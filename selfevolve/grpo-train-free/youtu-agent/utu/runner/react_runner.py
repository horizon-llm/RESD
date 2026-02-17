"""ReactRunner: Custom agent runner implementing the ReAct (Reasoning-Acting) pattern.

This module provides a custom agent execution loop that offers more flexibility than the
default openai-agents Runner, including configurable max-turns handling and future
extensibility for context management.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from agents import Agent, RunConfig, RunResultStreaming, Session
from agents._run_impl import (
    AgentToolUseTracker,
    NextStepFinalOutput,
    NextStepHandoff,
    NextStepRunAgain,
    QueueCompleteSentinel,
    RunImpl,
    SingleStepResult,
    get_model_tracing_impl,
)
from agents.agent_output import AgentOutputSchema, AgentOutputSchemaBase
from agents.exceptions import ModelBehaviorError, UserError
from agents.handoffs import Handoff, handoff
from agents.items import HandoffCallItem, ItemHelpers, ModelResponse, RunItem, TResponseInputItem
from agents.lifecycle import RunHooks
from agents.models.interface import Model
from agents.run import DEFAULT_MAX_TURNS, CallModelData, ModelInputData
from agents.run_context import RunContextWrapper, TContext
from agents.stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent
from agents.tool import Tool
from agents.tracing import SpanError, agent_span, get_current_trace, trace
from agents.tracing.span_data import AgentSpanData
from agents.usage import Usage
from agents.util import _coro, _error_tracing
from openai.types.responses import ResponseCompletedEvent

if TYPE_CHECKING:
    from agents.tracing import Span


@dataclass
class ReactRunnerConfig:
    """Configuration options for ReactRunner."""

    max_turns: int = DEFAULT_MAX_TURNS
    on_max_turns: Literal["error", "reply"] = "error"
    """Behavior when max turns is exceeded:
    - "error": Raise MaxTurnsExceeded (default, SDK-compatible)
    - "reply": Remove tools and generate a final reply
    """


class ReactRunner:
    """Custom agent runner implementing the ReAct (Reasoning-Acting) pattern.

    This runner provides:
    - Streaming-only execution via `run_streamed`
    - Full hooks integration for lifecycle callbacks
    - Tracing integration with openai-agents
    - Configurable max-turns handling
    - Core ReAct loop: think → act → observe → repeat

    Usage:
        result = ReactRunner.run_streamed(agent, input, max_turns=10)
        async for event in result.stream_events():
            process(event)
    """

    @classmethod
    def run_streamed(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        context: TContext | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: Session | None = None,
        **kwargs,
    ) -> RunResultStreaming:
        """Run an agent with streaming output.

        Args:
            starting_agent: The agent to run.
            input: User input (string or list of input items).
            context: Optional context object passed to tools and hooks.
            max_turns: Maximum number of turns before stopping.
            hooks: Lifecycle hooks for the run.
            run_config: Run configuration (tracing, model settings, etc).
            session: Optional session for conversation history persistence.

        Returns:
            RunResultStreaming object that can be used to stream events.
        """
        print(">> using ReactRunner")
        if run_config is None:
            run_config = RunConfig()
        if hooks is None:
            hooks = RunHooks[TContext]()

        # Create trace if not already in one
        new_trace = (
            None
            if get_current_trace()
            else trace(
                workflow_name=run_config.workflow_name,
                trace_id=run_config.trace_id,
                group_id=run_config.group_id,
                metadata=run_config.trace_metadata,
                disabled=run_config.tracing_disabled,
            )
        )

        output_schema = cls._get_output_schema(starting_agent)
        context_wrapper: RunContextWrapper[TContext] = RunContextWrapper(context=context)

        streamed_result = RunResultStreaming(
            input=cls._copy_input(input),
            new_items=[],
            current_agent=starting_agent,
            raw_responses=[],
            final_output=None,
            is_complete=False,
            current_turn=0,
            max_turns=max_turns,
            input_guardrail_results=[],
            output_guardrail_results=[],
            tool_input_guardrail_results=[],
            tool_output_guardrail_results=[],
            _current_agent_output_schema=output_schema,
            trace=new_trace,
            context_wrapper=context_wrapper,
        )

        # Start the streaming loop in background
        streamed_result._run_impl_task = asyncio.create_task(
            cls._streaming_loop(
                starting_input=input,
                streamed_result=streamed_result,
                starting_agent=starting_agent,
                max_turns=max_turns,
                hooks=hooks,
                context_wrapper=context_wrapper,
                run_config=run_config,
                session=session,
            )
        )
        return streamed_result

    @classmethod
    async def _streaming_loop(
        cls,
        starting_input: str | list[TResponseInputItem],
        streamed_result: RunResultStreaming,
        starting_agent: Agent[TContext],
        max_turns: int,
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        session: Session | None = None,
    ) -> None:
        """Core ReAct loop: think → act → observe → repeat."""
        if streamed_result.trace:
            streamed_result.trace.start(mark_as_current=True)

        current_span: Span[AgentSpanData] | None = None
        current_agent = starting_agent
        current_turn = 0
        should_run_agent_start_hooks = True
        tool_use_tracker = AgentToolUseTracker()

        # Emit initial agent event
        streamed_result._event_queue.put_nowait(AgentUpdatedStreamEvent(new_agent=current_agent))

        try:
            # Prepare input with session if enabled
            prepared_input = await cls._prepare_input_with_session(
                starting_input, session, run_config.session_input_callback
            )
            streamed_result.input = prepared_input

            # Save original input to session (empty new_items at start)
            await cls._save_result_to_session(session, starting_input, [])

            while True:
                # Check for cancellation
                if streamed_result._cancel_mode == "after_turn":
                    streamed_result.is_complete = True
                    streamed_result._event_queue.put_nowait(QueueCompleteSentinel())
                    break

                if streamed_result.is_complete:
                    break

                # Get all tools for the current agent
                all_tools = await cls._get_all_tools(current_agent, context_wrapper)
                # await RunImpl.initialize_computer_tools(tools=all_tools, context_wrapper=context_wrapper)

                # Start agent span if not already started
                if current_span is None:
                    handoff_names = [h.agent_name for h in await cls._get_handoffs(current_agent, context_wrapper)]
                    output_type_name = cls._get_output_schema(current_agent)
                    output_type_name = output_type_name.name() if output_type_name else "str"

                    current_span = agent_span(
                        name=current_agent.name,
                        handoffs=handoff_names,
                        output_type=output_type_name,
                    )
                    current_span.start(mark_as_current=True)
                    current_span.span_data.tools = [t.name for t in all_tools]

                current_turn += 1
                streamed_result.current_turn = current_turn

                # Check max turns - if exceeded, attach error to span and break
                # The MaxTurnsExceeded exception is raised by RunResultStreaming._check_errors()
                # when consumer calls stream_events()
                if current_turn > max_turns:
                    _error_tracing.attach_error_to_span(
                        current_span,
                        SpanError(
                            message="Max turns exceeded",
                            data={"max_turns": max_turns},
                        ),
                    )
                    streamed_result._event_queue.put_nowait(QueueCompleteSentinel())
                    break

                # Run single turn
                turn_result = await cls._run_single_turn_streamed(
                    streamed_result=streamed_result,
                    agent=current_agent,
                    hooks=hooks,
                    context_wrapper=context_wrapper,
                    run_config=run_config,
                    should_run_agent_start_hooks=should_run_agent_start_hooks,
                    tool_use_tracker=tool_use_tracker,
                    all_tools=all_tools,
                )
                should_run_agent_start_hooks = False

                # Update result state
                streamed_result.raw_responses = streamed_result.raw_responses + [turn_result.model_response]
                streamed_result.input = turn_result.original_input
                streamed_result.new_items = turn_result.generated_items

                # Handle next step
                if isinstance(turn_result.next_step, NextStepHandoff):
                    # Save to session before handoff
                    await cls._save_result_to_session(session, [], turn_result.new_step_items)

                    current_agent = cast(Agent[TContext], turn_result.next_step.new_agent)
                    current_span.finish(reset_current=True)
                    current_span = None
                    should_run_agent_start_hooks = True
                    streamed_result._event_queue.put_nowait(AgentUpdatedStreamEvent(new_agent=current_agent))

                    if streamed_result._cancel_mode == "after_turn":
                        streamed_result.is_complete = True
                        streamed_result._event_queue.put_nowait(QueueCompleteSentinel())
                        break

                elif isinstance(turn_result.next_step, NextStepFinalOutput):
                    # Save to session before completing
                    await cls._save_result_to_session(session, [], turn_result.new_step_items)

                    streamed_result.final_output = turn_result.next_step.output
                    streamed_result.is_complete = True
                    streamed_result._event_queue.put_nowait(QueueCompleteSentinel())

                elif isinstance(turn_result.next_step, NextStepRunAgain):
                    # Save to session before next turn
                    await cls._save_result_to_session(session, [], turn_result.new_step_items)

                    if streamed_result._cancel_mode == "after_turn":
                        streamed_result.is_complete = True
                        streamed_result._event_queue.put_nowait(QueueCompleteSentinel())
                        break

            streamed_result.is_complete = True
        except Exception:
            streamed_result.is_complete = True
            streamed_result._event_queue.put_nowait(QueueCompleteSentinel())
            raise
        finally:
            if current_span:
                current_span.finish(reset_current=True)
            if streamed_result.trace:
                streamed_result.trace.finish(reset_current=True)
            if not streamed_result.is_complete:
                streamed_result.is_complete = True
                streamed_result._event_queue.put_nowait(QueueCompleteSentinel())

    @classmethod
    async def _run_single_turn_streamed(
        cls,
        streamed_result: RunResultStreaming,
        agent: Agent[TContext],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_hooks: bool,
        tool_use_tracker: AgentToolUseTracker,
        all_tools: list[Tool],
    ) -> SingleStepResult:
        """Execute a single turn of the ReAct loop with streaming."""
        # emitted_tool_call_ids: set[str] = set()
        # emitted_reasoning_item_ids: set[str] = set()

        # Run agent start hooks
        if should_run_agent_start_hooks:
            await asyncio.gather(
                hooks.on_agent_start(context_wrapper, agent),
                agent.hooks.on_start(context_wrapper, agent) if agent.hooks else _coro.noop_coroutine(),
            )

        output_schema = cls._get_output_schema(agent)
        streamed_result.current_agent = agent
        streamed_result._current_agent_output_schema = output_schema

        # Get prompts and handoffs
        system_prompt, prompt_config = await asyncio.gather(
            agent.get_system_prompt(context_wrapper),
            agent.get_prompt(context_wrapper),
        )
        handoffs = await cls._get_handoffs(agent, context_wrapper)
        model = cls._get_model(agent, run_config)
        model_settings = agent.model_settings.resolve(run_config.model_settings)
        model_settings = RunImpl.maybe_reset_tool_choice(agent, tool_use_tracker, model_settings)

        # Prepare input
        input_items = ItemHelpers.input_to_new_input_list(streamed_result.input)
        input_items.extend([item.to_input_item() for item in streamed_result.new_items])

        # Optional input filter
        filtered = await cls._maybe_filter_model_input(
            agent=agent,
            run_config=run_config,
            context_wrapper=context_wrapper,
            input_items=input_items,
            system_instructions=system_prompt,
        )

        # Call on_llm_start hooks
        await asyncio.gather(
            hooks.on_llm_start(context_wrapper, agent, filtered.instructions, filtered.input),
            agent.hooks.on_llm_start(context_wrapper, agent, filtered.instructions, filtered.input)
            if agent.hooks
            else _coro.noop_coroutine(),
        )

        final_response: ModelResponse | None = None

        # Stream model response
        async for event in model.stream_response(
            filtered.instructions,
            filtered.input,
            model_settings,
            all_tools,
            output_schema,
            handoffs,
            get_model_tracing_impl(run_config.tracing_disabled, run_config.trace_include_sensitive_data),
            prompt=prompt_config,
        ):
            # Emit raw event
            streamed_result._event_queue.put_nowait(RawResponsesStreamEvent(data=event))

            if isinstance(event, ResponseCompletedEvent):
                usage = (
                    Usage(
                        requests=1,
                        input_tokens=event.response.usage.input_tokens,
                        output_tokens=event.response.usage.output_tokens,
                        total_tokens=event.response.usage.total_tokens,
                        input_tokens_details=event.response.usage.input_tokens_details,
                        output_tokens_details=event.response.usage.output_tokens_details,
                    )
                    if event.response.usage
                    else Usage()
                )
                final_response = ModelResponse(
                    output=event.response.output,
                    usage=usage,
                    response_id=event.response.id,
                )
                context_wrapper.usage.add(usage)

        # Call on_llm_end hooks
        if final_response is not None:
            await asyncio.gather(
                agent.hooks.on_llm_end(context_wrapper, agent, final_response)
                if agent.hooks
                else _coro.noop_coroutine(),
                hooks.on_llm_end(context_wrapper, agent, final_response),
            )

        if not final_response:
            raise ModelBehaviorError("Model did not produce a final response!")

        # Process response and execute tools
        single_step_result = await cls._get_single_step_result_from_response(
            agent=agent,
            original_input=streamed_result.input,
            pre_step_items=streamed_result.new_items,
            new_response=final_response,
            output_schema=output_schema,
            all_tools=all_tools,
            handoffs=handoffs,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
            tool_use_tracker=tool_use_tracker,
            event_queue=streamed_result._event_queue,
        )

        # Stream step results to queue
        RunImpl.stream_step_result_to_queue(single_step_result, streamed_result._event_queue)
        return single_step_result

    @classmethod
    async def _get_single_step_result_from_response(
        cls,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        pre_step_items: list[RunItem],
        new_response: ModelResponse,
        output_schema: AgentOutputSchemaBase | None,
        all_tools: list[Tool],
        handoffs: list[Handoff],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        tool_use_tracker: AgentToolUseTracker,
        event_queue: asyncio.Queue | None = None,
    ) -> SingleStepResult:
        """Process model response and execute tools."""
        processed_response = RunImpl.process_model_response(
            agent=agent,
            all_tools=all_tools,
            response=new_response,
            output_schema=output_schema,
            handoffs=handoffs,
        )
        tool_use_tracker.add_tool_use(agent, processed_response.tools_used)

        # Stream handoff items immediately
        if event_queue is not None and processed_response.new_items:
            handoff_items = [item for item in processed_response.new_items if isinstance(item, HandoffCallItem)]
            if handoff_items:
                RunImpl.stream_step_items_to_queue(cast(list[RunItem], handoff_items), event_queue)

        return await RunImpl.execute_tools_and_side_effects(
            agent=agent,
            original_input=original_input,
            pre_step_items=pre_step_items,
            new_response=new_response,
            processed_response=processed_response,
            output_schema=output_schema,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
        )

    # Helper methods
    @staticmethod
    def _copy_input(input: str | list[TResponseInputItem]) -> str | list[TResponseInputItem]:
        if isinstance(input, str):
            return input
        return list(input)

    @staticmethod
    def _get_output_schema(agent: Agent) -> AgentOutputSchemaBase | None:
        if agent.output_type is None or agent.output_type is str:
            return None
        return AgentOutputSchema(agent.output_type)

    @staticmethod
    def _get_model(agent: Agent, run_config: RunConfig) -> Model:
        if isinstance(run_config.model, Model):
            return run_config.model
        elif isinstance(run_config.model, str):
            return run_config.model_provider.get_model(run_config.model)
        elif isinstance(agent.model, Model):
            return agent.model

        return run_config.model_provider.get_model(agent.model)

    @staticmethod
    async def _get_all_tools(agent: Agent[TContext], context_wrapper: RunContextWrapper[TContext]) -> list[Tool]:
        return await agent.get_all_tools(context_wrapper)

    @staticmethod
    async def _get_handoffs(agent: Agent[TContext], context_wrapper: RunContextWrapper[TContext]) -> list[Handoff]:
        resolved = []
        for h in agent.handoffs:
            if isinstance(h, Handoff):
                resolved.append(h)
            elif isinstance(h, Agent):
                resolved.append(handoff(h))
            elif callable(h):
                result = h(context_wrapper)
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, Handoff):
                    resolved.append(result)
                elif isinstance(result, Agent):
                    resolved.append(handoff(result))
        return resolved

    @staticmethod
    async def _maybe_filter_model_input(
        agent: Agent[TContext],
        run_config: RunConfig,
        context_wrapper: RunContextWrapper[TContext],
        input_items: list[TResponseInputItem],
        system_instructions: str | None,
    ):
        """Apply optional input filter."""
        effective_instructions = system_instructions
        effective_input = input_items

        if run_config.call_model_input_filter is None:
            return ModelInputData(input=effective_input, instructions=effective_instructions)

        model_input = ModelInputData(input=effective_input.copy(), instructions=effective_instructions)
        filter_payload = CallModelData(model_data=model_input, agent=agent, context=context_wrapper.context)
        maybe_updated = run_config.call_model_input_filter(filter_payload)
        updated = await maybe_updated if inspect.isawaitable(maybe_updated) else maybe_updated
        return updated

    @classmethod
    async def _prepare_input_with_session(
        cls,
        input: str | list[TResponseInputItem],
        session: Session | None,
        session_input_callback=None,
    ) -> str | list[TResponseInputItem]:
        """Prepare input by combining it with session history if enabled."""
        if session is None:
            return input

        # If the user doesn't specify an input callback and pass a list as input
        if isinstance(input, list) and not session_input_callback:
            raise UserError(
                "When using session memory, list inputs require a "
                "`RunConfig.session_input_callback` to define how they should be merged "
                "with the conversation history. If you don't want to use a callback, "
                "provide your input as a string instead, or disable session memory "
                "(session=None) and pass a list to manage the history manually."
            )

        # Get previous conversation history
        history = await session.get_items()

        # Convert input to list format
        new_input_list = ItemHelpers.input_to_new_input_list(input)

        if session_input_callback is None:
            return history + new_input_list
        elif callable(session_input_callback):
            res = session_input_callback(history, new_input_list)
            if inspect.isawaitable(res):
                return await res
            return res
        else:
            raise UserError(
                f"Invalid `session_input_callback` value: {session_input_callback}. "
                "Choose between `None` or a custom callable function."
            )

    @classmethod
    async def _save_result_to_session(
        cls,
        session: Session | None,
        original_input: str | list[TResponseInputItem],
        new_items: list[RunItem],
    ) -> None:
        """Save the conversation turn to session."""
        if session is None:
            return

        # Convert original input to list format if needed
        input_list = ItemHelpers.input_to_new_input_list(original_input)

        # Convert new items to input format
        new_items_as_input = [item.to_input_item() for item in new_items]

        # Save all items from this turn
        items_to_save = input_list + new_items_as_input
        await session.add_items(items_to_save)
