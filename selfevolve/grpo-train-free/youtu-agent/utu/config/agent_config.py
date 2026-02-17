import logging
from collections.abc import Callable
from typing import Literal

from pydantic import Field, model_validator

from .base_config import ConfigBaseModel
from .model_config import ModelConfigs

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


class ProfileConfig(ConfigBaseModel):
    name: str | None = "default"
    instructions: str | Callable | None = DEFAULT_INSTRUCTIONS


class ToolkitConfig(ConfigBaseModel):
    """Toolkit config."""

    mode: Literal["builtin", "customized", "mcp"] = "builtin"
    """Toolkit mode."""
    env_mode: Literal["local", "e2b"] = "local"
    """Environment mode for the toolkit."""
    name: str | None = None
    """Toolkit name."""
    activated_tools: list[str] | None = None
    """Activated tools, if None, all tools will be activated."""
    config: dict | None = Field(default_factory=dict)
    """Specified  configs for certain toolkit. We use raw dict for simplicity"""
    config_llm: ModelConfigs | None = None  # | dict[str, ModelConfigs]
    """LLM config if used in toolkit."""
    customized_filepath: str | None = None
    """Customized toolkit filepath."""
    customized_classname: str | None = None
    """Customized toolkit classname."""
    mcp_transport: Literal["stdio", "sse", "streamable_http"] = "stdio"
    """MCP transport."""
    mcp_client_session_timeout_seconds: int = 20
    """The read timeout passed to the MCP ClientSession. We set it bigger to avoid timeout expections."""


class ContextManagerConfig(ConfigBaseModel):
    name: str | None = None
    config: dict | None = Field(default_factory=dict)


class EnvConfig(ConfigBaseModel):
    name: str | None = None
    config: dict | None = Field(default_factory=dict)


class AgentConfig(ConfigBaseModel):
    """Overall agent config"""

    type: Literal["simple", "orchestra", "orchestrator", "workforce"] = "simple"
    """Agent type"""

    # simple agent config
    model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Model config, with model_provider, model_settings, model_params"""
    agent: ProfileConfig = Field(default_factory=ProfileConfig)
    """Agent profile config"""
    context_manager: ContextManagerConfig = Field(default_factory=ContextManagerConfig)
    """Context manager config"""
    env: EnvConfig = Field(default_factory=EnvConfig)
    """Env config"""
    enabled_skills: list[str] = Field(default_factory=list)
    """Enabled skills for this agent, only available when env=`shell_local` for now."""
    toolkits: dict[str, ToolkitConfig] = Field(default_factory=dict)
    """Toolkits config"""
    max_turns: int = 50
    """Max turns for simple agent. This param is derived from @openai-agents"""
    stop_at_tool_names: list[str] | None = None
    """Stop at tools for simple agent. This param is derived from @openai-agents"""
    runner: Literal["openai", "react"] = "openai"
    """Runner name for simple agent."""

    # orchestra agent config
    planner_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Planner model config"""
    planner_config: dict = Field(default_factory=dict)
    """Planner config (dict)\n
    - `examples_path`: path to planner examples json file"""
    workers: dict[str, "AgentConfig"] = Field(default_factory=dict)
    """Workers config"""
    workers_info: list[dict] = Field(default_factory=list)
    """Workers info, list of {name, desc, strengths, weaknesses}\n
    - `name`: worker name
    - `desc`: worker description
    - `strengths`: worker strengths
    - `weaknesses`: worker weaknesses"""
    reporter_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Reporter model config"""
    reporter_config: dict = Field(default_factory=dict)
    """Reporter config (dict)\n
    - `template_path`: template Jinja2 file path, with `question` and `trajectory` variables"""

    # workforce agent config
    workforce_planner_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Workforce planner model config"""
    workforce_planner_config: dict = Field(default_factory=dict)
    """Workforce planner config (dict)"""
    workforce_assigner_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Workforce assigner model config"""
    workforce_assigner_config: dict = Field(default_factory=dict)
    """Workforce assigner config (dict)"""
    workforce_answerer_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Workforce answerer model config"""
    workforce_answerer_config: dict = Field(default_factory=dict)
    """Workforce answerer config (dict)"""
    workforce_executor_agents: dict[str, "AgentConfig"] = Field(default_factory=dict)
    """Workforce executor agents config"""
    workforce_executor_config: dict = Field(default_factory=dict)
    """Workforce executor config (dict)"""
    workforce_executor_infos: list[dict] = Field(default_factory=list)
    """Workforce executor infos, list of {name, desc, strengths, weaknesses}"""

    # orchestrator agent config
    orchestrator_router: "AgentConfig" = None
    """Orchestrator router agent config"""
    orchestrator_config: dict = Field(default_factory=dict)
    """Orchestrator config (dict)\n
    - `name`: name of the orchestrator-workers system
    - `examples_path`: path to planner examples. default utu/data/plan_examples/chain.json
    - `additional_instructions`: additional instructions for planner
    - `add_chitchat_subagent`: whether to add chitchat subagent. default True"""
    orchestrator_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Planner model config"""
    orchestrator_workers: dict[str, "AgentConfig"] = Field(default_factory=dict)
    """Workers config"""
    orchestrator_workers_info: list[dict] = Field(default_factory=list)
    """Workers info, list of {name, description}"""

    @model_validator(mode="after")
    def validate_enabled_skills(self):
        """Validate that enabled_skills is used with correct env and context_manager settings."""
        if not self.enabled_skills:
            return self

        # Check env is shell_local
        if not self.env or self.env.name != "shell_local":
            logger.warning(
                "enabled_skills requires env.name='shell_local'. "
                f"Current env: {self.env.name if self.env else None}. "
                "Skills may not work properly."
            )

        # Check context_manager is env
        if not self.context_manager or self.context_manager.name != "env":
            logger.warning(
                "enabled_skills requires context_manager.name='env' for skill prompts to be injected. "
                f"Current context_manager: {self.context_manager.name if self.context_manager else None}. "
                "Skills may not work properly."
            )

        return self
