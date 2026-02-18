import copy
import logging

from agents import RunContextWrapper, TContext, TResponseInputItem
from openai.types.responses import EasyInputMessageParam

from ..env import BaseEnv
from .base_context_manager import BaseContextManager

logger = logging.getLogger(__name__)


class EnvContextManager(BaseContextManager):
    def preprocess(
        self, input: str | list[TResponseInputItem], run_context: RunContextWrapper[TContext] = None
    ) -> str | list[TResponseInputItem]:
        if run_context is None or run_context.context.get("env", None) is None:
            logger.warning(f"run_context {run_context} or env is None")
            return input
        env: BaseEnv = run_context.context["env"]
        input = copy.deepcopy(input)
        if (extra_sp := env.get_extra_sp()) and _is_first_query(input):
            input = [EasyInputMessageParam(content=extra_sp, role="user")] + input
        if env_state := env.get_state():
            input.append(EasyInputMessageParam(content=env_state, role="user"))
        return input


def _is_first_query(input: str | list[TResponseInputItem]) -> bool:
    if isinstance(input, str):
        return True
    return len(input) < 2
