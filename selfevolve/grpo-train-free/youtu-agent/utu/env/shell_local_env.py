import logging
import shutil
import subprocess
from pathlib import Path

import frontmatter

from ..utils import DIR_ROOT
from .base_env import BaseEnv

logger = logging.getLogger(__name__)


def _check_openskills_installed() -> bool:
    """Check if openskills CLI is installed."""
    try:
        result = subprocess.run(["which", "openskills"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


TEMPLATE = r"""<env>
{env}
</env>
<instructions>
1. You CANNOT make any changes outside your workspace!
</instructions>
"""

SKILLS_PROMPT_TEMPLATE = r"""<!-- SKILLS_SYSTEM:START -->
<usage>
When users ask you to perform tasks, check if any of the available skills below can help complete the task
more effectively. Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke: read skill with bash command "openskills read <skill-name>"
- The skill content will load with detailed instructions on how to complete the task
- Base directory provided in output for resolving bundled resources (references/, scripts/, assets/)

Usage notes:
- Only use skills listed in <available_skills> below
- Do not invoke a skill that is already loaded in your context
- Each skill invocation is stateless
</usage>

<available_skills>
{skills_xml}
</available_skills>
<!-- SKILLS_SYSTEM:END -->
"""

SKILL_XML_TEMPLATE = """<skill>
<name>{name}</name>
<description>{description}</description>
</skill>"""


def parse_skill_metadata(skill_path: Path) -> dict:
    """Extract name and description from SKILL.md frontmatter."""
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        return {"name": skill_path.name, "description": "No description"}

    try:
        post = frontmatter.load(skill_md)
        return {
            "name": post.get("name", skill_path.name),
            "description": post.get("description", "No description"),
        }
    except Exception as e:
        logger.warning(f"Failed to parse SKILL.md at {skill_md}: {e}")
        return {"name": skill_path.name, "description": "No description"}


class ShellLocalEnv(BaseEnv):
    workspace: str
    enabled_skills: list[str]

    def __init__(self, config: dict = None, trace_id: str = None, enabled_skills: list[str] = None):
        config = config or {}
        workspace = config.get("workspace_root")
        if not workspace:
            workspace = DIR_ROOT / "workspace" / trace_id
            workspace.mkdir(parents=True, exist_ok=True)
        print(f"> Workspace: {workspace}")
        self.workspace = workspace
        self.enabled_skills = enabled_skills or []
        self._setup_skills()

    def _setup_skills(self):
        """Copy enabled skills to workspace .agent/skills/ directory."""
        if not self.enabled_skills:
            return

        # Check if openskills CLI is installed
        if not _check_openskills_installed():
            logger.warning(
                "openskills CLI is not installed. Skills require openskills to work properly. "
                "Please install it from: https://github.com/numman-ali/openskills"
            )

        source_root = DIR_ROOT / ".agent" / "skills"
        target_root = Path(self.workspace) / ".agent" / "skills"

        for skill_name in self.enabled_skills:
            source = source_root / skill_name
            target = target_root / skill_name
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target, dirs_exist_ok=True)
                logger.info(f"Skill '{skill_name}' copied to {target}")
            else:
                logger.warning(f"Skill '{skill_name}' not found at {source}")

    def get_state(self) -> str:
        env_strs = [
            f"Time: {self.get_time()}",
            f"Workspace: {self.workspace}",
        ]
        sp_prefix = TEMPLATE.format(env="\n".join(env_strs))
        return sp_prefix

    def get_extra_sp(self) -> str:
        if not self.enabled_skills:
            return ""

        skills_xml_parts = []
        for skill_name in self.enabled_skills:
            skill_path = Path(self.workspace) / ".agent" / "skills" / skill_name
            metadata = parse_skill_metadata(skill_path)
            skills_xml_parts.append(SKILL_XML_TEMPLATE.format(**metadata))

        skills_xml = "\n".join(skills_xml_parts)
        return SKILLS_PROMPT_TEMPLATE.format(skills_xml=skills_xml)
