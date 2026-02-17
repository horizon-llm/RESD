# How to Use Agent Skills

Agent Skills provide modular, domain-specific knowledge and workflows that agents can invoke on-demand. This feature is inspired by [Anthropic's Claude Code skills](https://docs.anthropic.com/en/docs/claude-code/skills).

## Prerequisites

1. **Environment**: Skills are only supported in `shell_local` environment mode
2. **Context Manager**: Must use `env` context manager for skill prompts to be injected
3. **openskills CLI**: Install the [openskills](https://github.com/numman-ali/openskills) CLI tool:
   ```bash
   npm i -g openskills
   ```

## Quick Start

### 1. Create a Skill

Skills are stored in `.agent/skills/<skill-name>/` directory. Each skill needs a `SKILL.md` file with YAML frontmatter:

```
.agent/skills/
└── my-skill/
    ├── SKILL.md           # Main skill file (required)
    └── references/        # Optional reference files
        └── guide.md
```

**SKILL.md structure:**

```markdown
---
name: my-skill
description: A brief description of what this skill does and when to use it.
---

# My Skill Guide

Detailed instructions for the agent to follow when this skill is invoked.

## When to Use

- Use this skill when...
- This skill helps with...

## Instructions

1. Step one...
2. Step two...

## Reference Files

For more details, see [guide.md](./references/guide.md).
```

### 2. Configure the Agent

Create an agent config file (e.g., `configs/agents/my_agent.yaml`):

```yaml
# @package _global_
defaults:
  - /tools/bash@toolkits.BashTool
  - _self_

agent:
  name: MyAgent
  instructions: You are a helpful assistant.

env:
  name: shell_local  # Required for skills

context_manager:
  name: env  # Required for skill prompts to be injected

enabled_skills:
  - my-skill
  - another-skill
```

### 3. Run the Agent

```bash
python scripts/cli_chat.py --config my_agent
```

## How Skills Work

When an agent with `enabled_skills` starts:

1. **Skill Deployment**: Skill folders are copied from `.agent/skills/` to the workspace `.agent/skills/` directory
2. **Prompt Injection**: A skills system prompt is added to the agent's context listing available skills
3. **On-demand Reading**: The agent can read skill content using `openskills read <skill-name>`

The agent sees a prompt like:

```xml
<!-- SKILLS_SYSTEM:START -->
<usage>
When users ask you to perform tasks, check if any of the available skills below
can help complete the task more effectively.

How to use skills:
- Invoke: read skill with bash command "openskills read <skill-name>"
- The skill content will load with detailed instructions
</usage>

<available_skills>
<skill>
<name>my-skill</name>
<description>A brief description of what this skill does.</description>
</skill>
</available_skills>
<!-- SKILLS_SYSTEM:END -->
```

## References

- [openskills CLI](https://github.com/numman-ali/openskills) - CLI tool for reading skills
- [Anthropic Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code/skills) - Original inspiration
- [Agent Configuration Guide](config.md) - General agent configuration
