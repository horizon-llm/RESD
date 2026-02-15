from verl import DataProto

class ACEContextUpdater:
    def __init__(self, config):
        self.config = config

        self.playbook = self._initialize_empty_playbook()
    
    def _initialize_empty_playbook(self) -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""

    def update(self, batch: DataProto):
        pass