"""
Generator prompts for ACE system.
"""

# Retrieval and Reason Generator prompt that outputs bullet IDs
STUDENT_PROMPT = """You are an analysis expert tasked with answering questions using your knowledge.

**Instructions:**
- Show your reasoning step-by-step
- Be concise but thorough in your analysis
- Double-check your calculations and logic before providing the final answer

Your output should be a json object, which contains the following fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- final_answer: your concise final answer

**Prompt:**
{prompt}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",  
  "final_answer": "[Your concise final answer here]"
}}

---
"""

TEACHER_PROMPT = """You are an analysis expert tasked with answering questions using your knowledge, a curated playbook of strategies and insights, a previous trial of yours and a reflection that goes over the diagnosis of all previous mistakes made while answering the question, an environment feedback and a successful trial.

**Instructions:**
- Read the playbook carefully and apply relevant strategies, formulas, and insights
- Pay attention to common mistakes listed in the playbook and avoid them
- Review the previous trial and reflection to understand where you went wrong before, and make sure to not repeat the same mistakes
- Show your reasoning step-by-step
- Be concise but thorough in your analysis
- If the playbook contains relevant code snippets or formulas, use them appropriately
- Double-check your calculations and logic before providing the final answer

Your output should be a json object, which contains the following fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- final_answer: your concise final answer


**Playbook:**
{playbook}

**Previous Trial**
{previous_trial}

**Reflection:**
{reflection}

**Environment Feedback:**
{feedback}

**Successful Trial:**
{solution}

**Prompt:**
{prompt}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",  
  "final_answer": "[Your concise final answer here]"
}}

---
"""