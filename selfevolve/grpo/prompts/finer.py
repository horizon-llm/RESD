FINER_PROMPT_TEMPLATE = """
Answer the given question based on the provided context.

### Context:
{context}

### Question:
{question}

Reason about it and then put your final answer inside <answer> and </answer> tags. For example, <answer>TagOne,TagTwo,TagThree,TagFour</answer>.
"""