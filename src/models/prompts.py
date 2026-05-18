"""Shared prompts for zero-shot vulnerability analysis.

These prompts are designed to be overrideable by ORPO automatic prompt optimization.
"""

ZERO_SHOT_PROMPT = """Please analyze the following code:
```
{func}
```
Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""

SYS_INST = "You are a security expert that is good at static program analysis."
