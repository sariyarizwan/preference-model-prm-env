"""prm_env — RL environment for training Process Reward Models.

This package provides the prompt, judge, gates, evaluators, corruption
procedures, and toy data needed to run an RL environment in which an
LLM agent trains a PRM end-to-end and is graded against an OutcomeRM
baseline and a frozen PRM800K-style reference PRM.

The full design (the assessment writeup) lives at /docs/assessment.md.
"""

from prm_env.prompt import AGENT_PROMPT
from prm_env.judge import Judge, JudgeResult

__all__ = ["AGENT_PROMPT", "Judge", "JudgeResult"]
