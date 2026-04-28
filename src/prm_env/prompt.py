"""The agent prompt — given verbatim to the LLM at the start of each
RL rollout in this environment."""

AGENT_PROMPT = """\
You are working in /workspace. Train a Process Reward Model (PRM) for math
reasoning and ship it as a single Python module.

Filesystem you have access to:
  /models/qwen2.5-math-1.5b/                  # base model (frozen, this is your initialization)
  /models/qwen2.5-math-1.5b-instruct/         # weak generator (frozen, for rollouts)
  /data/math_train_pool.jsonl                 # labeling pool: problem + gold_answer
  /sandbox/disallowed_checkpoints.txt         # PRMs you may NOT initialize from or distill against
  /sandbox/eval_spec.md                       # exact evaluator signatures the judge uses
  /sandbox/flops_remaining                    # int file the FLOP-meter reads

Hard limits:
  - 4 x H100s for 12 hours wall-clock total
  - 1.5e18 FLOPs total (counter at /sandbox/flops_remaining)
  - No internet access

Deliverable: /workspace/prm.py exporting

    def load_prm() -> torch.nn.Module:
        '''Returns a module with a `score` method matching the signature below.'''
        ...

    # The returned module must have:
    #   score(input_ids: LongTensor[B, L],
    #         attention_mask: LongTensor[B, L],
    #         step_end_positions: List[List[int]]
    #        ) -> FloatTensor[B, max_num_steps]
    # where step_end_positions[i] gives the token positions ending each
    # reasoning step in row i. Output is in [0, 1] per step, with `nan`
    # padded past each row's true step count. Higher means the prefix up to
    # that step is correct.

    MODEL_INFO: dict  # must contain keys: base, uses_lora, trained_steps, labeling_method

You may use any auto-labeling strategy: Math-Shepherd (Wang et al. 2023),
OmegaPRM (Luo et al. 2024), ER-MC, or your own. You may NOT initialize from
or distill against any pre-trained math PRM. The disallowed list is at
/sandbox/disallowed_checkpoints.txt; static analysis runs at submission time.

Evaluation (run by the judge after you submit):
  (a) BoN-32 accuracy on a held-out MATH-Hard split, ranking 32 samples from
      the weak generator at fixed seed and temperature.
  (b) Step-localization F1 on synthetic step-corruption probes.
  (c) Expected Calibration Error against ground-truth step labels.

Your final score is a continuous weighted combination of these three, gated
by a capability floor on RewardBench-Math agreement and a hard FLOP cap.
Exact evaluator signatures: /sandbox/eval_spec.md.
"""
