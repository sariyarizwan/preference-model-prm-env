# Process Reward Model RL Environment

A runnable scaffold for an RL environment in which an LLM agent trains a
Process Reward Model (PRM) end-to-end and is graded against an Outcome
Reward Model baseline and a frozen PRM800K-style reference PRM, on a held-out
slice of MATH.

The full design writeup — prompt, judge, reward-hacking analysis, the works
— lives at `docs/assessment.md`. This README explains how to run the project.

## What's interesting here

The judge is the centerpiece. It runs seven gates from the writeup and scores
a submission continuously only when every hard gate passes. The most important
gate is **step-localization F1**, which is what catches the *outcome-correlation
shortcut* — the open failure mode in the PRM literature where an auto-labeled
PRM degenerates into an outcome predictor with no per-step signal.

To prove the gate actually catches what the writeup claims it catches, the
project ships **four example submissions**, each engineered to trigger a
specific gate:

| submission                       | designed to                          | expected gate failure       |
| -------------------------------- | ------------------------------------ | --------------------------- |
| `real_prm_submission`            | pass everything                      | none                        |
| `outcome_classifier_submission`  | demonstrate Hack #2 from §5          | step-localization F1        |
| `uniform_submission`             | demonstrate trivially-bad submission | capability floor            |
| `threshold_gamer_submission`     | demonstrate Hack #3 from §5          | BoN-32 lift                 |

The end-to-end test (`tests/test_judge_end_to_end.py`) asserts each submission
actually triggers its predicted failure. If any test starts passing or failing
in a different way, the judge has lost its teeth.

## Run it

Tested on Python 3.10+. CPU only.

```bash
cd preference_model_prm_env
pip install -e ".[dev]"

# Run the test suite (corruptions, symbolic checker, judge end-to-end).
pytest

# Run the demo: judge runs on all four submissions, prints verdict matrix.
python scripts/run_demo.py
```

The demo finishes in a few seconds and produces output like:

```
submission                          | exist | disal | flop_ | capab | bon32 | step_ | calib | score
-------------------------------------------------------------------------------------------
real PRM                            |  PASS |  PASS |  PASS |  PASS |  PASS |  PASS |  PASS | 0.770
outcome-classifier-in-disguise      |  PASS |  PASS |  PASS |  PASS |  PASS |  FAIL |  FAIL | 0.000
uniform 0.5 scorer                  |  PASS |  PASS |  PASS |  FAIL |  FAIL |  FAIL |  FAIL | 0.000
threshold-shape gamer               |  PASS |  PASS |  PASS |  FAIL |  FAIL |  PASS |  FAIL | 0.000
```

## Layout

```
preference_model_prm_env/
├── docs/assessment.md               # the design writeup (full text of the assessment answers)
├── pyproject.toml                   # numpy + sympy; pytest for dev
├── README.md                        # this file
├── examples/                        # four submissions (proof that each gate fires)
│   ├── real_prm_submission.py
│   ├── outcome_classifier_submission.py
│   ├── uniform_submission.py
│   └── threshold_gamer_submission.py
├── scripts/
│   └── run_demo.py                  # judge over all four submissions
├── src/prm_env/
│   ├── prompt.py                    # the agent prompt verbatim
│   ├── corruptions.py               # the step-corruption probe generator
│   ├── symbolic_check.py            # PRM800K-style answer equivalence
│   ├── evaluators.py                # BoN-32, step-loc F1, ECE, capability agreement
│   ├── gates.py                     # existence / disallowed-checkpoint / FLOP gates
│   ├── judge.py                     # the orchestrator
│   ├── baselines/
│   │   ├── outcome_rm.py
│   │   └── reference_prm.py
│   ├── data/
│   │   └── splits.py                # toy MATH-style problems with structured steps
│   └── labeling/                    # placeholders for Math-Shepherd / OmegaPRM auto-label
└── tests/
    ├── test_corruptions.py
    ├── test_symbolic_check.py
    └── test_judge_end_to_end.py
```

## What is real and what is a stand-in

For the demo to run on a laptop in seconds, the project ships toy data and
mock baselines. The contracts are real:

| component                     | toy demo                                                                                    | production                                                                                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Problems                      | 200 procedurally generated arithmetic word problems with structured steps (40-each eval splits) | the MATH dataset, partitioned by problem id                                                                                                              |
| Step-corruption probes        | mechanical sign-flip / operand-swap / miscopy / off-by-one with forward propagation         | the same procedure, run over corrupted variants of human-annotated solutions; ideally augmented with a small human-labeled probe to validate the proxy     |
| Submission interface          | `score(problem_statement, step_texts) -> List[float]`                                       | the full torch interface in the writeup: `score(input_ids, attention_mask, step_end_positions)`                                                            |
| Submission internals          | closure over an internal table populated at `load_prm()` time (stand-in for learned weights) | actual model weights                                                                                                                                        |
| OutcomeRM / ReferencePRM      | heuristic with calibrated noise                                                             | a fine-tuned outcome head and a PRM800K-trained reference head respectively                                                                                  |
| FLOP gate                     | accepts a numeric argument                                                                   | reads from `/sandbox/flops_remaining`, populated by an enforced FLOP meter on the agent's training stack                                                    |
| Disallowed-checkpoint gate    | static AST scan for known-bad substrings                                                    | same scan + weight-hash comparison against published PRM checkpoints                                                                                         |

The judge code, the gates, the seven-gate orchestration, the BoN-32
aggregation logic, the step-localization F1 calculation, and the ECE
calculation are all the production design — not a mock.

## Citations

The design draws on:

- Lightman et al., *Let's Verify Step by Step* (PRM800K), 2023.
- Wang et al., *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*, 2023.
- Luo et al., *Improve Mathematical Reasoning in Language Models by Automated Process Supervision* (OmegaPRM), 2024.
- Skywork-PRM team, *Skywork-PRM*, 2024.
- Zhang et al., *Generative Verifiers: Reward Modeling as Next-Token Prediction*, 2024.
- DeepSeek-AI, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO), 2024.
- Lambert et al., *RewardBench*, 2024.
