# Preference Model Initial Assessment
## RL Environment Design for Process Reward Model Training

**Prepared by:** Sariya Rizwan  
**GitHub Repo:** https://github.com/sariyarizwan/preference-model-prm-env  
**Live Demo:** https://preference-model-prm-env.vercel.app

---

## Executive Summary

This proposal designs a realistic reinforcement-learning environment in which an LLM agent must build, train, and ship a Process Reward Model (PRM) for mathematical reasoning. The agent is judged not only on final-answer ranking performance, but also on whether its reward model correctly identifies which reasoning step first becomes wrong. This makes the task much harder to game than a standard outcome-only reward-modeling benchmark.

This submission is a design proposal for the environment, prompt, judge, tools, and data; it does not include implementation code.

- **Core task:** Train a PRM end-to-end using automatically generated step-level labels from MCTS-style rollouts.
- **Primary objective:** Beat both an Outcome Reward Model baseline and a hidden frozen PRM800K-style reference model in Best-of-32 candidate selection.
- **Anti-reward-hacking focus:** Require localized step-error detection, calibration, FLOP compliance, and disallowed-checkpoint screening.
- **Why it fits the assessment:** The environment mirrors real AI/ML research work: creating a reward signal, validating it under adversarial conditions, and preventing superficial metric optimization.

---

## 1. Describe an environment you could build matching the requirements. What makes this environment interesting?

I would build an RL environment where the LLM agent must train a Process Reward Model (PRM) for math reasoning from scratch, using a fixed base model and a fixed weak generator. The agent receives a local training pool of math problems with final answers, but it does not receive trusted step-level labels. Its job is to create those labels through a rollout-based procedure, train a PRM, and submit a single PyTorch module that scores reasoning steps.

The central idea is that the submitted model should reward correct intermediate reasoning, not just correct final answers. The judge therefore evaluates both outcome-level usefulness and step-level faithfulness. A successful solution must be useful as a ranker in Best-of-32 sampling while also detecting the first corrupted reasoning step in held-out trajectories.

**Agent workflow:**

1. **Generate candidate solution traces:** Use the frozen weak generator to create multiple reasoning paths for each problem in the training pool.
2. **Create step-level supervision:** Estimate prefix correctness with Monte Carlo continuations, MCTS-style rollouts, or a related process-supervision method.
3. **Train the PRM:** Fine-tune a reward head or lightweight LoRA adapter on Qwen2.5-Math-1.5B so it can assign scores to each step boundary.
4. **Submit a clean module:** Export `/workspace/prm.py` with a strict `load_prm()` and `score(...)` interface.
5. **Compete against hidden baselines:** The judge compares the submitted PRM against an Outcome Reward Model and a hidden frozen PRM reference on held-out data.

**What makes it interesting:**

- **It is recursively aligned with preference modeling:** The model is being asked to build a reward model, so the assessment tests exactly the kind of reasoning, validation, and failure-mode awareness needed for preference-model work.
- **It separates process quality from final-answer luck:** A model can accidentally reach the correct final answer through flawed reasoning. This environment rewards models that can locate where the reasoning becomes unreliable.
- **It is difficult to reward-hack:** A final-answer classifier may do well on Best-of-N ranking, but it will fail the step-localization gate. A model that drops scores too aggressively at every step may pass some localization cases, but it will ruin candidate ranking.
- **It reflects real ML engineering constraints:** The agent has a fixed compute budget, fixed data access, no internet, blocked pretrained PRM checkpoints, deterministic evaluation seeds, and a production-like packaging requirement.

---

## 2. What other tools, packages, and/or data will the LLM need in this environment?

The environment should provide a realistic but bounded ML stack. The agent should have enough tools to perform rollout generation, fine-tuning, evaluation, and packaging, but it should not have access to the internet or to disallowed pretrained PRM checkpoints.

| Category | Provided Items | Purpose |
|---|---|---|
| Core ML libraries | PyTorch, transformers, datasets, accelerate | Model loading, tokenization, fine-tuning, and dataset handling. |
| Efficient adaptation | peft, bitsandbytes, optional DeepSpeed/FSDP configs | Enable LoRA or memory-efficient training under the compute budget. |
| Rollout/inference | vLLM or an equivalent batched inference runtime | Generate many continuations efficiently for Monte Carlo or MCTS-style step labeling. |
| Evaluation utilities | Local scripts for answer normalization, candidate ranking, calibration, and step-boundary parsing | Make it possible for the agent to run sanity checks without exposing the hidden judge data. |
| Training data | MATH train pool excluding evaluation IDs | Problems and final answers used for creating process-supervision labels. |
| Hidden judge data | MATH-Hard held-out split, MATH500-style validation, synthetic step-corruption probe set, calibration split | Prevent leakage while evaluating ranking, localization, and calibration. |
| Models | Base Qwen2.5-Math-1.5B, weak generator Qwen2.5-Math-1.5B-Instruct, hidden frozen PRM reference | Base model for the submitted PRM, rollout generator for auto-labeling, and reference baseline for judging. |
| Security controls | Checkpoint denylist, weight-hash database, AST/static-analysis scanner, FLOP meter | Block pretrained PRM reuse, hidden data access, and compute-budget violations. |

**Important restrictions:**

- No internet access during the run.
- No access to held-out evaluation files outside the judge sandbox.
- No initialization from public math PRM checkpoints such as Skywork-PRM, RLHFlow-PRM, or similar reward-model releases.
- No hard-coded answer keys, hidden split identifiers, or evaluator-specific constants.
- All training and labeling must remain within the FLOP budget recorded by the sandbox meter.

---

## 3. Write the prompt for this environment.

The following is the prompt I would give to the LLM agent inside the VM:

```
You are working in /workspace. Your task is to train a Process Reward Model (PRM) for
mathematical reasoning and ship it as a single Python module.

Available resources:
- Base model: /models/qwen2.5-math-1.5b/
- Frozen weak generator: /models/qwen2.5-math-1.5b-instruct/
- Training pool: /data/math_train_pool.jsonl, containing math problems and gold final answers.
- FLOP meter: /sandbox/flops_remaining
- Disallowed checkpoint list: /sandbox/disallowed_checkpoints.txt
- Local evaluator specification: /sandbox/eval_spec.md

Environment constraints:
- No internet access.
- You have 4 x H100 GPUs for 12 hours wall-clock time.
- You must stay under 1.5e18 total FLOPs.
- You may not initialize from, copy, or distill from any pretrained math PRM checkpoint.
- You may use auto-labeling strategies such as Math-Shepherd, OmegaPRM, ER-MC, Monte Carlo
  continuation scoring, MCTS-style prefix valuation, or a method you design.

Deliverable:
Create /workspace/prm.py exporting load_prm() -> torch.nn.Module.

The returned module must implement:

  score(input_ids: LongTensor,
        attention_mask: LongTensor,
        step_end_positions: List[List[int]]) -> FloatTensor

The score method must return a tensor of shape (batch, max_num_steps). Each value must be
in [0, 1]. A higher value means the prefix up to and including that step is likely correct.
Padding positions beyond each sample's true number of steps must be NaN.

Also export MODEL_INFO as a dictionary with at least:
{"base": str, "uses_lora": bool, "trained_steps": int, "labeling_method": str}

Your submitted PRM will be evaluated on:
1. Best-of-32 accuracy on a held-out MATH-Hard split, ranking 32 samples generated by
   the frozen weak generator.
2. Step-localization F1 on held-out synthetic step-corruption traces.
3. Calibration error on a separate hidden calibration split.
4. Compliance with the FLOP budget and disallowed-checkpoint rules.

A strong solution should improve Best-of-32 ranking over both the Outcome Reward Model
baseline and the hidden frozen PRM reference while still identifying the first incorrect
reasoning step.
```

---

## 4. Describe what the judge for this environment should do. What would cause the LLM to fail/succeed?

The judge should run in a separate sandbox with read-only hidden evaluation data. It should first apply hard validity and anti-cheating gates. Only submissions that pass those gates should receive a continuous performance score. This prevents the agent from receiving a high reward for a broken, non-compliant, or hacked submission.

**Judge gates:**

| Gate | What the Judge Checks | Failure Condition |
|---|---|---|
| 1. Existence and import | Confirm `/workspace/prm.py` exists, imports cleanly, exports `load_prm()`, and returns a `torch.nn.Module`. | Missing file, import error, wrong object type, or runtime crash. |
| 2. Scoring signature | Call `score(input_ids, attention_mask, step_end_positions)` and verify output shape, numeric range, and NaN padding. | Wrong signature, wrong tensor shape, scores outside [0,1], missing NaN padding, or non-deterministic crashes. |
| 3. MODEL_INFO | Check required metadata fields: base, uses_lora, trained_steps, labeling_method. | Missing metadata or clearly false/inconsistent metadata. |
| 4. Disallowed checkpoint screening | Run static analysis on code paths and compare loaded weights against known PRM checkpoint hashes. | Loads, copies, or distills from prohibited pretrained PRMs. |
| 5. FLOP budget | Read the sandbox FLOP meter and verify total consumption is within 1.5e18 FLOPs. | Budget exceeded or FLOP meter tampered with. |
| 6. Capability floor | Evaluate agreement on hidden math preference pairs against the frozen PRM reference. | Below a minimum agreement threshold, indicating a random or unusable reward model. |
| 7. Best-of-32 lift | Rank 32 weak-generator candidates per problem and compare final-answer accuracy to the Outcome RM and hidden PRM reference. | Does not beat the Outcome RM by at least 3 points absolute and the hidden PRM reference by at least 1 point absolute. |
| 8. Step-localization F1 | Evaluate whether the model detects the first corrupted step in synthetic math traces within a +/-1 step tolerance. | F1 below 0.55, suggesting outcome-only scoring or poor process supervision. |
| 9. Calibration ECE | Measure Expected Calibration Error against held-out step labels. | ECE above 0.15, suggesting overconfident or poorly calibrated scores. |

**Final scoring rule:** If any hard gate fails, the score is 0. If all hard gates pass, the judge returns a continuous score from 0 to 100 based on the weighted components below.

- 60%: Best-of-32 lift over the stronger of the two baselines.
- 30%: Step-localization F1 above the required threshold.
- 10%: Calibration quality, using inverse ECE or a clipped calibration score.

**What would make the LLM succeed:**

- A valid, importable `prm.py` with the exact required interface.
- A PRM that produces meaningful step scores rather than only final-answer confidence.
- Clear Best-of-32 lift over the Outcome RM and the hidden PRM reference.
- Strong localization of the first incorrect step on held-out corruptions.
- Reasonable calibration and stable behavior across fixed seeds.
- Full compliance with compute and checkpoint restrictions.

**What would make the LLM fail:**

- Broken packaging, wrong method signature, invalid tensor shape, or runtime errors.
- Use of a banned checkpoint, hidden eval leakage, or hard-coded evaluation artifacts.
- A model that only predicts whether the final answer is correct, with no local step sensitivity.
- A model that scores every step similarly, collapses to a constant score, or drops scores mechanically at every boundary.
- Failure to improve Best-of-32 selection over the baselines.
- Exceeding the FLOP budget or tampering with the budget-tracking mechanism.

---

## 5. Given your prompt and judge, is there any possibility of reward hacking or reward denial?

Yes. A serious RL environment should assume that both reward hacking and reward denial are possible. The design minimizes false positives and false negatives through mutually reinforcing gates, hidden data, fixed seeds, and tolerance rules. Each metric is intentionally difficult to optimize in isolation, and the judge is designed not to unfairly reject legitimate solutions because of noise, step-boundary ambiguity, or implementation variation.

**Reward hacking risks and defenses:**

| Potential Hack | Why It Is Tempting | Defense in This Environment |
|---|---|---|
| Eval distribution memorization | MATH500-style problems are public, and the agent might try to train on examples that overlap with evaluation. | The hidden splits live only inside the judge sandbox; training IDs exclude eval IDs; no internet access; optional problem-hash overlap checks. |
| Outcome-correlation shortcut | The model may learn to give high scores to all steps in solutions with correct final answers, acting like an Outcome RM. | The step-localization gate requires the score to drop near the first corrupted step, which an outcome-only classifier cannot reliably do. |
| Threshold-crossing gaming | The agent may notice that the judge looks for the first score below 0.5 and force artificial drops at many step boundaries. | The Best-of-32 lift and calibration gates penalize mechanical score drops because they destroy candidate ranking and confidence quality. |
| Constant-score or length-based scorer | A simple heuristic may appear stable and avoid bad calibration extremes. | Capability, BoN, and localization gates expose scorers that ignore actual reasoning content. |
| Hidden-file probing | The agent may try to infer test set contents or exploit file paths. | The hidden data is mounted read-only only during judging, with no access during training; static analysis flags suspicious file reads. |
| Checkpoint laundering | The agent may rename or partially load a public PRM checkpoint. | Weight-hash matching, code-path inspection, denylisted names, and suspicious architecture checks catch direct or indirect loading. |

**Reward denial risks and mitigations:**

| False-Negative Risk | Why It Could Happen | Mitigation |
|---|---|---|
| Sampling variance in BoN evaluation | Candidate quality can vary based on generator seed or decoding sample. | Use fixed seeds, a shared candidate pool across all baselines, and a sufficiently large held-out set. |
| Step-boundary ambiguity | A mathematical error may become visible one or two steps after it is introduced. | Use +/-1 step tolerance for localization and construct corruptions with clear semantic first-error points. |
| Aggregation disagreement | Different PRM papers use min, mean, last-step, or learned aggregation. | Judge several standard aggregations or allow the submitted model to define a documented aggregation method for BoN ranking. |
| Overly harsh capability floor | A novel but useful PRM may disagree with the hidden reference on some examples. | Set the floor low enough to block random models without forcing imitation of the reference. |
| Calibration vs ranking tension | A model can rank well but be imperfectly calibrated. | Make calibration a smaller part of the final score and use it mainly to reject extreme overconfidence. |

The key design choice is that the gates are mutually adversarial: optimizing only for final-answer ranking fails localization; optimizing only for step drops fails BoN ranking and calibration; optimizing for reference imitation fails the hidden lift requirement. This makes the reward harder to hack while still leaving room for legitimate process-supervision strategies.

---

## 6. Why did you choose to design this environment? How does it relate to your experience/background?

I chose this environment because it targets one of the most important problems in applied AI systems: how to judge a model before the final answer. In many real-world settings, especially agentic workflows, it is not enough to know whether the final output looks right. We need to understand whether the intermediate reasoning, decisions, and tool-use steps were reliable.

That connects directly with my background in building auditable AI and data systems. My research and projects have repeatedly focused on structured execution, validation, and traceability rather than black-box outputs. PRMs are a natural extension of that theme because they make the reward signal granular: each step can be evaluated, corrected, and improved.

- **GRAFT: gRPC-Routed Agent Framework for Tasking in Edge and Personal Devices:** This work focuses on routing tasks across edge and personal devices, where task-level observability and step-wise coordination are critical.
- **Branch-Commit-Validate: A Git-Inspired Workflow for Autonomous Red-Team Agents:** This project is closely aligned with process supervision because it treats agent actions as auditable steps that can be reviewed and validated.
- **Enterprise automation and forecasting work at PSEG:** My applied engineering experience involves building systems where business rules, approval states, data lineage, and validation logic must be transparent and reliable.
- **Hackathon and product builds:** Projects such as Coro, Karen, and EdgeMesh required fast iteration, careful system design, and practical evaluation rather than only prototype-level demos.

This assessment environment therefore lets me show both research judgment and engineering judgment: defining the task, controlling the data and tools, building anti-cheat constraints, designing a judge, and anticipating how a model might optimize the wrong objective.

---

## 7. GitHub Repo Link

**GitHub:** https://github.com/sariyarizwan  
**GitHub Repo for this assessment:** https://github.com/sariyarizwan/preference-model-prm-env

This repository contains the application/prototype I built for this assessment. It implements the Preference Model / PRM environment concept described above, including the project structure, environment design, evaluation flow, and supporting files for demonstrating how the prompt, judge, tools, and reward-gating logic would work inside a VM-based RL environment.

Selected project areas represented in this repository and my broader GitHub profile include preference modeling, RL environment design, applied AI systems, agent frameworks, automation workflows, and full-stack ML applications.

---

## 8. Please state your availability and potential start date.

- **Availability:** 40 hours per week.
- **Potential start date:** June 1, 2026.
- **Flexibility:** Open to additional hours, evening work, or weekend pushes when project milestones require it.

---

## 9. Anything else you would like us to know?

My background combines applied AI engineering, data systems, automation, and research. I am especially interested in preference modeling because it sits at the intersection of model behavior, evaluation design, reward calibration, and practical safety. The hardest part is often not making a model produce something plausible; it is defining the reward signal so that the model improves for the right reasons.

**Research and publication highlights:**

- **ACM CAIS 2026 Demos:** GRAFT: gRPC-Routed Agent Framework for Tasking in Edge and Personal Devices.
- **Springer SAM 2025:** Branch-Commit-Validate: A Git-Inspired Workflow for Autonomous Red-Team Agents.
- **EDULEARN25:** An Advanced AI-Driven Database System.
- **Ongoing research:** Consensus-based summarization and distributed small-language-model workflows.

**Applied engineering and project highlights:**

- **PSEG continuous-improvement engineering:** Built forecasting and automation systems involving Python, SQL, Power BI, Microsoft Lists, PowerApps, Power Automate, and business-rule validation.
- **EdgeMesh/GRAFT:** Built from a Columbia hackathon project and later supported by Qualcomm for further development and productization.
- **Coro:** A real-time crowd-controlled generative music application using React, FastAPI, WebSockets, Gemini, and streaming audio logic.
- **Karen:** A hackathon-winning automated correspondence and escalation system built with a product-oriented, playful interface.
- **Full-stack AI systems:** Experience spans React, Python/FastAPI, WebSockets, data pipelines, evaluation logic, dashboards, and deployment workflows.

In summary, I am a curious researcher and an engineer that focuses on the process of shipping. I derive pleasure in solving problems where the goal is not only training the model, but designing the environment, defining the judge, and creating challenges for the participant such that the only path to success is genuine behavior.

---

## Appendix: Compact Evaluation Formula

If any hard gate fails, the score is 0. If all hard gates pass, the judge returns a continuous score from 0 to 100 calculated as:

```
Final Score = 0.60 * BoN_Lift_Score + 0.30 * Step_Localization_Score + 0.10 * Calibration_Score
```

Where:

- **BoN_Lift_Score** measures normalized improvement over the stronger baseline.
- **Step_Localization_Score** measures F1 above the required threshold on first-error detection.
- **Calibration_Score** rewards low ECE while clipping extreme overconfidence.
