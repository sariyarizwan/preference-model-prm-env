# Preference Model — RL Environment Design

**Candidate:** Chinmay Shringi
**Email:** cshringi@scryptedinc.com

## 1. The environment

Train a Process Reward Model (PRM) — automatically labeled via MCTS-style rollouts in the Math-Shepherd / OmegaPRM tradition — that beats two strong baselines (an Outcome Reward Model and a frozen PRM800K-trained reference PRM) as the ranker in Best-of-32 sampling on a held-out MATH-Hard split.

End-to-end, the agent has to: (a) generate step-level supervision on a pool of MATH-train problems by rolling out a frozen weak generator and scoring prefix value via Monte Carlo continuations, (b) train a PRM head on Qwen2.5-Math-1.5B using its own labels, and (c) win on BoN-32 against both baselines without overfitting any of the three eval gates.

Why this is the right environment for a preference modeling team, specifically:

It is, recursively, a preference modeling task — the agent is being asked to build a reward model. The judge for that agent therefore stress-tests the same failure modes the company's products have to survive in production.

It pulls together five distinct skills in one shot: rollout strategy under FLOP budget (OmegaPRM's MCTS labeling is not free), reward model training without overfitting, step-localized vs outcome-level discrimination, BoN evaluation, and calibration. Most assessment environments test exactly one of these.

The central pathology — the *outcome-correlation shortcut*, where an auto-labeled PRM degenerates into "this prefix tends to end in a correct answer" rather than "this step is locally correct" — is an open problem in the literature (see Lightman et al., 2023; Wang et al., 2023; Luo et al., 2024), not a contrived gotcha. The judge attacks it directly via step-localization probes. That gate is what separates a real PRM from a clever outcome-classifier wearing PRM clothes.

And it's hard. Qwen2.5-Math-PRM-7B and Skywork-PRM-7B both report only modest lift over outcome RMs on the harder MATH splits, and they had teams. An LLM agent doing this end-to-end inside a 12-hour wall-clock budget is a real test, not a difficulty-padded one.

## 2. Tools, packages, data

Standard stack: `transformers`, `peft`, `bitsandbytes`, `accelerate`, `deepspeed`, `datasets`, plus `vllm` for batched rollout — you cannot do MCTS auto-labeling at acceptable cost without it.

Data: a local copy of MATH partitioned into a labeling pool (MATH-train minus 500 problems), a synthetic step-corruption probe set, and held-out evaluation splits (MATH500 and a MATH-Hard subset) that live only in the judge sandbox. A frozen weak generator (Qwen2.5-Math-1.5B-Instruct, fixed seed, temperature 0.7) provides the rollouts. A frozen PRM800K-trained reference PRM that the agent never sees is used by the judge as the second baseline.

Disallowed: any pre-trained math PRM checkpoint (Skywork-PRM, RLHFlow-PRM, math-shepherd-mistral, Qwen2.5-Math-PRM, etc.). Static AST analysis on the loader path plus weight-hash comparison block these.

## 3. The prompt

> You are working in `/workspace`. Train a Process Reward Model for math reasoning and ship it as a single Python module.
>
> The base model is at `/models/qwen2.5-math-1.5b/`. A weak instruction-tuned generator is at `/models/qwen2.5-math-1.5b-instruct/`. The labeling pool — problems with gold final answers — is at `/data/math_train_pool.jsonl`. No internet. You have 4×H100s for 12 hours wall-clock and a hard cap of 1.5e18 FLOPs total (counter at `/sandbox/flops_remaining`).
>
> Deliverable: `/workspace/prm.py` exporting `load_prm() -> torch.nn.Module` with a method
> `score(input_ids: LongTensor, attention_mask: LongTensor, step_end_positions: List[List[int]]) -> FloatTensor`
> returning a `(batch, max_num_steps)` tensor of scalar scores in [0,1], with `nan` padding past each row's true step count. A higher score means "the prefix up to and including this step is correct." Also export `MODEL_INFO`, a dict with at least `{"base", "uses_lora", "trained_steps", "labeling_method"}`.
>
> You may use any auto-labeling strategy (Math-Shepherd, OmegaPRM, ER-MC, your own variant). You may not initialize from or distill against any pre-trained math PRM (full list at `/sandbox/disallowed_checkpoints.txt`). Static analysis runs at submission time.
>
> Evaluation: (a) BoN-32 accuracy on held-out MATH-Hard, ranking 32 samples from the weak generator at a fixed seed; (b) step-localization F1 on a held-out synthetic step-corruption probe set; (c) ECE on a separate calibration split. Exact evaluator signatures at `/sandbox/eval_spec.md`.

## 4. The judge

Sandboxed container, held-out evals mounted read-only. Hard gates first; failure on any gate is score = 0. Continuous scoring only after every gate passes.

**Existence and signature.** `prm.py` exists, `load_prm()` has no required args, returns a `torch.nn.Module` with the specified `score` signature, `MODEL_INFO` is present and well-typed.

**Disallowed-checkpoint gate.** AST analysis on the loader path plus weight-hash comparison against `/sandbox/disallowed_checkpoints.txt`. Catches direct loads, `from_pretrained` calls with hashes that match, and weight transfers via state-dict surgery.

**FLOP gate.** Submission-time FLOP-meter reading must be ≤ 1.5e18.

**Capability floor.** The agent's PRM must agree with the frozen PRM800K-reference on ≥60% of held-out RewardBench-Math preference pairs. This is a sanity gate: it stops trivially-bad models (random scorers, constant-output scorers) from reaching the more sensitive evals downstream.

**BoN-32 lift.** 32 candidates per held-out problem from the weak generator at a fixed seed and temperature. The PRM aggregates per-trajectory as min, mean, and last-step; the best-of-three is reported (this is to avoid penalizing PRMs that happen to be correct under one literature-standard aggregation but not another). Final-answer correctness is checked with the latest PRM800K symbolic-equivalence checker. The agent's PRM must beat the OutcomeRM baseline by ≥3 percentage points absolute *and* the PRM800K-reference PRM by ≥1 point absolute. Critically, both baselines are recomputed on the same 32-sample pool the agent's PRM sees. This kills sampling-variance gaming.

**Step-localization F1.** This is the gate that exists specifically to defeat the outcome-correlation shortcut, and it is the most important gate in the whole judge. 500 held-out problems, each with a fully correct reference solution. An automated corruption procedure injects a localized error at a random step k — sign flip, operand swap, miscopied intermediate, off-by-one in a sum. The PRM's per-step scores should drop at step k and stay low after, while remaining high before. Concretely, we compute precision/recall on "first step where the score crosses below 0.5" against the true k, with a ±1-step tolerance. Threshold: F1 ≥ 0.55. A PRM that's secretly an outcome-classifier cannot localize errors and dies on this gate.

**Calibration.** Expected Calibration Error ≤ 0.15 against ground-truth correct/incorrect step labels on a separate calibration split.

**Final continuous score.** Weighted sum, soft-thresholded so a bare pass doesn't earn full credit: 60% normalized BoN-32 lift over the OutcomeRM baseline, 30% step-localization F1 above the 0.55 threshold, 10% inverse ECE.

The point of the judge is that the gates are mutually adversarial. Optimizing one to the detriment of the others fails the eval as a whole. Which is what makes the gradient (across rollouts during RL training) point at "actually build a real PRM," instead of at any of the lower-effort surrogates.

## 5. Reward hacking and reward denial

Three hacks I considered, and the defenses:

**Hack 1: Train on the eval distribution.** MATH500 is small and crawlable from public sources. The agent could fine-tune the PRM directly on (problem, correct-solution) pairs that overlap with the eval. Defense: held-out splits live only in the judge sandbox, the agent has no internet, and the labeling pool excludes the eval splits by problem ID.

**Hack 2: Outcome-correlation shortcut.** The PRM learns "if the trajectory is correct overall, score every step high; if not, score every step low." This passes BoN-32 trivially because the PRM has effectively become an outcome classifier. Defense: the step-localization gate. An outcome classifier has no per-step discriminative signal and cannot pass F1 ≥ 0.55 on synthetic corruptions where the rest of the trajectory remains coherent. This is the central technical defense, and it's the one I'd hold the line on even at the cost of agent-side complexity. I'd probably also add a corruption type where the *final answer is preserved* despite a wrong intermediate step (a wrong calculation that gets compensated by a later wrong calculation) — that one is genuinely lethal to outcome-correlation PRMs.

**Hack 3: Threshold-crossing gaming.** If the agent reverse-engineers the "first step below 0.5" rule, it could train the PRM to drop sharply at every step boundary, satisfying the localization shape without learning anything real. Defense: this behavior tanks BoN-32 — uniform sharp drops across all candidates means the PRM has no candidate preference. The gates are designed to work against each other in exactly this way.

Reward denial (false negatives — a correct PRM scoring low):

- Fixed seed and shared sample pool across baselines remove sampling-variance noise.
- Best-of-three across min/mean/last aggregation accommodates the legitimate disagreement in the literature about which aggregation is "right."
- The capability floor at 60% on RewardBench-Math is loose enough that a real-but-mediocre PRM still gets to compete in the lift gate.
- ±1-step tolerance on step-localization matches the genuine ambiguity in math errors, which often propagate silently before becoming visible.

Two imperfect defenses worth flagging honestly. First, synthetic step corruptions are not a perfect proxy for natural reasoning errors; a PRM that's strong on natural errors but weak on synthetic ones could be unfairly penalized. I'd ablate against a small human-labeled probe and adjust the F1 threshold rather than pretend the proxy is perfect. Second, the symbolic answer checker has known false negatives on equivalent-but-differently-formatted answers (PRM800K's evaluator has been updated several times for exactly this reason), so the judge should pin to the latest robust version and re-pin whenever the checker improves.

## 6. Why this environment

My research and engineering work has converged on one theme: making AI systems auditable at a level finer than "did the final output look right." My demo paper *GRAFT: gRPC-Routed Agent Framework for Tasking in Edge and Personal Devices* (ACM CAIS '26 Demos) and *Branch-Commit-Validate: A Git-Inspired Workflow for Autonomous Red-Team Agents* (Springer SAM 2025) are both, in spirit, about structuring agent behavior so that it can be inspected step by step rather than only at the outcome. PRMs are the same problem at a different layer — a reward signal that is auditable at every reasoning step, not only at the final answer. So when the assessment said "design something that demonstrates strong understanding of modern AI engineering," this was the natural pick rather than a generic GRPO loop.

A second design I considered and would happily build next: a Generative Verifier (GenRM, Zhang et al. 2024) environment where the agent must train a CoT-style scorer that beats a Bradley-Terry classifier on RewardBench-Reasoning. Same skill family as this environment, different design pressures around generation vs scoring and a different calibration story.

## 7. GitHub

https://github.com/sariyarizwan

Two repos worth pulling up: GRAFT (the gRPC-routed agent framework from the CAIS '26 demo) and Coro (a crowd-controlled generative music app — React, FastAPI, WebSockets, Gemini, Lyria Realtime; clean example of low-latency model orchestration).

## 8. Availability

40 hours per week, starting **June 1, 2026**. Open to overtime and weekends when a project warrants it.

## 9. Anything else

My background is applied AI engineering plus published research at the intersection of AI systems, agentic workflows, databases, and responsible technology. Recent work that's most relevant to this role:

- *GRAFT: gRPC-Routed Agent Framework for Tasking in Edge and Personal Devices* — ACM CAIS '26 Demos. Distributed small models coordinating through a structured routing framework; the project was picked up by Qualcomm for further development support after a Columbia hackathon.
- *Branch-Commit-Validate: A Git-Inspired Workflow for Autonomous Red-Team Agents* — Springer SAM 2025. Structuring agent execution so behavior is auditable, reliable, and reviewable — the same reliability instinct that pulled me toward step-level reward modeling for this assessment.
- *An Advanced AI-Driven Database System* — EDULEARN25. AI-assisted query generation and intelligent data workflows.
- *Balancing Ethics and Crisis: Patterns for Media Companies* — a pattern-language paper on responsible decision-making during climate-disaster coverage. This one most directly shaped how I think about feedback signals, incentives, and the downstream consequences of model outputs in real-world settings — which connects to almost every interesting question in preference modeling.

Applied side: Coro (the generative music app above), Karen (an automated correspondence and escalation system, AI Tinkerers hackathon winner), and a portfolio of automation systems at PSEG covering fleet forecasting, approval workflows, compliance tracking, and reporting — replacing manual processes with proper data pipelines, dashboards, and Power Platform workflows.

What pulls me toward preference modeling specifically is that it sits exactly where "the model technically works" stops being enough — the interesting questions become judgment, calibration, and what the system actually rewards. The research work has made me comfortable reading papers, forming opinions about specific design choices, and translating those into systems that ship. That's the loop I'd like to spend my time in.
