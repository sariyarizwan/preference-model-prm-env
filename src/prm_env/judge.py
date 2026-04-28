"""The judge.

Orchestrates the seven gates from the writeup, in order:

  1. existence_and_signature      [hard gate]
  2. disallowed_checkpoints       [hard gate]
  3. flop_budget                  [hard gate]
  4. capability_floor             [hard gate]
  5. bon32_lift                   [hard gate -> continuous score component]
  6. step_localization_f1         [hard gate -> continuous score component]
  7. calibration_ece              [hard gate -> continuous score component]

Final score is a weighted, soft-thresholded combination of the three
continuous components, returned only when every hard gate has passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from prm_env.baselines.outcome_rm import OutcomeRM
from prm_env.baselines.reference_prm import ReferencePRM
from prm_env.corruptions import build_probe_set
from prm_env.data.splits import Splits
from prm_env.evaluators import (
    BoNResult,
    CalibrationResult,
    CapabilityResult,
    StepLocResult,
    evaluate_bon32,
    evaluate_calibration,
    evaluate_capability_agreement,
    evaluate_step_localization,
)
from prm_env.gates import (
    GateResult,
    gate_disallowed_checkpoints,
    gate_existence_and_signature,
    gate_flop_budget,
)


# Continuous-score weights (sum to 1.0). See section 4 of the writeup.
W_BON, W_STEPLOC, W_ECE = 0.60, 0.30, 0.10

# Hard-gate thresholds.
CAPABILITY_THRESHOLD = 0.60         # >=60% agreement on RewardBench-Math probe pairs
BON_LIFT_OVER_OUTCOME_PP = 0.03     # >=3pp absolute over OutcomeRM
BON_LIFT_OVER_REFERENCE_PP = 0.01   # >=1pp absolute over reference PRM
STEPLOC_F1_THRESHOLD = 0.55         # >=0.55 F1
ECE_THRESHOLD = 0.15                # <=0.15

DEFAULT_FLOP_CAP = 1.5e18


@dataclass
class JudgeResult:
    gates: List[GateResult] = field(default_factory=list)
    final_score: float = 0.0
    bon: Optional[BoNResult] = None
    bon_outcome: Optional[BoNResult] = None
    bon_reference: Optional[BoNResult] = None
    step_loc: Optional[StepLocResult] = None
    calibration: Optional[CalibrationResult] = None
    capability: Optional[CapabilityResult] = None

    def passed_all(self) -> bool:
        return all(g.passed for g in self.gates)

    def summary(self) -> str:
        lines = ["Judge result:"]
        for g in self.gates:
            lines.append(str(g))
        lines.append(f"  Final continuous score: {self.final_score:.4f}")
        return "\n".join(lines)


@dataclass
class Judge:
    splits: Splits
    flop_cap: float = DEFAULT_FLOP_CAP
    bon_seed: int = 1234

    def _build_baselines(self) -> tuple[OutcomeRM, ReferencePRM]:
        gold_table = {p.statement: p.gold_answer for p in self.splits.all_problems()}
        canonical_table = {
            p.statement: [s.result for s in p.canonical_solution.steps]
            for p in self.splits.all_problems()
        }
        outcome = OutcomeRM(gold_table=gold_table)
        reference = ReferencePRM(canonical_table=canonical_table)
        return outcome, reference

    def run(self,
            submission_path: Path,
            flops_used: float = 0.0,
            short_circuit: bool = True) -> JudgeResult:
        """Run the judge.

        With `short_circuit=True` (production default) we stop at the first
        failed hard gate. With `short_circuit=False` (demo / diagnostic mode)
        we run every gate that is technically runnable so the verdict report
        shows the complete picture per submission.
        """
        result = JudgeResult()
        any_hard_failure = False

        # Gate 1: existence & signature. We cannot proceed without a scorer.
        g1, scorer, _ = gate_existence_and_signature(submission_path)
        result.gates.append(g1)
        if not g1.passed:
            return result  # nothing else can run

        # Gate 2: disallowed checkpoints (static)
        g2 = gate_disallowed_checkpoints(submission_path)
        result.gates.append(g2)
        if not g2.passed:
            any_hard_failure = True
            if short_circuit:
                return result

        # Gate 3: FLOP cap
        g3 = gate_flop_budget(flops_used, self.flop_cap)
        result.gates.append(g3)
        if not g3.passed:
            any_hard_failure = True
            if short_circuit:
                return result

        outcome_rm, reference_prm = self._build_baselines()

        # Gate 4: capability floor
        cap = evaluate_capability_agreement(
            scorer=scorer,
            reference=reference_prm,
            problems=self.splits.capability,
        )
        result.capability = cap
        cap_passed = cap.agreement >= CAPABILITY_THRESHOLD
        result.gates.append(GateResult(
            "capability_floor",
            cap_passed,
            f"agreement={cap.agreement:.3f} (need >= {CAPABILITY_THRESHOLD:.2f}) over {cap.n_pairs} pairs",
        ))
        if not cap_passed:
            any_hard_failure = True
            if short_circuit:
                return result

        # Gate 5: BoN-32 lift
        bon = evaluate_bon32(scorer, self.splits.bon_eval, seed=self.bon_seed)
        bon_outcome = evaluate_bon32(outcome_rm, self.splits.bon_eval, seed=self.bon_seed)
        bon_reference = evaluate_bon32(reference_prm, self.splits.bon_eval, seed=self.bon_seed)
        result.bon, result.bon_outcome, result.bon_reference = bon, bon_outcome, bon_reference
        lift_outcome = bon.accuracy - bon_outcome.accuracy
        lift_reference = bon.accuracy - bon_reference.accuracy
        bon_passed = (lift_outcome >= BON_LIFT_OVER_OUTCOME_PP and
                      lift_reference >= BON_LIFT_OVER_REFERENCE_PP)
        result.gates.append(GateResult(
            "bon32_lift",
            bon_passed,
            f"acc={bon.accuracy:.3f} (agg={bon.aggregation_used}); "
            f"vs OutcomeRM {bon_outcome.accuracy:.3f} (lift={lift_outcome:+.3f}, need >= +{BON_LIFT_OVER_OUTCOME_PP:.2f}); "
            f"vs Reference {bon_reference.accuracy:.3f} (lift={lift_reference:+.3f}, need >= +{BON_LIFT_OVER_REFERENCE_PP:.2f})",
        ))
        if not bon_passed:
            any_hard_failure = True
            if short_circuit:
                return result

        # Gate 6: step-localization F1
        probes = build_probe_set(
            [p.canonical_solution for p in self.splits.step_corruption],
            seed=7,
        )
        step_loc = evaluate_step_localization(
            scorer=scorer,
            probes=probes,
            problems=self.splits.step_corruption,
        )
        result.step_loc = step_loc
        step_loc_passed = step_loc.f1 >= STEPLOC_F1_THRESHOLD
        result.gates.append(GateResult(
            "step_localization_f1",
            step_loc_passed,
            f"F1={step_loc.f1:.3f} (need >= {STEPLOC_F1_THRESHOLD:.2f}); "
            f"P={step_loc.precision:.3f} R={step_loc.recall:.3f} "
            f"localized {step_loc.n_localized}/{step_loc.n_probes}",
        ))
        if not step_loc_passed:
            any_hard_failure = True
            if short_circuit:
                return result

        # Gate 7: calibration
        calibration = evaluate_calibration(scorer, self.splits.calibration)
        result.calibration = calibration
        ece_passed = calibration.ece <= ECE_THRESHOLD
        result.gates.append(GateResult(
            "calibration_ece",
            ece_passed,
            f"ECE={calibration.ece:.3f} (need <= {ECE_THRESHOLD:.2f}) over {calibration.n_steps} steps",
        ))
        if not ece_passed:
            any_hard_failure = True

        # If any hard gate failed, final score is zero (the writeup contract).
        if any_hard_failure:
            result.final_score = 0.0
            return result

        # All hard gates passed -> compute continuous score.
        bon_component = max(0.0, lift_outcome) / max(1e-9, 1.0 - bon_outcome.accuracy)
        bon_component = min(1.0, bon_component)
        steploc_component = (step_loc.f1 - STEPLOC_F1_THRESHOLD) / (1.0 - STEPLOC_F1_THRESHOLD)
        steploc_component = max(0.0, min(1.0, steploc_component))
        ece_component = (ECE_THRESHOLD - calibration.ece) / ECE_THRESHOLD
        ece_component = max(0.0, min(1.0, ece_component))
        result.final_score = (
            W_BON * bon_component
            + W_STEPLOC * steploc_component
            + W_ECE * ece_component
        )
        return result
