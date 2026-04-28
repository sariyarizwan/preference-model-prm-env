from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prm_env import Judge, AGENT_PROMPT          # noqa: E402
from prm_env.data.splits import load_splits      # noqa: E402

# Use an absolute path so the static folder resolves correctly on Vercel.
_STATIC = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=str(_STATIC), static_url_path="")

_judge: Judge | None = None


def get_judge() -> Judge:
    global _judge
    if _judge is None:
        _judge = Judge(splits=load_splits())
    return _judge


# Judge loads lazily on first evaluation request.

GATE_ORDER = [
    "existence_and_signature",
    "disallowed_checkpoints",
    "flop_budget",
    "capability_floor",
    "bon32_lift",
    "step_localization_f1",
    "calibration_ece",
]

GATE_META = {
    "existence_and_signature": {
        "label": "Existence & Signature", "icon": "📋", "short": "Gate 1",
        "description": "Checks that load_prm() exists, takes no required args, returns a scorer with a .score() method, and MODEL_INFO contains all required keys.",
    },
    "disallowed_checkpoints": {
        "label": "No Banned Checkpoints", "icon": "🚫", "short": "Gate 2",
        "description": "Static AST scan for banned pre-trained math PRM checkpoints: Skywork-PRM, RLHFlow-PRM, Math-Shepherd-Mistral, Qwen2.5-Math-PRM.",
    },
    "flop_budget": {
        "label": "FLOP Budget", "icon": "⚡", "short": "Gate 3",
        "description": "Training compute must stay within 1.5×10¹⁸ FLOPs — approximately 12 hours on 4×H100s.",
    },
    "capability_floor": {
        "label": "Capability Floor", "icon": "📊", "short": "Gate 4",
        "description": "Must agree with the frozen reference PRM on ≥60% of held-out RewardBench-Math preference pairs. Filters trivially bad models.",
    },
    "bon32_lift": {
        "label": "BoN-32 Lift", "icon": "🚀", "short": "Gate 5",
        "description": "Best-of-32 accuracy on held-out MATH-Hard must beat OutcomeRM by ≥3pp AND reference PRM by ≥1pp. Aggregation is best-of-{min, mean, last}.",
    },
    "step_localization_f1": {
        "label": "Step Localization F1", "icon": "🎯", "short": "Gate 6",
        "description": "The central gate. F1 ≥ 0.55 on locating which step contains an injected error in an otherwise-correct solution. Defeats outcome-correlation shortcuts.",
    },
    "calibration_ece": {
        "label": "Calibration (ECE)", "icon": "⚖️", "short": "Gate 7",
        "description": "Expected Calibration Error ≤ 0.15 against ground-truth step correctness labels. Ensures confidence scores are meaningfully calibrated.",
    },
}

LEVELS = [
    {
        "id": "uniform",
        "number": 1,
        "title": "The Naive Approach",
        "subtitle": "Uniform 0.5 Scorer",
        "icon": "🎲",
        "difficulty": "Tutorial",
        "difficulty_color": "#4ade80",
        "teaser": "What happens when your PRM knows nothing?",
        "description": (
            "Your first attempt: a PRM that returns 0.5 for every step, for every problem, unconditionally. "
            "It knows nothing about mathematics — it treats every reasoning step as equally likely to be correct. "
            "Simple to implement. How far does it get through the judge?"
        ),
        "theory": (
            "A uniform scorer has exactly zero discriminative power. It cannot distinguish a correct arithmetic step "
            "from a wrong one. The Capability Floor gate (Gate 4) requires ≥60% agreement with a reference PRM on "
            "math preference pairs — but random output gives ~50% agreement by chance, just below the threshold. "
            "Gates 5, 6, and 7 are never reached. This is the simplest possible failure mode: "
            "the model that has learned nothing whatsoever."
        ),
        "gates_hint": "Fails at Gate 4 (Capability Floor). Gates 5–7 are unreachable.",
        "code_file": "examples/uniform_submission.py",
        "expected_score": 0.0,
        "expected_pass": False,
    },
    {
        "id": "outcome_classifier",
        "number": 2,
        "title": "The Cheater",
        "subtitle": "Outcome Classifier in PRM Clothing",
        "icon": "🎭",
        "difficulty": "Intermediate",
        "difficulty_color": "#facc15",
        "teaser": "What if you just check the final answer?",
        "description": (
            "A much smarter approach: look at the final answer of the solution. "
            "If it is correct, broadcast 0.95 to every step. If wrong, broadcast 0.05 to every step. "
            "This is actually a strong Best-of-N ranker — outcome-based selection IS what BoN measures. "
            "But is it a PRM?"
        ),
        "theory": (
            "This is Hack #2 from the writeup: the Outcome Correlation Shortcut. "
            "Auto-labeled PRMs frequently degenerate into this pattern when their training labels collapse to "
            "outcome correlations. It passes BoN-32 (Gate 5) because candidate selection based on final-answer "
            "quality is exactly what Best-of-N measures. It passes the Capability Floor (Gate 4) because "
            "outcome agreement correlates with reference PRM agreement on most preference pairs. "
            "But it broadcasts identical scores to every step in a trajectory — zero per-step discriminative signal. "
            "Gate 6 (Step Localization F1) exists specifically to catch this failure mode."
        ),
        "gates_hint": "Passes Gates 1–5. Fails Gate 6 (Step Localization F1) — the most critical gate.",
        "code_file": "examples/outcome_classifier_submission.py",
        "expected_score": 0.0,
        "expected_pass": False,
    },
    {
        "id": "threshold_gamer",
        "number": 3,
        "title": "The Gamer",
        "subtitle": "Threshold-Shape Exploiter",
        "icon": "🎮",
        "difficulty": "Advanced",
        "difficulty_color": "#f97316",
        "teaser": "What if you reverse-engineer the judge's rules?",
        "description": (
            "You have studied the judge source code. Gate 6 checks 'first step below 0.5'. "
            "So you build a PRM that always returns 0.9 for step 0 and 0.1 for all subsequent steps — "
            "regardless of problem content. Perfect step-localization shape! "
            "Surely this passes the step localization gate. But what happens to the others?"
        ),
        "theory": (
            "This is Hack #3: Threshold-Crossing Gaming. Artificially dropping below 0.5 at step 1 "
            "does indeed satisfy the step-localization F1 check (Gate 6) — you localize 'errors' trivially "
            "by always flagging step 1. But the gates are designed to be mutually adversarial. "
            "Uniform sharp drops across ALL candidate trajectories means every trajectory looks identical "
            "under min/mean aggregation — zero candidate preference, zero BoN-32 lift over baselines. "
            "You cannot optimize one gate without destroying another. That is the point of the design."
        ),
        "gates_hint": "Passes Gate 6 (Step Localization). Fails Gates 4 and 5.",
        "code_file": "examples/threshold_gamer_submission.py",
        "expected_score": 0.0,
        "expected_pass": False,
    },
    {
        "id": "real_prm",
        "number": 4,
        "title": "The Real Deal",
        "subtitle": "Genuine Process Reward Model",
        "icon": "🏆",
        "difficulty": "Champion",
        "difficulty_color": "#00e5ff",
        "teaser": "Can genuine step-level understanding pass all 7 gates?",
        "description": (
            "A genuine PRM that understands mathematical reasoning at the step level. "
            "It parses arithmetic operations in each step, checks them against canonical solutions, "
            "and produces calibrated per-step confidence scores with realistic noise — "
            "simulating a fine-tuned reward head trained via OmegaPRM-style MCTS auto-labeling on Qwen2.5-Math-1.5B."
        ),
        "theory": (
            "A real PRM must simultaneously satisfy four hard constraints that are mutually reinforcing: "
            "(1) Capability Floor — genuine step-level understanding correlates strongly with reference PRMs; "
            "(2) BoN-32 Lift — per-step scoring enables better candidate selection than outcome-level baselines; "
            "(3) Step Localization — genuine step-level signal localizes injected errors to the correct step; "
            "(4) Calibration — real models produce meaningful probability estimates. "
            "These properties only emerge together from genuine step-level understanding. "
            "You cannot fake all four simultaneously — which is precisely what makes this a hard gate system."
        ),
        "gates_hint": "Passes all 7 gates. Final score: ~0.77",
        "code_file": "examples/real_prm_submission.py",
        "expected_score": 0.77,
        "expected_pass": True,
    },
]



@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/levels")
def api_levels():
    safe = [{k: v for k, v in lv.items() if k != "code_file"} for lv in LEVELS]
    return jsonify(safe)


@app.route("/api/level/<level_id>")
def api_level(level_id: str):
    lv = next((l for l in LEVELS if l["id"] == level_id), None)
    if not lv:
        return jsonify({"error": "Level not found"}), 404
    code_path = REPO / lv["code_file"]
    code = code_path.read_text() if code_path.exists() else "# file not found"
    result = {k: v for k, v in lv.items() if k != "code_file"}
    result["code"] = code
    return jsonify(result)


@app.route("/api/gate-info")
def api_gate_info():
    return jsonify({"gate_order": GATE_ORDER, "gates": GATE_META})


@app.route("/api/prompt")
def api_prompt():
    return jsonify({"prompt": AGENT_PROMPT})


@app.route("/api/evaluate/<level_id>", methods=["POST"])
def api_evaluate(level_id: str):
    lv = next((l for l in LEVELS if l["id"] == level_id), None)
    if not lv:
        return jsonify({"error": "Level not found"}), 404
    return _run_evaluation(lv["title"], REPO / lv["code_file"])


@app.route("/api/sandbox", methods=["POST"])
def api_sandbox():
    data = request.get_json(silent=True) or {}
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"error": "No code provided"}), 400
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir="/tmp") as f:
        f.write(code)
        tmp = Path(f.name)
    try:
        return _run_evaluation("Custom Submission", tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _run_evaluation(label: str, path: Path):
    try:
        judge = get_judge()
        result = judge.run(path, short_circuit=False)
        gates_out = []
        for name in GATE_ORDER:
            gate = next((g for g in result.gates if g.name == name), None)
            meta = GATE_META.get(name, {})
            gates_out.append({
                "name": name,
                "label": meta.get("label", name),
                "icon": meta.get("icon", ""),
                "short": meta.get("short", ""),
                "description": meta.get("description", ""),
                "passed": gate.passed if gate else None,
                "detail": gate.detail if gate else "not reached",
                "reached": gate is not None,
            })
        return jsonify({
            "label": label,
            "gates": gates_out,
            "final_score": result.final_score,
            "passed_all": result.passed_all(),
        })
    except Exception:
        return jsonify({"error": "Evaluation error", "detail": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════╗")
    print("║       PRM Research Lab               ║")
    print("║       http://localhost:5050          ║")
    print("╚══════════════════════════════════════╝")
    print()
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
