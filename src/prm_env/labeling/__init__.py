"""Auto-labeling strategies (Math-Shepherd, OmegaPRM, etc.).

The reference implementations in this package are deliberately
lightweight scaffolds. A real RL training run would invoke vLLM rollouts
under a FLOP budget; here we provide the pipeline shape and the data
contract so an agent can be evaluated on its labeling choices.
"""
