"""stat_board — a statistical-analysis board (STORM-like, adversarial).

The `engine` subpackage is the shared, pure-Python statistics core: it does the
actual math and returns JSON-serializable results. Both front-ends — the batch
report board (`/stat-board`) and the interactive advisor (`/stat-advisor`) —
call it as their single source of numeric truth, so conclusions are always
grounded in a real computation rather than a model's guess.
"""

__all__ = ["engine"]
