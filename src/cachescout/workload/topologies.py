"""Six-agent coordination topologies reconstructed from paper Fig. 5.

Agents (Fig. 5 axis labels): P, A, C, T, R, D. Only P (Planner), C (Coder), R (Reviewer) and a
judge are named in the text (Sec. 2.2.3); the mapping of the other labels is unknown.

[UNCERTAIN] Fig. 5's text layer gives each row's non-zero probabilities but not reliably their
column positions. Column placement below is our reconstruction (interpretation, see
docs/decisions.md); it is validated by comparing the entropy reduction R measured on generated
traces with the paper's values (Fig. 4a: 1.00 / 0.78 / 0.57 / 0.12).
"""

AGENTS = ("P", "A", "C", "T", "R", "D")

# Each row: successor -> probability (rows are renormalised; diagonal is always empty in Fig. 5).
TOPOLOGIES: dict[str, dict[str, dict[str, float]]] = {
    # "Pipeline follows a fixed chain through all six agents" (Sec. 2.2.3). Order assumed P..D.
    "pipeline": {
        "P": {"A": 1.0}, "A": {"C": 1.0}, "C": {"T": 1.0}, "T": {"R": 1.0}, "R": {"D": 1.0},
        "D": {},
    },
    # "proposer-challenger exchange between Coder and Reviewer periodically adjudicated by a
    # judge" (Sec. 2.2.3). Fig. 5 shows three rows with a single 1 and one row 0.53/0.47.
    # Assumed: P->C, C->R, judge D->C, R->C 0.53 / R->D 0.47.
    "debate": {
        "P": {"C": 1.0}, "C": {"R": 1.0}, "R": {"C": 0.53, "D": 0.47}, "D": {"C": 1.0},
        "A": {}, "T": {},
    },
    # SelectorGroupChat (Fig. 5 "Selector", R = 0.57). Row values from Fig. 5. Column placement
    # chosen by exhaustive search (216,000 placements) for analytic stationary R = 0.570 that keeps
    # Planner -> Coder dominant and Analyst -> Planner (docs/decisions.md). Rows T/R/D as read.
    "selector": {
        "P": {"C": 0.77, "T": 0.10, "A": 0.07, "R": 0.07},
        "A": {"P": 0.62, "R": 0.23, "T": 0.08, "D": 0.08},
        "C": {"D": 0.50, "T": 0.33, "A": 0.08, "R": 0.08},
        "T": {"P": 0.91, "C": 0.09},
        "R": {"P": 1.0},
        "D": {"P": 1.0},
    },
    # Random (Fig. 5, R = 0.12): five off-diagonal values per row, in column order.
    "random": {
        "P": {"A": 0.17, "C": 0.20, "T": 0.22, "R": 0.20, "D": 0.22},
        "A": {"P": 0.22, "C": 0.31, "T": 0.14, "R": 0.19, "D": 0.14},
        "C": {"P": 0.13, "A": 0.27, "T": 0.22, "R": 0.25, "D": 0.13},
        "T": {"P": 0.22, "A": 0.16, "C": 0.24, "R": 0.16, "D": 0.20},
        "R": {"P": 0.16, "A": 0.27, "C": 0.20, "T": 0.27, "D": 0.10},
        "D": {"P": 0.13, "A": 0.29, "C": 0.20, "T": 0.22, "R": 0.16},
    },
}

PAPER_R = {"pipeline": 1.00, "debate": 0.78, "selector": 0.57, "random": 0.12}  # Fig. 4a


def normalized(topology: str) -> dict[str, dict[str, float]]:
    """Row-normalised transition table for a topology."""
    table = TOPOLOGIES[topology]
    out: dict[str, dict[str, float]] = {}
    for a, row in table.items():
        total = sum(row.values())
        out[a] = {b: p / total for b, p in row.items()} if total > 0 else {}
    return out
