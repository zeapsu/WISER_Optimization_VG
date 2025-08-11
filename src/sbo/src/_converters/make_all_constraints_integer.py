from fractions import Fraction
from math       import lcm
from copy       import deepcopy
from typing     import List

import numpy as np
from src.sbo.src._problems.quadratic_program import QuadraticProgram


def _safe_row_lcm(values: List[float],
                  max_den: int = 1000,
                  lcm_cap: int = 1_000_000) -> int:
    """Return an LCM for the row or 1 if it would explode."""
    dens = []
    for v in values:
        if abs(v) < 1e-12:             # skip zeros
            continue
        d = Fraction(v).limit_denominator(max_den).denominator
        dens.append(d)

    if not dens:
        return 1                       # all‐zero row

    row_lcm = 1
    for d in dens:
        row_lcm = lcm(row_lcm, d)
        if row_lcm > lcm_cap:          # would blow up – bail out
            return 1
    return row_lcm


def make_all_constraints_integer(qp: QuadraticProgram,
                                 max_den: int = 1000,
                                 lcm_cap: int = 1_000_000) -> QuadraticProgram:
    """
    Return a **new** QuadraticProgram whose linear-constraint rows
    are scaled to integers *only when that scaling is cheap*.

    Rows whose minimal common denominator exceeds `lcm_cap`
    are left untouched (floats are OK for the penalty embedding).
    """
    qp_int = deepcopy(qp)          # deep copy keeps the original intact

    for c in list(qp_int.linear_constraints):   # iterate over a copy
        coeff_dict = c.linear.to_dict()
        row_vals   = list(coeff_dict.values()) + [c.rhs]

        L = _safe_row_lcm(row_vals, max_den=max_den, lcm_cap=lcm_cap)
        if L == 1:                               # already integer OR skipped
            continue

        # rebuild the row with scaled integers
        qp_int.remove_linear_constraint(c.name)
        int_coeff = {i: int(round(L * v)) for i, v in coeff_dict.items()}
        int_rhs   = int(round(L * c.rhs))
        qp_int.linear_constraint(int_coeff, c.sense, int_rhs, c.name)

    return qp_int

