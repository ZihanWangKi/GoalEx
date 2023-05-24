import numpy as np
from typing import List
import pulp


def maximum_set_coverage(
    can_cover: np.ndarray,
    overlap_penalty: float,
    not_cover_penalty: float,
    num_clusters: int = None,
) -> List[int]:
    # Create the problem
    problem = pulp.LpProblem("TextCoveringProblem", pulp.LpMinimize)

    # Define decision variables
    I, J = can_cover.shape
    s = [pulp.LpVariable(f"s_{j}", cat="Binary") for j in range(J)]
    y = [pulp.LpVariable(f"y_{i}", 0, None, cat="Continuous") for i in range(I)]
    z = [pulp.LpVariable(f"z_{i}", 0, None, cat="Continuous") for i in range(I)]

    # Define objective function
    if num_clusters is not None:
        problem += pulp.lpSum(z)
        problem += pulp.lpSum(s) == num_clusters
    else:
        problem += pulp.lpSum(z) + pulp.lpSum(s)

    # Define constraints
    for i in range(I):
        problem += y[i] == pulp.lpSum(can_cover[i, j] * s[j] for j in range(J))
        problem += z[i] >= overlap_penalty * (y[i] - 1)
        problem += z[i] >= not_cover_penalty * (1 - y[i])

    # Solve the problem
    pulp.LpSolverDefault.msg = 0
    problem.solve()

    # Extract the solution
    selected_idxes = [j for j in range(J) if pulp.value(s[j]) == 1]

    return selected_idxes