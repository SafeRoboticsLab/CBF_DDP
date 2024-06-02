import numpy as np

import cvxpy as cp
from cvxpy import SolverError


def unroll_task_policy(initial_obs, plan_env, task_policy, horizon):
    task_obses = np.zeros((horizon, plan_env.state_dim))
    initial_obs = initial_obs.ravel()
    task_obses[0] = initial_obs

    for j in range(horizon - 1):
        control_perf = task_policy(plan_env, initial_obs)
        initial_obs, _ = plan_env.step(initial_obs, control_perf)
        task_obses[j + 1] = initial_obs

    return task_obses


def barrier_filter_quadratic_one(P, p, c):
    def is_neg_def(x):
        # Check if a matrix is PSD
        return np.all(np.real(np.linalg.eigvals(x)) <= 0)

    # CVX faces numerical difficulties otherwise
    check_nd = (P < 0)
    # Check if P is PD
    if (check_nd):
        u = cp.Variable((1))
        P = np.array(P)
        p = np.array(p)

        prob = cp.Problem(cp.Minimize(1.0 * cp.square(u[0])),
                          [cp.quad_form(u, P) + p.T @ u + c >= 0])
        try:
            prob.solve(verbose=False)
        except SolverError:
            pass

    if (not check_nd or u[0] is None or prob.status not in [
            "optimal", "optimal_inaccurate"]):
        u = cp.Variable((1))
        prob = cp.Problem(cp.Minimize(1.0 * cp.square(u[0])),
                          [p @ u + c >= 0])
        try:
            prob.solve(verbose=False)
        except SolverError:
            pass

    if prob.status not in ["optimal", "optimal_inaccurate"] or u[0] is None:
        return np.array([0.])
    return np.array([u[0].value])


def barrier_filter_quadratic(P, p, c):

    if p.size == 1:
        return barrier_filter_quadratic_one(P, p, c)

    def is_neg_def(x):
        # Check if a matrix is PSD
        return np.all(np.real(np.linalg.eigvals(x)) <= 0)

    # CVX faces numerical difficulties otherwise
    check_nd = is_neg_def(P)
    # Check if P is PD
    if (check_nd):
        u = cp.Variable((2))
        P = np.array(P)
        p = np.array(p)

        prob = cp.Problem(cp.Minimize(1.0 *
                                      cp.square(u[0]) +
                                      1.0 *
                                      cp.square(u[1])), [cp.quad_form(u, P) +
                                                         p.T @ u +
                                                         c >= 0])
        try:
            prob.solve(verbose=False)
        except SolverError:
            pass

    if (not check_nd or u[0] is None or prob.status not in [
            "optimal", "optimal_inaccurate"]):
        u = cp.Variable((2))
        prob = cp.Problem(cp.Minimize(
            1.0 * cp.square(u[0]) + 1.0 * cp.square(u[1])), [p @ u + c >= 0])
        try:
            prob.solve(verbose=False)
        except SolverError:
            pass

    if prob.status not in ["optimal", "optimal_inaccurate"] or u[0] is None:
        return np.array([0., 0.])
    return np.array([u[0].value, u[1].value])


def barrier_filter_multiple_quadratic(P, p, c):
    def is_neg_def(x):
        # Check if a matrix is PSD
        return np.all(np.real(np.linalg.eigvals(x)) <= 0)

    # CVX faces numerical difficulties otherwise
    check_nd = is_neg_def(P[0]) and is_neg_def(P[1]) and is_neg_def(P[2])

    P0 = np.array(P[0])
    p0 = np.array(p[0])

    P1 = np.array(P[1])
    p1 = np.array(p[1])

    P2 = np.array(P[2])
    p2 = np.array(p[2])

    # Check if P is PD
    if (check_nd):
        u = cp.Variable((2))

        prob = cp.Problem(cp.Minimize(1.0 *
                                      cp.square(u[0]) +
                                      1.0 *
                                      cp.square(u[1])), [cp.quad_form(u, P0) +
                                                         p0.T @ u +
                                                         c >= 0, cp.quad_form(u, P1) +
                                                         p1.T @ u +
                                                         c >= 0, cp.quad_form(u, P2) +
                                                         p2.T @ u +
                                                         c >= 0])
        try:
            prob.solve(verbose=False)
        except SolverError:
            pass

    if (not check_nd or u[0] is None or prob.status not in [
            "optimal", "optimal_inaccurate"]):
        u = cp.Variable((2))
        prob = cp.Problem(cp.Minimize(1.0 *
                                      cp.square(u[0]) +
                                      1.0 *
                                      cp.square(u[1])), [p0 @ u +
                                                         c >= 0, p1 @ u +
                                                         c >= 0, p2 @ u +
                                                         c >= 0])
        try:
            prob.solve(verbose=False)
        except SolverError:
            pass

    if prob.status not in ["optimal", "optimal_inaccurate"] or u[0] is None:
        return np.array([0., 0.])
    return np.array([u[0].value, u[1].value])
