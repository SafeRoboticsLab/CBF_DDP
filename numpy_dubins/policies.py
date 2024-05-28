import numpy as np
import time


class ReachabilityLQPolicy:
    def __init__(
            self,
            state_dim,
            action_dim,
            marginFunc,
            env,
            horizon=50,
            Rc=1e-5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.marginFunc = marginFunc
        self.horizon = horizon
        self.env = env
        self.env.reset()

        self.R = np.diag(Rc * np.ones((env.action_dim, )))

        self.tol = 1e-6
        self.eps = 1e-5
        self.max_iters = 30
        self.line_search = "baseline"

    def initialize_trajectory(self, obs, nominal_controls):
        margins = np.zeros((self.horizon, ))
        nominal_states = np.zeros((self.horizon + 1, self.state_dim))
        nominal_states[0] = np.array(obs)

        for t in range(self.horizon):
            obs, nominal_controls[t] = self.env.step(obs, nominal_controls[t])
            nominal_states[t + 1] = np.array(obs)

        return nominal_states, nominal_controls, margins

    def forward_pass(
            self,
            nominal_states,
            nominal_controls,
            K_closed_loop,
            k_open_loop,
            alpha=1.0):
        new_states = np.array(nominal_states)
        new_controls = np.array(nominal_controls)

        obs = np.array(nominal_states[0])
        for t in range(self.horizon):
            new_controls[t] = new_controls[t] + \
                K_closed_loop[t] @ (obs - nominal_states[t]) + alpha * k_open_loop[t]
            obs, new_controls[t] = self.env.step(obs, new_controls[t])
            new_states[t + 1] = np.array(obs)

        reachable_margin, critical_margin, state_margins, critical_index = self.get_margin(
            new_states, new_controls)

        return new_states, new_controls, reachable_margin, critical_margin, state_margins, critical_index

    def get_margin(self, nominal_states, nominal_controls):
        R = self.R

        reachable_margin = np.inf
        critical_margin = np.inf

        state_margins = np.zeros((self.horizon, ))
        critical_index = -1

        t = self.horizon - 1
        # Find margin over truncated horizon
        while t >= 0:
            obs_curr = nominal_states[t:t + 1]
            action_curr = nominal_controls[t:t + 1]

            failure_margin, _, _ = self.marginFunc.eval(
                obs_curr, action_curr)
            state_margins[t] = failure_margin.ravel()[0]

            if (failure_margin < reachable_margin):
                critical_index = t
                reachable_margin = failure_margin.ravel()[0]
                critical_margin = failure_margin.ravel()[0]

            reachable_margin = reachable_margin - 0.5 * \
                nominal_controls[t] @ R @ nominal_controls[t]
            t = t - 1
        return reachable_margin, critical_margin, state_margins, critical_index

    def backward_pass(
            self,
            nominal_states,
            nominal_controls):
        # Perform an ILQ backward pass
        V_x = np.zeros((self.horizon + 1, self.state_dim))
        V_xx = np.zeros((self.horizon + 1, self.state_dim, self.state_dim))

        Q_x = np.zeros((self.horizon, self.state_dim))
        Q_u = np.zeros((self.horizon, self.action_dim))
        Q_xx = np.zeros((self.horizon, self.state_dim, self.state_dim))
        Q_uu = np.zeros((self.horizon, self.action_dim, self.action_dim))
        Q_ux = np.zeros((self.horizon, self.action_dim, self.state_dim))

        margins = np.zeros((self.horizon, ))
        state_margins = np.zeros((self.horizon, ))

        k_open_loop = np.zeros((self.horizon, self.action_dim))
        K_closed_loop = np.zeros(
            (self.horizon, self.action_dim, self.state_dim))

        index_lists = []

        reachable_margin = np.inf

        # Backward pass
        t = self.horizon - 1
        while t >= 0:
            obs_curr = nominal_states[t:t + 1]
            action_curr = nominal_controls[t:t + 1]

            failure_margin, failure_index, failure_subindex = self.marginFunc.eval(
                obs_curr, action_curr)

            if (failure_margin < reachable_margin):
                index_lists.append(
                    ["Failure", failure_index, failure_subindex])
                reachable_margin = failure_margin.ravel()[0]
            else:
                index_lists.append(
                    ["Propagate", failure_index, failure_subindex])
            reachable_margin = reachable_margin - 0.5 * \
                nominal_controls[t] @ self.R @ nominal_controls[t]
            margins[t] = reachable_margin
            state_margins[t] = failure_margin.ravel()[0]
            t = t - 1

        t = self.horizon - 1
        while t > -1:
            current_index = index_lists[self.horizon - 1 - t]
            obs_curr = nominal_states[t:t + 1]
            action_curr = nominal_controls[t:t + 1]

            # Failure derivatives
            c_x_failure = self.marginFunc.dcdx(
                obs_curr, action_curr, current_index[1], current_index[2])
            c_xx_failure = self.marginFunc.dcdx2(
                obs_curr, action_curr, current_index[1], current_index[2])
            Ad, Bd, _, _ = self.env.get_jacobian(
                obs_curr.ravel(), action_curr.ravel())

            if (current_index[0] == "Failure"):
                Q_x[t] = np.array(c_x_failure)
                Q_xx[t] = np.array(c_xx_failure)
                Q_u[t] = -self.R @ nominal_controls[t] + Bd.T @ V_x[t + 1]
                Q_ux[t] = Bd.T @ V_xx[t + 1] @ Ad
                Q_uu[t] = -self.R @ np.eye(action_curr.size) + \
                    Bd.T @ V_xx[t + 1] @ Bd
                Q_uu_delta = -self.R @ np.eye(action_curr.size) + Bd.T @ (
                    V_xx[t + 1] - self.eps * np.eye(self.state_dim)) @ Bd
                Q_ux_delta = Bd.T @ (V_xx[t + 1] -
                                     self.eps * np.eye(self.state_dim)) @ Ad
            else:
                Q_x[t] = Ad.T @ V_x[t + 1]
                Q_xx[t] = Ad.T @ V_xx[t + 1] @ Ad
                Q_u[t] = -self.R @ nominal_controls[t] + Bd.T @ V_x[t + 1]
                Q_ux[t] = Bd.T @ V_xx[t + 1] @ Ad
                Q_uu[t] = -self.R @ np.eye(action_curr.size) + \
                    Bd.T @ V_xx[t + 1] @ Bd
                Q_uu_delta = -self.R @ np.eye(action_curr.size) + Bd.T @ (
                    V_xx[t + 1] - self.eps * np.eye(self.state_dim)) @ Bd
                Q_ux_delta = Bd.T @ (V_xx[t + 1] -
                                     self.eps * np.eye(self.state_dim)) @ Ad

            Q_uu_inv = np.linalg.inv(Q_uu_delta)

            # Signs for maximization
            k_open_loop[t] = - Q_uu_inv @ Q_u[t]
            K_closed_loop[t] = - Q_uu_inv @ Q_ux_delta

            # Update value function derivative for the previous time step
            if (current_index[0] == "Failure"):
                V_x_critical = np.copy(c_x_failure)
                V_xx_critical = np.copy(c_xx_failure)
                V_x[t] = Q_x[t] + Q_ux[t].T @ k_open_loop[t]
                V_xx[t] = Q_xx[t] + Q_ux[t].T @ K_closed_loop[t]
            else:
                V_x[t] = Q_x[t] + Q_ux[t].T @ k_open_loop[t]
                V_xx[t] = Q_xx[t] + Q_ux[t].T @ K_closed_loop[t]

            t = t - 1

        self.Q_u = Q_u
        barrier_constraint_data = {
            "V_xx": V_xx[0],
            "V_x": V_x[0],
            "V_xx_critical": V_xx_critical,
            "V_x_critical": V_x_critical,
            "V_t": margins[0],
            "state_margins": state_margins}

        return K_closed_loop, k_open_loop, barrier_constraint_data

    def get_action(self, initial_state, initial_controls=None, animate=False):
        start_time = time.time()
        # Get initial trajectory with naive controls
        if initial_controls is None:
            initial_controls = np.zeros((self.horizon, self.action_dim))
            initial_controls[:, 0] = 0.01

        states, controls, _ = self.initialize_trajectory(
            initial_state, initial_controls)

        # Update control with ILQ updates
        iters = 0
        J, critical_margin, _, _ = self.get_margin(
            states, controls)

        converged = False
        status = 0
        updated_constraints_data = None

        convergence_sequence = []
        convergence_sequence.append(critical_margin)
        self.margin = 5.0
        while iters < self.max_iters and not converged:
            iters = iters + 1
            # Backward pass
            K_closed_loop, k_open_loop, _ = self.backward_pass(
                states, controls)

            # Choose the best alpha scaling using appropriate line search
            # methods
            alpha_chosen = self.baseline_line_search(
                states, controls, K_closed_loop, k_open_loop, critical_margin, J)

            if alpha_chosen < 1e-13:
                J_new = J
                break

            states, controls, J_new, critical_margin, _, _ = self.forward_pass(
                states, controls, K_closed_loop, k_open_loop, alpha_chosen)

            convergence_sequence.append(critical_margin)

            # Small improvement.
            if J_new > 0 and np.abs((J_new - J) / J) < self.tol:
                converged = True
            J = J_new

        # Backward pass
        _, _, updated_constraints_data = self.backward_pass(
            states, controls)

        process_time = time.time() - start_time
        solver_dict = {
            "states": states,
            "controls": controls,
            "status": status,
            "margin": critical_margin,
            "reachable_margin": J,
            "iteration_no": iters,
            "critical_constraint_type": 0,
            "convergence": convergence_sequence,
            "label": "Optimal safety plan",
            "id": 'Optimal',
            "t_process": process_time,
            "iterations": iters}
        return controls[0], solver_dict, updated_constraints_data

    def baseline_line_search(
            self,
            states,
            controls,
            K_closed_loop,
            k_open_loop,
            critical,
            J,
            beta=0.3):
        alpha = 1.0
        while alpha > 1e-13:
            _, _, J_new, _, _, _ = self.forward_pass(
                states, controls, K_closed_loop, k_open_loop, alpha)

            # Accept if there is improvement
            margin_imp = (J_new - J)
            if margin_imp > 0.:
                return alpha
            alpha = beta * alpha

        return alpha

    def armijo_line_search(
            self,
            states,
            controls,
            K_closed_loop,
            k_open_loop,
            state_margins,
            critical_index,
            J,
            beta=0.3):
        # Armijo only condition - Wolfe condition is not implemented.
        alpha = 1.0

        alpha_converged = False

        while not alpha_converged:
            X, _, J_new, _, _, _ = self.forward_pass(
                states, controls, K_closed_loop, k_open_loop, alpha)

            # cu is zero so we propagate cx one step back and use Qu as
            # gradient
            grad_u = self.Q_u[critical_index - 1]

            delta_u = K_closed_loop[critical_index - 1] @ (
                X[critical_index - 1] - states[critical_index - 1]) + k_open_loop[critical_index - 1]

            t = 0.5 * grad_u @ delta_u

            if J_new - t * alpha >= J:
                alpha_converged = True
                return alpha
            else:
                # Reduce alpha and check for decrease condition
                alpha = beta * alpha
                if alpha < 1e-18:
                    # Stop iterating here
                    return alpha

    def trust_region_ratio(
            self,
            states,
            controls,
            K_closed_loop,
            k_open_loop,
            V_x_critical,
            V_xx_critical,
            state_margins,
            critical_index,
            J,
            beta=0.6):
        alpha = 1.0
        # Margin around which to search for the next descent point.
        alpha_converged = False

        margin_old = state_margins[critical_index]
        while not alpha_converged:
            X, _, J_new, _, state_margins_new, _ = self.forward_pass(
                states, controls, K_closed_loop, k_open_loop, alpha)

            # Find new state margin at old critical point
            margin_new = state_margins_new[critical_index]

            # Trajectory difference
            traj_diff = max([np.linalg.norm(np.array(x_new) - np.array(x_old))
                            for x_new, x_old in zip(np.array(X)[:-1, :2], np.array(states)[:-1, :2])])

            # Decrease in margin based on second order approximation at pinch
            # point.
            x_diff = np.array(
                [
                    (np.array(x_new) - np.array(x_old)) for x_new,
                    x_old in zip(
                        (np.array(X))[
                            critical_index,
                            :],
                        (np.array(states))[
                            critical_index,
                            :])])

            delta_margin_quadratic_approx = 0.5 * \
                (np.transpose(x_diff) @ V_xx_critical[0] + 2 * np.transpose(V_x_critical[0])) @ x_diff

            # Actual decrease and approximated decrease.
            delta_margin_quadratic_actual = margin_new - margin_old

            if delta_margin_quadratic_approx != 0:
                rho = delta_margin_quadratic_actual / delta_margin_quadratic_approx
            else:
                rho = 0.

            # Trajectory difference is within margin and there is improvement.
            # Accept new alpha and change margin.
            # These conditions are obtained from Reach-avoid games:
            # https://github.com/SafeRoboticsLab/Reach-Avoid-Games

            if abs(traj_diff) < self.margin and J_new > J:
                if rho >= 0.85 and abs(traj_diff - self.margin) < 0.01:
                    self.margin = min(1.25 * self.margin, 10)
                elif rho <= 0.15:
                    self.margin = max(0.85 * self.margin, 0.1)

                alpha_converged = True
                return alpha
            else:
                alpha = beta * alpha
                if alpha < 1e-19:
                    return alpha
        return alpha
