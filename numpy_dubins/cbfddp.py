import copy
import numpy as np
import time

from policies import ReachabilityLQPolicy
from utils import unroll_task_policy, barrier_filter_quadratic

from plotting.plotting import helper_plot


def run_control_barrier_ilq(
        env,
        marginFunc,
        horizon,
        max_steps,
        gamma=0.9,
        barrier_scaling=1.0,
        policy_type="Reachability",
        task_policy=None,
        barrier_tol=-1e-5,
        Rc=1e-3,
        barrier_type="linear",
        plot_folder="./",
        animate=False):
    """
    Control Barrier filtering with iLQ
    """
    plan_env_1 = copy.deepcopy(env)
    plan_env_2 = copy.deepcopy(env)
    run_env = copy.deepcopy(env)

    current_horizon = horizon
    run_env_obs = run_env.reset()

    states_cbf_ilq = np.zeros((max_steps, plan_env_1.state_dim))
    controls_cbf_ilq = np.zeros((max_steps, plan_env_2.action_dim))
    deviations_cbf_ilq = np.zeros((max_steps, plan_env_2.action_dim))
    optimal_safe_controls_ilq = np.zeros((max_steps, plan_env_1.action_dim))
    optimal_values_ilq = np.zeros((max_steps, ))
    types_ilq = np.zeros((max_steps, ))
    process_times = np.zeros((max_steps, ))
    barrier_entries_ilq = np.zeros((max_steps, ))
    barrier_margins = np.zeros((max_steps, ))

    complete_filter_indices = []
    barrier_filter_indices = []
    yaw_filter_indices = []
    barrier_filter_steps = 0
    complete_filter_steps = 0

    reinit_controls = None
    if policy_type == "Reachability":
        lq_policy_1 = ReachabilityLQPolicy(
            plan_env_1.state_dim,
            plan_env_1.action_dim,
            marginFunc,
            plan_env_1,
            horizon=current_horizon,
            Rc=Rc)
        lq_policy_2 = ReachabilityLQPolicy(
            plan_env_2.state_dim,
            plan_env_2.action_dim,
            marginFunc,
            plan_env_2,
            horizon=current_horizon,
            Rc=Rc)

    for time_step in range(max_steps):
        print("Observation", run_env_obs)

        barrier_start_time = time.time()

        # Run first trajectory optimization
        control_safe_1, solver_dict_plan_1, _ = lq_policy_1.get_action(
            np.array(run_env_obs), initial_controls=reinit_controls)
        optimal_safe_controls_ilq[time_step] = control_safe_1

        # Run one counterfactual step with task policy
        control_perf = task_policy(plan_env_1, run_env_obs)
        imag_obs, control_perf = plan_env_2.step(run_env_obs, control_perf)

        # Initialize controls and perform second trajectory optimization
        boot_controls = solver_dict_plan_1['controls']
        boot_controls[0:-1] = boot_controls[1:]
        _, solver_dict_plan_2, constraints_data_plan_2 = lq_policy_2.get_action(
            np.array(imag_obs), initial_controls=boot_controls)
        boot_controls = solver_dict_plan_2['controls']

        # Initialize for DDP-CBF QCQP optimization
        control_cbf_candidate = control_perf.copy()
        constraint_violation = solver_dict_plan_2['reachable_margin'] - gamma * (
            solver_dict_plan_1['reachable_margin'])
        scaled_c = constraint_violation

        # Auxiliary storage for animation
        if constraint_violation < 0:
            solver_dict_plan_2['task_active'] = False
        else:
            solver_dict_plan_2['task_active'] = True

        if animate:
            # Animation creation overheads.
            solver_dict_plan_2['trace_id'] = 'Barrier'
            solver_dict_plan_2['title'] = 'Barrier filtering'
            solver_dict_plan_2['trace_label'] = "CBF-DDP" + \
                "($\\gamma$=" + str(gamma) + ")"
            solver_dict_plan_2['previous_trace'] = states_cbf_ilq[0:time_step]
            solver_dict_plan_2['run_steps'] = solver_dict_plan_2['states'].shape[0]
            solver_dict_plan_2['task_trace'] = unroll_task_policy(
                run_env_obs, copy.deepcopy(plan_env_1), task_policy, horizon=current_horizon)
            solver_dict_plan_2['controls_deviation'] = solver_dict_plan_2['controls']
            helper_plot(
                [solver_dict_plan_2],
                marginFunc.obstacle_center,
                marginFunc.obstacle_radius,
                save_prefix=str(time_step),
                save_folder=plot_folder +
                "barrier_ilq_progress/",
                xlimits=env.xlimits,
                ylimits=env.ylimits,
                ego_radius=env.ego_radius,
                animate=True)

        _, Bd, _, _ = plan_env_2.get_jacobian(run_env_obs, control_cbf_candidate)

        # Initializing barrier process time if while loop is never entered
        barrier_process_time = time.time() - barrier_start_time
        barrier_entries = 0

        eps_regularization = 1e-8

        # Obtain P and p for DDPCBF optimization from rollout.
        P = -eps_regularization * \
            np.eye(lq_policy_1.action_dim) + Bd.T @ constraints_data_plan_2['V_xx'] @ Bd
        p = Bd.T @ constraints_data_plan_2['V_x']
        p_norm = np.linalg.norm(p)

        while constraint_violation < barrier_tol and barrier_entries < 5:
            barrier_entries = barrier_entries + 1
            if barrier_type == "linear":
                control_correction = -p * scaled_c / ((p_norm)**2 + 1e-12)
            else:
                control_correction = barrier_filter_quadratic(P, p, scaled_c)
            # Update filtered control
            control_cbf_updated_candidate = control_cbf_candidate + control_correction

            # Testing barrier quality
            imag_obs_2, control_cbf_updated_candidate = plan_env_2.step(
                run_env_obs, control_cbf_updated_candidate)
            _, solver_dict_plan_3, constraints_data_plan_3 = lq_policy_2.get_action(
                np.array(imag_obs_2), initial_controls=boot_controls)

            # Re-initialize and re-optimize after scaling constraint violation
            solver_dict_plan_2 = solver_dict_plan_3
            constraints_data_plan_2 = constraints_data_plan_3
            control_cbf_candidate = control_cbf_updated_candidate

            # Re-optimization setup
            _, Bd, _, _ = plan_env_2.get_jacobian(run_env_obs, control_cbf_candidate)
            P = -eps_regularization * \
                np.eye(lq_policy_1.action_dim) + Bd.T @ constraints_data_plan_2['V_xx'] @ Bd
            p = Bd.T @ constraints_data_plan_2['V_x']
            p_norm = np.linalg.norm(p)
            constraint_violation = solver_dict_plan_2['reachable_margin'] - gamma * (
                solver_dict_plan_1['reachable_margin'])
            boot_controls = solver_dict_plan_2['controls']
            scaled_c = barrier_scaling * constraint_violation
            barrier_process_time = time.time() - barrier_start_time

        print(
            "[{}]: solver 1 returns margin {:.3f} and uses {:.3f}s, solver 2 returns margin {:.3f} and uses {:.3f}s.".format(
                time_step,
                solver_dict_plan_1['margin'],
                solver_dict_plan_1['t_process'],
                solver_dict_plan_2['margin'],
                solver_dict_plan_2['t_process']))

        # If DDP-CBF is failing despite iterations, fallback to DDP-LR filtering.
        if (solver_dict_plan_2['reachable_margin'] <= 0):
            complete_filter_indices.append(time_step)
            filtered_control = control_safe_1
            reinit_controls = np.array(solver_dict_plan_1['controls'])
            complete_filter_steps = complete_filter_steps + 1
            if solver_dict_plan_1['critical_constraint_type'] > 1:
                yaw_filter_indices.append(time_step)
            types_ilq[time_step] = 1
        else:
            filtered_control = control_cbf_candidate
            reinit_controls = np.array(solver_dict_plan_2['controls'])
            if (barrier_entries > 0):
                barrier_filter_indices.append(time_step)
                barrier_filter_steps = barrier_filter_steps + 1
                types_ilq[time_step] = 2

        run_env_obs, filtered_control_clip = run_env.step(run_env_obs, filtered_control)

        states_cbf_ilq[time_step] = np.array(run_env_obs)
        optimal_safe_controls_ilq[time_step] = control_safe_1
        controls_cbf_ilq[time_step] = np.array(filtered_control_clip)
        optimal_values_ilq[time_step] = solver_dict_plan_1['margin']
        process_times[time_step] = barrier_process_time
        barrier_entries_ilq[time_step] = barrier_entries
        barrier_margins[time_step] = constraint_violation
        deviations_cbf_ilq[time_step] = filtered_control_clip - control_perf

        print(
            "[{}]: Total barrier solver and uses {:.3f}s.".format(
                time_step,
                barrier_process_time))

        #if run_env_obs.size > 3 and (
        #        run_env_obs[1] > env.track_length or run_env_obs[2] < env.vmin):
        #    break

    solver_dict = {
        "states": states_cbf_ilq,
        "controls": controls_cbf_ilq,
        "safe_controls": optimal_safe_controls_ilq,
        "values": optimal_values_ilq,
        "types": types_ilq,
        "id": "Barrier",
        "task_trace": None,
        "task_active": False,
        "controls_deviation": deviations_cbf_ilq,
        "status": 1,
        "label": "CBF-DDP",
        "barrier_filter": barrier_filter_steps,
        "complete_filter": complete_filter_steps,
        "process_times": process_times,
        "previous_trace": None,
        "title": "",
        "barrier_margins": barrier_margins,
        "barrier_entries": barrier_entries_ilq,
        "run_steps": time_step,
        "complete_filter_indices": complete_filter_indices,
        "barrier_filter_indices": barrier_filter_indices}

    return solver_dict
