import copy
import numpy as np
import time

from policies import ReachabilityLQPolicy
from utils import unroll_task_policy

from plotting.plotting import helper_plot


def run_lr_ilq(
        env,
        marginFunc,
        horizon,
        max_steps,
        threshold=0.,
        seed=None,
        policy_type="Reachability",
        task_policy=None,
        barrier_tol=-5e-4,
        Rc=1e-3,
        plot_folder="./",
        animate=False):
    """
    LR shielding with iLQ
    """
    plan_env_1 = copy.deepcopy(env)
    plan_env_2 = copy.deepcopy(env)
    run_env = copy.deepcopy(env)

    current_horizon = horizon
    run_env_obs = run_env.reset()

    states_ilq = np.zeros((max_steps, plan_env_1.state_dim))
    controls_ilq = np.zeros((max_steps, plan_env_2.action_dim))
    safe_controls_ilq = np.zeros((max_steps, plan_env_1.action_dim))
    deviations_ilq = np.zeros((max_steps, plan_env_1.action_dim))
    values_ilq = np.zeros((max_steps, ))
    types_ilq = np.zeros((max_steps, ))
    process_times = np.zeros((max_steps, ))

    filter_steps = 0

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

    complete_filter_indices = []
    resolve = True
    for time_step in range(max_steps):
        start_time = time.time()
        print("Observation", run_env_obs)

        if resolve:
            control_safe_1, solver_dict_plan_1, _ = lq_policy_1.get_action(
                np.array(run_env_obs), initial_controls=reinit_controls)
        else:
            control_safe_1 = reinit_controls[:, 0]
            solver_dict_plan_1 = solver_dict_plan_2

        control_perf = task_policy(plan_env_1, run_env_obs)

        safe_controls_ilq[time_step] = control_safe_1

        imag_obs, control_perf = plan_env_2.step(run_env_obs, control_perf)

        if reinit_controls is None:
            _, solver_dict_plan_2, _ = lq_policy_2.get_action(
                np.array(imag_obs), initial_controls=None)
        else:
            _, solver_dict_plan_2, _ = lq_policy_2.get_action(
                np.array(imag_obs), initial_controls=reinit_controls)

        print("[{}]: solver 1 returns margin {:.3f} and uses {:.3f} iterations, solver 2 returns margin {:.3f} and uses {:.3f} iterations.".format(
            time_step, solver_dict_plan_1['margin'], solver_dict_plan_1['iterations'], solver_dict_plan_2['margin'], solver_dict_plan_2['iterations']))

        if (solver_dict_plan_2['reachable_margin'] > threshold):
            control = np.array(control_perf)
            solver_dict_plan_1['task_active'] = True
            reinit_controls = np.array(solver_dict_plan_2['controls'])
            resolve = True
        else:
            control = np.array(control_safe_1)
            solver_dict_plan_1['task_active'] = False
            reinit_controls = np.array(solver_dict_plan_1['controls'])
            filter_steps += 1
            complete_filter_indices.append(time_step)
            resolve = True
            types_ilq[time_step] = 1

        if animate:
            solver_dict_plan_1['title'] = 'Least restrictive filtering'
            solver_dict_plan_1['trace_id'] = 'LR'
            solver_dict_plan_1['trace_label'] = 'LR-DDP' + \
                '($\\epsilon$=' + str(threshold) + ')'
            solver_dict_plan_1['previous_trace'] = states_ilq[0:time_step]
            solver_dict_plan_1['run_steps'] = solver_dict_plan_2['states'].shape[0]
            solver_dict_plan_1['task_trace'] = unroll_task_policy(
                run_env_obs, copy.deepcopy(plan_env_1), task_policy, horizon=current_horizon)
            solver_dict_plan_1['controls_deviation'] = solver_dict_plan_1['controls']
            helper_plot(
                [solver_dict_plan_1],
                marginFunc.obstacle_center,
                marginFunc.obstacle_radius,
                save_prefix=str(time_step),
                save_folder=plot_folder +
                "lr_ilq_progress/",
                xlimits=env.xlimits,
                ylimits=env.ylimits,
                ego_radius=env.ego_radius,
                animate=True)

        run_env_obs, control_clip = run_env.step(run_env_obs, control)

        process_times[time_step] = time.time() - start_time
        states_ilq[time_step] = np.array(run_env_obs)
        safe_controls_ilq[time_step] = control_safe_1
        controls_ilq[time_step] = np.array(control_clip)
        values_ilq[time_step] = solver_dict_plan_1['margin']

        deviations_ilq[time_step] = control_clip - control_perf

    solver_dict = {
        "states": states_ilq,
        "run_steps": time_step + 1,
        "controls": controls_ilq,
        "controls_deviation": deviations_ilq,
        "safe_controls": safe_controls_ilq,
        "values": values_ilq,
        "id": "LR",
        "task_trace": None,
        "task_active": False,
        "types": types_ilq,
        "filter": filter_steps,
        "status": 1,
        "process_times": process_times,
        "label": "LR-DDP",
        "previous_trace": None,
        "title": "",
        "complete_filter_indices": complete_filter_indices,
        "run_steps": time_step}

    return solver_dict
