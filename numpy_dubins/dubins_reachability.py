import numpy as np
import math
from matplotlib import pyplot as plt
import pickle
import os

from costFunc.reachabilityFunc import ReachabilityMarginFunction3D
from dynamics import Dubins
from cbfddp import run_barrier_ilq
from lrddp import run_lr_ilq
from plotting.plotting import helper_plot, make_animation, plot_dubins_trajectory


def dubins_task_policy(env, run_env_obs):
    goal = env.goal
    direction = math.atan2(goal[1] - run_env_obs[1], goal[0] - run_env_obs[0])
    control_task = np.array([5.0 * (direction - run_env_obs[2])])

    return control_task


def run_heuristic_cbf(run_env, task_policy, marginFunc, max_steps):
    run_env_obs = env.reset()

    states_hcbf = np.zeros((max_steps, env.state_dim))
    controls_hcbf = np.zeros((max_steps, run_env.action_dim))
    deviations_hcbf = np.zeros((max_steps, run_env.action_dim))
    safe_controls_hcbf = np.zeros((max_steps, run_env.action_dim))
    values_hcbf = np.zeros((max_steps, ))
    process_times = np.zeros((max_steps, ))
    obs_position = marginFunc.obstacle_center[0]
    obs_radius = marginFunc.obstacle_radius[0]
    ego_radius = marginFunc.ego_radius
    R = obs_radius + ego_radius

    for t in range(max_steps):
        print("Time t with observation", t, run_env_obs)
        states_hcbf[t, :] = run_env_obs

        r = np.linalg.norm(run_env_obs[0:2] - obs_position)
        direction_obstacle = math.atan2(
            obs_position[1] - run_env_obs[1],
            obs_position[0] - run_env_obs[0])
        theta = run_env_obs[2] - direction_obstacle

        control_task = task_policy(env, run_env_obs)
        control_task = np.clip(
            control_task, -1 * env.theta_limit, env.theta_limit)

        rhs = -1.0 * (r**2 * np.sin(theta) - R**2)
        lhs = -r * np.sin(theta) * np.cos(theta) * env.velocity
        cbf_c = lhs - rhs
        cbf_p = r * r * np.cos(theta)

        if cbf_p * control_task[0] + cbf_c > 0:
            run_env_obs, control_perf = env.step(
                run_env_obs, np.array(control_task))
        else:
            control_perf = np.array(control_task)
            control_perf[0] = -cbf_c / cbf_p
            run_env_obs, control_perf = env.step(
                run_env_obs, np.array(control_perf))
        deviations_hcbf[t] = control_perf - control_task
        controls_hcbf[t, :] = control_perf
        values_hcbf[t] = 0.4 * (r**2 * np.sin(theta) - R**2)

    solver_dict = {
        "states": states_hcbf,
        "run_steps": t + 1,
        "controls": controls_hcbf,
        "safe_controls": safe_controls_hcbf,
        "values": values_hcbf,
        "controls_deviation": deviations_hcbf,
        "id": "HCBF",
        "label": "Manual CBF",
        "task_trace": None,
        "task_active": False,
        "previous_trace": None,
        "title": "",
        "types": None,
        'process_times': process_times}

    return solver_dict


obstacle_list = np.array([[0.0, 1.0]])
obstacle_radius = np.array([1.0])

horizon = 40
max_steps = 150

theta_limit = 1.0
gamma = 0.95
Rc = 5e-3
barrier_tol = -1e-5
barrier_scaling = 1.4
ego_radius = 0.15

plot_folder = "./dubins_plots_animate/"
env = Dubins(theta_limit)
env.xlimits = [-3.5, 2.0]
env.ylimits = [-1.0, 3.5]
env.road_boundary = None
env.ego_radius = ego_radius
env.track_length = 4.5

marginFunc = ReachabilityMarginFunction3D(
    ego_radius=ego_radius,
    obstacles=obstacle_list,
    obstacle_radius=obstacle_radius)

run = True

if run:
    solver_dict_hcbf = run_heuristic_cbf(
        env, dubins_task_policy, marginFunc, max_steps)

    # helper_plot([ solver_dict_hcbf ], save_prefix="barrier_shield_receding_",
    # save_folder = plot_folder + "param_" + str(gamma_l) + "," + str(Rc)+"/",
    # obstacle_list=obstacle_list, obstacle_radius=obstacle_radius,
    # value_plot=True)

    pickle.dump(
        solver_dict_hcbf,
        open(
            "./dubins_plots_animate/hcbf.pkl",
            "wb"))

    solver_dict_barrier_1 = run_barrier_ilq(
        env,
        marginFunc,
        horizon,
        max_steps,
        gamma=gamma,
        policy_type="Reachability",
        barrier_type="quadratic",
        task_policy=dubins_task_policy,
        barrier_tol=barrier_tol,
        barrier_scaling=barrier_scaling,
        Rc=Rc,
        plot_folder=plot_folder,
        animate=True)

    fig_prog_folder = plot_folder + "barrier_ilq_progress/"
    make_animation(fig_prog_folder, solver_dict_barrier_1['run_steps'])

    # helper_plot([ solver_dict_barrier_1, solver_dict_hcbf ], save_prefix="barrier_shield_receding_",
    # save_folder = plot_folder + "param_" + str(gamma_l) + "," + str(Rc)+"/",
    # obstacle_list=obstacle_list, obstacle_radius=obstacle_radius,
    # value_plot=True)

    pickle.dump(solver_dict_barrier_1, open(
        "./dubins_plots_animate/cbfddp.pkl", "wb"))

    solver_dict_lr_1 = run_lr_ilq(
        env,
        marginFunc,
        horizon,
        max_steps,
        threshold=0.,
        policy_type="Reachability",
        task_policy=dubins_task_policy,
        Rc=Rc,
        plot_folder=plot_folder,
        animate=True)

    pickle.dump(
        solver_dict_lr_1,
        open(
            "./dubins_plots_animate/lrddp.pkl",
            "wb"))

    fig_prog_folder = plot_folder + "lr_ilq_progress/"
    make_animation(fig_prog_folder, solver_dict_lr_1['run_steps'])

    # helper_plot([ solver_dict_lr_1, solver_dict_barrier_1, solver_dict_hcbf  ], save_prefix="barrier_shield_receding_",
    # save_folder = plot_folder + "param_" + str(gamma_l) + "," + str(Rc)+"/",
    # obstacle_list=obstacle_list, obstacle_radius=obstacle_radius,
    # value_plot=True)
else:
    solver_dict_hcbf = pickle.load(
        open("./dubins_plots_animate/hcbf.pkl", "rb"))
    solver_dict_barrier_1 = pickle.load(
        open("./dubins_plots_animate/cbfddp.pkl", "rb"))
    solver_dict_lr_1 = pickle.load(
        open("./dubins_plots_animate/lrddp.pkl", "rb"))

plotting = True
show_label = False
fig, ax = plt.subplots(
    1, 2, figsize=(
        6.4, 3.0), gridspec_kw={
            'height_ratios': [1]})
if plotting:
    plot_dubins_trajectory(fig,
                           ax[0],
                           [solver_dict_hcbf,
                            solver_dict_barrier_1,
                            solver_dict_lr_1],
                           obstacle_list=obstacle_list,
                           obstacle_radius=obstacle_radius,
                           xlimits=[-3.5,
                                    2.0],
                           ylimits=[-0.2,
                                    3.5],
                           ego_radius=0.15,
                           show_legend=True,
                           show_label=show_label,
                           legend_fontsize=10)

# plt.tight_layout()
ax[0].set_yticks(ticks=np.array([-1.5, 3.0]),
                 labels=np.array([-1.5, 3.0]), fontsize=10)
ax[0].set_xticks(ticks=np.array([-3.5, 2.0]),
                 labels=np.array([-3.5, 2.0]), fontsize=10)
ax[0].tick_params(axis='both', labelsize=10)
ax[0].set_ylim([-1.5, 3.0])
ax[0].set_xlim([-3.5, 2.0])
ax[0].legend(
    framealpha=0,
    fontsize=8,
    bbox_to_anchor=(
        1.3,
        1.2),
    ncol=3,
    fancybox=False,
    shadow=False)
ax[0].tick_params("both", labelsize=10)

if show_label:
    ax[0].set_xlabel("X position", fontsize=10)
    ax[0].set_ylabel("Y position", fontsize=10)
else:
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
ax[0].yaxis.set_label_coords(-0.05, 0.5)
ax[0].xaxis.set_label_coords(.5, -.05)
# ax[0].set_aspect('equal')

ax[1].plot(solver_dict_lr_1["controls_deviation"][:, 0],
           linewidth=1, color='r', linestyle='dashed')
ax[1].plot(solver_dict_hcbf["controls_deviation"][:, 0],
           linewidth=1, color='m', linestyle='solid')
ax[1].plot(solver_dict_barrier_1["controls_deviation"][:, 0],
           linewidth=1, color='b', linestyle='dashdot')

ax[1].xaxis.set_label_coords(.5, -.05)
ax[1].yaxis.set_label_coords(-0.05, 0.5)
ax[1].tick_params("both", labelsize=10)
ax[1].set_yticks(ticks=[0, 2], labels=[0, 2])
ax[1].set_xticks(ticks=[0, 150], labels=[0, 150])
ax[1].set_xlim([0, 150])
ax[1].set_ylim([0, 2.2])

if show_label:
    ax[1].set_ylabel("Control deviation", fontsize=10)
    ax[1].set_xlabel("Time step", fontsize=10)
else:
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])

fig.tight_layout()
fig.savefig(
    "./dubins_plots_animate/paper_combined" +
    str(show_label) +
    ".pdf",
    dpi=200,
    bbox_inches='tight',
    transparent=not show_label)
fig.savefig(
    "./dubins_plots_animate/paper_combined" +
    str(show_label) +
    ".png",
    dpi=200,
    bbox_inches='tight',
    transparent=not show_label)
