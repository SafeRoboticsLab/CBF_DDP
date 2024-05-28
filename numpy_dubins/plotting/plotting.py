import matplotlib
from matplotlib import pyplot as plt
import os
import numpy as np

import imageio
from IPython.display import Image

def plot_dubins_trajectory(fig, ax, solver_dicts, obstacle_list=None, obstacle_radius=0,
                        show_label=True, xlimits=[-3.5, 2.0], ylimits=[-0.2, 3.5], marginFunc=None, show_legend=True
                        , ego_radius=0.1, legend_fontsize=8):
    colors = {}
    colors['color0'] = 'r'
    colors['color1'] = 'm'
    colors['color2'] = 'c'
    colors['Barrier'] = 'b'
    colors['HCBF'] = 'm'
    colors['LR'] = 'r'
    colors['Optimal'] = 'k'

    styles = {}
    styles['Barrier'] = 'dashdot'
    styles['LR'] = 'dashed'
    styles['HCBF'] = 'solid'
    styles['Optimal'] = 'solid'
    styles['color0'] = 'solid'
    styles['color1'] = 'solid'
    styles['color2'] = 'solid'

    if marginFunc is not None:
        ix = 0
        iy = 0
            
        nx, ny = (1000, 100)
        xvals= np.linspace(xlimits[0], xlimits[1], nx)
        yvals = np.linspace(ylimits[0], ylimits[1], ny)
        cxy = np.zeros((nx, ny))

        for x in xvals:
            iy = 0
            for y in yvals:
                obs_constr = np.array([[x, y, 0.0]])
                ctrl = np.array([0.0])
                cost, _, _= marginFunc.eval(obs_constr, ctrl)
                cxy[ix, iy] = cost
                iy = iy + 1
            ix = ix + 1

        sc0 = ax.imshow(
              cxy.T, interpolation='none', extent=[xlimits[0],xlimits[1], ylimits[0], ylimits[1]],
              origin="lower", cmap='viridis', vmin=cxy.min(), vmax=cxy.max(), zorder=-1, alpha=0.3
            )
    

    for _, solver_dict in enumerate(solver_dicts):
        dict_id = solver_dict['id']
        if solver_dict['task_active']:
            if show_label:
                ax.plot(solver_dict['states'][:, 0], solver_dict['states'][:, 1], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=1.0, color=colors[dict_id])
            else:
                ax.plot(solver_dict['states'][:, 0], solver_dict['states'][:, 1], label='                 ', linestyle=styles[dict_id], linewidth=1.0, color=colors[dict_id])
        else:
            if show_label:
                ax.plot(solver_dict['states'][:, 0], solver_dict['states'][:, 1], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=1.0, color=colors[dict_id])
            else:
                ax.plot(solver_dict['states'][:, 0], solver_dict['states'][:, 1], label='                 ', linestyle=styles[dict_id], linewidth=1.0, color=colors[dict_id])

        #if solver_dict['label'][0:3]=="CBF":
        #    complete_shield_indices = solver_dict['complete_shield_indices']
        #    barrier_shield_indices = solver_dict['barrier_shield_indices']
            #plt.plot(solver_dict['states'][complete_shield_indices, 0], solver_dict['states'][complete_shield_indices, 1], 'mo')
            #plt.plot(solver_dict['states'][barrier_shield_indices, 0], solver_dict['states'][barrier_shield_indices, 1], 'ko')

    task_active = True
    for _, solver_dict in enumerate(solver_dicts):
        if solver_dict['previous_trace'] is not None:
            dict_id = solver_dict['trace_id']
            ax.plot(solver_dict['previous_trace'][:, 0], solver_dict['previous_trace'][:, 1], label=solver_dict['trace_label'], linestyle='solid', linewidth=2, color=colors[dict_id])
        if solver_dict['task_trace'] is not None:
            dict_id = solver_dict['id']
            task_active = solver_dict['task_active']
            if solver_dict['task_active']:
                ax.plot(solver_dict['task_trace'][:, 0], solver_dict['task_trace'][:, 1], label='Task policy plan ', linestyle='solid', linewidth=1.0, color='c')
            else:
                ax.plot(solver_dict['task_trace'][:, 0], solver_dict['task_trace'][:, 1], label='Task policy plan ', linestyle='solid', linewidth=1.0, color='c')

    if task_active:
        circle_ego = plt.Circle(solver_dict['states'][0, 0:2], ego_radius, color='c', alpha=0.8)
        ax.add_patch(circle_ego)
    else:
        circle_ego = plt.Circle(solver_dict['states'][0, 0:2], ego_radius, color='k', alpha=0.8)
        ax.add_patch(circle_ego)

    for indx, obstacle in enumerate(obstacle_list):
        circle_avoid = plt.Circle(obstacle, obstacle_radius[indx], color='k', alpha=0.6)
        ax.add_patch(circle_avoid)
    
    circle_goal = plt.Circle([1.5, 1.5], 0.15, color='g', alpha=0.6)
    ax.add_patch(circle_goal)

    if show_legend:
        ax.legend(fontsize=legend_fontsize, loc='upper left')

def helper_plot(solver_dicts, obstacle_list=None, obstacle_radius=0, save_prefix="safety", save_folder="./dubins_plots/",  
                        value_plot=False, convergence_plot=False, xlimits=[-3.7, 3.0], ylimits=[-0.4, 3.5], 
                        ego_radius=0.1, animate=False):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    matplotlib.rc('xtick', labelsize=9) 
    matplotlib.rc('ytick', labelsize=9) 

    legend_fontsize = 8
    
    colors = {}
    colors['color0'] = 'r'
    colors['color1'] = 'm'
    colors['color2'] = 'c'
    colors['Barrier'] = 'b'
    colors['HCBF'] = 'k'
    colors['LR'] = 'r'
    colors['Optimal'] = 'k'

    styles = {}
    styles['Barrier'] = 'solid'
    styles['LR'] = 'solid'
    styles['HCBF'] = 'solid'
    styles['Optimal'] = 'solid'
    styles['color0'] = 'solid'
    styles['color1'] = 'solid'
    styles['color2'] = 'solid'
    
    plt.figure()
    if not animate:
        fig1 = plt.figure(figsize=(1.7, 1.45), dpi=100)
        ax = plt.axes()
        label_size = 7
        legend_fontsize = 6
        plot_dubins_trajectory(fig1, ax, solver_dicts, obstacle_list=obstacle_list, obstacle_radius=obstacle_radius, ego_radius=ego_radius,
                         xlimits=xlimits, ylimits=ylimits,
                        legend_fontsize=legend_fontsize)
    else:
        fig1 = plt.figure(figsize=(3.0, 3.0), dpi=100)
        ax = plt.axes()
        label_size = 8
        legend_fontsize = 6
        plot_dubins_trajectory(fig1, ax, solver_dicts, obstacle_list=obstacle_list, obstacle_radius=obstacle_radius, ego_radius=ego_radius,
                         xlimits=xlimits, ylimits=ylimits,
                        legend_fontsize=legend_fontsize)

    ax.set_yticks(ticks=np.array([-0.2, 1.0, 3.2]), labels=np.array([-0.2, 1.0, 3.2]), fontsize=label_size)
    ax.set_xticks(ticks=np.array([-3.5, 0.0, 2.0]), labels=np.array([-3.5, 0.0, 2.0]), fontsize=label_size)
    ax.tick_params(axis='both', labelsize=label_size)
    ax.legend(fontsize=6)
    plt.ylabel("Y position", fontsize=label_size)
    plt.xlabel("X position", fontsize=label_size)
    plt.savefig(save_folder + save_prefix + "_debugging.pdf", bbox_inches='tight')
    plt.savefig(save_folder + save_prefix + "_debugging.png", bbox_inches='tight')

    plt.figure(figsize=(18, 12), dpi=200)
    plt.subplot(2, 3, 1)
    for _, solver_dict in enumerate(solver_dicts):
        dict_id = solver_dict['id']
        nsteps = solver_dict['run_steps']
        plt.plot(solver_dict['states'][0:nsteps, 0], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=2, color=colors[dict_id])
    plt.ylabel("X position", fontsize=legend_fontsize)
    plt.grid()
    plt.legend(fontsize=legend_fontsize)
    plt.subplot(2, 3, 2)
    for _, solver_dict in enumerate(solver_dicts):
        dict_id = solver_dict['id']
        nsteps = solver_dict['run_steps']
        plt.plot(solver_dict['states'][0:nsteps, 1], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=2, color=colors[dict_id])
    plt.ylabel("Y position", fontsize=legend_fontsize)
    plt.grid()
    plt.legend(fontsize=legend_fontsize)
    plt.subplot(2, 3, 3)
    for _, solver_dict in enumerate(solver_dicts):
        dict_id = solver_dict['id']
        nsteps = solver_dict['run_steps']
        plt.plot(solver_dict['states'][0:nsteps, 2], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=2, color=colors[dict_id])
    plt.ylabel("Velocity", fontsize=legend_fontsize)
    plt.grid()
    plt.legend(fontsize=legend_fontsize)

    plt.figure(figsize=(1.7, 1.45), dpi=100)
    ax = plt.gca()
    legend_fontsize=6
    label_size = 7
    for _, solver_dict in enumerate(solver_dicts):
        dict_id = solver_dict['id']
        nsteps = solver_dict['run_steps']
        plt.plot(solver_dict['controls_deviation'][0:nsteps, 0], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=1.0, color=colors[dict_id])
    ax.tick_params(axis='both', labelsize=label_size)
    plt.ylabel("Steer control deviation", fontsize=label_size)
    plt.xlabel("Time step", fontsize=label_size)
    ax.set_yticks(ticks=[0.0 , 1.0, 2.0], labels=[0.0 , 1.0, 2.0], fontsize=label_size)
    ax.set_xticks(ticks=[0, 30, 60, 90, 120, 150], labels=[0, 30, 60, 90, 120, 150], fontsize=label_size)
    ax.set_ylim([0, 2.5])
    #plt.grid()
    plt.legend(fontsize=legend_fontsize, ncol=1)
    plt.savefig(save_folder+save_prefix+"_controls.pdf",bbox_inches='tight')
    plt.savefig(save_folder+save_prefix+"_controls.png",bbox_inches='tight')
    
    if value_plot:
        fig = plt.figure(figsize=( 3.5, 2.3 ))
        ax = plt.gca()
        legend_fontsize=8
        for idx, solver_dict in enumerate(solver_dicts):
            dict_id = solver_dict['id']
            nsteps = solver_dict['run_steps']
            plt.plot(solver_dict['values'][0:nsteps], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=1.5, color=colors[dict_id])
            if solver_dict['label'][0:3]=="CBF":
                ax.fill_between(range(nsteps), min(solver_dict['values'][0:nsteps]), max(solver_dict['values'][0:nsteps]), where=(solver_dict['types'][0:nsteps]==2), color='b', alpha=0.2)
                ax.fill_between(range(nsteps), min(solver_dict['values'][0:nsteps]), max(solver_dict['values'][0:nsteps]), where=(solver_dict['types'][0:nsteps]==1), color='r', alpha=0.2)
        plt.ylabel("Receding horizon value function", fontsize=legend_fontsize)
        plt.xlabel("Time step", fontsize=legend_fontsize)
        #plt.ylim([-0.1, 1.0])
        ax.tick_params(axis='both', labelsize=legend_fontsize)
        plt.grid()
        plt.legend(fontsize=legend_fontsize, ncol=2)
        
        #ymin, ymax = ax.get_ylim()
        #ax.set_yticks(np.round(np.linspace(ymin, ymax, 9), 1))
        
        plt.savefig(save_folder+save_prefix+"_values.pdf",bbox_inches='tight')
        plt.savefig(save_folder+save_prefix+"_values.png",bbox_inches='tight')

        plt.figure(figsize=( 12, 7 ))
        for _, solver_dict in enumerate(solver_dicts):
            dict_id = solver_dict['id']
            nsteps = solver_dict['run_steps']
            if solver_dict['types'] is not None:
                plt.plot(solver_dict['types'][0:nsteps], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=2, color=colors[dict_id])
        plt.ylabel("Shielding type over the horizon", fontsize=legend_fontsize)
        plt.xlabel("Time step", fontsize=legend_fontsize)
        #plt.ylim([-0.45, 0.15])
        plt.grid()
        plt.legend(fontsize=legend_fontsize)
        plt.yticks(np.arange(3), ['None', 'Complete', 'Barrier'], horizontalalignment='right')
        plt.savefig(save_folder+save_prefix+"_shields.pdf")
        plt.savefig(save_folder+save_prefix+"_shields.png")

        plt.figure(figsize=( 12, 7 ))
        for _, solver_dict in enumerate(solver_dicts):
            dict_id = solver_dict['id']
            nsteps = solver_dict['run_steps']
            if dict_id=='Barrier':
                plt.plot(solver_dict['barrier_margins'][0:nsteps], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=2, color=colors[dict_id])
        plt.ylabel("Barrier shielding margin", fontsize=legend_fontsize)
        plt.xlabel("Time step", fontsize=legend_fontsize)
        #plt.ylim([-0.45, 0.15])
        plt.grid()
        plt.legend(fontsize=legend_fontsize)
        plt.savefig(save_folder+save_prefix+"_margins.pdf")
        plt.savefig(save_folder+save_prefix+"_margins.png")

        plt.figure(figsize=( 12, 7 ))
        for _, solver_dict in enumerate(solver_dicts):
            dict_id = solver_dict['id']
            nsteps = solver_dict['run_steps']
            if dict_id=='Barrier':
                plt.plot(solver_dict['barrier_entries'][0:nsteps], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=2, color=colors[dict_id])
        plt.ylabel("Barrier shielding entries", fontsize=legend_fontsize)
        plt.xlabel("Time step", fontsize=legend_fontsize)
        #plt.ylim([-0.45, 0.15])
        plt.grid()
        plt.legend(fontsize=legend_fontsize)
        plt.savefig(save_folder+save_prefix+"_entries.pdf")
        plt.savefig(save_folder+save_prefix+"_entries.png")

        plt.figure(figsize=( 12, 7 ))
        for _, solver_dict in enumerate(solver_dicts):
            dict_id = solver_dict['id']
            nsteps = solver_dict['run_steps']
            plt.plot(solver_dict['process_times'][0:nsteps], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=2, color=colors[dict_id])
        plt.ylabel("Time taken to shield", fontsize=legend_fontsize)
        plt.xlabel("Time step", fontsize=legend_fontsize)
        #plt.ylim([-0.45, 0.15])
        plt.grid()
        plt.legend(fontsize=legend_fontsize)
        plt.savefig(save_folder+save_prefix+"_times.pdf")
        plt.savefig(save_folder+save_prefix+"_times.png")
    plt.close('all')

    if convergence_plot:
        plt.figure(figsize=( 7, 7 ))
        for _, solver_dict in enumerate(solver_dicts):
            dict_id = solver_dict['id']
            nsteps = solver_dict['run_steps']
            plt.plot(solver_dict['convergence'][0:nsteps], label=solver_dict['label'], linestyle=styles[dict_id], linewidth=2, color=colors[dict_id])
        plt.ylabel("Convergence sequence of costs", fontsize=legend_fontsize)
        plt.xlabel("Iteration", fontsize=legend_fontsize)
        plt.savefig(save_folder+save_prefix+"_convergence.pdf")
        plt.savefig(save_folder+save_prefix+"_convergence.png")


def make_animation(fig_prog_folder, num_steps):
    gif_path = os.path.join(fig_prog_folder, 'rollout.gif')
    frame_skip = 1
    with imageio.get_writer(gif_path, mode='I') as writer:
        for i in range(num_steps - 1):
            if frame_skip != 1 and (i+1) % frame_skip == 0:
                continue
            filename = os.path.join(fig_prog_folder, str(i) + "_debugging" + ".png")
            image = imageio.imread(filename)
            writer.append_data(image)
            Image(open(gif_path, 'rb').read(), width=400)