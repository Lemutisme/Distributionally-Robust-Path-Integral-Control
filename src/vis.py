import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from utils import extract_para, extract_log_files

def trajectory_plot(x_hist, x_vis, x_init, x_goal, environment):
    fig, ax = plt.subplots()
    ax.plot([x_init[0]], [x_init[1]], '8', markersize = 10, markerfacecolor = 'k', label = 'Initial State',markeredgecolor = 'none' )
    ax.plot([x_goal[0]], [x_goal[1]], '*', markersize = 10, markerfacecolor = 'k', label = 'Target State',markeredgecolor = 'none' )
                    
    environment.plot_map(ax)
    ax.plot(x_hist[:,0], x_hist[:,1], 'r', label='Past state')
    ax.plot(x_vis[:,:,0].T, x_vis[:,:,1].T, 'k', alpha=0.1, zorder=3)
                    
    ax.set_xlim(environment.obstacles[0].boundary_x)
    ax.set_ylim(environment.obstacles[0].boundary_y)
                    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def simulation_plot(x_hist, x_init, x_goal, environment, k, num_simulation):
    fig, ax = plt.subplots()
    ax.plot([x_init[0]], [x_init[1]], '8', markersize=20, markerfacecolor='lime', label='Initial State', markeredgecolor='k', zorder=6)
    ax.plot([x_goal[0]], [x_goal[1]], '*', markersize=20, markerfacecolor='lime', label='Target State', markeredgecolor='k', zorder=6)
    environment.plot_map(ax)

    ax.plot(x_hist[:,0], x_hist[:,1], 'r', label='Past state')
    ax.set_xlim(environment.obstacles[0].boundary_x)
    ax.set_ylim(environment.obstacles[0].boundary_y)
    ax.set_xlabel(r'$p_{x}$')
    ax.set_ylabel(r'$p_{y}$')
    plt.gcf().set_dpi(600)
    ax.set_title(f'Trajectories {k+1} in {num_simulation} simulations.')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def final_plot(x_hists, x_init, x_goal, success_index, success_time, fail_index, environment, Visualization, SAVE_LOG):
    sns.set_style("white")
    soft_green = "#9bf80c"
    dark_blue = "#02075d"

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot([x_init[0]], [x_init[1]], '8', markersize=20, markerfacecolor=sns.color_palette()[0], markeredgecolor='k', zorder=6)
    ax.plot([x_goal[0]], [x_goal[1]], '*', markersize=25, markerfacecolor=sns.color_palette()[1], markeredgecolor='k', zorder=6)

    environment.plot_map(ax)

    ax.set_xlim(environment.obstacles[0].boundary_x)
    ax.set_ylim(environment.obstacles[0].boundary_y)
    ax.set_xlabel(r'$p_{x}$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$p_{y}$', fontsize=14, fontweight='bold')

    for i in success_index:
        ax.plot(x_hists[i,:,0], x_hists[i,:,1], color=soft_green, linewidth=0.8)

    for i in fail_index:
        ax.plot(x_hists[i,:,0], x_hists[i,:,1], color=dark_blue, linewidth=0.8)

    ax.plot([], [], '8', markersize=10, markerfacecolor=sns.color_palette()[0], label='Initial State', markeredgecolor='k') # Reduced size for legend
    ax.plot([], [], '*', markersize=10, markerfacecolor=sns.color_palette()[1], label='Target State', markeredgecolor='k') # Reduced size for legend
    ax.plot([], [], color=soft_green, linewidth=0.8, label='Successful Trajectories')
    ax.plot([], [], color=dark_blue, linewidth=0.8, label='Failed Trajectories')

    ax.set_aspect('equal')
    ax.set_xlabel(r'$p_{x}$', fontsize=10)
    ax.set_ylabel(r'$p_{y}$', fontsize=10)

    # ax.legend(loc="lower left", prop={'size': 7})

    if Visualization:
        plt.tight_layout()
        plt.show()

    if SAVE_LOG:    
        dir_path = f"./log/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filename = os.path.join(dir_path, f"Result.png")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi = 1000)

        df = pd.DataFrame({'success_index': success_index, 'success_time': success_time})
        df.to_csv(os.path.join(dir_path, 'success_data.csv'), index=False)
        with open(os.path.join(dir_path, 'x_hists.json'), 'w') as file:
            json.dump(x_hists.tolist(), file)

        return dir_path, df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Set parameters for the experiment.")

    parser.add_argument("--Experiment", choices=["1", "2"], default="1", help="Model number.")
    parser.add_argument("--Name", choices=["log/DRExp1", "log/DRExp2", "log/RNExp1", "log/RNExp2"], default="log/DRExp1", help="File name.")
    parser.add_argument("--SAVE_LOG", action='store_true', help="Save figure or not")
    parser.add_argument("--Visualization", action='store_true', help="Visualize figure or not")

    Exp = parser.parse_args().Experiment
    Name = parser.parse_args().Name
    SAVE_LOG = parser.parse_args().SAVE_LOG
    Visualization = parser.parse_args().Visualization

    x_hists, success_index, success_time, fail_index = extract_log_files(Name)
    
    x_init, x_goal, environment = extract_para(Exp)

    final_plot(x_hists, x_init, x_goal, success_index, success_time, fail_index, environment, Visualization, SAVE_LOG)