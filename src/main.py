import os
import random
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm

from env import *
from utils import OnlineMeanEstimator, stat_info, dist_to_goal_function, path_integral
from vis import final_plot


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Set parameters for the experiment.")

    # Add arguments
    parser.add_argument("--DR_method", choices=["DR NM", "DR GM", "RN"], default="DR NM", help="Set DR method.")
    parser.add_argument("--Experiment", choices=["1", "2"], default="1", help="Set experiment number.")
    parser.add_argument("--Visualization", action='store_true', help="Enable visualization. Default is False.")
    parser.add_argument("--seed_value", type=int, default=0, help="Set seed value, default is False.")
    parser.add_argument("--num_simulation", type=int, default=100, help="Set number of simulations.")
    parser.add_argument("--Online", action='store_true', help="Online Learning")
    parser.add_argument("--observations", type=int, default=1, help="Set number of observations.")
    parser.add_argument("--gamma", type=float, default=300, help="Set Gamma.")
    parser.add_argument("--sigma", type=float, default=0.5, help="Set Sigma")
    parser.add_argument("--SAVE_LOG", action='store_true', help="Save log")
    parser.add_argument("--mu", type=float, nargs=2, default=[-0.0, 0.0], help="Set mu value.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Set max steps.")
    parser.add_argument("--num_trajs", type=int, default=500, help="Set number of trajectories.")
    parser.add_argument("--num_vis", type=int, default=500, help="Set number of vis.")
    parser.add_argument("--T", type=float, default=2.0, help="Set T value.")
    parser.add_argument("--dt", type=float, default=0.05, help="Set dt value.")

    args = parser.parse_args()

    DR_method = args.DR_method
    Experiment = args.Experiment
    Visualization = args.Visualization
    seed_value = args.seed_value
    num_simulation = args.num_simulation
    Online = args.Online
    observations = args.observations
    sigma = args.sigma
    mu = np.array(args.mu)
    max_steps = args.max_steps
    num_trajs = args.num_trajs
    num_vis = args.num_vis
    T = args.T
    dt = args.dt
    gamma = args.gamma
    SAVE_LOG = args.SAVE_LOG

    Experiment_info = (
        f"DR_method: {DR_method}\n"
        f"Experiment: {Experiment}\n"
        f"Visualization: {Visualization}\n"
        f"seed_value: {seed_value}\n"
        f"num_simulation: {num_simulation}\n"
        f"Online: {Online}\n"
        f"observations: {observations}\n"
        f"gamma: {gamma}\n"
        f"sigma: {sigma}\n"
        f"SAVE_LOG: {SAVE_LOG}\n"
        f"mu: {mu}\n"
        f"max_steps: {max_steps}\n"
        f"num_trajs: {num_trajs}\n"
        f"num_vis: {num_vis}\n"
        f"T: {T}\n"
        f"dt: {dt}\n"
    )

    if seed_value:
        np.random.seed(seed_value)
        random.seed(seed_value)

    if not Online:
        mu_hat = np.mean(np.random.multivariate_normal(mu, np.identity(2), observations), axis=0)

    if Experiment == "1":
        # Dynamics
        dynamics = Dynamics_1(dt, sigma)
        
        x_goal = np.array([0,0])
        x_init = np.array([-3.5, 2.5, 0, 0])
        
        # Map parameters
        boundary_x = [-4.0, 1.0]
        boundary_y = [-1.0, 4.0]
        obs_cost = 1e2
        goal_tolerance = 0.1
        dist_weight = 0.01
        obstacle_positions = np.array([[-2.75, 0.25]])
        obstacle_radius = np.array([2.0])

        obstacle = Obstacle(obstacle_positions = obstacle_positions, obstacle_radius = obstacle_radius, boundary_x = boundary_x, boundary_y = boundary_y, obs_cost = obs_cost)
        environment = Map([obstacle])

    elif Experiment == "2":

        # Dynamics
        dynamics = Dynamics_2(dt, sigma)
        
        x_goal = np.array([0,0])
        x_init = np.array([0, 5, 5.7])

        # Map parameters
        boundary_x = [-2, 4]
        boundary_y = [-2, 6]
        obs_cost = 1e2
        goal_tolerance = 0.05
        dist_weight = 0.01
        obstacle_positions = np.array([[-1, 3]])
        obstacle_radius = np.array([[2, 1]])

        obstacle = RectangularObstacle(obstacle_positions = obstacle_positions, obstacle_dimensions = obstacle_radius,
                                        boundary_x = boundary_x, boundary_y = boundary_y, obs_cost = obs_cost)
        environment = Map([obstacle])

    else:
        raise ValueError("Experiment number not recognized")

    print(Experiment_info)
    success_time = []
    success_index = []
    fail_index = []
    x_hists = np.zeros( (num_simulation, max_steps+1, 2) )*np.nan

    for k in tqdm(range(num_simulation), desc="Simulating", unit="sim"):

        terminate = False
        hit_obstacle = False
        hit_boundary = False

        u_curr = np.zeros((int(T//dt), 2))
        x_hist = np.zeros( (max_steps+1, len(x_init)) )*np.nan
        u_hist = np.zeros( (max_steps+1, 2) )*np.nan
        x_hist[0] = x_init
        
        plot_every_n = 10
        
        if Online:
            online_estimator = OnlineMeanEstimator(mu)
            mu_hat = online_estimator.get_mean()
        
        gamma_t = gamma
        
        for t in range(max_steps) :
            u_curr, x_vis = path_integral(dynamics, environment, mu, x_hist[t], x_goal, dist_weight, 
                                obs_cost ,obstacle_positions, obstacle_radius, 
                                T, dt, num_trajs, num_vis, gammas=gamma_t, sigma=sigma, DR_method = DR_method, mu_hat = mu_hat)
            u_hist[t] = u_curr[0]  
            
            x_hist[t+1] = dynamics.compute_next_state(x_hist[t], u_curr[0], np.random.multivariate_normal(np.zeros(2), np.identity(2)), mu, dt)
                                    
            if Online:
                gamma_t = gamma/(t+1)
                if Experiment == "1":
                    dsi = np.linalg.pinv(dynamics.S) @ (x_hist[t+1] - x_hist[t] - dynamics.F @ x_hist[t] * dt)
                    mu_hat = online_estimator.update(dsi)

                elif Experiment == "2":
                    Sigma_Matrix =  np.array([[np.cos(x_hist[t][2]), 0],
                                            [np.sin(x_hist[t][2]), 0],
                                            [0, sigma]])
                    dsi = np.linalg.pinv(Sigma_Matrix) @ (x_hist[t+1] - x_hist[t] - np.identity(3) @ x_hist[t] * dt)
                    mu_hat = online_estimator.update(dsi)

                else:
                    raise ValueError("Experiment number not recognized")

            if environment.check_hit_any_obstacle(x_hist[t+1]):
                terminate = True
                fail_index.append(k)
                print("Hit obstacle")
                break

            if environment.check_hit_any_boundary(x_hist[t+1]):
                terminate = True
                fail_index.append(k)
                print("Hit boundary")
                break    

            if dist_to_goal_function(x_hist[t+1, :2], x_goal) <= goal_tolerance :
                print("Goal reached at t={:.2f}s".format(t*dt))
                success_time.append(t*dt)
                terminate = True
                success_index.append(k)
                break

            if t == max_steps - 1:
                terminate = True
                fail_index.append(k)
                print("MAX STEPS REACHED")
                break  

            if Visualization:
                if t % plot_every_n == 0 :

                    fig, ax = plt.subplots()
                    ax.plot([x_init[0]], [x_init[1]], '8', markersize = 10, markerfacecolor = 'k', label = 'Initial State',markeredgecolor = 'none' )
                    ax.plot([x_goal[0]], [x_goal[1]], '*', markersize = 10, markerfacecolor = 'k', label = 'Target State',markeredgecolor = 'none' )
                    
                    environment.plot_map(ax)


                    ax.plot(x_hist[:,0], x_hist[:,1], 'r', label='Past state')
                    
                    ax.plot(x_vis[:,:,0].T, x_vis[:,:,1].T, 'k', alpha=0.1, zorder=3)
                    
                    ax.set_xlim(boundary_x)
                    ax.set_ylim(boundary_y)
                    
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
                    ax.set_aspect('equal')
                    plt.tight_layout()
                    plt.show()
                
        x_hists[k, :t, :] = x_hist[:t, :2]

        if Visualization:
            if terminate:
                fig, ax = plt.subplots()
                ax.plot([x_init[0]], [x_init[1]], '8', markersize=20, markerfacecolor='lime', label='Initial State', markeredgecolor='k', zorder=6)
                ax.plot([x_goal[0]], [x_goal[1]], '*', markersize=20, markerfacecolor='lime', label='Target State', markeredgecolor='k', zorder=6)
                environment.plot_map(ax)

                ax.plot(x_hist[:,0], x_hist[:,1], 'r', label='Past state')
                ax.set_xlim(boundary_x)
                ax.set_ylim(boundary_y)
                ax.set_xlabel(r'$p_{x}$')
                ax.set_ylabel(r'$p_{y}$')
                plt.gcf().set_dpi(600)
                ax.set_title(f'Trajectories {k+1} in {num_simulation} simulations.')
                ax.set_aspect('equal')
                plt.tight_layout()
                plt.show()

    dir_path = final_plot(x_hists, x_init, x_goal, boundary_x, boundary_y, success_index, fail_index, environment, Visualization, SAVE_LOG)

    if SAVE_LOG:
        file_path = os.path.join(dir_path, 'x_hists.json')

        with open(file_path, 'w') as file:
            json.dump(x_hists.tolist(), file)

        df = pd.DataFrame({'success_index': success_index, 'success_time': success_time})

        stat_info(df, dir_path, num_simulation, Experiment_info)

        df.to_csv(os.path.join(dir_path, 'success_data.csv'), index=False)

        
