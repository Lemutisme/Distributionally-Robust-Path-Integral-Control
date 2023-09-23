import random
import argparse
import numpy as np
from tqdm import tqdm

from env import Dynamics_Input_Integrator, Dynamics_Unicycle
from utils import OnlineMeanEstimator, stat_info, path_integral, extract_para
from vis import trajectory_plot, simulation_plot, final_plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Set parameters for the experiment.")

    parser.add_argument("--DR_method", choices=["DR NM", "DR GM", "RN"], default="DR NM", help="Set DR method.")
    parser.add_argument("--Experiment", choices=["1", "2"], default="1", help="Set experiment number.")
    parser.add_argument("--Visualization", action='store_true', help="Enable visualization. Default is False.")
    parser.add_argument("--seed_value", type=int, default=0, help="Set seed value, default is False.")
    parser.add_argument("--num_simulation", type=int, default=100, help="Set number of simulations.")
    parser.add_argument("--Online", action='store_true', help="Online Learning")
    parser.add_argument("--observations", type=int, default=1, help="Set number of observations.")
    parser.add_argument("--sigma", type=float, default=0.5, help="Set Sigma")
    parser.add_argument("--mu", type=float, nargs=2, default=[0.0, 0.0], help="Set mu value.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Set max steps.")
    parser.add_argument("--num_trajs", type=int, default=500, help="Set number of trajectories.")
    parser.add_argument("--num_vis", type=int, default=500, help="Set number of vis.")
    parser.add_argument("--T", type=float, default=2.0, help="Set T value.")
    parser.add_argument("--dt", type=float, default=0.05, help="Set dt value.")
    parser.add_argument("--goal_tolerance", type=float, default=0.1, help="Set goal tolerance.")
    parser.add_argument("--dist_weight", type=float, default=0.01, help="Set dist weight.")

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
    goal_tolerance = args.goal_tolerance
    dist_weight = args.dist_weight
    T = args.T
    dt = args.dt
    gamma = 1 / observations
    success_time = []
    success_index = []
    fail_index = []
    x_hists = np.zeros( (num_simulation, max_steps+1, 2) )*np.nan
    time_steps = int(np.floor(T/dt))

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
        f"mu: {mu}\n"
        f"max_steps: {max_steps}\n"
        f"num_trajs: {num_trajs}\n"
        f"num_vis: {num_vis}\n"
        f"T: {T}\n"
        f"dt: {dt}\n"
        f"goal_tolerance: {goal_tolerance}\n"
        f"dist_weight: {dist_weight}\n"
        f"time_step: {time_steps}\n"
    )

    if seed_value:
        np.random.seed(seed_value)
        random.seed(seed_value)

    if Experiment == "1":
        dynamics = Dynamics_Input_Integrator(dt, sigma)

    elif Experiment == "2":
        dynamics = Dynamics_Unicycle(dt, sigma)

    else:
        raise ValueError("Experiment number not recognized")
    
    x_init, x_goal, environment = extract_para(Experiment)

    print(Experiment_info)

    for k in tqdm(range(num_simulation), desc="Simulating", unit="sim"):

        terminate = False
        hit_obstacle = False
        hit_boundary = False

        u_curr = np.zeros((time_steps, 2))
        x_hist = np.zeros((max_steps+1, len(x_init))) * np.nan
        u_hist = np.zeros((max_steps+1, 2)) * np.nan
        x_hist[0] = x_init
        
        plot_every_n = 10
        
        if Online:
            online_estimator = OnlineMeanEstimator(mu, observations)
            mu_hat = online_estimator.get_mean()
        else:
            mu_hat = np.mean(np.random.multivariate_normal(mu, np.identity(2), observations), axis=0)
        
        gamma_t = gamma
        
        for t in range(max_steps) :
            u_curr, x_vis = path_integral(dynamics, environment, mu, x_hist[t], x_goal, dist_weight, time_steps,
                                T, dt, num_trajs, num_vis, gammas=gamma_t, DR_method = DR_method, mu_hat = mu_hat)
            u_hist[t] = u_curr[0]  
            
            x_hist[t+1] = dynamics.compute_next_state(x_hist[t], u_hist[t], np.random.multivariate_normal(np.zeros(2), np.identity(2)), mu)
                                    
            if Online:
                gamma_t = gamma/(t+1)
                mu_hat = online_estimator.update(dynamics.dxi(x_hist, t))

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

            if np.linalg.norm(x_hist[t+1, :2] - x_goal) <= goal_tolerance :
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
                    trajectory_plot(x_hist, x_vis, x_init, x_goal, environment)
                
        x_hists[k, :t, :] = x_hist[:t, :2]

        if Visualization:
            if terminate:
                simulation_plot(x_hist, x_init, x_goal, environment, k, num_simulation)

    dir_path, df = final_plot(x_hists, x_init, x_goal, success_index, success_time, fail_index, environment, Visualization, SAVE_LOG = True)
    stat_info(df, dir_path, num_simulation, Experiment_info)