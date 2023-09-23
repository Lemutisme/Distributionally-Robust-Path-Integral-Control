import os
import json
import numpy as np
import pandas as pd
from env import *
from scipy.optimize import minimize, basinhopping, differential_evolution

class OnlineMeanEstimator:
    def __init__(self, mu, N):
        self.n = N
        self.mean = np.mean(np.random.multivariate_normal(mu, np.identity(len(mu)), self.n), axis=0)

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        return self.mean

    def get_mean(self):
        return self.mean

def stage_cost(dist2, dist_weight = 1) :
    return dist_weight * dist2

def term_cost(dist2, goal_reached) :
    return (1 - float(goal_reached)) * dist2

def sample_noise(mu, time_steps, num_trajs=500) :
    return np.random.multivariate_normal(mu, np.identity(len(mu)) , [num_trajs, time_steps])

def rollout(dynamics, environment, x_init, x_goal, time_steps, noise_samples, dist_weight, mu_hat, num_trajs=500, num_vis=500, goal_tolerance=0.1) :
    costs = np.zeros(num_trajs)
    x_vis = np.zeros( (num_vis, time_steps, 2) )*np.nan
    
    for k in range(num_trajs) :

        x_curr = x_init.copy()
        
        if k < num_vis :
            x_vis[k, 0, :] = x_curr[:2]    
        
        for t in range(time_steps) :
            x_curr = dynamics.compute_next_state(x_curr, np.zeros(2), noise_samples[k, t, :], mu_hat)
            
            if k < num_vis :
                x_vis[k, t, :] = x_curr[:2]    
                
            dist_to_goal = np.linalg.norm(x_curr[:2] - x_goal)
            costs[k] += stage_cost(dist_to_goal, dist_weight)
            
            if dist_to_goal <= goal_tolerance :
                break
            
            num_obs = len(environment.obstacles[0].obstacle_positions)
            
            if num_obs != 0 :
                # Obstacle cost
                costs[k] += environment.compute_total_obstacle_cost(x_curr)
                if environment.check_hit_any_obstacle(x_curr):
                    break
                # Boundary cost
                costs[k] += environment.compute_total_boundary_penalty(x_curr)
                if environment.check_hit_any_boundary(x_curr):
                    break
            
    return costs, x_vis

def opt_cost_func(lambda_, gamma, costs, num_trajs):
    epsilon = 1e-10
    lambda_prime = 1 / (1 - lambda_ + epsilon)
    return gamma / (lambda_ + epsilon) - lambda_prime * np.log(1 / num_trajs * np.sum(np.exp(- costs / lambda_prime)))

def update_useq_risk_neutral(costs, noise_samples, T, dt, lambda_neut=1, n=2) :
    costs = np.exp( - (costs) / lambda_neut )     
    sum_costs = np.sum(costs)    
    
    time_steps = int(np.floor(T/dt))
    u_curr = np.zeros((time_steps, n))
    for t in range(time_steps) :
        for k in range(len(costs)) :
            u_curr[t,:] += (costs[k] / sum_costs ) * noise_samples[k,t,:] / np.sqrt(dt) 

    return  u_curr

def update_useq_NM(costs, noise_samples, gamma, T, dt):
    num_trajs = len(noise_samples)
    lambda_r = 1  

    best_result = None
    best_value = float('inf')  

    methods = ['Powell', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']
    bound_for_lambda = [(0, None)]  

    for method in methods:
        try:
            result = minimize(opt_cost_func, x0=lambda_r, args=(gamma, costs, num_trajs), method=method, bounds=bound_for_lambda)
                
            if result.fun < best_value:
                best_result = result
                best_value = result.fun

        except ValueError as e:
            print(f"Error using method {method} for gamma = {gamma}: {e}")
        
    if best_result: 
        lambda_r = best_result.x[0]
    else:
        print("Optimization failed for gamma =", gamma)

    return update_useq_risk_neutral(costs, noise_samples, T, dt, lambda_neut=lambda_r, n=2)

def update_useq_GM(costs, noise_samples, gamma, T, dt):
    num_trajs = len(noise_samples)
    lambda_r = 1 

    best_result = None
    best_value = float('inf')  

    bound_for_lambda = [(0, 100)]  
    try:
        result = basinhopping(opt_cost_func, x0=lambda_r, minimizer_kwargs={"method": "L-BFGS-B", "args": (gamma, costs, num_trajs), "bounds": bound_for_lambda})
        
        if result.fun < best_value:
            best_result = result
            best_value = result.fun
    except ValueError as e:
        print(f"Error in basinhopping for gamma = {gamma}: {e}")

    try:
        result = differential_evolution(opt_cost_func, bounds=bound_for_lambda, args=(gamma, costs, num_trajs), x0=lambda_r)
        
        if result.fun < best_value:
            best_result = result
            best_value = result.fun
    except ValueError as e:
        print(f"Error in differential_evolution for gamma = {gamma}: {e}")

    if best_result:  
        lambda_r = best_result.x[0]
    else:
        print("Optimization failed for gamma =", gamma)

    return update_useq_risk_neutral(costs, noise_samples, T, dt, lambda_neut=lambda_r, n=2)

def path_integral(dynamics, environment, mu, x_init, x_goal, dist_weight, time_steps, T, dt, num_trajs, num_vis, gammas, DR_method, mu_hat):
    
    noise_samples = sample_noise(mu, time_steps, num_trajs)
    costs, x_vis = rollout(dynamics, environment, x_init, x_goal, time_steps, noise_samples, dist_weight, mu_hat, num_trajs, num_vis, goal_tolerance=0.1)   
    if DR_method == "DR NM" :
        u_curr = update_useq_NM(costs, noise_samples, gammas, T, dt)
    elif DR_method == "DR GM" :
        u_curr = update_useq_GM(costs, noise_samples, gammas, T, dt)
    elif DR_method == "RN" :
        u_curr = update_useq_risk_neutral(costs, noise_samples, T, dt, lambda_neut=1, n=2)
    else :
        raise ValueError("DR_method not recognized")
    return u_curr, x_vis

def stat_info(df, dir_path, num_simulation, Experiment_info):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    mean_success_time = df['success_time'].mean()
    variance_success_time = df['success_time'].var()
    standard_deviation_success_time = df['success_time'].std()
    percentiles_success_time = df['success_time'].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    success_rate = len(df['success_time']) / num_simulation
    
    info = (
        "\nStatistics info:\n"
        f"Success rate: {success_rate * 100}%\n"
        f"Mean of success_time: {mean_success_time}\n"
        f"Variance of success_time: {variance_success_time}\n"
        f"Standard Deviation of success_time: {standard_deviation_success_time}\n"
        "Percentiles of success_time:\n"
        f"{percentiles_success_time}\n"
    )

    print(info)
    file_path = os.path.join(dir_path, 'statistics_info.txt')
    with open(file_path, 'w') as file:
        file.write(Experiment_info + info)

def extract_para(Experiment):

    if Experiment == "1":

        x_goal = np.array([0,0])
        x_init = np.array([-3.5, 2.5, 0, 0])
        
        # Map parameters
        boundary_x = [-4.0, 1.0]
        boundary_y = [-1.0, 4.0]
        obs_cost = 1e2
        
        obstacle_positions = np.array([[-2.75, 0.25]])
        obstacle_radius = np.array([2.0])

        obstacle = Obstacle(obstacle_positions = obstacle_positions, obstacle_radius = obstacle_radius, boundary_x = boundary_x, boundary_y = boundary_y, obs_cost = obs_cost)

        environment = Map([obstacle])

    elif Experiment == "2":

        x_goal = np.array([0,0])
        x_init = np.array([0, 5, 5.7])
        
        # Map parameters
        boundary_x = [-2, 4]
        boundary_y = [-2, 6]
        obs_cost = 1e2

        obstacle_positions = np.array([[-1, 3]])
        obstacle_radius = np.array([[2, 1]])

        obstacle = RectangularObstacle(obstacle_positions = obstacle_positions, obstacle_dimensions = obstacle_radius,
                                        boundary_x = boundary_x, boundary_y = boundary_y, obs_cost = obs_cost)
        environment = Map([obstacle])

    else:
        raise ValueError("Experiment number not recognized")
    
    return x_init, x_goal, environment

def extract_log_files(base_path):

    success_data = pd.read_csv(os.path.join(base_path, 'success_data.csv'))
    
    json_path = os.path.join(base_path, 'x_hists.json')
    with open(json_path, 'r') as f:
        x_hists_data = json.load(f)
    x_hists_ndarray = np.array(x_hists_data)

    num_simulation = len(x_hists_ndarray)
    success_index = success_data['success_index'].to_numpy()
    success_time = success_data['success_time'].to_numpy()
    fail_index = np.setdiff1d(np.arange(num_simulation), success_index)
    
    return x_hists_ndarray, success_index, success_time, fail_index