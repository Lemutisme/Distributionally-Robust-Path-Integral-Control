import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class Dynamics_1:
    def __init__(self, dt, sigma):
        self.dt = dt
        self.sigma = sigma
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        self.G = np.array([[0, 0],
                           [0, 0],
                           [sigma, 0],
                           [0, sigma]])
        
        self.S = np.array([[0, 0],
                           [0, 0],
                           [sigma, 0],
                           [0, sigma]])

    def compute_next_state(self, state, control, noise, mu, dt):
        return self.F @ state + self.G @ (control * dt) + self.S @ (mu * dt + noise * np.sqrt(dt))
    

class Dynamics_2:
    def __init__(self, dt, sigma):
        self.dt = dt
        self.sigma = sigma
    
    def compute_next_state(self, state, control, noise, mu, dt):
        next_state = np.zeros_like(state)
        next_state[0] = state[0] + (control[0] * dt + mu[0] * dt + noise[0] * np.sqrt(dt)) * np.cos(state[2])
        next_state[1] = state[1] + (control[0] * dt + mu[0] * dt + noise[0] * np.sqrt(dt)) * np.sin(state[2])
        next_state[2] = state[2] + self.sigma * (control[1]* dt + mu[1] * dt + noise[1] * np.sqrt(dt))
        return next_state


class Obstacle:
    def __init__(self, obstacle_positions, obstacle_radius, boundary_x, boundary_y, obs_cost):
        self.obstacle_positions = np.array(obstacle_positions)
        self.obstacle_radius = np.array(obstacle_radius)
        self.obs_cost = obs_cost
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y

    def compute_obstacle_cost(self, x_curr):
        num_obs = len(self.obstacle_positions)
        cost = 0
        for obs_i in range(num_obs):
            op = self.obstacle_positions[obs_i]
            cost += float(x_curr[0] > op[0] and x_curr[0] < op[0] + self.obstacle_radius[obs_i] and
                          x_curr[1] > op[1] and x_curr[1] < op[1] + self.obstacle_radius[obs_i]) * self.obs_cost
        return cost

    def compute_boundary_penalty(self, x_curr):
        return float(x_curr[0] < self.boundary_x[0] or x_curr[0] > self.boundary_x[1] or
                     x_curr[1] < self.boundary_y[0] or x_curr[1] > self.boundary_y[1]) * self.obs_cost

    def check_hit_obstacle(self, x_curr):
        for pos, r in zip(self.obstacle_positions, self.obstacle_radius):
            if (x_curr[0] > pos[0] and x_curr[0] < pos[0] + r and x_curr[1] > pos[1] and x_curr[1] < pos[1] + r):
                return True
        return False

    def check_hit_boundary(self, x_curr):
        return x_curr[0] < self.boundary_x[0] or x_curr[0] > self.boundary_x[1] or x_curr[1] < self.boundary_y[0] or x_curr[1] > self.boundary_y[1]
    
    def plot_obstacles(self, ax):
        for obs_pos, obs_r in zip(self.obstacle_positions, self.obstacle_radius):
            obs = plt.Rectangle(obs_pos, obs_r, obs_r, color='k', fill=True, zorder=6)
            ax.add_patch(obs)

class RectangularObstacle:
    def __init__(self, obstacle_positions, obstacle_dimensions, boundary_x, boundary_y, obs_cost):
        self.obstacle_positions = np.array(obstacle_positions)
        self.obstacle_dimensions = np.array(obstacle_dimensions)
        self.obs_cost = obs_cost
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y

    def compute_obstacle_cost(self, x_curr):
        num_obs = len(self.obstacle_positions)
        cost = 0
        for obs_i in range(num_obs):
            op = self.obstacle_positions[obs_i]
            od = self.obstacle_dimensions[obs_i]
            cost += float(x_curr[0] > op[0] and x_curr[0] < op[0] + od[0] and
                          x_curr[1] > op[1] and x_curr[1] < op[1] + od[1]) * self.obs_cost
        return cost

    def compute_boundary_penalty(self, x_curr):
        return float(x_curr[0] < self.boundary_x[0] or x_curr[0] > self.boundary_x[1] or
                     x_curr[1] < self.boundary_y[0] or x_curr[1] > self.boundary_y[1]) * self.obs_cost

    def check_hit_obstacle(self, x_curr):
        for pos, dim in zip(self.obstacle_positions, self.obstacle_dimensions):
            if (x_curr[0] > pos[0] and x_curr[0] < pos[0] + dim[0] and x_curr[1] > pos[1] and x_curr[1] < pos[1] + dim[1]):
                return True
        return False

    def check_hit_boundary(self, x_curr):
        return x_curr[0] < self.boundary_x[0] or x_curr[0] > self.boundary_x[1] or x_curr[1] < self.boundary_y[0] or x_curr[1] > self.boundary_y[1]

    def plot_obstacles(self, ax):
        for obs_pos, obs_dim in zip(self.obstacle_positions, self.obstacle_dimensions):
            obs = plt.Rectangle(obs_pos, obs_dim[0], obs_dim[1], color='k', fill=True, zorder=6)
            ax.add_patch(obs)


class CircleObstacle(Obstacle):
    def __init__(self, obstacle_positions, obstacle_radius, boundary_x, boundary_y, obs_cost):
        super().__init__(obstacle_positions, obstacle_radius, boundary_x, boundary_y, obs_cost)

    def compute_obstacle_cost(self, x_curr):
        num_obs = len(self.obstacle_positions)
        cost = 0
        for obs_i in range(num_obs):
            op = self.obstacle_positions[obs_i]
            distance_to_center = np.sqrt((x_curr[0] - op[0])**2 + (x_curr[1] - op[1])**2)
            cost += float(distance_to_center <= self.obstacle_radius[obs_i]) * self.obs_cost
        return cost

    def plot_obstacles(self, ax):
        for obs_pos, obs_r in zip(self.obstacle_positions, self.obstacle_radius):
            obs = plt.Circle(obs_pos, obs_r, color='k', fill=True, zorder=6)
            ax.add_patch(obs)

class PolygonObstacle(Obstacle):
    def __init__(self, obstacle_positions, boundary_x, boundary_y, obs_cost):
        super().__init__(obstacle_positions, None, boundary_x, boundary_y, obs_cost)

    def compute_obstacle_cost(self, x_curr):
        num_obs = len(self.obstacle_positions)
        cost = 0
        for obs_i in range(num_obs):
            polygon = Polygon(self.obstacle_positions[obs_i])
            cost += float(polygon.contains_point(x_curr)) * self.obs_cost
        return cost

    def plot_obstacles(self, ax):
        for obs_pos in self.obstacle_positions:
            obs = Polygon(obs_pos, color='k', fill=True, zorder=6)
            ax.add_patch(obs)

class Map:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def compute_total_obstacle_cost(self, x_curr):
        total_cost = 0
        for obstacle in self.obstacles:
            total_cost += obstacle.compute_obstacle_cost(x_curr)
        return total_cost

    def compute_total_boundary_penalty(self, x_curr):
        total_penalty = 0
        for obstacle in self.obstacles:
            total_penalty += obstacle.compute_boundary_penalty(x_curr)
        return total_penalty

    def check_hit_any_obstacle(self, x_curr):
        for obstacle in self.obstacles:
            if obstacle.check_hit_obstacle(x_curr):
                return True
        return False

    def check_hit_any_boundary(self, x_curr):
        for obstacle in self.obstacles:
            if obstacle.check_hit_boundary(x_curr):
                return True
        return False

    def plot_map(self, ax):
        for obstacle in self.obstacles:
            obstacle.plot_obstacles(ax)


if __name__ == "__main__":

    boundary_x = [-4.0, 1.0]
    boundary_y = [-1.0, 4.0]
    obs_cost = 1e2
    goal_tolerance = 0.1
    dist_weight = 0.01
    obstacle_positions = np.array([[-2.75, 0.25]])
    obstacle_radius = np.array([2.0])

    obstacle = Obstacle(obstacle_positions = obstacle_positions, obstacle_radius = obstacle_radius, boundary_x = boundary_x, boundary_y = boundary_y, obs_cost = obs_cost)
    environment = Map([obstacle])
    fig, ax = plt.subplots(figsize=(10, 6))
    environment.plot_map(ax)
    plt.show()

    dynamics = Dynamics_1(0.05, 0.5)
    print(dynamics.compute_next_state(np.array([0, 0, 0, 0]), np.array([1, 1]), np.array([0, 0]), np.array([0, 0]), 0.05))