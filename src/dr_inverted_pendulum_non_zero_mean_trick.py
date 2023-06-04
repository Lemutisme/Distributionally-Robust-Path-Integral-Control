# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:35:37 2023

@author: Hank Park
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def dlqr(A,B,Q,R) :
    P = scipy.linalg.solve_discrete_are(A,B,Q,R)
    K = - np.linalg.pinv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K

def dlqr_mine(A,B,Q,R,T=250) :
    P = [None] * (T + 1)
    P[-1] = Q
    for t in range(T-1, -1, -1) :
        P[t] = Q + A.T @ P[t+1] @ A - \
        ( B.T @ P[t+1] @ A).T @ np.linalg.pinv( (R + B.T @ P[t+1] @ B) ) \
        @ (B.T @ P[t+1] @ A)
        
    K = [None] * T
    for t in range(T-1, -1, -1) :
        K[t] = -np.linalg.inv(R + B.T @ P[t+1] @ B) @ (B.T @ P[t+1] @ A)

    # return K[0], P
    return K, P

def dlqg(A,B,Q,R,Cov,T=250) :
    P = [None] * (T + 1)
    P[-1] = Q
    for t in range(T-1, -1, -1) :
        P[t] = Q + A.T @ P[t+1] @ A - \
        ( B.T @ P[t+1] @ A).T @ np.linalg.pinv( (R + B.T @ P[t+1] @ B) ) \
        @ (B.T @ P[t+1] @ A)
        
    K = [None] * T
    for t in range(T-1, -1, -1) :
        K[t] = -np.linalg.inv(R + B.T @ P[t+1] @ B) @ (B.T @ P[t+1] @ A)
    
    ctg = np.sum(Cov @ P[1:])
    # return K[0], P[0], ctg
    return K, P[0], ctg

def dleqg(A,B,Q,R,theta,Cov,T = 250) :
    P = [None] * (T + 1)
    P_tilde = [None] * T
    P[-1] = Q
    for t in range(T-1, -1, -1) :
        P_tilde[t] = np.linalg.inv( (np.linalg.inv(P[t+1]) -  Cov * theta) )
        P[t] = Q + A.T @ P_tilde[t] @ A - \
        ( B.T @ P_tilde[t] @ A).T @ np.linalg.inv( (R + B.T @ P_tilde[t] @ B) ) \
        @ (B.T @ P_tilde[t] @ A)
    
    K = [None] * T
    for t in range(T-1, -1, -1) :
        K[t] = -np.linalg.inv(R + B.T @ P_tilde[t] @ B) @ (B.T @ P_tilde[t] @ A)
    
    ctg = - 1/ theta * np.sum( [ np.log(
                                    np.linalg.det( np.identity(A.shape[0]) -  theta * Cov @ P[t] ) 
                                    ) for t in range(1, T+1) ] )
    # return K, P, ctg
    return K, P[0], ctg

def dleqg_inv(A,B,Q,R,theta,Cov,T = 250) :
    P = [None] * (T + 1)
    P_tilde = [None] * T
    P[-1] = Q
    for t in range(T-1, -1, -1) :
        P_tilde[t] = np.linalg.inv( (np.linalg.inv(P[t+1]) -  Cov / theta) )
        P[t] = Q + A.T @ P_tilde[t] @ A - \
        ( B.T @ P_tilde[t] @ A).T @ np.linalg.inv( (R + B.T @ P_tilde[t] @ B) ) \
        @ (B.T @ P_tilde[t] @ A)
    
    K = [None] * T
    for t in range(T-1, -1, -1) :
        K[t] = -np.linalg.inv(R + B.T @ P_tilde[t] @ B) @ (B.T @ P_tilde[t] @ A)
    
    ctg = - theta * np.sum( [ np.log(
                                    np.linalg.det( np.identity(A.shape[0]) - (1 / theta) * Cov @ P[t] ) 
                                    ) for t in range(1, T+1) ] )
    # return K, P, ctg
    return K, P[0], ctg

def opt_cost(A, B, Q, R, gamma, theta, Cov) : 
    # K, P, ctg = dleqg_inv(A, B, Q, R, theta, Cov, T=nsteps)
    K, P, ctg = dleqg_inv(A, B, Q, R, theta, Cov, T=nsteps)
    xk = np.matrix("0 ; 0 ; .2 ; 0; 1")
    target_xk = np.matrix("0 ; 0 ; 0.0 ; 0; 1")
    state_error = xk - target_xk
    cost = gamma * theta + (state_error.T @ P @ state_error) + ctg
    return cost

# %%
# Construct 2D True Gaussian Mixture Dist 
true_mus = [np.array([0,0]), np.array([-1, 0]), np.array([1.01, 0])]
true_covs = [np.array([[1e-1,0],[0,1e-2]]), np.array([[1e-3,0],[0,1e-4]]), np.array([[1e-3,0],[0,1e-4]])]
true_probs = np.array([0, 0.5, 0.5])
acc_probs = [np.sum(true_probs[:i]) for i in range(1, len(true_probs)+1)]

# Plot 2D True Gaussian Mixture Samples
n = 10
samples = np.zeros([n,2])

for n in range(len(samples)) :
    # sample uniform
    r = np.random.uniform(0, 1)
    # select gaussian
    k = 0
    for i, threshold in enumerate(acc_probs):
        if r < threshold:
            k = i
            break

    selected_mu = true_mus[k]
    selected_cov = true_covs[k]

    # sample from selected gaussian
    lambda_, gamma_ = np.linalg.eig(selected_cov)

    dimensions = len(lambda_)
    # sampling from multivariate normal distribution
    x_multi = np.random.multivariate_normal(selected_mu,selected_cov)
    samples[n,:] = x_multi

# Try Uniform
# samples = np.c_[np.random.uniform(-0.1,0.1, n), np.random.uniform(-0.01,0.01, n)]
# samples = np.c_[np.random.uniform(-10,1, n), np.random.uniform(-10,1, n)]

# Try Triangular
# samples = np.c_[np.random.triangular(-0.6, 0.2, 0.3, n), np.random.triangular(-0.06, 0.02, 0.03, n)]
# samples = np.c_[np.random.triangular(-5, 2, 3, n), np.random.triangular(-0.5, 0.2, 0.3, n)]
# samples = np.c_[np.random.triangular(-3, 0, 3, n), np.random.triangular(-0.3, 0.0, 0.3, n)]

n_estimates = 5000
samples_estimates = np.random.multivariate_normal(np.mean(samples, axis=0),np.cov(samples.T),
                                                  n_estimates)

fig = plt.figure()
ax = fig.add_subplot(111)
samples = np.array(samples)
ax.scatter(samples[:, 0], samples[:, 1], label="GMM", s = 3.0, alpha = 0.5)
ax.scatter(samples_estimates[:, 0], samples_estimates[:, 1], label="Nominal Gaussian", s = 3.0, alpha = 0.5)

plt.title('Distribution Mismatch')
plt.ylabel('Disturbance in Angle')
plt.xlabel('Disturbance in Location')
plt.legend(loc='best')
plt.show()    

Cov_ = np.cov(samples.T)  
Cov = np.array([
                [Cov_[0,0],  0,     Cov_[0,1], 0, 0],
                [0,          0,     0,         0, 0],
                [Cov_[0,1],  0,     Cov_[1,1], 0, 0],
                [0,          0,     0,         0, 0],
                [0,          0,     0,         0, 0]
               ])

mu_ = np.mean(samples, axis=0)
mu = np.matrix([mu_[0], 0, mu_[1], 0, 0]).T


###################################################
############ Inverted Pendulum Problem ############
###################################################


l = .22 # rod length is 2l
m = (2*l)*(.006**2)*(3.14/4)*7856 # rod 6 mm diameter, 44cm length, 7856 kg/m^3
M = .4
dt = .02 # 20 ms
g = 9.8

A = np.matrix([[1, dt, 0, 0],[0,1, -(3*m*g*dt)/(7*M+4*m),0],[0,0,1,dt],[0,0,(3*g*(m+M)*dt)/(l*(7*M+4*m)),1]])
A_ = np.block(
             [[A,                        np.zeros([A.shape[0],1])],
              [np.zeros([1,A.shape[0]]), 1                      ]]
              )
B = np.matrix([[0],[7*dt/(7*M+4*m)],[0],[-3*dt/(l*(7*M+4*m))]])
B_ = np.block([[B],[0]])
Q_ = np.matrix("1 0 0 0 0; 0 10e-10 0 0 0; 0 0 1 0 0; 0 0 0 10e-10 0; 0 0 0 0 10e-10")
# Q_ = np.block([[np.identity(4), x_bar.T]]).T @ Q @ np.block([[np.identity(4), x_bar.T]])
R = np.matrix("0.0005")
nsteps = 100
time = np.linspace(0, 2, nsteps, endpoint=True)

############### Out-of-sample simulation ##################
############### Out-of-sample simulation ##################
############### Out-of-sample simulation ##################



############################
##### Ternary search method #####
############################
gammas = [10**-3,10**-2,10**-1,10**0,2**1,3**1]
optimal_values = []
optimal_lambdas = []
for gamma in gammas :
    lambda_l = 0; lambda_r = 1e7
    # lambda_l = -1e10; lambda_r = 0
    k = 0
    K = 300
    
    while k < K :
        lambda_1 = lambda_l + (lambda_r - lambda_l) / 3
        lambda_2 = lambda_l + 2 * (lambda_r - lambda_l) / 3
        
        opt_cost_1 = opt_cost(A_, B_, Q_, R, gamma, lambda_1, Cov )
        opt_cost_2 = opt_cost(A_, B_, Q_, R, gamma, lambda_2, Cov )
        
        
        if opt_cost_1 > opt_cost_2:
            lambda_l = lambda_1
        elif opt_cost_1 < opt_cost_2 :
            lambda_r = lambda_2
        # else :
        #     if lambda_1 == lambda_2 :
        #         break
        #     else :
        #         lambda_l, lambda_r = lambda_1, lambda_2
        
        optimal_cost = min(opt_cost_1, opt_cost_2)
        k += 1
        # print(f"iteration{k} {abs(lambda_l - lambda_r)}" )
        # print(f"iteration{k} Cost: {optimal_cost}")
        # print(f"iteration{k} Cost: {obj_val_two}, lambda_r:{lambda_r}")    
    optimal_values.append(optimal_cost)
    optimal_lambdas.append(lambda_l)    
    print(f"gamma is {gamma} and optimal lambda is {optimal_lambdas[-1]}")
    # print(opt_cost( lambda_r )    )

# Create OOS test data
rep = 300
samples_OOS = np.zeros([rep,nsteps,2])
for j in range(rep) :
    for n in range(nsteps) :
        r = np.random.uniform(0,1)
        k = 0
        for i, threshold in enumerate(acc_probs):
            if r < threshold:
                k = i
                break
        selected_mu = true_mus[k]
        selected_cov = true_covs[k]
        samples_OOS[j,n,:] = np.random.multivariate_normal(selected_mu,selected_cov)

# Try Uniform
# for j in range(rep) :
#     samples_OOS[j,:,:] = np.c_[np.random.uniform(-0.1,0.1, nsteps), np.random.uniform(-0.01,0.01, nsteps)]

# Try Triangular
# for j in range(rep) :
#     samples_OOS[j,:,:] = np.c_[np.random.triangular(-0.6, 0.2, 0.3, nsteps), np.random.triangular(-0.06, 0.02, 0.03, nsteps)]

# # Try the case where true distribution is indeed Normal.
# for j in range(rep) :
#     samples_OOS[j,:,:] = np.random.multivariate_normal([0,0],Cov_,nsteps)


t_w = [t for t in range(nsteps)]
 
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
lower_qantile = 0.10 ; upper_qantile = 0.90

OOS_costs_lqg = []
OOS_costs_lqg_trj = []
X_lqg = []
T_lqg = []
U_lqg = []
X_leqg = []
T_leqg = []
U_leqg = []
# for j in range(rep) :
for i in range(2) :
    if i == 0 :
        K, P, ctg_ = dlqg(A_, B_, Q_, R, Cov, T=nsteps)
        # K, P, ctg = dleqg_inv(A, B, Q, R, 10, Cov, T=nsteps)

        for n in range(rep) :
            xk = np.matrix("0 ; 0 ; .2 ; 0 ; 1")
            target_xk = np.matrix("0 ; 0 ; 0.0 ; 0 ; 1")
            # OOS_cost = (xk - target_xk).T*Q*(xk - target_xk)
            X = []
            T = []
            U = []
            OOS_costs = []
            OOS_cost = 0
            OOS_costs.append(OOS_cost)
            for t in range(len(time)):
                uk = K[t]*(xk - target_xk)
                X.append(xk[0,0])
                T.append(xk[2,0])
                v = xk[1,0]
                force = uk[0,0]
                accel = force/(M+m)
                u = ((1-.404)*v + dt*accel)/.055/10
                U.append(u)
                if t in t_w :
                    noise = np.matrix([samples_OOS[n,t,0] - mu_[0], 0, samples_OOS[n,t,1] - mu_[1], 0, 0]).T
                    xk = A_*xk + B_*uk + noise 
                    # xk = A*xk + B_robust*uk + noise
                else :
                    xk = A_*xk + B_*uk
                if t < len(time) - 1 :
                    OOS_cost += xk.T*Q_*xk + uk.T*R*uk
                    OOS_costs.append(OOS_cost[0,0])
                else : 
                    OOS_cost += xk.T*Q_*xk
                    OOS_costs.append(OOS_cost[0,0])
            OOS_costs_lqg.append(OOS_cost[0,0])
            OOS_costs_lqg_trj.append(OOS_costs)
            X_lqg.append(X)
            T_lqg.append(T)
            U_lqg.append(U)
            
    else :
        # X_leqg = []
        # T_leqg = []
        # U_leqg = []
        for i, lmbda in enumerate([optimal_lambdas[-1]]) :
        
            # theta = 79.3935376942728
            K, P, ctg = dleqg_inv(A_, B_, Q_, R, lmbda, Cov, T=nsteps)
            # K, P = dlqr_mine(A, B, Q, R, T=nsteps)
            OOS_costs_leqg = []
            OOS_costs_leqg_trj = []
            # X_leqg = []
            # T_leqg = []
            # U_leqg = []
            for n in range(rep) :
                xk = np.matrix("0 ; 0 ; .2 ; 0; 1")
                target_xk = np.matrix("0 ; 0 ; 0.0 ; 0 ; 1")
                # OOS_cost = (xk - target_xk).T*Q*(xk - target_xk)
                X = []
                T = []
                U = []
                OOS_costs = []
                OOS_cost = 0
                OOS_costs.append(OOS_cost)
                for t in range(len(time)):
                    uk = K[t]*(xk - target_xk)
                    # uk = K*(xk - target_xk)
                    X.append(xk[0,0])
                    T.append(xk[2,0])
                    v = xk[1,0]
                    force = uk[0,0]
                    accel = force/(M+m)
                    u = ((1-.404)*v + dt*accel)/.055/10
                    U.append(u)
                    if t in t_w :
                        noise = np.matrix([samples_OOS[n,t,0] - mu_[0], 0, samples_OOS[n,t,1] - mu_[1], 0, 0]).T
                        xk = A_*xk + B_*uk + noise 
                        # xk = A*xk + B_robust*uk + noise
                    else :
                        xk = A_*xk + B_*uk
                    if t < len(time) - 1 :
                        OOS_cost += xk.T*Q_*xk + uk.T*R*uk
                        OOS_costs.append(OOS_cost[0,0])
                    else : 
                        OOS_cost += xk.T*Q_*xk
                        OOS_costs.append(OOS_cost[0,0])
                OOS_costs_leqg.append(OOS_cost[0,0])
                OOS_costs_leqg_trj.append(OOS_costs)
                X_leqg.append(X)
                T_leqg.append(T)
                U_leqg.append(U)
                
                # if i == 4 :
                #     axes[0].plot(range(len(time)), np.median(np.array(X_leqg[i]), axis=0), label=f"DRO $\gamma$={5}")                    
                #     axes[1].plot(range(len(time)), np.median(np.array(T_leqg[i]), axis=0), label=f"DRO $\gamma$={5}")
                # else :
                #     axes[0].plot(range(len(time)), np.median(np.array(X_leqg[i]), axis=0), label=f"DRO $\gamma$={10**(-3+i)}")
                #     axes[1].plot(range(len(time)), np.median(np.array(T_leqg[i]), axis=0), label=f"DRO $\gamma$={10**(-3+i)}")

axes[0].plot(range(len(time)), np.median(np.array(X_lqg), axis=0),'--', color='k', label="LQG", linewidth=3, alpha = 0.8)            
axes[1].plot(range(len(time)), np.median(np.array(T_lqg), axis=0),'--', color='k', label="LQG", linewidth=3, alpha = 0.8)
axes[0].fill_between(range(len(time)), y1 = np.quantile(np.array(X_lqg), lower_qantile, axis=0), y2 = np.quantile(np.array(X_lqg), upper_qantile, axis=0),alpha=0.2,label='_nolegend_',color="k")
axes[1].fill_between(range(len(time)), y1 = np.quantile(np.array(T_lqg), lower_qantile, axis=0), y2 = np.quantile(np.array(T_lqg), upper_qantile, axis=0),alpha=0.2,label='_nolegend_',color="k")

axes[0].plot(range(len(time)), np.median(np.array(X_leqg), axis=0), label=f"DRO $\gamma$={gammas[np.argwhere(np.array(optimal_lambdas) == lmbda)[0,0]]}", color="g")
axes[1].plot(range(len(time)), np.median(np.array(T_leqg), axis=0), label=f"DRO $\gamma$={gammas[np.argwhere(np.array(optimal_lambdas) == lmbda)[0,0]]}", color="g")
axes[0].fill_between(range(len(time)), y1 = np.quantile(np.array(X_leqg), lower_qantile, axis=0), y2 = np.quantile(np.array(X_leqg), upper_qantile, axis=0),alpha=0.2,label='_nolegend_',color="g")
axes[1].fill_between(range(len(time)), y1 = np.quantile(np.array(T_leqg), lower_qantile, axis=0), y2 = np.quantile(np.array(T_leqg), upper_qantile, axis=0),alpha=0.2,label='_nolegend_',color="g")

axes[0].set_title('Cart Position (meters)')
axes[0].legend(loc='best')
axes[0].grid()
axes[1].set_title('Pendulum Angle (radians)')
axes[1].legend(loc='best')
axes[1].grid()
plt.show()

fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)

ax.boxplot([OOS_costs_lqg,OOS_costs_leqg])
ax.set_xticklabels(["LQG", "DRO"])
ax.set_title("Cost under Gaussian mixture with non-zero mean")
# # %%

# plt.plot(range(len(time)+1), np.median(np.array(OOS_costs_lqg_trj), axis=0), label=f"DRO $\gamma$={gammas[np.argwhere(np.array(optimal_lambdas) == lmbda)[0,0]]}", color="g")
# plt.fill_between(range(len(time)+1), y1 = np.quantile(np.array(OOS_costs_lqg_trj), lower_qantile, axis=0), y2 = np.quantile(np.array(OOS_costs_lqg_trj), upper_qantile, axis=0),alpha=0.2,label='_nolegend_',color="g")
# plt.plot(range(len(time)+1), np.median(np.array(OOS_costs_leqg_trj), axis=0), label=f"DRO $\gamma$={gammas[np.argwhere(np.array(optimal_lambdas) == lmbda)[0,0]]}", color="k")
# plt.fill_between(range(len(time)+1), y1 = np.quantile(np.array(OOS_costs_leqg_trj), lower_qantile, axis=0), y2 = np.quantile(np.array(OOS_costs_leqg_trj), upper_qantile, axis=0),alpha=0.2,label='_nolegend_',color="k")


# plt.set_title('Pendulum Angle (radians)')
# plt.legend(loc='best')
# plt.grid()
# plt.show()