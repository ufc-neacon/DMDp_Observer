
import os
from IPython import get_ipython
# get_ipython().magic('reset -f')
# get_ipython().run_line_magic('clear', '')
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from SOSPy import *
from sympy import symbols, Matrix
import random
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset

import matplotlib as mpl


#############################################################################
#                       PENDULUM RELATED FUNCTIONS
#############################################################################

def pendulum(v, t, u):
    x, y = v

    #Parameters
    b = 0
    m = 1
    g = 9.81
    l = 1
        
    dxdt = y
    dydt = - (b/m)*y - (g/l)*np.sin(x) + u

    return [dxdt, dydt]

# Define the system of differential equations with Ts=1e-6 including input u
def system(x, xss, u):

    # Solve ODE
    t = [0, 1e-6]
    ode1 = odeint(pendulum, x, np.linspace(t[0], t[1], 2), args=(u,))
    ode2 = odeint(pendulum, xss, np.linspace(t[0], t[1], 2), args=(u,))
    _x = ode1[-1] 
    xs = ode2[-1]
    return _x, xs

#############################################################################
#                       INPUT AND STATE DEFINITIONS
#############################################################################

# Initial conditions


#############################################################################
#--------------------- EXTENDED KALMAN FILTER OBSERVER ----------------------
#############################################################################

#############################################################################
#                       PENDULUM RELATED FUNCTIONS
#############################################################################
def pendulum_model(v, t, u):
    x, y = v

    #Parameters
    b = 0
    m = 1
    g = 9.81
    l = 1
        
    dxdt = y
    dydt = - (b/m)*y - (g/l)*np.sin(x) + u

    return np.array([dxdt, dydt])

# Define the system of differential equations with Ts=1e-6 including input u
def pendulum_model_discrete(x, u):

    # Solve ODE
    t = [0, 1e-6]
    ode1 = odeint(pendulum_model, x, np.linspace(t[0], t[1], 2), args=(u,))
    _x = ode1[-1] 
    return _x

# Pendulum Jacobian function
def jacobian_pendulum(xk, uk):
    epsilon = 1e-5
    f_x_plus = pendulum_model_discrete(xk + np.array([epsilon, 0]), uk)
    f_x_minus = pendulum_model_discrete(xk - np.array([epsilon, 0]), uk)
    
    f_y_plus = pendulum_model_discrete(xk + np.array([0, epsilon]), uk)
    f_y_minus = pendulum_model_discrete(xk - np.array([0, epsilon]), uk)
    
    # Computing the partials
    jacobian_matrix = np.array([
        [(f_x_plus[0] - f_x_minus[0]) / (2 * epsilon), (f_x_plus[0] - f_x_minus[0]) / (2 * epsilon)],  # Derivadas parciais de x e y em relação a x
        [(f_y_plus[0] - f_y_minus[0]) / (2 * epsilon), (f_y_plus[1] - f_y_minus[1]) / (2 * epsilon)]   # Derivadas parciais de x e y em relação a y
    ])
    
    return jacobian_matrix

# EKF
class EKF:
    def __init__(self, f_model, jacobian_f, P0=0.1):
        self.P = np.array([[P0, 0], [0, P0]])   # Initial covariation (2 states)
        self.f_model = f_model                  # Model prediction
        self.jacobian_f = jacobian_f            # Jacobian function
    
    def step(self, x_prev, u, y_measured, Q, R):
        F = self.jacobian_f(x_prev, u)          # Jacobian
        x_pred = self.f_model(x_prev, u)        # Prediction
        P_pred = F @ self.P @ F.T + Q           # Covariance update

        H = np.array([[1, 0], [0, 1]])          # Observation matrix (identity for 2 states)
        S = H @ P_pred @ H.T + R                # Inovation
        K = P_pred @ H.T @ np.linalg.inv(np.array(S))   # Kalman Gain
        x_est = x_pred + K @ (y_measured - H @ x_pred)  # Estimation update
        P_est = (np.eye(2) - K @ H) @ P_pred            # Covariance update
        
        self.P = P_est                          # Next step covariance
        return x_est                            # Retorns the estimated state

# Simulation Parameters
Ta = np.arange(100000)

x_real = np.array([0, 0])       # Initial position and velocity
x_est = np.array([0.25,-0.25])  # Initial estimation
xT_real = [x_real]
xT_est = [x_est]
uT = []
ruido = []
erro_medio_T = []

ekf = EKF(f_model=pendulum_model_discrete, jacobian_f=jacobian_pendulum)

# Simulation loop
for i in Ta:
    u = (-0.5*2500* np.cos(2 * np.pi * i / 2500))   # Plant input
    x_real = pendulum_model_discrete(x_real, u)     # State of the plant
    eta = 1e-2 * (np.random.rand(2) - 0.5)          # Noise
    y_measured = x_real + eta                       # Noisy measurements
    
    ruido.append(eta)
    xT_real.append(x_real)
    uT.append(u)
    
    R_est = 2e0     # Input noise (EKF parameter)
    Q_est = 1e-3    # State noise (EKF parameter)
    
    x_est = ekf.step(x_est, u, x_real, Q_est, R_est)  # State update
    xT_est.append(x_est)

# Results
xT_real = np.array(xT_real)
xT_est = np.array(xT_est)

print(xT_real.shape)
print(xT_est.shape)

# Errors
e_2_x = xT_real[:, 0] - xT_est[:, 0]  # Position error
e_2_y = xT_real[:, 1] - xT_est[:, 1]  # Angle error

#############################################################################
#                               PLOTTING AREA
#############################################################################

time = np.arange(0,1e-1,1e-6)
time = time[0:-40001]

plt.figure(figsize=(10, 6))
plt.plot(time, e_2_x[:-40001], label='EKF: $x_1$ (position)', color="orange",linewidth=2)
plt.plot(time, e_2_y[:-40001], label='EKF: $x_2$ (angle)', linestyle="dashed", color="red",linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Absolute value")
plt.title("State error convergence")
plt.grid()
plt.legend()
plt.show()