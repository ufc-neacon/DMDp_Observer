# tank_EKF_example.py

# This script implements the Extended Kalman Filter (EKF) for the tank system example presented in the project.
# Functionality:

#     Implements the EKF algorithm for state estimation of the nonlinear tank system.

#     Simulates the estimation process with measurement noise to reflect realistic conditions.

#     Saves the estimated state data to support later comparison with other methods.


# Usage Notes:

#     This script should be executed before running tank_polynomial_example.py, in order to generate the EKF data required for comparison.

#     Ensure all dependencies are properly installed before execution.







import numpy as np
import matplotlib.pyplot as plt


#Definition of the tank modeling functions  differential equations.
#_______________________________________________________________________________________________________________________________
def tank_model_discrete(xk, uk):
    """ Reservoir discrete model (tank) """
    r = 0.635
    R = 0.73
    H = 1.38
    q_nominal = 14 / 3600
    ki = q_nominal / (2 * np.pi * 60)
    ko = 0.6 * ki * 2 * np.pi * 60 / (np.sqrt(0.60 * H))

    xk = max(xk, 1e-6)  # ensure that xk​ remains non-negative
    xk1 = xk + (ki * uk - ko * np.sqrt(xk)) / (
        np.pi * r**2 + 2 * np.pi * r * xk * (R - r) / H
        + np.pi * xk**2 * (R - r)**2 / H**2
    )
    return max(min(xk1, H), 0)  # Guarantee that xx stays within the physical limits of the tank.


#__________________________________________________________________________________________________________________
#Jacobian computation 
def jacobian_tank(xk, uk):
    """ Cálculo numérico da Jacobiana """
    epsilon = 1e-5
    f_x_plus = tank_model_discrete(xk + epsilon, uk)
    f_x_minus = tank_model_discrete(xk - epsilon, uk)
    return (f_x_plus - f_x_minus) / (2 * epsilon)

#Extended Kalman filter implementation
class EKF:
    def __init__(self, f_model, jacobian_f, P0=0.1):
        self.P = np.array([P0], dtype=np.float64)  # Initial covariance
        self.f_model = f_model  # Model prediction function.
        self.jacobian_f = jacobian_f  # Jacobian update
    
    def step(self, x_prev, u, y_measured, Q, R):
        F = self.jacobian_f(x_prev, u)  # Jacobian
        x_pred = self.f_model(x_prev, u)  # predicted state
        P_pred = F * self.P * F + Q  # covariance update

        H = 1  # Observer Matrix (y = x)
        S = H * P_pred * H + R  # Inovação
        K = P_pred * H / S  # Kalman gain
        x_est = x_pred + K * (y_measured - H * x_pred)  # prediction update
        P_est = (1 - K * H) * P_pred  #
        
        self.P = P_est  # covariance update for next iteration
        return x_est  # predicted state




    
T_max_ekf = 20000 ##total Simulation duration.
dt = 1
Ta_ekf = np.arange(0, T_max_ekf, dt)
cu = 1  #Physical parametric constants of the reservoir.

x_real_ekf = 0
x_est_ekf = 0.105
xT_real_ekf = []
xT_est_ekf = []
uT_ekf = []
ruido_ekf = []
erro_medio_T_ekf = []

ekf = EKF(f_model=tank_model_discrete, jacobian_f=jacobian_tank)

for i in Ta_ekf:
    u = cu * np.pi * 60 * (1 - np.cos(2 * np.pi * i / 2500))
    x_real_ekf = tank_model_discrete(x_real_ekf, u)
    eta = 0*1e-2 * (np.random.rand() - 0.5)
    y_measured = x_real_ekf + eta  # noise  added to enhance realism and reflect practical signal
    
    ruido_ekf.append(eta)
    xT_real_ekf.append(x_real_ekf)
    uT_ekf.append(u)
    
    R_est = 0.1325 
    Q_est = 1 
    
    x_est_ekf = ekf.step(x_est_ekf, u, x_real_ekf, Q_est, R_est) 
    xT_est_ekf.append(x_est_ekf)
    
xT_real_ekf = np.array(xT_real_ekf) #real output
xT_est_ekf = np.array(xT_est_ekf)[:, 0]   #Filter predicted output
e_2k = np.abs(xT_real_ekf - xT_est_ekf)  #prediction error
print(np.mean(e_2k))

#______________________________________________________________________________________________________________
#ploting obtained EKF results


#_______________________________________________________________________________________________________________
#tank level Real VS EKF
plt.figure(figsize=(10, 6))
plt.plot(Ta_ekf[0:11], e_2k[0:11], label="EKF Error", color="green")
plt.xlabel("Time (s)")
plt.grid()
plt.legend(loc='upper right')
plt.show()

#EKF Error analysis
plt.figure(figsize=(10, 6))
plt.plot(Ta_ekf, xT_real_ekf, label='Real', color="red", linestyle ="dashed")
plt.plot(Ta_ekf, xT_est_ekf, label='EKF', color="green", linestyle =':', )
plt.xlabel("Time (s)")
plt.ylabel("Tank Level (m)")
plt.legend(loc='lower right')
plt.grid()
plt.show()

#______________________________________________________________________________________________
#Save EKF data to allow direct comparison with the proposed method
#Available in tank_polynomial_example.py"

np.save('xT_est_ekf_data.npy', xT_est_ekf)
np.save('e_2k_data.npy', e_2k)
