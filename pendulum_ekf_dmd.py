import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from SOSPy import *
from sympy import symbols, Matrix
import random
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset

import matplotlib as mpl

######################### FONT DEFINITIONS ###################################

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font',**font)

rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

######################### PENDULUM RELATED FUNCTIONS ##########################

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

# Initial conditions
x0 = 0
y0 = 0
x = [x0, y0]
xs=x
uV = []
xV = []
xVV=[]
t = np.arange(100000)
dec=2500
ruidox1=[]
ruidox2=[]

###################### INPUT AND STATE DEFINITIONS ################################

for i in t:     
    u = (-0.5*dec* np.cos(2 * np.pi * i / dec)) # Computes input u
    x,xs = system(x, xs, u)
    xV.append(x)
    xVV.append(xs)
    uV.append(u)

xV = np.vstack(xV)
xVV = np.vstack(xVV)

time = np.arange(0,1e-1,1e-6)
time = time[0:-2]

u = uV[:-1]
x0 = xV[:-1, :]
x1 = xV[1:, :]

H = lambda x: [x[:,0], x[:,1], x[:,0]**2, x[:,1]**2, x[:,0]*x[:,1], x[:,0]**2*x[:,1], x[:,0]*x[:,1]**2]
H0 = np.array(H(x0))

W0 = np.array([u])

sx = np.dot(x1.T, np.linalg.pinv(np.vstack([W0, H0])))

A = sx[:,1:]
B = sx[:,:1]

Sx = lambda u, x: B * u + np.dot(A, np.array([[x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], x[0]**2*x[1], x[0]*x[1]**2 ]]).T)

X = []
x0 = xV[0,:]

for i in t[:-1]:
    u = uV[i]
    x0 = Sx(u, x0)
    x0 = x0[:,0]
    X.append(x0)

X = np.array(X)

xr = xV[2:,0]
XX = X[1:,0]
yr = xV[2:,1]
yrr = xVV[2:,1]
YY = X[1:,1]

###################### POLYNOMIAL OBSERVER ################################

Kpol = np.array([0.35,0.95])
Sob = lambda u, x, ee: B * u + np.dot(A, np.array([[x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], x[0]**2*x[1], x[0]*x[1]**2 ]]).T) + Kpol@ee.T #(xV[2:,1]-X[2:,1])

X_sob = []
x0_sob = np.array([0.25,-0.25])


for i in t[:-1]:
    u_sob = uV[i]
    x0_sob = Sob(u_sob, x0_sob, xV[i,:] - x0_sob)
    x0_sob = x0_sob[:,0]
    X_sob.append(x0_sob)

X_sob = np.array(X_sob)

XX_sob = X_sob[1:,0]
YY_sob = X_sob[1:,1]

options = {}
options['solver'] = 'cvxopt'

x1, x2 = symbols("x1, x2")
vartable = [x1, x2]

lin,col = np.shape(sx)

cvar = np.arange(0,col-1,1)
lvar = np.arange(0,lin-1,1)

for j in lvar:
    for i in cvar:
        sx[j,i] = '{:.2g}'.format(sx[j,i])
    
eps = 0.11e-9
Ip = np.eye(8)
S = sx

Ba = np.hstack((sx[:,:1],np.zeros((2,7))))
Aa = np.hstack((np.zeros((2,1)),sx[:,1:]))

S = np.concatenate((Ba,Aa),axis=0)

prog = sosprogram(vartable)

prog, Q = sospolymatrixvar(prog, monomials(vartable,[0]), [4, 4])
print(Q)

eq = S.T@Q@S

prog = sosmatrixineq(prog, - Q)
prog = sosmatrixineq(prog, - eq + eps*Ip)
prog = sossolve(prog)

Q = np.double(sosgetsol(prog, Q))
print(Q)

Ip = np.eye(2)

x1, x2 = symbols('x1 x2')
vartable = [x1, x2]

prog = sosprogram(vartable)

x =  np.array([
    [x1],
    [x2],
    [x1**2],
    [x2**2],
    [x1*x2],
    [x1**2*x2],
    [x2**2*x1]
])

bound = np.abs(np.sum(Ba.T@Q[:2,:2]@Ba)+np.sum(Aa.T@Q[2:,2:]@Aa))#np.trace(Q)#sigma
print(bound)
xbound = bound
ybound = bound

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)  
plt.plot(time[:-1],XX, 'b', label='Proposed Polynomial Approximation', linewidth=2)
plt.plot(time[:-1],xr, color='orange', linestyle='--', label='Real', linewidth=2)
plt.legend(fancybox=True, loc='right')
plt.title('$x_1$ state (position)')#,**csfont)
plt.ylabel('$x_1$ state (position)',fontsize = 16)
plt.grid(True)
plt.subplot(2, 1, 2)  
plt.plot(time[:-1],[(x - xr) for x, xr in zip(XX, xr)], 'b', linewidth=2, label='Proposed Polynomial Approximation')
plt.plot(time[:-1],bound*np.ones(len(xr)), color='red', linestyle='-.', linewidth=2, label='$\sigma$')
plt.legend(fancybox=True, loc='right')#, framealpha=0.1)
plt.xlabel('Time (s)',fontsize = 16)
plt.title('Output (position) error')
plt.grid(True)
plt.tight_layout()
plt.savefig('x1.eps')

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)  # Creating a subplot with 2 rows, 1 column, and select the first subplot
plt.plot(time[:-1], YY,'b', label='Proposed Polynomial Approximation', linewidth=2)
plt.plot(time[:-1], yr, color='orange', linestyle='--', label='Real', linewidth=2)
plt.legend(fancybox=True, loc='right')
plt.title('$x_2$ state (angle)')
plt.ylabel('$x_2$ state (angle)',fontsize = 16)
plt.grid(True)
plt.subplot(2, 1, 2)  # Creating a subplot with 2 rows, 1 column, and select the second subplot
plt.plot(time[:-1],[(y - yr) for y, yr in zip(YY, yr)], 'b', linewidth=2, label='Proposed Polynomial Approximation')
plt.plot(time[:-1],bound*np.ones(len(yr)), color='red', linestyle='-.', linewidth=2, label='$\sigma$')
plt.legend(fancybox=True, loc='right')
plt.xlabel('Time (s)',fontsize = 16)
plt.title('Output (angle) error')
plt.grid(True)
plt.tight_layout()
plt.savefig('x2.eps')

""" e_x1=XX_sob-xr
e_x2=YY_sob-yr
print('erro obs x1: ',np.mean(e_x1))
print('erro obs x2: ',np.mean(e_x2))
fig = plt.figure(figsize=(10, 6))
plt.plot(time[:-1],XX_sob-xr, 'b', label= '$x_1$ state (position)', linewidth=2)
plt.plot(time[:-1],YY_sob-yr, color='orange', linestyle='--', label='$x_2$ state (angle)', linewidth=2)
# plt.plot(time[:-2],[(x - xr) for x, xr in zip(XX, xr+bound)], color='red', linestyle='-.', linewidth=2, label='Upper Bound')
# plt.plot(time[:-2],[(x - xr) for x, xr in zip(XX, xr-bound)], color='red', linestyle='-.', linewidth=2, label='Lower Bound')
plt.grid()
plt.legend()
plt.savefig('Observer.eps') """


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

# Define the system of differential equations including input u
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
    
    # Calculando a Jacobiana (derivadas parciais)
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

# Loop de simulação
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

# Error plotting
e_2_x = xT_real[:, 0] - xT_est[:, 0]  # Position error
e_2_y = xT_real[:, 1] - xT_est[:, 1]  # Angle error

time = np.arange(0,1e-1,1e-6)
time = time[0:-40001]

plt.figure(figsize=(10, 6))
plt.plot(time,XX_sob[:-39998]-xr[:-39998], color='blue', label= 'Proposed: $x_1$ (position)', linewidth=3)
plt.plot(time,YY_sob[:-39998]-yr[:-39998], color='cyan', linestyle='--', label='Proposed: $x_2$ (angle)', linewidth=3)
plt.plot(time, e_2_x[:-40001], label='EKF: $x_1$ (position)', color="orange",linewidth=2)
plt.plot(time, e_2_y[:-40001], label='EKF: $x_2$ (angle)', linestyle="dashed", color="red",linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Absolute value")
plt.title("State error convergence")
plt.grid()
plt.legend()
plt.savefig('Observer_Comparison.eps')
plt.show()