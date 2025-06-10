

import os
# from IPython import get_ipython
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
from IPython import get_ipython




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

#############################################################################
#              DMD ALGORITHM FOR POLYNOMIAL APPROXIMATION: DMDp 
#############################################################################

u = uV[:-1]
x0 = xV[:-1, :]
x1 = xV[1:, :]

H = lambda x: [x[:,0], x[:,1], x[:,0]**2, x[:,1]**2, x[:,0]*x[:,1], x[:,0]**2*x[:,1], x[:,0]*x[:,1]**2] # set of monomials definition
H0 = np.array(H(x0))
W0 = np.array([u])

sx = np.dot(x1.T, np.linalg.pinv(np.vstack([W0, H0]))) # The effective dmd operation

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

#############################################################################
#               POLYNOMIAL OBSERVER BASED ON THE ABOVE DMDp
#############################################################################

Kpol = np.array([0.35,0.95])                # Observer gain
Sob = lambda u, x, ee: B * u + np.dot(A, np.array([[x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], x[0]**2*x[1], x[0]*x[1]**2 ]]).T) + Kpol@ee.T 

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

#############################################################################
#         POLYNOMIAL APPROXIMATION ANALYSIS OF THE DMDp APPROACH
#############################################################################

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

bound = np.abs(np.sum(Ba.T@Q[:2,:2]@Ba)+np.sum(Aa.T@Q[2:,2:]@Aa))
print(bound)
xbound = bound
ybound = bound

#############################################################################
#                              PLOTTING AREA
#############################################################################
plt.ioff()
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
plt.show()

plt.ioff()
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
plt.show()



#______________________________________________________________________________________________
# #Save Proposed and DMDc data to allow direct comparison with the proposed method
# #Available in tank_EKF_example.py"

# np.save('x1_est_DMDp_data.npy', XX)
#np.save('x1_real_data.npy', xr)

# np.save('x2_est_DMDp_data.npy', YY)
#np.save('x2_real_data.npy', yr)

#x1_Aprox_DMDp_er=[(x - xr) for x, xr in zip(XX, xr)]
# np.save('x1_Aprox_DMDp_er_data.npy', x1_Aprox_DMDp_er)
# np.save('x1bound_DMDp_data.npy', xbound)

#x2_Aprox_DMDp_er=[(x - yr) for x, yr in zip(YY, yr)]
# np.save('x2_Aprox_DMDp_er_data.npy', x2_Aprox_DMDp_er)
# np.save('x2bound_DMDp_data.npy', ybound)


#np.save('x1_sob_DMDp.npy', XX_sob)
#np.save('x2_sob_DMDp.npy', YY_sob)






#############################################################################
#                               PLOTTING AREA
# #############################################################################

# time = np.arange(0,1e-1,1e-6)
# time = time[0:-40001]

# plt.figure(figsize=(10, 6))
# plt.plot(time,XX_sob[:-39998]-xr[:-39998], color='blue', label= 'Proposed: $x_1$ (position)', linewidth=3)
# plt.plot(time,YY_sob[:-39998]-yr[:-39998], color='cyan', linestyle='--', label='Proposed: $x_2$ (angle)', linewidth=3)
# plt.plot(time, e_2_x[:-40001], label='EKF: $x_1$ (position)', color="orange",linewidth=2)
# plt.plot(time, e_2_y[:-40001], label='EKF: $x_2$ (angle)', linestyle="dashed", color="red",linewidth=2)
# plt.xlabel("Time (s)")
# plt.ylabel("Absolute value")
# plt.title("State error convergence")
# plt.grid()
# plt.legend()
# plt.savefig('Observer_Comparison.eps')
# plt.show()