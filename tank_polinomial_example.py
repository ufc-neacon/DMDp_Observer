# tank_polynomial_example.py

# This script implements and performs a comparative analysis between the proposed polynomial-based method and the Dynamic Mode Decomposition.
# Functionality:

#     Implements the polynomial modeling method described in the project.

#     Compares the model's predictive performance against DMDc.

#     Plots the simulation results, highlighting the error between the proposed method and DMDc.

#     Saves the generated data to enable subsequent comparison with the Extended Kalman Filter (EKF) implementation (tank_EKF_example.py).

# Usage Notes:

# Make sure this script is executed after running the tank_EKF_example.py script if you plan to include EKF in the performance comparison.





import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from SOSPy import *
from sympy import symbols, Matrix, diff
import random

import matplotlib as mpl


#Definition of the tank modeling functions and their differential equations.
#_______________________________________________________________________________________________________________________________
def cylinder_volume(r, R, H, h):
    return np.pi * r**2 * h + np.pi * r * h**2 * (R - r) / H + np.pi * h**3 * (R - r)**2 / (3 * H**2)

def ode_system(h, t, ki, ko, r, R, H, w):
    return (ki * w - ko * np.sqrt(h)) / (np.pi * r**2 + 2 * np.pi * r * h * (R - r) / H + np.pi * h**2 * (R - r)**2 / H**2)

def tank_model(xk, uk, cko):
    # Parameters
    r = 0.635
    R = 0.73
    H = 1.38
    q_nominal = 14 / 3600
    ki = q_nominal / (2 * np.pi * 60)
    ko = cko * ki * 2 * np.pi * 60 / (np.sqrt(0.60 * H))

    # Solve ODE
    t_span = [0, 1]
    ode_solution = odeint(ode_system, xk, np.linspace(t_span[0], t_span[1], 1000), args=(ki, ko, r, R, H, uk))

    # Extract the final value
    xk1 = ode_solution[-1]

    # Ensure the value is within the limits [0, H]
    xk1 = abs(xk1)
    xk1 = max(min(xk1, H), 0)

    return float(xk1), float(ki), float(ko)
#________________________________________________________________________________________________________________________________


#definition of the function that implements the proposed polynomial model.
# and Computing [B A] from data driven polynomial decomposition (from returned S)
#_________________________________________________________________________________________________________________________________
def DMDp(xOLHistory, uOLHistory, mtype):

    try:
        # Signal X0
        X0 = np.array(xOLHistory[:-1], dtype=object)

        # Signal X1 = X'
        X1 = np.array(xOLHistory[1:], dtype=object)

        # Input Signal U
        U = np.array(uOLHistory[:-1])

        # Signal Zo polinômio
        if mtype == 1:
            Z = lambda x: [x]
        elif mtype == 2:
            Z = lambda x: [x, x**2]
        elif mtype == 3:
            Z = lambda x: [x, x**2, x**3]
        elif mtype == 4:
            Z = lambda x: [x, x**2, x**3, x**4]
        elif mtype == 5:
            Z = lambda x: [x, x**2, x**3, x**4, x**5]
        elif mtype == 6:
            Z = lambda x: [x, x**2, x**3, x**4, x**5, x**6]    
        elif mtype == 7:
            Z = lambda x: [x, x**2, x**3, x**4, x**5, x**6, x**7]

        Zo = Z(X0)

        # W(x) polinôm
        W = lambda x: 1

        # Signal Wo
        Wo = W(X0) * U

        # make sure that Wo e Zo are numeric NumPy 
        Wo = np.asarray(Wo, dtype=float)
        Zo = np.asarray(Zo, dtype=float)
        X0 = np.asarray([X0], dtype=float)
        X1 = np.asarray([X1], dtype=float)
        U = np.asarray([U], dtype=float)

        # Matrix S = [B A]
        S = np.dot(X1, np.linalg.pinv(np.vstack([Wo, Zo])))

        Li, Lj = np.shape(X0)
        Li = Li + 1
        rU = np.array([U[0][:-1-Li]])

        for j in range(1,Li):
            rU = np.vstack((rU, U[0][j:-1-Li+(j+1)-1]))

        rank_UX = np.linalg.matrix_rank(np.vstack([U, X0]))
        size_UX = np.vstack([U, X0]).shape
        rank_X = np.linalg.matrix_rank(X0)
        size_X = X0.shape
        size_U = U.shape
        rank_QU = np.linalg.matrix_rank(rU)
        size_QU = np.array(rU).shape

        if rank_QU == size_X[0] + 1:
            InPE = True
            if rank_UX == size_U[0] + size_X[0]:
                OutPE = True

        S = S[0]
        if mtype == 1:
            Sx = lambda u, x: S[0]*u + S[1]*x
        elif mtype == 2:
            Sx = lambda u, x: S[0]*u + S[1]*x + S[2]*x**2
        elif mtype == 3:
            Sx = lambda u, x: S[0]*u + S[1]*x + S[2]*x**2 + S[3]*x**3 
        elif mtype == 4:
            Sx = lambda u, x: S[0]*u + S[1]*x + S[2]*x**2 + S[3]*x**3 + S[4]*x**4
        elif mtype == 5:
            Sx = lambda u, x: S[0]*u + S[1]*x + S[2]*x**2 + S[3]*x**3 + S[4]*x**4 + S[5]*x**5
        elif mtype == 6:
            Sx = lambda u, x: S[0]*u + S[1]*x + S[2]*x**2 + S[3]*x**3 + S[4]*x**4 + S[5]*x**5 + S[6]*x**6
        elif mtype == 7:
            Sx = lambda u, x: S[0]*u + S[1]*x + S[2]*x**2 + S[3]*x**3 + S[4]*x**4 + S[5]*x**5 + S[6]*x**6 + S[7]*x**7

        return S, Sx, InPE, OutPE

    except Exception as e:

        return None
#__________________________________________________________________________________________________________________________________________




deg = 5   # Desired polynomial degree.
Ts = 1    #sampleTime
Duration = 20000 #total Simulation duration.
cko = 0.6   #Physical parametric constants of the reservoir.
cu = 1

Ta = np.arange(0, Duration, Ts).tolist()

u = 0
x = 0
xT = []
uT = []
ruido = []





for i in Ta:
    u = cu * np.pi * 60 * (1 - np.cos(2 * np.pi * i / 2500)) # Simulated input signal applied to the system.
    x, _, _ = tank_model(x, u, cko) # System response: reservoir level.
    eta = 1*1e-2*(random.random()-0.5) # Measurement noise was added to the simulation to enhance realism and reflect practical signal acquisition conditions
    ruido.append(eta)
    xT.append(x)
    uT.append(u+eta)

xT = np.array(xT)
uT = np.array(uT)

#Matrices obtained for the proposed model with the selected polynomial degree and the DMDc comparison solution.
S1, Sx1, InPE1, OutPE1 = DMDp(xT, uT, 1)
S0, Sx, InPE, OutPE = DMDp(xT, uT, deg)

xDMDp_1 = []
xDMDp = []
x1 = xT[0]
x = xT[0]

for i in Ta:

    x1 = Sx1(uT[i], x1)
    xDMDp_1.append(x1)

    x = Sx(uT[i], x)
    xDMDp.append(x)

####### SOS Options #######
erro_x = x1-x
options = {}
options['solver'] = 'cvxopt'

####### SOS Problem 1: Find Q #######

x1, x2 = symbols("x1 x2")
vartable = [x1, x2]

ar = np.arange(0,deg+1,1)
for i in ar:
    S0[i] = '{:.2g}'.format(S0[i])


w = (np.mean(ruido)-np.var(ruido))
Ip = np.eye(deg+1)
R = np.hstack((S0[0],np.zeros(deg)))
S = np.array([R,np.hstack((np.zeros(1),S0[1:]))])
Ortho=np.dot(R,np.hstack((np.zeros(1),S0[1:])))

prog = sosprogram(vartable)
prog, Q = sospolymatrixvar(prog, monomials(vartable,[0]), [2,2])
eq1 = S.T@Q@S

prog = sosmatrixineq(prog,  -Q)
prog = sosmatrixineq(prog,  -eq1 + 0.11*1e-9*Ip)

prog = sossolve(prog)

Q = np.double(sosgetsol(prog, Q))
print(Q)

####### SOS Problem 2: Find w #######

x1, x2 = symbols("x1 x2")
vartable = [x1, x2]

prog = sosprogram(vartable)

xx =  np.array([
    [x1],
    [x1**2],
    [x1**3],
    [x1**4],
    [x1**5],
    [x1**6],
    [x1**7],
])
xx = xx[0:deg]
A = S0[1:]
B = S0[0]

K_est = 1.1

if deg == 1:
    Sob = lambda u, x, ee: B * u + np.dot(A, np.array([[x ]]).T) + K_est*ee #(xV[2:,1]-X[2:,1])
    outfile = 'Output1.eps'
    obsfile = 'Observer1.eps'
elif deg == 2:
    Sob = lambda u, x, ee: B * u + np.dot(A, np.array([[x, x**2 ]]).T) + K_est*ee #(xV[2:,1]-X[2:,1])
    outfile = 'Output2.eps'
    obsfile = 'Observer2.eps'
elif deg == 3:
    Sob = lambda u, x, ee: B * u + np.dot(A, np.array([[x, x**2, x**3 ]]).T) + K_est*ee #(xV[2:,1]-X[2:,1])
    outfile = 'Output3.eps'
    obsfile = 'Observer3.eps'
elif deg == 4:
    Sob = lambda u, x, ee: B * u + np.dot(A, np.array([[x, x**2, x**3, x**4 ]]).T) + K_est*ee #(xV[2:,1]-X[2:,1])
    outfile = 'Output4.eps'
    obsfile = 'Observer4.eps'
elif deg == 5:
    Sob = lambda u, x, ee: B * u + np.dot(A, np.array([[x, x**2, x**3, x**4, x**5 ]]).T) + K_est*ee #(xV[2:,1]-X[2:,1])
    outfile = 'Output5.eps'
    obsfile = 'Observer5.eps'
elif deg == 6:
    Sob = lambda u, x, ee: B * u + np.dot(A, np.array([[x, x**2, x**3, x**4, x**5, x**6 ]]).T) + K_est*ee #(xV[2:,1]-X[2:,1])
    outfile = 'Output6.eps'
    obsfile = 'Observer6.eps'
elif deg == 7:
    Sob = lambda u, x, ee: B * u + np.dot(A, np.array([[x, x**2, x**3, x**4, x**5, x**6, x**7 ]]).T) + K_est*ee #(xV[2:,1]-X[2:,1])
    outfile = 'Output7.eps'
    obsfile = 'Observer7.eps'

inpfile = 'Input.eps'   

X_sob = []
x0_sob = 0.105          #

for i in Ta[:-1]:
    u_sob = uT[i]
    x0_sob = Sob(u_sob, x0_sob, xT[i] - x0_sob)
    x0_sob = x0_sob[-1]
    X_sob.append(x0_sob)

X_sob = np.array(X_sob)



f = A @ xx + B * 1 

prog, eps = sospolymatrixvar(prog, monomials(vartable,[0]), [1,1])
prog, sigma = sospolymatrixvar(prog, monomials(vartable,[0]), [1,1])

lam, vec = np.linalg.eigh(Q)

g = np.vstack((f.T+eps,sigma))
eq2 = g.T @ Q @ g

#
prog = sosmatrixineq(prog, -eq2)
prog = sossolve(prog)
eps2 = np.double(sosgetsol(prog, eps))
sigma = np.double(sosgetsol(prog, sigma))
eps=sigma
print('raio = ', eps)

# ####### Print obtained w bounds #######
Ba = S[0,:]
Aa = S[1,:]
bound = np.abs(np.sum(Ba*Q[0,0]*Ba.T)+np.sum(Aa*Q[1,1]*Aa.T)) 
print(bound)

print('Erro Modelo DMDp:', np.max(np.abs(xT[:-1] - xDMDp[:-1])))
print('Erro Modelo DMDc:', max(np.abs(xT - xDMDp_1)))

Aprox_DMDp_er=[np.abs((x - xr)) for x, xr in zip(xDMDp, xT)]
Aprox_DMDc_er=[np.abs((x - xr)) for x, xr in zip(xDMDp_1, xT)]


xDMDp = xDMDp[:-1]
xDMDp_1 = xDMDp_1[:-1]
xT = xT[:-1]







plt.close('all')
plt.ioff()


#______________________________________________________________________________________________________________
#ploting obtained results

#_______________________________________________________________________________________________________________
#tank level Proposed VS DMDc
plt.figure(figsize=(20, 12))
plt.plot(xDMDp, label=f'{deg} order Proposed', color='blue', linestyle='-', linewidth=2)
plt.plot(xDMDp_1, label='DMDc', color='green', linestyle=':', linewidth=2)
plt.plot(xT, label='Real', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('(m)')
plt.title('Tank Level')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("FIG_3a.png", dpi=300, bbox_inches='tight')
plt.show()



#analysis of proposed boundary limit
plt.figure(figsize=(20, 12))
plt.plot([np.abs((x - xr)) for x, xr in zip(xDMDp, xT)], color='blue', linestyle='-', linewidth=2, label=f'{deg} order Proposed')
plt.plot([np.abs((x - xr)) for x, xr in zip(xDMDp_1, xT)], color='green', linestyle=':', linewidth=2, label='DMDc')
plt.plot(bound*np.ones(len(xDMDp)), color='red', linestyle='-.', linewidth=2, label='$\sigma$')
plt.grid(True)
plt.legend(loc='lower right')
plt.title('Output error')
plt.savefig("FIG_3b.png", dpi=300, bbox_inches='tight')
plt.show()

plt.close('all')


#Error analysis
plt.show()
plt.ioff()
e_2 = (xT-X_sob)
plt.figure(figsize=(10, 6))
plt.plot(Ta[0:11],e_2[0:11], 'b', label=f'{deg} order'  , linewidth=2)
plt.xlabel('Time (s)')
plt.ylim(-0.002, 0.015)
plt.legend()
plt.grid(True)
plt.savefig("FIG_7.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close('all')

#______________________________________________________________________________________________
#Save Proposed and DMDc data to allow direct comparison with the proposed method
#Available in tank_EKF_example.py"
np.save('xT_est_DMDp_data.npy', xDMDp)
np.save('Aprox_DMDp_er_data.npy', Aprox_DMDp_er)
np.save('xT_est_DMDc_data.npy', xDMDp_1)
np.save('Aprox_DMDc_er_data.npy', Aprox_DMDc_er)
np.save('bound_last', bound)
np.save('deg_last', deg)
np.save('xT_real_data.npy', xT)
np.save('e_2_data.npy', e_2)


