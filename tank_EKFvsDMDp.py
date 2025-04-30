


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, Matrix, diff



# Comparison Between Different Implementations

# This script performs a comparative analysis between the proposed polynomial-based method and the DMD approach.

# Prerequisites:
# Before running this script, ensure the following:

#     Execute the EKF implementation by running:
#     tank_EKF_example.py

#     Then, run the comparison script:
#     tank_polynomial_example.py

# These steps are necessary to ensure that all required data for comparison is properly generated and available.




xDMDp=np.load('xT_est_DMDp_data.npy')
Aprox_DMDp_er=np.load('Aprox_DMDp_er_data.npy')
xDMDp_1=np.load('xT_est_DMDc_data.npy')
Aprox_DMDc_er=np.load('Aprox_DMDc_er_data.npy')
bound=np.load('bound_last.npy')
deg=np.load('deg_last.npy')
xT=np.load('xT_real_data.npy')
e_2=np.load('e_2_data.npy')
xT_est_ekf=np.load('xT_est_ekf_data.npy')
e_2k=np.load('e_2k_data.npy')

T_max=20000
dt=1

Ta = np.arange(0, T_max, dt)

#Error analysis
plt.show()
plt.ioff()
plt.figure(figsize=(10, 6))
plt.plot(Ta[0:11],e_2[0:11], 'b', label=f'{deg} order'  , linewidth=2)
plt.plot(Ta[0:11], e_2k[0:11], label="EKF", color="green")
plt.xlabel('Time (s)')
plt.ylim(-0.002, 0.015)
plt.legend()
plt.grid(True)
plt.savefig("FIG_7.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close('all')


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
