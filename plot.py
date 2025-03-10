import matplotlib.pyplot as plt
import numpy as np


MSE = np.array([0.059, 0.061, 0.071, 0.087, 0.101])  
KS = np.array([0.360, 0.350, 0.311, 0.237, 0.171]) 
MSE_std = np.array([0.006, 0.006, 0.006, 0.007, 0.008])  
KS_std = np.array([0.029, 0.029, 0.032, 0.040, 0.039])  


plt.figure(figsize=(6, 4))
plt.errorbar(KS, MSE, xerr=KS_std, yerr=MSE_std, fmt='o', capsize=5)


plt.xlabel("KS")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig("Different_ce.png")
