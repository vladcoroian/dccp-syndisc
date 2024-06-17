import os

import dit
from dccp_syndisc import dccp_synergy, compute_synergy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Enable LaTeX rendering
mpl.rcParams['text.usetex'] = True

# Set the font to Charter
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{XCharter}'

# Generate a distribution with two binary sources with correlation rho
# AND gate with correlated inputs
a = 0.5
b = 0.5

def CorrelatedAND(r):
    return dit.Distribution(['000','010','100','111'], [1-a-b+r, b-r, a-r, r])

rho_range = np.linspace(0.25, 0.5, 30)
gt_values, dccp_values = [], []
for rho in rho_range:
    dist = CorrelatedAND(rho)
    gt_synergy, gt_time = compute_synergy(dist)
    dccp_value, dccp_time = dccp_synergy(dist, iterations=20, eps=1e-10)
    gt_values.append(gt_synergy)
    dccp_values.append(dccp_value)
    print(f'rho={rho}, GT={gt_synergy}, DCCP={dccp_value}')


# Plot two side by side subplots for comparison
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plt.rcParams.update({'font.size': 16})
fig.suptitle('Correlated AND gate')
axs[0].plot(rho_range, gt_values, label='Ground Truth', color='green')
axs[1].plot(rho_range, dccp_values, label='DCCP', color='red')
axs[0].set_xlabel('Correlation', fontsize=16)
axs[0].set_ylabel('Synergy', fontsize=16)
axs[1].set_xlabel('Correlation', fontsize=16)
axs[1].set_ylabel('Synergy', fontsize=16)

# Add some better spacing
plt.tight_layout()
# Ticks font size
axs[0].tick_params(axis='both', which='major', labelsize=16)
axs[1].tick_params(axis='both', which='major', labelsize=16)

axs[0].set_title('Ground truth values')
axs[1].set_title('DCCP estimated values')

fig.savefig(os.path.expanduser("~/Desktop/project_data/corr_and_gate.png"))
plt.show()
