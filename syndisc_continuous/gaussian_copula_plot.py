import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, multivariate_normal, norm
from scipy.special import expm1, log1p

import matplotlib as mpl

# Enable LaTeX rendering
mpl.rcParams['text.usetex'] = True

# Set the font to Charter
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{XCharter}'
# Plotting Boxplot
plt.rcParams.update({'font.size': 22})
# Also font size for ticks and labels
plt.rcParams.update({'xtick.labelsize': 22, 'ytick.labelsize': 22})

# Axis label font size
plt.rcParams.update({'axes.labelsize': 22})

# Set style for seaborn
sns.set(style="white")


# Function to generate Gaussian copula
def generate_copula_data(n, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    data = multivariate_normal.rvs(mean=mean, cov=cov, size=n)
    copula_data = norm.cdf(data)
    return copula_data

# Generate data
n_samples = 50000
rho = 0.7
data = generate_copula_data(n_samples, rho)

# Create a joint plot with marginal histograms
g = sns.JointGrid(x=data[:, 0], y=data[:, 1], space=0)
g.plot_joint(sns.kdeplot, cmap="YlOrRd", fill=True)
g.plot_marginals(sns.histplot, color="r", bins=30, kde=False)

# Adjust limits and labels
g.ax_joint.set_xlim(0, 1)
g.ax_joint.set_ylim(0, 1)
g.ax_joint.set_xlabel('Uniform Dimension 1')
g.ax_joint.set_ylabel('Uniform Dimension 2')
g.ax_marg_x.set_xlim(0, 1)
g.ax_marg_y.set_ylim(0, 1)

# Change axis labels size and ticks size
g.ax_joint.tick_params(axis='both', which='major', labelsize=14)
g.ax_joint.set_xlabel('Uniform Marginal $u_1$', fontsize=16)
g.ax_joint.set_ylabel('Uniform Marginal $u_2$', fontsize=16)
g.ax_marg_x.tick_params(axis='both', which='major', labelsize=14)

# Add margins and padding to the plot g
g.fig.subplots_adjust(top=0.85, right=0.9, left=0.15, bottom=0.1)

# Add colorbar for the density
cbar_ax = g.fig.add_axes([.91, .25, .02, .4])  # x, y, width, height
plt.colorbar(g.ax_joint.collections[0], cax=cbar_ax)
# Format numbers in the colorbar with 2 decimal places
cbar_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

plt.subplots_adjust(right=0.9)
plt.margins(0.1)

output_path = f'~/Desktop/project_data/gaussian_example_{rho}.png'
plt.savefig(os.path.expanduser(output_path))

plt.show()
