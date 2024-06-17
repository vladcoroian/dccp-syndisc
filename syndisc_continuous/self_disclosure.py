import cvxpy as cvx
import dccp
from matplotlib.ticker import LogLocator

import numpy as np
from scipy.stats import norm, multivariate_normal
import itertools
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# Quantize an array to a given precision
def quantize(array, pres=0.01):
    return ((pres ** (-1)) * array).round() * pres

# Set up the matplotlib for LaTeX rendering and font styles
def setup_matplotlib():
    # Enable LaTeX rendering
    mpl.rcParams['text.usetex'] = True

    # Set the font to Charter
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{XCharter}'  # XCharter is an extended set of Charter fonts
    # Plotting Boxplot
    plt.rcParams.update({'font.size': 22})  # Adjust the 14 to whatever size you need
    # Also font size for ticks and labels
    plt.rcParams.update({'xtick.labelsize': 22, 'ytick.labelsize': 22})
    plt.figure(figsize=(12, 8))

def maximise_for(n, copulas=None):
    if copulas is None:
        copulas = np.ones((n, n))
    x = []
    constraints = []
    expr = 0
    for i in range(n):
        x.append(cvx.Variable(n))
        x[i].value = [1] * n
        constraints += [x[i] >= 0]
        constraints += [cvx.sum(x[i]) == n]
        expr += cvx.sum(-cvx.entr(x[i] / copulas[i, :]) @ copulas[i, :])

    for j in range(n):
        c = 0
        for i in range(n):
            c += x[i][j]
        constraints += [c == n]
    objective = cvx.Maximize(cvx.sum(expr))
    prob = cvx.Problem(objective, constraints)
    prob.solve(method='dccp', max_iter=50, solver=cvx.GUROBI, verbose=False, ccp_times=1, warm_start=True)
    return [x[i].value for i in range(n)], objective.value, prob.status

def maximise_row_for(n, row_index, copulas=None):
    if copulas is None:
        copulas = np.ones((n, n))
    log_copulas = np.log(copulas)
    print(log_copulas[row_index, :])
    x = []
    constraints = []
    expr = 0
    x = cvx.Variable(n)
    x.value = [1] * n
    constraints += [x >= 0]
    constraints += [cvx.sum(x) == n]
    # expr += cvx.sum(-cvx.entr(x / copulas[row_index, :]) @ copulas[row_index, :])
    expr = -cvx.sum(x @ log_copulas[row_index, :])

    objective = cvx.Maximize(cvx.sum(expr))
    prob = cvx.Problem(objective, constraints)
    prob.solve(method='dccp', max_iter=50, solver=cvx.GUROBI, verbose=False, ccp_times=1, warm_start=True)
    return x.value, objective.value, prob.status

def gaussian_copula(n, rho):
    C = np.zeros((n + 1, n + 1))
    xis = np.linspace(0, 1, n + 1)
    cov_matrix = [[1, rho], [rho, 1]]
    for i in range(n + 1):
        for j in range(n + 1):
            u_1 = norm.ppf(xis[i])
            u_2 = norm.ppf(xis[j])
            C[i, j] = multivariate_normal.cdf([u_1, u_2], cov=cov_matrix)
            if np.isnan(C[i, j]):
                C[i, j] = 0
    return C

def gumbel_copula(n, theta):
    C = np.zeros((n + 1, n + 1))
    xis = np.linspace(0, 1, n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            C[i, j] = np.exp(-((- np.log(xis[i]))**theta + (- np.log(xis[j]))**theta)**(1 / theta))
    return C

def mean_copula_density(n, copulas):
    copula_density = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            copula_density[i, j] = copulas[i + 1, j + 1] - copulas[i, j + 1] - copulas[i + 1, j] + copulas[i, j]
    return copula_density * n**2

def all_sums(n, mean_copula):
    perms_n = list(itertools.permutations(range(n)))
    best_sum = 0
    best_perm = []
    for perm_i in perms_n:
        for perm_j in perms_n:
            zips = list(zip(perm_i, perm_j))
            s = sum([mean_copula[i, j] for i, j in zips])
            if s > best_sum:
                best_sum = s
                best_perm = zips
    print(best_sum, best_perm)


def plot_synergy():
    synergy_dict = {}
    
    # Vary grid size for independence copula
    for n in range(2, 25):
        print("n: ", n)
        print("-" * 10)
        xis, obj, status = maximise_for(n)
        for i in range(n):
            quant_xi = quantize(xis[i])
            print(quant_xi)
        print(obj / n**2)
        print(status, np.log(n))
        synergy_dict[n] = obj / n**2
        print('-----------------')
        
    # Make a plot of the synergy values for different grid sizes
    plt.plot(synergy_dict.keys(), synergy_dict.values())
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.ylabel("Synergy")
    plt.title("Synergy values for different grid sizes")
    plt.scatter(synergy_dict.keys(), synergy_dict.values())
    plt.xlabel("Grid size $n$")
    plt.savefig(os.path.expanduser("~/Desktop/project_data/self_disclosure_independence.png"))

    
# Plotting functions for specific copulas
def plot_gaussian_results(synergy_dict, n):
    plt.figure(figsize=(12, 8))
    rhos, synergies = synergy_dict.keys(), synergy_dict.values()
    plt.plot(rhos, synergies)
    plt.xlabel("Correlation $\\rho$")
    plt.ylabel("Synergy")
    
    # Add a horizontal line at log(n)
    plt.axhline(y=np.log(n), color='r', linestyle='--', label='Synergy lower bound')
    plt.legend()
    plt.title("Synergy values with Gaussian Copula")
    plt.savefig(os.path.expanduser("~/Desktop/project_data/gaussian_copula_synergy.png"))
    plt.show()

def plot_gumbel_results(synergy_dict, n):
    plt.figure(figsize=(12, 8))
    plt.xscale("log")
    tick_values = [1, 2, 5, 10]
    thetas, synergies = zip(*sorted(synergy_dict.items()))
    plt.plot(thetas, synergies)
    plt.xlabel("Parameter $\\theta$")
    plt.ylabel("Synergy")
    plt.xticks(tick_values, labels=[str(x) for x in tick_values], fontsize=22)
    plt.yticks(fontsize=22)
    
    # Add a horizontal line at log(n)
    plt.axhline(y=np.log(n), color='r', linestyle='--', label='Synergy lower bound')
    plt.legend()
    plt.title("Synergy values with Gumbel Copula")
    plt.savefig(os.path.expanduser("~/Desktop/project_data/gumbel_copula_synergy.png"))
    plt.show()


def collect_and_plot_gaussian(n):
    gaussian_synergy = {}
    for rho in np.linspace(-0.9, 0.9, 19):
        copulas = gaussian_copula(n, rho)
        mean_copula = mean_copula_density(n, copulas)
        _, obj, _ = maximise_for(n, mean_copula)
        gaussian_synergy[rho] = obj / (n**2)
    plot_gaussian_results(gaussian_synergy, n)

def collect_and_plot_gumbel(n):
    gumbel_synergy = {}
    for theta in np.logspace(0, 1, 20):
        copulas = gumbel_copula(n, theta)
        mean_copula = mean_copula_density(n, copulas)
        _, obj, _ = maximise_for(n, mean_copula)
        gumbel_synergy[theta] = obj / (n**2)
    plot_gumbel_results(gumbel_synergy, n)


def perform_experiments():
    setup_matplotlib()
    n = 10
    # collect_and_plot_gaussian(n)
    # collect_and_plot_gumbel(n)
    plot_synergy()

if __name__ == "__main__":
    perform_experiments()

