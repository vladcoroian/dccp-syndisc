import os
import cvxpy as cvx
import dccp
import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.linalg import det, inv, LinAlgError
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


# Function to generate bivariate Gaussian copula
def bivariate_gaussian_copula(n, rho):
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


# Function to generate trivariate Gaussian copula
def trivariate_gaussian_copula(n, rho12=0.5, rho23=0.5, rho13=0.5):
    C = np.zeros((n + 1, n + 1, n + 1))
    xis = np.linspace(0, 1, n + 1)
    cov_matrix = [[1, rho12, rho13],
                  [rho12, 1, rho23],
                  [rho13, rho23, 1]]

    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                u_1 = norm.ppf(xis[i])
                u_2 = norm.ppf(xis[j])
                u_3 = norm.ppf(xis[k] if k != 0 else 1e-6)
                C[i, j, k] = multivariate_normal.cdf([u_1, u_2, u_3], cov=cov_matrix)
                if np.isnan(C[i, j, k]):
                    C[i, j, k] = 0
    return C


# Function to compute trivariate mean copula density
def trivariate_mean_copula_density(n, copulas):
    copula_density = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                copula_density[i, j, k] = copulas[i + 1, j + 1, k + 1] - copulas[i, j + 1, k + 1] - copulas[i + 1, j, k + 1] - copulas[i + 1, j + 1, k] + copulas[i, j, k + 1] + copulas[i, j + 1, k] + copulas[i + 1, j, k] - copulas[i, j, k]
    return copula_density / (1 / n) ** 3


# Function to compute bivariate mean copula density
def bivariate_mean_copula_density(n, copulas):
    copula_density = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            copula_density[i, j] = copulas[i + 1, j + 1] - copulas[i, j + 1] - copulas[i + 1, j] + copulas[i, j]
    return copula_density / (1 / n) ** 2


# Function to maximize the given problem
def maximise_for(n, coeffs):
    x = []
    constraints = []
    for i in range(n):
        x.append(cvx.Variable(n))
        x[i].value = [1] * n
        constraints += [x[i] >= 0]
        constraints += [cvx.sum(x[i]) == n]

    for j in range(n):
        c = 0
        for i in range(n):
            c += x[i][j]
        constraints += [c == n]

    final_expr = 0
    for w in range(n):
        expr = 0
        for i in range(n):
            expr += x[i] @ coeffs[w, i, :]
        final_expr += cvx.entr(expr / n ** 2)

    objective = cvx.Maximize(-final_expr)
    prob = cvx.Problem(objective, constraints)
    prob.solve(method='dccp', max_iter=50, solver=cvx.GUROBI, verbose=False, ccp_times=1, warm_start=True)
    return [x[i].value for i in range(n)], objective.value, prob.status


# Function to check if a matrix is positive definite
def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except LinAlgError:
        return False


# Function to compute the Gaussian copula density
def gaussian_copula_density(rho12, rho23, rho13, u1, u2, u3):
    R = np.array([
        [1.0, rho12, rho13],
        [rho12, 1.0, rho23],
        [rho13, rho23, 1.0]
    ])

    if not is_positive_definite(R):
        raise ValueError("The correlation matrix R is not positive definite.")

    eps = 1e-10
    u1 = np.clip(u1, eps, 1 - eps)
    u2 = np.clip(u2, eps, 1 - eps)
    u3 = np.clip(u3, eps, 1 - eps)

    R_inv = inv(R)
    det_R = det(R)
    phi_inv_u = np.array([norm.ppf(u1), norm.ppf(u2), norm.ppf(u3)])
    exponent = -0.5 * np.dot(phi_inv_u.T, np.dot((R_inv - np.eye(3)), phi_inv_u))
    c_R_Gauss = (1.0 / np.sqrt(det_R)) * np.exp(exponent)
    return c_R_Gauss


# Function to compute the Gaussian copula density for 2D
def gaussian_copula_density_2d(rho, u1, u2):
    R = np.array([
        [1.0, rho],
        [rho, 1.0]
    ])

    if not is_positive_definite(R):
        raise ValueError("The correlation matrix R is not positive definite.")

    eps = 1e-10
    u1 = np.clip(u1, eps, 1 - eps)
    u2 = np.clip(u2, eps, 1 - eps)
    R_inv = inv(R)
    det_R = det(R)
    phi_inv_u = np.array([norm.ppf(u1), norm.ppf(u2)])
    exponent = -0.5 * np.dot(phi_inv_u.T, np.dot((R_inv - np.eye(2)), phi_inv_u))
    c_R_Gauss = (1.0 / np.sqrt(det_R)) * np.exp(exponent)
    return c_R_Gauss


# Function for Monte Carlo integration
def monte_carlo_integration(n, rho12, rho23, rho13):
    n_samples = 100
    copula_density = np.zeros((n, n, n))
    print("Starting Monte Carlo integration")
    for i in range(n):
        print("i = ", i)
        for j in range(n):
            for k in range(n):
                points = np.random.rand(n_samples, 3)
                for l in range(n_samples):
                    points[l, 0] = i / n + points[l, 0] / n
                    points[l, 1] = j / n + points[l, 1] / n
                    points[l, 2] = k / n + points[l, 2] / n
                gaussian_3d = [gaussian_copula_density(rho12, rho23, rho13, point[0], point[1], point[2]) for point in points]
                gaussian_2d = [gaussian_copula_density_2d(rho12, point[0], point[1]) for point in points]
                copula_density[k, i, j] = np.mean([gaussian_3d[l] / gaussian_2d[l] for l in range(n_samples)])
    return copula_density


# Function for non-self-disclosure experiment
def non_self_disclosure(grid_size, rho12, rho23, rho13):
    coeffs = monte_carlo_integration(grid_size, rho12, rho23, rho13)
    values, obj, status = maximise_for(n=grid_size, coeffs=coeffs)
    disclosure = 1 / grid_size * obj
    for j in range(grid_size):
        print(values[j])
    print("-" * 10)
    print("Final disclosure: ", disclosure)

    # Value of main diagonal and secondary diagonal
    # Sometimes dccp misses this optimal solutions
    main_diag = 0
    for w in range(grid_size):
        expr = 0
        for i in range(grid_size):
            expr += coeffs[w, i, grid_size - i - 1] / grid_size
        main_diag += 1 / grid_size * expr * np.log(expr)
    print("Value of main diagonal: ", main_diag)
    secondary_diag = 0
    for w in range(grid_size):
        expr = 0
        for i in range(grid_size):
            expr += coeffs[w, i, i] / grid_size
        secondary_diag += 1 / grid_size * expr * np.log(expr)
    print("Value of secondary diagonal: ", secondary_diag)
    return max(disclosure, main_diag, secondary_diag)

def mutual_information_gaussian(rho12, rho23, rho13):
    # Construct the full covariance matrix for X1, X2, X3
    Sigma = np.array([
        [1, rho12, rho13],
        [rho12, 1, rho23],
        [rho13, rho23, 1]
    ])

    Sigma_12 = Sigma[:2, :2]
    Sigma_3 = np.array([[1]])

    # Calculate entropies
    H_X1X2X3 = 0.5 * np.log((2 * np.pi * np.e) ** 3 * np.linalg.det(Sigma))
    H_X1X2 = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(Sigma_12))
    H_X3 = 0.5 * np.log(2 * np.pi * np.e * np.linalg.det(Sigma_3))

    # Mutual Information I(X1, X2; X3)
    mutual_info = H_X1X2 + H_X3 - H_X1X2X3
    return mutual_info

def plot_synergy_by_rho(n, m):
    rhos = np.linspace(0.0, 0.95, m)
    data1, data2, data3 = [], [], []
    for rho12 in rhos:
        data1.append(non_self_disclosure(n, rho12=rho12, rho23=0.65, rho13=0.65))
        data2.append(non_self_disclosure(n, rho12=rho12, rho23=0.5, rho13=0.5))
        data3.append(non_self_disclosure(n, rho12=rho12, rho23=0.2, rho13=0.2))

    plt.figure(figsize=(12, 10))
    plt.plot(rhos, data1, label='$\\rho_{23}=0.65, \\rho_{13}=0.65$')
    plt.scatter(rhos, data1)
    plt.plot(rhos, data2, label='$\\rho_{23}=0.5, \\rho_{13}=0.5$')
    plt.scatter(rhos, data2)
    plt.plot(rhos, data3, label='$\\rho_{23}=0.2, \\rho_{13}=0.2$')
    plt.scatter(rhos, data3)

    plt.xlabel("Correlation $\\rho_{12}$")
    plt.ylabel("Synergy")
    plt.title("Synergy for different values of $\\rho_{12}$")
    plt.legend()
    plt.savefig(os.path.expanduser("~/Desktop/project_data/synergy_gaussian_rho12_multiple.png"))
    plt.show()

    # Save data to CSV
    df = pd.DataFrame({
        'rho12': rhos,
        'rho23=0.65, rho13=0.65': data1,
        'rho23=0.5, rho13=0.5': data2,
        'rho23=0.2, rho13=0.2': data3
    })
    df.to_csv(os.path.expanduser("~/Desktop/project_data/synergy_gaussian_rho12_multiple.csv"), index=False)

def plot_synergy_heatmap(n, m):
    rho23_values = np.linspace(0.15, 0.65, m)
    rho13_values = np.linspace(0.15, 0.65, m)
    disclosures = np.zeros((m, m))

    for j, k in itertools.product(range(m), range(m)):
        print(f"Starting rho12 = {0.0}, rho23 = {rho23_values[j]}, rho13 = {rho13_values[k]}")
        disclosures[j, k] = non_self_disclosure(n, rho12=0, rho23=rho23_values[j], rho13=rho13_values[k])
        print(f"Completed rho23 = {rho23_values[j]}, rho13 = {rho13_values[k]}")

    # Plot the disclosures on a seaborn heatmap
    sns.heatmap(data=disclosures, xticklabels=np.round(rho13_values, 2), yticklabels=np.round(rho23_values, 2), cmap="flare")
    plt.xlabel("Correlation $\\rho_{13}$")
    plt.ylabel("Correlation $\\rho_{23}$")
    plt.title("Synergy values for independent sources")
    plt.savefig(os.path.expanduser("~/Desktop/project_data/synergy_gaussian_independent_2.png"))
    plt.show()

def plot_synergy_grid_size():
    # Correlation parameters
    rho12 = 0.3
    correlations = [(0.3, 0.65, 0.65), (0, 0.65, 0.65), (0.7, 0.2, 0.2)]
    
    # Initialize the data collection
    results = {}
    
    ns = range(3, 26, 2)
    for rho12, rho13, rho23 in correlations:
        mi = mutual_information_gaussian(rho12, rho23, rho13)
        data = []
        for n in ns:
            data.append(non_self_disclosure(n, rho12=rho12, rho23=rho23, rho13=rho13))
        results[(rho12, rho13, rho23)] = data
    
        # Write label in latex
        label = f"$\\rho_{{12}} = {rho12}, \\rho_{{13}} = {rho13}, \\rho_{{23}} = {rho23}$"
        # Plot the data
        plt.plot(ns, data, label=label)
        plt.scatter(ns, data)
    
    plt.xlabel("Grid size $n$")
    plt.ylabel("Synergy")
    plt.title("Synergy for different values of $n$ and correlations")
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.expanduser("~/Desktop/project_data/synergy_gaussian_grid_correlations.png"))
    plt.show()
    
    # Save the data to a CSV file
    df = pd.DataFrame(results, index=ns)
    df.index.name = "Grid size"
    df.to_csv(os.path.expanduser("~/Desktop/project_data/synergy_gaussian_grid_correlations.csv"))

if __name__ == "__main__":
    n, m = 10, 10
    # plot_synergy_by_rho(n, m)
    # plot_synergy_heatmap(n, m)
    plot_synergy_grid_size()

