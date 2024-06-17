import itertools
from time import perf_counter
import cvxpy as cvx
import gurobipy

from dccp_utils import components, compute_matrix_A, flatten, quantize, pick_point_inside_polytope
import dit
import numpy as np
import dccp

from syndisc.syndisc import disclosure_channel
from scipy.stats import entropy



def synergy(py, pXgyk, Px, PWgX):
    """Calculate the synergy of a given set of probability distributions."""
    HW = entropy(np.matmul(PWgX, Px), base=2)
    minH = sum(py[k] * entropy(PWgX @ pXgyk[k], base=2) for k in range(len(py)))
    return HW - minH


def select_random_distribution(n):
    """Generate a random probability distribution with constraints on the random variable functions."""
    new_dist = dit.random_distribution(n, ['01'])
    new_dist = dit.insert_rvf(new_dist, lambda x: '1' if all(map(bool, map(int, x))) else '0')
    return new_dist


def compute_synergy(dist):
    """Compute the synergy for a given distribution and measure the computation time."""
    start_time = perf_counter()
    disc_channel = disclosure_channel(dist)
    end_time = perf_counter()
    regular_computation_time = end_time - start_time
    return disc_channel[0], regular_computation_time


def valid_constraints_check(py, pXgyk, Px, PWgX, A, ApX, eps):
    """Check if the constraints are satisfied."""
    if py is None or pXgyk is None:
        return False
    valid_constraints = True
    for k in range(len(py)):
        if np.abs(sum(pXgyk[k]) - 1) > eps:
            valid_constraints = False
            print('ERROR - not prob distrib')
            print(sum(pXgyk[k]))
        if np.allclose(A @ pXgyk[k], ApX, atol=eps) is False:
            valid_constraints = False
            print('ERROR - not in null space')
            print(A @ pXgyk[k])
            print(ApX)
    if np.allclose(sum([py[k] * pXgyk[k] for k in range(len(py))]), Px, atol=eps) is False:
        valid_constraints = False
        print('ERROR - not equal to Px')
        print(sum([py[k] * pXgyk[k] for k in range(len(py))]))
        print(Px)
    if not valid_constraints:
        print('constraints not satisfied')
    return valid_constraints


def dccp_optimisation_polytope(py, PWgX, Px, A, ApX):
    # print('starting polytope phase...')
    pXgyk = []
    for k in range(len(py)):
        x = cvx.Variable(len(Px))
        x.value = Px
        pXgyk.append(x)
    cons = []
    for k in range(len(py)):
        cons += [0 <= pXgyk[k], pXgyk[k] <= 1]
        cons += [cvx.sum(pXgyk[k]) == 1]
        cons += [A @ pXgyk[k] == ApX]
    pXgyk_full = cvx.hstack(pXgyk)
    big_py = np.eye(len(Px)) * py[0]
    for k in range(1, len(py)):
        big_py = np.hstack((big_py, np.eye(len(Px)) * py[k]))
    cons += [big_py @ pXgyk_full == Px]
    big_PWgX = np.kron(np.eye(len(py)), PWgX)
    W_elems = PWgX.shape[0]
    big_py_for_W = []
    for k in range(len(py)):
        big_py_for_W += [py[k]] * W_elems
    prob = cvx.Problem(cvx.Minimize(big_py_for_W @ cvx.entr(big_PWgX @ pXgyk_full)), cons)

    prob.solve(method='dccp', max_iter=50, solver=cvx.GUROBI, verbose=False, ccp_times=1, warm_start=True)
    return [pXgyk[k].value for k in range(len(py))]


def dccp_optimisation_distribution(pXgyk, PWgX, Px, len_py, py_guess=None):
    # print('starting distribution phase...')
    entropies = []
    for k in range(len_py):
        entropies.append(entropy(PWgX @ pXgyk[k], base=2))

    py_var = cvx.Variable(len_py)
    py_var.value = py_guess
    cons = [0 <= py_var, py_var <= 1, cvx.sum(py_var) == 1]
    sum_px_cons = 0
    for k in range(len_py):
        sum_px_cons += py_var[k] * pXgyk[k]
    cons += [sum_px_cons == Px]
    expr = 0
    for k in range(len_py):
        if entropies[k] > 0:
            expr += py_var[k] * entropies[k]
    prob = cvx.Problem(cvx.Minimize(expr), cons)
    prob.solve(solver=cvx.GUROBI, verbose=False, warm_start=True)
    return py_var.value


def dccp_synergy(dist, len_py=10, py_guess=None, iterations=20, eps=1e-10, verbose=False, all_iterations=False):
    start_time = perf_counter()
    P, Px, PWgX = components(dist)
    A, _, _, _ = compute_matrix_A(P, Px)
    ApX = np.matmul(A, Px)
    py = py_guess
    if py_guess is None:
        py = np.random.rand(len_py)
        py = py / sum(py)
    pXgyk = [Px] * len_py
    iter_dict = {}
    best_syn, best_py, best_pXgyK = 0, None, None
    for i in range(iterations):
        py = np.random.rand(len_py)
        py = py / sum(py)
        try:
            pXgyk = dccp_optimisation_polytope(py, PWgX, Px, A, ApX)
        except Exception as e:
            print('ERROR: ', e)
            continue
        if not valid_constraints_check(py, pXgyk, Px, PWgX, A, ApX, eps):
            continue
        if verbose:
            print(f'{i}-1: syn: ', synergy(py, pXgyk, Px, PWgX))
        try:
            py = dccp_optimisation_distribution(pXgyk, PWgX, Px, len_py, py)
        except Exception as e:
            print('ERROR: ', e)
            continue
        if not valid_constraints_check(py, pXgyk, Px, PWgX, A, ApX, eps):
            continue
        syn = synergy(py, pXgyk, Px, PWgX)
        if syn > best_syn:
            best_syn, best_py, best_pXgyK = syn, py, pXgyk
        end_time_iter = perf_counter()
        iter_dict[i + 1] = (best_syn, end_time_iter - start_time)
        if verbose:
            print(f'{i}-2: syn: ', syn)
    if verbose:
        print('------------')
        print('best syn: ', best_syn)
    if all_iterations:
        return iter_dict
    end_time = perf_counter()
    return best_syn, end_time - start_time

def optimal_synergy_channel(dist, len_py=10, py_guess=None, iterations=10, eps=1e-12, verbose=False):
    start_time = perf_counter()
    P, Px, PWgX = components(dist)
    A, _, _, _ = compute_matrix_A(P, Px)
    ApX = np.matmul(A, Px)
    py = py_guess
    if py_guess is None:
        py = np.random.rand(len_py)
        py = py / sum(py)
    pXgyk = [Px] * len_py
    best_syn, best_py, best_pXgyK = 0, None, None
    for i in range(iterations):
        try:
            pXgyk = dccp_optimisation_polytope(py, PWgX, Px, A, ApX)
        except Exception as e:
            print('ERROR: ', e)
            return None, None
        if not valid_constraints_check(py, pXgyk, Px, PWgX, A, ApX, eps):
            return None, None
        if verbose:
            print(f'Iteration {i}-1: Synergy = {synergy(py, pXgyk, Px, PWgX)}')
        try:
            py = dccp_optimisation_distribution(pXgyk, PWgX, Px, len_py, py)
        except Exception as e:
            print('ERROR: ', e)
            return None, None
        if not valid_constraints_check(py, pXgyk, Px, PWgX, A, ApX, eps):
            return None, None
        syn = synergy(py, pXgyk, Px, PWgX)
        if syn > best_syn:
            best_syn, best_py, best_pXgyK = syn, py, pXgyk
        if verbose:
            print(f'Iteration {i}-2: Synergy = {syn}')

    if verbose:
        print('------------')
        print(f'Best synergy achieved: {best_syn}')
    end_time = perf_counter()
    return best_syn, best_pXgyK, best_py


def experiment(n: int, eps: float = 1e-13):
    dist = select_random_distribution(n)
    original_synergy, time = compute_synergy(dist)
    print(f'Original synergy: {original_synergy}')
    dccp_synergy(dist, iterations=2, len_py=10, eps=eps, verbose=True)

# To run an experiment with a distribution of size 4:
# experiment(4)
