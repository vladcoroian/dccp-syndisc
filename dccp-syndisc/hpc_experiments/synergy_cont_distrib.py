import pandas as pd
from neuromaps import datasets, images
import numpy as np
from scipy.stats import rankdata
from pyvinecopulib import Vinecop, RVineStructure
from neuromaps import datasets, nulls, transforms
import dit
from dccp_syndisc import compute_synergy, dccp_synergy
import itertools
from neuromaps import images
import sys
import cvxpy as cvx

def quantize(data, n_bins=2):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:]
    percentile_values = np.percentile(data, percentiles)
    quantized_array = np.digitize(data, bins=percentile_values, right=True)
    return quantized_array


def mi_computation(sources, target, n_bins=3):
    binary_sources = [quantize(source, n_bins) for source in sources]
    binary_target = quantize(target, n_bins)

    data = np.array(binary_sources + [binary_target])
    distribution = np.zeros([n_bins] * (len(sources) + 1))

    # Get all the combinations of the outcomes with [0, 1]
    outcomes = list(itertools.product(range(n_bins), repeat=(len(sources) + 1)))
    data_T = data.T
    for datapoint in data_T:
        distribution[tuple(datapoint)] += 1

    # normalize the distribution to sum to 1
    distribution /= distribution.sum()

    # Add them to a list of outcomes
    outcomes = list(itertools.product(range(n_bins), repeat=(len(sources) + 1)))
    Px = [distribution[outcome] for outcome in outcomes]
    outcomes = ["".join([str(i) for i in outcome]) for outcome in outcomes]
    dist = dit.Distribution(outcomes, Px)

    # Compute the mutual information between any source and the target
    mi = {}
    for i in range(len(sources)):
        mi[f"({[i]}, {len(sources)})"] = dit.shannon.mutual_information(dist, [i], [len(sources)])

    mi[f"({list(range(len(sources)))}, {len(sources)})"] = dit.shannon.mutual_information(dist,
                                                                                          list(range(len(sources))),
                                                                                          [len(sources)])

    return mi


def annotation_synergy(sources, target, n_bins, len_py):
    binary_sources = [quantize(source, n_bins) for source in sources]
    binary_target = quantize(target, n_bins)

    data = np.array(binary_sources + [binary_target])
    distribution = np.zeros([n_bins] * (len(sources) + 1))

    # Get all the combinations of the outcomes with [0, 1]
    data_T = data.T
    for datapoint in data_T:
        distribution[tuple(datapoint)] += 1

    # normalize the distribution to sum to 1
    distribution /= distribution.sum()

    # Add them to a list of outcomes
    outcomes = list(itertools.product(range(n_bins), repeat=(len(sources) + 1)))
    Px = [distribution[outcome] for outcome in outcomes]
    outcomes = ["".join([str(i) for i in outcome]) for outcome in outcomes]
    dist = dit.Distribution(outcomes, Px)
    return dccp_synergy(dist, len_py=len_py, eps=1e-8)[0]
    # return compute_synergy(dist)[0]
    
def estimate_bivar(copula, i, j, grid_size):
    # where c(x, y) = int_{0}^{1} c(x, y, w) dw, which is also done by the monte carlo method
    n_samples = 100
    points = np.random.rand(n_samples, 1)
    # Create triples where the first two elements are i, j and the third element is the random point
    points = np.concatenate((points, np.ones((n_samples, 1)) * i, np.ones((n_samples, 1)) * j), axis=1)
    cop_density = copula.pdf(points)
    return np.mean(cop_density)


def monte_carlo(copula, grid_size):
    coeffs = np.zeros((grid_size, grid_size, grid_size))
    n_samples = 100
    for i in range(grid_size):
        print("i: ", i)
        for j in range(grid_size):
            for k in range(grid_size):
                points = np.random.rand(n_samples, 3)
                for l in range(n_samples):
                    points[l, 0] = i / grid_size + points[l, 0] / grid_size
                    points[l, 1] = j / grid_size + points[l, 1] / grid_size
                    points[l, 2] = k / grid_size + points[l, 2] / grid_size
                c_ijw = np.mean(copula.pdf(points))
                c_ij = np.mean([estimate_bivar(copula, point[1], point[2], grid_size) for point in points])
                coeffs[i, j, k] = c_ijw / c_ij
    return coeffs
    
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

def continuous_synergy(sources, target, grid_size):
    print(target.shape)
    data = np.stack([target, sources[0], sources[1]], axis=-1)
    print(data.shape)
    def to_uniform(data):
        return (rankdata(data, method='average') - 0.5) / len(data)

    uniform_data = np.array([to_uniform(data[:, i]) for i in range(data.shape[1])]).T
    print(uniform_data.shape)
    vine_structure = RVineStructure.simulate(3)
    copula = Vinecop(vine_structure)
    copula.select(uniform_data)
    coeffs = monte_carlo(copula, grid_size)
    max_disc = 0
    for i in range(10):
        values, obj, status = maximise_for(n=grid_size, coeffs=coeffs)
        disclosure = 1 / grid_size * obj
        max_disc = max(max_disc, disclosure)
    return max_disc
    


def singular_spin_test(sources, target, target_perms, grid_size, perm_index):
    s = None
    counter = 0
    while s is None and counter < 10:
        try:
            print('here')
            s = continuous_synergy(sources, target_perms[:, perm_index] if perm_index != 0 else target, grid_size)
        except Exception as e:
            print(e)
            s = None
        counter += 1
    return s if s is not None else 0


fgradient_annotation = datasets.fetch_annotation(source='margulies2016', desc='fcgradient02')
fgradient = images.load_data(fgradient_annotation)


myelinmap_annotation = datasets.fetch_annotation(source='hcps1200', desc='myelinmap')
thickness_annotation = datasets.fetch_annotation(source='hcps1200', desc='thickness')
myelinmap = images.load_data(myelinmap_annotation)
thickness = images.load_data(thickness_annotation)
print("starting..")
group = 10
n_perms = int(sys.argv[2])
grid_size = int(sys.argv[3])
fgrad_perms = nulls.alexander_bloch(fgradient, 'fsLR', '32k', n_perm=n_perms, seed=42)
sources = [myelinmap, thickness]
synergy = singular_spin_test(sources, fgradient, fgrad_perms, grid_size, perm_index=int(sys.argv[1]) - 1)
print(synergy)

with open(f"microstructure_continuous_p={n_perms}_g={grid_size}.csv", "a") as f:
    f.write(str(int(sys.argv[1]) - 1) + "," + str(synergy) + "\n")