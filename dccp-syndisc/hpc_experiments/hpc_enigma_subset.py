from enigmatoolbox.datasets import fetch_ahba
import numpy as np
import dit
from dccp_syndisc import dccp_synergy
import itertools
import sys
from enigmatoolbox.datasets import risk_genes, load_summary_stats, load_fsa5
from enigmatoolbox.permutation_testing import spin_test, shuf_test, rotate_parcellation, centroid_extraction_sphere
import os
import importlib_resources

import h5py
import pandas as pd

def quantize(data, n_bins=2):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:]
    percentile_values = np.percentile(data, percentiles)
    quantized_array = np.digitize(data, bins=percentile_values, right=True)
    return quantized_array


def mi_computation(sources, target, n_bins=3):
    binary_sources = [quantize(source, n_bins) for source in sources]
    binary_target = target

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
    binary_target = target

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
    return dccp_synergy(dist, len_py=len_py, eps=1e-10, iterations=20)[0]


def singular_spin_test(sources, target, target_perms, perm_index, n_bins, len_py):
    s = None
    counter = 0
    while s is None and counter < 20:
        try:
            s = annotation_synergy(sources, target_perms[perm_index] if perm_index != 0 else target, n_bins=n_bins, len_py=len_py)
        except Exception as e:
            s = None
        counter += 1
    return s if s is not None else 0

def load_epilepsy_data(resolution):
    print("Path at terminal when executing this file")
    print(os.getcwd() + "\n")

    print("This file path, relative to os.getcwd()")
    print(__file__ + "\n")

    print("This file full path (following symlinks)")
    full_path = os.path.realpath(__file__)
    print(full_path + "\n")

    print("This file directory and name")
    path, filename = os.path.split(full_path)
    print(path + ' --> ' + filename + "\n")

    print("This file directory only")
    print(os.path.dirname(full_path))

    dir = os.path.dirname(full_path) + "/data/"
    file = f'Schaefer{resolution}_EpilepsyNeuroSynth.mat'
    print(os.listdir())
    fullpath = os.path.join(dir, file)
    with h5py.File(fullpath, 'r') as f:
        data = f.get('pattern')
        cortex = resolution - resolution % 100
        data = np.array(data)[:, :cortex]
        bin_target = np.where(data > 0, 1, 0)[0, :]
        return bin_target

def load_gene_expression_data(df, resolution, subset_genes):
    # Get the names of epilepsy-related genes (Focal HS phenotype)
    cortex = resolution - resolution % 100
    sources_df = df[df.columns.intersection(subset_genes)].to_numpy()[:cortex, :].T
    sources_df = np.nan_to_num(sources_df) # Change NaN values to 0
    return sources_df

def syn_epilepsy_with_res(resolution, n_bins=2, len_py=10):
    bin_target = load_epilepsy_data(resolution)
    sources_df = load_gene_expression_data(resolution)
    synergy = annotation_synergy(sources_df, bin_target, n_bins=n_bins, len_py=len_py, ground_truth=False, is_binary_target=True)
    return synergy


n_perms = int(sys.argv[2])
n_bins = 2
len_py = int(sys.argv[3])
resolution = 400

surface_name = 'fsa5'
parcellation_name = f'schaefer_{resolution}'
gene_expression_data = pd.read_csv(f'~/synergy/examples/data/Schaefer{resolution}_gene_expression.csv')

epilepsy_genes = risk_genes('epilepsy')['allepilepsy']
subset_size = int(sys.argv[4])
sources_subsets = [list(subset) for subset in itertools.combinations(epilepsy_genes, subset_size)]

target = load_epilepsy_data(resolution)
# Run spin test
sphere_lh, sphere_rh = load_fsa5(as_sphere=True)

root_pth = str(importlib_resources.files('enigmatoolbox').joinpath('permutation_testing'))

# get sphere coordinates of parcels
annotfile_lh = os.path.join(root_pth, 'annot', surface_name + '_lh_' + parcellation_name + '.annot')
annotfile_rh = os.path.join(root_pth, 'annot', surface_name + '_rh_' + parcellation_name + '.annot')

lh_centroid = centroid_extraction_sphere(sphere_lh.Points, annotfile_lh)
rh_centroid = centroid_extraction_sphere(sphere_rh.Points, annotfile_rh)

# generate permutation maps
permutation = rotate_parcellation(lh_centroid, rh_centroid, n_perms)
target_perms = []
for i in range(n_perms):
    perm = [int(x) for x in permutation[:, i]]
    target_perms.append(target[perm])

for (i, subset) in enumerate(sources_subsets): 
    sources = load_gene_expression_data(gene_expression_data, resolution, subset)
    synergy = singular_spin_test(sources, target, target_perms, perm_index=int(sys.argv[1]) - 1, n_bins=n_bins, len_py=len_py)
    print(synergy)
    with open(f"subsets_{subset_size}_{resolution}_len_py={len_py}.csv", "a") as f:
        f.write(str(i) + "," + str(int(sys.argv[1]) - 1) + "," + str(synergy) + "\n")
