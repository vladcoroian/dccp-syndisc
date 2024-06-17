from neuromaps import datasets, nulls, transforms
import numpy as np
import dit
from dccp_syndisc import compute_synergy, dccp_synergy
import itertools
from neuromaps import images
import sys

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


def singular_spin_test(sources, target, target_perms, perm_index, n_bins, len_py):
    s = None
    counter = 0
    while s is None and counter < 20:
        try:
            s = annotation_synergy(sources, target_perms[:, perm_index] if perm_index != 0 else target, n_bins=n_bins, len_py=len_py)
        except Exception as e:
            s = None
        counter += 1
    return s if s is not None else 0


print(f"given argument was {int(sys.argv[1])}, perms={int(sys.argv[2])}, bins={int(sys.argv[3])}")
print('starting')
fgradient_annotation = datasets.fetch_annotation(source='margulies2016', desc='fcgradient02')
fgradient = images.load_data(fgradient_annotation)

# Uncomment for other sets of sources

# --- Set 1 ----
myelinmap_annotation = datasets.fetch_annotation(source='hcps1200', desc='myelinmap')
thickness_annotation = datasets.fetch_annotation(source='hcps1200', desc='thickness')
myelinmap = images.load_data(myelinmap_annotation)
thickness = images.load_data(thickness_annotation)

# --- Set 2 ----
# cbf = datasets.fetch_annotation(source='raichle', desc='cbf')
# cdf = images.load_data(cbf)
# cbv = datasets.fetch_annotation(source='raichle', desc='cbv')
# cbv = images.load_data(cbv)
# cmr02 = datasets.fetch_annotation(source='raichle', desc='cmr02')
# cmr02 = images.load_data(cmr02)
# cmrglc = datasets.fetch_annotation(source='raichle', desc='cmrglc')
# cmrglc = images.load_data(cmrglc)
# fgradient = images.load_data(transforms.fslr_to_fslr(fgradient_annotation, target_density='164k'))

# --- Set 3 ----
# delta_power = datasets.fetch_annotation(source='hcps1200', desc='megdelta')
# delta_power_32k = images.load_data(transforms.fslr_to_fslr(delta_power, target_density='32k'))

# theta_power = datasets.fetch_annotation(source='hcps1200', desc='megtheta')
# theta_power_32k = images.load_data(transforms.fslr_to_fslr(theta_power, target_density='32k'))

# alpha_power = datasets.fetch_annotation(source='hcps1200', desc='megalpha')
# alpha_power_32k = images.load_data(transforms.fslr_to_fslr(alpha_power, target_density='32k'))

# beta_power = datasets.fetch_annotation(source='hcps1200', desc='megbeta')
# beta_power_32k = images.load_data(transforms.fslr_to_fslr(beta_power, target_density='32k'))

# low_gamma_power = datasets.fetch_annotation(source='hcps1200', desc='meggamma1')
# low_gamma_power_32k = images.load_data(transforms.fslr_to_fslr(low_gamma_power, target_density='32k'))

# high_gamma_power = datasets.fetch_annotation(source='hcps1200', desc='meggamma2')
# high_gamma_power_32k = images.load_data(transforms.fslr_to_fslr(high_gamma_power, target_density='32k'))

# gamma_power_32k = low_gamma_power_32k + high_gamma_power_32k
# fgradient = images.load_data(transforms.fslr_to_fslr(fgradient_annotation, target_density='4k'))

group = 10
n_perms = int(sys.argv[2])
n_bins = int(sys.argv[3])
len_py = int(sys.argv[4])
print('really starting')
fgrad_perms = nulls.alexander_bloch(fgradient, 'fsLR', '32k', n_perm=n_perms, seed=42)

print('really really starting')

# sources = [cdf, cbv, cmr02, cmrglc]
# sources = [delta_power_32k, theta_power_32k, alpha_power_32k, beta_power_32k, gamma_power_32k]
sources = [myelinmap, thickness]
synergy = singular_spin_test(sources, fgradient, fgrad_perms, perm_index=int(sys.argv[1]) - 1, n_bins=n_bins, len_py=len_py)
print(synergy)

with open(f"microstructure_spin_test_p={n_perms}_b={n_bins}_len_py={len_py}.txt", "a") as f:
    f.write(str(int(sys.argv[1]) - 1) + " " + str(synergy) + "\n")