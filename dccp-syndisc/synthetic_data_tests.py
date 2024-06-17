import os
import dit
from dccp_syndisc import dccp_synergy, compute_synergy
import numpy as np
from dccp_utils import components, pick_point_inside_polytope, ext_pts



def select_big_small_random_distribution(n: int, unique_comp: int = None):
    if unique_comp is None:
        unique_comp = n
    # Generate a random distribution with unique_comp random variables
    d = dit.random_distribution(unique_comp, ['01'])

    # Extend the distribution to n random variables, by adding the same random variable multiple times
    perm = list(range(unique_comp))
    if n - 3 > 0:
        perm += list(np.random.choice(np.arange(unique_comp), n - 3))

    # Create a big distribution with n random variables and a small distribution with unique_comp random variables
    big_dist = d.coalesce([perm], extract=True)
    big_dist = dit.insert_rvf(big_dist, lambda x: '1' if all(map(bool, map(int, x))) else '0')
    small_dist = dit.insert_rvf(d, lambda x: '1' if all(map(bool, map(int, x))) else '0')
    return big_dist, small_dist


def compute_syn_metrics(prediction, ground_truth, algorithm=""):
    prediction, ground_truth = np.array(prediction), np.array(ground_truth)
    efficiency = prediction / ground_truth
    # Mean, max, min, std
    mean, max, min, std = np.mean(efficiency), np.max(efficiency), np.min(efficiency), np.std(efficiency)
    # Print the lowest 5 values and the corresponding gt_synergies, dccp values
    print("Algorithm: ", algorithm)
    print(f"Lowest 5 efficiencies: {efficiency[np.argsort(efficiency)[:5]]}")
    print(f"Corresponding 5 gt synergies: {ground_truth[np.argsort(efficiency)[:5]]}")
    print(f"Corresponding 5 predictions: {prediction[np.argsort(efficiency)[:5]]}")

    print(f"Efficiency: Mean: {mean}, Max: {max}, Min: {min}, Std: {std}")


def pick_almost_uniform(n: int):
    init_dist = dit.example_dists.And(n)
    P, Px, _ = components(init_dist)
    ext_points = ext_pts(P, Px)

    # Pick a point inside the polytope which will act as new distribution Px
    Px = pick_point_inside_polytope(ext_points)
    outcomes = dit.modify_outcomes(init_dist.coalesce(init_dist.rvs[:-1]), lambda x: ''.join(x)).outcomes
    new_dist = dit.Distribution(outcomes, Px)
    new_dist = dit.insert_rvf(new_dist, lambda x: '1' if any(map(bool, map(int, x))) else '0')
    return new_dist


def alphabet_size_testing(iters: int, eps=1e-10):
    # Write the first line in the file "discrete_testing/alphabet_size.csv"
    # Write the header in the file "discrete_testing/alphabet_size.csv" so that columns are separated by commas
    # size5, size10, size15, size20, size25, size30
    with open("discrete_testing/alphabet_size.csv", "w") as f:
        f.write(
            "n,gt_synergy,regular_time,size5_value,size5_time,size10_value,size10_time,size15_value,size15_time,size20_value,size20_time,size25_value,size25_time,size30_value,size30_time\n")

    for n in range(4, 5):
        print(f"---- Testing with {n} random variables, precision {eps} ----")
        regular_syn, regular_time = [], []
        size_dict = {5: [], 10: [], 15: [], 20: [], 25: [], 30: []}
        time_dict = {5: [], 10: [], 15: [], 20: [], 25: [], 30: []}
        for _ in range(iters):
            if n < 5:
                dist = select_random_distribution(n)
            else:
                dist = pick_almost_uniform(n)
            print(f"Iteration {_}")
            gt_synergy, gt_time = compute_synergy(dist)
            if gt_synergy < 1e-3:
                continue

            regular_syn.append(gt_synergy)
            regular_time.append(gt_time)

            for len_py in [5, 10, 15, 20, 25, 30]:
                one_iter, one_iter_time = dccp_synergy(dist, iterations=20, eps=1e-10, len_py=len_py)
                size_dict[len_py].append(one_iter)
                time_dict[len_py].append(one_iter_time)

            with open("discrete_testing/alphabet_size.csv", "a") as f:
                f.write(
                    f"{n},{gt_synergy},{gt_time},{size_dict[5][-1]},{time_dict[5][-1]},{size_dict[10][-1]},{time_dict[10][-1]},{size_dict[15][-1]},{time_dict[15][-1]},{size_dict[20][-1]},{time_dict[20][-1]},{size_dict[25][-1]},{time_dict[25][-1]},{size_dict[30][-1]},{time_dict[30][-1]}\n"
                )

        compute_syn_metrics(size_dict[5], regular_syn, "Size 5")
        compute_syn_metrics(size_dict[10], regular_syn, "Size 10")
        compute_syn_metrics(size_dict[15], regular_syn, "Size 15")
        compute_syn_metrics(size_dict[20], regular_syn, "Size 20")
        compute_syn_metrics(size_dict[25], regular_syn, "Size 25")
        compute_syn_metrics(size_dict[30], regular_syn, "Size 30")
        print(f"Regular time: {np.mean(regular_time)}")
        print(f"Size 5 time: {np.mean(time_dict[5])}")
        print(f"Size 10 time: {np.mean(time_dict[10])}")
        print(f"Size 15 time: {np.mean(time_dict[15])}")
        print(f"Size 20 time: {np.mean(time_dict[20])}")
        print(f"Size 25 time: {np.mean(time_dict[25])}")
        print(f"Size 30 time: {np.mean(time_dict[30])}")



def small_system_testing(iters: int, eps=1e-10):
    # Write the header in the file "discrete_testing/small_system.csv"
    with open("dccp-syndisc/discrete_testing/small_system.csv", "w") as f:
        f.write(
            "n,gt_synergy,regular_time,one_iter_value,one_iter_time,five_iter_value,five_iter_time,ten_iter_value,ten_iter_time,dccp_value,dccp_time\n")

    for n in range(2, 6):
        print(f"---- Testing with {n} random variables, precision {eps} ----")
        regular_syn, regular_time = [], []
        one_iter_values, one_iter_times = [], []
        five_iter_values, five_iter_times = [], []
        ten_iter_values, ten_iter_times = [], []
        dccp_syn_values, dccp_times = [], []
        for _ in range(iters):
            if n < 5:
                dist = select_random_distribution(n)
            else:
                dist = pick_almost_uniform(n)
            print(f"Iteration {_}")
            gt_synergy, gt_time = compute_synergy(dist)
            if gt_synergy < 1e-3:
                continue
            iter_dict = dccp_synergy(dist, iterations=20, eps=1e-10, all_iterations=True)
            if iter_dict is None:
                continue
            regular_syn.append(gt_synergy)
            regular_time.append(gt_time)

            one_iter, one_iter_time = iter_dict[1]
            one_iter_values.append(one_iter)
            one_iter_times.append(one_iter_time)

            five_iter, five_iter_time = iter_dict[5]
            five_iter_values.append(five_iter)
            five_iter_times.append(five_iter_time)

            ten_iter, ten_iter_time = iter_dict[10]
            ten_iter_values.append(ten_iter)
            ten_iter_times.append(ten_iter_time)

            dccp_syn, dccp_syn_time = iter_dict[20]
            dccp_syn_values.append(dccp_syn)
            dccp_times.append(dccp_syn_time)
            print(gt_synergy, one_iter, five_iter, ten_iter, dccp_syn)

            with open("dccp-syndisc/discrete_testing/small_system.csv", "a") as f:
                f.write(
                    f"{n},{gt_synergy},{gt_time},{one_iter},{one_iter_time},{five_iter},{five_iter_time},{ten_iter},{ten_iter_time},{dccp_syn},{dccp_syn_time}\n")

        compute_syn_metrics(one_iter_values, regular_syn, "One iter")
        compute_syn_metrics(five_iter_values, regular_syn, "Five iter")
        compute_syn_metrics(ten_iter_values, regular_syn, "Ten iter")
        compute_syn_metrics(dccp_syn_values, regular_syn, "DCCP")
        print(f"Regular time: {np.mean(regular_time)}")
        print(f"One iter time: {np.mean(one_iter_times)}")
        print(f"Five iter time: {np.mean(five_iter_times)}")
        print(f"Ten iter time: {np.mean(ten_iter_times)}")
        print(f"DCCP time: {np.mean(dccp_times)}")


def select_random_distribution(n: int) -> dit.Distribution:
    new_dist = dit.random_distribution(n, ['01'])
    new_dist = dit.insert_rvf(new_dist, lambda x: '1' if all(map(bool, map(int, x))) else '0')
    return new_dist

def stress_testing(iters: int, eps=1e-10):
    # Write the header in the file "dccp-syndisc/discrete_testing/stress_testing.csv"
    with open("dccp-syndisc/discrete_testing/stress_testing.csv", "w") as f:
        f.write(
            "n,unique_comp,gt_synergy,regular_time,one_iter_value,one_iter_time,five_iter_value,five_iter_time,ten_iter_value,ten_iter_time,dccp_value,dccp_time\n")
    for n in range(12, 14):
        for unique_comp in [3]:
            print(f"---- Testing with {n} random variables {unique_comp} unique components, precision {eps} ----")
            regular_syn, regular_time = [], []
            one_iter_values, one_iter_times = [], []
            five_iter_values, five_iter_times = [], []
            ten_iter_values, ten_iter_times = [], []
            dccp_syn_values, dccp_times = [], []
            for _ in range(iters):
                big_dist, small_dist = select_big_small_random_distribution(n, unique_comp=unique_comp)
                print(f"Iteration {_}")
                gt_synergy, gt_time = compute_synergy(small_dist)
                if gt_synergy < 1e-3:
                    continue
                iter_dict = dccp_synergy(big_dist, iterations=20, eps=1e-10, all_iterations=True, len_py=10)
                if iter_dict is None:
                    continue
                regular_syn.append(gt_synergy)
                regular_time.append(gt_time)

                one_iter, one_iter_time = iter_dict[1]
                one_iter_values.append(one_iter)
                one_iter_times.append(one_iter_time)

                five_iter, five_iter_time = iter_dict[5]
                five_iter_values.append(five_iter)
                five_iter_times.append(five_iter_time)

                ten_iter, ten_iter_time = iter_dict[10]
                ten_iter_values.append(ten_iter)
                ten_iter_times.append(ten_iter_time)

                dccp_syn, dccp_syn_time = iter_dict[20]
                dccp_syn_values.append(dccp_syn)
                dccp_times.append(dccp_syn_time)
                print(gt_synergy, one_iter, five_iter, ten_iter, dccp_syn)

                with open("dccp-syndisc/discrete_testing/stress_testing.csv", "a") as f:
                    f.write(
                        f"{n},{unique_comp},{gt_synergy},{gt_time},{one_iter},{one_iter_time},{five_iter},{five_iter_time},{ten_iter},{ten_iter_time},{dccp_syn},{dccp_syn_time}\n")
            compute_syn_metrics(one_iter_values, regular_syn, "One iter")
            compute_syn_metrics(five_iter_values, regular_syn, "Five iter")
            compute_syn_metrics(ten_iter_values, regular_syn, "Ten iter")
            compute_syn_metrics(dccp_syn_values, regular_syn, "DCCP")
            
small_system_testing(10)