import csv
import numpy as np
from os import listdir
from os.path import isfile, join
import multiprocessing as mp
from itertools import combinations


def read_file(filename):
    """Reads a CSV file and returns topic-wise document entries."""
    result = {}
    with open(filename, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            topic = int(line[0])
            entry = {'doc': line[1], 'score': float(line[2]), 'rel': int(line[3])}
            result.setdefault(topic, []).append(entry)
    return result


def get_common_items(ranking_1, ranking_2):
    """Filters rankings to retain only documents common to both."""
    filtered_1, filtered_2 = {}, {}
    for topic in ranking_1:
        if topic in ranking_2:
            docs1 = {entry['doc'] for entry in ranking_1[topic]}
            docs2 = {entry['doc'] for entry in ranking_2[topic]}
            common_docs = docs1 & docs2
            filtered_1[topic] = [entry for entry in ranking_1[topic] if entry['doc'] in common_docs]
            filtered_2[topic] = [entry for entry in ranking_2[topic] if entry['doc'] in common_docs]
    return filtered_1, filtered_2


def lexicographic_ranking(primary, secondary):
    """Returns lexicographic ranking based on primary and secondary scores."""
    idxs = list(range(len(primary)))
    idxs.sort(key=lambda i: (-primary[i], -secondary[i]))
    ranks = [0] * len(primary)
    for rank, idx in enumerate(idxs):
        ranks[idx] = rank
    return ranks


# Weight Functions
def tau_ap_weight(i, j, x, y):
    return 1.0 / (max(y[j], y[i]) - 1)


def tau_h_weight(i, j, xy_ranks, yx_ranks):
    return 0.5 * (
        1.0 / (xy_ranks[j] + 1) + 1.0 / (xy_ranks[i] + 1) +
        1.0 / (yx_ranks[j] + 1) + 1.0 / (yx_ranks[i] + 1)
    )


# Concordance Functions
def sign_concordance(rel_xi, rel_xj, rel_yi, rel_yj, max_rel):
    dx = np.sign(rel_xi - rel_xj)
    dy = np.sign(rel_yi - rel_yj)
    return 1 if dx == dy else -1


def additive_concordance_relevance(rel_xi, rel_xj, rel_yi, rel_yj, max_rel):
    dx = np.sign(rel_xi - rel_xj)
    dy = np.sign(rel_yi - rel_yj)
    concordance = 1 if dx == dy else -1

    if rel_xi == rel_yi and rel_xj == rel_yj:
        return 1
    coeff = 1 - abs((rel_xi + rel_xj) - (rel_yi + rel_yj)) / (2 * max_rel)
    return 0.9 * concordance * coeff


def additive_concordance(rel_i, rel_j, max_rel):
    return (rel_i + rel_j) / (2 * max_rel)


def multiplicative_concordance(rel_i, rel_j, max_rel):
    return (rel_i * rel_j) / (max_rel ** 2)


# Metric Computation
def weighted_tau(x, y, rel_x, rel_y, max_rel):
    n = len(rel_x)
    if n < 2:
        print("Returning 0!")
        return (0.0,) * 15

    xy_ranks = lexicographic_ranking(x, y)
    yx_ranks = lexicographic_ranking(y, x)

    # Numerator and denominator pairs for different tau versions
    metrics = np.zeros((15, 2))

    for i in range(n):
        for j in range(i + 1, n):
            concordance = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            sign_conc = sign_concordance(rel_x[i], rel_x[j], rel_y[i], rel_y[j], max_rel)
            additive_conc_rel = additive_concordance_relevance(rel_x[i], rel_x[j], rel_y[i], rel_y[j], max_rel)
            additive_conc = additive_concordance(rel_x[i], rel_x[j], max_rel)
            multiplicative_conc = multiplicative_concordance(rel_x[i], rel_x[j], max_rel)

            tauAP_weight = tau_ap_weight(i, j, x, y)
            tauH_weight = tau_h_weight(i, j, xy_ranks, yx_ranks)
            dummy_rank_i = [i + 1] * n
            dummy_rank_j = [j + 1] * n
            tauAP_rel_weight = tau_ap_weight(i, j, dummy_rank_i, dummy_rank_j)
            tauH_rel_weight = tau_h_weight(i, j, xy_ranks, xy_ranks)

            entries = [
                (concordance, 1.0),                              # tau
                (sign_conc, 1.0),                                # tau_s
                (additive_conc_rel, 1.0),                        # tau_ar
                (concordance * additive_conc, additive_conc),   # tau_a
                (concordance * multiplicative_conc, multiplicative_conc),  # tau_m

                (concordance * tauAP_weight, tauAP_weight),     # tauAP
                (sign_conc * tauAP_rel_weight, tauAP_rel_weight),  # tauAP_s
                (additive_conc_rel * tauAP_rel_weight, tauAP_rel_weight),  # tauAP_ar
                (concordance * tauAP_weight * additive_conc, tauAP_weight * additive_conc),  # tauAP_a
                (concordance * tauAP_weight * multiplicative_conc, tauAP_weight * multiplicative_conc),  # tauAP_m

                (concordance * tauH_weight, tauH_weight),       # tauH
                (sign_conc * tauH_rel_weight, tauH_rel_weight),  # tauH_s
                (additive_conc_rel * tauH_rel_weight, tauH_rel_weight),  # tauH_ar
                (concordance * tauH_weight * additive_conc, tauH_weight * additive_conc),  # tauH_a
                (concordance * tauH_weight * multiplicative_conc, tauH_weight * multiplicative_conc),  # tauH_m
            ]

            for idx, (num, denom) in enumerate(entries):
                metrics[idx][0] += num
                metrics[idx][1] += denom

    return tuple(num / denom for num, denom in metrics)


# Processing
def process_file_pair(args):
    folder, file1, file2 = args
    path = lambda f: join("untied_data", folder, f)
    res_1 = read_file(path(file1))
    res_2 = read_file(path(file2))
    filtered_1, filtered_2 = get_common_items(res_1, res_2)

    max_rel = 3 if folder in {'2010', '2011'} or 'simulated_data' in folder else 4
    results = []

    for topic in list(filtered_1.keys()):
        print(f"Status update. Topic: {topic}, file 1: {file1}, file 2: {file2}")
        items_1 = [item["doc"] for item in filtered_1[topic]]
        items_2 = [item["doc"] for item in filtered_2[topic]]

        rel_1 = [item["rel"] for item in filtered_1[topic]]
        rel_2 = [item["rel"] for item in filtered_2[topic]]

        index_map_1 = {doc: i for i, doc in enumerate(items_1, start=1)}
        index_map_2 = {doc: i for i, doc in enumerate(items_2, start=1)}
        sorted_X = [index_map_1[doc] for doc in items_1]
        sorted_Y = [index_map_2[doc] for doc in items_1]

        results.append(weighted_tau(sorted_X, sorted_Y, rel_1, rel_2, max_rel))
    return results


def get_folder_file_pairs(folder):
    path = f"untied_data/{folder}/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return [(folder, f1, f2) for f1, f2 in combinations(files, 2)]


# Main Execution
if __name__ == "__main__":
    # folders = ['2010', '2011', '2012', '2013', '2014']
    folders = ['simulated_data']
    all_jobs = []
    for folder in folders:
        all_jobs.extend(get_folder_file_pairs(folder))

    with mp.Pool() as pool:
        results = pool.map(process_file_pair, all_jobs)

    folder_outputs = {}
    for (folder, file1, file2), pair_result in zip(all_jobs, results):
        folder_outputs.setdefault(folder, []).extend(pair_result)

    for folder, metrics in folder_outputs.items():
        output_path = join("output", f"{folder}.csv")
        with open(output_path, 'w') as f:
            header = 'tau,tau_s,tau_ar,tau_a,tau_m,tauAP,tauAP_s,tauAP_ar,tauAP_a,tauAP_m,tauH,tauH_s,tauH_ar,tauH_a,tauH_m\n'
            f.write(header)
            for row in metrics:
                f.write(','.join(map(str, row)) + '\n')
