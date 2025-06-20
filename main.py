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
def sign_concordance_relevance(rel_xi, rel_xj, rel_yi, rel_yj, max_rel, a_f = False):
    if a_f and max_rel == 3:
        a = -0.7
    elif a_f and max_rel == 4:
        a = -0.58
    else:
        a = -1 / (2 * max_rel)

    dx = np.sign(rel_xi - rel_xj)
    dy = np.sign(rel_yi - rel_yj)

    if dx == dy:
        return 1

    if dx == 0 or dy == 0:
        return a

    return -1


def additive_concordance_relevance(rel_xi, rel_xj, rel_yi, rel_yj, rel_xni, rel_xnj, rel_yni, rel_ynj, max_rel):
    if rel_xi == rel_yi == rel_yni and rel_xj == rel_yj == rel_ynj and (rel_xni != rel_yni or rel_xnj != rel_ynj):
        return 0

    if (rel_xi != rel_yi or rel_xj != rel_yj) and ((rel_xi == rel_yni and rel_xj == rel_ynj) or (rel_yi == rel_xni and rel_yj == rel_xnj)):
        concordance = sign_concordance_relevance(rel_xi, rel_xj, rel_yni, rel_ynj, max_rel, a_f=True)
        coeff = 1 - abs((rel_xi + rel_xj) - (rel_yni + rel_ynj)) / (2 * max_rel)
    else:
        concordance = sign_concordance_relevance(rel_xi, rel_xj, rel_yi, rel_yj, max_rel, a_f=True)
        coeff = 1 - abs((rel_xi + rel_xj) - (rel_yi + rel_yj)) / (2 * max_rel)

    return concordance * coeff


def distance_concordance(rel_i, rel_j, max_rel):
    if rel_i == 0 and rel_j == 0:
        return 0
    return abs(rel_i - rel_j) / max(rel_i, rel_j)


# Metric Computation
def weighted_tau(x, y, rel_x, rel_y, max_rel):
    n = len(rel_x)
    if n < 2:
        print("Returning 0!")
        return (0.0,) * 12

    xy_ranks = lexicographic_ranking(x, y)
    yx_ranks = lexicographic_ranking(y, x)

    # Numerator and denominator pairs for different tau versions
    metrics = np.zeros((12, 2))

    for i in range(n):
        for j in range(i + 1, n):
            concordance = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            sign_conc_rel = sign_concordance_relevance(rel_x[i], rel_x[j], rel_y[i], rel_y[j], max_rel)
            additive_conc_rel = additive_concordance_relevance(rel_x[i], rel_x[j], rel_y[i], rel_y[j], rel_x[n-1-i], rel_x[n-1-j], rel_y[n-1-i], rel_y[n-1-j], max_rel)
            distance_conc = distance_concordance(rel_x[i], rel_x[j], max_rel)

            if len(set(rel_x)) == 1:
                distance_conc = 1

            tauAP_weight = tau_ap_weight(i, j, x, y)
            tauH_weight = tau_h_weight(i, j, xy_ranks, yx_ranks)
            dummy_rank_i = [i + 1] * n
            dummy_rank_j = [j + 1] * n
            tauAP_rel_weight = tau_ap_weight(i, j, dummy_rank_i, dummy_rank_j)
            tauH_rel_weight = tau_h_weight(i, j, xy_ranks, xy_ranks)

            entries = [
                (concordance, 1.0),                              # tau
                (sign_conc_rel, 1.0),                                # tau_sc
                (additive_conc_rel, 1.0),                        # tau_ac
                (concordance * distance_conc, distance_conc),   # tau_dw

                (concordance * tauAP_weight, tauAP_weight),     # tauAP
                (sign_conc_rel * tauAP_rel_weight, tauAP_rel_weight),  # tauAP_ss
                (additive_conc_rel * tauAP_rel_weight, tauAP_rel_weight),  # tauAP_ac
                (concordance * tauAP_weight * distance_conc, tauAP_weight * distance_conc),  # tauAP_dw

                (concordance * tauH_weight, tauH_weight),       # tauH
                (sign_conc_rel * tauH_rel_weight, tauH_rel_weight),  # tauH_sc
                (additive_conc_rel * tauH_rel_weight, tauH_rel_weight),  # tauH_ac
                (concordance * tauH_weight * distance_conc, tauH_weight * distance_conc),  # tauH_dw
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

    max_rel = 3 if folder in {'2010', '2011'} or 'simulated_data' == folder else 4

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
            header = 'tau,tau_sc,tau_ac,tau_dw,tauAP,tauAP_sc,tauAP_ac,tauAP_dw,tauH,tauH_sc,tauH_ac,tauH_dw\n'
            f.write(header)
            for row in metrics:
                f.write(','.join(map(str, row)) + '\n')
