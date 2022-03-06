#!/usr/bin/python3.7
import scipy
from sklearn.metrics import accuracy_score


def test_5x2_ftest(real_instances_method_1: list,
                   predicted_instances_method_1: list,
                   real_instances_method_2: list,
                   predicted_instances_method_2: list):
    variances = []
    differences = []
    real_differences = []

    for repetition in range(5):
        acc_m1_1 = accuracy_score(real_instances_method_1[repetition*2],
                                  predicted_instances_method_1[repetition*2])

        acc_m2_1 = accuracy_score(real_instances_method_2[repetition*2],
                                  predicted_instances_method_2[repetition*2])

        acc_m1_2 = accuracy_score(real_instances_method_1[repetition*2 + 1],
                                  predicted_instances_method_1[repetition*2 + 1])

        acc_m2_2 = accuracy_score(real_instances_method_2[repetition*2 + 1],
                                  predicted_instances_method_2[repetition*2 + 1])

        score_diff_1 = acc_m1_1 - acc_m2_1
        score_diff_2 = acc_m1_2 - acc_m2_2

        score_mean = (score_diff_1 + score_diff_2) / 2.
        score_var = ((score_diff_1 - score_mean)**2 +
                     (score_diff_2 - score_mean)**2)

        real_differences.extend([score_diff_1, score_diff_2])
        differences.extend([score_diff_1**2, score_diff_2**2])
        variances.append(score_var)

    numerator = sum(differences)
    denominator = 2*(sum(variances))
    f_stat = numerator / denominator
    p_value = scipy.stats.f.sf(f_stat, 10, 5)
    print("Score diffs=", real_differences)
    print("F-stat=", float(f_stat), "p-value=", float(p_value))
