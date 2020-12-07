from readCorpus import getTermList_TermFrequency
from readCorpus import getFilterTermList_CoappearMatrix
from readCorpus import load_object
from readCorpus import save_object

from math import sqrt
from math import log
from math import log2
import os

def get_t_statistic_topK(term_frequency_dict_filtered, term_coappear_matrix_dict, topK=100):
    if not os.path.exists('./results'):
        os.makedirs('./results')

    path_output = './results/t_statistic_topK.pkl'
    if os.path.exists(path_output):
        t_statistic_topK = load_object(path_output)
        return t_statistic_topK
    else:
        t_statistic_dict = {}

        for term_left in term_coappear_matrix_dict:
            for term_right in term_coappear_matrix_dict[term_left]:
                x_bar = term_coappear_matrix_dict[term_left][term_right] / cnt_pairs
                mu = term_frequency_dict_filtered[term_left] * term_frequency_dict_filtered[term_right] / cnt_pairs / cnt_pairs
                S_2 = x_bar * (1 - x_bar)
                t = (x_bar - mu) / sqrt(S_2 / cnt_pairs)

                pair = (term_left, term_right)
                t_statistic_dict[pair] = t

        t_statistic_topK = sorted(t_statistic_dict.items(), key=lambda d: d[1], reverse=True)[:topK]

        save_object(t_statistic_topK, path_output)
        return t_statistic_topK


def get_chi_square_topK(term_frequency_dict_filtered, term_coappear_matrix_dict, topK=100):
    if not os.path.exists('./results'):
        os.makedirs('./results')

    path_output = './results/chi_square_topK.pkl'
    if os.path.exists(path_output):
        chi_square_topK = load_object(path_output)
        return chi_square_topK
    else:
        chi_square_dict = {}
        for term_left in term_coappear_matrix_dict:
            for term_right in term_coappear_matrix_dict[term_left]:
                O_11 = term_coappear_matrix_dict[term_left][term_right]
                O_12 = term_frequency_dict_filtered[term_left] - O_11
                O_21 = term_frequency_dict_filtered[term_right] - O_11
                O_22 = cnt_pairs - O_11 - O_12 - O_21

                num = cnt_pairs * ((O_11 * O_22 - O_12 * O_21) ** 2)
                den = (O_11 + O_12) * (O_11 + O_21) * (O_12 + O_22) * (O_21 + O_22)
                chi_square = num / den

                pair = (term_left, term_right)
                chi_square_dict[pair] = chi_square

        chi_square_topK = sorted(chi_square_dict.items(), key=lambda d: d[1], reverse=True)[:topK]

        save_object(chi_square_topK, path_output)
        return chi_square_topK


def get_LLR_statistic_topK(term_frequency_dict_filtered, term_coappear_matrix_dict, topK=100):
    if not os.path.exists('./results'):
        os.makedirs('./results')

    path_output = './results/LLR_statistic_topK.pkl'
    if os.path.exists(path_output):
        LLR_statistic_topK = load_object(path_output)
        return LLR_statistic_topK
    else:
        def L_function(k, n, x):
            if (x == 0.):
                if (k == 0):
                    return 0.
                else:
                    return -1e1000
            elif (x == 1.):
                if (k == n):
                    return 0.
                else:
                    return -1e1000
            else:
                return k * log(x) + (n - k) * log(1 - x)


        LLR_statistic_dict = {}
        N = cnt_pairs
        for term_left in term_coappear_matrix_dict:
            for term_right in term_coappear_matrix_dict[term_left]:
                c_12 = term_coappear_matrix_dict[term_left][term_right]
                c_1 = term_frequency_dict_filtered[term_left]
                c_2 = term_frequency_dict_filtered[term_right]
                p = c_2 / N
                p_1 = c_12 / c_1
                p_2 = (c_2 - c_12) / (N - c_1)

                logL_num_left = L_function(c_12, c_1, p)
                logL_num_right = L_function(c_2 - c_12, N - c_1, p)
                logL_den_left = L_function(c_12, c_1, p_1)
                logL_den_right = L_function(c_2 - c_12, N - c_1, p_2)

                log_lambda = logL_num_left + logL_num_right - logL_den_left - logL_den_right

                pair = (term_left, term_right)
                LLR_statistic_dict[pair] = -2 * log_lambda

        LLR_statistic_topK = sorted(LLR_statistic_dict.items(), key=lambda d: d[1], reverse=True)[:topK]
        save_object(LLR_statistic_topK, path_output)
        return LLR_statistic_topK


def get_MI_statistic_topK(term_frequency_dict_filtered, term_coappear_matrix_dict, topK=100):
    if not os.path.exists('./results'):
        os.makedirs('./results')

    path_output = './results/MI_statistic_topK.pkl'
    if os.path.exists(path_output):
        MI_statistic_topK = load_object(path_output)
        return MI_statistic_topK
    else:
        MI_statistic_dict = {}
        for term_left in term_coappear_matrix_dict:
            for term_right in term_coappear_matrix_dict[term_left]:
                p_xy = term_coappear_matrix_dict[term_left][term_right] / cnt_pairs
                p_x = term_frequency_dict_filtered[term_left] / cnt_pairs
                p_y = term_frequency_dict_filtered[term_right] / cnt_pairs

                MI = log2(p_xy) - log2(p_x) - log2(p_y)

                pair = (term_left, term_right)
                MI_statistic_dict[pair] = MI

        MI_statistic_topK = sorted(MI_statistic_dict.items(), key=lambda d: d[1], reverse=True)[:topK]

        save_object(MI_statistic_topK, path_output)
        return MI_statistic_topK


if __name__ == '__main__':
    term_frequency_dict = getTermList_TermFrequency()
    term_coappear_matrix_dict, cnt_pairs = getFilterTermList_CoappearMatrix(term_frequency_dict)

    topK = 10

    t_statistic_topK = get_t_statistic_topK(term_frequency_dict, term_coappear_matrix_dict,
                                            topK=topK)

    print(t_statistic_topK)



    chi_square_topK = get_chi_square_topK(term_frequency_dict, term_coappear_matrix_dict,
                                          topK=topK)

    print(chi_square_topK)


    LLR_statistic_topK = get_LLR_statistic_topK(term_frequency_dict, term_coappear_matrix_dict,
                                          topK=topK)

    print(LLR_statistic_topK)


    MI_statistic_topK = get_MI_statistic_topK(term_frequency_dict, term_coappear_matrix_dict,
                                          topK=topK)

    print(MI_statistic_topK)







