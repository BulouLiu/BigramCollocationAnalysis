from readCorpus import getTermList_TermFrequency
from readCorpus import getFilterTermList_CoappearMatrix
from readCorpus import load_object
from readCorpus import save_object
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', required=True)
    args = parser.parse_args()
    path_t = './results/'+args.function+'_topK.pkl'

    term_frequency_dict = getTermList_TermFrequency()
    term_coappear_matrix_dict, cnt_pairs = getFilterTermList_CoappearMatrix(term_frequency_dict)

    t_statistic_topK = load_object(path_t)
    print("rank\tleft\tright\tscore\tcnt_left\tcnt_right\tcnt_bigram")
    all_list = []
    for i in range(10):
        pair, score = t_statistic_topK[i]
        all_list.append([i+1, pair[0], pair[1], score, term_frequency_dict[pair[0]],
                                          term_frequency_dict[pair[1]], term_coappear_matrix_dict[pair[0]][pair[1]]])
        print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(i+1, pair[0], pair[1], score, term_frequency_dict[pair[0]],
                                          term_frequency_dict[pair[1]], term_coappear_matrix_dict[pair[0]][pair[1]]))
    df = pd.DataFrame(all_list, columns=['Rank', 'Left_term', 'Right_term', 't_statistic', 'Cnt_left', 'Cnt_right', 'Cnt_bigram'])
    df.to_excel('./results/' + args.function+".xlsx", index=False)