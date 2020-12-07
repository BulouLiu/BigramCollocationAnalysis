from readCorpus import getTermList_TermFrequency
from readCorpus import getFilterTermList_CoappearMatrix
from readCorpus import load_object
from readCorpus import save_object
import argparse
from io import open
from conllu import parse_incr
import pandas as pd

def cnt_relation(relation_pair, term1, term2):
    if term1 in relation_pair.keys():
        if term2 in relation_pair[term1].keys():
            return relation_pair[term1][term2]
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', required=True)
    args = parser.parse_args()
    path_t = './results/'+args.function+'_topK.pkl'

    term_frequency_dict = getTermList_TermFrequency()
    term_coappear_matrix_dict, cnt_pairs = getFilterTermList_CoappearMatrix(term_frequency_dict)

    t_statistic_topK = load_object(path_t)

    relation_pair = {}
    data_file = open("ud.conllu", "r", encoding="utf-8")
    for tokenlist in parse_incr(data_file):
        id2form = {}
        for i in range(len(tokenlist)):
            id2form[tokenlist[i]['id']] = tokenlist[i]['form'] + '/' + tokenlist[i]['upos']
        for i in range(len(tokenlist)):
            if tokenlist[i]['head'] != 0:
                if id2form[tokenlist[i]['id']] not in relation_pair.keys():
                    relation_pair[id2form[tokenlist[i]['id']]] = {}
                    relation_pair[id2form[tokenlist[i]['id']]][id2form[tokenlist[i]['head']]] = 1
                elif id2form[tokenlist[i]['head']] not in relation_pair[id2form[tokenlist[i]['id']]].keys():
                    relation_pair[id2form[tokenlist[i]['id']]][id2form[tokenlist[i]['head']]] = 1
                else:
                    relation_pair[id2form[tokenlist[i]['id']]][id2form[tokenlist[i]['head']]] += 1
    print("rank\tleft\tright\tleft_right\tright_left")
    all_list = []
    for i in range(10):
        pair, score = t_statistic_topK[i]
        all_list.append([i+1, pair[0], pair[1], cnt_relation(relation_pair, pair[0], pair[1]), cnt_relation(relation_pair, pair[1], pair[0])])
        print('{}\t{}\t{}\t{}\t{}'.format(i+1, pair[0], pair[1], cnt_relation(relation_pair, pair[0], pair[1]), cnt_relation(relation_pair, pair[1], pair[0])))
    df = pd.DataFrame(all_list,
                      columns=['Rank', 'Left_term', 'Right_term', 'Cnt_left_right', 'Cnt_right_left'])

    df.to_excel('./results/' + args.function + "_relation.xlsx", index=False)
