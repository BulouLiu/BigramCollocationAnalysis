import os
import pickle as pkl
import json

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

def getTermList_TermFrequency():
    if not os.path.exists('./output'):
        os.makedirs('./output')

    path_pkl = './output/term_frequency_dict_daily.pkl'
    if os.path.exists(path_pkl):
        term_frequency_dict = load_object(path_pkl)
        return term_frequency_dict
    else:
        term_frequency_dict = {}

        path_corpus = 'corpus.json'

        with open(path_corpus, 'r') as f:
            load_dict = json.load(f)
            for sentence in load_dict:
                for i in range(len(sentence)):
                    if sentence[i]['upos'] != "PUNCT":
                        term = sentence[i]['form'] + '/' + sentence[i]['upos']
                        if term in term_frequency_dict:
                            term_frequency_dict[term] += 1.
                        else:
                            term_frequency_dict[term] = 1.

        save_object(term_frequency_dict, path_pkl)

        return term_frequency_dict



def getFilterTermList_CoappearMatrix(term_frequency_dict):
    if not os.path.exists('./output'):
        os.makedirs('./output')

    path_coappear_matrix = './output/coappear_matrix.pkl'
    path_scnt = './output/cnt_pairs.pkl'

    if not os.path.exists(path_coappear_matrix):
        term_coappear_matrix_dict = {}

        for term in term_frequency_dict:
            term_coappear_matrix_dict[term] = {}

        cnt_pairs = 0.

        path_corpus = 'corpus.json'
        with open(path_corpus, 'r') as f:
            load_dict = json.load(f)
            for sentence in load_dict:
                for i in range(len(sentence) - 1):
                    if sentence[i]['upos'] != "PUNCT" and sentence[i+1]['upos'] != "PUNCT":
                        cnt_pairs += 1.
                        item_left = sentence[i]['form'] + '/' + sentence[i]['upos']
                        item_right = sentence[i+1]['form'] + '/' + sentence[i+1]['upos']
                        if item_right in term_coappear_matrix_dict[item_left]:
                            term_coappear_matrix_dict[item_left][item_right] += 1.
                        else:
                            term_coappear_matrix_dict[item_left][item_right] = 1.

        save_object(term_coappear_matrix_dict, path_coappear_matrix)
        save_object(cnt_pairs, path_scnt)
        return term_coappear_matrix_dict, cnt_pairs
    else:
        cnt_pairs = load_object(path_scnt)
        term_coappear_matrix_dict = load_object(path_coappear_matrix)
        return term_coappear_matrix_dict, cnt_pairs





if __name__ == '__main__':
    term_frequency_dict = getTermList_TermFrequency()
    term_coappear_matrix_dict, cnt_pairs = getFilterTermList_CoappearMatrix(term_frequency_dict)

    for term in term_frequency_dict:
        print(term, term_coappear_matrix_dict[term])

    print('sentence number = {}'.format(cnt_pairs))

