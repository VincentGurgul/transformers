import pandas as pd

import json

from wordwise.core import Extractor
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings("ignore")

ps = PorterStemmer()


def calc_ap(true_kw, pred_kw):
    res = 0
    corr = 0
    true_kw_stemmed = [ps.stem(kw) for kw in true_kw]
    pred_kw_stemmed = [ps.stem(kw) for kw in pred_kw]
    list_length = min(len(true_kw_stemmed), len(pred_kw_stemmed))
    for i in range(0, list_length):
        if pred_kw[i] in true_kw:
            corr += 1
            res += corr / (i + 1)
    if corr == 0:
        return 0
    else:
        return res / corr


def calc_map(abstracts_dict, true_kw_dict, pred_kw_dict):
    avg_prec_list = []
    with tqdm(total=len(abstracts_dict)) as pbar:
        for key in abstracts_dict.keys():
            true_kw = true_kw_dict[key]
            pred_kw = pred_kw_dict[key]
            avg_prec_list.append(calc_ap(true_kw, pred_kw))
            pbar.update(1)
    return sum(avg_prec_list) / len(avg_prec_list)


def extract_keywords(model):
    extractor = Extractor(bert_model=model)
    suggested_keywords_dict = {}
    with tqdm(total=len(abstracts_dict)) as pbar:
        for key in abstracts_dict.keys():
            abstract = abstracts_dict[key]
            top_k = len(keywords_dict[key])
            try:
                keyword_suggestions = extractor.generate(abstract, top_k=top_k)
            except RuntimeError:
                b = abstract[0:1200]
                keyword_suggestions = extractor.generate(b, top_k=top_k)
            except IndexError:
                b = abstract[0:1000]
                keyword_suggestions = extractor.generate(b, top_k=top_k)

            suggested_keywords_dict.update({key: keyword_suggestions})
            pbar.update(1)

    bert_map = calc_map(abstracts_dict, keywords_dict, suggested_keywords_dict)
    return bert_map


if __name__ == '__main__':

    n_gram_range = (1, 2)
    stop_words = "english"

    count = 0
    count_dict = {}
    candidates_dict = {}
    abstracts_dict = {}
    keywords_dict = {}
    candidates_list = []
    abstracts_list = []
    keywords_list = []
    titles_list = []

    for line in open("data/kp20k_testing.json", 'r'):
        line = json.loads(line)

        title = line['title']
        abstract = line['abstract']
        keywords = line['keyword']

        # print('For title:', title)
        countVect = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([abstract])

        candidates = countVect.get_feature_names_out()
        # print('Candidates: ', candidates)
        # print()

        abstracts_dict.update({title: abstract})
        keywords_dict.update({title: keywords.split(';')})
        candidates_dict.update({title: candidates})

        abstracts_list.append(abstract)
        keywords_list.append(keywords.split(';'))
        candidates_list.append(candidates)
        titles_list.append(title)

        count += 1
        #if count == 30:
        #    break

    df = pd.DataFrame.from_dict({'abstract': abstracts_list, 'title': titles_list,
                                 'candidates': candidates_list, 'keywords': keywords_list})

    bert_map = extract_keywords("bert-base-uncased")
    print('MAP of BERT is: ', bert_map)

    distilbert_map = extract_keywords("distilbert-base-uncased")
    print('MAP of DistilBERT is: ', distilbert_map)

    roberta_map = extract_keywords("roberta-base")
    print('MAP of RoBERTa is: ', roberta_map)

    distilroberta_map = extract_keywords("distilroberta-base")
    print('MAP of DistilRoBERTa is: ', distilroberta_map)

    print('Final results')
    print('MAP of BERT is: ', bert_map)
    print('MAP of DistilBERT is: ', distilbert_map)
    print('MAP of RoBERTa is: ', roberta_map)
    print('MAP of DistilRoBERTa is: ', distilroberta_map)
