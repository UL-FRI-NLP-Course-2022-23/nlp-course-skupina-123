import os
import json
import numpy as np
import pandas as pd
import ners.ner_stanza as ns

from collections import Counter
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer

def sentiment_mapping(sentiment_value):
    if sentiment_value == 0:
        return -1
    elif sentiment_value == 1:
        return 0
    else:
        return 1

def get_person_df(persons):
    rows = []
    for person in persons:
        row = {
            "text": person.text,
            "type": person.type,
            "start_char": person.start_char,
            "end_char": person.end_char,
            "sentence_sentiment": sentiment_mapping(person._sent.sentiment)
        }
        rows.append(row)

    return pd.DataFrame(rows)

def read_story_from_file(file_name):
    with open(f'../data/stories/{file_name}.txt', encoding="utf8") as f:
        return f.read()

def get_person_entities(doc):
    return list(filter(lambda e: e.type == "PERSON", doc.entities))

def perform_sentiment(df):
    sentiment = df.copy()
    sentiment = sentiment.drop(columns=["type", "start_char", "end_char"])
    sentiment = sentiment.groupby("text").sum().reset_index()

    return sentiment.sort_values("sentence_sentiment")

def calculate_matrix(name_list, sentences, cor_res_sentences):
    '''
    Function to calculate the co-occurrence matrix and sentiment matrix among all the top characters
    :param name_list: the list of names of the top characters in the novel.
    :param sentences: the list of sentences in the novel.
    :param align_rate: the sentiment alignment rate to align the sentiment score between characters due to the writing style of
    the author. Every co-occurrence will lead to an increase or decrease of one unit of align_rate.
    :return: the co-occurrence matrix and sentiment matrix.
    '''

    # calculate a sentiment score for each sentence in the novel
    sentiment_score = []
    # for sentence in sentences:
    doc = stanza_pretokenized.tagger(sentences)
    for doc_sentence in doc.sentences:
        sentiment_score.append(float(doc_sentence.sentiment) - 1)

    l = len(np.nonzero(sentiment_score)[0])
    if l == 0:
        l = 1
    
    align_rate = np.sum(sentiment_score) / l * -2

    # replace name occurrences with names that can be vectorized
    for i in range(len(cor_res_sentences)):
        cor_res_sentences[i] = cor_res_sentences[i].lower()

        for name in name_list:
            tmp = name.split(" ")
            tmp = "_".join(tmp)
            cor_res_sentences[i] = cor_res_sentences[i].replace(name, tmp)

    for i in range(len(name_list)):
        tmp = name_list[i].split(" ")
        name_list[i] = "_".join(tmp)

    name_vec = CountVectorizer(vocabulary=name_list, binary=True)

    # calculate occurrence matrix and sentiment matrix among the top characters
    if (len(name_list) == 0):
        return np.array([]), np.array([]), np.array([]), np.array([])
    else:
        occurrence_each_sentence = name_vec.fit_transform(cor_res_sentences).toarray()

    co_occurrence_matrix = np.dot(occurrence_each_sentence.T, occurrence_each_sentence)
    sentiment_matrix = np.dot(occurrence_each_sentence.T, (occurrence_each_sentence.T * sentiment_score).T)
    sentiment_matrix += align_rate * co_occurrence_matrix
    co_occurrence_matrix = np.tril(co_occurrence_matrix)
    sentiment_matrix = np.tril(sentiment_matrix)

    # diagonals of the matrices are set to be 0 (co-occurrence of name itself is meaningless)
    shape = co_occurrence_matrix.shape[0]
    co_occurrence_matrix[[range(shape)], [range(shape)]] = 0
    sentiment_matrix[[range(shape)], [range(shape)]] = 0

    return co_occurrence_matrix, sentiment_matrix

def generate_json(f_name, name_list, sentiment_matrix):
    data = {}
    for i, name in enumerate(name_list):
        data[name] = {}
        for j, name2 in enumerate(name_list):
            divisor = 0
            if len(name_list) > 1:
                divisor = np.abs(sentiment_matrix).max()
            if divisor == 0:
                divisor = 1
            data[name][name2] = sentiment_matrix[i][j] / divisor 

    with open(f"../results/{f_name}.json", "w+", encoding="utf8") as f:
        f.write(json.dumps({"characters": name_list, "sentiments" : data}, indent=2))

if __name__ == "__main__":
    stanza = ns.StanzaNer()
    stanza_pretokenized = ns.StanzaNer(tokenize_pretokenized=True)
    for fn in os.listdir("../data/stories"):
        if not fn.startswith("0") and not fn.startswith("split"):
            f = fn.split(".")[0]
            
            print(f"Running for {f}")
            story = read_story_from_file(file_name=f)
            doc, cr_story = stanza.ner_stanza_whole_doc(story, use_cr=True)

            sentences = sent_tokenize(story)
            cr_sentences = sent_tokenize(cr_story)
            
            person_entities = [x.text.lower().replace("'s", "") for x in get_person_entities(doc)]
            person_entities = [x.split(' ') for x in person_entities]
            person_entities = [[word for word in x if not word in ['the', 'an', 'a', 'and']] for x in person_entities]
            person_entities = [' '.join(x) for x in person_entities]
            counts = Counter(person_entities)
            person_entities = [x for x in counts]
            counts = [counts[x] for x in counts]
            
            cooccurrence_matrix, sentiment_matrix = calculate_matrix(person_entities, sentences, cr_sentences)
            person_entities = [name.replace("_", " ") for name in person_entities]

            generate_json(f, person_entities, sentiment_matrix)