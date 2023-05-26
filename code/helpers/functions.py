import copy
import json
from Levenshtein import jaro
import os
import re


def read_story_from_file(file_name):
    with open(f"../data/stories/{file_name}.txt", encoding="utf8") as f:
        return f.read()


def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    total = sum(numbers)
    average = total / len(numbers)
    return average


def get_characters_from_file(file_name, gt=True):
    if gt:
        with open(f"../data/annotations/{file_name}.json", encoding="utf8") as f:
            return json.load(f)["characters"]
    else:
        with open(f"../results/{file_name}.json", encoding="utf8") as f:
            return json.load(f)["characters"]


def get_sentiment_from_file(file_name, gt=True):
    if gt:
        with open(f"../data/annotations/{file_name}.json", encoding="utf8") as f:
            return json.load(f)["sentiments"]
    else:
        with open(f"../results/{file_name}.json", encoding="utf8") as f:
            return json.load(f)["sentiments"]


def compare_lists(list1, list2):
    if len(list1) != len(list2):
        return False
    return set(map(str.lower, list1)) == set(map(str.lower, list2))


def get_file_names():
    path = "../data/stories/"
    file_names = os.listdir(path)
    file_names = sorted([re.sub(r"\.txt$", "", file_name) for file_name in file_names])
    return file_names


def string_similarity(s1, s2, threshold=0.5):
    return jaro(s1.lower(), s2.lower()) >= threshold


def true_positive(list1, list2):
    tp = 0
    l2 = copy.deepcopy(list2)
    for x in list1:
        for y in l2:
            if string_similarity(x, y):
                tp += 1
                l2.remove(y)
                break
    return tp


def precision_score(list1, list2):
    if len(list1) + len(list2) == 0:
        return 1
    if len(list2) == 0:
        return 0
    return true_positive(list1, list2) / len(list2)


def recall_score(list1, list2):
    if len(list1) + len(list2) == 0:
        return 1
    if len(list1) == 0:
        return 0
    return true_positive(list1, list2) / len(list1)


def f1_score(list1, list2):
    per = precision_score(list1, list2)
    recall = recall_score(list1, list2)
    try:
        return 2 * per * recall / (per + recall)
    except ZeroDivisionError:
        return 0


def overall_scores(flair=None, stanza=None, spacy=None, use_cr=False):
    p_scores, r_scores, f1_scores = [], [], []

    if flair == None and stanza == None and spacy == None:
        return None

    for name in get_file_names():
        story = read_story_from_file(name)
        characters = get_characters_from_file(name)

        if flair != None:
            print(f"Running FLAIR for {name}")
            pred = flair.ner_flair(story, use_cr)
        elif stanza != None:
            print(f"Running STANZA for {name}")
            pred = stanza.ner_stanza(story, use_cr)
        elif spacy != None:
            print(f"Running SPACY for {name}")
            pred = spacy.ner_spacy(story, use_cr)

        p_scores.append(precision_score(pred, characters))
        r_scores.append(recall_score(pred, characters))
        f1_scores.append(f1_score(pred, characters))
    return (
        sum(p_scores) / len(p_scores),
        sum(r_scores) / len(r_scores),
        sum(f1_scores) / len(f1_scores),
    )
