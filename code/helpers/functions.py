import json
from Levenshtein import jaro, ratio
import os
import re
import difflib


def read_story_from_file(file_name):
    with open(f"../data/stories/{file_name}.txt", encoding="utf8") as f:
        return f.read()


def get_characters_from_file(file_name):
    with open(f"../data/annotations/{file_name}.json", encoding="utf8") as f:
        return json.load(f)["characters"]


def get_file_names():
    path = "../data/stories/"
    file_names = os.listdir(path)
    file_names = sorted([re.sub(r"\.txt$", "", file_name) for file_name in file_names])
    return file_names


def string_similarity(s1, s2, threshold=0.5):
    print(s1, s2)
    return difflib.SequenceMatcher(a=s1.lower(), b=s2.lower()).ratio() >= threshold


def true_positive(list1, list2):
    tp = 0
    for x in list1:
        for y in list2:
            if string_similarity(x, y):
                tp += 1
                break
    return tp


def percision_score(list1, list2):
    try:
        return true_positive(list1, list2) / len(list1)
    except ZeroDivisionError:
        return 1


def recall_score(list1, list2):
    return true_positive(list1, list2) / len(list2)


def f1_score(list1, list2):
    per = percision_score(list1, list2)
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

        p_scores.append(percision_score(pred, characters))
        r_scores.append(recall_score(pred, characters))
        f1_scores.append(f1_score(pred, characters))
    return (
        sum(p_scores) / len(p_scores),
        sum(r_scores) / len(r_scores),
        sum(f1_scores) / len(f1_scores),
    )
