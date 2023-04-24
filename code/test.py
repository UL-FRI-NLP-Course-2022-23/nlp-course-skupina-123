import os
import json
import spacy
from ners.ner_spacy import ner_spacy

ANNOTATIONS_PATH = "./data/annotations/"
STORIES_PATH = "./data/stories/"
USE_CR = False

def run_spacy():
    nlp_spacy = spacy.load("en_core_web_sm")
    ground_truth = []
    results = []

    file_name = "002"
    gt_file = ANNOTATIONS_PATH + f"{file_name}.json"
    if (os.path.isfile(gt_file)):
        with open(gt_file, encoding='utf8') as f:
            annotation = f.read()
            annotation_json = json.loads(annotation)
            ground_truth = list(set(annotation_json['characters']))

    filename = STORIES_PATH + f'{file_name}.txt'
    if os.path.isfile(filename):
        with open(filename, encoding='utf8') as f:
            story = f.read()
            persons = ner_spacy(story, nlp_spacy, USE_CR)
            results = list(persons)

    print(f'Ground truths: {ground_truth}')
    print(f'Results: {results}')


def run_afinn():
    print("Running afinn")

if __name__ == "__main__":
    run_spacy()
    # run_afinn()