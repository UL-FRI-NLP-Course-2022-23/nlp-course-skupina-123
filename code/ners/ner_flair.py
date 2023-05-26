from flair.data import Sentence
from flair.nn import Classifier
import sys
import re

sys.path.append("..")

from helpers.helper_cr import perform_cr


class FlairNer:
    def __init__(self):
        self.tagger = Classifier.load("ner")

    def ner_flair(self, story, use_cr=False):
        if use_cr == True:
            story = perform_cr(story)
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", story)]
        entities = set()
        for s in sentences:
            sentence = Sentence(s)
            self.tagger.predict(sentence)
            for entity in sentence.get_spans("ner"):
                if entity.tag == "PER":
                    entities.add(entity.text)
        return list(entities)

    def test_ner_flair(self):
        text = "Joseph Robinette Biden Jr. is an American politician who is the 46th and\
    current president of the United States. A member of the Democratic Party, \
    he served as the 47th vice president from 2009 to 2017 under Barack Obama and\
    represented Delaware in the United States Senate from 1973 to 2009."
        print("Entities without CR", self.ner_flair(text))
        print("Entities with CR", self.ner_flair(text, use_cr=True))
