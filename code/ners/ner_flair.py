from flair.data import Sentence
from flair.nn import Classifier
import sys

sys.path.append("..")

from helpers.helper_cr import perform_cr


# story - text (whole story) (string)
# use_cr - use coreference resolution (boolean)
def ner_flair(story, use_cr=False):
    tagger = Classifier.load("ner")

    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", story)]
    entities = set()
    for s in sentences:
        sentence = Sentence(s)
        tagger.predict(sentence)
        for entity in sentence.get_spans("ner"):
            if entity.tag == "PER":
                entities.add(entity.text)
    return list(entities)


def test_ner_flair():
    text = "Joseph Robinette Biden Jr. is an American politician who is the 46th and\
  current president of the United States. A member of the Democratic Party, \
  he served as the 47th vice president from 2009 to 2017 under Barack Obama and\
  represented Delaware in the United States Senate from 1973 to 2009."
    print("Entities without CR", ner_flair(text))
    print("Entities with CR", ner_flair(text, use_cr=True))
