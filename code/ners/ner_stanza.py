import sys

sys.path.append("..")

from helpers.helper_cr import perform_cr


def ner_stanza(story, stanza_nlp, use_cr=False):
    if use_cr == True:
        story = perform_cr(story)

    doc = stanza_nlp(story)
    return doc


# story - text (whole story) (string)
# stanza_nlp - initialized stanza pipeline (check stanza_sentiment.py main)
# use_cr - use coreference resolution (boolean)
def ner_stanza_person_entities(story, stanza_nlp, use_cr=False):
    doc = ner_stanza(story, stanza_nlp, use_cr)
    return list(filter(lambda e: e.type == "PERSON", doc.entities))


def test_ner_stanza():
    text = "Joseph Robinette Biden Jr. is an American politician who is the 46th and\
    current president of the United States. A member of the Democratic Party, \
    he served as the 47th vice president from 2009 to 2017 under Barack Obama and\
    represented Delaware in the United States Senate from 1973 to 2009."
    print("Entities without CR", ner_stanza(text))
    print("Entities with CR", ner_stanza(text, use_cr=True))
