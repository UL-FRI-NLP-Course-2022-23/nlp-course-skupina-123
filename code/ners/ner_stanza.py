# pip install stanza

import stanza
import sys

sys.path.append('..')

from helpers.helper_cr import perform_cr

stanza.download('en')
stanza_nlp = stanza.Pipeline('en')

def ner_stanza(story, use_cr=False):
  if use_cr == True:
    story = perform_cr(story)

  doc = stanza_nlp(story)
  entities = list(set([ent.text.lower() for ent in filter(lambda e: e.type == "PERSON", doc.ents)]))
  return entities


def test_ner_stanza():
    text = "Joseph Robinette Biden Jr. is an American politician who is the 46th and\
    current president of the United States. A member of the Democratic Party, \
    he served as the 47th vice president from 2009 to 2017 under Barack Obama and\
    represented Delaware in the United States Senate from 1973 to 2009."
    print("Entities without CR", ner_stanza(text))
    print("Entities with CR", ner_stanza(text, use_cr=True))

test_ner_stanza()