import sys
import stanza

sys.path.append("..")

from helpers.helper_cr import perform_cr

class StanzaNer:
    def __init__(self, tokenize_pretokenized=False):
        stanza.download('en')
        self.tagger = stanza.Pipeline('en', tokenize_pretokenized=tokenize_pretokenized)

    # story - text (whole story) (string)
    # use_cr - use coreference resolution (boolean)
    def ner_stanza(self, story, use_cr=False):
        if use_cr == True:
            story = perform_cr(story)

        doc = self.tagger(story)
        persons = filter(lambda e: e.type == "PERSON", doc.entities)
        return list(set([p.text for p in persons]))

    def ner_stanza_whole_doc(self, story, use_cr=False):
        if use_cr == True:
            story = perform_cr(story)

        doc = self.tagger(story)
        return doc, story

    def test_ner_stanza(self):
        text = "Joseph Robinette Biden Jr. is an American politician who is the 46th and\
        current president of the United States. A member of the Democratic Party, \
        he served as the 47th vice president from 2009 to 2017 under Barack Obama and\
        represented Delaware in the United States Senate from 1973 to 2009."
        print("Entities without CR", self.ner_stanza(text))
        print("Entities with CR", self.ner_stanza(text, use_cr=True))
