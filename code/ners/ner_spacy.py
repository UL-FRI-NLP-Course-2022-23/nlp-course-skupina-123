import spacy
from helpers.helper_cr import perform_cr

class SpacyNer:
    def __init__(self):
        self.tagger = spacy.load("en_core_web_sm")

    def ner_spacy(self, story, use_cr=False):
        if use_cr == True:
            story = perform_cr(story)

        # Perform Spacy
        doc = self.tagger(story)

        # Extract only PERSON entities
        persons = [ent for ent in doc.ents if ent.label_ in ['PERSON']]

        # To lower case and remove 's
        persons = set([str(person).lower().replace("'s", "") for person in persons])
        return persons

    def test_ner_spacy(self):
        text = "Joseph Robinette Biden Jr. is an American politician who is the 46th and\
                current president of the United States. A member of the Democratic Party, \
                he served as the 47th vice president from 2009 to 2017 under Barack Obama and\
                represented Delaware in the United States Senate from 1973 to 2009."
        print("Entities without CR", self.ner_spacy(text))
        print("Entities with CR", self.ner_spacy(text, use_cr=True))