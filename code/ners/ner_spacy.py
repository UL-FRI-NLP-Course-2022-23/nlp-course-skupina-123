import sys

sys.path.append('..')

from helpers.helper_cr import perform_cr

def ner_spacy(story, spacy_nlp, use_cr=False):
    if use_cr == True:
        story = perform_cr(story)

    # Perform Spacy
    doc = spacy_nlp(story)

    # Extract only PERSON entities
    persons = [ent for ent in doc.ents if ent.label_ in ['PERSON']]

    # To lower case and remove 's
    persons = set([str(person).lower().replace("'s", "") for person in persons])
    return persons