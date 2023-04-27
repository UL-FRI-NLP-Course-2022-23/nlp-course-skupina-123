import pandas as pd
import ners.ner_stanza as ns

def sentiment_mapping(sentiment_value):
    if sentiment_value == 0:
        return -1
    elif sentiment_value == 1:
        return 0
    else:
        return 1

def get_person_df(persons):
    rows = []
    for person in persons:
        row = {
            "text": person.text,
            "type": person.type,
            "start_char": person.start_char,
            "end_char": person.end_char,
            "sentence_sentiment": sentiment_mapping(person._sent.sentiment)
        }
        rows.append(row)

    return pd.DataFrame(rows)

def read_story_from_file(file_name):
    with open(f'../data/stories/{file_name}.txt', encoding="utf8") as f:
        return f.read()

def get_person_entities(doc):
    return list(filter(lambda e: e.type == "PERSON", doc.entities))

def perform_sentiment(df):
    sentiment = df.copy()
    sentiment = sentiment.drop(columns=["type", "start_char", "end_char"])
    sentiment = sentiment.groupby("text").sum().reset_index()

    return sentiment.sort_values("sentence_sentiment")

if __name__ == "__main__":
    stanza = ns.StanzaNer()

    story = read_story_from_file(file_name="007")
    
    doc = stanza.ner_stanza_whole_doc(story, use_cr=True)

    person_df = get_person_df(get_person_entities(doc))

    sentiment_df = perform_sentiment(person_df)

    print(sentiment_df.head())
