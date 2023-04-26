import json


def read_story_from_file(file_name):
    with open(f"../data/stories/{file_name}.txt", encoding="utf8") as f:
        return f.read()


def get_characters_from_file(file_name):
    with open(f"../data/annotations/{file_name}.json", encoding="utf8") as f:
        return json.load(f)
