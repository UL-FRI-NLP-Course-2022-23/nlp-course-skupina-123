import os
import json
import re
os.listdir()


from flair.data import Sentence
from flair.nn import Classifier


f = "002"
story_path = "nlp-course-skupina-123/data/stories/" + f + ".txt"
story = open(story_path, "r").read()


sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', story)]

tagger = Classifier.load('ner')
for s in sentences:
  sentence = Sentence(s)
  tagger.predict(sentence)
  print(sentence)



# annotation_path = "nlp-course-skupina-123/data/annotations/" + f + ".json"
# a = json.load(open(annotation_path, "r"))
# a['characters']

stories_path = "nlp-course-skupina-123/data/stories"
annotations_path = "nlp-course-skupina-123/data/annotations"

# Loop through all files in the folder
for story_name in os.listdir(folder_path):
    # Check if the file has a .txt extension
    if story_name.endswith('.txt'):
        # Construct the full file path by joining the folder path and file name
        story_path = os.path.join(stories_path, story_name)
        annotation_path = os.path.join(annotations_path, story_name.split(".")[0], ".json")



        story = open(story_path, "r").read()

        annotation = json.load(open(annotation_path, "r"))