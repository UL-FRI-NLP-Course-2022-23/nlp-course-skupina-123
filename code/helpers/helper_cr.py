# pip install allennlp allennlp-models

from allennlp.predictors.predictor import Predictor

model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
predictor = Predictor.from_path(model_url)


def perform_cr(story):
    return predictor.coref_resolved(story)
