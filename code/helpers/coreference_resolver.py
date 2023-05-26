from allennlp.predictors.predictor import Predictor

class CoreferenceResolver:
    def __init__(self):
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

    def resolve_coreferences(self, text):
        return self.predictor.coref_resolved(text)
