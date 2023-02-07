import os, sys
from libs import *

class NER():
    def __init__(self, 
        ckp_dir, 
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "vinai/phobert-large", 
            use_fast = False, 
        )
        self.model = transformers.pipeline("token-classification", 
            ckp_dir, aggregation_strategy = "simple", 
        )

    def ner_predict(self, 
        sentence, 
    ):
        sentence = vitools.normalize_diacritics(sentence)
        sentence = underthesea.word_tokenize(sentence, format = "text")

        output = {
            "PATIENT_ID":[], 
            "NAME":[], 
            "AGE":[], 
            "GENDER":[], 
            "JOB":[], 
            "LOCATION":[], 
            "ORGANIZATION":[], 
            "SYMPTOM_AND_DISEASE":[], 
            "TRANSPORTATION":[], 
            "DATE":[], 
        }
        pred = self.model(sentence)
        for entity in pred:
            if entity["entity_group"] in output:
                output[entity["entity_group"]].append(entity["word"])

        return output