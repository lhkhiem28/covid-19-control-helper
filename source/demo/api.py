import os, sys
from libs import *

class NER():
    def __init__(self, 
        ckp_dir, 
    ):
        self.segmenter = vncorenlp.VnCoreNLP(annotators = ["wseg"], 
            save_dir = "/home/ubuntu/khiem.lh/Free/ViNER/source/demo/VnCoreNLP/", 
        )
        self.model = transformers.pipeline("token-classification", 
            ckp_dir, aggregation_strategy = "simple", 
        )

    def ner_predict(self, 
        sentence, 
    ):
        sentence = vitools.normalize_diacritics(sentence)
        sentence = self.segmenter.word_segment(sentence)[0]
        output = {
            "ACCOUNT_NUMBER":[], 
            "ACCOUNT_NAME":[], 
            "DATE":[], 
            "UNIT_PRICE":[], 
            "MONEY":[], 
            "TAX_TYPE":[], 
            "TAX_RATE":[], 
            "VOUCHER_TYPE":[], 
        }
        pred = self.model(sentence)
        for entity in pred:
            if entity["entity_group"] in output:
                output[entity["entity_group"]].append(entity["word"])

        for entity_group, entities in output.items():
            fixed_entities = []
            i = 0
            while i < len(entities):
                if entities[i].endswith("@@"):
                    fixed_entity = entities[i][:-2] + entities[i + 1]
                    fixed_entities.append(fixed_entity.replace("_", " "))
                    i += 2
                else:
                    fixed_entity = entities[i]
                    fixed_entities.append(fixed_entity.replace("_", " "))
                    i += 1
            output[entity_group] = fixed_entities

        return output