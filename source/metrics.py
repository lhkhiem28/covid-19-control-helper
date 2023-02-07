import os, sys
from libs import *

def ner_f1_score(
    tags, preds
    , tag_names
):
    masks = tags != -100
    tags, preds = tags[masks].tolist(), preds[masks].tolist()
    tags, preds = [tag_names[tag] for tag in tags], [tag_names[pred] for pred in preds]

    f1 = seqmetrics.f1_score(
        [tags], [preds]
        , scheme = seqscheme.BIO2, mode = "strict"
        , average = "micro"
    )

    return f1

def ner_classification_report(
    tags, preds
    , tag_names
):
    masks = tags != -100
    tags, preds = tags[masks].tolist(), preds[masks].tolist()
    tags, preds = [tag_names[tag] for tag in tags], [tag_names[pred] for pred in preds]

    report = seqmetrics.classification_report(
        [tags], [preds]
        , scheme = seqscheme.BIO2, mode = "strict"
        , digits = 4
    )

    return report