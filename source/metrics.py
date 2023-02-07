import os, sys
from libs import *

def ner_f1_score(
    tags, preds, 
    tag_names, 
):
    masks = tags != -100
    tags, preds,  = tags[masks].tolist(), preds[masks].tolist(), 
    tags, preds,  = [tag_names[tag] for tag in tags], [tag_names[pred] for pred in preds], 

    f1_score = seqmetrics.f1_score(
        [tags], [preds], 
        scheme = seqscheme.BIO2, mode = "strict", 
        average = "micro", 
    )

    return f1_score

def ner_classification_report(
    tags, preds, 
    tag_names, 
):
    masks = tags != -100
    tags, preds,  = tags[masks].tolist(), preds[masks].tolist(), 
    tags, preds,  = [tag_names[tag] for tag in tags], [tag_names[pred] for pred in preds], 

    classification_report = seqmetrics.classification_report(
        [tags], [preds], 
        scheme = seqscheme.BIO2, mode = "strict", 
        digits = 4, 
    )

    return classification_report