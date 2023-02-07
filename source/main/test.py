import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import NERDataset
from engines import *

tag_names = [
    "B-PATIENT_ID", 
    "B-NAME", 
    "B-AGE", 
    "B-GENDER", 
    "B-JOB", 
    "B-LOCATION", 
    "B-ORGANIZATION", 
    "B-SYMPTOM_AND_DISEASE", 
    "B-TRANSPORTATION", 
    "B-DATE", 
    "I-PATIENT_ID", 
    "I-NAME", 
    "I-AGE", 
    "I-GENDER", 
    "I-JOB", 
    "I-LOCATION", 
    "I-ORGANIZATION", 
    "I-SYMPTOM_AND_DISEASE", 
    "I-TRANSPORTATION", 
    "I-DATE", 
    "O"
]

test_loader = torch.utils.data.DataLoader(
    NERDataset(
        data_path = "../../datasets/PhoNER-COVID-19/word/test.json", 
        tag_names = tag_names, 
    ), 
    num_workers = 4, batch_size = 8, 
    shuffle = False, 
)
model = transformers.RobertaForTokenClassification.from_pretrained(
    "vinai/phobert-large", 
    num_labels = len(tag_names), 
)

save_ckp_dir = "../../ckps/PhoNER-COVID-19/word"
model = torch.load(
    "{}/best.ptl".format(save_ckp_dir), 
    map_location = "cuda", 
)
test_fn(
    test_loader, 
    model, 
)