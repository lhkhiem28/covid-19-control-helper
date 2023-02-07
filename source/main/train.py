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

train_loaders = {
    "train":torch.utils.data.DataLoader(
        NERDataset(
            data_path = "../../datasets/PhoNER-COVID-19/word/train.json", 
            tag_names = tag_names, 
        ), 
        num_workers = 4, batch_size = 8, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        NERDataset(
            data_path = "../../datasets/PhoNER-COVID-19/word/val.json", 
            tag_names = tag_names, 
        ), 
        num_workers = 4, batch_size = 8, 
        shuffle = True, 
    ), 
}
model = transformers.RobertaForTokenClassification.from_pretrained(
    "vinai/phobert-large", 
    num_labels = len(tag_names), 
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr = 5e-5, 
)

save_ckp_dir = "../../ckps/PhoNER-COVID-19/word"
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, num_epochs = 10, 
    model = model, 
    optimizer = optimizer, 
    save_ckp_dir = save_ckp_dir, 
)