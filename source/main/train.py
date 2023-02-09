import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import NERDataset
from engines import *

tag_names = [
    "B-ACCOUNT_NUMBER", 
    "B-ACCOUNT_NAME", 
    "B-DATE", 
    "B-UNIT_PRICE", 
    "B-MONEY", 
    "B-TAX_TYPE", 
    "B-TAX_RATE", 
    "B-VOUCHER_TYPE", 
    "I-ACCOUNT_NUMBER", 
    "I-ACCOUNT_NAME", 
    "I-DATE", 
    "I-UNIT_PRICE", 
    "I-MONEY", 
    "I-TAX_TYPE", 
    "I-TAX_RATE", 
    "I-VOUCHER_TYPE", 
    "O"
]

train_loaders = {
    "train":torch.utils.data.DataLoader(
        NERDataset(
            data_path = "../../datasets/AccountingNER/word/train.csv", 
            tag_names = tag_names, 
        ), 
        num_workers = 4, batch_size = 8, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        NERDataset(
            data_path = "../../datasets/AccountingNER/word/val.csv", 
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
optimizer = torch.optim.Adam(
    model.parameters(), lr = 1e-5, 
)

save_ckp_dir = "../../ckps/AccountingNER/word"
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, num_epochs = 20, 
    model = model, 
    optimizer = optimizer, 
    save_ckp_dir = save_ckp_dir, 
)