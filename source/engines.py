import os, sys
from libs import *

from metrics import *

def train_fn(
    train_loaders, num_epochs, 
    model, 
    optimizer, 
    tag_names, 
    save_ckp_dir = "./", 
):
    print("\nStart Training ...\n" + " = "*16)
    model = model.cuda()