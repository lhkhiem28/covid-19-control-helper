import os, sys
from libs import *

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_path, 
        tag_names, 
    ):
        self.data = pd.read_csv(data_path)
        self.tag_names = tag_names
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "vinai/phobert-large", 
            use_fast = False, 
        )

    def __len__(self, 
    ):
        return len(self.data)

    def __getitem__(self, 
        index, 
    ):
        sample = self.data.iloc[index]
        words, tags = sample["words"].split(), sample["tags"].split()
        encoded_words, encoded_tags = vncorenlp.ner_encode(
            words, tags
            , tag_names = self.tag_names
            , tokenizer = self.tokenizer
        )

        return np.array(encoded_words), np.array(encoded_tags)