import torch
from torch.utils.data import Dataset
from Utils import convert_data_to_text
import pickle


class MovesDataset(Dataset):
    def __init__(self, paths, tokenizer, max_length=768):

        self.comment_encoding = tokenizer.get_added_vocab()['<comment>']

        self.proccessed_data = []
        self.attn_masks = []
        self.labels = []

        for path in paths:
            with open(path, 'rb') as file:
                raw_data = pickle.load(file)
            for data_object in raw_data:
                text = convert_data_to_text(data_object)

                enc_text = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")

                inputs = enc_text['input_ids']
                label_idx = inputs.index(self.comment_encoding) + 1
                labels = [-100] * label_idx + inputs[label_idx:]

                self.proccessed_data.append(torch.tensor(inputs))
                self.attn_masks.append(torch.tensor(enc_text['attention_mask']))
                self.labels.append(torch.tensor(labels))

    def __len__(self):
        return len(self.proccessed_data)

    def __getitem__(self, index):
        return self.proccessed_data[index], self.attn_masks[index], self.labels[index]
