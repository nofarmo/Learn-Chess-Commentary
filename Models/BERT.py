from transformers import BertTokenizer, BertLMHeadModel, BertConfig
from Utils import get_chess_tokens, dataset_tokens
import torch

BERT_TYPE = 'bert-base-uncased'

class BERT:
    def __init__(self):
        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)

        # special tokens
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens': dataset_tokens})

        # chess tokens
        self.tokenizer.add_tokens(get_chess_tokens())

        # model
        self.configuration = BertConfig.from_pretrained(BERT_TYPE)
        self.configuration.is_decoder = True

        self.model = BertLMHeadModel.from_pretrained(BERT_TYPE, config=self.configuration).cuda()

        self.model.resize_token_embeddings(len(self.tokenizer))

    def load_model(self, model_path):
        self.model = BertLMHeadModel(self.configuration)
        self.model.load_state_dict(torch.load(model_path))
