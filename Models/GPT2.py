from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from Utils import get_chess_tokens, dataset_tokens
import torch

GPT2_TYPE = "gpt2"


class GPT2:
    def __init__(self):
        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2_TYPE)

        # special tokens
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens': dataset_tokens})

        # chess tokens
        self.tokenizer.add_tokens(get_chess_tokens())

        # model
        self.configuration = GPT2Config.from_pretrained(GPT2_TYPE)
        self.model = GPT2LMHeadModel.from_pretrained(GPT2_TYPE, config=self.configuration).cuda()

        self.model.resize_token_embeddings(len(self.tokenizer))

    def load_model(self, model_path):
        self.model = GPT2LMHeadModel(self.configuration)
        self.model.load_state_dict(torch.load(model_path))
