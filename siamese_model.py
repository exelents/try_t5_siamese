
import os
import json

import torch
from torch import nn

from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5EncoderModel
from transformers import T5Tokenizer, T5Config
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map


class T5Siamese(T5PreTrainedModel):
    def __init__(self, encoder_left=None, encoder_right=None, embeddings=None, config=None):
        super(T5Siamese, self).__init__(config)
        self.encoder_left = encoder_left
        self.encoder_right = encoder_right
        self.embeddings = embeddings
        self.cos = nn.CosineSimilarity(dim=1)
        self.config = config

    @staticmethod
    def init_from_base_t5_model(model_name_or_path='t5-base', output_root='./'):
        MODEL = model_name_or_path
        MODEL_OUTPUT = output_root
        EMBEDDINGS_OUTPUT_FILE = os.path.join(MODEL_OUTPUT, 'embedding.pth')
        os.makedirs(MODEL_OUTPUT, exist_ok=True)

        tokenizer = T5Tokenizer.from_pretrained(MODEL)
        # model = T5ForConditionalGeneration.from_pretrained(MODEL)
        model_left = T5EncoderModel.from_pretrained(MODEL)
        model_right = T5EncoderModel.from_pretrained(MODEL)

        left_dir = os.path.join(MODEL_OUTPUT, 'left')
        right_dir = os.path.join(MODEL_OUTPUT, 'right')
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)

        # torch.save(model.encoder.embed_tokens.state_dict(), EMBEDDINGS_OUTPUT_FILE)
        model_left.save_pretrained(left_dir)
        model_right.save_pretrained(right_dir)
        tokenizer.save_pretrained(MODEL_OUTPUT)
        model_right.config.save_pretrained(MODEL_OUTPUT)

    @staticmethod
    def from_pretrained(model_path, with_tokenizer=False):
        MODEL_OUTPUT = model_path
        EMBEDDINGS_OUTPUT_FILE = os.path.join(MODEL_OUTPUT, 'embedding.pth')

        left_dir = os.path.join(MODEL_OUTPUT, 'left')
        right_dir = os.path.join(MODEL_OUTPUT, 'right')

        config_left = json.load(open(os.path.join(left_dir, 'config.json')))

        # embedding = nn.Embedding(config_left['vocab_size'], config_left['d_model'])
        # embedding.load_state_dict(torch.load(EMBEDDINGS_OUTPUT_FILE))
        encoder_left = T5EncoderModel.from_pretrained(left_dir) #, embed_tokens=embedding
        encoder_right = T5EncoderModel.from_pretrained(right_dir) #, embed_tokens=embedding
        config = T5Config.from_pretrained(MODEL_OUTPUT)
        if with_tokenizer:
            tokenizer = T5Tokenizer.from_pretrained(MODEL_OUTPUT)

            return tokenizer, T5Siamese(encoder_left=encoder_left,
                                        encoder_right=encoder_right,
                                        config=config)

        return T5Siamese(encoder_left=encoder_left,
                         encoder_right=encoder_right,
                         config=config)

    def save_pretrained(self, model_path):
        MODEL_OUTPUT = model_path

        left_dir = os.path.join(MODEL_OUTPUT, 'left')
        right_dir = os.path.join(MODEL_OUTPUT, 'right')
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)

        # torch.save(self.embeddings.state_dict(), EMBEDDINGS_OUTPUT_FILE)
        self.encoder_left.save_pretrained(left_dir)
        self.encoder_right.save_pretrained(right_dir)
        self.config.save_pretrained(MODEL_OUTPUT)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder_left.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder_left.block))
        self.encoder_left.parallelize(self.device_map)
        self.encoder_right.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder_left.deparallelize()
        self.encoder_right.deparallelize()
        self.encoder_left = self.encoder.to("cpu")
        self.encoder_right = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def encode_left(self,
                input_left_ids,
                attention_mask_left):

        output_left = self.encoder_left(
            input_ids=input_left_ids,
            attention_mask=attention_mask_left,
            return_dict=True
        )
        emb_left = output_left.last_hidden_state
        emb_left = torch.mean(emb_left, dim=1)
        return emb_left

    def encode_right(self,
                input_right_ids,
                attention_mask_right):

        output_right = self.encoder_right(
            input_ids=input_right_ids,
            attention_mask=attention_mask_right,
            return_dict=True
        )
        emb_right = output_right.last_hidden_state
        emb_right = torch.mean(emb_right, dim=1)
        return emb_right

    def forward(self,
                input_left_ids,
                input_right_ids,
                attention_mask_left,
                attention_mask_right,
                labels=None):
        if input_left_ids.shape[0] != input_right_ids.shape[0]:
            raise Exception('In cosine similarity you should pass equal size batches')

        output_left = self.encoder_left(
            input_ids=input_left_ids,
            attention_mask=attention_mask_left,
            return_dict=True
        )
        # with torch.no_grad(): # даёт ошибку при тренировке в режиме cpu_offload
        output_right = self.encoder_right(
            input_ids=input_right_ids,
            attention_mask=attention_mask_right,
            return_dict=True
        )

        # get the final hidden states
        emb_left = output_left.last_hidden_state  # * batch_left_tensor["attention_mask"]
        emb_right = output_right.last_hidden_state  # * batch_right_tensor["attention_mask"]
        emb_left = torch.mean(emb_left, dim=1)
        emb_right = torch.mean(emb_right, dim=1)
        cos = self.cos(emb_left, emb_right)

        if labels is not None:
            cross_entropy_loss = nn.BCEWithLogitsLoss()
            loss = cross_entropy_loss(cos, labels)
            return loss, cos

        return cos

