
import os
import json

import torch
from torch import nn
from typing import Optional, Tuple
from dataclasses import dataclass

from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5EncoderModel
from transformers import T5Tokenizer, T5Config
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.file_utils import ModelOutput


@dataclass
class T5SiameseModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    left_embedding_vector: torch.FloatTensor = None
    right_embedding_vector: torch.FloatTensor = None
    cos: Optional[torch.FloatTensor] = None


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def cosine_similarity_dim1(a: torch.Tensor, b: torch.Tensor, eps=1e-5):
    dot_prod = torch.einsum('bn,bn->b', a, b)
    vecs_lens = torch.norm(a, dim=1) * torch.norm(b, dim=1)
    epsv = torch.Tensor([eps] * vecs_lens.shape[0])
    cos = dot_prod / torch.where(torch.less(vecs_lens, epsv), epsv, vecs_lens)
    return cos


class T5Siamese(T5PreTrainedModel):
    def __init__(self, config=None):
        super(T5Siamese, self).__init__(config)
        self.encoder_left = T5EncoderModel(config)
        self.encoder_right = T5EncoderModel(config)
        self.cos = nn.CosineSimilarity(dim=1)
        self.config = config
        self.model_parallel = False
        self.loss_func = nn.CosineEmbeddingLoss()

    @staticmethod
    def init_from_base_t5_model(model_name_or_path='t5-base', output_root='./'):
        os.makedirs(output_root, exist_ok=True)

        tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        model_config = T5Config.from_pretrained(model_name_or_path)

        # torch.save(model.encoder.embed_tokens.state_dict(), EMBEDDINGS_OUTPUT_FILE)
        tokenizer.save_pretrained(output_root)

        model = T5Siamese(config=model_config)
        model.encoder_left = T5EncoderModel.from_pretrained(model_name_or_path)
        model.encoder_right = T5EncoderModel.from_pretrained(model_name_or_path)
        model.save_pretrained(output_root)
        model_config.save_pretrained(output_root)

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
                input_ids,
                attention_mask):

        output_left = self.encoder_left(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        emb_left = output_left.last_hidden_state
        attention_mask_expand = attention_mask.unsqueeze(-1)
        emb_left = emb_left * attention_mask_expand
        emb_left = emb_left.sum(1) / attention_mask_expand.sum(1)
        return emb_left

    def encode_right(self,
                input_ids,
                attention_mask):

        output_right = self.encoder_right(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        emb_right = output_right.last_hidden_state
        attention_mask_expand = attention_mask.unsqueeze(-1)
        emb_right = emb_right * attention_mask_expand
        emb_right = emb_right.sum(1) / attention_mask_expand.sum(1)
        return emb_right

    def forward(self,
                input_left_ids,
                input_right_ids,
                attention_mask_left,
                attention_mask_right,
                labels=None):
        if input_left_ids.shape[0] != input_right_ids.shape[0]:
            raise Exception('In cosine similarity you should pass equal size batches')

        emb_left = self.encode_left(
            input_ids=input_left_ids,
            attention_mask=attention_mask_left,
        )

        emb_right = self.encode_right(
            input_ids=input_right_ids,
            attention_mask=attention_mask_right,
        )

        # get the final hidden states
        cos = self.cos(emb_left, emb_right)

        if labels is not None:
            loss = self.loss_func(emb_left, emb_right, labels)

            return T5SiameseModelOutput(
                loss=loss,
                cos=cos,
                left_embedding_vector=emb_left,
                right_embedding_vector=emb_right
            )

        return T5SiameseModelOutput(
            cos=cos,
            left_embedding_vector=emb_left,
            right_embedding_vector=emb_right
        )

