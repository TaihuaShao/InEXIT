# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoModel, AutoTokenizer
import math
import copy

def dropout(input_tensor, dropout_prob):
  """Perform dropout.
  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `torch.nn.dropout`).
  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  x = torch.nn.dropout(1.0 - dropout_prob)
  output = x(input_tensor) 
  return output


def layer_norm(input_tensor):
  """Run layer normalization on the last dimension of the tensor."""
  return torch.nn.LayerNorm(
      input_tensor, elementwise_affine=True)


def layer_norm_and_dropout(input_tensor, dropout_prob):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class BertMatchingModel(nn.Module):
    def __init__(self, args):
        super(BertMatchingModel, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.bert_path)
      
        for param in self.bert.parameters():
            param.requires_grad = True

        self.encoder = Encoder(args.word_emb_dim, args.num_heads, args.hidden_size, args.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(args.num_layers)])

        self.encoder_2 = Encoder(args.word_emb_dim, args.num_heads, args.hidden_size, args.dropout)
        self.encoders_2 = nn.ModuleList([
            copy.deepcopy(self.encoder_2)
            for _ in range(args.num_layers)])

        self.geek_pool = nn.AdaptiveAvgPool2d((1, args.word_emb_dim))
        self.job_pool = nn.AdaptiveAvgPool2d((1, args.word_emb_dim))

        self.mlp = MLP(
            input_size=args.word_emb_dim * 3,
            output_size=1
        )    

    def forward(self, geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask, job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask, geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask, job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask):

        # taxon_embedding
        geek_taxon = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 0, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 0, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 0, :].squeeze(1)).last_hidden_state).squeeze(1) # batch_size * max_len * word_embedding_size -> batch_size * word_embedding_size
        job_taxon = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 0, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 0, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 0, :].squeeze(1)).last_hidden_state).squeeze(1)

        # key_ewmbedding        
        geek_key_0 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 1, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 1, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 1, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_1 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 2, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 2, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 2, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_2 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 3, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 3, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 3, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_3 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 4, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 4, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 4, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_4 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 5, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 5, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 5, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_5 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 6, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 6, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 6, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_6 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 7, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 7, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 7, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_7 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 8, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 8, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 8, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_8 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 9, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 9, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 9, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_9 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 10, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 10, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 10, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_10 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 11, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 11, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 11, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_key_11 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 12, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 12, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 12, :].squeeze(1)).last_hidden_state).squeeze(1)

        job_key_0 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 1, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 1, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 1, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_key_1 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 2, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 2, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 2, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_key_2 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 3, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 3, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 3, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_key_3 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 4, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 4, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 4, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_key_4 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 5, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 5, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 5, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_key_5 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 6, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 6, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 6, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_key_6 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 7, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 7, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 7, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_key_7 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 8, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 8, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 8, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_key_8 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 9, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 9, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 9, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_key_9 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 10, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 10, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 10, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_key_10 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 11, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 11, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 11, :].squeeze(1)).last_hidden_state).squeeze(1)

        # value_ewmbedding
        geek_value_0 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 13, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 13, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 13, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_1 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 14, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 14, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 14, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_2 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 15, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 15, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 15, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_3 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 16, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 16, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 16, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_4 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 17, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 17, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 17, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_5 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 18, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 18, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 18, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_6 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 19, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 19, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 19, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_7 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 20, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 20, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 20, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_8 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 21, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 21, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 21, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_9 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 22, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 22, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 22, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_10 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 23, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 23, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 23, :].squeeze(1)).last_hidden_state).squeeze(1)
        geek_value_11 = self.geek_pool(self.bert(input_ids=geek_sent_tokens_input_ids.squeeze(1), token_type_ids=geek_tokens_sent_token_type_ids.squeeze(1), attention_mask=geek_sent_tokens_attention_mask.squeeze(1)).last_hidden_state).squeeze(1)

        job_value_0 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 12, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 12, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 12, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_value_1 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 13, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 13, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 13, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_value_2 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 14, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 14, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 14, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_value_3 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 15, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 15, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 15, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_value_4 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 16, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 16, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 16, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_value_5 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 17, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 17, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 17, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_value_6 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 18, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 18, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 18, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_value_7 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 19, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 19, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 19, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_value_8 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 20, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 20, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 20, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_value_9 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 21, :].squeeze(1), token_type_ids=job_tokens_token_type_ids[:, 21, :].squeeze(1), attention_mask=job_tokens_attention_mask[:, 21, :].squeeze(1)).last_hidden_state).squeeze(1)
        job_value_10 = self.job_pool(self.bert(input_ids=job_sent_tokens_input_ids.squeeze(1), token_type_ids=job_sent_tokens_token_type_ids.squeeze(1), attention_mask=job_sent_tokens_attention_mask.squeeze(1)).last_hidden_state).squeeze(1)
        

        # Inner interaction
        if self.args.fusion =='cat':
            geek_0 = torch.cat([geek_key_0, geek_value_0], dim=1).unsqueeze(1)  # batch_size * 1 * 2 word_embedding_size
            geek_1 = torch.cat([geek_key_1, geek_value_1], dim=1).unsqueeze(1)
            geek_2 = torch.cat([geek_key_2, geek_value_2], dim=1).unsqueeze(1)
            geek_3 = torch.cat([geek_key_3, geek_value_3], dim=1).unsqueeze(1)
            geek_4 = torch.cat([geek_key_4, geek_value_4], dim=1).unsqueeze(1)
            geek_5 = torch.cat([geek_key_5, geek_value_5], dim=1).unsqueeze(1)
            geek_6 = torch.cat([geek_key_6, geek_value_6], dim=1).unsqueeze(1)
            geek_7 = torch.cat([geek_key_7, geek_value_7], dim=1).unsqueeze(1)
            geek_8 = torch.cat([geek_key_8, geek_value_8], dim=1).unsqueeze(1)
            geek_9 = torch.cat([geek_key_9, geek_value_9], dim=1).unsqueeze(1)
            geek_10 = torch.cat([geek_key_10, geek_value_10], dim=1).unsqueeze(1)
            geek_11 = torch.cat([geek_key_11, geek_value_11], dim=1).unsqueeze(1)
            
            job_0 = torch.cat([job_key_0, job_value_0], dim=1).unsqueeze(1)   # batch_size * 1 * word_embedding_size
            job_1 = torch.cat([job_key_1, job_value_1], dim=1).unsqueeze(1)
            job_2 = torch.cat([job_key_2, job_value_2], dim=1).unsqueeze(1)
            job_3 = torch.cat([job_key_3, job_value_3], dim=1).unsqueeze(1)
            job_4 = torch.cat([job_key_4, job_value_4], dim=1).unsqueeze(1)
            job_5 = torch.cat([job_key_5, job_value_5], dim=1).unsqueeze(1)
            job_6 = torch.cat([job_key_6, job_value_6], dim=1).unsqueeze(1)
            job_7 = torch.cat([job_key_7, job_value_7], dim=1).unsqueeze(1)
            job_8 = torch.cat([job_key_8, job_value_8], dim=1).unsqueeze(1)
            job_9 = torch.cat([job_key_9, job_value_9], dim=1).unsqueeze(1)
            job_10 = torch.cat([job_key_10, job_value_10], dim=1).unsqueeze(1)
        else:
            geek_0 = (geek_key_0 + geek_value_0).unsqueeze(1)
            geek_1 = (geek_key_1 + geek_value_1).unsqueeze(1)
            geek_2 = (geek_key_2 + geek_value_2).unsqueeze(1)
            geek_3 = (geek_key_3 + geek_value_3).unsqueeze(1)
            geek_4 = (geek_key_4 + geek_value_4).unsqueeze(1)
            geek_5 = (geek_key_5 + geek_value_5).unsqueeze(1)
            geek_6 = (geek_key_6 + geek_value_6).unsqueeze(1)
            geek_7 = (geek_key_7 + geek_value_7).unsqueeze(1)
            geek_8 = (geek_key_8 + geek_value_8).unsqueeze(1)
            geek_9 = (geek_key_9 + geek_value_9).unsqueeze(1)
            geek_10 = (geek_key_10 + geek_value_10).unsqueeze(1)
            geek_11 = (geek_key_11 + geek_value_11).unsqueeze(1)
            job_0 = (job_key_0 + job_value_0).unsqueeze(1)
            job_1 = (job_key_1 + job_value_1).unsqueeze(1)
            job_2 = (job_key_2 + job_value_2).unsqueeze(1)
            job_3 = (job_key_3 + job_value_3).unsqueeze(1)
            job_4 = (job_key_4 + job_value_4).unsqueeze(1)
            job_5 = (job_key_5 + job_value_5).unsqueeze(1)
            job_6 = (job_key_6 + job_value_6).unsqueeze(1)
            job_7 = (job_key_7 + job_value_7).unsqueeze(1)
            job_8 = (job_key_8 + job_value_8).unsqueeze(1)
            job_9 = (job_key_9 + job_value_9).unsqueeze(1)
            job_10 = (job_key_10 + job_value_10).unsqueeze(1)
        geek = torch.cat([geek_0, geek_1, geek_2, geek_3, geek_4, geek_5, geek_6, geek_7, geek_8, geek_9, geek_10, geek_11], dim=1) # batch_size * 12 * (2) word_embedding_size
        job = torch.cat([job_0, job_1, job_2, job_3, job_4, job_5, job_6, job_7, job_8, job_9, job_10], dim=1) # batch_size * 11 * (2) word_embedding_size

        for encoder in self.encoders:
            geek, job = encoder(geek), encoder(job)    # batch_size * 12 * word_embedding_size


        if self.args.fusion =='cat':
            geek = torch.cat([torch.repeat_interleave(geek_taxon.unsqueeze(1), repeats=12, dim=1), geek], dim=2) # batch_size * 12 * (3) word_embedding_size
            job = torch.cat([torch.repeat_interleave(job_taxon.unsqueeze(1), repeats=11, dim=1), job], dim=2) # batch_size * 11 * (3) word_embedding_size
        else:
            geek = torch.repeat_interleave(geek_taxon.unsqueeze(1), repeats=12, dim=1) + geek # batch_size * 12 * word_embedding_size
            job = torch.repeat_interleave(job_taxon.unsqueeze(1), repeats=11, dim=1) + job # batch_size * 11 * word_embedding_size

        geek_job = torch.cat([geek, job], dim=1) # batch_size * 12 + 11 * (3) word_embedding_size

        for encoder_2 in self.encoders_2:
            geek_job = encoder_2(geek_job)  # batch_size * 12 + 11 * (3) word_embedding_size

        geek_vec, job_vec = torch.split(geek_job, (12,11), dim=1)
        geek_vec, job_vec = self.geek_pool(geek_vec).squeeze(1),self.job_pool(job_vec).squeeze(1)
        x = torch.cat([job_vec, geek_vec, job_vec - geek_vec], dim=1)
        output = self.mlp(x).squeeze(1)

        return output


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to('cuda')
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5 
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x 
        out = self.layer_norm(out)
        return out