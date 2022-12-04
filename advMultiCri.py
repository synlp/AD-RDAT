import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss import EntropyLoss

import math

from pytorch_transformers import BertModel,BertConfig
from embedding import DiaEmbedding

class BiLSTM(nn.Module):
    def __init__(self, in_feature, out_feature, num_layers=1, batch_first = True):
        super(BiLSTM, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=in_feature,
            hidden_size=out_feature,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True
        )

    def rand_init_hidden(self, batch_size, device):
        return (torch.zeros(2 * self.num_layers, batch_size, self.out_feature).to(device),
                torch.zeros(2 * self.num_layers, batch_size, self.out_feature).to(device))

    def forward(self, input):
        batch_size, seq_len, hidden_size = input.shape
        hidden = self.rand_init_hidden(batch_size, input.device)
        output, hidden = self.lstm(input, hidden)
        return output.contiguous().view(batch_size, seq_len, self.out_feature * 2)

class Transformer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=512, dropout=0.2, num_layers=1):
        super(Transformer, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, input):
        output = self.transformer(input.transpose(0, 1))
        return output.transpose(0, 1)

class AdvMultiCriModel(nn.Module):
    def __init__(self, tag_size=100, bert_path=None, embedding='embedding', encoder="bilstm",
                 num_layers=1, criteria_size=2, multi_criteria=True, adversary=True, adv_coefficient=1):
        super(AdvMultiCriModel, self).__init__()
        self.tag_size = tag_size
        self.criteria_size = criteria_size
        self.num_layers = num_layers
        self.embedding_type = embedding
        self.encoder_type = encoder
        self.multi_criteria = multi_criteria
        self.adversary = adversary
        self.adv_coefficient = adv_coefficient
        bert_config = BertConfig.from_pretrained(bert_path)
        hidden_size = bert_config.hidden_size
        self.config = bert_config
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        if multi_criteria is False:
            self.criteria_size = 1
        if encoder == "bilstm":
            self.embedding = DiaEmbedding(bert_path, embedding, hidden_size, bert_config=bert_config)
            self.shared_encoder = BiLSTM(in_feature=hidden_size, out_feature=hidden_size, num_layers=num_layers,
                                         batch_first=True)
            self.private_encoder = BiLSTM(in_feature=hidden_size, out_feature=hidden_size, num_layers=num_layers,
                                          batch_first=True)
            self.private_encoder_knowledge = BiLSTM(in_feature=hidden_size, out_feature=hidden_size,
                                                    num_layers=num_layers, batch_first=True)
            self.classifier = torch.nn.Linear(hidden_size * 4, tag_size)
            self.classifier_knowledge = torch.nn.Linear(hidden_size * 4, tag_size)
            self.discriminator = torch.nn.Linear(hidden_size * 2, 2, bias=True)

        elif encoder == "transformer":
            self.embedding = DiaEmbedding(bert_path, embedding, hidden_size, using_position=True, bert_config=bert_config)
            self.shared_encoder = Transformer(hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1,
                                              num_layers=num_layers)
            self.private_encoder = Transformer(hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1,
                                               num_layers=num_layers)
            self.private_encoder_knowledge = Transformer(hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1,
                                                         num_layers=num_layers)
            self.classifier = torch.nn.Linear(hidden_size * 2, tag_size)
            self.classifier_knowledge = torch.nn.Linear(hidden_size * 2, tag_size)
            self.discriminator = torch.nn.Linear(hidden_size, 2, bias=True)

        else:
            raise Exception("Invalid encoder")

    def _reset_params(self, initializer):
        for child in self.children():
            if type(child) == DiaEmbedding and "bert" in self.embedding_type:
                continue
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_valid_seq_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output

    def forward(self, input_ids, token_type_ids=None, valid_ids=None, labels = None, c2w_map=None, wordpiece_ids=None,
                criteria_index=0):
        if criteria_index not in [0, 1]:
            raise Exception("criteria_index Invalid")

        embedding_output = self.embedding(input_ids, token_type_ids, wordpiece_ids, c2w_map)
        shared_output = self.shared_encoder(embedding_output)
        if criteria_index == 0:
            private_output = self.private_encoder(embedding_output)
        elif criteria_index == 1:
            private_output = self.private_encoder_knowledge(embedding_output)

        if valid_ids is not None:
            shared_output = self.get_valid_seq_output(shared_output, valid_ids)
            private_output = self.get_valid_seq_output(private_output, valid_ids)

        sequence_output = torch.cat([shared_output, private_output], dim=-1)
        sequence_output = self.dropout(sequence_output)

        if criteria_index == 0:
            logits = self.classifier(sequence_output)
        elif criteria_index == 1:
            logits = self.classifier_knowledge(sequence_output)
        tag_seq = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            total_loss = loss_fct(logits.view(-1, self.tag_size), labels.view(-1))

            if self.adversary is True:
                loss_discriminator = CrossEntropyLoss(ignore_index=-100)
                loss_entropy = EntropyLoss(coefficient=self.adv_coefficient)
                d_input = torch.mean(shared_output, dim=1)
                d_result = self.discriminator(d_input)
                d_labels = [criteria_index] * d_result.shape[0]
                d_loss = loss_discriminator(d_result.view(-1, 2), torch.LongTensor(d_labels).to(input_ids.device))
                probability_distribution = torch.softmax(d_result, dim=-1)
                h_loss = loss_entropy(probability_distribution)
            else:
                d_loss, h_loss = 0, 0
            return tag_seq, total_loss, d_loss, h_loss
        else:
            return tag_seq