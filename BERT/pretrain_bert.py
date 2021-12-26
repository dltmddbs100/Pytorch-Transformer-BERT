import torch
from torch import nn

from BERT import BertLayerNorm, BertModel, gelu


# MLM : Masked Language Model

class MaskedWordPredictions(nn.Module):
    def __init__(self, config):
        '''BertLMPredictionHead'''
        super(MaskedWordPredictions, self).__init__()

        # BERT로부터 출력된 특징량을 변환하는 모듈
        self.transform = BertPredictionHeadTransform(config)

        # self.transform의 출력으로부터, 각 위치의 단어가 어느 쪽인지를 맞추는 전체 결합층
        self.decoder = nn.Linear(in_features=config.hidden_size,  # 'hidden_size': 768
                                 out_features=config.vocab_size,  # 'vocab_size': 30522
                                 bias=False)
        
        self.bias = nn.Parameter(torch.zeros(
            config.vocab_size))  # 'vocab_size': 30522

    def forward(self, hidden_states):
        '''
        hidden_states：BERT output [batch_size, seq_len, hidden_size]
        '''
        # BERT output
        # [batch_size, seq_len, hidden_size]
        hidden_states = self.transform(hidden_states)

        # 각 위치의 단어가 어휘의 어느 단어인지를 클래스 분류로 예측
        # [batch_size, seq_len, vocab_size]
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    '''MaskedWordPredictions에서, BERT로부터의 특징량을 변환하는 모듈'''

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 'hidden_size': 768

        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        '''hidden_states : [batch, seq_len, hidden_size]'''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# NSP：Next Sentence Prediction Model

class SeqRelationship(nn.Module):
    def __init__(self, config, out_features):
        '''NSP를 수행하는 모듈'''
        super(SeqRelationship, self).__init__()

        # 두문장이 연결되어 있는지 예측
        self.seq_relationship = nn.Linear(config.hidden_size, out_features) # 'hidden_size': 768, 'out_features': 2

    def forward(self, pooled_output): # [CLS] 토큰의 특징량을 사용하므로 pooled_output 사용
        return self.seq_relationship(pooled_output)


# Final Module

class BertPreTrainingHeads(nn.Module):
    '''사전 학습 문제를 위한 어댑터 모듈'''

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()

        # MLM
        self.predictions = MaskedWordPredictions(config)

        # NSP
        self.seq_relationship = SeqRelationship(config, 2)

    def forward(self, sequence_output, pooled_output):
        '''
        sequence_output:[batch_size, seq_len, hidden_size]
        pooled_output:[batch_size, hidden_size]
        '''
        # 입력의 마스크된 각 단어가 어떤 단어인지 예측
        # [batch, seq_len, vocab_size]
        prediction_scores = self.predictions(sequence_output)

        # [CLS]의 특징량으로부터 두문장이 연결되어 있는지 예측
        seq_relationship_score = self.seq_relationship(
            pooled_output)  # [batch, 2]

        return prediction_scores, seq_relationship_score


class BertForMaskedLM(nn.Module):
    '''최종적으로 BERT 사전 학습 과제를 수행하는 모듈'''
    def __init__(self, config, net_bert):
        super(BertForMaskedLM, self).__init__()

        # BERT 모델
        self.bert = BertModel(config)

        # 사전 학습 문제를 위한 어댑터 모듈
        self.cls = BertPreTrainingHeads(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        '''
        input_ids： [batch_size, seq_len]
        token_type_ids： [batch_size, seq_len]
        attention_mask：masking
        '''
        # BERT output 출력
        encoded_layers, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=False)

        # Pretrain 시 output
        # 이를 바탕으로 loss를 갱신하며 pretraining
        prediction_scores, seq_relationship_score = self.cls(
            encoded_layers, pooled_output)

        return prediction_scores, seq_relationship_score