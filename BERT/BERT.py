import math

import torch
from torch import nn


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """LayerNormalization"""

        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weight
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # bias
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BertEmbeddings(nn.Module):
    """Token Embedding, Sentence Embedding, Positional Embedding을 구해 더한 텐서를 반환    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        # Token Embedding： inputs_id를 벡터로 변환
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)

        # Transformer Positional Embedding：위치 정보 텐서를 벡터로 변환
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

        # Sentence Embedding：첫번째, 두번째 문장인지 구별하는 벡터로 변환
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        '''
        input_ids： [batch_size, seq_len]
        token_type_ids：[batch_size, seq_len]
        '''

        # 1. Token Embeddings
        words_embeddings = self.word_embeddings(input_ids)

        # 2. Sentence Embedding
        # token_type_ids가 없는 경우는 모두 0으로 설정
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 3. Transformer Positional Embedding：
        # [0, 1, 2 ・・・]로 길이만큼 숫자가 올라감
        seq_length = input_ids.size(1) 
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # 3개의 embedding 텐서를 합침
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # LayerNormalization & Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()

        # Self-Attention
        self.attention = BertAttention(config)

        # Self-Attention의 출력을 처리
        self.intermediate = BertIntermediate(config)

        # Self-Attention의 출력과 BertLayer의 input을 더함
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states：Embedder 모듈의 출력 [batch_size, seq_len, hidden_size]
        attention_mask：마스킹
        attention_show_flg：Self-Attention 가중치를 반환
        '''
        if attention_show_flg == True:
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output  # [batch_size, seq_length, hidden_size]


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        '''
        input_tensor：Embeddings또는 이전 BertLayer의 출력
        attention_mask：masking
        attention_show_flg
        '''
        if attention_show_flg == True:
            self_output, attention_probs = self.selfattn(
                input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.selfattn(
                input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads # 12

        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)  # 768/12=64

        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64 = 768

        # Query, Key, Value 설정
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        '''multi-head Attention용으로 텐서 형변환
        [batch_size, seq_len, hidden] --> [batch_size, 12(num_heads), seq_len, hidden/12] 
        '''
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states：Embeddings또는 이전 BertLayer의 출력
        attention_mask：masking
        attention_show_flg
        '''

        mixed_query_layer = self.query(hidden_states) # [batch_size, seq_len, hidden]
        mixed_key_layer = self.key(hidden_states) # [batch_size, seq_len, hidden]
        mixed_value_layer = self.value(hidden_states) # [batch_size, seq_len, hidden]

        # multi-head Attention용으로 텐서 변환
        query_layer = self.transpose_for_scores(mixed_query_layer) # [batch_size, 12, seq_len, hidden/12] 
        key_layer = self.transpose_for_scores(mixed_key_layer) # [batch_size, 12, seq_len, hidden/12] 
        value_layer = self.transpose_for_scores(mixed_value_layer) # [batch_size, 12, seq_len, hidden/12] 
 
        # 특징량끼리 곱해 Attention score를 구함
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [batch_size, 12, seq_len, seq_len] 
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # masking
        attention_scores = attention_scores + attention_mask

        # Attention 정규화
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        # Attention Map을 곱함
        context_layer = torch.matmul(attention_probs, value_layer) # [batch_size, 12, seq_len, hidden/12] 

        # multi-head Attention의 텐서를 원래 형태로 되돌림
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [batch_size, seq_len, hidden]

        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states：BertSelfAttention의 출력
        input_tensor：Embeddings또는 이전 BertLayer의 출력
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) 
        return hidden_states


def gelu(x):
    '''Gaussian Error Linear Unit으로 LeLU가 0에서 거칠고
    불연속적이어서 그부분은 연속적으로 매끄럽게 한 LeLU 형태
    '''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    '''TransformerBlock의 FeedForward에 해당'''
    def __init__(self, config):
        super(BertIntermediate, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size) # 768, 3072
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        '''
        hidden_states： BertAttention의 output
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states) # [batch_size, seq_len, intermediate_size]
        return hidden_states


class BertOutput(nn.Module):
    '''TransformerBlock의 FeedForward에 해당'''
    def __init__(self, config):
        super(BertOutput, self).__init__()

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size) # 3072,768

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states： BertIntermediate의 output
        input_tensor：BertAttention의 ouput
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # [batch_size, seq_len, intermediate_size]
        return hidden_states


class BertEncoder(nn.Module):
    def __init__(self, config):
        '''BertLaye를 반복해서 Layer를 구성'''
        super(BertEncoder, self).__init__()

        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)]) # 12번 반복

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        hidden_states：Embeddings의 output
        attention_mask：masking
        output_all_encoded_layers：모든 BertLayer의 output을 출력할지 
        최종 layer의 output을 출력할지를 조정
        attention_show_flg
        '''
        all_encoder_layers = []

        # BertLayer를 반복
        for layer_module in self.layer:

            if attention_show_flg == True:
                hidden_states, attention_probs = layer_module(
                    hidden_states, attention_mask, attention_show_flg)
            elif attention_show_flg == False:
                hidden_states = layer_module(
                    hidden_states, attention_mask, attention_show_flg)

            # 12층분을 모두 저장함
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # 마지막 layer의 output만 저장
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        if attention_show_flg == True:
            return all_encoder_layers, attention_probs
        elif attention_show_flg == False:
            return all_encoder_layers


class BertPooler(nn.Module):
    '''입력 문장의 [CLS] 토큰의 특징량만을 반환'''
    def __init__(self, config):
        super(BertPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 768, 768
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # [CLS] 토큰의 특징량만 추출
        first_token_tensor = hidden_states[:, 0] # [batch, hidden]

        # 특징량 변환
        pooled_output = self.dense(first_token_tensor) # [batch, hidden]

        # Tanh
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()

        # 3つのモジュールを作成
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        input_ids： [batch_size, sequence_length] 문장의 token_id
        token_type_ids： [batch_size, sequence_length] 각 단어의 문장 id
        attention_mask：masking
        output_all_encoded_layers：모든 BertLayer의 output을 출력할지 
        최종 layer의 output을 출력할지를 조정
        attention_show_flg
        '''
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # multi-head Attention에서 사용할 형태로 하기위해
        # maks를 [batch, 1, 1, seq_length] 형태로 변환
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) 

        # softmax 연산시 masking을 수행하기위해 0과 -inf(-10000)으로 설정
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # propagate
        # BertEmbeddins 모듈
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # BertLayer를 반복하는 BertEncoder 모듈
        if attention_show_flg == True:
            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers, attention_show_flg)
        elif attention_show_flg == False:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers, attention_show_flg)

        # BertPooler 모듈
        # encoder의 마지막 BertLayer의 output을 사용
        pooled_output = self.pooler(encoded_layers[-1])

        # output_all_encoded_layers가 False이면 마지막 BertLayer의 output만 반환
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # attention_show의 경우 마지막 layer의 attention_probs도 반환
        if attention_show_flg == True:
            return encoded_layers, pooled_output, attention_probs
        elif attention_show_flg == False:
            return encoded_layers, pooled_output