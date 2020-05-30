"""Basic Lstm model for the question."""

# Author: Ziqi Yuan <1564123490@qq.com>

import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from Generator.dataset import StringToPaddedIndexesWithLength
from Generator.utils import char_to_idx, idx_to_char, PAD_IDX, EOS


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size=64, lstm_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.embedding = nn.Embedding(input_size + 1, hidden_size, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_with_length):
        x, _ = self._forward(x_with_length)
        return x

    def _forward(self, tokenseq_with_length, lstm_state=None):
        tokenseq = tokenseq_with_length[:, :-1]
        length = tokenseq_with_length[:, -1]
        total_padded_length = tokenseq.size(-1)
        # tokeseq : [batch_size, seq_len]
        # lenght : [batch_size, 1]
        tokenseq = self.embedding(tokenseq)
        # tokenseq : [batch_size, seq_len, embedding_size(这里等于hidden_size)]
        # 核心在与使用了 pack_padded_sequence 解决一个 batch 中长度不一致问题
        tokenseq = pack_padded_sequence(tokenseq, length, batch_first=True)
        tokenseq, lstm_state = self.lstm(tokenseq, lstm_state)
        tokenseq, _ = pad_packed_sequence(tokenseq, batch_first=True, padding_value=PAD_IDX,
                                          total_length=total_padded_length)
        # tokenseq : [batch_size, seq_len, hidden_size]
        tokenseq = self.linear(tokenseq)
        # tokenseq : [batch_size, seq_len, vocab_size]
        tokenseq = tokenseq.view(-1, tokenseq.size(-1))
        # tokenseq : [batch_size * seq_len, vocab_size]
        # lstm_state : (h_state: [1, batch_size, hidden_size], c_state: [1, batch_size, hidden_size])
        return tokenseq, lstm_state

    def generate(self, device, max_len=40):
        index = 0
        generated_str = ''
        # 随机挑选一个开始字母
        selected_char = np.random.choice(list(char_to_idx.keys() - ['', EOS]))
        transformer = StringToPaddedIndexesWithLength(max_len)
        # 全零初始化 lstm 的状态向量
        hidden_size = (self.lstm_layers, 1, self.hidden_size)
        h_n, c_n = torch.zeros(*hidden_size, device=device), torch.zeros(*hidden_size, device=device)
        while selected_char != EOS and index < max_len:
            x = transformer(selected_char).unsqueeze(0).to(device)
            x, (h_n, c_n) = self._forward(x, (h_n, c_n))
            x = x[0]
            distribution = Categorical(logits=x)
            selected_char = idx_to_char[distribution.sample().item()]
            if selected_char == EOS:
                break
            generated_str += selected_char
            index += 1

        return generated_str
