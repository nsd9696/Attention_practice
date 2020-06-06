import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, N, lstm_hidden_size=64):
        super(LSTM, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self._lstm = nn.LSTM(batch_first=True, num_layers=N, hidden_size=lstm_hidden_size, input_size=128)
        self._fc = nn.Linear(lstm_hidden_size, 1)
    def forward(self, features):
        result, _ = self._lstm(features)
        print(result.shape)
        result = result.reshape(-1, self.lstm_hidden_size)
        # print(result.shape)
        fc_result = self._fc(result)
        return fc_result

class LSTM2(nn.Module):
    def __init__(self, N, lstm_hidden_size=64):
        super(LSTM2, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self._lstm = nn.LSTM(batch_first=True, num_layers=N, hidden_size=lstm_hidden_size, input_size=128)

    def forward(self, features):
        result, _ = self._lstm(features)
        return result

sample_input = torch.rand(size=(4,5,128))
# print(sample_input)
LSTM_Model = LSTM(3)
result = LSTM_Model(sample_input)
print(result.reshape(4, -1))


import torchtext

# The case above is for, when we don't use the embeddings.
# For example, the comparison will be with just integers (yes/no)

# However, in real sentence, the comparison should be with
# actual embeddings.
