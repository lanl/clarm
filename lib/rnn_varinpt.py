import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc= nn.Linear(hidden_size, num_classes)
      
    def forward(self, x, lengths, device):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        packed_input = pack_padded_sequence(x, lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        packed_output, (hn,cn) = self.lstm(packed_input,(h0,c0))        
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # out: (Nb,seq_len,hidden_size) --> out:(N, hidden_size) taking last seq value
        lstm_out = lstm_out[:,-1,:]
        linear_out = self.fc(lstm_out)
        #print("linear shape", linear_out.shape)
        
        return linear_out