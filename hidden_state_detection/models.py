import torch
from torch import nn
import math

class MLPNet(nn.Module):
    def __init__(self, dropout, in_dim=4096, out_dim=2):
        super(MLPNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512), # 4096 * 512
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, out_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class ParallelLinearLayers(nn.Module):
    def __init__(self, input_dim=4096, output_dim=2, num_layers=17):
        super(ParallelLinearLayers, self).__init__()
        self.num_layers = num_layers
        self.linear_layers = nn.ModuleList([MLPNet(input_dim, output_dim) for _ in range(num_layers)])
        self.attn = nn.Linear(num_layers * 2, output_dim)
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        outputs = [self.linear_layers[idx](x[:,idx+15,:]) for idx in range(len(self.linear_layers))]
        # 将多个输出组合成一个张量
        # 例如：可以将它们连接在一起
        output = torch.stack(outputs, dim=1).view(-1, self.num_layers * 2)
        logits = self.attn(output)
        logits == nn.ReLU()
        output = nn.Softmax(-1)(logits)
        return output

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size=(3, 12),stride=1), # (32, 32, 1)->(30, 21, 64)
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64,256,kernel_size=(3, 12),stride=1), # (28, 10, 256)
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2)) # (14, 5, 256)
        self.dense = torch.nn.Sequential(torch.nn.Linear(14*5*256, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(1024, 32),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(32, 2),
                                         torch.nn.Softmax())
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*5*256)
        x = self.dense(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim, 1)) #(512, 1)
        nn.init.xavier_uniform_(self.attention_weights.data)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, num_layers, hidden_dim)
        # 计算注意力分数
        attention_scores = torch.matmul(hidden_states, self.attention_weights).squeeze(-1)
        attention_weights = nn.Softmax(dim=1)(attention_scores)
        # print(attention_weights)
        # 加权求和
        weighted_hidden_states = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return weighted_hidden_states

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2, dropout=0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, _ = self.lstm(x)
        # x = self.dropout(x)
        attn_out = self.attention(x)
        out = self.fc(attn_out)
        # out = self.fc(x[:, -1, :])
        out = self.softmax(out)
        return out
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2), d_model <= 2会报错
        pe[:, 0::2] = torch.sin(position * div_term) # (max_len, d_model / 2)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, layers): # x:(bt_size, seq_len, d_model)
        x += self.pe[layers, :] # (layers, d_model)
        return x
    
class TransforModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_classes=2, dropout=0, need_layers=[]):
        super(TransforModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.need_layers = need_layers
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_embedding = PositionalEncoding(hidden_size, 33)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_embedding(src, self.need_layers)
        output = self.transformer_encoder(src)
        output = self.dropout(output[:, -1, :])
        output = self.fc(output)
        output = self.softmax(output)
        return output

if __name__ == '__main__':
    model = TransforModel(2, 10, 1, num_heads=1, dropout=0.1)
    inputs = torch.randn(3, 3, 2)
    model(inputs, list(range(3)))


