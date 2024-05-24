import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__  
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class word_embedding(nn.Module):
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))#均匀分布embedding随机初始化
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))#加载已经训练好的词向量
    def forward(self,input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz ,vocab_len ,word_embedding,embedding_dim, lstm_hidden_dim):
        super(RNN_model,self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim

        self.rnn_lstm = nn.LSTM(input_size=self.word_embedding_dim,hidden_size=self.lstm_dim,num_layers=2,batch_first=True)
        self.rnn = nn.RNN(input_size=self.word_embedding_dim,hidden_size=self.lstm_dim,num_layers=2,batch_first=True)    
        # self.gru = nn.GRU(input_size=self.word_embedding_dim,hidden_size=self.lstm_dim,num_layers=2,batch_first=True,bidirectional=True)
        # self.gru2 = nn.GRU(input_size=self.word_embedding_dim,hidden_size=self.lstm_dim,num_layers=2,batch_first=True, bidirectional=True)

        self.conv = nn.Conv1d(self.word_embedding_dim, self.lstm_dim, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(self.word_embedding_dim, self.lstm_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.lstm_dim, self.lstm_dim, kernel_size=3, padding=1)

        self.transformer = nn.Transformer(d_model=self.word_embedding_dim, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.1, activation='relu')

        self.attention = Attention(self.lstm_dim)
        # self.normalization = nn.BatchNorm1d(self.lstm_dim)
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len,self.word_embedding_dim)
        self.apply(weights_init)

        self.softmax = nn.LogSoftmax()
        self.tanh = nn.Tanh()

    def forward(self,sentence,is_test = False):
        """
        输入数据 -> RNN -> 全连接层 -> Normalization -> ReLU -> Softmax -> 输出数据
        """
        batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
        output, _ = self.rnn(batch_input, torch.zeros(2, 1, self.lstm_dim).to(device))


        # 注意力机制
        # 在RNN和全连接层之间添加注意力机制
        # output, hidden = self.rnn(batch_input, torch.zeros(2, 1, self.lstm_dim).to(device))
        # hidden = hidden.expand(-1, -1, output.size(2))
        # attn_weights = self.attention(hidden, output)
        # context = attn_weights.bmm(output.transpose(0, 1))  # (B,1,N)
        # context = context.transpose(0, 1)  # (1,B,N)
        # output = context.squeeze(0)  # (B,N)


        out = output.contiguous().view(-1,self.lstm_dim)
        
        # Normalization
        out =  F.normalize(out, p=2, dim=1)

        out =  F.relu(self.fc(out))
        
        # dropout层
        out = nn.Dropout(0.5)(out)
        out = self.softmax(out)




        if is_test:
            prediction = out[ -1, : ].view(1,-1)    
            output = prediction
        else:
           output = out 
        return output

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.shape[1]
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)), dim=2)
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]




