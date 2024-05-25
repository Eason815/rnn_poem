import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
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
    def __init__(self, vocab_length , embedding_dim, pretrained_embeddings_path, vocab):
        super(word_embedding, self).__init__()

        # 加载预训练的词嵌入模型
        word2vec = KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=False)

        # 创建一个形状为(vocab_length, embedding_dim)的零张量
        embedding_matrix = torch.zeros((vocab_length, embedding_dim))
        itos = {i: word for word, i in vocab.items()}  # 创建一个索引到词汇的映射
        # 遍历词汇表，将词嵌入矩阵中对应的行设置为预训练的词嵌入
        for i in range(vocab_length):
            try:
                word = itos[i]
                if word in word2vec:
                    embedding_matrix[i] = torch.tensor(word2vec[word])
            except KeyError:
                continue
        self.word_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

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

        # self.normalzation = nn.BatchNorm1d(self.lstm_dim)
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len,self.word_embedding_dim)
        self.apply(weights_init)

        self.softmax = nn.LogSoftmax(dim = 1)
        self.tanh = nn.Tanh()

    def forward(self,sentence,is_test = False):
        """
        输入数据 -> LSTM -> 全连接层 -> ReLU -> Dropout -> Softmax -> 输出数据
        """
        batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
        output, _ = self.rnn_lstm(batch_input, (torch.zeros(2, 1, self.lstm_dim).to(device), torch.zeros(2, 1, self.lstm_dim).to(device)))
        out = output.contiguous().view(-1,self.lstm_dim)
        out =  F.relu(self.fc(out))
        out = nn.Dropout(0.5)(out)
        out = self.softmax(out)


        if is_test:
            prediction = out[ -1, : ].view(1,-1)    
            output = prediction
        else:
           output = out 
        return output
