import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gensim.models import KeyedVectors
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


# class word_embedding(nn.Module):
#     def __init__(self,vocab_length , embedding_dim):
#         super(word_embedding, self).__init__()
#         w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))#均匀分布embedding随机初始化
#         self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
#         self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))#加载已经训练好的词向量

#     def forward(self,input_sentence):
#         """
#         :param input_sentence:  a tensor ,contain several word index.
#         :return: a tensor ,contain word embedding tensor
#         """
#         sen_embed = self.word_embedding(input_sentence)
#         return sen_embed

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

        # self.rnn_lstm = nn.LSTM(input_size=self.word_embedding_dim,hidden_size=self.lstm_dim,num_layers=2,batch_first=True)
        # self.rnn = nn.RNN(input_size=self.word_embedding_dim,hidden_size=self.lstm_dim,num_layers=2,batch_first=True)    
        # self.gru = nn.GRU(input_size=self.word_embedding_dim,hidden_size=self.lstm_dim,num_layers=2,batch_first=True,bidirectional=True)
        # self.gru2 = nn.GRU(input_size=self.word_embedding_dim,hidden_size=self.lstm_dim,num_layers=2,batch_first=True, bidirectional=True)

        # self.conv = nn.Conv1d(self.word_embedding_dim, self.lstm_dim, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(self.word_embedding_dim, self.lstm_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.lstm_dim, self.lstm_dim, kernel_size=3, padding=1)

        self.attention = SelfAttention(self.lstm_dim)

        self.fc = nn.Linear(lstm_hidden_dim, vocab_len,self.word_embedding_dim)
        self.apply(weights_init)

        self.softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self,sentence,is_test = False):
        """
        输入数据 -> 两层卷积层CNN -> 注意力机制 -> 全连接层  -> ReLU/Leaky_Relu -> Dropout -> Softmax -> 输出数据
        """
        batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
        output = self.conv1(batch_input.permute(0,2,1)).permute(0,2,1)
        output = self.conv2(output.permute(0,2,1)).permute(0,2,1)
        output = self.attention(output)        # 在CNN和全连接层之间添加注意力机制
        out = output.contiguous().view(-1,self.lstm_dim)
        out = F.leaky_relu(self.fc(out))
        out = nn.Dropout(0.5)(out)        # dropout层
        out = self.softmax(out)


        if is_test:
            prediction = out[ -1, : ].view(1,-1)    
            output = prediction
        else:
           output = out 
        return output


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, input):
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)

        attention_weights = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.size(-1)), dim=-1)
        output = attention_weights @ v

        return output


