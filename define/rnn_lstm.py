import torch.nn as nn
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):# 初始化神经网络中的权重(Xavier初始化或Glorot初始化)
    classname = m.__class__.__name__  
    if classname.find('Linear') != -1:  # 每个模块（m）的类名是否包含'Linear',是则初始化
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
# 训练开始时保持每一层的输入和输出的方差一致，以避免在深度神经网络中出现梯度消失或梯度爆炸的问题。

class word_embedding(nn.Module):# 将词索引poems_vector 转换为 词向量/词嵌入矩阵
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))#均匀分布embedding随机初始化
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        # self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial)) #加载已经训练好的词向量
    def forward(self,input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed
"""
举例 vocab_length =  5 ,embedding_dim = 2
(poems_vector) input_sentence = torch.tensor([1,2,3,4,5])（即一句五个字的诗歌，每个数字代表词汇表中的一个词）
(word_to_int) 词汇表：{'我': 1, '爱': 2, '你': 3, '中国': 4, '人': 5}
[[ 0.1,  0.2],  # '我'的词嵌入向量
 [ 0.3,  0.4],  # '爱'的词嵌入向量
 [ 0.5,  0.6],  # '你'的词嵌入向量
 [ 0.7,  0.8],  # '中国'的词嵌入向量
 [ 0.9,  1.0]]  # '人'的词嵌入向量
"""
# __init__ 方法定义了模型的结构，forward方法定义了模型的计算逻辑。
# 在train中先实例化这些类，传入参数，PyTorch会自动调用该模型的forward方法。


class RNN_model(nn.Module):
    def __init__(self, batch_sz ,vocab_len ,word_embedding,embedding_dim, lstm_hidden_dim):
        super(RNN_model,self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim

        self.rnn_lstm = nn.LSTM(input_size=self.word_embedding_dim,hidden_size=self.lstm_dim,num_layers=2,batch_first=True)

        self.fc = nn.Linear(lstm_hidden_dim, vocab_len,self.word_embedding_dim)
        self.apply(weights_init)

        self.softmax = nn.LogSoftmax(dim = 1)
        self.tanh = nn.Tanh()

    def forward(self,sentence,is_test = False):
        """
        输入数据 -> LSTM -> 全连接层 -> ReLU -> Softmax -> 输出数据
        """
        batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
        output, _ = self.rnn_lstm(batch_input, (torch.zeros(2, 1, self.lstm_dim).to(device), torch.zeros(2, 1, self.lstm_dim).to(device)))
        out = output.contiguous().view(-1,self.lstm_dim)
        out =  nn.functional.relu(self.fc(out))
        out = self.softmax(out)


        if is_test:
            prediction = out[ -1, : ].view(1,-1)    
            output = prediction
        else:
           output = out 
        return output
