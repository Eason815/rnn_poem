import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim
from opencc import OpenCC
import os
import time
import re
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_token = 'G'
end_token = 'E'
batch_size = 64

# 选择模式：1.训练模式 2.生成模式
mode = 2

# 注意！模型要与源文本一同对应导入
# 请选择模型(1-5)：
choices = 4


def getargs(choices):
    project_path = os.path.dirname(os.path.abspath(__file__))
    
    # 自训练模型1
    if choices == 1:
        putmodel=savemodel= project_path + '\\model\\rnn_model1'
        dest= project_path + '\\setdata\\song7.txt'
        epochs = 100
        return putmodel,savemodel,dest,epochs
    # 自训练模型2    
    elif choices == 2:
        embeding_path = project_path + '\\define\\sgns.literature.word'
        putmodel=savemodel= project_path + '\\model\\rnn_model2'
        dest= project_path + '\\setdata\\song7.txt'
        epochs = 10
        return putmodel,savemodel,dest,epochs,embeding_path
    # 自训练模型3
    elif choices == 3:
        putmodel=savemodel= project_path + '\\model\\rnn_model3'
        dest= project_path + '\\setdata\\tang.txt'
        epochs = 11
        return putmodel,savemodel,dest,epochs
    # 自训练模型4
    elif choices == 4:
        putmodel=savemodel= project_path + '\\model\\rnn_model4'
        dest= project_path + '\\setdata\\song7.txt'
        epochs = 41
        return putmodel,savemodel,dest,epochs
    # 自训练模型5
    elif choices == 5:
        embeding_path = project_path + '\\define\\sgns.literature.word'
        putmodel=savemodel= project_path + '\\model\\rnn_model5'
        dest= project_path + '\\setdata\\song5.txt'
        epochs = 20
        return putmodel,savemodel,dest,epochs,embeding_path
    # 自训练模型...

    else:
        print("输入错误")
        exit(0)

def process_poems():
    # 数据预处理
    # 1 文本清洗
    poems = []
    with open(dest, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token 
                poems.append(content)
            except ValueError as e:
                # print("error")
                pass

    poems = sorted(poems, key=lambda line: len(line))
    # 2 词频统计
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 词频排列
    words, _ = zip(*count_pairs) # 
    # 3 词索引映射      "字"--->id(一个整数)
    words = words[:len(words)] + (' ',)         # 诗中所有出现的字，按照它们出现的频率排序
    word_int_map = dict(zip(words, range(len(words))))  # 映射字典
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]# 生成vector 数字id表示
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size #生成batch的number
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
       
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    # 处理数据集
    poems_vector, word_to_int, vocabularies = process_poems()
    # 生成batch
    print("finished loadding data")
    BATCH_SIZE = 100

    torch.manual_seed(5)

    if choices == 3 or choices == 4:
        word_embedding = rnn_lstm.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
        rnn_model = rnn_lstm.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)
    elif choices == 1:
        word_embedding = rnn.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
        rnn_model = rnn.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)
    elif choices == 2:
        word_embedding = cnn.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 300,pretrained_embeddings_path=embeding_path, vocab=word_to_int)
        rnn_model = cnn.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 300, lstm_hidden_dim=128)
    elif choices == 5:
        word_embedding = lstm.word_embedding(vocab_length=len(word_to_int) + 1, embedding_dim=300, pretrained_embeddings_path=embeding_path, vocab=word_to_int)
        rnn_model = lstm.RNN_model(batch_sz=BATCH_SIZE, vocab_len=len(word_to_int) + 1, word_embedding=word_embedding, embedding_dim=300, lstm_hidden_dim=128)

    rnn_model.to(device)
    if choices == 1 or choices == 2:
        optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)# 优化器
    elif choices == 3 or choices == 4 or choices == 5:
        optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)# 优化器
    # optimizer = optim.SGD(rnn_model.parameters(), lr=0.001, momentum=0.9)# 优化器
    # optimizer = optim.Adagrad(rnn_model.parameters(), lr=0.01)# 优化器

    loss_fun = torch.nn.NLLLoss() # 损失函数

    # 在每个epoch开始时记录开始时间
    start_time = time.time()
    for epoch in range(epochs):
        # 在每个epoch开始时重置batch计数器和总时间
        total_batch_time = 0
        batch_count = 0

        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_start_time = time.time()

            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch]
            loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype = np.int64)
                y = np.array(batch_y[index], dtype = np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x,axis=1)))#变量化
                y = Variable(torch.from_numpy(y ))
                x, y = x.to(device), y.to(device)
                pre = rnn_model(x)#这里就进入model，得到预测结果
                loss += loss_fun(pre , y)#与真实label对比得loss
                if index == 0:
                    _, pre = torch.max(pre, dim=1)#输出预测概率最大的那一个word
                    print('prediction', pre.data.tolist()) # 输出预测结果（现在都是数字id形式的）
                    print('b_y       ', y.data.tolist())   # 输出label，真正的古诗，也是数字id形式
                    print('*' * 30)

            loss  = loss  / BATCH_SIZE #计算平均损失
            
            total_batch_time += time.time() - batch_start_time  # 在每个batch结束时更新总时间和batch计数器
            batch_count += 1
            avg_batch_time = total_batch_time / batch_count     # 计算平均每个batch所需的时间
            remaining_time = avg_batch_time * (n_chunk - batch) # 用平均时间乘以剩余的batch数量来估计剩余时间
            m, s = divmod(remaining_time, 60)

            total_epoch_time = time.time() - start_time  # 在每个epoch结束时更新总时间
            progress1=(batch+1)/n_chunk
            avg_epoch_time = total_epoch_time / (epoch + progress1)  # 计算平均每个epoch所需的时间
            remaining_time1 = avg_epoch_time * (epochs - epoch - progress1)  # 用平均时间乘以剩余的epoch数量来估计剩余时间
            m1, s1 = divmod(remaining_time1, 60)
            h1, m1 = divmod(m1, 60)

            print("epoch  ",epoch,'batch number',batch,"loss is: ", loss.data.tolist(), "\033[32m" + "epoch ETA: " + "%02d:%02d" % (m, s) + "\033[0m", "\033[36m" + "ETA: " + "%02d:%02d:%02d" % (h1, m1, s1) + "\033[0m")
            optimizer.zero_grad()#梯度归零
            loss.backward()#反向传播
            torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1)#对所有的梯度乘以一个clip_coef，缓解梯度爆炸问题（小于1）
            optimizer.step()#通过梯度下降执行一步参数更新

        if epoch % 5 ==0:
            torch.save(rnn_model.state_dict(), savemodel)#每五个epoch保存一次model
            print("finished save model")

    torch.save(rnn_model.state_dict(), savemodel)#保存最终model
    print(len(word_to_int))
"""
假设我们有一个样本，其输入是诗歌的前n个词，输出是第n+1个词。
我们首先将输入诗歌的前n个词通过词嵌入层转换为向量，然后将这些向量输入到模型中，得到第n+1个词的预测结果。
然后，我们将这个预测结果和真实的第n+1个词在NLLLoss损失函数下的差异作为损失。

反向传播和参数更新：计算完一个batch的平均损失后，我们将梯度归零，然后进行反向传播，计算每个参数的梯度。然后，我们使用优化器更新这些参数。
"""



def to_word(predict, vocabs):  # 预测的结果转化成汉字

    sample = np.argmax(predict)
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]



# 作一段诗
def gen_poem(begin_word):
    
    poems_vector, word_int_map, vocabularies = process_poems()

    if choices == 3 or choices == 4:
        word_embedding = rnn_lstm.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
        rnn_model = rnn_lstm.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,embedding_dim=100, lstm_hidden_dim=128)
    elif choices == 1: 
        word_embedding = rnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
        rnn_model = rnn.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,embedding_dim=100, lstm_hidden_dim=128)
    elif choices == 2:
        word_embedding = cnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=300, pretrained_embeddings_path=embeding_path, vocab=word_int_map)
        rnn_model = cnn.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,embedding_dim=300, lstm_hidden_dim=128)
    elif choices == 5:
        word_embedding = lstm.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=300, pretrained_embeddings_path=embeding_path, vocab=word_int_map)
        rnn_model = lstm.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,embedding_dim=300, lstm_hidden_dim=128)

    rnn_model.load_state_dict(torch.load(putmodel))#加载模型

    # 指定开始的字
    rnn_model.to(device)
    poem = begin_word
    word = begin_word

    while word != end_token:
        input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
        input = Variable(torch.from_numpy(input)).to(device)
        output = rnn_model(input, is_test=True)
        word = to_word(output.data.tolist()[-1], vocabularies)
        poem += word 

        if len(poem) > 50:#限制长度(一般不会超)
            return poem
        
    return poem
    
# 格式任意诗句(含续写)
def pretty_print_poem1(poem1):  # 打印

    # print(poem1)

    # 无法生成
    if len(poem1) <= 3:
        print("生成失败，换个字试试吧！") 
        return 

    poem_sentences = poem1.split('。')
    for s in poem_sentences:
        # 删除}字符
        if '}' in s:
            s = s.replace('}', '')

        if s != '' and len(s) > 8 :
            s = OpenCC('t2s').convert(s)
            print(s + '。')
        elif s==poem_sentences[0]:
            s = OpenCC('t2s').convert(s)
            print(s)

    #print(len(poem1))

    #续写
    if len(poem1) < 24 and (len(poem1) >= 3 and poem1[-3] != ''):
        #print(poem1[-3])
        poem2 = gen_poem(poem1[-3])
        #print(poem2)
        poem_sentences = poem2.split('。')
        for s in poem_sentences:
            if '}' in s:
                s = s.replace('}', '')
            if s != '' and len(s) > 8 :
                s = OpenCC('t2s').convert(s)
                print(s + '。')

        # # print(len(poem1) + len(poem2))

        # if len(poem1) + len(poem2) < 24 and (len(poem2) >= 3 and poem2[-3] != ''):
        #     poem3 = gen_poem(poem2[-3])
        #     poem_sentences = poem3.split('。')
        #     for s in poem_sentences:
        #         if s != '' and len(s) > 8 :
        #             s = OpenCC('t2s').convert(s)
        #             print(s + '。')

# 七言绝句
def pretty_print_poem2(poem1):  # 打印

    #print(poem)

    # 无法生成
    if len(poem1) <= 3:
        print("生成失败，换个字试试吧！") 
        return 

    poem_sentences = poem1.split('。')
    for s in poem_sentences:
        if s != '' and len(s) == 15 :
            s = OpenCC('t2s').convert(s)
            print(s + '。')
        elif s==poem_sentences[0]:
            s = OpenCC('t2s').convert(s.split('，')[0])
            print(s)

    #print(len(poem1))

    #续写
    if len(poem1) < 24 and (len(poem1) >= 3 and poem1[-3] != ''):
        #print(poem1[-3])
        poem2 = gen_poem(poem1[-3])
        #print(poem2)
        poem_sentences = poem2.split('。')
        for s in poem_sentences:
            if s != '' and len(s) == 15 :
                s = OpenCC('t2s').convert(s)
                print(s + '。')

# 五言绝句
def pretty_print_poem3(poem1):  # 打印

    # print(poem1)

    # 无法生成
    if len(poem1) <= 3:
        print("生成失败，换个字试试吧！") 
        return 

    poem_sentences = re.split('。|，', poem1)

    count = 0
    for s in poem_sentences:
        if s != '' and len(s) == 5:
            s = OpenCC('t2s').convert(s)
            if count % 2 == 0:    
                print(s + '，', end='')
            else:
                print(s + '。')
            count += 1

    #print(len(poem1))

# 格式任意诗句(不含续写)
def pretty_print_poem4(poem1):  # 打印

    # print(poem1)

    # 无法生成
    if len(poem1) <= 3:
        print("生成失败，换个字试试吧！") 
        return 

    poem_sentences = re.split('。|，', poem1)
    for s in poem_sentences:
        # 删除}字符
        if '}' in s:
            s = s.replace('}', '')

        if s != '' and len(s) > 8 :
            s = OpenCC('t2s').convert(s)
            print(s + '。')
        elif s==poem_sentences[0]:
            s = OpenCC('t2s').convert(s)
            print(s)


if __name__ == '__main__':

    if choices == 3 or choices == 4:
        import define.rnn_lstm as rnn_lstm
        putmodel,savemodel,dest,epochs=getargs(choices)
    elif choices == 1:
        import define.rnn as rnn
        putmodel,savemodel,dest,epochs=getargs(choices)
    elif choices == 2:
        import define.cnn as cnn
        putmodel,savemodel,dest,epochs,embeding_path=getargs(choices)
        # 测试embeding_path是否存在
        if not os.path.exists(embeding_path):
            print("预训练词向量不存在，请自行下载sgns.literature.word文件后放入define文件夹")
            exit(0)
    elif choices == 5:
        import define.lstm as lstm
        putmodel,savemodel,dest,epochs,embeding_path=getargs(choices)
        if not os.path.exists(embeding_path):
            print("预训练词向量不存在，请自行下载sgns.literature.word文件后放入define文件夹")
            exit(0)
    else:
        print("choices输入错误")
        exit(0)



    if mode == 1:
        print("start to train model")
        run_training()
        print("finish training")
    elif mode == 2:
        print("start to generate poem")
        print("(仅输入为" + "\033[36m" + "Enter" + "\033[0m" +"退出)请给出一个字:",end='')
        while word1 := input():
            if '&' in word1:
                break
            word1 = OpenCC('s2t').convert(word1)
            try:
                if choices == 1:
                    pretty_print_poem1(gen_poem(word1))
                elif choices == 2:
                    pretty_print_poem4(gen_poem(word1))
                elif choices == 3:
                    pretty_print_poem1(gen_poem(word1))
                elif choices == 4:
                    pretty_print_poem2(gen_poem(word1))
                elif choices == 5:
                    pretty_print_poem3(gen_poem(word1))
            except KeyError as e:
                print("生成失败，换个字试试吧！") 
            print("(仅输入为" + "\033[36m" + "Enter" + "\033[0m" +"退出)请给出一个字:",end='')
    else:
        print("输入错误")
        exit(0)
