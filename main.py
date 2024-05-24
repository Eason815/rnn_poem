import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim
from opencc import OpenCC
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_token = 'G'
end_token = 'E'
batch_size = 64

# 选择模式：1.训练模式 2.生成模式
mode=2

# 注意！模型要与源文本一同对应导入
# 请选择模型(1或2)：
choices=1


def getargs(choices):
    # 自训练模型1
    if choices == 1:
        putmodel=savemodel='./model/rnn_model1'
        dest='./setdata/tang.txt'
        epochs = 11
        return putmodel,savemodel,dest,epochs
    # 自训练模型2    
    elif choices == 2:
        putmodel=savemodel='./model/rnn_model2'
        dest='./setdata/song.txt'
        epochs = 41
        return putmodel,savemodel,dest,epochs
    # 自训练模型3
    elif choices == 3:
        putmodel=savemodel='./model/rnn_model3'
        dest='./setdata/song.txt'
        epochs = 100
        return putmodel,savemodel,dest,epochs
    # 自训练模型4
    elif choices == 4:
        putmodel=savemodel='./model/rnn_model4'
        dest='./setdata/song.txt'
        epochs = 100
        return putmodel,savemodel,dest,epochs
    # 自训练模型...  

    else:
        print("输入错误")
        exit(0)

def process_poems():
    # 数据预处理
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

    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 词频排列
    words, _ = zip(*count_pairs)

    words = words[:len(words)] + (' ',)
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

    if choices == 1 or choices == 2:
        word_embedding = rnn_lstm.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
        rnn_model = rnn_lstm.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)
    elif choices == 3:
        word_embedding = rnn.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
        rnn_model = rnn.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)
    elif choices == 4:
        word_embedding = cnn.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
        rnn_model = cnn.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)

    rnn_model.to(device)
    if choices == 1 or choices == 2:
        optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)# 优化器
    elif choices == 3 or choices == 4:
        optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)# 优化器
    # optimizer = optim.SGD(rnn_model.parameters(), lr=0.001, momentum=0.9)# 优化器
    # optimizer = optim.Adagrad(rnn_model.parameters(), lr=0.01)# 优化器

    loss_fun = torch.nn.NLLLoss() # 损失函数


    for epoch in range(epochs):

        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
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
            loss  = loss  / BATCH_SIZE #计算平均损失吧
            print("epoch  ",epoch,'batch number',batch,"loss is: ", loss.data.tolist())
            optimizer.zero_grad()#梯度归零
            loss.backward()#反向传播
            torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1)#对所有的梯度乘以一个clip_coef，缓解梯度爆炸问题（小于1）
            optimizer.step()#通过梯度下降执行一步参数更新

        if epoch % 5 ==0:
            torch.save(rnn_model.state_dict(), savemodel)#每五个epoch保存一次model
            print("finished save model")

    torch.save(rnn_model.state_dict(), savemodel)#保存最终model
    print(len(word_to_int))


def to_word(predict, vocabs):  # 预测的结果转化成汉字

    sample = np.argmax(predict)
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]



# 作一段诗
def gen_poem(begin_word):
    
    poems_vector, word_int_map, vocabularies = process_poems()

    if choices == 1 or choices == 2:
        word_embedding = rnn_lstm.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
        rnn_model = rnn_lstm.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,embedding_dim=100, lstm_hidden_dim=128)
    elif choices == 3: 
        word_embedding = rnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
        rnn_model = rnn.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,embedding_dim=100, lstm_hidden_dim=128)
    elif choices == 4:
        word_embedding = cnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
        rnn_model = cnn.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,embedding_dim=100, lstm_hidden_dim=128)

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
    
# 格式任意诗句
def pretty_print_poem1(poem1):  # 打印

    #print(poem)

    # 无法生成
    if len(poem1) <= 3:
        print("生成失败，换个字试试吧！") 
        return 

    poem_sentences = poem1.split('。')
    for s in poem_sentences:
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

    print(poem1)

    # 无法生成
    if len(poem1) <= 3:
        print("生成失败，换个字试试吧！") 
        return 

    poem_sentences = poem1.split('。')
    for s in poem_sentences:
        if s != '' and len(s) == 11 :
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
            if s != '' and len(s) == 11 :
                s = OpenCC('t2s').convert(s)
                print(s + '。')


if __name__ == '__main__':

    if choices == 1 or choices == 2:
        import define.rnn_lstm as rnn_lstm
    elif choices == 3:
        import define.rnn as rnn
    elif choices == 4:
        import define.cnn as cnn
    else:
        print("choices输入错误")
        exit(0)

    putmodel,savemodel,dest,epochs=getargs(choices)


    if mode == 1:
        print("start to train model")
        run_training()
        print("finish training")
    elif mode == 2:
        print("start to generate poem")
        print("(仅输入为Enter退出)请给出一个字:",end='')
        while word1 := input():
            if '&' in word1:
                break
            word1 = OpenCC('s2t').convert(word1)
            try:
                if choices == 1:
                    pretty_print_poem1(gen_poem(word1))
                elif choices == 2:
                    pretty_print_poem2(gen_poem(word1))
                elif choices == 3:
                    pretty_print_poem1(gen_poem(word1))
                elif choices == 4:
                    pretty_print_poem1(gen_poem(word1))
            except KeyError as e:
                print("生成失败，换个字试试吧！") 
            print("(仅输入为Enter退出)请给出一个字:",end='')
    else:
        print("输入错误")
        exit(0)
