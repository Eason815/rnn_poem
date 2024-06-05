import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import os
from opencc import OpenCC

# 使用Python的Tkinter库来创建UI窗口

import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim
import re
import define.rnn_lstm as rnn_lstm
import define.rnn as rnn
import define.cnn as cnn
import define.lstm as lstm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_token = 'G'
end_token = 'E'
batch_size = 64


def getargs(choices):

    project_path = os.path.dirname(os.path.abspath(__file__))
    
    # 自训练模型1
    if choices == 1:
        putmodel=savemodel= project_path + '\\model\\rnn_model1'
        dest= project_path + '\\setdata\\song7.txt'
        epochs = 100
        embeding_path = project_path + '\\define\\sgns.literature.word'
        return putmodel,savemodel,dest,epochs,embeding_path
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
        embeding_path = project_path + '\\define\\sgns.literature.word'
        return putmodel,savemodel,dest,epochs,embeding_path
    # 自训练模型4
    elif choices == 4:
        putmodel=savemodel= project_path + '\\model\\rnn_model4'
        dest= project_path + '\\setdata\\song7.txt'
        epochs = 41
        embeding_path = project_path + '\\define\\sgns.literature.word'
        return putmodel,savemodel,dest,epochs,embeding_path
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

def model_import(choices):
    # 创建一个字典来映射choices的值到对应的参数
    choices_map = {
        1: ('putmodel', 'savemodel', 'dest', 'epochs'),
        2: ('putmodel', 'savemodel', 'dest', 'epochs', 'embeding_path'),
        3: ('putmodel', 'savemodel', 'dest', 'epochs'),
        4: ('putmodel', 'savemodel', 'dest', 'epochs'),
        5: ('putmodel', 'savemodel', 'dest', 'epochs', 'embeding_path')
    }

    # 检查choices的值是否在字典中
    if choices in choices_map:
        # 获取参数
        putmodel, savemodel, dest, epochs, embeding_path = getargs(choices)

        # 如果choices的值是2或5，检查embeding_path是否存在
        if choices in [2, 5] and  not os.path.exists(embeding_path):
            print("预训练词向量不存在，请自行下载sgns.literature.word文件后放入define文件夹")
            exit(0)
    else:
        print("choices输入错误")
        exit(0)

    return putmodel, savemodel, dest, epochs, embeding_path

def use_model(choices,is_train):
    if is_train == 1:
        putmodel, savemodel, dest, epochs, embeding_path = model_import(choices)
        poems_vector, word_to_int, vocabularies = process_poems()
        BATCH_SIZE = 100
        torch.manual_seed(5)
        word_int_map = word_to_int

    elif is_train == 2:
        putmodel, savemodel, dest, epochs, embeding_path = model_import(choices)
        poems_vector, word_int_map, vocabularies = process_poems()

    else:
        print("输入错误")
        exit(0)

    # 创建一个字典来映射choices的值到对应的模块和嵌入维度
    modules = {
        1: ('rnn', 100),
        2: ('cnn', 300),
        3: ('rnn_lstm', 100),
        4: ('rnn_lstm', 100),
        5: ('lstm', 300)
    }
    
    # 检查choices的值是否在字典中
    if choices in modules:
        # 获取对应的模块和嵌入维度
        module, embedding_dim = modules[choices]

        module = __import__('define.' + module, fromlist=[module])
        # 创建词嵌入和模型
        if choices in [2, 5]:
            word_embedding = module.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=embedding_dim, pretrained_embeddings_path=embeding_path, vocab=word_int_map)
        else:
            word_embedding = module.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=embedding_dim)
        
        rnn_model = module.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,embedding_dim=embedding_dim, lstm_hidden_dim=128)
    else:
        print("choices输入错误")
        exit(0)

    return rnn_model

def process_poems():
    putmodel, savemodel, dest, epochs, embeding_path = model_import(getchoices())
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





def to_word(predict, vocabs):  # 预测的结果转化成汉字

    sample = np.argmax(predict)
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]



# 作一段诗
def gen_poem(begin_word):
    choices = getchoices()
    putmodel, savemodel, dest, epochs, embeding_path = model_import(choices)
    poems_vector, word_int_map, vocabularies = process_poems()
    
    rnn_model = use_model(choices,2)
    rnn_model.load_state_dict(torch.load(putmodel))     #加载模型

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

    result = ''
    # print(poem1)

    # 无法生成
    if len(poem1) <= 3:
        result += "生成失败，换个字试试吧！"
        return result

    poem_sentences = poem1.split('。')
    for s in poem_sentences:
        # 删除}字符
        if '}' in s:
            s = s.replace('}', '')

        if s != '' and len(s) > 8 :
            s = OpenCC('t2s').convert(s)
            result += s + '。' + '\n'
        elif s==poem_sentences[0]:
            s = OpenCC('t2s').convert(s)
            result += s + '\n'

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
                result += s + '。' + '\n'

    return result

# 七言绝句
def pretty_print_poem2(poem1):  # 打印

    #print(poem)
    result = ''
    # 无法生成
    if len(poem1) <= 3:
        result += "生成失败，换个字试试吧！"
        return result

    poem_sentences = poem1.split('。')
    for s in poem_sentences:
        if s != '' and len(s) == 15 :
            s = OpenCC('t2s').convert(s)
            result += s + '。' + '\n'
        elif s==poem_sentences[0]:
            s = OpenCC('t2s').convert(s.split('，')[0])
            result += s + '\n'

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
                result += s + '。' + '\n'
    
    return result

# 五言绝句
def pretty_print_poem3(poem1):  # 打印

    # print(poem1)
    result = ''
    # 无法生成
    if len(poem1) <= 3:
        result += "生成失败，换个字试试吧！"
        return result

    poem_sentences = re.split('。|，', poem1)

    count = 0
    for s in poem_sentences:
        if s != '' and len(s) == 5:
            s = OpenCC('t2s').convert(s)
            if count % 2 == 0:    
                result += s + '，'
            else:
                result += s + '。\n'
            count += 1

    #print(len(poem1))
    return result

# 格式任意诗句(不含续写)
def pretty_print_poem4(poem1):  # 打印

    # print(poem1)
    result = ''
    # 无法生成
    if len(poem1) <= 3:
        result += "生成失败，换个字试试吧！"
        return result

    poem_sentences = re.split('。|，', poem1)
    for s in poem_sentences:
        # 删除}字符
        if '}' in s:
            s = s.replace('}', '')

        if s != '' and len(s) > 8 :
            s = OpenCC('t2s').convert(s)
            result += s + '。' + '\n'
        elif s==poem_sentences[0]:
            s = OpenCC('t2s').convert(s)
            result += s + '\n'
    return result


def generate_poem(choice, word):
    # 创建一个字典，键是choices的可能值，值是对应的函数
    func_dict = {
        1: pretty_print_poem1,
        2: pretty_print_poem4,
        3: pretty_print_poem1,
        4: pretty_print_poem2,
        5: pretty_print_poem3
    }

    # 从字典中获取对应的函数并调用
    func = func_dict.get(choice)
    if func:
        return func(gen_poem(word))
    else:
        return '生成失败，换个字试试吧！'


def getchoices():  
    choices = (int)(choice_entry.get())
    return choices



def do_gen():

    choices = (int)(choice_entry.get())
    word = word_entry.get()
    # 生成诗歌
    word1 = OpenCC('s2t').convert(word)
    poem = generate_poem(choices, word1)

    poem+=f"\n\n模型：{choices}词语：{word}\n"
    output_text.config(state='normal')
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, poem)
    output_text.config(state='disabled')

    # 在插入文本后，使用tag_configure和tag_add方法来实现文本居中
    output_text.tag_configure("center", justify='center')
    output_text.tag_add("center", 1.0, "end")


if __name__ == "__main__":


    root = tk.Tk()
    root.title("诗歌生成器")

    # 设置窗口大小
    root.geometry('1000x600')  

    # 设置默认字体大小
    default_font = tkFont.nametofont("TkDefaultFont")
    default_font.configure(size=20)

    # 设置整体背景颜色
    root.configure(bg='#f0f0f0')

    # 设置标签和输入框的样式
    style = ttk.Style()
    style.configure("TLabel", background='#f0f0f0', font=('YaHei', 16))
    style.configure("TButton", font=('YaHei', 16))
    style.configure("TEntry", font=('YaHei', 10))

    # 标签和输入框
    ttk.Label(root, text="模型:").grid(row=0, column=0, padx=20, pady=20, sticky='E')
    choice_entry = ttk.Entry(root, width=20, font=('YaHei', 16))
    choice_entry.grid(row=0, column=1, padx=20, pady=20)
    choice_entry.insert(0, '4')

    ttk.Label(root, text="词语:").grid(row=1, column=0, padx=20, pady=20, sticky='E')
    word_entry = ttk.Entry(root, width=20, font=('YaHei', 16))
    word_entry.grid(row=1, column=1, padx=20, pady=20)
    word_entry.insert(0, '望')

    # 文本框和滚动条
    output_frame = ttk.Frame(root)
    output_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=20)
    output_text = tk.Text(output_frame, width=60, height=10, font=('YaHei', 24))
    output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(output_frame, command=output_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    output_text.config(yscrollcommand=scrollbar.set)
    output_text.config(state='disabled')



    # 生成按钮
    ttk.Button(root, text="生成诗歌", command=do_gen).grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    root.mainloop()

