# RNN进行文本生成

## 题目简介

自然语言处理（NLP）的文本生成是一门复杂而强大的技术领域，通过深度学习和自然语言处理算法的结合，模型能够理解并生成出与人类语言相近甚至更加自然流畅的文本。这项技术涉及多层次的学习，从单词和语法结构到语义和上下文的综合考量，使得模型能够以多样的方式生成出准确、连贯且富有逼真感的文本内容。文本生成的应用领域极为广泛，包括但不限于自动文案创作、新闻报道、故事和诗歌创作、对话系统、自动摘要生成以及智能客服对话等。这项技术的发展不仅推动了人机交互的智能化进程，还为各行各业提供了强大的语言处理和沟通工具，提高了工作效率、创造了新的应用场景，并推动了智能化系统和服务的不断发展和进步。

在本题目中，我们将使用最全中文诗歌古典文集数据库进行学习，并且使用RNN进行诗歌的生成。



## 项目要求

1.	熟悉各种网络的框架，能熟练使用Pytorch搭建神经网络，了解每一个模块的作用；
1.	熟悉自然语言处理领域中对数据的预处理方法；
1.	要求使用多种网络（RNN、CNN或是它们的变体）实验，记录网络表现情况，并且进行对比；


## 数据集

### 诗歌数据

主页 https://github.com/chinese-poetry/chinese-poetry

- 下载 https://github.com/chinese-poetry/chinese-poetry/tree/master/%E5%85%A8%E5%94%90%E8%AF%97

### 预训练的词嵌入

主页 https://github.com/Embedding/Chinese-Word-Vectors

- 下载 https://pan.baidu.com/s/1ciq8iXtcrHpu3ir_VhK0zg

## 实现过程

### 数据预处理

在dealdata.py文件中

- 定义`deal_data`函数：将.json文件的诗歌数据录入到.txt文件中


### 定义模型

模型:
    
    rnn.py          (基于RNN)
    cnn.py          (基于CNN)
    rnn_lstm.py     (基于LSTM)
    lstm.py         (基于LSTM)

词向量Word2vec:

    sgns.literature.word    (文学作品)


- 定义`weights_init`函数：用于初始化神经网络中的权重

- 定义`word_embedding`类：一个词嵌入模型，它将词的索引转换为词嵌入向量。



RNN 的基本原理是利用神经网络的输出作为下一步的输入，形成一种“循环”的结构，这使得 RNN 能够处理序列数据。然而，传统的 RNN 存在梯度消失和梯度爆炸的问题，这使得它难以处理长序列数据。

CNN（卷积神经网络）：虽然CNN通常用于处理图像数据，但它也可以用于处理序列数据。一维的CNN可以用于处理时间序列数据，或者用于NLP任务中的文本分类、情感分析等。

LSTM 通过引入“门”结构和“记忆细胞”来解决这个问题。门结构可以控制信息的流动，记忆细胞可以存储长期的状态信息。这使得 LSTM 能够更好地处理长序列数据。

### 训练

在main.py文件中

- 定义`getargs`函数：输入文件等参数

- 定义`process_poems`函数：处理数据，映射得到词汇表

- 定义`generate_batch`函数：生成批次数据并处理，计算损失进行反向传播

- 定义`run_training`函数：

    - 初始化词嵌入层和模型
    - 定义优化器     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `RMSprop`/`Adam`
    - 定义损失函数   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       `NLLLoss`(负对数似然损失)
    - 进入训练循环
    - 所有epoch完成后，保存最终的模型状态。

- 定义`to_word`函数：通过词汇表转换

- 定义`gen_poem`函数：调用模型预测，生成诗句

- 定义`pretty_print_poem`函数：优化格式，根据结尾续写等


## 关于优化


###  **Dropout**

尝试调整dropout比例值以改善模型的性能。

ReLU激活函数之后的Dropout层和全连接层之前的Dropout层都是为了防止过拟合

1. ReLU激活函数之后的Dropout层：ReLU激活函数的输出是非负的，因此在ReLU之后添加Dropout层可以随机将一些神经元的输出设置为0，这样可以增加模型的稳健性，使模型对输入的小变化更加鲁棒。同时，这也可以防止模型过度依赖某些特定的神经元，从而防止过拟合。

2. 全连接层之前的Dropout层：全连接层的作用是将前一层的输出映射到新的空间，如果模型过度依赖某些特定的神经元，可能会导致过拟合。在全连接层之前添加Dropout层可以随机将一些神经元的输出设置为0，这样可以防止模型过度依赖某些特定的神经元，从而防止过拟合。





### **优化器**

尝试不同的优化器，如Adam、RMSprop等以改善模型的性能。


1. **随机梯度下降（SGD）**：这是最基本的优化器，它的主要思想是沿着梯度的反方向更新参数以最小化损失函数。SGD的一个主要问题是它可能会陷入局部最优解，而不是全局最优解。

2. **带动量的SGD**：动量可以帮助SGD在相关方向上加速，减少振荡，从而更快地收敛。

3. **Adagrad**：Adagrad会为每个参数保持一个学习率，这使得它能够对稀疏数据进行有效的优化。但是，Adagrad的学习率在训练过程中是单调递减的，这可能会导致训练过早停止。

4. **RMSprop**：RMSprop通过使用一个衰减系数来解决Adagrad学习率过早下降的问题，使得它在非凸优化问题上表现得更好。

5. **Adam**：Adam结合了RMSprop和动量的思想，它既有自适应学习率的特性，又有动量项。Adam通常被认为是一个在许多任务上表现都很好的优化器。


    - Adam一般作为首选，它在许多任务上都表现得相当好。
    - 如果数据稀疏或处理非凸优化问题，RMSprop或Adagrad会更好。
    - 如果损失函数波动较大，SGD更好。




### **激活函数**

尝试使用其他的激活函数以改善模型的性能。


1. **ReLU（Rectified Linear Unit）**：ReLU是最常用的激活函数，特别是在卷积神经网络（CNN）和深度学习模型中。ReLU函数在输入大于0时直接输出该值，在输入小于0时输出0。ReLU函数的优点是计算简单，而且不会出现梯度消失问题。但是，ReLU函数也有所谓的"死亡ReLU"问题，即某些神经元可能永远不会被激活，导致相应的参数无法更新。

2. **Sigmoid**：Sigmoid函数可以将任何输入转换为0到1之间的输出，因此常用于二分类问题的最后一层，表示概率输出。但是，Sigmoid函数在输入值的绝对值较大时，梯度接近0，容易出现梯度消失问题。

3. **Tanh**：Tanh函数的输出范围是-1到1，因此比Sigmoid函数的输出范围更广。Tanh函数在隐藏层中的使用比Sigmoid函数更常见。但是，Tanh函数仍然存在梯度消失问题。

4. **Leaky ReLU**：Leaky ReLU是ReLU的一个变种，解决了"死亡ReLU"问题。Leaky ReLU允许在输入小于0时有一个小的正斜率，而不是完全输出0。

5. **Softmax**：Softmax函数常用于多分类问题的最后一层，可以将一组输入转换为概率分布。



### **网络结构**

尝试使用LSTM或者其他的RNN结构以改善模型的性能。

尝试增加隐藏层的数量或使用更复杂的RNN结构，双向RNN或者堆叠RNN。


1. **双向RNN**：双向RNN可以同时处理过去和未来的信息。在某些任务中，如语言模型或者序列标注，双向RNN可以取得更好的效果。

2. **堆叠RNN**：堆叠RNN是指在一个RNN的输出上再叠加一个RNN。这可以使模型有更深的层次，能够学习到更复杂的模式。

3. **注意力机制**：注意力机制可以让模型在生成输出时，对输入的不同部分赋予不同的注意力。这在一些任务中，如机器翻译或者文本摘要，可以取得更好的效果。

4. **Transformer**：Transformer是一种基于自注意力机制的模型，它在许多NLP任务中都取得了最好的效果。






### **批量归一化**


尝试在全连接层之后添加Batch Normalization(批量归一化)以改善模型的性能。


Batch Normalization的主要思想是对每一层的输入进行归一化处理，使得结果的均值为0，方差为1。这样做的好处是可以防止梯度消失或梯度爆炸问题，使得网络可以使用更高的学习率，从而加速训练过程。

Batch Normalization的操作通常在全连接层或卷积层之后、激活函数之前进行。如下：

1. 计算当前批次数据的均值和方差。
2. 使用均值和方差对当前批次数据进行归一化处理。
3. 对归一化的结果进行缩放和平移，这两个操作的参数是可以学习的。

常用函数包括`torch.nn.BatchNorm1d`、`torch.nn.BatchNorm2d`和`F.normalize`

1. `torch.nn.BatchNorm1d`和`torch.nn.BatchNorm2d`是Batch Normalization的实现，它们在每个mini-batch上计算输入的均值和方差，然后用这些值来对数据进行归一化，使得结果的均值为0，方差为1。这两个函数还包含两个可学习的参数（scale和shift），用于对归一化的结果进行缩放和平移。适合用于深度学习模型

2. `F.normalize`是一个更简单的归一化函数，它直接对输入数据进行归一化，使得结果的L2范数（欧几里得长度）为1。`F.normalize`不计算输入的均值和方差，也不包含任何可学习的参数。这个函数通常用于对特征向量进行归一化，以确保它们在空间中的分布不受原始数据范围的影响。适合用于简单的特征向量归一化




## 效果展示

尝试剔除含有占比过高字的诗歌 并只保留七言绝句（250000余首宋诗->97067首）

### 自训练模型1

RNN模型网络如下 
> rnn.py

    """
    输入数据 -> RNN -> 全连接层 -> Normalization -> ReLU -> Dropout -> Softmax -> 输出数据
    """
    batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
    output, _ = self.rnn(batch_input, torch.zeros(2, 1, self.lstm_dim).to(device))
    out = output.contiguous().view(-1,self.lstm_dim)
    out =  F.normalize(out, p=2, dim=1)        # Normalization
    out =  F.relu(self.fc(out))
    out = nn.Dropout(0.5)(out)        `        # dropout层
    out = self.softmax(out)



包含90000余首七言宋诗

    (仅输入为Enter退出)请给出一个字:林
    林外雨，千尺黄花春已空。
    老来不作一行客，不是高山无处行。
    一年相语不可怜，一雨不是青山月。

    (仅输入为Enter退出)请给出一个字:酒
    酒长春风起一回
    天上一时人不老，一声同上月明中外。
    老来不作两人传，老病如今不能分寞。

    (仅输入为Enter退出)请给出一个字:古
    古今无人
    不如梅子雪中春有，不知不是一年人。
    一枝未觉无多子，一笑何人说一声轻。

    (仅输入为Enter退出)请给出一个字:古
    古今无人语
    语声名无几日还，不知何处更何如明。
    何时一雨不如雪，不是人间无几年微。


### 自训练模型2

CNN在此项目中还是不太合适

使用了预训练的词向量

CNN模型网络如下
> cnn.py

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


包含90000余首七言宋诗

    (仅输入为Enter退出)请给出一个字:林
    林明名禹禹，E
    (仅输入为Enter退出)请给出一个字:百
    百百不，E



### 自训练模型3

> rnn_lstm.py

    """
    输入数据 -> LSTM -> 全连接层 -> ReLU -> Softmax -> 输出数据
    """
    batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
    output, _ = self.rnn_lstm(batch_input, (torch.zeros(2, 1, self.lstm_dim).to(device), torch.zeros(2, 1, self.lstm_dim).to(device)))
    out = output.contiguous().view(-1,self.lstm_dim)
    out =  nn.functional.relu(self.fc(out))
    out = self.softmax(out)

包含20000余首唐诗

    (仅输入为Enter退出)请给出一个字:新
    新春光不可寻
    君不有时意，今日不相思。
    可怜君不见，空有一生情。

    (仅输入为Enter退出)请给出一个字:望
    望见千里春
    何时不可见，今日不可寻。
    一身不可见，今日不可为。

    (仅输入为Enter退出)请给出一个字:照
    照云正秋
    风流落日照，风吹照水空。
    风光飞不可，归去不知知。

    (仅输入为Enter退出)请给出一个字:霜
    霜露滴滴花香
    风流落日照云色，风吹花枝落露香。
    谁家一朝有文子，不知君不可如头。

### 自训练模型4

> rnn_lstm.py

    """
    输入数据 -> LSTM -> 全连接层 -> ReLU -> Softmax -> 输出数据
    """
    batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
    output, _ = self.rnn_lstm(batch_input, (torch.zeros(2, 1, self.lstm_dim).to(device), torch.zeros(2, 1, self.lstm_dim).to(device)))
    out = output.contiguous().view(-1,self.lstm_dim)
    out =  nn.functional.relu(self.fc(out))
    out = self.softmax(out)

包含90000余首七言宋诗

    (仅输入为Enter退出)请给出一个字:寒
    寒日夜声
    不知何处无人语，不见山中一水落。
    一声风雨不可寻，一日一声天地长。

    (仅输入为Enter退出)请给出一个字:亭
    亭下铄
    山中有客无人语，诗在山中有几人。
    老有一身无处有，一时人在白云间。

    (仅输入为Enter退出)请给出一个字:清
    清如玉
    不知身在玉堂山，一日一声天地间。
    人言万古不如在，不见一时无一言。

    (仅输入为Enter退出)请给出一个字:洞 
    洞天龙密云
    老人不作一身语，老身一日无人看。
    君不见天下一尺，一时白日无人愁。

    (仅输入为Enter退出)请给出一个字:寻
    寻老不如君
    人言无人不如归，老身不作人间人。
    君不见不见前日，不知何处不如何。

    (仅输入为Enter退出)请给出一个字:老树
    老树山深目风
    人间有酒不可得，老觉不知君有声。
    不见东风无一语，一时人在白云间。

    (仅输入为Enter退出)请给出一个字:羌笛
    羌笛声清风
    人间万古无人语，一曲清风一水长。
    人言万古不如枯，人间有酒如何人。

### 自训练模型5

使用了预训练的词向量
> lstm.py

    """
    输入数据 -> LSTM -> 全连接层 -> ReLU -> Dropout -> Softmax -> 输出数据
    """
    batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
    output, _ = self.rnn_lstm(batch_input, (torch.zeros(2, 1, self.lstm_dim).to(device), torch.zeros(2, 1, self.lstm_dim).to(device)))
    out = output.contiguous().view(-1,self.lstm_dim)
    out =  F.relu(self.fc(out))
    out = nn.Dropout(0.5)(out)
    out = self.softmax(out)

包含50000余首五言宋诗

    (仅输入为Enter退出)请给出一个字:春
    春月落飞飞，青山无古尘。
    落日无余子，谁家日上真。
    飞落青山下，春霜入晚回。

    (仅输入为Enter退出)请给出一个字:林
    山灵无远远，水上水中低。
    古石犹无路，春霜日上青。
    春日无时看，古石如飞影。

    (仅输入为Enter退出)请给出一个字:晓
    青春上上龙，孤城无远后。
    春日日前春，古石犹无日。
    春余水上尘，春霜如石路。
    岁月入秋霜，谁家非下远。

    (仅输入为Enter退出)请给出一个字:齐
    齐家犹断落，春日上山门。
    无时非古时，欲见古中春。
    谁家无远春，大子日明月。

> **导入使用模型时应注意：模型要与源文本一同对应导入**

效果仅供参考

模型3,4为固定输出 

其余模型均为随机输出

## 总结

### 模型实现

1. RNN模型（rnn.py）

输入数据 -> RNN -> 全连接层 -> Normalization -> ReLU -> Dropout -> Softmax -> 输出数据

使用90000余首七言宋诗进行训练

效果：生成的诗句基本连贯，但有时语义上不够准确

2. CNN模型（cnn.py）

输入数据 -> 两层卷积层 -> 注意力机制 -> 全连接层 -> Normalization -> ReLU/Leaky_Relu -> Dropout -> Softmax -> 输出数据

使用90000余首七言宋诗进行训练，预训练的词向量

效果：生成的诗句不够连贯，存在乱码

3. LSTM模型（rnn_lstm.py 和 lstm.py）

输入数据 -> LSTM -> 全连接层 -> ReLU -> Dropout -> Softmax -> 输出数据

使用20000余首唐诗和90000余首七言宋诗进行训练

效果：生成的诗句较为连贯，语义也较准确

#### 尝试

4. GRU（Gated Recurrent Unit）

简化结构：GRU 是 LSTM 的简化版本，只有两个门（更新门和重置门）。它保留了 LSTM 的一些优点，但结构更简单，计算效率更高。

性能表现：GRU 的表现与 LSTM 相当，但由于其更简化的结构，计算速度更快，适用于一些对计算效率要求较高的应用场景。

推荐使用RNN,GRU,LSTM系列模型。

### 优化与改进
1. Dropout：通过添加Dropout层来防止过拟合。
1. 优化器：尝试了RMSprop和Adam等优化器以改善模型性能。
1. 激活函数：尝试了ReLU和Leaky ReLU等激活函数。
1. 网络结构：尝试引入双向RNN、堆叠RNN和注意力机制等更复杂的结构。
1. 批量归一化：在全连接层之后添加Batch Normalization以改善模型的性能。


### 实现效果
通过对比不同模型的生成效果，可以得出以下结论：

1. RNN模型：适合处理短文本生成任务，但在长文本生成方面可能会遇到梯度消失问题。
2. CNN模型：在文本生成任务中表现不如RNN和LSTM，适合处理图像和短序列数据。
3. LSTM模型：在处理长序列数据时表现优异，生成的诗句较为连贯，语义准确，是中文古诗词生成任务中的较优选择。


### 结言

本项目通过对比不同神经网络模型生成古诗词的效果，发现LSTM模型在语义连贯性和准确性上表现最佳。项目展示了深度学习在自然语言处理和文本生成领域的潜力，并为进一步的研究和应用提供了重要参考。

## 实验环境


### Python 

- 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]

### Torch

- 2.2.1+cu121

    * Device 0: NVIDIA GeForce RTX 4060 Laptop GPU &nbsp;&nbsp; 8.0GB
    * Compute capability: 8.9


## 参考文献


基于项目 https://github.com/nndl/exercise/blob/master/chap6_RNN

- 改进数据预处理部分

- 调整训练参数进行优化

- 自行训练词汇表模型

- 整理数据结果输出


### 署名

22软工lxx
