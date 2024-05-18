import json
import os
# 数据集(origindata) 主页           https://github.com/chinese-poetry/chinese-poetry
# 数据集(origindata) 本项目使用部分  https://github.com/chinese-poetry/chinese-poetry/tree/master/%E5%85%A8%E5%94%90%E8%AF%97

# 通过dealdata.py处理后数据集(setdata) 

def dest():
    return './setdata/tang.txt'

def data_deal():

    # 清空文件
    with open(dest() , 'w') as f:
        f.write('')

    for i in range(20):

        path='./origindata/poet.tang.'+str(i*1000)+'.json'
        # path = './origindata/唐诗三百首.json'

        # 打开并读取json文件
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 打开test.txt文件，准备写入    w重写，a追加
        with open(dest(), 'a', encoding='utf-8') as f:
            for item in data:
                # 获取"title"和"paragraphs"的值
                title = item.get('title', '')
                paragraphs = ' '.join(map(str, item.get('paragraphs', '')))

                # 将"title"和"paragraphs"的值写入到test.txt文件中
                f.write(title + ':' + paragraphs + '\n')

        # 关闭文件
        f.close()
        print('第'+str(i)+'次写入完成')

if __name__ == '__main__':
    data_deal()
    print('数据处理完成')