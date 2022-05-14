# python批量更换后缀名
import datetime
import os
import sys

from tqdm import tqdm

os.chdir(r'D:\桌面\创作\AI预测高考作文\Spider-People-s-daily')

# 列出当前目录下所有的文件
files = os.listdir('./')
print('files', files)

for root, dirs, files in os.walk(r'D:\桌面\创作\AI预测高考作文\Spider-People-s-daily\data'):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>2", root, dirs, files)
    input_col = []
    output_col = []
    count = 0
    bar = tqdm(files,
               total=len(files),
               desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Process NFT, POAP, Donation",
               ncols=150)
    for filename in bar:
        print('filename', filename)
        portion = os.path.splitext(filename)
        os.chdir(root)
        # 如果后缀是.dat
        if portion[1] == ".csv":
            #把原文件后缀名改为 txt
            newName = portion[0] + ".csv"
            os.renames(filename, newName)


### 提取摘要-这部分放在Tokenization函数里
def tmp():
    import jieba,os,re
    from gensim import corpora, models, similarities

    """创建停用词列表"""
    def stopwordslist():
        stopwords = [line.strip() for line in open('../stopwords.txt', encoding='UTF-8').readlines()]
        return stopwords

    """对句子进行中文分词"""
    def seg_depart(sentence):
        sentence_depart = jieba.cut(sentence.strip())
        stopwords = stopwordslist()
        outstr = ''
        for word in sentence_depart:
            if word not in stopwords:
                outstr += word
                outstr += " "
        return outstr

    train = []
    for j,line in tqdm(enumerate(df["摘要"])):
        line = re.sub(r'[^\u4e00-\u9fa5]+','',line)
        line_seg = seg_depart(line.strip())
        line = [word.strip() for word in line_seg.split(' ')]
        train.append(line[:-1])
