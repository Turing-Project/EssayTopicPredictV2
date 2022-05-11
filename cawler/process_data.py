#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("data/文化.csv", encoding="utf-8", usecols=[1,3,5,6])
data.drop_duplicates(subset="标题", inplace=True)
pd.set_option("display.max_colwidth", 100)
data
# data['index']=range(7902)


# In[2]:


data.reset_index().drop('index',axis=1)


# In[3]:


data['id']=range(7902)
data


# In[4]:


data = pd.concat([data, pd.DataFrame(columns=["ORG","PERSON","GPE-P","GPE-C","TOPIC"])])
data = data.fillna("")


# In[5]:


df = data.reset_index().drop(["index","版面"],axis=1)


# In[6]:


df[['id']] = df[['id']].astype(int)


# In[7]:


df.head()


# ### 时间处理函数

# In[8]:


df["DATE"] = pd.to_datetime(df["发表时间"], format="%Y/%m/%d").dt.date


# In[9]:


time_df = df[['id','标题','摘要']]
time_df.to_csv('basic_information.csv')


# ### 地区处理

# In[10]:


import json
with open("CitiesChina.json","r", encoding="UTF-8") as f:
     city_dict = json.load(f)


# In[12]:


province_list = ["河北","山西","辽宁","吉林","黑龙江","江苏","浙江","安徽","福建","江西","山东","河南","湖北","湖南","广东","海南","四川","贵州","云南","陕西","甘肃","青海","台湾","上海","北京","天津","重庆"]


# In[11]:


def find_province(city):
    if city[-1:]=="省":
        return 1,city[:-1]
    if city in province_list:
        return 1,city
    if city[-1:] in ["市","区"]:
        city = city[:-1]
    for loc in city_dict:
        if loc["city"] == city:
            return 2,loc["province"]
    return 0,city


# ### 提取实体

# In[13]:


find_province("郑州")


# In[11]:


import spacy
nlp = spacy.load("zh_core_web_lg")
from tqdm import tqdm
type_list = ["ORG", "PERSON","GPE-P","GPE-C"]
for j,text in tqdm(enumerate(df["摘要"])):
    ent_dict ={"ORG":[],"PERSON":[],"GPE-P":[],"GPE-C":[]}
    doc = nlp(text)
    for token in doc.ents:
        if token.label_ in type_list:
            idx = type_list.index(token.label_)
            ent_dict[token.label_].append(token.text.replace("习近","习近平"))
        elif token.label_ == "GPE":
            # 去中国等字
            if token.text in ["中华人民共和国","中国","中华"]:
                pass
            #找省份
            elif find_province(token.text)[0] == 1:
                ent_dict["GPE-P"].append(find_province(token.text)[1])
                # ent_dict['GPE-C'].append(token.text)
            elif find_province(token.text)[0] == 2:
                ent_dict["GPE-P"].append(find_province(token.text)[1])
                ent_dict["GPE-C"].append(find_province(token.text)[1]+token.text)
                # ent_dict['GPE-C'].append(token.text)
            elif find_province(token.text)[0] == 0:
                ent_dict['GPE-C'].append(token.text)
        for i in type_list:
            ent_dict[i]=list(set(ent_dict[i]))
    df.iloc[j,3]=" ".join(ent_dict["ORG"])
    df.iloc[j,4]=" ".join(ent_dict["PERSON"])
    df.iloc[j,5]=" ".join(ent_dict["GPE-P"])
    df.iloc[j,6]=" ".join(ent_dict["GPE-C"])


# In[14]:


# 写入csv
import spacy
nlp = spacy.load("zh_core_web_lg")
from tqdm import tqdm
with open('location.txt','w',encoding='utf-8') as f:
    f.write('id'+'province'+'\n')
    for j,text in tqdm(enumerate(df["摘要"])):
        j = str(j)
        doc = nlp(text)
        for token in doc.ents:
            if token.label_ =='GPE':
                if token.text in ["中华人民共和国","中国","中华"]:
                    pass
                elif find_province(token.text)[0] == 1:
                    f.write(j+','+find_province(token.text)[1]+'\n')
                # ent_dict['GPE-C'].append(token.text)
                elif find_province(token.text)[0] == 2:
                    f.write(j+','+find_province(token.text)[1]+'\n')
                # ent_dict['GPE-C'].append(token.text)
                elif find_province(token.text)[0] == 0:
                    f.write(j+','+token.text+'\n')


# In[69]:


f =open('people.csv','w',encoding='utf-8')
s = open('org.csv','w',encoding='utf-8')
f.write('id'+','+'people'+'\n')
s.write('id'+','+'org'+'\n')
for j,text in tqdm(enumerate(df["摘要"])):
    j = str(j)
    doc = nlp(text)
    for token in doc.ents:
        if token.label_ =='PERSON':
            f.write(j+','+token.text+'\n')
        if token.label_ =='ORG':
            s.write(j+','+token.text+'\n')
f.close()
s.close()


# ### 提取摘要

# In[27]:


import jieba,os,re
from gensim import corpora, models, similarities

"""创建停用词列表"""
def stopwordslist():
    stopwords = [line.strip() for line in open('./stopwords.txt',encoding='UTF-8').readlines()]
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


# In[28]:


train = []
for j,line in tqdm(enumerate(df["摘要"])):
    line = re.sub(r'[^\u4e00-\u9fa5]+','',line)
    line_seg = seg_depart(line.strip())
    line = [word.strip() for word in line_seg.split(' ')]
    train.append(line[:-1])


# In[31]:


"""构建词频矩阵，训练LDA模型"""
dictionary = corpora.Dictionary(train)
# corpus[0]: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),...]
# corpus是把每条新闻ID化后的结果，每个元素是新闻中的每个词语，在字典中的ID和频率
corpus = [dictionary.doc2bow(text) for text in train]

lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
topic_list = lda.print_topics(10)
print("20个主题的单词分布为：\n")
for topic in topic_list:
    print(topic)


# In[32]:


import pyLDAvis.gensim_models
data = pyLDAvis.gensim_models.prepare(lda, corpus=corpus,dictionary=dictionary)
pyLDAvis.display(data=data)

