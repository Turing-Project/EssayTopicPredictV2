# EssayTopicPredict

![image](https://img.shields.io/badge/License-Apache--2.0-green) ![image](https://img.shields.io/badge/License-MIT-orange)  ![image](https://img.shields.io/badge/License-Anti--996-red)  ![image](https://img.shields.io/badge/pypi-v0.0.1a4-yellowgreen) ![image](https://img.shields.io/badge/stars-%3C%201k-blue) ![image](https://img.shields.io/badge/issues-1%20open-brightgreen)  

通用型高考作文题目预测模型 v1.0 -人工智能框架，仅限交流与科普。


## 项目简介
EssayTopicPredict是基于无监督学习、模式识别与NLP领域的最新模型所构建的生成式考试题目AI框架，目前第一版finetune模型针对高考作文，可以有效生成符合人类认知的文章题目。

| 项目作者        | 主页1           | 主页2  | 主页3 |
| ------------- |:-------------:|:----:|:---:|
| 图灵的猫       | [知乎](https://www.zhihu.com/people/dong-xi-97-29) |[B站](https://space.bilibili.com/371846699) | [Youtube](https://www.youtube.com/channel/UCoEVP6iTw5sfozUGLLWJyDg/featured) |


**致谢**

感谢开源作者[@imcaspar](https://github.com/imcaspar)  提供GPT-2中文预训练框架与数据支持。
<br>

## 框架说明
- [x] 基于哈工大RoBerta-WWM-EXT、Bertopic、GAN模型的高考题目预测AI
- [x] 支持bert tokenizer，当前版本基于clue chinese vocab
- [x] 17亿参数多模块异构深度神经网络，超2亿条预训练数据
- [x] 可结合作文生成器一起使用：[17亿参数作文杀手](https://colab.research.google.com/github/EssayKillerBrain/EssayKiller_V2/blob/master/colab_online.ipynb)
- [x] 端到端生成，从试卷识别到答题卡输出一条龙服务


## 本地环境
* Ubuntu 18.04.2/ Windows10 x86
* Pandas 0.24.2
* Regex 2019.4.14
* h5py 2.9.0
* Numpy 1.16.2
* Tensorboard 1.15.2
* Tensorflow-gpu 1.15.2
* Requests 2.22.0
* OpenCV 3.4.2
* CUDA >= 10.0
* CuDNN >= 7.6.0

## 开发日志

* 2022.04.23 本地Git项目建立
* 2022.05.03 整体模型架构搭建，开始语料收集
* 2022.05.13 数据集清洗、语料处理
* 2022.05.21 Bertopic+DBSCAN聚类算法
* 2022.05.31 RoBerta与摘要模型调整
* 2022.05.30 代码Review与开源发布


## 模型结构
整个框架分为Proprocess、Bert、DNSCAN 3个模块，每个模块的网络单独训练，参数相互独立。

### 1. 例子
高考语文试卷作文题
>![浙江卷](https://images.shobserver.com/img/2020/7/7/37b2224ee3de441a8a040cb4f5576c2d.jpg)


**数据准备**

人民日报、央视新闻、微博客户端、人民网4个主要爬虫渠道，通过不同API进行爬取（时间为过去12个月内）

*修改/train/config.py中train_data_root，validation_data_root以及image_path*

**训练**
```bash
cd train  
python train.py
```

**训练结果**
```python
Epoch 3/100
25621/25621 [==============================] - 15856s 619ms/step - loss: 0.1035 - acc: 0.9816 - val_loss: 0.1060 - val_acc: 0.9823
Epoch 4/100
25621/25621 [==============================] - 15651s 611ms/step - loss: 0.0798 - acc: 0.9879 - val_loss: 0.0848 - val_acc: 0.9878
Epoch 5/100
25621/25621 [==============================] - 16510s 644ms/step - loss: 0.0732 - acc: 0.9889 - val_loss: 0.0815 - val_acc: 0.9881
Epoch 6/100
25621/25621 [==============================] - 15621s 610ms/step - loss: 0.0691 - acc: 0.9895 - val_loss: 0.0791 - val_acc: 0.9886
Epoch 7/100
25621/25621 [==============================] - 15782s 616ms/step - loss: 0.0666 - acc: 0.9899 - val_loss: 0.0787 - val_acc: 0.9887
Epoch 8/100
25621/25621 [==============================] - 15560s 607ms/step - loss: 0.0645 - acc: 0.9903 - val_loss: 0.0771 - val_acc: 0.9888
```


<br>

### 2. 网络结构
#### 2.1 BERT

**Whole Word Masking (wwm)**，暂翻译为`全词Mask`或`整词Mask`，是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。
简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个子词，在生成训练样本时，这些被分开的子词会随机被mask。
在`全词Mask`中，如果一个完整的词的部分WordPiece子词被mask，则同属该词的其他部分也会被mask，即`全词Mask`。

**需要注意的是，这里的mask指的是广义的mask（替换成[MASK]；保持原词汇；随机替换成另外一个词），并非只局限于单词替换成`[MASK]`标签的情况。
更详细的说明及样例请参考：[#4](https://github.com/ymcui/Chinese-BERT-wwm/issues/4)**

同理，由于谷歌官方发布的`BERT-base, Chinese`中，中文是以**字**为粒度进行切分，没有考虑到传统NLP中的中文分词（CWS）。
我们将全词Mask的方法应用在了中文中，使用了中文维基百科（包括简体和繁体）进行训练，并且使用了[哈工大LTP](http://ltp.ai)作为分词工具，即对组成同一个**词**的汉字全部进行Mask。

下述文本展示了`全词Mask`的生成样例。
**注意：为了方便理解，下述例子中只考虑替换成[MASK]标签的情况。**

| 说明 | 样例 |
| :------- | :--------- |
| 原始文本 | 使用语言模型来预测下一个词的probability。 |
| 分词文本 | 使用 语言 模型 来 预测 下 一个 词 的 probability 。 |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |


#### 2.2 DBSCAN

基于密度的噪声应用空间聚类(DBSCAN)是一种无监督的ML聚类算法。无监督的意思是它不使用预先标记的目标来聚类数据点。聚类是指试图将相似的数据点分组到人工确定的组或簇中。它可以替代KMeans和层次聚类等流行的聚类算法。

KMeans vs DBSCAN：
KMeans尤其容易受到异常值的影响。当算法遍历质心时，在达到稳定性和收敛性之前，离群值对质心的移动方式有显著的影响。此外，KMeans在集群大小和密度不同的情况下还存在数据精确聚类的问题。K-Means只能应用球形簇，如果数据不是球形的，它的准确性就会受到影响。最后，KMeans要求我们首先选择希望找到的集群的数量。

另一方面，DBSCAN不要求我们指定集群的数量，避免了异常值，并且在任意形状和大小的集群中工作得非常好。它没有质心，聚类簇是通过将相邻的点连接在一起的过程形成的。

## 中文模型下载
本目录中主要包含base模型，故我们不在模型简称中标注`base`字样。对于其他大小的模型会标注对应的标记（例如large）。

* **`BERT-large模型`**：24-layer, 1024-hidden, 16-heads, 330M parameters  
* **`BERT-base模型`**：12-layer, 768-hidden, 12-heads, 110M parameters  

**注意：开源版本不包含MLM任务的权重；如需做MLM任务，请使用额外数据进行二次预训练（和其他下游任务一样）。**

| 模型简称 | 语料 | Google下载 | 百度网盘下载 |
| :------- | :--------- | :---------: | :---------: |
| **`RBT6, Chinese`** | **EXT数据<sup>[1]</sup>** | - | **[TensorFlow（密码hniy）](https://pan.baidu.com/s/1_MDAIYIGVgDovWkSs51NDA?pwd=hniy)** |
| **`RBT4, Chinese`** | **EXT数据<sup>[1]</sup>** | - | **[TensorFlow（密码sjpt）](https://pan.baidu.com/s/1MUrmuTULnMn3L1aw_dXxSA?pwd=sjpt)** |
| **`RBTL3, Chinese`** | **EXT数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1Jzn1hYwmv0kXkfTeIvNT61Rn1IbRc-o8)**<br/>**[PyTorch](https://drive.google.com/open?id=1qs5OasLXXjOnR2XuGUh12NanUl0pkjEv)** | **[TensorFlow（密码s6cu）](https://pan.baidu.com/s/1vV9ClBMbsSpt8wUpfQz62Q?pwd=s6cu)** |
| **`RBT3, Chinese`** | **EXT数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1-rvV0nBDvRCASbRz8M9Decc3_8Aw-2yi)**<br/>**[PyTorch](https://drive.google.com/open?id=1_LqmIxm8Nz1Abvlqb8QFZaxYo-TInOed)** | **[TensorFlow（密码5a57）](https://pan.baidu.com/s/1AnapwWj1YBZ_4E6AAtj2lg?pwd=5a57)** |
| **`RoBERTa-wwm-ext-large, Chinese`** | **EXT数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1dtad0FFzG11CBsawu8hvwwzU2R0FDI94)**<br/>**[PyTorch](https://drive.google.com/open?id=1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq)** | **[TensorFlow（密码dqqe）](https://pan.baidu.com/s/1F68xzCLWEonTEVP7HQ0Ciw?pwd=dqqe)** |
| **`RoBERTa-wwm-ext, Chinese`** | **EXT数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt)** <br/>**[PyTorch](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)** | **[TensorFlow（密码vybq）](https://pan.baidu.com/s/1oR0cgSXE3Nz6dESxr98qVA?pwd=vybq)** |
| **`BERT-wwm-ext, Chinese`** | **EXT数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi)** <br/>**[PyTorch](https://drive.google.com/open?id=1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_)** | **[TensorFlow（密码wgnt）](https://pan.baidu.com/s/1x-jIw1X2yNYHGak2yiq4RQ?pwd=wgnt)** |
| **`BERT-wwm, Chinese`** | **中文维基** | **[TensorFlow](https://drive.google.com/open?id=1RoTQsXp2hkQ1gSRVylRIJfQxJUgkfJMW)** <br/>**[PyTorch](https://drive.google.com/open?id=1AQitrjbvCWc51SYiLN-cJq4e0WiNN4KY)** | **[TensorFlow（密码qfh8）](https://pan.baidu.com/s/1HDdDXiYxGT5ub5OeO7qdWw?pwd=qfh8)** |
| `BERT-base, Chinese`<sup>Google</sup> | 中文维基 | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) | - |
| `BERT-base, Multilingual Cased`<sup>Google</sup>  | 多语种维基 | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) | - |
| `BERT-base, Multilingual Uncased`<sup>Google</sup>  | 多语种维基 | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) | - |

> [1] EXT数据包括：中文维基百科，其他百科、新闻、问答等数据，总词数达5.4B。

查看更多哈工大讯飞联合实验室（HFL）发布的资源：https://github.com/ymcui/HFL-Anthology

```bash
python run.py --model bert
```
<br>

![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-16-40-19.png)

测试时，需要用正则表达式过滤考试专用词，包括“阅读下面的材料，根据要求写作”，“要求：xxx”，“请完成/请结合/请综合xx”。

比如
>![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-17-17-30.png)


    人们用眼睛看他人、看世界，却无法直接看到完整的自己。所以，在人生的旅程中，我们需要寻找各种“镜子”、不断绘制“自画像”来审视自我，尝试回答“我是怎样的人”“我想过怎样的生活”“我能做些什么”“如何生活得更有意义”等重要的问题。

<br>


## Citation
```
@misc{EssayKillerBrain,
  author = {Turing's Cat},
  title = {Autowritting Ai Framework},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AlanTur1ng/EssayTopicPredict}},
}
```

<br>


## 参考资料  
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
[2] ERNIE: Enhanced Representation through Knowledge Integration  
[3] Fine-tune BERT for Extractive Summarization  
[4] EAST: An Efficient and Accurate Scene Text Detector  
[5] An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition  
[6] Language Models are Unsupervised Multitask Learners  
[7] https://github.com/Morizeyao/GPT2-Chinese  
[8] https://github.com/argman/EAST  
[9] https://github.com/bgshih/crnn  
[10] https://github.com/zhiyou720/chinese_summarizer  
[11] https://zhuanlan.zhihu.com/p/64737915  
[12] https://github.com/ouyanghuiyu/chineseocr_lite  
[13] https://github.com/google-research/bert  
[14] https://github.com/rowanz/grover  
[15] https://github.com/wind91725/gpt2-ml-finetune-  
[16] https://github.com/guodongxiaren/README  
[17] https://www.jianshu.com/p/55560d3e0e8a  
[18] https://github.com/YCG09/chinese_ocr  
[19] https://github.com/xiaomaxiao/keras_ocr  
[20] https://github.com/nghuyong/ERNIE-Pytorch  
[21] https://zhuanlan.zhihu.com/p/43534801  
[22] https://blog.csdn.net/xuxunjie147/article/details/87178774/  
[23] https://github.com/JiangYanting/Pre-modern_Chinese_corpus_dataset  
[24] https://github.com/brightmart/nlp_chinese_corpus  
[25] https://github.com/SophonPlus/ChineseNlpCorpus  
[26] https://github.com/THUNLP-AIPoet/Resources  
[27] https://github.com/OYE93/Chinese-NLP-Corpus  
[28] https://github.com/CLUEbenchmark/CLUECorpus2020  
[29] https://github.com/zhiyou720/chinese_summarizer  


## 免责声明
该项目中的内容仅供技术研究与科普，不作为任何结论性依据，不提供任何商业化应用授权
