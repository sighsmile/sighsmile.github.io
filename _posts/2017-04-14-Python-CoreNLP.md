---
layout: post
title: "用 Python 和 Stanford CoreNLP 进行中文自然语言处理"
---


实验环境：Windows 7 / Python 3.6.1 / CoreNLP 3.7.0


## 一、下载 CoreNLP

在 [Stanford NLP 官网](https://stanfordnlp.github.io/CoreNLP/index.html
) 下载最新的模型文件：

- CoreNLP 完整包 [stanford-corenlp-full-2016-10-31.zip](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip)：下载后解压到工作目录。

- 中文模型[stanford-chinese-corenlp-2016-10-31-models.jar](http://nlp.stanford.edu/software/stanford-chinese-corenlp-2016-10-31-models.jar)：下载后复制到上述工作目录。

## 二、安装 stanza

[stanza](https://github.com/stanfordnlp/stanza) 是 Stanford CoreNLP 官方最新开发的 Python 接口。

根据 [StanfordNLPHelp 在 stackoverflow 上的解释](http://stackoverflow.com/questions/43210972/standpostagger-in-python-could-not-find-or-load-main-class/43215707#43215707)，推荐 Python 用户使用 stanza 而非 nltk 的接口。

> If you want to use our tools in Python, I would recommend using the Stanford CoreNLP 3.7.0 server and making small server requests (or using the stanza library).

> If you use nltk what I believe happens is Python just calls our Java code with subprocess and this can actually be very inefficient since distinct calls reload all of the models.

注意 `stanza\setup.py` 文件临近结尾部分，有一行是

    packages=['stanza', 'stanza.text', 'stanza.monitoring', 'stanza.util'],

这样安装后缺少模块，需要手动修改为

    packages=['stanza', 'stanza.text', 'stanza.monitoring', 'stanza.util', 'stanza.corenlp', 'stanza.ml', 'stanza.cluster', 'stanza.research'],


## 三、测试

在CoreNLP工作目录中，打开cmd窗口，启动服务器：

- 如果处理英文，输入    
    `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000`

- 如果处理中文，输入    
    `java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9000 -timeout 15000`    
    
注意stanford-chinese-corenlp-2016-10-31-models.jar应当位于工作目录下。

可在浏览器中键入 http://localhost:9000/ 或 corenlp.run 进行直观测试。

Python示例代码：

```
from stanza.nlp.corenlp import CoreNLPClient
client = CoreNLPClient(server='http://localhost:9000', default_annotators=['ssplit', 'lemma', 'tokenize', 'pos', 'ner']) # 注意在以前的版本中，中文分词为 segment，新版已经和其他语言统一为 tokenize

# 分词和词性标注测试
test1 = "深蓝的天空中挂着一轮金黄的圆月，下面是海边的沙地，都种着一望无际的碧绿的西瓜，其间有一个十一二岁的少年，项带银圈，手捏一柄钢叉，向一匹猹尽力的刺去，那猹却将身一扭，反从他的胯下逃走了。"
annotated = client.annotate(test1)
for sentence in annotated.sentences:
    for token in sentence:
        print(token.word, token.pos)

# 命名实体识别测试
test2 = "大概是物以希为贵罢。北京的白菜运往浙江，便用红头绳系住菜根，倒挂在水果店头，尊为胶菜；福建野生着的芦荟，一到北京就请进温室，且美其名曰龙舌兰。我到仙台也颇受了这样的优待……"
annotated = client.annotate(test2)
for sentence in annotated.sentences:
    for token in sentence:
        if token.ner != 'O':
          print(token.word, token.ner)
```
