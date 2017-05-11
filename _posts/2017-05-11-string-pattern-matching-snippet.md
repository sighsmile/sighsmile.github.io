---
layout: post
title: "简单粗暴进行中文文本匹配的Python代码片段"
description: ""
category:
tags:
---


## 汉字匹配

给定一个字符串（Unicode编码），判断其中是否含有汉字或者是否全为汉字，并提取出其中的汉字。

在处理网络爬取的乱七八糟的原始文本时非常有用。

这段代码只用到初级的正则表达式。

```
import re

hanzi = re.compile(u"[\u4e00-\u9fa5]") #一个汉字
hanzi2 = re.compile(u"[\u4e00-\u9fa5]{2}") #两个汉字
# [\u4e00-\u9fa5] 是汉字的Unicode编码范围
# 注意这里不包括标点符号，标点符号一般会在停用词过滤时去掉

all_hanzi = re.compile(u"^[\u4e00-\u9fa5]+$") #全是汉字
# ^和$分别匹配字符串的开头和结尾

s1 = "这句话全是汉字"
s2 = "这1句话不全是汉字s"
s3 = "this sentence has one 字"
s4 = "this sentence has two 汉字s"
s5 = "this 句 has three 汉字s"

for s in [s1, s2, s3, s4, s5]:
  print(s)
  print(bool(all_hanzi.match(s))) #是否全是汉字
  print(bool(hanzi.search(s))) #是否含有（至少）一个汉字
  print(bool(hanzi2.search(s))) #是否含有（至少）两个汉字
  print(hanzi.findall(s)) #字符串中含有的所有汉字的列表

```

## 词典匹配

给定一个词典（字符串的列表）`words`和一个句子（字符串）`s`，要求判断出`words`中的哪些词出现在`s`中。

当词典较大时，对每个词循环的效率太低，需要采用更高效的数据结构和算法，通常是有穷状态自动机（DFA）或前缀树（trie），使用 Aho–Corasick 算法。

本文使用的Python工具包是 [pyahocorasick](https://github.com/WojciechMula/pyahocorasick)。（另一个常用工具是 esmre，但是似乎很久没有维护，在Windows上安装略麻烦。）

```
pip install pyahocorasick
```


```
import ahocorasick

checkwords = ["英文", "英国", "英国人", "希腊", "瑞典", "瑞士"]
query = "一个英国人在瑞典，一个美国人在丹麦。"

A = ahocorasick.Automaton()
for index, word in enumerate(checkwords):
    A.add_word(word, (index, word))
A.make_automaton()
# A可以用pickle.dump()和load()方法保存到本地并读取

for i in A.iter(query):
    print(i)    
# (3, (1, '英国'))  
# (4, (2, '英国人'))
# (7, (4, '瑞典'))
# 每一项都是 (end_index, value)，其中end_index是匹配到的字符串末尾在query中的位置索引
```
