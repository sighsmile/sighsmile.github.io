---
layout: post
title: "今日头条NLP面试半日游"
description: ""
category:
tags:
---

之前参加今日头条技术开放日的活动，对这家公司的氛围颇有好感，因为两个细节：听到员工互相称呼同学，看到技术分享的教室里满满的全是人，很多员工拿着小本本站着听。非常想去成长型的公司，于是当场就投了简历（我是一个随身带纸质版简历的好孩子！），很快约了面试，中途人力又要求更改日期，等待了好久终于等来了这次槽点满满的半日游。

## 一面

一面小哥可能比我还要年轻吧。基本没问什么问题，直接在记事本里做了一道编程题：

给定 100G 文本，存放在 text 变量中。

```
text = "深蓝的天空中挂着一轮金黄的圆月，下面是海边的沙地，都种着一望无际的碧绿的西瓜，其间有一个十一二岁的少年，项带银圈，手捏一柄钢叉，向一匹猹尽力的刺去，那猹却将身一扭，反从他的胯下逃走了。……"
```
sent_split 是标点，sent_end 是标识句尾的标点。

```
sent_split = ['，', '。', '；', '！', '？', '、']
sent_end = ['。', '！', '？']
```

用一个bool变量T表示一个标点是否为句尾标识。

对任意连续的五元组 A B T C D（其中 T 如上定义，A B C D 是汉字），可以定义条件概率 \(P(T|B)\)，\(P(T|C)\), \(P(T|B, C)\) 等等。

本题要求 \(P(T|B, C)\)。进一步，给定测试文本
```
test_text = "眼前展开一片海边碧绿的沙地来[]上面深蓝的天空中挂着一轮金黄的圆月"
```
要求返回[]处的T的概率分布。



简单分析一下，这是一个二元分类问题：给定前后各一个汉字，确定中间的标点是否标识句子的边界。为了简化问题，引入自我感觉很合理的独立性假设：假定标点的性质只与其前后的两个汉字有关，而且前后的两个汉字彼此独立。那么相应的有向图模型为 B -> T <- C，于是 
$$P(T, B, C) = P(B) P(C) P(T|B,C)$$

因此 
$$P(T|B,C) = \frac{P(T,B,C)}{P(B) P(C)}$$

不过，既然求的是条件概率，只需要进行计数就可以了。可以采取合适的平滑措施以解决数据稀疏问题，当然，反正训练文本有 100G，也不会稀疏到哪里去，倒是需要格外注意一下效率问题。实际应该可以拆分成一堆子串分别计数再合并。鉴于没有跟大数据打交道的经验，我不太确定有多必要这样做。

下面代码里的 counterB 和 counterC 对这个问题貌似没什么用。

```
from itertools import islice
from collections import deque, defaultdict

def slide_window(iterable, n):
    """yields a sliding window of width n over iterable
    see: https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
    and https://docs.python.org/release/2.3.5/lib/itertools-example.html
    """
    it = iter(iterable)
    result = deque(maxlen=n)
    result.extend(list(islice(it, n)))
    if len(result) == n:
        yield result    
    for elem in it:
        result.append(elem)
        yield result


def update_counters(text,
        counterB = defaultdict(lambda: [0, 0])),
        counterC = defaultdict(lambda: [0, 0])),
        counterBC = defaultdict(lambda: [0, 0])):
    """update counters
    perhaps divide text into segments and do sth. like map-reduce?
    """
    for B, T, C in slide_window(text, 3):
        if not T in sent_split: continue
        T = int(T in sent_split)
        counterB[B][T] += 1
        counterC[C][T] += 1
        counterBC[tuple(B,C)][T] += 1
    return counterB, counterC, counterTBC

def normalize(lst):
    sum_lst = sum(lst)
    if sum_lst > 0:
        return list(map(lambda x: x / sum_lst, lst))
    else:
        return [1/len(lst)] * len(lst)

def get_conditional_prob(B, C, counterBC):
    if tuple(B, C) in counterBC:
        occurs = counterBC[tuple(B,C)]
        # do some data smoothing stuff here
        return normalize(occurs)
    else:
        # B T C never occurred in text, so fall back to a default value
        # this could be calculated and saved beforehand to make things faster
        total_occurs = [sum(x) for x in zip(*counterBC.values())]
        return normalize(total_occurs)


counterB, counterC, counterBC = update_counters(text)
get_conditional_prob(B, C, counterBC)
```



## 二面

二面，等待时间半小时，实际面试时间五分钟。面试官是个大boss，在打电话的间隙问了我两个问题：

- LSTM 的定义。
- CRF 的 cost function。

虽然我知道 LSTM 是用到 input gate 和 forget gate 对状态进行更新，再用 output gate 对输出进行控制；我也知道 CRF 是对无向图计算条件概率，最大化 log likelihood，并且因为其中的势函数并不是真正的概率分布，所以最后要进行全局归一化。但是这样的程度显然远远不能达到对方的要求。

还是要好好学习啊。
