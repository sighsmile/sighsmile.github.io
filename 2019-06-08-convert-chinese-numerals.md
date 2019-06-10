---
layout: post
title: "中文与阿拉伯数字的转换"
description: ""
category:
tags:
---

由于语音识别结果或者爬虫结果等等文本的格式不统一，为了后续的抽取或比对，常常需要规范一些内容的呈现格式，例如数字统一转化为阿拉伯数字。

英文和阿拉伯数字的转换比较简单，在 Leetcode 上都有，这里只分享一下中文正整数的转换方法，免得每次使用都重新写一遍。

数字转中文，限定是日常通用领域，比如 `10` 就转换为`十`，不是会计领域的`壹拾零元`，后者其实简单很多，直接一位一位处理就行。前者倒也不难，大体上是四位一组递归处理。

```python
def num_to_zh(num):
    digits = '零一二三四五六七八九'
    scales = [(1, '十'), (2, '百'), (3, '千'), (4, '万'), (8, '亿'), (16, 'X')]
    
    def words(n):
        if n < 10:
            return [digits[n]] if n > 0 else []
        for i, (p, w) in enumerate(scales[:-1]):
            if n < 10 ** (scales[i+1][0]):
                denom = 10 ** p
                zero_fill = '零' if i > 0 and 0 < n%denom < denom//10 else ''
                return words(n//denom) + [w] + [zero_fill] + words(n%denom)
        raise ValueError('输入无法解析')
    return ''.join(words(num)) or '零'
```

中文转数字，同上，只是增加了对“两”的处理，例如`两百零二`可转为`202`。与前一函数不同，这里没有递归，而是用栈的思想。代码写得有点糙，有空可以重构下，目前测试没发现什么错误，不过也不能保证没有漏掉边角用例。

中文的麻烦主要是量级的组合和管辖范围，例如`一万零五亿零一万`，应该是 `一万零五 * 一亿 + 一万`，而不是 `一万 + 五亿 + 一万`。解决了这种样例，其他的应该问题不大了。

```python
def zh_to_num(s):
    digits = {k:i for i,k in enumerate('零一二三四五六七八九')}
    digits['两'] = 2
    scales = {k:i for i,k in [(1, '十'), (2, '百'), (3, '千'), (4, '万'), (8, '亿')]}
    
    res = []
    for c in s:
        if c in digits:
            d = digits[c]
            if d > 0:
                res.append(d)
            else: pass
        elif c in scales:
            d = 10 ** scales[c]
            if scales[c] % 4 == 0:
                if res and res[-1] < d:
                    p = 0
                    while res and res[-1] < d:
                        p += res.pop()
                    res.append(p * d)
                else:
                    res.append(d)
            else:
                if res: 
                    p = res.pop()
                    if p < d:
                        res.append(p * d)
                    else:
                        res.append(p + d)
                else:
                    res.append(d)
        else:
            raise ValueError('输入无法解析')
    return sum(res)
```

最后补充一下，中文的数量级的基数，据中国物理学会的大小数命名法，日常使用的万、亿级别之上，大数还有兆（mega）、京（giga）、垓（tera，读gai1）、秭（peta，读zi3）、穰（exa，读rang2）；小数还有微（micro）、纤（nano，感觉用“纳”的更常见）、沙（pico）、尘（femto）、渺（atto）。以上都是依次递增或递减 1000 倍。

