---
layout: post
title: "Python for Data Analysis 笔记"
description: ""
category:
tags:
---

这是我对 Python for Data Analysis: Data Wrangling with Pandas, Numpy, and IPython 这本书的阅读笔记，只摘录了我不太熟悉的部分，主要是层次索引和时间序列。

如果想要学习数据分析，建议完整阅读这本书并搭配 Kaggle 等数据竞赛实战。

> 该书配套代码见 https://github.com/wesm/pydata-book

该书涵盖的工具包括：Python 3.6, Numpy, pandas, matplotlib/seaborn, IPython (Jupyter), SciPy, scikit-learn, statsmodels.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm

plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

%matplotlib inline # in Jupyter
```

## 离散化技巧

将连续数值划分到不相交区间中，获得 0-1 编码。

```python
np.random.seed(12345)
values = np.random.rand(10)
print(values)
"""
array([0.9296, 0.3164, 0.1839, 0.2046, 0.5677, 0.5955, 0.9645, 0.6532, 0.7489, 0.6536])
"""       

bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
print(pd.cut(values, bins))
"""
[(0.8, 1.0], (0.2, 0.4], (0.0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.4, 0.6], (0.8, 1.0], (0.6, 0.8], (0.6, 0.8], (0.6, 0.8]]
Categories (5, interval[float64]): [(0.0, 0.2] < (0.2, 0.4] < (0.4, 0.6] < (0.6, 0.8] < (0.8, 1.0]]
"""
```

在下面的代码中，如果没有重新指定 labels，列名将是
(0.0, 0.2],  (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]。

```python
pd.get_dummies(pd.cut(values, bins, labels=list('abcde')))
"""
    a   b   c   d   e
0   0   0   0   0   1
1   0   1   0   0   0
2   1   0   0   0   0
3   0   1   0   0   0
4   0   0   1   0   0
5   0   0   1   0   0
6   0   0   0   0   1
7   0   0   0   1   0
8   0   0   0   1   0
9   0   0   0   1   0
"""
```

## 层次索引

Pandas 提供 hierarchical indexing，即同一个 axis 可以有多个层次的索引。

对层次索引，可以按照不同的层次计算 sum 等等，实际机制等价于 groupby。

```python
data = pd.Series(np.random.randn(9),
                 index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                        [1, 2, 3, 1, 3, 1, 2, 2, 3]])
"""
a  1    1.007189
   2   -1.296221
   3    0.274992
b  1    0.228913
   3    1.352917
c  1    0.886429
   2   -2.001637
d  2   -0.371843
   3    1.669025
dtype: float64
"""

data.index
"""
MultiIndex(levels=[['a', 'b', 'c', 'd'], [1, 2, 3]],
           labels=[[0, 0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 0, 2, 0, 1, 1, 2]])
"""

data.loc['b']
"""
1    0.228913
3    1.352917
dtype: float64
"""

data.loc[:, 2]
"""
a   -1.296221
c   -2.001637
d   -0.371843
dtype: float64
"""
```


通过 stack 和 unstack，set_index 和 reset_index，很容易使信息在索引和列之间转换。

```python
data.unstack()
"""
    1           2           3
a   1.007189    -1.296221   0.274992
b   0.228913    NaN         1.352917
c   0.886429    -2.001637   NaN
d   NaN         -0.371843   1.669025
"""
```

```python
data.reset_index()  # opposite of set_index
"""
    level_0 level_1 0
0   a       1       1.007189
1   a       2       -1.296221
2   a       3       0.274992
3   b       1       0.228913
4   b       3       1.352917
5   c       1       0.886429
6   c       2       -2.001637
7   d       2       -0.371843
8   d       3       1.669025
"""
```

通过 pivot 和 melt 可以将数据在 long 和 stacked 格式之间转换。要注意索引的处理。

```python
# 省略预处理步骤

# long format for data with more than one key
ldata
"""
    date        item    value
0   1959-03-31  realgdp 2710.349
1   1959-03-31  infl    0.000
2   1959-03-31  unemp   5.800
3   1959-06-30  realgdp 2778.801
4   1959-06-30  infl    2.340
"""

# stacked format, which is shorter
pivoted = ldata.pivot('date', 'item', 'value')
"""
item        infl    realgdp     unemp
date            
1959-03-31  0.00    2710.349    5.8
1959-06-30  2.34    2778.801    5.1
1959-09-30  2.74    2775.488    5.3
1959-12-31  0.27    2785.204    5.6
1960-03-31  2.31    2847.699    5.2
"""

# ldata.pivot_table(index=['date', 'item']).unstack('item')
unstacked = ldata.set_index(['date', 'item']).unstack('item')
"""
            value
item        infl    realgdp     unemp
date            
1959-03-31  0.00    2710.349    5.8
1959-06-30  2.34    2778.801    5.1
1959-09-30  2.74    2775.488    5.3
1959-12-31  0.27    2785.204    5.6
1960-03-31  2.31    2847.699    5.2
"""

melted = pd.melt(sdata.reset_index(), ['date'])
"""
    date        item    value
0   1959-03-31  infl    0.00
1   1959-06-30  infl    2.34
2   1959-09-30  infl    2.74
3   1959-12-31  infl    0.27
4   1960-03-31  infl    2.31
"""
```

这一部分非常复杂，许多操作都能实现类似的转换。我目前还不清楚它们的差异和适用场景。

## 画图

如果想要产生两行两列子图，以下两种方法是等价的，第一种是分别添加每个 subplot，第二种是一次性创建所有 subplots：

```python
# 分别添加
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
# plt.plot 在当前活跃的子图上绘制，即最近添加的左下角子图
plt.plot(np.random.randn(50).cumsum(), 'k--')

_  = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
```

```python
# 一次性统一创建
fig, axes = plt.subplots(2, 2)

axes[1][0].plot(np.random.randn(50).cumsum(), 'k--')
_  = axes[0][0].hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
axes[0][1].scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
```

可以调整子图之间的距离。但是 matplotlib 不会自动检查标注是否重叠遮挡，需要手动调整。

```python
plt.subplots_adjust(wspace=0, hspace=0)
```

Pandas Series 和 DataFrame 都自带基本的 plot，例如 df.plot.bar() 很方便。但是如果想要对数据整合统计再作图，建议使用 seaborn。


## Split-Apply-Combine

如果想要对数据分组处理，可以使用 df.groupby(keys).apply(func)，它的机制是按照某种原则将元素切分，分别进行处理，将结果拼接在一起。

```python
# select the rows with the largest values in a particular column
def top(df, n=5, columns='tip_pct'):
    return df.sort_values(by=column)[-n:]

# 5 highest tips from smoker and 5 from non-smoker 
tips.groupby('smoker').apply(top)

# highest bill from each of the different combinations of smoker and day
tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')
```

对于缺失的值，可以用数据填充，例如 df.fillna(df.mean())。事实上，可以分组填充。

```python
fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)
```

在采样数据时，也可以分组或分层采样。

```python
# Hearts, Spades, Clubs, Diamonds
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)

deck = pd.Series(card_val, index=cards)
"""
AH    1
2H    2
3H    3
...
JD     10
KD     10
QD     10
"""

def draw(deck, n=5):
    return deck.sample(n)
draw(deck)
"""
5S     5
7C     7
6C     6
JC    10
9H     9
"""

get_suit = lambda card: card[-1]
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)
"""
5C      5
7C      7
JD     10
8D      8
10H    10
6H      6
6S      6
QS     10
"""
```

通过 groupby 可以完成更复杂的任务。

## 时间序列

时间序列数据通常是固定或非固定时间采样的数据，时间信息会表示为时间戳、固定周期或不固定时间间隔之一的形式。

内置的时间处理非常灵活，例如 date_range 的参数 freq 包含非常丰富的选项。

```python
ts = pd.Series(np.random.randn(1000),
               index=pd.date_range('1/1/2000', periods=100, freq='W-MON'))

# convenient indexing 
ts['2001']       # a year
ts['2001-05': '2001-07']  # a range, including 3 months
```

计算相对变化时，可以用 shift 将序列整体前后移动，例如 `ts / ts.shift(1, freq='D') - 1`。

时间戳和周期很容易相互转换。

```python
ts = pd.Series(np.random.randn(3), index=pd.date_range('2000-01-01', periods=3, freq='M'))
ts
"""
2000-01-31   -0.679335
2000-02-29   -0.129697
2000-03-31   -1.930931
Freq: M, dtype: float64
"""

pts = ts.to_period()
pts
"""
2000-01   -0.679335
2000-02   -0.129697
2000-03   -1.930931
Freq: M, dtype: float64
"""

pts.to_timestamp(how='end')
```

时间序列索引可以从列数据中计算出。
```python
index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
```

我们往往希望将时间戳或不固定周期通过整合或补空的方式转换为固定周期的表示。resample 的 API 与 groupby 有些类似。

```python
ts = pd.Series(np.random.randn(100), 
    index=pd.date_range('2000-01-01', periods=100, freq='D'))
ts.resample('M', kind='period').mean()
"""
2000-01   -0.164294
2000-02   -0.112794
2000-03   -0.124408
2000-04   -0.483544
"""
```

时间戳按照固定周期整合，需要指定是左开右闭区间还是左闭右开区间。
金融时间序列数据经常要求对每个区间计算 OHLC 四个值：open（开始）、close（最终）、high（最大）、low（最小）。
不完整的时间戳，上采样补成连续固定周期表示。
这些常用的操作都可以通过 resample 直接实现。

```python
ts.resample('5min', closed='right', label='right').sum()
ts.resample('5min').ohlc()
ts.resample('D').ffill()
```

滑动窗口的 API 同样与 groupby 类似。任何自定义函数只要满足从一个区间返回一个值，就可以用 apply 使用。

```python
# 累积
data.expanding().mean()

# 滑动窗口
data.rolling(60).mean()
data.rolling('20D').mean()

# 指数权重
data.ewm(span=30).mean()

# 自定义函数
from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x, 0.02)
data.rolling(250).apply(score_at_2percent)
```

更复杂的操作可以用 pd.TimeGrouper 完成。

## 函数链

多步操作往往可以合并为一个复合操作，但是可读性也许会下降。

```python
df = load_data()                                        # 1
df2 = df[df['col2'] < 0]                                # 2
df2['col1_demeaned'] = df2['col1'] - df2['col1'].mean() # 3
result = df2.groupby('key').col1_demeaned.std()         # 4

# 3-4 合并为一步
result = (df2.assign(col1_demeaned = df2.col1 - df2.col2.mean())
    .groupby('key').col1_demeaned.std())

# 1-2 合并为一步
df = (load_data()[lambda x: x['col2'] < 0])

# 1-4 合并为一步
result = (load_data()
    [lambda x: x['col2'] < 0]
    .assign(col1_demeaned = df2.col1 - df2.col2.mean())
    .groupby('key').col1_demeaned.std())
```

外部函数或自定义函数也可以用 pipe 合并。

```python
a = f(df, arg1=v1)
b = g(a, v2, arg3=v3)
c = h(b, arg4=v4)

# 如果这些函数都返回 Series 或者 DataFrame，就可以合并
result = (df.pipe(f, arg1=v1)
    .pipe(g, v2, arg3=v3)
    .pipe(h, arg4=v4))
```

使用 pipe，函数的复用变得更简单。

```python
def group_demean(df, by, cols):
    result = df.copy()
    g = df.groupby(by)
    for c in cols:
        result[c] = df[c] - g[c].transform('mean')
    return result
result = (df[df.col1 < 0]
    .pipe(group_demean, ['key1', 'key2'], ['col1']))    
```

## 建模

常用的两个包是 statsmodels 和 scikit-learn。

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# X 是多维数组，y 是一维数组
model = sm.OLS(y, X)
results = model.fit()
results.summary()

data = pd.DataFrame(X, columns=['col0', 'col1', 'col2'])
data['y'] = y
results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
```

时间序列分析，挖掘自回归性质。

```python
MAXLAGS = 5
model = sm.tsa.AR(values)
results = model.fit(MAXLAGS)
results.params
```

推荐阅读

- *Introduction to Machine Learning with Python* by Andreas Mueller and Sarah Guido
- *Python Data Science Handbook* by Jake VanderPlas 
- *Data Science from Scratch: First Principles with Python* by Joel Grus
- *Python Machine Learning* by Sebastian Raschka 
- *Hands-On Machine Learning with Scikit-Learn and TensorFlow* by Aurélien Géron

最后，多读文档！
