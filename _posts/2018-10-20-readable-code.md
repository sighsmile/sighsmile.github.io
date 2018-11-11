
---
layout: post
title: "《编写可读代码的艺术》笔记"
description: ""
category:
tags:
---

本文是 The Art of Readable Code（编写可读代码的艺术）的读书笔记。

大量的工作是在维护而不是从头开发，因此，本书就抛开项目结构和设计模式，单纯谈如何从小处入手，提高代码的可读性。

这本书的精华在于实例，因此务必仔细阅读原书的例子。

核心原则：代码应当容易看懂。

可读性基本定理：代码应当写得使别人能用最短的时间看懂它。


## 表层结构

### 命名

通过命名，整合信息。

- 选择特定的词汇
    - 例如二叉树的 size 有歧义，不如 height, num_nodes, memory_bytes 明确
- 避免宽泛的名字
    - 例如 i,j 适合用作循环变量，但是如果有嵌套的循环就容易搞混，应该写得更准确，例如 item_i, user_i
- 名字应当具体而不抽象
    - 例如 ServerCanStart() 很抽象，CanListenOnPort() 则更具体
- 用前缀后缀补充重要信息
    - 例如十六进制字符串标识符不妨命名为 hex_id 而不是 id
    - 例如 delay 改为 delay_secs，size 改为 size_mb
- 长度适当
    - 名字的长度限制与 scope 的大小相关
    - 避免看不懂的短名或缩写
- 用格式来传达信息
    - 例如 C++ class 命名格式为 CamelCase，变量名格式为 lower_separated

细查深思，避免误解。

- 例如 clip(text, length) 在文本的尾部截断，有多种理解方法，而 truncate(text, max_chars) 则没有这些歧义
- 例如 in_range(start, stop) 是左闭右开区间还是闭区间不确定，所以应该避免，而 in_range(first, last) 很明确是闭区间，in_range(begin, end) 按惯例表示左闭右开区间
- 布尔变量可以用 is,has,can,should,need 使之作用更明确
- 应该符合预期，例如一般会预设 get_size() 是 O(1) 操作，如果实际是 O(n) 操作就应该使用 compute_ 或者 count_size()

### 审美

- 布局、顺序一致有规律
- 功能相似的代码应该具有视觉相似性
- 按照关联疏密分组形成代码块，用空行隔开。

### 注释

将代码作者所知道的一切传递给读者。

- 不要注释不言自明的信息
- 不要用注释弥补代码的问题，例如难懂的函数名，与其注释不如改正
- 应当注释重要的想法，例如该数据使用二叉树比哈希表快 40%
- 应当注释缺陷，常见的提示词包括 TODO/FIXME/HACK/XXX 等
- 应当注释常量（超参数）的选择理由和限制
- 应当用注释来回答可能的疑问，提示可能的坑
- 应该用 high-level 注释从整体上概括说明一个代码段落或模块的作用

注释应该用最少的空间传递最多的信息。

- 避免 it 之类代词的歧义
- 准确描述函数的行为
    - 例如，返回文件行数（如何定义行？），更准确的说法是返回换行符 `\n` 个数
- 给出特殊的用例
- 说明代码的总体意图，而不是重复描述代码的细节
- 函数调用时，给出参数的关键字（可以用行内注释实现）
- 使用含义丰富的术语来指代常见的情况，例如 canonicalize, naive

## 简化循环和逻辑

### 控制流

循环、条件分支尽可能自然，不要让读者需要停下来回头重读。

- 条件语句中建议左边是变量，右边是更稳定的参照量
- if/else 建议先写更关注的分支
- `cond ? a : b` 只在极其简单的情形下使用
- 不要写 do/while
- 通常不要写 goto 
- 有时可以提前返回值，减少代码嵌套

### 超长表达式

拆分超长表达式，使每一个部分容易理解。

- 解释变量
    - 例如`if line.split(':')[0].strip() == 'root':` 改为 `username = ...; if username == 'root':`
    - 例如代码中多次用到 `request.user.id == document.owner.id`，可以将它定义为 `user_owns_document`
- 将复杂的 if/else 简化为逻辑等价的函数

### 变量

变量越多，作用域越大，改变次数越多，就越难管控它的值。

- 减少变量个数
    - 移除没有复用的临时变量
    - 有时可以立即返回最终结果，从而移除中间结果
- 减小变量作用域
    - 静态方法
    - 注意不同语言的差异
- 减少变量修改次数
    - 使用 immutable 变量
    - 如果赋值之后反复在不同的地方修改值，会导致变量的状态难以跟踪

## 重新组织代码

这部分的总结比较简略，请阅读原文的示例。

### 分离无关的子问题

如果一段代码不是直接服务于当前的项目目标，而是解决一个无关的子问题，将它分离成一个单独的函数。

子问题有时可以写成更加通用、可复用的 util 函数，例如读写文件、处理字符串。

如果现有的接口过于糟糕和复杂，可以定义自己的 wrapper，使它符合需求。

### 一次只做一件事

每个时刻，代码应该专注于完成一项任务。

如果代码难以读懂，列出它在完成的所有任务，将每项任务分离为单独的函数或者类，或者分隔为有序的段落。

### 先用语言描述代码

先用简单的语言描述代码要做什么，注意提到的关键词，然后根据描述来编写代码。

这种方法也称为 rubber ducking：对着橡皮鸭讲解。

### 如何少写代码

可读代码的最佳实践：不写代码。

- 质疑和拆分需求，削减不必要的需求
- 熟悉你用的代码库，经常通读它们的 API

## Selected Topics

### 测试

测试代码应当可读，方便其他人修改和补充。

- 顶层的测试代码应当尽可能简洁，分离无关的细节
- 新的测试用例应当容易添加
- 测试失败报错时，应当提供有用信息
- 测试用例应当简单而充分
- 测试函数命名应当指出被测试的函数或情形

不容易测试的特性容易带来设计上的困难。

- 全局变量
- 依赖于许多外部组件的代码
- 具有不确定行为的代码

容易测试的特性带来设计上的便利。

- 没有内部状态的类
- 只做一件事的函数或者类
- 解耦合，依赖少的代码
- 具有简单、定义完备的接口的函数

但是，不要让测试喧宾夺主，开发才是你的真正目标。


### 案例：设计与实现一种专用数据结构

问题描述：跟踪记录一个网络服务器在最近一分钟和一小时内传输了多少字节。

原书是 C++ 代码，我改用 Python。

首先，运用之前的方法，讨论和完善命名及相关注释。

```python
class MinuteHourCounter(object):
    """track the cumulative counts over the past minute and over the past hour
    e.g. to track recent bandwidth usage"""

    def __init__(self):
        pass

    def add(self, count):
        """add a new data point: count >= 0
        for the next minute, minite_count will be larger by + count
        for the next hour, hour_count will be larger by + count"""
        pass

    def minute_count(self):
        """return the accumulated count over the past 60 seconds"""
        pass
    
    def hour_count(self):
        """return the accumulated count over the past 3600 seconds"""
        pass
```

然后，解决问题。

Naive 方法：用一个 list 记录所有数据，每次对过去指定时间范围内的数据统计求和。显然速度慢，有很多过期作废的数据占据空间。

改进：用两个 list 分别记录最近一分钟、一小时的数据，超出时间范围的数据就移除，这提示我们用 deque。这样会导致最近一分钟的数据重复存放两次，这提示我们将第二个 list 改成不包括第一个 list 的数据。

继续改进：一个 deque 记录最近一分钟的数据，另一个 deque 记录最近一小时（不包括最近一分钟）的数据。

```python
from collections import deque, namedtuple
from time import time

Event = namedtuple('Event', ['count', 'time'])

class MinuteHourCounter(object):
    """track the cumulative counts over the past minute and over the past hour
    e.g. to track recent bandwidth usage"""

    def __init__(self):
        self.minute_events = deque()
        self.hour_events = deque() # only contains elements NOT in minute_events

        self.minute_count = 0
        self.hour_count = 0


    def add(self, count):
        """add a new data point: count >= 0
        for the next minute, minite_count will be larger by + count
        for the next hour, hour_count will be larger by + count"""
        now_secs = time()

        # TODO: popleft()
        self.shift_old_events(now_secs)

        # feed into the minute list
        # (not into the hour list -- that will happen later)
        self.minute_events.append(
            Event(count=count, time=now_secs))

        self.minute_count += count
        self.hour_count += count

    def minute_count(self):
        """return the accumulated count over the past 60 seconds"""
        self.shift_old_events(time())
        return self.minute_count
    
    def hour_count(self):
        """return the accumulated count over the past 3600 seconds"""
        self.shift_old_events(time())
        return self.hour_count

    def shift_old_events(self, now_secs):
        """find and delete old events, and decrease hour_count and minute_count accordingly"""
        minute_ago = now_secs - 60
        hour_ago = now_secs - 3600

        # move events more than 1 minute old from minute list to hour list
        while self.minute_events and self.minute_events[0].time <= minute_ago:
            self.minute_count -= self.minute_events[0].count
            self.hour_events.append(self.minute_events.popleft())

        # remove events more than 1 hour old from hour list
        while self.hour_events and self.hour_events[0].time <= hour_ago:
            self.hour_count -= self.hour_events[0].count
            self.hour_events.popleft()
```

现在，问题已经解决。但是解决方案也存在一些问题。

首先，扩展性不好，如果新需求是返回最近一天（24小时）的计数，就要改动很多代码。`shift_old_events` 函数非常复杂，涉及许多数据的改动。

其次，上面代码的 memory footprint 很大，如果每秒有很多数据进来，存储空间就会大得超出预期。我们希望改进为固定空间，这提示采用 bucket，时间戳取为整数秒。

由于使用了 deque，没有像原书那样重新定义 `ConveyorQueue`。

```python
class TrailingBucketCounter(object):
    """keep counts for the past N buckets of time"""
    def __init__(self, num_buckets, secs_per_bucket):
        """e.g. TrailingBucketCounter(30, 60) tracks the last 30 minute-buckets of time"""
        self.buckets = deque()
        self.total_sum = 0  # sum of all items in buckets
        self.num_buckets = num_buckets
        self.secs_per_bucket = secs_per_bucket
        self.last_update_time = None

    def _shift(self, num_shifted):
        # in case too many item shifted, just clear
        if num_shifted >= self.num_buckets:
            self.buckets.clear()
            self.total_sum = 0
            return

        # push all the needed 0's
        while num_shifted:
            self.buckets.append(0)
            num_shifted -= 1

        # pop all the excess items
        while len(self.buckets) > self.num_buckets:
            self.total_sum -= self.buckets.popleft()

    def _update(self, now):
        current_bucket = now // self.secs_per_bucket
        last_update_bucket = self.last_update_time // self.secs_per_bucket

        self._shift(current_bucket - last_update_bucket)
        self.last_update_time = now

    def add(self, count, now):
        self._update(now)

        if not self.buckets:
            self._shift(1)
        self.buckets[-1] += count
        self.total_sum += count

    def trailing_count(self, now):
        """return the total count over the last num_buckets worth of time"""
        self._update(now)
        return self.total_sum

class MinuteHourCounter(object):

    def __init__(self):
        self.minute_counts = TrailingBucketCounter(num_buckets=60, secs_per_bucket=1)
        self.hour_counts = TrailingBucketCounter(num_buckets=60, secs_per_bucket=60)

    def add(self, count):
        now = int(time())
        self.minute_counts.add(count, now)
        self.hour_counts.add(count, now)

    # TODO: combine the in-progress bucket with a complementary fraction of the oldest bucket to make up for the half-full current bucket
    def minute_count(self):
        now = int(time())
        return self.minute_counts.trailing_count(now)

    def hour_count(self):
        now = int(time())
        return self.hour_counts.trailing_count(now)
```

总结：我们首先考虑 naive 的解决方案，发现设计上的瓶颈和难点：效率，存储空间。然后把问题拆分为子问题，采用更合适的数据结构，有针对性地改进。反复分析设计的缺陷，进一步改进。

事实上，容易推导出：

- hour_count 的时间复杂度，naive 解法为 O(#events-per-hour) 即正比于每小时数据更新次数，另外两种解法都为 O(1) 即常数
- 存储空间，naive 解法无上限，deque 解法为 O(#events-per-hour) 正比于每小时数据更新次数，buckets 解法为 O(#buckets) 也可以认为是常数

但是，buckets 解法由于取整数时间戳，因此它的计数是近似值。思考：如何改进估算？


## 推荐书目

这里直接摘录了原书最后的推荐书目，大部分应该都有中文版。

关于高质量代码

- Code Complete: A Practical Handbook of Software Construction, by S. McConnell
- Refactoring: Improving the Design of Existing Code, by M. Fowler et al.
- The Practice of Programming, by B. Kernighan and R. Pike
- The Pragmatic Programmer: From Journeyman to Master, by A. Hunt and D. Thomas
- Clean Code: A Handbook of Agile Software Craftsmanship, by R. C. Martin

关于各种编程话题：

- JavaScript: The Good Parts, by D. Crockford
- Effective Java, 2nd ed, by J. Bloch
- Design Patterns: Elements of Reusable Object-Oriented Software, by E. Gamma et al.
- Programming Pearls, 2nd ed, by J. Bentley
- High Performance Web Sites, by S. Souders
- Joel on Software: ANd on Diverse and ..., by J. Spolsky

年份更早的好书

- Writing Solid Code, by S. Maguire
- Smalltalk Best Practice Patterns, by K. Beck
- The Elements of Programming Style, by B. Kernighan and P. J. Plauger
- Literate Programming, by D. E. Knuth