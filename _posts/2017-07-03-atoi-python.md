---
layout: post
title: "面试题之 atoi 实现"
description: ""
category:
tags:
---

记录一下上次面试，代码题是实现 `atoi()`，字符串转整数。这是一道常见的面试题，考察点不在于算法，而在于讨论交流，确认需求，考虑到所有可能的输入。

按照 C/C++ 的标准，忽略输入开始的空格，可能有正负号，对接下来的连续数字字符进行转换，遇到非数字字符就停止转换。例如输入 `     -30 ` 转换为整数 `-30`，输入 `100m` 转换为整数 `100`。而 Python 则不同，遇到非数字字符时会 `raise ValueError`。

注意这与 LeetCode 第 8 题不同之处，一是异常输入的处理方式，二是溢出。

```
def atoi(s):
  """
  simple implementation of atoi function
  """
  s = s.strip()  # strip whitespaces from both ends

  sign = 1
  i = 0
  if s[0] == '-':
      sign = -1
      i = 1
  elif s[0] == '+':
      i = 1

  res = 0
  while i < len(s):
      d = ord(s[i]) - ord('0')
      if d < 0 or d > 9:
          raise ValueError("invalid literal for atoi(): %s" % s)
          # to follow C/C++ style, just replace the above line with break
      res = res * 10 + d
      i += 1

  return res * sign
```  

上述代码只覆盖了标准的数字格式。当然，数字还有其他常见格式，例如带分隔符（123,456,789）或者科学记数法（4e+10）。
