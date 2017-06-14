---
layout: post
title: "Python反转单链表"
description: ""
category:
tags:
---

前一篇回忆了今日头条的面试，今天又看到有人在 [V2EX](https://www.v2ex.com/t/345807) 吐槽之，说面试官无法理解他的递归反转单链表的如下代码：

```
def reverse(head):
    if head==None or head.next==None:
        return head
    result = reverse(head.next)
    head.next.next = head
    head.next = None
    return result
```

说实话，代码确实不难理解，但是这个问题用递归似乎并没有什么特别的优势，因为很容易想到怎样一遍扫描完成。

假设在某一步已经反转了前面若干个结点，它们的新的头结点为 `new_head`，后面剩下若干个结点，它们的头结点为 `head`，则当前这一步操作应该把
```
reversed nodes <- prev_ (new_head) | curr_ (head) -> next_ -> remain_nodes
```
变成
```
reversed nodes <- prev_ <- curr_ (new_head) | next_ (head) -> remain_nodes
```

如果直接修改原链表，代码如下：

```
def reverse_list(head):
    """reverse a linked list in-place"""
    new_head = None  # head of reversed list

    while head:
        new_head, head, head.next = head, head.next, new_head

        # this is equivalent to the following lines:
        # prev_, curr_, next_ = new_head, head, head.next
        # curr_.next = prev_
        # new_head = curr_
        # head = next_

    return new_head
```

如果创建一个新的链表，代码如下

```
from copy import copy

def get_reversed_list(head):
    """return reversed copy of a linked list"""
    new_head = None # head of reversed list

    while head:
        prev_ = new_head
        new_head = copy(head)
        new_head.next = prev_
        head = head.next

    return new_head
```

话说回来，实际工作中从来没有遇到过Python处理单链表的问题，一次也没有！例如这道题的背景是大整数运算，这对Python用户根本就不是什么需要特别处理的事儿。

有意思的是，J.F. Sebastian 在 [Stackoverflow](https://stackoverflow.com/questions/280243/python-linked-list/283630#283630) 给出了函数式的 Python 单链表，可以学习一下：

```
cons   = lambda el, lst: (el, lst)
mklist = lambda *args: reduce(lambda lst, el: cons(el, lst), reversed(args), None)
car = lambda lst: lst[0] if lst else lst
cdr = lambda lst: lst[1] if lst else lst
nth = lambda n, lst: nth(n-1, cdr(lst)) if n > 0 else car(lst)
length  = lambda lst, count=0: length(cdr(lst), count+1) if lst else count
begin   = lambda *args: args[-1]

w = sys.stdout.write
display = lambda lst: begin(w("%s " % car(lst)), display(cdr(lst))) if lst else w("nil\n")
```

当然，双链表还是会用到的，而反转双链表就更简单了……
