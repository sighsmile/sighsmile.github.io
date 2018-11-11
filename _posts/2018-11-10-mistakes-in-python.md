
根据 *How to Make Mistakes in Python* 一书，以及我自己的经验，总结了 Python 新手容易踩的坑。

## 配置

直接 `pip install` 把包安装在 `site-packages` 目录下，容易导致冲突。一定要养成配置虚拟环境的习惯，例如 `virtualenv`，不过我更习惯用 conda（在这之前我也搞坏过别人的环境……）。

不要用自带的 REPL，建议改用 IPython/Jupyter Notebook。

## 手滑

写函数的时候不要忘记返回值。关于测试，不要偷懒。

使用支持自动补全的编辑器，最好支持代码检查，例如未定义、未使用变量，从而避免手滑带来的难以发觉的错误。

## 风格

没有必要用匈牙利命名法对每个变量写出它的类型，这对代码逻辑没有实质帮助，反而会把变量名搞得很长。关于命名和其他代码习惯问题，我推荐阅读 The Art of Readable Code（编写可读代码的艺术）。

了解 PEP-8。

不要滥用 `lambda` 语句。

## 结构

如果有类似 `switch case` 的需求，用过多的 `if/elif/else` 可能看起来混乱。建议用字典映射到所采取的选择，或者采用面向对象的方式。

没有必要像 Java 那样写一大堆 `getter/setter`。如果想要私有属性，善用 `@property`：

```python
@property
def event_number(self):
    return self._event_number

@event_number.setter
def _set_event_number(self, x):
    self._event_number = int(x)

@event_number.deleter
def _delete_event_number(self):
    self._event_number = None

event.event_number = 10    
print(event.event_number)
```

遵循 The Law of Demeter，减少不必要的耦合，更不要将耦合隐藏在深处。

不要滥用全局状态。

## 意外

不要 `from some_module import *`，否则你无法掌控 import 了哪些东西。一个折中的方法是在 some_module 里面定义 `__all__ = ['foo', 'bar', ...]`，这样 `import *` 就只会引入它们。不过根本上还是建议彻底摒弃 `import *`。

不要在 `import` 的时候做高代价、高依赖的事情，例如在 `__init__.py` 或者模块的开头部分从数据库读入，后果是慢而且容易出错，导致模块的其他函数也无法 `import`。

不要滥用 `try: .... except: pass`，因为这样会导致所有的错误（及其发生的上下文）都无声无息掩盖过去。正确的做法是指明 `except Exception` 或者 `except IOError`，并且记录 `logging.exception('message')`。

造轮子之前，查找标准库、PyPI 和 GitHub。

警惕 mutable 变量作为缺省的关键字参数，非常不安全，例如

```python
def set_reminders(self, event, reminders=[]):
    reminders.extend(self.get_default_reminders())
    for reminder in reminders:
        ...
```

这样写的结果是多次调用之后 `reminders` 变得非常长，每次调用它的信息都积累下来，非常可能导致隐私泄露。这是因为列表是 mutable 变量，同一个 instance 贯穿始终。应该采用 immutable 变量，保证每次调用都是一个新的 instance，例如

```python
def set_reminders(self, event, reminders=None):
    reminders = reminders or []
    # 或者 reminders = [] if reminders if None else reminders
    reminders.extend(self.get_default_reminders())
    for reminder in reminders:
        ...
```

重视 logging，重视测试，不要找借口偷懒。