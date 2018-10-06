---
layout: post
title: "关于 python 的 import 导入模块"
description: ""
category:
tags:
---

Python 初学者经常为 `ImportError` 头疼。本文解释这个问题及其解决方法。

模块（module）：包含定义与声明的 Python 文件。

包（package）：一组模块及一个 `__init__.py` 文件。

导入（import）：从某个包或模块导入相应的名称（可以指模块或函数、变量等）。

在模块文件末尾写入 `if __name__ == "__main__": main()`，该文件就可以作为脚本使用，执行 `main()` 函数。这是因为 `__name__` 是当前的作用域（scope）的名称，如果代码是直接运行而不是导入的，`__name__` 就是 `"__main__"`。如果是导入的，`__name__` 就是它的模块名，即文件名（不包含 `.py` 后缀）。


建议提前设计好代码组织逻辑，避免循环导入。

以一个标注工具包为例，其文件组织为：

```    
tagger/
|- __init__.py
|- utils.py
|- main.py  
|- data/
|- model/
|- |- __init__.py
|- |- base_model.py
|- |- nn_model.py
```

在这个包里，我们希望 `nn_model.py` 从 `base_model.py` 导入 `BaseModel` 类，从 `utils.py` 导入 `preprocess` 函数。

有相对导入（relative import）和绝对导入（absolute import）两种方式。

无论是哪种导入方式，都可以用 `import xxx` 导入包、模块，也可以用 `from xxx import yyy` 导入函数、类的定义。

相对导入模块

```python
"""in nn_model.py"""
# import sibling module from current package
from . import base_model  

# import sibling package from parent package
from .. import utils  
```

相对导入函数定义

```python
"""in nn_model.py"""
# import from sibling module
from .base_model import BaseModel  

# import from sibling package
from ..utils import preprocess  
```

绝对导入模块

```python
"""in nn_model.py"""
# import sibling module from current package
from tagger.model import base_model

# import sibling package from parent package
from tagger import utils
```

绝对导入函数定义


```python
"""in nn_model.py"""
# import from sibling module
from tagger.model.base_model import BaseModel

# import from sibling package
from tagger.model.utils import preprocess 
```

如果需要运行 `nn_model.py` 中的 `main()`，该怎么做？

直接运行 `python tagger/model/nn_model.py`，可能会遇到 `ImportError`，是命名空间与作用域的问题：因为直接运行单个文件时，并没有导入所在的包、模块，所以并不知道模块的组织结构，无法从当前模块定位到其他模块。

常见的建议是手动修改 PATH，将每个被导入的模块的目录都加到 `sys.path` 中，使之能够被搜索到。但是这样修改 PATH 并不可靠，不仅会改变搜索目录优先级，还可能引入风险。

一种更安全的方式是不修改 PATH，在顶级目录运行代码。例如，在顶级目录的 `main.py` 中导入或写入有关 `nn_model` 的代码，执行 `python main.py`。

也可以从顶级目录执行 `python -Im tagger.model.nn_model`。按照文档解释，`-m` 会将当前目录加到 `sys.path` 的最前面，然后在 `sys.path` 中搜索该模块并执行；但是 `-I` 会使之隔离运行，不会真正修改 `sys.path`，从而规避上述风险。

最后，为了解这两种导入方式在实践中的使用情况，我查看了 *The Hitchhiker's Guide to Python*推荐的几个代码风格良好的项目，以及我平时常用的开源项目，观察其源代码采用的导入方式。

相对导入：

- Requests 
> https://github.com/requests/requests
- Flask 
> https://github.com/pallets/flask
- Keras 
> https://github.com/keras-team/keras

绝对导入：

- Diamond 
> https://github.com/python-diamond/Diamond
- Werkzeug （在源码注释中强调采用绝对导入）
> https://github.com/pallets/werkzeug
- Tensorflow 
> https://github.com/tensorflow/tensorflow

两种导入方式均有：

- Tablib 
> https://github.com/kennethreitz/tablib

可见两种导入方式都有广泛应用，有可能新的、更复杂的项目会倾向于采用绝对导入。


> 参考：

> - https://stackoverflow.com/questions/72852/how-to-do-relative-imports-in-python
> - https://docs.python.org/3/tutorial/modules.html
> - https://docs.python.org/3/using/cmdline.html