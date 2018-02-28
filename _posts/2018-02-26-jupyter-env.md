---
layout: post
title: "远程 jupyter notebook 配置"
description: ""
category:
tags:
---

首先 ssh 登录远程服务器。

假设已经安装好了 conda，配置环境，例如事先将下列内容写入配置文件中：

```
name: ml3
dependencies:
- python=3.6
- notebook 
# 以下省略
```

激活环境

```conda env create -f ml3.yml```
```activate ml3```

（在服务器端做各种事情之前可以先托管，避免掉线

```tmux```

重新登录以后，输入`tmux list-sessions`可以查看运行状况，`tmux attach -t [session-id]`就可以重连，`tmux detach`则脱离。）

现在可以做各种事情，比如开启 jupyter notebook

```jupyter notebook --no-browser --port=8888```

然后回到**本地**

```ssh -N -L 8888:localhost:8888 username@remote-server```

（当然本地端口可以不一样啦。）

在本地浏览器打开 ```localhost:8888``` 就可以愉快地玩 jupyter notebook 了。
