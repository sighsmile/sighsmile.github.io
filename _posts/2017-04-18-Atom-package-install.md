---
layout: post
title: "Atom手动安装插件和模块的解决方案"
description: ""
category:
tags:
---

**更新：经测试，以管理员身份运行Atom即可在编辑器内直接安装插件，无需手动下载安装。**


最近开始使用Atom编辑器写作。为了预览带LaTeX公式的markdown文档，尝试安装插件markdown-preview-plus，但是总是失败。经过仔细查看错误输出和网上相关问答，发现尽管报错为`Compiler tools not found`，真实原因其实是网络不通畅（GFW）。由于无法使用代理上网，最后只能手动安装完成，摸索过程记录如下。

简单摘要：从github下载并手动安装插件；用nrm测试和切换npm源；用npm-install-missing批量安装模块。

## 安装插件

首先找到该package的代码库。

- https://atom.io/packages/markdown-preview-plus
- https://github.com/atom-community/markdown-preview-plus

下载zip文件，并解压到 `Users\..\.atom\packages\` 文件夹中。

在 cmd 下切换到该目录，执行 `apm install markdown-preview-plus`。

此时仍会报错，并且打开Atom后会出现 `Cannot find module fs-plus` 错误信息，但是已经可以在`Packages`菜单下找到该插件。

这是因为手动安装时，只安装了这个包，没有安装它的若干依赖模块。

## 安装缺失模块

首先安装 [node.js](https://nodejs.org/)。

如果想手动安装fs-plus这一个模块，可以在上述markdown-preview-plus的目录下执行 `npm install fs-plus`。但是这样安装完成之后还会源源不断地提示缺少其他模块……由于 markdown-preview-plus 这个包的依赖模块比较多，不能一个一个手动安装，最好借助其他工具批量安装。

为了批量安装所有依赖模块，首先安装 npm-install-missing 工具，即执行 `npm install -g npm-install-missing`。

然后在markdown-preview-plus目录里执行 `npm-install-missing`。

这一步再次报错 `Registry returned 404 for GET on....`，同样是因为墙的缘故。解决方案是切换源。

首先执行 `npm install -g nrm`，这是源的管理工具。安装成功后，便可以用 `nrm ls` 列出可选源，`nrm test` 测试连接时间，方便地在不同源之间切换。测试结果显示，我这里最快的源是 cnpm，于是执行 `nrm use cnpm` 来切换到它。

最后，再次 `npm-install-missing`，这次安装成功！

现在可以打开 Atom 编辑器，停用默认的 markdown-preview 插件，启用 markdown-preview-plus，通过快捷键 `Ctrl+Shift+m` 切换源文件和预览窗口。


