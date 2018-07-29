---
layout: post
title: "向量、矩阵的求导计算方法"
description: ""
category:
tags:
---

本文简单介绍一下向量、矩阵的求导计算方法。

首先约定记号：

$x$ 和 $y$ 是标量；$\mathbf{x}$ 是 $n$ 维向量，$\mathbf{y}$ 是 $m$ 维向量（本文的向量默认指的是列向量）；$\mathbf{X}$ 表示矩阵。

另外，约定本文使用 numerator layout，具体解释见下文，或 [Wikipedia](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions)。

## 求导法则

### 对标量 $x$ 求导

$$\frac{\partial{\mathbf{y}}}{\partial{x}} = 
\begin{pmatrix} 
\frac{\partial{y_1}}{\partial{x}} \\
\vdots  \\
\frac{\partial{y_m}}{\partial{x}} 
\end{pmatrix}
$$

$$\frac{\partial{\mathbf{Y}}}{\partial{x}} = 
\begin{pmatrix} 
\frac{\partial{y_{11}}}{\partial{x}} & \cdots & \frac{\partial{y_{1n}}}{\partial{x}} \\
\vdots & \ddots & \vdots \\
\frac{\partial{y_{m1}}}{\partial{x}} & \cdots & \frac{\partial{y_{mn}}}{\partial{x}} 
\end{pmatrix}
$$ 

上式又称为 tangent matrix。


### 对向量 $\mathbf{x}$ 求导

$$\frac{\partial{y}}{\partial{\mathbf{x}}} = 
\begin{pmatrix} 
\frac{\partial{y}}{\partial{x_1}} &
\cdots  &
\frac{\partial{y}}{\partial{x_n}} 
\end{pmatrix}
$$


$$\frac{\partial{\mathbf{y}}}{\partial{\mathbf{x}}} = 
\begin{pmatrix} 
\frac{\partial{y_{1}}}{\partial{x_1}} & \cdots & \frac{\partial{y_{1}}}{\partial{x_n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial{y_{m}}}{\partial{x_1}} & \cdots & \frac{\partial{y_{m}}}{\partial{x_n}} 
\end{pmatrix}
\equiv \frac{\partial{\mathbf{y}}}{\partial{\mathbf{x}}^{\top}}
$$

上式又称为
Jacobian matrix。



### 对矩阵 $\mathbf{X}$ 求导

$$\frac{\partial{y}}{\partial{\mathbf{X}}} = 
\begin{pmatrix} 
\frac{\partial{y}}{\partial{x_{11}}} & \cdots & \frac{\partial{y}}{\partial{x_{m1}}} \\
\vdots & \ddots & \vdots \\
\frac{\partial{y}}{\partial{x_{m1}}} & \cdots & \frac{\partial{y}}{\partial{x_{mn}}} 
\end{pmatrix}
\equiv \frac{\partial{y}}{\partial{\mathbf{X}}^{\top}}
$$ 

上式又称为 gradient matrix。

### numerator layout vs. denominator layout

我们之前曾经约定本文使用 numerator layout，这指的是 $\frac{\partial{\mathbf{y}}}{\partial{\mathbf{x}}}$ 矩阵的两种写法之一：例如 $\mathbf{y}$ 是 $m$ 维向量，$\mathbf{x}$ 是 $n$ 维向量（这里都是指的列向量），则该矩阵有两种写法

- $m \times n$ 矩阵，即 $\frac{\partial{\mathbf{y}}}{\partial{\mathbf{x^{\top}}}}$ ，分子不变，分母转置，是 numerator layout；

- $n \times m$ 矩阵，即 $\frac{\partial{\mathbf{y^{\top}}}}{\partial{\mathbf{x}}}$，分母不变，分子转置，是 denominator layout。

对变量是标量、向量、矩阵的情况，都有类似的约定。

很多人倾向于选择 numerator layout，主要是因为 $\frac{\partial{\mathbf{Y}}}{\partial{x}}$ 的自然写法与这种格式是一致的。通常的建议是不要混用两种写法：或者只使用 numerator layout，或者在任何时候都不省略转置符号。

## 常用结论

对标量 $a$、向量 $\mathbf{a}$、矩阵 $\mathbf{A}$，若它们都不是 $x$ 或 $\mathbf{x}$ 的函数，在此给出一些常用结论，证明从略。

$$\frac{\partial{\mathbf{a^\top x}}}{\partial{\mathbf{x}}} = 
\frac{\partial{\mathbf{x^\top a}}}{\partial{\mathbf{x}}} = \mathbf{a}^{\top}$$

$$\frac{\partial{\mathbf{x^\top x}}}{\partial{\mathbf{x}}} = 2\mathbf{x}^{\top}$$

$$\frac{\partial{(\mathbf{x^\top a}})^2}{\partial{\mathbf{x}}} = 2\mathbf{x^\top a a^\top}$$

$$\frac{\partial{\mathbf{Ax}}}{\partial{\mathbf{x}}} = \mathbf{A}$$

$$\frac{\partial{\mathbf{x^\top A}}}{\partial{\mathbf{x}}} = \mathbf{A}^{\top}$$

$$\frac{\partial{\mathbf{x^\top Ax}}}{\partial{\mathbf{x}}} = \mathbf{x}^\top (\mathbf{A}+\mathbf{A}^{\top})$$


关于求导的加法、数乘、乘法、链式法则，此处不再赘述。

## 练习：最小二乘法

$\mathbf{X}$ 是 $m \times n$ 矩阵，$\mathbf{y}$ 是 $m$ 维矩阵，求 $\mathbf{w}$ 使得 $\lVert{\mathbf{Xw - y}}\rVert^2$ 最小。

解：

令 $ J = \lVert{\mathbf{Xw - y}}\rVert^2 = (\mathbf{Xw - y})^{\top} (\mathbf{Xw - y})
= \mathbf{w^{\top} X^{\top}Xw - w^{\top}X^{\top}y - y^{\top}Xw + y^{\top}y}$。

求导，根据上面给出的常用结论，有
$ \frac{\partial{\mathbf{J}}}{\partial{\mathbf{w}}} = 
 \frac{\partial{\mathbf{w^{\top}X^{\top}Xw}}}{\partial{\mathbf{w}}} - \frac{\partial{\mathbf{w^{\top}X^{\top}y}}}{\partial{\mathbf{w}}} - \frac{\partial{\mathbf{y^{\top}Xw}}}{\partial{\mathbf{w}}} + \frac{\partial{\mathbf{y^{\top}y}}}{\partial{\mathbf{w}}}
 = 2(\mathbf{w^{\top}X^{\top}X - y^{\top}X})$。

令 $ \frac{\partial{\mathbf{J}}}{\partial{\mathbf{w}}} = 0 $，即 $\mathbf{w^{\top}X^{\top}X = y^{\top}X}$。
转置，得 $\mathbf{X^{\top}Xw = X^{\top}y}$。

如果 $\mathbf{X^{\top}X}$ 可逆，则 $\mathbf{w} = (\mathbf{X^{\top}X})^{-1} \mathbf{X^{\top}y}$。这正是 Moore-Penrose 广义逆矩阵。


参考文献及扩展阅读：

- Leow Wee Kheng. CS5240 slides. https://comp.nus.edu.sg/~cs5240/lecture/matrix-differentiation.pdf
- K. B. Petersen and M. S. Pedersen, The Matrix Cookbook, 2012.
