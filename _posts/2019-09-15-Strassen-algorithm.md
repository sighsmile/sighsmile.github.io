---
layout: post
title: "矩阵相乘的 Strassen 算法"
description: ""
category:
tags:
---

计算矩阵相乘的 Strassen 算法是怎样发明的？

## Master Method

首先复习一下用于推导分治算法时间复杂度的 Master Method：

如果
$$T(n) \leq a \cdot T(\frac{n}{b}) + O(n^d)$$
那么 
$$T(n) \leq c n^d \sum_{j=0}^{\log_{b}{n}} \left[ \frac{a}{b^d}\right] ^j $$

于是
$$T(n) = \left\{ \begin{array}{l}
O(n^d), \quad \text{if } a < b^d\\
O(n^d \log n), \quad \text{if }a = b^d\\
O(n^{log_b{a}}), \quad \text{if }a > b^d
\end{array} \right.$$

例如，$a = 8$，$b = 2$，$d = 2$，则 $T(n) = O(n^3)$。如果能改进为 $a = 7$，则 $T(n) = O(n^{\log_2{7}}) \approx O(n^{2.8})$。

## Strassen 算法

Volker Strassen 发表于 1968 年的文章，标题为 Gaussian Elimination is not Optimal（高斯消元法不是最优的，这是针对矩阵求逆而讲的，因为快速的矩阵乘法可以给出更快的矩阵求逆）。原论文只是直接给出了算法结果，没有提及构造思路。本文尝试启发式探索一下如何构造出这种算法。

题目是，计算矩阵乘法，
$XY = \begin{pmatrix}
A & B \\ C & D 
\end{pmatrix}
\begin{pmatrix}
E & F \\ G & H
\end{pmatrix}
= \begin{pmatrix}
AE+BG & AF+BH \\ CE+DG & CF+DH
\end{pmatrix}
$。其中 $X$ 和 $Y$ 是 $n$ 阶方阵，其它字母都是 $\frac{n}{2}$ 阶方阵。

按照定义计算，需要做 $4 \times 2 = 8$ 次规模为 $\frac{n}{2}\cdot \frac{n}{2}$ 的矩阵乘法。尽管是分而治之了，复杂度依然是 $O(n^3)$，和直接计算大的方阵乘法一样。

我们希望通过引入一些其它的项，进行组合，使乘法结果得到复用，减少需要计算的乘法次数。如何做这样的组合？利用合并同类项，假如要计算 $AG+DG$，可以计算 $(A+D)G$，乘法次数由 2 次减少为 1 次；假如要计算 $BG+DG+BF+DF$，可以计算 $(B+D)(G+F)$，乘法次数由 4 次减少为 1 次。

以结果矩阵中的第一项 $AE+BG$ 为例，第一种拆法是不妨通过引入 $AG$ 项，改写 
$$AE+BG = AE+BG+0\cdot AG = A(E-G) + (A+B)G$$
类似引入 $BE$ 项也可以，此时计算该部分结果所需的乘法次数仍然是 2 次。

第二种拆法是同时引入 $AG$ 和 $BE$ 项，改写 
$$AE+BG = AE+BG+0\cdot AG + 0\cdot BE = (A+B)(G+E) - (AG+BE)$$
但是此时计算该部分结果所需的乘法次数变成 3 次（除非我们复用其中的部分结果）。总之，原则是尽可能复用出现过的变量。

考虑对称性，一方面不难发现此处的符号选择不是唯一的，如果改写 $AE+BG = A(E+G) + (B-A)G$，只需后续相应修改其他项的正负号即可；另一方面不能将每一项都按照相同的方式分拆，因为那样复用性完全没有提高，计算效率没有提升。

因此，不妨首先将乘积矩阵的对角元按照上述第一种方法不对称地改写。例如 
$$AE+BG = AE+BG+0\cdot AG = A(E-G) + (A+B)G$$
$$CF+DH = CF+DH+0\cdot DF = (C+D)F + D(H-F)$$
注意到第二处引入的是 $DF$ 而不是 $CH$。

现在需要设法将剩下的 $CE+DG$ 和 $AF+BH$ 改写为类似的组合，复用前面计算过的 4 项，并引入少于 4 次（即不超过 3 次）新的矩阵乘法。

观察发现，前面的 4 项中，有两项与 $CE+DG$ 的重复元为 2，直觉上复用可能性更大，不妨写 
$$CE+DG = A(E-G) + (C+D)F + R, \quad R = CE+DG-AE+AG-CF-DF$$
同理有 
$$AF+BH = (A+B)G + D(H-F) + S, \quad S = AF+BH-AG-BG-DH+DF$$

比较 $R$ 和 $S$ 发现其中的相同部分为 $AG-DF$，同时 $R$ 里面还出现了 $DG$，$S$ 里面还出现了 $AF$（与 $AG$ 符号相反）。因此，按照第二种方法将它改写为 $AG-DF+DG-AF = (A+D)(G-F)$，这是我们想看到的计算复用。

再次整理得 
$$R = (A+D)(G-F) + (C-A)(E-F)$$
$$S = -(A+D)(G-F) + (B-D)(H-G)$$
在这 4 个乘法项中，只需要独立计算 3 次，总的乘法次数得以节省。

注意，本文不是 Strassen 算法的唯一发现方式，也并未给出这是最优算法的证明，仅仅希望本文有助于读者理解该算法的思路，并将它运用于解决其他的问题上。
