---
layout: post
title: "EM 算法"
description: ""
category:
tags:
---

统计学习里的一类基本问题是，通过观察到的数据 <span>$X$</span>，对背后的参数 <span>$\theta$</span> 进行最大似然估计，即 <span>$\hat{\theta} = \arg \max_{\theta} L(\theta; X)$</span>，其中似然一般定义为 <span>$\mathcal{L}(\theta; X) = \mathcal{P}_ {\theta}(X) = p(X|\theta)$</span>。

> 上式中的分号和竖线都表示“以后者为条件”，但是在似然记号中似乎一般习惯用冒号或分号，而在概率中用竖线较多，因此这里按照这种习惯混用两种符号。

可惜，很多情况下存在隐变量 <span>$Z$</span>，直接的最大似然估计行不通。这种情况下，EM（Expectation Maximization）算法可以帮助我们估计。

很多文章只是不加解释地抛出结论，EM 算法每次迭代分为 E 和 M 两步：

- E 步计算期望函数 <span>$Q(\theta|\theta^{(t)}) = \mathbb{E}_ {Z|X, \theta^{(t)}}[\log \mathcal{L}(\theta; X, Z)]$</span>
- M 步计算更新估计 <span>$\theta^{(t+1)} = \arg\max_{\theta} Q(\theta|\theta^{(t)})$</span>

下面将详细地阐述 EM 算法为何物。首先看一个小例子。

## 抛硬币的例子

摘录 What is the expectation maximization algorithm? (Chuong B Do & Serafim Batzoglou，2008) 里面的例子：

- 有两枚不公平硬币 A 和 B，用 <span>$\theta_A, \theta_B$</span> 表示它们分别掷出正面的概率
- 在第 <span>$i=1,2,\ldots,5$</span> 轮试验中，首先用一枚公平硬币 C 随机抛掷，记录结果 <span>$z_i$</span>，正面为 A，反面为 B；然后将相应的硬币独立重复掷 10 次，其中的正面次数记为 <span>$x_i$</span>
- 如何用 <span>$x_1, \ldots, x_5; z_1, \ldots, z_5$</span> 估计 <span>$\theta_A, \theta_B$</span>？很简单，借助 <span>$z_i$</span> 可以知道硬币 A 和 B 分别抛掷多少次，分别有多少次正面，直接最大似然估计即可
- 如何用 <span>$x_1, \ldots, x_5$</span> 估计 <span>$\theta_A, \theta_B$</span>？没有隐变量 <span>$z$</span> 的信息，做法就不那么显然了

例如，<span>$(z_1, \ldots, z_5) = (B, A, A, B, A)$</span>，<span>$(x_1, \ldots, x_5) = (5, 9, 8, 4, 7)$</span>，在已知隐变量信息的情况下，直接估计 <span>$\hat{\theta_A} = \frac{24}{30} = 0.80$</span>，<span>$\hat{\theta_B} = \frac{9}{20} = 0.45$</span>。

在未知隐变量信息的情况下，首先拍脑袋假设一个 <span>$\theta_A^{(0)} = 0.6$</span>，<span>$\theta_B^{(0)} = 0.5$</span>，然后设法迭代改进估计：

> 以下计算取小数点后两位。

第一轮观察到正面 5 次，反面 5 次，贝叶斯公式有 
<div>$$\frac{p(A| i=1, \theta_A^{(0)},\theta_B^{(0)})}{p(B| i=1, \theta_A^{(0)},\theta_B^{(0)})} = \frac{0.6^5 0.4^5 0.5}{0.5^5 0.5^5 0.5} = \frac{0.45}{0.55}$$</div>
也就是说隐变量 <span>$z_1$</span> 取 A 的条件（后验）概率是 <span>$p(A|i=1,\theta^{(0)})= p_1 = 0.45$</span>。

依此类推得到，五轮试验中，隐变量取 A 的贝叶斯概率分别为 <span>$(p_1, \ldots, p_5) = (0.45, 0.80, 0.73, 0.35, 0.65)$</span>。以此为条件，计算硬币 A 抛掷的期望：

- 正面次数期望为 <span>$(5,9,8,4,7) \cdot (0.45, 0.80, 0.73, 0.35, 0.65) = 21.25$</span>
- 反面次数期望为 <span>$ (5,1,2,6,3) \cdot (0.45, 0.80, 0.73, 0.35, 0.65)  = 8.55$</span>

现在可以更新 <span>$\theta_A$</span> 的最大似然估计为 <span>$\theta_A^{(1)} = \frac{21.25}{21.25+8.55} = 0.71$</span>。类似地得到 <span>$\theta_B^{(1)} = 0.58$</span>。继续迭代直到收敛。

当然这只是一个 toy example，为了展示大致思路。下面介绍真正的 EM 算法。

## 复习一下信息论

熵 
<div>$$H(p) = -\sum_x [p(x) \log p(x)] = - \mathbb{E}_ {x \sim p} \log p(x)$$</div>
衡量的是分布 <span>$p$</span> 本身的信息量。编码中，以 2 为底，熵代表编码信息所需的最少比特数。

交叉熵 
<div>$$H(p, q) = -\sum_x [p(x) \log q(x)] = - \mathbb{E}_ {x \sim p} \log q(x)$$</div>
衡量的是使用 <span>$q$</span> 来编码 <span>$p$</span> 的信息所需要的最少比特数。

KL 散度 
<div>$$KL(p||q) = \sum_x [p(x) \log \frac{p(x)}{q(x)}] = \mathbb{E}_ {x \sim p} \log \frac{p(x)}{q(x)}$$</div>
衡量的是使用 <span>$q$</span> 来编码 <span>$p$</span> 的信息所增加的比特数。换言之，这是衡量分布 <span>$q$</span> 和分布 <span>$p$</span> 的差异的一种方法，通常 <span>$p$</span> 是真实分布，<span>$q$</span> 是估测分布。可以证明 KL 散度非负，显然 <span>$KL(p||p) = 0$</span> 是它的最小值。

在真实分布 <span>$p$</span> 确定不变的情况下，交叉熵和 KL 散度的单调性是一致的，作为优化的目标函数可以相互转换。

对于似然，引入记号 <span>$p_{\theta}(x) = p(x|\theta)$</span>。采取独立假设，<span>$p(X|\theta) = \prod_i p_{\theta} (x_i)$</span>，又假设真实分布为 <span>$p(x)$</span>，取对数就有 
<div>$$\log \mathcal{L}(\theta; X) = \sum_i \log p_{\theta}(x_i) = \sum_x p(x) \log p_{\theta}(x) = \mathbb{E}_ {x \sim p} \log p_{\theta}(x) = -H(p, p_{\theta})$$</div>
因此很多时候我们做最大似然估计，实际是设法求 <span>$\arg \max_{\theta} \mathbb{E}_ {x \sim p} \log p_{\theta}(x)$</span>。而这等价于求 <span>$\arg \min_{\theta} H(p, p_{\theta})$</span> 或者 <span>$\arg \min_{\theta} KL(p||p_{\theta})$</span>。


## EM 算法

EM 算法大致是首先根据当前的参数猜测，估计隐变量的分布，计算似然的期望，然后据此修正参数猜测。

现在详细解释其中的计算步骤，可比照前面的硬币例子理解。注意引入记号 <span>$p_{\theta}(x) = p(x|\theta)$</span>：

1. 以现有的（初始化或者之前迭代更新过的） <span>$\theta^{(t)}$</span> 为条件，计算 <span>$p_{\theta^{(t)}}(z)$</span> 和 <span>$p_{\theta^{(t)}}(x|z)$</span>，然后用贝叶斯公式对隐变量的分布做估计，计算 
<div>$$p_{\theta^{(t)}}(z|x)$$</div>（在硬币例子中，就是计算每轮隐变量取 A 的贝叶斯概率）
2. 为进行最大似然估计，我们想要 <span>$\arg \max_{\theta} \log p_{\theta}(x)$</span>。设隐变量的真实分布为 <span>$p(z)$</span>，由于 
<div>$$p_{\theta}(x) = \sum_z p_{\theta}(x, z) = \sum_z p(z) \frac{p_{\theta}(x, z)}{p(z)} = \mathbb{E}_ {z \sim p}\frac{p_{\theta}(x, z)}{p(z)}$$</div>
对数为凸函数，利用 Jensen 不等式可以证明
$$\log \mathbb{E}_ {z \sim p}\frac{p_{\theta}(x, z)}{p(z)} \geq 
\mathbb{E}_ {z \sim p} \log \frac{p_{\theta}(x, z)}{p(z)} = 
-KL(p(z)||p_{\theta}(x, z)) = 
\mathbb{E}_ {z \sim p} \log p_{\theta}(x, z) + H(p(z))$$
因而我们可以将优化目标从左边的 <span>$\log \mathbb{E}$</span> 改为右边的第一项 <span>$\mathbb{E} \log$</span>。
现在假设前一步的隐变量分布就是真实分布（当然我们知道它并不是），我们的目标就变成
$$\mathbb{E}_ {z \sim p_{\theta^{(t)}}(z|x)} \log p_{\theta}(x,z)
$$（在硬币例子中，我们计算硬币 A 和 B 的抛掷结果的期望，对应于此步）
3. 更新对参数的最大似然估计 
<div>$$\theta^{(t+1)} = \arg \max_{\theta} \mathbb{E}_ {z \sim p_{\theta^{(t)}}(z|x)} \log p_{\theta}(x,z)$$</div>
当然这也等价于 <span>$\arg \min_{\theta} -KL(p(z)||p_{\theta}(x, z))$</span>。

为了使计算可行，我们需要保证适当选取分布假设，使得 <span>$p_{\theta^{(t)}}(z)$</span> 和 <span>$p_{\theta^{(t)}}(x|z)$</span> 可以计算；并且使得 <span>$p_{\theta}(x,z)$</span> 的对数似然期望的 argmax 可以直接求出。

我们关心两个问题，每步是否保证更优，最终是否能到全局最优。答案是，每步都在优化，最终必然收敛，但可能陷入局部最优。具体证明很容易搜到，不写了。

这篇就写到这里，后续有空再写一下几个相关的模型：HMM，k-Means，LDA。
