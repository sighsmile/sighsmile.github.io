---
layout: post
title: "EM 算法与 K-means 和 GMM 聚类"
description: ""
category:
tags:
---

之前已经大致推导过 EM 算法。这里再进一步思考一下它到底是怎么来的，要怎么用。

回顾一下 EM 算法的目的，是最大似然估计 <span>$\arg \max_{\theta} \log p_{\theta}(x)$</span>。但是除了可观测的变量 <span>$X$</span> 以外，还存在观测不到的隐变量 <span>$Z$</span>，其真实分布是未知的 <span>$p(z)$</span>，因而难以直接求解。

引入记号 <span>$p_{\theta}(x) = p(x|\theta)$</span>，由于 
<div>$$p_{\theta}(x) = \sum_z p_{\theta}(x, z) = \sum_z p(z) \frac{p_{\theta}(x, z)}{p(z)} = \mathbb{E}_ {z \sim p}\frac{p_{\theta}(x, z)}{p(z)}$$</div>
这个 <span>$\log \mathbb{E}$</span> 的优化目标不好解决，利用 Jensen 不等式，可以放缩为 <span>$\mathbb{E} \log$</span>，这是因为可以证明
<div>$$\log \mathbb{E}_ {z \sim p}\frac{p_{\theta}(x, z)}{p(z)} \geq 
\mathbb{E}_ {z \sim p} \log \frac{p_{\theta}(x, z)}{p(z)} = 
-KL(p(z)||p_{\theta}(x, z)) = 
\mathbb{E}_ {z \sim p} \log p_{\theta}(x, z) + H(p(z))$$</div>

虽然真实的 <span>$p(z)$</span> 是未知的，但是我们可以利用当前条件做一个尽可能好的估计，那就是
<div>$$p_{\theta^{(t)}}(z|x)$$</div>
把它代入上面的 <span>$\mathbb{E}_ {z \sim p} \log p_{\theta}(x, z)$</span> 的 <span>$z \sim p$</span> 部分，即可更新对参数的最大似然估计 
<div>$$\theta^{(t+1)} = \arg \max_{\theta} \mathbb{E}_ {z \sim p_{\theta^{(t)}}(z|x)} \log p_{\theta}(x,z)$$</div>

通常文献中将 <span>$\mathbb{E}_ {z \sim p_{\theta^{(t)}}(z|x)} \log p_{\theta}(x,z)$</span> 称为 Q 函数。

## Jensen 不等式

之前我们未加解释，直接利用 Jensen 不等式对目标函数进行放缩。这里还是稍微解释一下，此处放缩可以等价地改写为：
<div>$$J = \log \sum_z p(z) q(z) \geq \sum_z p(z) \log q(z) = Q + H(p(z))$$</div>
其中 <span>$\sum_z p(z)=1$</span>，<span>$p(z),q(z)>0$</span>。

Jensen 不等式是说，若 <span>$f(x)$</span> 是 <span>$(a,b)$</span> 上的凸函数，则对任意的 <span>$x_1,\ldots,x_n \in (a,b)$</span> 和正数 <span>$a_1 + \cdots + a_n = 1$</span>，有 
<div>$$f(a_1 x_1 + \cdots + a_n x_n) \leq a_1 f(x_1) + \cdots + a_n f(x_n)$$</div>

注意，上面说的凸函数是指的 convex 的凸函数（虽然中文有时也称之为凹函数），例如 <span>$f(x) = x^2$</span>。凸函数（convex）的定义是：对于区间内的任意 <span>$x_1, x_2$</span> 和任意 <span>$0 < \lambda < 1$</span>，都有 <span>$f(\lambda x_1 + (1-\lambda)x_2) \leq \lambda f(x_1) + (1-\lambda) f(x_2)$</span>。

反之，如果 <span>$f(x)$</span> 是 concave 函数（虽然中文有时也称为凸函数），例如 <span>$f(x) = \log x$</span>，上式为大于等于号。因此上面的放缩可以用 Jensen 不等式直接证明。

> 吐槽一下，凸函数的中文翻译是多么混乱啊。一个如此基本的概念，居然在各种教材中都不一致，甚至截然相反。总之，不要死记硬背凸函数的结论。只要注意观察 Jensen 不等式和凸函数定义式中的符号是一致的，很容易推出来。

## 总结

总的来说，如果我们想求一个目标函数的最小值，就用一系列与它相切、但在它之上的曲线去逼近它，不断去优化更紧的上界；反之，如果想求目标函数的最大值，就用一系列与它相切、但作为它的下界的曲线去逼近它。为了方便使用凸优化的工具，我们总希望构造可微的、凸的代价函数去替代或逼近原始的误差函数。

因此，我们可以重新用这套统一的思路来理解各种最优化方法。

什么是 EM 算法？就是为了求对数似然的最大值，改用它的下界去逼近它，这些下界就是 Q 函数，从而每步迭代更新 <span>$\theta_{n+1} = \arg \max Q(\theta_n)$</span>。

什么是牛顿法？就是为了求代价函数的最小值，改用它的上界去逼近它，这些上界就是切抛物线（为了省事儿这里只写一元的式子了） <span>$g(x) = f(x_n) + f'(x_n) \cdot (x-x_n) + \frac{1}{2} f''(x_n) \cdot (x-x_n)^2$</span>，从而每步迭代更新 <span>$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$</span>。

什么是梯度下降法？上面用的是二阶近似，这次我们用一阶近似去逼近它，也就是切线 <span>$g(x) = f(x_n) + f'(x_n) \cdot (x-x_n)$</span>，显然 <span>$x - x_n$</span> 与 <span>$f'(x_n)$</span> 反向时 <span>$f(x)$</span> 下降最大（请自行想象高维的情况），所以每步迭代更新 <span>$x_{n+1} = x_n - \alpha f'(x_n)$</span>，其中步长 <span>$\alpha$</span> 是一个正的系数。实际上也相当于用 <span>$\alpha$</span> 作为牛顿法中 <span>$\frac{1}{f''(x_n)}$</span> 的近似，所以直观上理解，如果希望跳出 <span>$f''(x_n)$</span> 很小的平原区域，就需要更大的 <span>$\alpha$</span>；如果希望收敛到谷底而不是在深谷的半山腰来回振荡，就需要更小的 <span>$\alpha < \frac{1}{f''(x_n)}$</span>。但是提醒读者注意，这条切线已经无法保证是一个严格的上界。

## K-Means

在聚类问题中，可观测的变量是 N 个点 <span>$X = (x_1, \ldots, x_N)$</span>，隐变量是 K 个类。为了形式化叙述聚类问题，我们需要定义每个类是什么样的，每个点与每个类的关系是什么样的。

一种方式是将每个类 <span>$C_j$</span> 定义为它的中心 <span>$\mu_j$</span>，每个点按照与中心的距离硬性划分到最近的类，并使总的代价最小。这就是 K-means 算法。

K-means 的操作步骤是，首先初始化选 K 个点作为各类的中心，每一步按照当前的聚类中心对全体 N 个点进行划分，对划分结果的每一类计算新的聚类中心，迭代直至收敛。

如果点到类的距离就是点到类的中心的欧氏距离，总的代价为距离的平方和，那么新的聚类中心是每一类的重心（该类中各点坐标的均值，这也是算法名称的来历）。当然，这是很直观的，大家一般不加思考地默认接受。但是为什么要用重心呢？

事实上，对于 K-means 聚类，总的代价是 <span>$J = \sum_i \min_j \lVert x_i - \mu_j \rVert^2 = \sum_j \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2$</span>。因此我们可以证明几个结论：

第一，每个点应该分在距离它最近的中心 <span>$\mu_j$</span> 所在的类 <span>$C_j$</span>，否则只需将该点重新分类就可以降低代价；

第二，新的聚类中心应该是每一类中各点坐标的均值，因为
<span>$\frac{\partial{J}}{\partial{\mu_j}} = \frac{\partial{\sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2}}{\partial{\mu_j}} = -2 \sum_{x_i \in C_j} (x_i - \mu_j)$</span>，令上式为零即可推出，当前分类下的最低代价对应 <span>$\mu_j = \frac{\sum_{x_i \in C_j} x_i}{\sum_{x_i \in C_j} 1}$</span>；

第三，因为 N 和 K 都是有限的，该算法迭代一定会收敛，即上面两步操作都无法继续降低代价，但是未必收敛到全局最优。实践中，为了尽可能避免陷入局部最优，一般会多次随机初始化并运行 K-means 算法，并且在初始化时会尽量分散各类中心点。

## GMM

另一种方式是将每个类 <span>$C_j$</span> 定义为一个中心为 <span>$\mu_j$</span>、方差为<span>$\Sigma_j$</span> 的高斯分布（是的，协方差矩阵与求和符号都用 Sigma 表示，请自行结合上下文区分）。每个点视为从 K 个类中随机选一个类（按照参数为 <span>$\phi$</span> 的多项式分布）、按照该类的高斯分布生成的，并使总的似然最大。这就是 GMM 算法，它假设全体数据是由若干个高斯分布混合而来的（mixture of Gaussians）。

我们复习一下高维的高斯分布 
<div>$$\mathcal{N}(x|\mu, \Sigma) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\{-\frac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\}$$</div>

GMM 的操作步骤是，首先初始化参数，每一步按照当前的分布对全体 N 个点进行软的划分，即计算每个点来自各个类的后验分布，再根据这个分布来更新各类的高斯分布参数，迭代直至收敛。

这样说可能会有些晕，不过实际上这就是之前说过的 EM 算法。

先利用当前参数 <span>$\theta^{(t)} = (\mu^{(t)}, \Sigma^{(t)}, \phi^{(t)})$</span>，对每个观察值估计隐变量的分布
<div>$$p_{\theta^{t}}(C_k|x_i) = \frac{p_{\theta^{t}}(C_k)p_{\theta^{t}}(x_i|C_k)}{\sum_j p_{\theta^{t}}(C_j)p_{\theta^{t}}(x_i|C_j)}$$</div>

然后代入 <span>$Q = \mathbb{E}_ {z \sim p_{\theta^{(t)}}(z|x)} \log p_{\theta}(x,z)$</span>，更新 <span>$\theta^{(t+1)} = \arg \max Q$</span>。可以求偏导令它们都为零，也可以将它们写成期望的形式，由于该模型是多项式分布和高斯分布，分布假设较为简单，这里省略推导过程直接写出
<div>$$\phi_j^{(t+1)} = \frac{\sum_{i} p_{\theta^{t}}(C_j|x_i)}{N}$$</div>
<div>$$\mu_j^{(t+1)} = \frac{\sum_{i} p_{\theta^{t}}(C_j|x_i)x_i}{\sum_{i} p_{\theta^{t}}(C_j|x_i)}$$</div>
<div>$$\Sigma_j^{(t+1)} = \frac{\sum_{i} p_{\theta^{t}}(C_j|x_i)(x_i-\mu_j)(x_i-\mu_j)^{\top}}{\sum_{i} p_{\theta^{t}}(C_j|x_i)}$$</div>

## Soft K-means

前面说过，K-means 的代价函数是 <span>$J = \sum_j \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2$</span>，所以乍一看与 EM 算法无关。但是接下来我会将它转化为相关的形式，发掘上述算法的深层联系。由于是在 K-means 中引入软的划分，我会姑且称之为 soft K-means。

对于 K-means 的设定，每个点归属于（或者来自于）中心离它最近的类，可以写出一个理想化的概率分布 
<div>$$p(C_j|x_i) = 1 \quad \mathrm{if} \quad \lVert x_i - \mu_j \rVert^2 = \min_k \lVert x_i - \mu_k \rVert^2$$</div>
<div>$$p(C_j|x_i) = 0 \quad \mathrm{otherwise}$$</div>

这让我们想到 softmax 函数，其实也就是 softargmax：<span>$p(z_i) \propto \exp(\beta z_i)$</span>，其中 <span>$\beta > 0$</span>，然后归一化。仿照此式可写出 softargmin
<div>$$p(C_j|x_i) \propto \exp(-\beta \lVert x_i - \mu_j \rVert^2)$$</div>
归一化，也就是说
<div>$$p(C_j|x_i) = \frac{\exp(-\beta \lVert x_i - \mu_j \rVert^2)}{\sum_k \exp(-\beta \lVert x_i - \mu_k \rVert^2)}$$</div>
与 GMM 聚类中的隐变量估计 <span>$p_{\theta^{t}}(C_k|x_i)$</span> 的计算式对比，可以看出可以看出上面这个式子是它的一个特例，也就是如果 GMM 的高斯分布的 <span>$\Sigma$</span> 是对角阵 <span>$\frac{1}{2\beta}I$</span>，其他系数配平，两个式子就是相等的。

> 关于 softmax 和 smooth maximum 技巧，建议阅读维基百科的 Softmax_function、Generalized_f-mean 和 Smooth_maximum 词条。

之前由 <span>$J = \sum_j \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2
= \sum_i \min_j \lVert x_i - \mu_j \rVert^2$</span> 并令 <span>$\frac{\partial{J}}{\partial{\mu_j}} = 0$</span> 推导过 K-means 的迭代式 <span>$\mu_j = \frac{\sum_{x_i \in C_j} x_i}{\sum_{x_i \in C_j} 1}$</span>。现在我们需要改写软化的代价函数
<div>$$J = \sum_i \sum_j p(C_j|x_i) \lVert x_i - \mu_j \rVert^2$$</div>

固定 <span>$p(C_j|x_i)$</span> 并假装它与 <span>$\mu_j$</span> 无关（更严谨的写法是把它换成 <span>$c_{i,j}$</span> 表示），于是 <span>$\frac{\partial{J}}{\partial{\mu_j}} = -2 \sum_i p(C_j|x_i) (x_i - \mu_j)$</span>，令它为零，则有
<div>$$\mu_j = \frac{\sum_i p(C_j|x_i) x_i}{\sum_i p(C_j|x_i) }$$</div>
可以看出这个式子与 GMM 聚类中的迭代式的相似性。如果将 <span>$p(C_j|x_i)$</span> 从连续分布收缩到 0/1 的硬取值，就变成 hard K-means 里计算的迭代式。

如此就将 K-means 和 GMM 以及 EM 算法联系在一起。


