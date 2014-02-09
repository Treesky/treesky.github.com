---
layout: post
title: "l1r-lr optimization"
date: 2014-02-09 18:33
comments: true
categories: 
---

在大规模机器学习中，可能最常见的机器学习算法就是 l1-regularized logisic regreesion. 这种算法适用于大规模稀疏数据，上亿维度的feature，只有几十维的非0特征，几十亿的instance。loss-function如下：

$$
\min \limits_{\mathbf{w}} f(\mathbf{w}) = ||\mathbf{w}||_1 + C \sum _{i=1}^{l} log(1+e^{-y \mathbf{w}^T x})
$$

owlqn是解决l1r-lr的“标准方法”，实现起来简单粗暴有效 =.=

介绍见：http://www.cnblogs.com/downtjs/archive/2013/07/29/3222643.html
源码见：http://research.microsoft.com/en-us/um/people/jfgao/
这个代码里面，同时计算 $f'(\mathbf{w})$ 和 $f(\mathbf{w})$ 的那部分思路还是挺有意思的。

在`A comparision of Optimization Methods and Software for Large-Scale L1-regularized Linear Classification`里面介绍了另外一些优化方法，比如CDN.

一、BBR算法
BBR(Bayesian Binary Regression)是`Large-scale Bayesian logistic regression for text categorization`中提出的一种使用trust-region newton methold来解决 coordnate descent 的 subproblem 的方法。

算法的基本思想如下：

step 1: 给定 $\mathbf{w}^1$.

step 2: while true (coordinate descent的外层循环)

step 3: 对 active sets 中的coordinate j (coordinate descent的内层循环)

step 4: 计算 
$$
U _j = C \sum \limits _{i=1}^l x _{ij}^2 F(y _i(\mathbf{w}^{k,j})^T \mathbf{x}_i, \Delta _j |x _{ij}| ) 
$$

$$
F(r, \delta) = 0.25 (|r| \leq \delta)
$$

$$
F(r, \delta) = \cfrac{1}{2+e^(|r| - \delta) + e^{\delta - |r|}} (otherwise)
$$

$$
\Delta _j \text{ is the trust-region}
$$

step 5: 计算
$$
d = \min ( \max( P( - \cfrac{g_j^{'}(0)}{U_j}, w_j^{k,j}), - \Delta _j) , \Delta _j)
$$

$$
P(z,w) = z \text{ for } (sgn(w+z) = sgn(w))
$$

$$
P(z,w) = -w \text{ (otherwise)}
$$

step 6: 更新
$$
\Delta _j = \max (2|d|, \Delta _j / 2)
$$

step2 和 step3 是标准的 coordinate descent 的算法框架。在step4-step6中，解决了这样一个subproblem，就是

$$
\min \limits_{z} g_j(z) = | \mathbf{w} _j + z | - | \mathbf{w} _j| + L(\mathbf{w} + z \mathbf{e} _j ) - L(\mathbf{w})
$$ 

这个 subproblem 的解就表示在选定了 coordinate j上我们应该 move 多长的距离。BBR 使用了 trust-region newton method 来解决这个问题。在使用trust-region过程中，我们需要一个函数来在trust-region内逼近原始函数，在BBR中使用了如下函数

 $$ g_j(z) = g_j(0) + g_j^{'}(0) z + \cfrac{1}{2} g_j^{''}(\eta z) z^2 $$

在解这个问题的过程中，有两个地方需要注意。

第一、找到一个"合适"的 $ {U_{jz}} $ 使得 
$$
U_{jz} \geq g_j^{''}(z), \forall |z| \leq \Delta _j 
$$ 

$$
\hat g_j(z) = g_j(0) + g_j^{'}(0)z + \cfrac{1}{2} {U_{jz}}^2
$$
在这种情况下，只要 step $z$ 能优化 $\hat g_j(z)$ 也就是 $\hat g_j(z) \leq \hat g_j(0)$ 就可以证明
$$
g_j(z) - g_j(0) = g_j(z) - \hat g_j(0) \leq \hat g_j(z) - \hat g_j(0) \le 0
$$
也就是能优化 $\hat g_j(z)$ 的step $z$ 也能优化 $g_j(z)$

第二、在 $w_j^{k,j} = 0$ 的时候 $g_j(z)$ 在 $ z = 0$的情况下，是不连续的。因此 $g_j^{'}(z)$ 是not well-defined的。在这种情况下，定义如下：如果 $L_j^{'}(0) + 1 \le 0, g_j^{'}(0) = L_j^{'}(0)+ 1$ 如果 $L_j^{'}(0) - 1 \ge 0, g_j^{'}(0) = L_j^{'}(0) - 1$。 这种定义和 OWL-QN 的sub-gradient的定义是如出一辙的，是从"目的"出发的一个定义，这样定义可以使得在 $0 \le z \leq -g_j^{'}(0)/U_j$ 的情况下，使得 $ \hat g _j(0) \le \hat g _j(0)$ 也就是使我们得到一个更好的解。 这个bound的好坏严重影响算法的性能。


二、 Coordinate Descent Method Using One-Dimensional Newton Directions (CDN).

在CDN中，最后一项使用了 $U_j$ 来代替 Hession值。 如果使用Hession值，然后得到 newon direction 的话，在算法收敛到最后的 local convergence 阶段会得到一个更快的收敛速度。
因此问题就变成了优化如下的目标函数
$$
\min \limits_z |w_j^{k,j} + z| - |w_j^{k,j}| + L_j^{'}(0) z + \cfrac{1}{2} L_j^{''}(0)z^2
$$

上面的问题有close-formed solution就是
$$
d = -\cfrac{L^{'}(0) + 1}{L^{''}(0)} \text{ if } L^{''}(0) + 1 \leq L_j^{''}(0) w_j^{k,j}
$$
$$
d = -\cfrac{L^{'}(0) - 1}{L^{''}(0)} \text{ if } L^{''}(0) - 1 \geq L_j^{''}(0) w_j^{k,j}
$$
$$
-w_j^{k,j} \text{ otherwise. }
$$

这个算法的里面有两个要注意的地方就是

1、 在计算 $L^{'}(0)$ 的过程中，作者把 $L^{'}(0)$ 的计算方法变了一下，和原始的计算方法是一致的，但是应该能快一些。这个技巧在后面计算的 line search 终止条件的时候，同样出现了。

2、 在最后的 line search 过程中，作者处理了一种特殊情况，就是所有的特征值都是正数的时候，作者使用了一个计算较为简单，不需要遍历所有instance的近似终止条件来判断。这在在实现过程中对算法的运行时间，应该也有较大帮助。