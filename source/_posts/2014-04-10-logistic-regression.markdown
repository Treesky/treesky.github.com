---
layout: post
title: "logistic regression"
date: 2014-04-10 16:58
comments: true
categories: 
---

logistic regression 的 prob 的定义是

$$
p(y=1|x)=\cfrac{1}{1+e^{- \mathbf{w} \mathbf{x}}} \\
p(y=-1|x)=\cfrac{1}{1+e^{\mathbf{w} \mathbf{x}}}
$$

整体loss是max-liklihood
$$
f(w) =  \sum log( {1+e^{-y \mathbf{w} \mathbf{x}}} )
$$

一阶倒数为
$$
\begin{aligned}
\cfrac{ \partial f}{\partial w_i} = \sum \cfrac{-y x_i}{1+e^{y \mathbf{w} \mathbf{x}}} \\
 &= \sum ((\cfrac{1}{1+e^{ - y \mathbf{w} \mathbf{x}}} - 1) y x_i)
\end{aligned}
$$ 

二阶倒数为
$$
\cfrac{\partial f}{\partial w_i \partial w_j} = \sum \cfrac{x_i x_j}{(1+e^{-y \mathbf{w} \mathbf{x}}) (1+e^{y \mathbf{w} \mathbf{x}})}
$$