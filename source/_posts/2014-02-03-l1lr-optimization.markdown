---
layout: post
title: "L1LR-Optimization"
date: 2014-02-03 23:28
comments: true
categories: optimization L1 LR Survey
---

`A comparision of Optimization Methods and Software for Large-Scale L1-regularized Linear Classification`

台大 **林智仁** 的文章，趁春节期间拜读一下。

优化目标是

$$
\min \limits_{\mathbf{w}} \ f(\mathbf{w}) = ||\mathbf{w}||_1 + C \sum \limits_{i=1}^{l}\xi(\mathbf{w};\mathbf{x}_i,y_i)
$$

对
$$
\xi(\mathbf{w};\mathbf{x}_i,y_i)
$$
的要求是 **非负，且凸，一阶连续可微**(为什么这三个要求？Appendix A表示这样的话，有至少有一个全局最优解)，这里主要讨论的两个loss就是 **log-loss** 和 **l2-loss** .

## Decomposition类的方法 ###

每次选择某些维度进行更新

### Cyclic Coordinate Descent ###

选择优化维度 $$ \mathbf{e}_j $$ 之后，通过优化
$$
\min \limits_{z} g_j(z) = f(\mathbf{w} + z \mathbf{e}_j) - f(\mathbf{w}) = |w_j^{k,j} + z| - |w_j^{k,j}| + L_j(z;\mathbf{w}^{k,j}) - L_j(0;\mathbf{w}^{k,j})
$$
从这个式子中，我们可以得到一个重要的信息，即某个维度 $$\mathbf{e}_j$$ 在当前迭代轮是否有效。有效的意思就是 $$z=0$$ 是上式的最优解——当$$z=0$$是上式的最优解的时候，我们就知道当然轮在当前维度上，不需要移动。$$z=0$$是上式的最优解的当且仅当：

$$
L_j^a (0) + 1 = 0, \ w_j^{k,j} > 0
$$
$$
L_j^a (0) - 1 = 0, \ w_j^{k,j} < 0
$$
$$
-1 \leq L_j^a (0) + 1 \leq 1, \ w_j^{k,j} = 0
$$

这个函数在 $$ z = -w_j $$ 的时候不可微
得到在维度 $$ \mathbf{e}_j $$ 上移动的距离

`Large-scale Bayesian logistic regression for text categorization` 中提出了BBR方法。BBR使用trust region和 一维牛顿step来解上面的问题。
CDN是扩展自`Coordinate descent method for large-scale L2-loss linear SVM`的方法，原方法使用一维newton direction+线性搜索来解决上面的问题，目标是l2约束的问题。
`A coordinate gradient descent method for nonsmooth convex optimization problems in machine learning` 文章中提出了一个通用的decomposition的框架，可以同时选择多个维度计算，CDN是其中的一个特例。

### Variable selection using Gradient information ###
使用梯度信息选择一组变量， Gauss-Southwell rule. 在使用了梯度信息之后，可以缩减迭代轮数，但是因为要计算梯度信息，因此单轮时间变长。
`A coordinate gradient descent method for l1-regularized convex optimization` 里面就是用了 CGD-GS方法， coordinate gradient descent & Gauss-Southwell rule

### Active Set ###
active set method 的特殊之处在于区分了0权重和非0权重特征，使得计算加快。
Grafting就是active set method for Log Loss

## 解带约束的优化 ##

### 光滑约束 ###

问题转化为
$$
\min \limits_{\mathbf{w}^+,\mathbf{w}^-} \sum _{j=1}^n w_j^+ + \sum _{j=1}^n w_j^- + C \sum _{i=1}^l \xi (\mathbf{w}^+ - \mathbf{w}^-; \mathbf{x}_i, y_i)
$$

subject to 
$$
 w_j^+ \geq 0, w_j^- \geq 0, j = 1,...,n 
$$

这个转化相当巧妙的把不可微的函数，拆了两个可微的函数的和

`An interior point method for large-scale l1-regularized logistic regression` 把这个问题转化为
$$
\min \limits_ {\mathbf{w}, \mathbf{u}} \sum _{j=1}^n u_j + C \sum _{i=1}^l \xi (\mathbf{w}; \mathbf{x} _i, y_i)
$$

subject to
$$
-u_j \leq w_j \leq u_j, j = 1,...,n
$$
然后使用interior point方法来解这个问题。

### 不光滑的约束 ###

$$
\min \limits_ {\mathbf{w}} \sum _{i=1}^l \xi(\mathbf{w};\mathbf{x}_i, y_i)
$$

subject to

$$
||\mathbf{w}||_1 \leq K.
$$

Active Set类的方法 `The generalized LASSO` 也可以被用来解决类似问题，

## 其他方法 ##

EM
随机梯度下降`Stochastic methods for l1 regularized loss minimization` `Sparse online learning via truncated gradient`
OWLQN
Hybrid Methods
Quadratic Approximation Followed by coordinate descent
Cutting Plane methods
Approximating L1 regularization by l2 regularization
Solution Path

提供了语法高亮和方便的快捷键功能，给您最好的 Markdown 编写体验。

来试一下：

- **粗体** (`Ctrl+B`) and *斜体* (`Ctrl+I`)
- 引用 (`Ctrl+Q`)
- 代码块 (`Ctrl+K`)
- 标题 1, 2, 3 (`Ctrl+1`, `Ctrl+2`, `Ctrl+3`)
- 列表 (`Ctrl+U` and `Ctrl+Shift+O`)

<!--more-->

### 实时预览，所见即所得 ###

无需猜测您的 [语法](http://markdownpad.com) 是否正确；每当您敲击键盘，实时预览功能都会立刻准确呈现出文档的显示效果。

### 自由定制 ###
 
100% 可自定义的字体、配色、布局和样式，让您可以将 MarkdownPad 配置的得心应手。

### 为高级用户而设计的稳定的 Markdown 编辑器 ###
 
 MarkdownPad 支持多种 Markdown 解析引擎，包括 标准 Markdown 、 Markdown 扩展 (包括表格支持) 以及 GitHub 风格 Markdown 。
 
 有了标签式多文档界面、PDF 导出、内置的图片上传工具、会话管理、拼写检查、自动保存、语法高亮以及内置的 CSS 管理器，您可以随心所欲地使用 MarkdownPad。
