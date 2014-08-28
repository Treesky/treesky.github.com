---
layout: post
title: "model_selection"
date: 2014-02-11 23:32
comments: true
categories: 
---

mostly from `Ensemble Methods: Foundations and Algorithms`

为了得到一个泛化能力强的模型，我们通常使用 cross-validation  + test error 来选择模型，但是交叉验证也不避免的会受到随机分割的数据集的影响。（也就是说，不同的数据集分割方法关于哪个模型的泛化能力更强，是会给出不同的答案的。）这时候我们就是使用一些假设检验的方法的帮助我们确定“模型A的效果优于模型B”，这个结论是否可信。

在交叉验证的假设检验中，我们最常使用的就是 10-fold cross-validation t-test.

10-fold cross-validation 就是随机把数据分成10份，然后9份训练，一份做测试，进而得到两个算法在10个测试集上的不同表现。

t-test 是从 z-test 中改进得到的，z-test 适用于样本量较大的时候，在样本量较小的时候，z-test有很大的误差，由于我们是 10-fold  validation 因此在假设检验的时候，我们只有10个样本，因此我们需要采用 t-test. t-test 需要满足一个条件，样本本身（即classifier 在检验集上的错误率/其他评价指标）服从高斯分布（在样本数量较大的情况下，即使不服从高斯分布，也可以使用t-test）。

在 Dependent-Samples t-test中，我们计算
$$
t = \cfrac{\sum \limits _{i=1}^n D_i}{\sqrt{\cfrac{n\sum \limits _{i=1}^n D_i^2 - (\sum \limits _{i=1}^n D_i)^2}{n-1}}}
$$

$D_i$ 表示在相同的 validation set 下的两个算法评测结果的差异。

通过如上计算的 t-值，以及给定的显著性水平 $ \alpha $ ，我们就可以得到拒绝阈 $ \| t \| $，并进而得到我们是否接受 $ H_0 $ 假设，也就是两个算法是否有差异。

但是 Dietterich[1998] 中指出 10-fold cross-validation t-test 会得到一个较小的 variability 并进一步会在两个算法没有差异的情况下，错误地认为两个算法有差异。因此 Dietterich 推荐使用 5*2 cross-validation paired t-test.

在5*2的 cross-validation 中，每次我们把数据集平均分成2份，两个算法分别在某一份上做训练，然后在另外一份数据上做测试，于是我们就得到四个错误率 $err_a^{(1)}, err_a^{(2)}, err_b^{(1)}, err_b^{2}$， $a，b$ 表示不同算法，$1,2$ 表示训练集。 $d^{(i)} = err_a^{(i)} - err_b^{(i)}$，然后计算均值和方差：

$$
\mu = \cfrac{d^{(1)} + d^{(2)}}{2}
$$

$$
s^2 = (d^{(1)} - \mu ) ^2 + (d^{(2)} - \mu)^2
$$

我们给 $\mu s^2$ 加上下标 $i$ 表示第 $i$ 次cross-validation（一共5次）
我们如下计算 t-值

$$
\tilde t = \cfrac{d_1^{(1)}}{\sqrt{\cfrac{1}{5}\sum \limits _{i=1}^5 s_i^2}} ~ t_5
$$

如果我们只有一次检验机会的话，比如 1*2 cross-validation（看起来好诡异的样子）的话，我们也可以使用  **McNemar's test** 方法，也就是

$$
\cfrac{(|err_{ab}-err_{ba}|-1)^2}{err_{ab} + err_{ba}} \sim X_1^2
$$

$ err_{ab} $ 表示 $a$ 预测正确，而 $b$ 预测错误的个数， 相应的 $err_{ba}$ 表示 $b$ 预测正确，而 $a$ 预测错误的个数。

如果我们要在多个测试集上验证多个算法的话，我们可以采用 **Friedman test**.这需要我们使用 *Nemenyi post-hoc test* [Demsar, 2006] 来计算 *critical differnce value*

$$
CD = q_{\alpha} \sqrt{\cfrac{K(K+1)}{6N}}
$$

$N$ 是测试用数据集的数量， $K$ 是算法的数量， $q_{\alpha}$ 是 *critical value*. 两个算法之间如果在【数据集上的表现的rank值】的平均 的差大于 $CD$ 值的话，就认为有显著差异。

我个人认为后面介绍的 **critical difference diagram** 会更加直观。就是每个算法一个bar，bar的中心中心是该算法在所有数据集上表现的rank值的平均，bar的宽度是 *critical difference value*.