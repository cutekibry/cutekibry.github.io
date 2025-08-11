---
title: 杂文：集成学习简介
date: 2023-07-30
category: 
- 学习
tags:
- 杂文
excerpt: 非常简单的关于 Bagging, Stacking, Boosting 的介绍。
---

## 定义
集成学习一般指用若干个弱模型（在二分类问题中指正确率略高于 $50\%$ 的模型）组成一个强模型，以此减小偏差和方差、缓和过拟合的技术。

此处应集成多个弱模型，若集成多个强模型，容易导致强模型模式极其相近，导致难以解决的过拟合；弱模型模式极其相近则导致欠拟合，然而欠拟合较容易解决——只需要提升弱模型的多样化即可。

常见的弱模型如：只有两到三层的决策树；只使用一部分特征训练的决策树等。

## 常见算法
**下面假设问题为简单二分类问题，输入 $\boldsymbol x \in \R^n$，输出 $y \in \{-1, 1\}$，训练算法 $\mathcal L$ 给出的模型输出为 $-1$ 或 $1$**。

### Bagging
Bagging 算法对每次训练一个子模型时，都随机抽样一部分数据 $\mathcal D_t \subseteq \mathcal D$，并以这一小部分数据训练子模型 $h_t = \mathcal L(\mathcal D_t)$，其中 $\mathcal L$ 是学习算法。最后使用所有模型输出的**简单算术**平均数作为输出。

Bagging 的实现是简易且可并行的。

### Stacking
Stacking 算法先训练若干个弱学习算法 $h_t$，然后将预测的分类 $h_t(\boldsymbol x)$ 作为新特征，再运行一个算法得到最终分类算法 $h'$。

具体地，该算法分为两个部分进行：

1. 第一层，用 $\mathcal L_1, \mathcal L_2, \ldots, \mathcal L_T$ 等 $T$ 种学习算法学习 $T$ 个模型 $h_t = \mathcal L_t(\mathcal D)$；
2. 第二层，对每个数据 $i = 1 \ldots m$，获得 $T$ 种模型的输出 $h_t(\boldsymbol x_i)$ 作为新的特征 $\boldsymbol z_i = (h_1(\boldsymbol x_i), h_2(\boldsymbol x_i), \ldots, h_T(\boldsymbol x_i))$，然后以 $\mathcal D' = \{(\boldsymbol z_i, y_i)\}$ 作为新训练集训练最终模型 $h' = \mathcal L'(\mathcal D')$。

最后使用 $h'(h_1(\boldsymbol x), \ldots, h_T(\boldsymbol x))$ 作为输出。

Stacking 在每一层各个模型 $h_i$ 的计算是可并行的。

### Boosting
Boosting 算法会根据第 $t$ 个模型 $h_t$ 的表现调整数据权重（即重要程度），并以此计算 $h_{t + 1}$。最后将所有模型输出的**加权**平均数作为输出，模型的加权会因模型准确率不同而不同。

Boosting 又分为 AdaBoost、GradientBoost 和 XGBoost。

#### AdaBoost
AdaBoost 是最早的 Boosting 算法，其理念是将降低正确数据的权值、提高错误数据的权值，使得每次模型改变时能够主动拟合新的数据。

记 $w_t(i)$ 是第 $t$ 个模型中，第 $i$ 个数据的权值。

算法流程如下：

1. 令 $w_1(i) = \frac 1m$。
2. 令 $t = 1 \ldots T$：
   1. 训练子模型 $h_t = \mathcal L(\mathcal D, w_t)$
   2. 计算加权错误率 $\epsilon_t = \sum_{i = 1}^m w_t(i) \cdot [h_t(\boldsymbol x_i) \neq y_i]$
   3. 计算子模型权值 $\alpha_t = \frac 12 \ln \frac {1 - \epsilon_t}{\epsilon_t}$
   4. 计算新的数据权值 $w_{t + 1}(i) = \frac {w_t(i) \exp(-\alpha_ty_ih_t(\boldsymbol x_i))}{\lVert w_t \rVert_1}$

输出为 $\operatorname{sign} f(\boldsymbol x) = \operatorname{sign}\sum_{t = 1}^T \alpha_t h_t(\boldsymbol x)$。

#### GradientBoost
GradientBoost 的理念是让 $h_{t + 1}$ 拟合 $H_t$ 的残差 $(\boldsymbol x_i, y_i - H_t(\boldsymbol x_i))$，然后将总模型进行调整，得到新的总模型 $H_{t + 1}(\boldsymbol x) = H_t(\boldsymbol x) + \eta h_{t + 1}(\boldsymbol x)$。

输出为 $\operatorname{sign} H(\boldsymbol x) = \operatorname{sign} \sum_{i = 1}^T \eta^{T + 1 - i} h_i(\boldsymbol x)$。

此处 $y_i - H_t(\boldsymbol x_i)$ 其实就是 MSE 的负梯度的一半。

#### XGBoost
XGBoost 是目前使用最多的 Boosting 方法，是对 GradientBoost 的改进。

实现较为复杂，略去。