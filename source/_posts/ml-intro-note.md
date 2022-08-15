---
title: Coursera 吴恩达机器学习入门课笔记（更新至第一课）
date: 2022-08-13
category: 
- 学习
---

吴恩达在 Coursera 上的机器学习入门课的笔记。只给自己看，所以写得非常简略。

使用的是较新的 2022 版本（和之前的不太一样），双语可见 [BV1Pa411X76s](//www.bilibili.com/video/BV1Pa411X76s)。

<!-- more -->

## 线性回归（Linear Regression）
设特征数为 $n$，样本数为 $m$。

模型：$f_{\vec w, b}(\vec x) = \left(\sum_{i = 1}^n w_ix_i\right) + b$

损失函数：

$$L(f_{\vec w, b}(\vec x^{(i)}), y^{(i)}) = (f_{\vec w, b}(\vec x^{(i)}) - y^{(i)})^2$$

费用函数：

$$J(\vec w, b) = \frac 1{2m} \left(\sum_{i = 1}^m \left(f_{\vec w, b}(\vec x^{(i)}) - y^{(i)} \right)^2 \right) + \frac \lambda{2m} \sum_{i = 1}^n w_i^2$$

此处 $\frac \lambda{2m} \sum_{i = 1}^n w_i^2$ 采用了**正则化**（Regularization），能使得 $w_i$ 的值减小，从而抑制**过拟合**（Overfitting）。

## 梯度下降（Gradient Descent）
$$
\begin{aligned}
    \frac {\delta J(\vec w, b)}{\delta w_j} &= \frac 1m \left(\sum_{i = 1}^m \left( f_{\vec w, b}(\vec x^{(i)}) - y^{(i)} \right) x_j^{(i)} \right) + \frac 1m w_j \\ 
\end{aligned}
$$

$$
\begin{aligned}
    \frac {\delta J(\vec w, b)}{\delta b} &= \frac 1m\sum_{i = 1}^m \left( f_{\vec w, b}(\vec x^{(i)}) - y^{(i)} \right) \\ 
\end{aligned}
$$

## 特征缩放（Feature Scaling）
* Mean Normalization：$x' = \frac {x - \mu}{\max - \min}$
* Z-Score Normalization：$x' = \frac {x - \mu}\sigma$

目标是让 $x$ 尽可能趋近于 $[-1, 1]$ 分布。

## 逻辑回归（Logistic Regression）
Sigmoid：$f(x) = \frac 1{1 + e^{-x}}$

ReLU：$f(x) = \max\{0, x\}$

模型：$f_{\vec w, b}(\vec x) = \text{sigmoid}\left(\left(\sum_{i = 1}^n w_ix_i\right) + b\right)$

也可以换成 ReLU。

损失函数：

$$
L(f_{\vec w, b}(\vec x^{(i)}), y^{(i)}) = 
\begin{cases}
    -\log\left(f_{\vec w, b}(\vec x^{(i)})\right), y^{(i)} = 1\\
    -\log\left(1 - f_{\vec w, b}(\vec x^{(i)})\right), y^{(i)} = 0 \\
\end{cases}
$$

实际操作中，使用该损失函数比直接套用线性回归的损失函数更优。