---
title: 《概率论》笔记
date: 2023-06-10
category: 
- 学习
tags:
- 文化课
excerpt: 怎么这么多课……
---

$$\phi'(x) = \frac 1{\sqrt {2\pi}\sigma} e^{-\frac {(x - \mu)^2}{2\sigma^2}}$$

## 第四章 数字特征
### 4.1 期望
柯西不等式：

$$E^2(XY) \leq EX^2EY^2$$

### 4.2 方差
$$DX = EX^2 - E^2X$$

**切比雪夫不等式**：$\forall \epsilon > 0$，

$$P(|X - EX| \geq \epsilon) \leq \frac {DX}{\epsilon^2}$$

等价于

$$P(|X - EX| < \epsilon) \geq 1 - \frac {DX}{\epsilon^2}$$

也就是说，$X$ 与其期望 $EX$ 的距离小于 $\epsilon$ 的概率大于 $1 - \frac {DX}{\epsilon^2}$。

### 4.3 矩

### 4.4 协方差与相关系数
相关系数 $|\rho| \leq 1$，且 $|\rho| = 1$ 当且仅当 $Y = aX + b$。

### 4.5 条件数学期望

## 第五章 大数定律和中心极限定理
### 5.1 大数定律

## 第六章 数理统计的基本概念
### 6.1 总体与样本
**经验分布函数**：取样本 $(X_1, \ldots, X_n)$，则经验分布函数为

$$F_n(x) = \sum_{X_i \leq x} \frac 1n$$

**统计量**：对样本 $(X_1, \ldots, X_n)$，若有连续函数 $g(x_1, \ldots, x_n)$ 中不含位置参数，则 $g(X_1, \ldots, X_n)$ 为统计量。统计量是随机变量。

样本方差：$S^2 = \frac 1{n - 1}\sum_{i = 1}^n (X_i - \bar X)^2$。**注意：这里的分母是 $(n - 1)$**。

### 6.2 抽样分布
**$\chi^2$ 分布**：设样本 $(X_1, \ldots, X_n)$ 是 $N(0, 1)$ 的样本，则称

$$\chi^2 = \sum_{i = 1}^n X_i^2$$

记为 $\chi^2 \sim \chi^2(n)$。

$E\chi^2 = n, D\chi^2 = 2n$，且具有可加性。

---

**$t$ 分布**：设 $X \sim N(0, 1), Y \sim \chi^2(n)$ 且相互独立，则称

$$T = \frac X{\sqrt {Y/n}}$$

为 $T \sim t(n)$。

---

**$F$ 分布**：设 $X \sim \chi^2(n_1), Y \sim \chi^2(n_2)$ 且相互独立，则称

$$F = \frac {X / n_1}{Y / n_2}$$

为 $F \sim F(n_1, n_2)$。

---

正态分布取 $n$ 个样，则

$$\overline X \sim N(\mu, \frac {\sigma^2}n)$$

$$(n - 1)\frac {S^2}{\sigma^2} \sim \chi^2(n - 1)$$

且 $\overline X, S^2$ 独立。

## 第七章 参数估计
### 7.2 矩估计法和极大似然估计法
