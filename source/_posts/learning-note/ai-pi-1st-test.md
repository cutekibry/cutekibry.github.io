---
title: AI 派 2023 年暑期招生第一次测试（理论题）
date: 2023-07-27
category: 
- 学习
excerpt: 理论题学习笔记。<br />感觉考得很深……
---

## 2 数学原理之LR

*出题人：吴佳黎*

​	在后续的问题中，你可能需要使用逻辑回归解决问题。由于逻辑回归的原理十分经典，这部分内容将考察你对其中数学原理的理解程度，你需要使用简单的数学公式回答以下问题：

### 具体要求

1. 给出利用**梯度下降法**求解**逻辑回归**的**前反向公式推导**，在这一部分中，你只需要考虑**最朴素的二分类逻辑回归模型**。
2. 假设标签集不是 {0,1} 而是 **{1,-1}**，将会有什么变化，请给出推导。
3. 在问题2.1的基础上，即标签集为 {0,1} 的情况，分别增加**L1正则化**和**L2正则化**，公式和模型效果分别会有什么变化，请给出推导。
4. 给出**核逻辑回归的对偶形式**。
5. （选做）如果你愿意，给出利用**二阶优化算法**如牛顿法求解逻辑回归的公式推导，只需要考虑最朴素的逻辑回归模型。

### 提示

- 除了问题2.4以外，请给出关键推导过程，仅仅给出结果是无效的。
- 如果你愿意，问题2.4可以给出关键性的公式推导或解释。
- 建议使用 LaTeX 语法书写数学公式，例如你可以很容易查询到，梯度公式如下（假设数据下标从1到m）:

$$
g(\boldsymbol w)=\frac{\partial J(\boldsymbol w)}{\partial \boldsymbol w}=\sum_{i=1}^m(\sigma(\boldsymbol w^\top\boldsymbol x_i)-y_i)\boldsymbol x_i
$$

- 参考资料：李航.统计学习方法[M].北京:清华大学出版社,2019.5.1

## 2 Solution
### 2.1.
Q：给出利用**梯度下降法**求解**逻辑回归**的**前反向公式推导**，在这一部分中，你只需要考虑**最朴素的二分类逻辑回归模型**。

A：记模型参数为 $\boldsymbol w \in \R^m$，输入为 $\boldsymbol x_i \in \R^m$，正确输出为 $y_i \in \{0, 1\}$，模型输出为 $f(\boldsymbol x_i) = \hat P(y_i = 1) \in [0, 1]$。

偏置 $b$ 可以在所有 $\boldsymbol x_i$ 后新加一维 $x_{i, m + 1} = 1$，因此下面的推导不考虑偏置。

#### 2.1.1. 前向传播
根据定义得：

$$f(\boldsymbol x_i) = \hat P(y_i = 1) = \frac 1{1 + \exp(-\boldsymbol w \cdot \boldsymbol x_i)}$$

$$\hat P(y_i = 0) = \frac 1{1 + \exp(\boldsymbol w \cdot \boldsymbol x_i)}$$

#### 2.1.2. 后向传播
极大似然估计法给出的似然函数为：

$$
\begin{aligned}
    L(\boldsymbol w) &= \ln \prod_{i = 1}^n f(\boldsymbol x_i)^{y_i} [1 - f(\boldsymbol x_i)]^{1 - y_i} \\
    &= \sum_{i = 1}^n y_i\ln f(\boldsymbol x_i) + (1 - y_i)\ln[1 - f(\boldsymbol x_i)] \\
\end{aligned}
$$

计算偏导数 $\frac {\partial L(\boldsymbol w)}{\partial \boldsymbol w}$：

$$
\begin{aligned}
    \frac {\partial L(\boldsymbol w)}{\partial w_j} &= \sum_{i = 1}^n y_i\frac {\partial \ln f(\boldsymbol x_i)}{\partial w_j} + (1 - y_i) \frac {\partial \ln [1 - f(\boldsymbol x_i)]}{\partial w_j} \\
    &= -\sum_{i = 1}^n y_i\frac {\partial [1 + \exp(-\boldsymbol w \cdot \boldsymbol x_i)]}{\partial w_j} + (1 - y_i) \frac {\partial [1 + \exp(\boldsymbol w \cdot \boldsymbol x_i)]}{\partial w_j} \\
    &= \sum_{i = 1}^n [y_i\exp(-\boldsymbol w \cdot \boldsymbol x_i) - (1 - y_i) \exp(\boldsymbol w \cdot \boldsymbol x_i)]x_{i, j} \\
\end{aligned}
$$

因此

$$\frac {\partial L(\boldsymbol w)}{\partial \boldsymbol w} = \sum_{i = 1}^n [y_i\exp(-\boldsymbol w \cdot \boldsymbol x_i) - (1 - y_i) \exp(\boldsymbol w \cdot \boldsymbol x_i)] \boldsymbol x_i$$

该模型的损失函数为

$$l(\boldsymbol w) = -\frac 1n L(\boldsymbol w)$$

梯度下降给出的反向传播公式：

$$\boldsymbol w' = \boldsymbol w - \eta \frac {\partial l(\boldsymbol w)}{\partial \boldsymbol w} = \boldsymbol w + \frac \eta n \frac {\partial L(\boldsymbol w)}{\partial \boldsymbol w}$$

其中 $\eta > 0$ 是学习率。

### 2.2
**Q**：假设标签集不是 {0,1} 而是 **{1,-1}**，将会有什么变化，请给出推导。

**A**：

类似地，前向传播公式：

$$f(\boldsymbol x_i) = \hat P(y_i = 1) = \frac 1{1 + \exp(-\boldsymbol w \cdot \boldsymbol x_i)}$$

$$\hat P(y_i = -1) = \frac 1{1 + \exp(\boldsymbol w \cdot \boldsymbol x_i)}$$


似然函数：

$$
\begin{aligned}
    L(\boldsymbol w) &= \ln \prod_{i = 1}^n f(\boldsymbol x_i)^{\frac {y_i + 1}2} [1 - f(\boldsymbol x_i)]^{\frac {1 - y_i}2} \\
    &= \sum_{i = 1}^n \frac {y_i + 1}2\ln f(\boldsymbol x_i) + \frac {1 - y_i}2 \ln[1 - f(\boldsymbol x_i)] \\
\end{aligned}
$$

同理可得：

$$\frac {\partial L(\boldsymbol w)}{\partial \boldsymbol w} = \sum_{i = 1}^n [\frac {y_i + 1}2\exp(-\boldsymbol w \cdot \boldsymbol x_i) - \frac {1 - y_i}2 \exp(\boldsymbol w \cdot \boldsymbol x_i)] \boldsymbol x_i$$

$$\boldsymbol w' = \boldsymbol w + \frac \eta n \frac {\partial L(\boldsymbol w)}{\partial \boldsymbol w}$$


与 **2.1** 中的式子相比，只有常数 $y_i$ 的计算发生了变化。

### 2.3
**Q**：在问题2.1的基础上，即标签集为 {0,1} 的情况，分别增加**L1正则化**和**L2正则化**，公式和模型效果分别会有什么变化，请给出推导。

**A**：增加 L1 和 L2 正则化后，前向传播公式不变。

损失函数变为

$$l(\boldsymbol w) = -\frac 1n L(\boldsymbol w) + \alpha \lVert \boldsymbol w \rVert + \beta \lVert \boldsymbol w \rVert_2^2$$

计算偏导：

$$
\begin{aligned}
    \frac {\partial l(\boldsymbol w)}{\partial \boldsymbol w} = -\frac 1n \frac {\partial L(\boldsymbol w)}{\partial \boldsymbol w} + \alpha \operatorname{sign} \boldsymbol w + 2\beta \boldsymbol w
\end{aligned}
$$

反向传播公式：

$$\boldsymbol w' = \boldsymbol w - \eta\left[\alpha \operatorname{sign} \boldsymbol w + 2\beta\boldsymbol w - \frac 1n \frac {\partial L(\boldsymbol w)}{\partial \boldsymbol w}\right]$$

其中 $\operatorname{sign}(x)$ 是符号函数：

$$
\operatorname{sign}(x) =
\begin{cases}
    -1, &x < 0 \\
    0,  &x = 0 \\
    1,  &x > 0 \\
\end{cases}
$$

### 2.4
**Q**：给出**核逻辑回归的对偶形式**。

**A**：核逻辑回归原问题：

$$
\begin{aligned}
    \min_{\boldsymbol w, b} \quad \sum_{i = 1}^n y_i \ln \{1 + \exp[-K(\boldsymbol w, \boldsymbol x_i) - b]\} + (1 - y_i) \ln \{1 + \exp[K(\boldsymbol w, \boldsymbol x_i) + b]\} \\ 
\end{aligned}
$$

其中 $K(\boldsymbol x, \boldsymbol z), K: (\R^m, \R^m) \rightarrow \R$ 是核函数。

因为没有条件约束，对偶问题即为原问题。

## 3 数学原理之SVM

*出题人：吴佳黎*

​	在后续的问题中，你可能需要使用支持向量机解决问题。SVM的部分原理如下。

- 支持向量机原问题：

$$
\min_{\boldsymbol{w},b}f(\boldsymbol{w})=\frac{1}{2}\|\boldsymbol{w}\|_2^2,\quad\mathrm{ s.t. }\ y_i(\boldsymbol{w}^\top\boldsymbol{x}_i+b)\ge 1,\forall 1\le i\le m
$$

- 支持向量机对偶问题：

$$
\max_{\boldsymbol{\alpha}\ge 0}g(\boldsymbol{}\alpha) =-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m \alpha_i\alpha_j y_iy_j\boldsymbol x_i^\top\boldsymbol{x}_j+\sum_{i=1}^m \alpha_i,\quad \mathrm{s.t.}\ \boldsymbol{y}^\top\boldsymbol\alpha=0
$$

你需要使用简单的数学公式回答以下问题：

### 具体要求

1. 给出支持向量机最大间隔准则原问题的推导，解释分类超平面和支持超平面的含义。
2. 给出对偶问题的推导，并回答问题：强对偶性在支持向量机中始终成立吗？
3. 根据前面的结果推导**SVM的KKT条件**。
4. 如果数据非线性可分怎么办？给出修改后的原问题和对偶问题。
5. 在问题3.4的基础上，若**允许少量样本破坏约束**，应增加怎样的损失函数，请给出修改后的原问题和对偶问题。

### 提示

- 可以直接使用点到超平面的距离公式。可以认为你已经掌握拉格朗日乘子法，无需对此再进行证明推导。
- 问题3.2中回答问题只需要回答是否，如果你愿意，也可以给出简单的解释。
- 除了问题3.4与问题3.5以外，请给出推导过程，仅仅给出结果是无效的。
- 如果你愿意，问题3.4与问题3.5可以给出关键性的公式推导或解释。
- 建议使用 LaTeX 语法书写数学公式。
- 参考资料：李航.统计学习方法[M].北京:清华大学出版社,2019.5.1

## 3 Solution
### 3.1
**Q**：给出支持向量机最大间隔准则原问题的推导，解释分类超平面和支持超平面的含义。

**A**：

#### 原问题推导
SVM 原问题：

$$
\begin{aligned}
    & \max_{\boldsymbol w, b} \quad \gamma \\
    & \text{s.t.} \quad y_i\frac {\boldsymbol w \cdot \boldsymbol x_i + b}{\lVert \boldsymbol w \rVert} \geq \gamma, \quad \forall 1 \leq i \leq n \\
\end{aligned}
$$

令 $\gamma' = \gamma \lVert \boldsymbol w \rVert$，转化为

$$
\begin{aligned}
    & \max_{\boldsymbol w, b} \quad \frac {\gamma'}{\lVert \boldsymbol w \rVert} \\
    & \text{s.t.} \quad y_i(\boldsymbol w \cdot \boldsymbol x_i + b) \geq \gamma', \quad  \forall 1 \leq i \leq n \\
\end{aligned}
$$

可以注意到，$\gamma'$ 的取值不影响问题的解：若 $\gamma_1' = \lambda \gamma_2'$，则若取 $\boldsymbol w_1^* = \lambda \boldsymbol w_2^*, b_1^* = \lambda b_2^*$，有

$$\frac {\gamma_1'}{\lVert \boldsymbol w_1^* \rVert} = \frac {\gamma_2'}{\lVert \boldsymbol w_2^* \rVert}$$

$$y_i(\boldsymbol w_1^* \cdot \boldsymbol x_i + b_1^*) \geq \gamma'_1 \Leftrightarrow y_i(\boldsymbol w_2^* \cdot \boldsymbol x_i + b_2^*) \geq \gamma'_2$$

因此不妨取 $\gamma' = 1$，问题转化为

$$
\begin{aligned}
    & \max_{\boldsymbol w, b} \quad \frac 1{\lVert \boldsymbol w \rVert} \\
    & \text{s.t.} \quad y_i(\boldsymbol w \cdot \boldsymbol x_i + b) \geq 1, \quad  \forall 1 \leq i \leq n \\
\end{aligned}
$$

即

$$
\begin{aligned}
    & \min_{\boldsymbol w, b} \quad \frac 12\lVert \boldsymbol w \rVert^2 \\
    & \text{s.t.} \quad y_i(\boldsymbol w \cdot \boldsymbol x_i + b) \geq 1, \quad  \forall 1 \leq i \leq n \\
\end{aligned}
$$

#### 分类超平面和支持超平面
**分类超平面**既是 SVM 参数对应的超平面 $\boldsymbol w \cdot \boldsymbol x + b = 0$。SVM 根据点在分类超平面的哪一侧为其预测分类。

**支持超平面**既是支持向量所在的超平面，即 $y_i(\boldsymbol w \cdot \boldsymbol x_i + b) = 1$。根据 $y_i \in \{-1, 1\}$ 取值不同，存在两个支持超平面 $H_1: \boldsymbol w \cdot \boldsymbol x_i + b = 1$ 和 $H_2: \boldsymbol w \cdot \boldsymbol x_i + b = -1$。求解分类超平面时，有且只有支持超平面上的点会起作用。

### 3.2
**Q**：给出对偶问题的推导，并回答问题：强对偶性在支持向量机中始终成立吗？

**A**：

#### 3.2.1 对偶问题的推导
根据拉格朗日乘子法，定义拉格朗日函数

$$L(\boldsymbol w, b, \boldsymbol a) = \frac 12 \lVert \boldsymbol w \rVert^2 - \sum_{i = 1}^n \alpha_i y_i (\boldsymbol w \cdot \boldsymbol x_i + b) + \sum_{i = 1}^n \alpha_i$$

可得原问题的对偶问题为

$$\max_{\boldsymbol \alpha \geq \boldsymbol 0} \min_{\boldsymbol w, b} L(\boldsymbol w, b, \boldsymbol a)$$

简单推导：

$$
\begin{aligned}
    L(\boldsymbol w, b, \boldsymbol a) &= \frac 12 \lVert \boldsymbol w \rVert^2 - \sum_{i = 1}^n \alpha_i [y_i (\boldsymbol w \cdot \boldsymbol x_i + b) - 1] \\
    &= \frac 12 \lVert \boldsymbol w \rVert^2 - \left(\sum_{i = 1}^n \alpha_i y_i \boldsymbol x_i\right) \cdot \boldsymbol w - \left(\sum_{i = 1}^n \alpha_i y_i \right) b + \sum_{i = 1}^n \alpha_i \\
\end{aligned}
$$

当 $\boldsymbol a$ 确定时，$\left(\sum_{i = 1}^n \alpha_i y_i \boldsymbol x_i\right), \left(\sum_{i = 1}^n \alpha_i y_i \right), \sum_{i = 1}^n \alpha_i$ 都是常数。不妨记 $\boldsymbol C_1(\boldsymbol \alpha) = \left(\sum_{i = 1}^n \alpha_i y_i \boldsymbol x_i\right), C_2(\boldsymbol \alpha) = \left(\sum_{i = 1}^n \alpha_i y_i \right)$，则对偶问题等价于

$$\max_{\boldsymbol \alpha \geq \boldsymbol 0} \left[\left(\sum_{i = 1}^n \alpha_i\right) + \min_{\boldsymbol w, b} \left(\frac 12 \lVert \boldsymbol w \rVert^2 - \boldsymbol C_1(\boldsymbol \alpha) \cdot \boldsymbol w - C_2(\boldsymbol \alpha) b\right)\right]$$

记函数

$$f(\boldsymbol w, b) = \left(\frac 12 \lVert \boldsymbol w \rVert^2 - \boldsymbol C_1(\boldsymbol \alpha) \cdot \boldsymbol w - C_2(\boldsymbol \alpha) b\right)$$

要求解 $\min_{\boldsymbol w, b} f(\boldsymbol w, b)$，考虑计算偏导为 $0$，有

$$
\begin{cases}
    \frac {\partial f(\boldsymbol w, b)}{\partial \boldsymbol w} = \boldsymbol w - \boldsymbol C_1(\boldsymbol \alpha) = 0, \\
    \frac {\partial f(\boldsymbol w, b)}{\partial b} = - C_2(\boldsymbol \alpha) = 0 
\end{cases}
$$

得

$$
\begin{cases}
    \boldsymbol w = \boldsymbol C_1(\boldsymbol \alpha), \\
    C_2(\boldsymbol \alpha) = 0
\end{cases}
$$

代入得 $\min_{\boldsymbol w, b} f(\boldsymbol w, b) = f(\boldsymbol C_1(\boldsymbol \alpha), 0) = -\frac 12 \lVert \boldsymbol C_1(\boldsymbol \alpha) \rVert^2$。

故对偶问题即为

$$
\begin{aligned}
    & \max_{\boldsymbol \alpha \geq \boldsymbol 0} \left[\left(\sum_{i = 1}^n \alpha_i\right) - \frac 12 \lVert \boldsymbol C_1(\boldsymbol \alpha) \rVert^2\right] \\
    & \text{s.t. } C_2(\boldsymbol \alpha) = 0
\end{aligned}
$$

拆开式子得

$$
\begin{aligned}
    & \max_{\boldsymbol \alpha \geq \boldsymbol 0} \left(\sum_{i = 1}^n \alpha_i - \frac 12 \sum_{i = 1}^n \sum_{j = 1}^n \alpha_i \alpha_j y_i y_j \boldsymbol x_i \cdot \boldsymbol x_j \right) \\
    & \text{s.t. } \boldsymbol \alpha \cdot \boldsymbol y  = 0
\end{aligned}
$$

即为原问题的对偶形式。

#### 3.2.2 是否满足强对偶性
满足。

首先，显然地，$f(\boldsymbol w)$ 和 $[y_i(\boldsymbol{w}^\top\boldsymbol{x}_i+b) - 1]$ 是关于 $\boldsymbol w$ 的凸函数，且因为数据集线性可分，所以存在严格可行解。根据定理可知，原问题和对偶问题均存在最优解，且最优解对应函数的值相等。

### 3.3
**Q**：根据前面的结果推导 **SVM 的 KKT 条件**。

**A**：因为问题满足了 **3.2.2** 的条件，所以可知 $\boldsymbol w^*, b^*, \boldsymbol a^*$ 是原问题和对偶问题的最优解，当且仅当满足 KKT 条件

$$
\frac {\partial L(\boldsymbol w^*, b^*, \boldsymbol a^*)}{\partial \boldsymbol w} = \boldsymbol w^* - \sum_{i = 1}^n \alpha_i y_i \boldsymbol x_i = \boldsymbol 0 
$$

$$
\frac {\partial L(\boldsymbol w^*, b^*, \boldsymbol a^*)}{\partial b} = - \sum_{i = 1}^n \alpha_i y_i = 0 
$$

$$
\boldsymbol [y_i(\boldsymbol w^* \cdot \boldsymbol x_i + b^*) - 1]  a^*_i = 0, \quad \forall 1 \leq i \leq n 
$$

$$
y_i(\boldsymbol{w}^* \cdot \boldsymbol{x}_i+b^*)\ge 1, \quad \forall 1 \leq i \leq n
$$

$$\boldsymbol a^* \geq \boldsymbol 0$$

### 3.4
**Q**：如果数据非线性可分怎么办？给出修改后的原问题和对偶问题。

**A**：若不允许样本破坏约束，则可以考虑将输入的维度进行变换。

具体地，记 $\Chi$ 是输入空间，$\Eta$ 是特征空间。对函数 $K(\boldsymbol x, \boldsymbol z), K: \Chi^2 \rightarrow \R$，若存在映射 $\boldsymbol \phi: \Chi \rightarrow \Eta$ 使得

$$K(\boldsymbol x, \boldsymbol z) = \boldsymbol \phi(\boldsymbol x) \cdot \boldsymbol \phi(\boldsymbol y)$$

则称 $K(\boldsymbol x, \boldsymbol z)$ 是**核函数**。

在给定核函数的前提下，我们可以将 SVM 问题进行修改，对偶问题变为

$$
\begin{aligned}
    & \max_{\boldsymbol \alpha \geq \boldsymbol 0} \left(\sum_{i = 1}^n \alpha_i - \frac 12 \sum_{i = 1}^n \sum_{j = 1}^n \alpha_i \alpha_j y_i y_j K(\boldsymbol x_i, \boldsymbol x_j) \right) \\
    & \text{s.t. } \boldsymbol \alpha \cdot \boldsymbol y  = 0
\end{aligned}
$$

相应地，原问题变为

$$
\begin{aligned}
    & \min_{\boldsymbol w, b} \quad \frac 12\lVert \boldsymbol w \rVert^2 \\
    & \text{s.t.} \quad y_i(K(\boldsymbol w, \boldsymbol x_i) + b) \geq 1, \quad  \forall 1 \leq i \leq n \\
\end{aligned}
$$

当 $\{\boldsymbol x_i\}$ 在核函数对应的特征空间 $\Eta$ 中线性可分时，就存在最优解。

### 3.5
**Q**：在问题3.4的基础上，若**允许少量样本破坏约束**，应增加怎样的损失函数，请给出修改后的原问题和对偶问题。

**A**：

由于非线性可分，所以部分点不能满足 $y_i(\boldsymbol w \cdot \boldsymbol x_i + b) \geq 1$。引入松弛变量 $\xi_i \geq 0$，不等式变为 $y_i(\boldsymbol w \cdot \boldsymbol x_i + b) \geq 1 - \xi_i$，同时给函数添加一项 $+C\sum_{i = 1}^n \xi_i$ 作为修正，防止 $\xi_i$ 过大。修改后的原问题：

$$
\begin{aligned}
    & \min_{\boldsymbol w, b, \boldsymbol \xi} \quad \frac 12\lVert \boldsymbol w \rVert^2 + C\sum_{i = 1}^n \xi_i \\
    & \text{s.t.} \quad y_i(K(\boldsymbol w, \boldsymbol x_i) + b) \geq 1 - \xi_i, \quad  \forall 1 \leq i \leq n \\
    & \qquad\  \boldsymbol \xi \geq \boldsymbol 0 \\
\end{aligned}
$$

与 **3.2.1** 同理，可知对偶问题为

$$
\begin{aligned}
    & \max_{\boldsymbol \alpha} \quad \sum_{i = 1}^n \alpha_i - \frac 12 \sum_{i = 1}^n \sum_{j = 1}^n \alpha_i \alpha_j y_i y_j K(\boldsymbol x_i, \boldsymbol x_j) \\
    & \text{s.t. } \quad \boldsymbol \alpha \cdot \boldsymbol y = 0 \\
    & \qquad\ \ 0 \leq \alpha_i \leq C, \quad \forall 1 \leq i \leq n \\
\end{aligned}
$$
