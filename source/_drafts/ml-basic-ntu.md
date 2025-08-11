---
title: 台大林轩田《机器学习基石》笔记
date: 2022-08-13
category: 
- 学习
---

[BV1aR4y1m7xU](//www.bilibili.com/video/BV1aR4y1m7xU)。

全是理论，要头秃了……

<!-- more -->

## 2. Learning to Answer Yes No - PLA
### 介绍
**PLA**（**Perceptron Linear Algorithm，线性感知机算法**）共有 $n + 1$ 个参数 $w_0, w_1, w_2, \ldots, w_n$。

对于输入向量 $\boldsymbol x$，输出 $\hat y = \text{sign}(\boldsymbol w \cdot \boldsymbol x)$。注意 $x_0 = 1$ 是我们额外加上的，这会使得 $w_0$ 在函数中起到**阈值**的作用，即

$$\hat y = \text{sign}\left(\sum_{i = 1}^n w_ix_i + w_0\right)$$

### 训练方法
采用调整法。

每次找到一个 $n(T)$ 使得 $\text{sign}(\boldsymbol w_T \cdot \boldsymbol x_{n(T)} + T) \neq y_{n(T)}$，令 $\boldsymbol w_{T + 1} \leftarrow \boldsymbol w_T + y_{n(T)}\boldsymbol x_{n(T)}$。此时第 ${n(T)}$ 组数据被修正。

当训练集线性可分（即存在一个 $w$ 使得所有训练集都被正确划分）时，PLA 会在一定时间后停止，否则会无限循环。

### 证明
设 $\boldsymbol w_f$ 能使得所有训练集都被正确划分，则这相当于 $p = \min_i \{y_i \boldsymbol w_f \cdot \boldsymbol x_i\} > 0$。不妨令 $\Vert \boldsymbol w_f \Vert = 1$。

再记 $R = \max_i \{\Vert \boldsymbol x_i \Vert\}$ 表示训练集半径。

令 $\boldsymbol w_0 = \boldsymbol 0$。

**引理 2.1** $\boldsymbol w_f \cdot \boldsymbol w_T > Tp$。

**证明 2.1** 通过调整，可以发现

$$
\begin{aligned}
  \boldsymbol w_f \cdot \boldsymbol w_{T + 1} &= \boldsymbol w_f \cdot (\boldsymbol w_T + y_{n(T)} \boldsymbol x_{n(T)}) \\
  &= \boldsymbol w_f \cdot \boldsymbol w_T + y_{n(T)} \boldsymbol w_f \boldsymbol x_{n(T)} \\
  &\geq \boldsymbol w_f \cdot \boldsymbol w_T + \min_i \{y_i \boldsymbol w_f \boldsymbol x_i\} \\
  &= \boldsymbol w_f \cdot \boldsymbol w_T + p \\
\end{aligned}
$$

故 $\boldsymbol w_f \cdot \boldsymbol w_T \geq \boldsymbol w_f \cdot \boldsymbol w_0 + Tp = Tp$。

**引理 2.2** $\Vert\boldsymbol w_T\Vert^2 \leq tR^2$。

**证明 2.2**

$$
\begin{aligned}
  \Vert \boldsymbol w_{T + 1} \Vert^2 &= \Vert \boldsymbol w_T + y_{n(T)}\boldsymbol x_{n(T)} \Vert^2 \\
  &= \Vert \boldsymbol w_T \Vert^2 + 2y_{n(T)}\boldsymbol w_T \cdot \boldsymbol x_{n(T)} + \Vert y_{n(T)} \boldsymbol x_{n(T)} \Vert^2 \\
  &\leq \Vert \boldsymbol w_T \Vert^2 + 0 + \Vert \boldsymbol x_{n(T)} \Vert^2 \\
  &\leq \Vert \boldsymbol w_T \Vert^2 + \max_i \Vert \boldsymbol x_i \Vert^2 \\
\end{aligned}
$$

故 $\Vert \boldsymbol w_T \Vert^2 \leq \Vert \boldsymbol w_0 \Vert^2 + TR^2 = TR^2$。

**定理 2.3** 令 $\boldsymbol w_0 = \boldsymbol 0$，则

$$\frac {\boldsymbol w_f \boldsymbol w_T}{\Vert \boldsymbol w_f\Vert \Vert \boldsymbol w_T \Vert} \geq \sqrt T \cdot \text{constant}$$

**证明 2.3** 

$$
\begin{aligned}
  \frac {\boldsymbol w_f \boldsymbol w_T}{\Vert \boldsymbol w_f\Vert \Vert \boldsymbol w_T \Vert} &\geq \frac {Tp}{\Vert \boldsymbol w_f \Vert \sqrt T R}  \\
  &\geq \sqrt T \cdot \frac pR  \\
\end{aligned}
$$

此外，注意到 $\frac {\boldsymbol w_f \boldsymbol w_T}{\Vert \boldsymbol w_f\Vert \Vert \boldsymbol w_T \Vert} = \cos \theta \leq 1$，因此也有

$$
\begin{aligned}
  1&\geq \sqrt T \cdot \frac pR  \\
  T&\leq \frac {R^2}{p^2}  \\
\end{aligned}
$$

### 总结
PLA 的优点：容易实现，支持多维，实践中速度很快

PLA 的缺点：需要训练集线性可分，且不能确定需要多久才会完全拟合（尽管一般很快）

### 改进
一个改进：我们不需要找到完美分割的，只需要正确率足够高就行，即

$$\boldsymbol w_f = \argmin_{\boldsymbol w} \sum_{n = 1}^N [y_n \neq \text{sign}(\boldsymbol w \cdot \boldsymbol x_n)]$$

——然而这个问题是 NP-Hard 问题，没有什么有效的算法。

一个可能的改进是，在每次修改时分别计算 $w_T$ 和 $w_{T + 1}$ 的正确率，若 $w_{T + 1}$ 比 $w_T$ 正确率更高则保留，否则舍弃；迭代若干次后返回答案。此算法也称为**口袋**（**Pocket**）算法。若训练集线性可分，则口袋算法需要比朴素 PLA 花更多时间，

## 3. Types of Learning
### 数据类型
* Conrete（结构化数据）：其强度和数值有很大关联性（如厨房面积、高度、分数、排名等）
* Raw（原始数据）：像音频、图像这类比较抽象的数据
* Abstract（抽象数据）：像道具编号、用户编号等，数值和其强度几乎没有关联的数据

## 4. Feasibility of Learning
### 天下没有免费的午餐（No Free Lunch）定理
即，没有一种通用的学习算法可以在各种任务中都有很好的表现。

模型在训练集上表现优异，并不能认为在测试集上就一定优异。

### 霍夫丁（Hoeffding）不等式
假设共有 $N$ 个黑白小球，其中黑色占比 $\mu$；再从中取出任意数量的小球，记占比为 $v$，则

$$\mathbb P[|v - \mu| > \epsilon] \leq 2e^{-2\epsilon^2N}$$

**即使我们不清楚 $\mu$ 的具体值，但可以根据 $v, \epsilon$ 来推测其取值**。

也就是说，假设我们有 $N$ 个数据和一个确定的函数 $h: \boldsymbol x \rightarrow y$，则训练集误差 $E_{\text{in}}(h)$ 和测试集误差 $E_{\text{out}}(h)$ 会比较相近，满足

$$\mathbb P[|E_{\text{in}}(h) - E_{\text{out}}(h)| > \epsilon] \leq 2e^{-2\epsilon^2N}$$

当数据够多的时候，可以简单认为 $E_{\text{in}}(h) \approx E_{\text{out}}(h)$。

### 有限函数与坏数据
对单个 $h$ 来说，若 $E_{\text{in}}$ 和 $E_{\text{out}}$ 差别过大，则对应的数据是坏数据。

若有 $M$ 个函数 $h_1, h_2, \ldots, h_M$，则对于一个数据集 $\mathcal D$，有

$$
\begin{aligned}
  &\ \mathbb P_{\mathcal D}[\mathcal D \text{ is bad}] \\
  = &\ \mathbb P_{\mathcal D}[(\mathcal D\text{ for }h_1\text{ is bad}) \lor (\mathcal D\text{ for }h_2\text{ is bad}) \lor \ldots \lor (\mathcal D\text{ for }h_M\text{ is bad})] \\
  \leq &\ \mathbb P_{\mathcal D}[\mathcal D\text{ for }h_1\text{ is bad}] + \mathbb P_{\mathcal D}[\mathcal D\text{ for }h_2\text{ is bad}] + \ldots + \mathbb P_{\mathcal D}[\mathcal D\text{ for }h_M\text{ is bad}] \\
  \leq &\ M \cdot 2 \cdot \exp(-2\epsilon^2 N) \\
\end{aligned}
$$

因此有

$$\mathbb P[|E_{\text{in}}(h) - E_{\text{out}}(h)| > \epsilon] \leq 2M\exp(-2\epsilon^2N)$$

可见当 $M$ 不够大，且 $N$ 足够大时，该数据集为好数据的概率很高。

## 5. Training versus Testing
### 无限函数和坏数据
实际情况中这样的 $h$ 有无数个，显然不能让 $M = +\infty$。考虑替换，替换为

$$\mathbb P[|E_{\text{in}}(h) - E_{\text{out}}(h)| > \epsilon] \leq 2m_{\mathcal H}\exp(-2\epsilon^2N)$$

考虑根据训练集的预测输出 $\hat y$ 对 $h$ 进行分类。

| $N$ | $\hat y$ 的方案数 $m_{\mathcal H}(N)$ |
| :-: | :-: |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |
| 4 | 14 |

首先显然有方案数 $\leq 2^N$，且容易注意到若 $m_{\mathcal H}(N_0) < 2^{N_0}$，则 $m_{\mathcal H}(N_0 + 1), m_{\mathcal H}(N_0 + 2), \ldots$ 都会满足 $m_{\mathcal H}(N_0 + k) < 2^{N_0 + k}$。

### Dichotomies
记 $\mathcal{H}(\boldsymbol x_1, \boldsymbol x_2, \ldots, \boldsymbol x_N) = (h(\boldsymbol x_1), h(\boldsymbol x_2), \ldots, h(\boldsymbol x_N)) \in \{1, -1\}^N$ 为固定 $\boldsymbol x_i$ 后将所有可能的 $h$ 对 $\boldsymbol x_i$ 取值得到的结果的集合，称之为一个 Dichtomy。

再定义**成长函数**（**Growth function**）$m_{\mathcal{H}}(N)$ 为在所有可能的 $\boldsymbol x_i$ 中，对不同 $h$ 产生结果种类的最大值

$$m_{\mathcal{H}}(N) = \max_{\boldsymbol x_1, \boldsymbol x_2, \ldots, \boldsymbol x_N} |\mathcal{H}(\boldsymbol x_1, \boldsymbol x_2, \ldots, \boldsymbol x_N)|$$

当 $m_{\mathcal{H}}(N) = O(\text{poly}(N))$ 时，用其替代 $M$ 会更优。

### 断点（Break Point）
若 $m_{\mathcal{H}}(N) < 2^N$，称 $N$ 为 $\mathcal{H}$ 的一个断点。

**结论 5.1** 若最小断点为 $k$，则 $m_{\mathcal{H}}(N) = O(N^{k - 1})$。

## 6. Theory of Generalization
### 引入 Bounding Function 
记

$$B(N, k) = \max_{\mathcal H: \text{ Breakpoint} = k} m_{\mathcal H}(N)$$

这样，我们考虑 $B(N, k)$ 时就不需要关心 $\mathcal H$ 是什么，只要找到最多的数据集 $\boldsymbol x_1, \boldsymbol x_2, \ldots, \boldsymbol x_M$ 使得仅凭任意 $k$ 个特征都不能分辨所有 $\boldsymbol x_i$。

打表找规律可以发现：

* $B(N, 1) = 1$（显然）
* $\forall N < k, B(N, k) = 2^N$（根据断点定义）
* $\forall N = k, B(N, k) = 2^k - 1$（只需要去掉其中一个数据即可）
* $\forall N > k, B(N, k) = B(N - 1, k - 1) + B(N - 1, k)$

其中第四条需要我们证明。 

### 证明
考虑 $B(N, k)$ 的所有数据 $S$。

定义配对：$\forall \boldsymbol a, \boldsymbol b \in S$，若 $\boldsymbol a, \boldsymbol b$ 在前 $N - 1$ 个特征都相等，则称 $\boldsymbol a, \boldsymbol b$ 有配对。

把 $S$ 分为两个部分：一部分是存在配对的，数据个数为 $2\alpha$；剩下一部分不存在配对，数据个数为 $\beta$。则 $B(N, k) = 2\alpha + \beta$。

首先，只考虑 $S$ 的前 $(N - 1)$ 个特征，则需要把这些配对的数据对，每对仅保留一个。此时我们拥有 $\alpha + \beta$ 个合法数据，根据定义，有 $\alpha + \beta \leq B(N - 1, k)$。

接着另起一行，只考虑这 $\alpha$ 个数据。反证，若我们能在前 $(N - 1)$ 个特征中选出恰好 $k - 1$ 个特征区分这 $\alpha$ 个数据，则我们也可以通过这 $k - 1$ 个特征，加上 $x_N$ 这个特征区分 $2\alpha$ 个数据。然而这样就相当于我们选了 $k$ 个特征区分了 $2\alpha$ 个数据，不满足 $B(N, k)$ 的要求，舍去。综上，我们不能通过任意 $k - 1$ 个特征区分这 $\alpha$ 个数据，即 $\alpha \leq B(N - 1, k - 1)$。

综上，$B(N, k) = (\alpha + \beta) + \alpha  \leq B(N - 1, k) + B(N - 1, k - 1)$。等号其实是可以取到的，这里不赘述。

这个式子和组合数的推导式极为相像，故可简单地得知

$$B(N, k) = \sum_{i = 0}^{k - 1} \left(\begin{matrix}N \\ i\end{matrix}\right) = O(N^{k - 1})$$

### Vapnik-Chervonenkis Bound
实际我们获得的公式并不是上面提到的那个，而是

$$\mathbb P\left[\exist h \in \mathcal H \text{ s.t. } |E_{\text{in}}(h) - E_{\text{out}}(h)| > \epsilon\right] \leq 2 \cdot \color{red}2\color{black} m_{\mathcal H}(\color{red}2\color{black}N) \cdot \exp(-\color{red}\frac 18\color{black}\epsilon^2N)$$

该式称为 **Vapnik-Chervonenkis Bound**，简称 **VC Bound**。

具体推导分为三步。

第一步：用 $E_{\text{in}}'(h)$ 替代 $E_{\text{out}}(h)$。注意到 $E_{\text{out}}(h)$ 有无穷多种，考虑取另外 $N$ 个数据作为验证集 $\mathcal D'$，在该集上计算 $E_{\text{in}}'(h)$。若 $E_{\text{in}}(h)$ 和 $E_{\text{out}}(h)$ 相差很大，那么也有很大概率 $E_{\text{in}}(h)$ 和 $E_{\text{in}}'(h)$ 相差很大。具体地，

$$\mathbb P\left[\exist h \in \mathcal H \text{ s.t. } |E_{\text{in}}(h) - E_{\text{out}}(h)| > \epsilon\right] \leq 2\mathbb P\left[\exist h \in \mathcal H \text{ s.t. } |E_{\text{in}}(h) - E_{\text{in}}'(h)| > \frac \epsilon2\right]$$

第二步：将 $\mathcal H$ 有限化。我们可以将 $h \in \mathcal H$ 根据 $\mathcal D, \mathcal D'$ 上的 $2N$ 个输出分成 $m_{\mathcal H}(2N) \leq 2^{2N}$ 类，用类似的方法，可见

$$2\mathbb P\left[\exist h \in \mathcal H \text{ s.t. } |E_{\text{in}}(h) - E_{\text{in}}'(h)| > \frac \epsilon2\right] \leq 2m_{\mathcal H}(2N) \mathbb P\left[\text{fixed } h \text{ s.t. } |E_{\text{in}}(h) - E_{\text{in}}'(h)|\right]$$

第三步：用 Hoeffding 定理估计概率。注意到

$$|E_{\text{in}}(h) - E_{\text{in}}'(h)| > \frac \epsilon2 \Leftrightarrow |E_{\text{in}}(h) - \frac {E_{\text{in}}(h) + E_{\text{in}}'(h)}2| > \frac \epsilon4$$

考虑现在从 $2N$ 个数据里选出 $N$ 个给 $E_{\text{in}}$，剩下给 $E_{\text{in}}'$，根据 Hoeffding 定理可知

$$\mathbb P\left[\text{fixed } h \text{ s.t. } |E_{\text{in}}(h) - E_{\text{in}}'(h)| > \frac \epsilon2\right]\\
= \mathbb P\left[|(\text{sum of randomly selected } N \text{ datas}) - \text{avg}(\mathcal D + \mathcal D') | > \frac \epsilon4\right]\\ 
\leq 2\exp\left(-2\left(\frac \epsilon 4\right)^2 N\right) = 2\exp\left(-\frac \epsilon8 N\right)$$

## 7. The VC Dimension
### 复习
回顾之前的 VC Bound，它证明了当我们的学习策略集 $\mathcal H$ 良好（即 $m_{\mathcal H}$ 在 $k$ 有 Breakpoint）且当 $N$ 足够大时，我们能保证 $E_{\text{out}} \approx E_{\text{in}}$。此时，若选择的 $h \in \mathcal H$ 满足 $E_{\text{in}}$ 很小，则 $E_{\text{out}}$ 也很小，也就是说，这个算法确实对没有见过的数据都适用。

**需要特别注意的是，该理论和输入数据 $\mathcal D$ 分布、目标实际函数 $f$ 等都没有关系，也并不能保证学习后 $E_{\text{in}}$ 很小。该理论只保证了学习的有效性。**

### VC Dimension
**VC Dimension** 指的是最大非断点，即

$$d_{\text{VC}}(\mathcal H) = \max_{m_{\mathcal H}(N) = 2^N} N$$

当 $N, d_{\text{VC}} \geq 2$，有 $m_{\mathcal H}(N) \leq N^{d_{\text {VC}}}$。

### 回顾 PLA
容易注意到，$1$ 维 PLA 的 $d_{\text{VC}} = 2$，而 $2$ 维 PLA 的 $d_{\text{VC}} = 3$。

**猜想 7.1** $k$ 维 PLA 的 $d_{\text{VC}} = k + 1$。

**证明 7.1** 接下来分为两部证明：$d_{\text{VC}} \geq k + 1$ 和 $d_{\text{VC}} \leq k + 1$。

先证 $d_{\text{VC}} \geq k + 1$。这意味着存在 $k + 1$ 个数据，使得该算法产生的输出能覆盖到所有 $2^{k + 1}$ 种不同的输出。

考虑构造数据 $\text X \in \{0, 1\}^{(k + 1) \times (k + 1)}$：

$$\text X = \begin{bmatrix}
  \boldsymbol x_1 \\ \boldsymbol x_2 \\ \vdots \\ \boldsymbol x_{k + 1}
\end{bmatrix}

= \begin{bmatrix}
  1 & 0 & 0 & \ldots & 0 \\
  1 & 1 & 0 & \ldots & 0 \\
  \vdots & \vdots &   & \ddots & 0 \\
  1 & 0 & 0 & \ldots & 1 \\
\end{bmatrix}
$$

要证明上述声明，相当于对于任意 $\boldsymbol y \in \{-1, 1\}^{k + 1}$，都能找到一个 $\boldsymbol w \in \R^{k + 1}$，使得

$$\text{sign}(\text X \boldsymbol w) = \boldsymbol y$$

即

$$\text{sign}(w_1) = y_1$$

$$\text{sign}(w_1 + w_{i + 1}) = y_i$$

这样的数据不难构造，令 $w_1 = y_1$，$w_{i + 1} = y_i - y_1$ 即可。

接下来证明 $d_{\text {VC}} \leq k + 1$。这意味着对于任何 $k + 2$ 个数据，该算法产生的输出都不能覆盖到所有 $2^{k + 2}$ 种不同的输出。

注意到数据是 $k + 2$ 个 $k + 1$ 维向量，所以该向量组必定线性相关。不妨设 $\boldsymbol x_{k + 2} = \sum_{i = 1}^{k + 1} a_i \boldsymbol x_i$。

取输出 $\boldsymbol y$ 使得 $\forall 1 \leq i \leq k + 1, y_i = \text{sign}(a_i)$，则有

$$
\text{sign}(\boldsymbol w \cdot \boldsymbol x_i) = y_i = \text{sign}(a_i) \\
\Leftrightarrow \text{sign}(a_i\boldsymbol w \cdot \boldsymbol x_i) = 1 \\
 \Leftrightarrow a_i\boldsymbol w \cdot \boldsymbol x_i > 0
$$

则

$$
\begin{aligned}
  y_{k + 2} &= \text{sign}(\boldsymbol w \cdot \boldsymbol x_{k + 2}) \\
  &= \text{sign}\left(\sum_{i = 1}^{k + 1} a_i \boldsymbol w \cdot \boldsymbol x_i\right) \\
  &> \text{sign}\left(\sum_{i = 1}^{k + 1} 0 \right) \\
  &= 1 \\
\end{aligned}
$$

可见 $y_{k + 2}$ 在此情形下不可能为 $-1$，即我们无法让输出 $\boldsymbol y = \{\text{sign}(a_1), \text{sign}(a_2), \ldots, \text{sign}(a_{k + 1}), -1\}$。证毕。

### 经验法则
$$d_{\text{VC}} \approx \#\text{free parameters}$$

当 $d_{\text{VC}}$ 过小，可能导致自由度太低，没法让 $E_{\text{in}}$ 尽可能小；当 $d_{\text{VC}}$ 过大，可能导致 VC Bound 太大，难以保证训练效果。

