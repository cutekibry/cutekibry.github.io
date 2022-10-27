---
title: Coursera 吴恩达机器学习入门课笔记（更新至第三课）
date: 2022-08-13
category: 
- 学习
---

吴恩达在 Coursera 上的机器学习入门课的笔记。只给自己看，所以写得非常简略。

使用的是较新的 2022 版本（和之前的不太一样），双语可见 [BV1Pa411X76s](//www.bilibili.com/video/BV1Pa411X76s)。

<!-- more -->

## 第一课
### 线性回归（Linear Regression）
设特征数为 $n$，样本数为 $m$。

模型：$f_{\vec w, b}(\vec x) = \vec w^{T} \vec x + b = \left(\sum_{i = 1}^n w_ix_i\right) + b$

损失函数：

$$L(f_{\vec w, b}(\vec x^{(i)}), y^{(i)}) = (f_{\vec w, b}(\vec x^{(i)}) - y^{(i)})^2$$

费用函数：

$$J(\vec w, b) = \frac 1{2m} \left(\sum_{i = 1}^m \left(f_{\vec w, b}(\vec x^{(i)}) - y^{(i)} \right)^2 \right) + \frac \lambda{2m} \sum_{i = 1}^n w_i^2$$

此处 $\frac \lambda{2m} \sum_{i = 1}^n w_i^2$ 采用了**正则化**（Regularization），能使得 $w_i$ 的值减小，从而抑制**过拟合**（Overfitting）。

### 梯度下降（Gradient Descent）
$$
\begin{aligned}
    \frac {\partial J(\vec w, b)}{\partial w_j} &= \frac 1m \left(\sum_{i = 1}^m \left( f_{\vec w, b}(\vec x^{(i)}) - y^{(i)} \right) x_j^{(i)} \right) + \frac 1m w_j \\ 
\end{aligned}
$$

$$
\begin{aligned}
    \frac {\partial J(\vec w, b)}{\partial b} &= \frac 1m\sum_{i = 1}^m \left( f_{\vec w, b}(\vec x^{(i)}) - y^{(i)} \right) \\ 
\end{aligned}
$$

### 特征缩放（Feature Scaling）
* Mean Normalization：$x' = \frac {x - \mu}{\max - \min}$
* Z-Score Normalization：$x' = \frac {x - \mu}\sigma$

目标是让 $x$ 尽可能趋近于 $[-1, 1]$ 分布。

### 逻辑回归（Logistic Regression）
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

## 第二课
### 输出层的转换函数选择
* 二元分类：$\text{Sigmoid}$
* 回归：$\text{id}$（带正负）或 $\text{ReLU}$（非负）

### Softmax 回归
Softmax 回归用来解决多分类问题。

假设类别共有 $k$ 种，则 Softmax 回归的输出为一个 $k$ 维向量 $\vec x$，$x_i$ 衡量分类为 $i$ 的相对可能性有多大。实际分类为 $i$ 的概率估计为

$$a_i = P(y = i) = \frac {e^{x_i}}{\sum_{j = 1}^k e^{x_j}}$$

损失函数为

$$
loss(\{a_k\}, y) = -\log a_y
$$

### 交叉验证
咕了

### 决策树解决二分类问题
给定特征向量 $\vec x = \{a_1, a_2, \ldots, a_k\} \ (a_i \in \{0, 1\})$，估计其分类 $\hat y \in \{0, 1\}$。

熵：若 $p$ 为当前节点数据中第一类的占比，那么定义熵 $H(p)$ 为

$$H(p) = -p \log_2 p - (1 - p) \log_2 (1 - p), \quad p \in (0, 1)$$

特别地，$H(0) = H(1) = 0$（这是因为 $\log_2 0$ 没有定义）。

{% note info %}

**INFO**

可以注意到 $\lim_{x \rightarrow 0} x \log_2 x = 0$，因此即使特殊定义 $H(0) = 0\log_2 0 = 0$ 也不会破坏函数连续性。

{% endnote %}

![H(x) 的图像](https://s2.loli.net/2022/10/12/GTM4DpSwrt9JHCx.png)

信息增益：若将根节点 $root$ 分为左子树 $left$ 和右子树 $right$，记节点数据个数为 $n$，则信息增益（Information Gain，实际上就是熵减）为

$$H(p^{root}) - \frac {n^{left}}{n^{root}} H(p^{left}) - \frac {n^{right}}{n^{root}} H(p^{right})$$

取信息增益最大分割即可。

### 停止条件
停止划分的条件有几种：

* 当当前节点深度已经达到规定的最大深度时；
* 当最大信息增益小于某个阈值时；
* 当节点数据数小于某个阈值时。

### One-hot 编码
对于 $a \in [1, n] \cup \N$，可以将 $a$ 转化为一个长度为 $n$ 的向量 $\vec b = \{[a = 1], [a = 2], [a = 3], \ldots, [a = n]\}$。可以注意到，转化后的 $\vec b$ 的每个元素都为 $0$ 或 $1$，并且有且仅有一个元素为 $1$（因此成为 One-hot 编码）。

对多分类问题的决策树应用 One-hot 编码，即可转化为二分类问题。

### 连续取值
在决策树上，对于取值为实数的特征进行分割时，可以取条件为 $x \leq C$。

若当前有 $m$ 个数据，其从小到大为 $x_1 \leq x_2 \leq \ldots \leq x_m$，则可以考虑取 $C = \frac {x_i + x_{i + 1}}2 \quad (1 \leq i \leq n - 1)$。

### 扩展到回归树
回归树和决策树类似，但其输出为一个 $y \in \R$，而不是预测其分类。

类似地，可以定义回归树的信息增益为

$$D(p^{root}) - \frac {n^{left}}{n^{root}} D(p^{left}) - \frac {n^{right}}{n^{root}} D(p^{right})$$

其中 $D$ 为方差函数。

### 决策树森林
单棵决策树对数据的微小改变较为敏感。为了使其更加健壮，可以生成多棵决策树，然后收集所有决策树计算后的结果，取众数作为结果返回。

生成决策树的方式是：每次有放回抽样，抽出与数据集大小个数相等的样本（这些样本可重复），根据它们建决策树。

决策树棵数一般在 $100$ 左右，因为每增加一棵决策树，其对准确率的增加是有边际递减的。一般来讲，当超过 $100$ 之后其对性能的提升就很小了，还会减慢计算速度。

### 随机森林算法
仅有放回抽样构造出的决策树，在根节点附近的分类标准仍然是高度相似的。记特征数为 $n$，为了更加随机，可以在每次分割时仅随机选出 $k < n$ 个特征进行判断，取信息增益最多的那个进行分割。

一般来讲取 $k \approx \sqrt n$（西瓜书上据说 $k \approx \log_2 n$）。

### Boost
更进一步地，在每次抽样时，我们可以更有意识（更高概率地）抽出那些被当前已有决策树森林错误分类的样本。

### XGBoost
XGBoost 是一个开源高效的决策树 / 回归树计算库。

### 如何选择：决策树还是神经网络
决策树的优点：

* 适合处理天然结构化的数据。简单来说，如果数据看起来像一个表格，则适合用决策树处理。
* 训练时间短。
* 较小的决策树可能是 Human-interpretable（人类可以理解的）。

决策树的缺点：

* 不适合处理非结构化数据（图像，文本，视频，音乐等）。

神经网络的优点：

* 适合处理几乎所有类型的数据（无论是否结构化）。
* 可以和迁移学习（Tranfer learning）一起使用。
* (*) 可以更容易做到让多个机器学习模型一起配合训练。
  - 原因很复杂，暂时不展开。

神经网络的缺点：

* 训练时间长。

## 第三课
### 引入
* 无监督学习算法
  * 聚类（Clustering）
  * 异常检测（Anormaly detection）
* 推荐系统
* 强化训练

### K-means 聚类算法
K-means 算法可以用来解决聚类问题。算法流程如下：

1. 随机 $k$ 个点的坐标 $\vec {\mu_1}, \vec {\mu_2}, \ldots, \vec {\mu_k}$作为 $k$ 个簇的中心点。
2. 重复以下流程直至中心点坐标无变化：
   1. 遍历所有数据点，将每个数据点分配给离它最近的那个中心点。
   2. 将中心点坐标改为它所支配的数据点坐标的平均值。

其中若中心点无支配数据点，则要么删去该中心点，要么重新随机中心点坐标。

上面的做法实际上是在优化代价函数

$$J(\{c^{(m)}\}, \{\vec {\mu_k}\}) = \frac 1m \sum_{i = 1}^m \| \vec {x^{(i)}} - \vec {\mu_{c^{(i)}}}\|^2$$

该代价函数也称为 Distortion 函数。

可以注意到，算法流程中的 2.1. 既是在保持 $\vec {\mu_k}$ 不变的前提下解得 $\{c^{(m)}\}$ 的最优解，2.2 既是在保持 $\{c^{(m)}\}$ 不变的前提下解得 $\vec {\mu_k}$ 的最优解。也就是说，每一步调整后，代价函数的值都应减小或不变。

### K-means 聚类算法的改进
初始化时，我们可以令 $k$ 个中心点坐标为 $k$ 个不同的样本数据，以取得更好的效果。

另外，该代价函数也有多个局部最小值，所以我们也可以运行多次 K-means 算法（一般为 $50 \ldots 1000$ 次），取代价函数最小的作为答案。

### 如何选择 K 值
肘部法则（Elbow Method）：令 $f(k)$ 为取 $K = k$ 时计算得到的最小代价，则取 $f(k) - k$ 图像上明显的拐点的横坐标作为 $k$ 的取值。但实际上不少时候这个图像是比较平缓的，所以其实也没有很大用处，吴恩达也说自己不用。

实际应用中，该算法一般是用来产生一些簇，提供给下游（Downstream）的开发人员，让他们进一步进行计算的。所以可以考虑根据下游开发人员的反馈调整 $K$ 值。

{% note warning %}

**WARNING**

注意：你并不是在选择 $k$ 使得 $f(k)$ 最小。实际上，大多数情况下 $f(k)$ 随着 $k$ 的增大是近似单调下降的，但这并不意味着簇数越多越好。

{% endnote %}

