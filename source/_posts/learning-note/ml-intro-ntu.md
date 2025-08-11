---
title: 台大李宏毅《机器学习》笔记
date: 2023-04-17
category: 
- 学习
---

[BV1wU4y1v7Eh](//www.bilibili.com/video/BV1wU4y1v7Eh)。

看了很久才发现看错老师了，应该是看李老师而不是林老师的课……

<!-- more -->

## 神经网络
### 类神经网络训练不起来怎么办
#### 局部最小值（Local minima）和鞍点（Saddle point）
将 $\frac {\partial L}{\partial x_i} = 0$ 的点称为**驻点**（Critical point）。

这样的点有可能是**极点**（Local minima/maxima，局部最小或局部最大），也有可能是**鞍点**（Saddle point，即沿着一些方向走会变大，沿着一些方向走会变小，但偏导数都等于 $0$）。

使用 Hessian 在驻点 $\boldsymbol x_0$ 进行展开，有

$$L(\boldsymbol x) \approx L(\boldsymbol x_0) + (\boldsymbol x - \boldsymbol x_0)^T \boldsymbol g + \frac 12 (\boldsymbol x - \boldsymbol x_0)^T\boldsymbol H(\boldsymbol x - \boldsymbol x_0)$$

其中 $\boldsymbol g = \text{grad } L \in \R^k$，满足 $g_i = \frac {\partial L}{\partial x_i}$；

$\boldsymbol H \in \R^{k \times k}$，满足 $H_{i, j} = \frac {\partial^2 L}{\partial x_i \partial x_j}$。

因为 $\boldsymbol x_0$ 为驻点，所以 $\boldsymbol g = \boldsymbol 0$，上式可写作

$$L(\boldsymbol x) \approx L(\boldsymbol x_0) + \frac 12 (\boldsymbol x - \boldsymbol x_0)^T\boldsymbol H(\boldsymbol x - \boldsymbol x_0)$$

下面我们考虑 $\boldsymbol H$ 的一个特征向量 $\boldsymbol u$ 和对应的特征值 $\lambda < 0$。可见

$$\boldsymbol u^T\boldsymbol H \boldsymbol u = \boldsymbol u^T \lambda \boldsymbol u = \lambda \lVert \boldsymbol u \rVert^2 < 0$$

令 $\boldsymbol x = \boldsymbol x_0 + k\boldsymbol u$，则有 $L(\boldsymbol x) < L(\boldsymbol x_0)$。即，沿着 $\boldsymbol u$ 方向即可降低 $L(\boldsymbol x)$。

注：由于该方法需要计算二次微分和特征值、特征向量，计算量非常大，所以实践中几乎不使用该方法。

---

在实践中，究竟是 Local minima 多还是 Saddle point 多？

理论上讲，一个点可能在二维平面是 Local minima，但在三维空间里可能就是 Saddle point 了。因此，在高维空间下，Saddle point 的数量也应该比 Local minima 多一些。

例如下方的这个图，竖轴代表模型最终收敛到的 Loss 的值，横轴代表正特征值个数占所有特征值个数的比例。可以发现实践中这个比例是很小的，且大部分模型都能收敛到一个不错的 Loss 值。

![示意图](/img/post/ml-intro-ntu/19-1.jpg)

#### Batch size 和动量（Momentum）
|                                  |    小    |    大    |
| :------------------------------- | :------: | :------: |
| 更新一次参数的速度               |    快    |    慢    |
| 更新一次参数的速度（带平行计算） | 基本相等 | 基本相等 |
| 进行一次迭代（Epoch）的速度      |    慢    |    快    |
| 梯度                             |  噪声大  |   稳定   |
| 最优化能力                       |    强    |    弱    |
| 泛化能力                         |    强    |    弱    |

Batch size 小或大都有各自的优点和缺点。有一些方法可以鱼和熊掌兼得之，可以参考下面的 paper：

* Large Batch Optimization for Deep Learning: Training BERT in 76 minutes(https://arxiv.org/aos/1904.00962)
* Extremely Large Minibatch SGD: Training ResNet-50 on ImageNet in 15 Minutes (https://arxiv.org/abs/1711.04325)
* Stochastic Weight Averaging in Parallel:Large-Batch Training That Generalizes Well (https://arxiv.org/abs/2001.02312)
* Large Batch Training of Convolutional Networks (https://arxiv.org/abs/1708.03888)
* Accurate, large minibatch sgd: Training imagenet in 1 hour (https://arxiv.org/abs/1706.02677)

上述 paper 里采用的 batch size 一般很大（甚至上千上万）。

---

在朴素的梯度下降法中，当梯度几乎为 $0$ 时算法将会停止。然而在现实世界中，即使一个球从高处滚落到平原，它也会因为惯性而继续往前滚动而不直接停下。

在带**动量**（Momentum）的梯度下降中，每次更新的值 $\Delta \boldsymbol x$ 不再是纯粹由梯度 $\boldsymbol g$ 决定，而是

$$\Delta \boldsymbol x = \lambda \Delta \boldsymbol x^{(\text{prev})} - \eta \boldsymbol g$$

其中 $\Delta x^{(\text{prev})}$ 是上次更新的位移差。此时我们要走的方向不是纯粹沿着梯度 $\boldsymbol g$ 了，还会受上次移动的惯性（严谨地，还有之前的所有移动）的影响。

总结：

* 驻点是梯度全为 $0$ 的点。
* 驻点可能是局部最小点（Local minima）或鞍点（Saddle point）。
  * 可以用 Hessian 矩阵的特征值正负性判定，且能逃离 Saddle point。
  * 局部最小点大概率很少。
* 小的 Batch size 和动量（Momentum）算法能帮助算法逃离驻点。

#### 最大难题：学习率调整
即使采用了合适的参数进行梯度下降，也可能遇到 Loss 难以下降的情况。

一种非常棘手的情况是，假设 $L = L(x, y)$，其中 $|\frac {\partial L}{\partial x}|$ 相当大但 $|\frac {\partial L}{\partial y}|$ 相当小。当学习率 $\eta$ 过大时，会导致 $x$ 参数来回震荡，无法抵达理想值 $x_f$；当学习率 $\eta$ 过小时，会导致 $y$ 参数几乎不改变，也同样无法达到理想值 $y_f$，如下图所示。

![一个仅含参数 w, b 的模型](/img/post/ml-intro-ntu/21-1.jpg)

因此，我们需要对不同参数使用不同的学习率。令

$$\Delta w_i^{(t + 1)} = -\frac \eta{\sigma_i^{(t)}} g_i^{(t)}$$

此处 $\sigma_i^{(t)}$ 是一个和 $i, t$ 都有关的量。

##### Adagrad 算法
一个可能的算法是 Root Mean Square（RMS，均方根）算法，其中

$$\sigma_i^{(T)} = \sqrt {\frac 1{T + 1} \cdot \sum_{t = 0}^T (g^{(t)}_i)^2}$$

代表过去所有时刻的梯度的均方根。

Adagrad 采用了 RMS 算法。

##### RMSProp 算法
纯粹的 Adagrad 有一个明显的缺点，那就是早期的梯度值对后期也会有很大的影响。因为损失函数可能是相当复杂的，导致在某些区域上梯度大，某些区域上梯度小。

RMSProp 算法采用了**指数平滑**法，即

$$
\begin{aligned}
  \sigma_i^{(T)} &= \sqrt {\sum_{t = 0}^T \alpha^t (1 - \alpha)(g^{(T - t)}_i)^2} \\
  &= \sqrt {\alpha (\sigma_i^{(T - 1)})^2 + (1 - \alpha)(g^{(t)}_i)^2} \\
\end{aligned}
$$

其中 $\alpha \in (0, 1)$ 是一个常数。不难注意到，越久远的梯度产生的影响是越小的（因为要乘上 $\alpha^t$），所以实际也会更优秀一点。

##### Adam 算法
目前最常用的算法就是 Adam 算法，它是 RMSProp 和 Momentum 的结合。Paper 原文可在 https://arxiv.org/aos/1904.00962 找到。虽然该算法也有一些超参数要调整，但一般情况下使用默认超参数即可。

##### 学习率调整策略（Learning rate scheduling）
除此之外，还可以考虑在 $\eta$ 上进行调整，即 $\eta^{(t)}$ 会随更新次数 $t$ 的增加而改变。

一个很朴素的想法是，让 $\eta$ 平滑地减少。这称之为 **Learning rate decay**。

还有一个有些匪夷所思的想法：让 $\eta$ 快速从 $0$ 变大，然后再缓慢变小。这称之为 **Warm up**。在很多远古论文中出现，称之为一种黑科技。具体为什么 Warm up 有效目前也不清楚，一种感性的解释是，先让模型在附近探索一下，能提高 $\sigma^{(t)}_i$ 的统计意义。对 Warm up 的更进一步探究可以考虑看 RAdam（https://arvix.org/abs/1908.03265 ）。

突然提到的 Arxiv 小知识：Arxiv 的编号格式为 `YYMM.NNNNN`，其中 `YY` 代表年份，`MM` 代表月份，`NNNNN` 表示这是该年度已经提交的论文数量。

### Hoeffding 不等式
$$P(\mathcal D_{\text{train}} \text{ is bad due to } h) \leq 2\exp(-2N\epsilon^2)$$

其中要求 $L(\boldsymbol x) \in [0, 1]$，$N$ 是训练集的样例数量。因此有

$$P(\mathcal D_{\text{train}} \text{ is bad} ) \leq |\mathcal H| \cdot 2\exp(-2N\epsilon^2)$$

当然，这样的上界未免有些太宽松了，以至于很多时候算出来都是大于 $1$ 的。要更进一步探究上界（VC-dimension），可以参考《机器学习基石》。

但至少它为我们反映了一点：**理论上，要提高普适性，则应当降低模型复杂度 $|\mathcal H|$ 或增加数据量 $N$**。

当然，我们也不能盲目降低 $\mathcal H$ 的复杂性，因为虽然这样导致普适性变高了，却会导致 Loss 太大，无法充分拟合任意数据。理论上来讲，$\mathcal H$ 大小都有各自的优缺点，那能不能做到两全其美呢？实际上是可以的：通过**深度学习**的方式，可以做到两全其美。

### 附加内容：学习率学习 / 调整方法
#### Adam 算法原理
首先，Adam 采用动量算法计算当前动量 $m^{(t)}$，其中

$$m^{(t)} = \beta_1 m^{(t - 1)} + (1 - \beta_1) g^{(t - 1)}$$

再用 RMSProp 修正学习率，修正因子为 $v^{(t)}$，其中

$$v^{(t)} = \beta_2v^{(t - 1)} + (1 - \beta_2)(g^{(t - 1)})^2$$

则

$$\Delta w_i = -\frac {\eta}{\sqrt {\hat v^{(t)}} + \epsilon}\hat m^{(t)}$$

其中 

$$\hat m^{(t)} = \frac {m^{(t)}}{1 - \beta_1^t}$$
$$\hat v^{(t)} = \frac {v^{(t)}}{1 - \beta_2^t}$$

这个方法被称为 **de-biasing**，目的是为了在训练开始 $m, v$ 过小、尚未稳定的时候保证学习速度。

默认超参数一般取 $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$。

#### Adam vs SGDM
Adam：训练快，泛化鸿沟（generalization gap）大（即泛化性弱），不稳定

SGDM：泛化性强，稳定，容易收敛

二者结合为 **SWATS** [Keskar, et al., arXiv'17]，开始用 Adam，收尾用 SGDM。但 SWATS 仍有需要研究的地方，如：

* 什么时候切换？
* Adam 会自适应学习率，但 SGDM 不会，要如何处理？

#### 改进 Adam
研究者发现在训练末期，大部分数据上梯度都接近于 $0$，仅有少量的数据梯度相当大，导致平均梯度很小。并且，当 $\beta_2 = 0.999$ 时，

$$v^{(t)} = 0.999v^{(t - 1)} + 0.001(g^{(t - 1)})^2$$

其中 $0.999$ 很大，导致很久以前的梯度仍然会对当前 $v^{(t)}$ 产生影响（具体来说，$0.999^{999} \approx \frac 1e \approx 0.37$），所以 $v^{(t)}$ 也会相当大。这导致在训练末期模型对梯度较大的模型不敏感。结论是，单个数据的 movement（更改量）上界为 $\sqrt {\frac 1{1 - \beta_2}}\eta = 10\sqrt {10}\eta \approx 32\eta$。

于是提出了 **AMSGrad** [Reddi, et al., ICLR'18]。该算法中 $v^{(t)}$ 会取历史梯度最大值。然而，虽然这保证了梯度小的数据学习率低、而梯度大的数据会影响参数，但因为取历史梯度最大值，所以和 SGD 有一样的缺点：前期很可能走了几步就不走了。

之后就有了 **AdaBound** [Luo, et al., ICLR'19]，公式为

$$\Delta w^{(t)} = -\text{Clip}\left(\frac \eta{\sqrt {\hat v^{(t)}} + \epsilon}\right) \hat m^{(t)}$$

其中 

$$\text{Clip}(x) = \text{Clip}(x, 0.1 - \frac {0.1}{(1 - \beta_2)t + 1}, 0. 1 + \frac {0.1}{(1 - \beta_2)t})$$

这里给出的上界和下界是作者给出的经验公式，并不清楚为什么这样取。

#### 改进 SGD
SGD 在学习率过小或过大时表现都不优。我们可以枚举学习率进行训练，推测学习率等于多少时最优。这样的测试叫 **LR range test** [Smith, WACV'17]。

一个改进方法是，我们让学习率随着时间从我们钦定的下界 $\eta_L$ 开始线性增大到上界 $\eta_R$，然后再线性减回来，不断重复，产生一个三角波的形状。这种方法叫做 **Cyclical LR** [Smith, WACV'17]。

另一个版本是，令学习率的减少为 $\cos$ 函数，从 $\eta_R$ 减小到 $\eta_L$，然后立即增大到 $\eta_R$，不断重复，产生 $\cos$ 从 $[0, \pi]$ 截得的图像。该方法称为 **SGDR** [Loshchilov, et al., ICLR'17]。

还有一个版本，先从 $0$ 线性增大到 $\eta_R$（称为 warm up），然后线性减小到较小的下界 $\eta_L$（称为 annealing），最后从 $\eta_L$ 线性慢慢减小到 $0$（称为 fine-tuning）。fine-tuning 的学习率减小速度应该比前两者都低。这个过程只执行一次，执行结束即训练完毕，不像前两个方法会周期不断重复。该方法称为 **One-cycle LR** [Smith, et al., arXiv'17]。

#### Adam 和 warm up
在 adam 的前十次更新中，每次更新都会使数据梯度发生明显的变化，$v^{(t)}$ 也会很不稳定，直到后面才会稳定下来。此时如果预先进行 warm up，可以使数据梯度更加稳定。

#### RAdam
有人提出了 **RAdam** [Liu, et al., ICLR'20]，但是太复杂了，听不懂 TT

#### Lookahead
**Lookahead** [Zhang, et al., arXiv'19] 是一个类似外包装（wrapper）性质的算法。该算法以一个初始参数 $\phi$ 为起点，用某种指定的 optimizer 更新 $k$ 次（$k$ 是常数）得到 $\theta^{(k)}$，最后再更新 $\phi ' = \phi + \alpha(\theta^{(k)} - \phi)$。可以认为是从 $\phi$ 开始往外随便探索 $k$ 步，最后才往那个方向挪动一点距离。

Lookahead 的好处是能避免太极端的探索，使得训练调参更加稳定，且能寻找到更平滑的局部最小值，因此而有更高的泛用性。

#### Nesterov accelerated gradient（NAG）
**NAG** [Nesterov, joul Dokl. Akad. Nauk SSSR'83] 对 SGDM 做出了改进，能够让 SGDM 稍微预测未来的影响。

原 SGDM：

$$m^{(t)} = \lambda m^{(t - 1)} + \eta L(w^{(t - 1)})$$

改进后的 NAG：

$$m^{(t)} = \lambda m^{(t - 1)} + \eta L(w^{(t - 1)} - \lambda m^{(t - 1)})$$

其中 $w^{(t - 1)} - \lambda m^{(t - 1)}$ 就是在预测下一步的 $w^{(t)}$ 应该是多少。

这里改进后的 NAG，在实现时我们只记录两个变量 $w'^{(t)}$ 和 $m^{(t)}$，其中 $w'^{(t)} = w^{(t)} - \lambda m^{(t)}$。经过推导有

$$w'^{(t)} = w'^{(t - 1)} - \lambda m^{(t - 1)} - \eta\nabla L(w'^{(t - 1)})$$

$$m^{(t)} = \lambda m^{(t - 1)} + \eta\nabla L(w'^{(t - 1)})$$
 
上式与 $w^{(t)}$ 无关，仅和 $w'^{(t)}, m^{(t)}$ 有关。

#### Nadam
**Nadam** [Dozat, ICLR workshop'16] 相当于对 adam 使用了 NAG 的超前部署的方法。

#### 其他方法
**Gradient noise** [Neelakantan, et al., arXiv'15] 算法会在每个梯度中加入一个随机噪声 $N(0, \sigma_t^2)$，其中 $\sigma_t = \frac c{(1 + t)^\gamma}$。

**Curriculum learning** [Bengio, et al., ICML'09] 先用简单数据训练模型，再用困难数据训练模型。

#### 总结
SGDM 一般用于

* CV（图像识别，图像分割，对象检测）

Adam 一般用于

* NLP（QA，机器翻译，总结）
* 语音识别
* GAN（生成对抗网络）
* RL（强化学习）

**注意：没有万能的 Optimizer**。

## CNN
### 入门
每个神经元照顾的感知区域大小称为 **kernal size**（如 $3 \times 3$），一般会涉及到所有通道，每次计算后会向右 / 向下移动固定步数，该步数称为 **stride**（如 $2$）。

注意到同样的模式可能在图片的任何一个地方出现，所以可以考虑共用参数，即每个神经元只会有 $(\text{kernal size} + 1)$ 个参数。

除了卷积层外，还有池化层（Pooling layer），因为缩小图片不会改变图片内的内容。如一个 $2 \times 2$ 大小的 Max pooling 层，会把大小为 $2n \times 2n$ 的图片缩小到 $n \times n$ 的图片。

一个经典的 CNN 架构为：卷积层 -> 池化层 -> 卷积层 -> (卷积层) -> (池化层) -> ... -> 摊平 -> 全连接层 -> 全连接层 -> ... -> Softmax。

CNN 是为图形专门设计的，但也可以用到类似的情境下，例如 Alpha Go 使用的便是 CNN。这是因为围棋也可以视为一个 $19 \times 19$ 的图片，且 patterns 通常比较小，出现在哪个位置都影响不是很大。唯一要注意的是 Alpha Go 没有池化层，因为实际下围棋时我们不可能忽略奇数行或奇数列，否则会极大影响战略。CNN 也可以用于语音处理和 NLP 等，但需要对一些特性进行修改，此处不展开。**要想清楚数据的特性，并为之设计方案**。

此外，原始 CNN 不能处理图片局部放大、缩小或旋转的问题。关于类似的问题，可以参考 Spatial Transformer Layer。

## 深度学习
### 为什么使用验证集仍然 Overfitting
当我们用验证集从 $k$ 个模型 $\mathcal H' = \{h_1, h_2, \ldots, h_k\}$ 里选取 Loss 最小的那个模型时，其实就是把验证集当作训练数据，筛选出在验证集上表现最好的那个模型。因此，实际上

$$P(\mathcal D_{\text{val}} \text{ is bad}) \leq 2|\mathcal H'| \exp(-2\epsilon^2N)$$

理论上来讲，当 $|\mathcal H'|$ 越大，右边的上界就越大，该验证集为坏数据的可能性就有概率更大（虽然不一定）。也就是说，验证集更有可能“偏爱”某个并不准确的模型，导致那个模型在验证集上 overfit，从而你选出的模型也是 overfit 的。

### DL 拟合折线
对于任意一个有 $n$ 个折点的折线，我们可以用不超过 $n + 2$ 条简单折线 $y_i$ 之和来拟合，其中

$$y_i =
\begin{cases}
  0, & x \leq L_i\\
  \frac {x - L_i}{R_i - L_i}U_i, & L_i \leq x \leq R_i \\
  U_i, & x \geq R_i \\
\end{cases}
$$

这个函数被称为 **hard sigmoid** 函数。在实际中，我们多用 **sigmoid** 或 **ReLU** 函数来近似 hard sigmoid，其中 sigmoid 满足

$$\sigma(x) = \frac 1{1 + e^{-x}}$$

ReLU 满足

$$\text{ReLU}(x) = \max\{0, x\}$$

考虑一个仅有一层 hidden layer 的模型，让 hidden layer 的每个神经元掌管一条折线，则只要神经元够多，我们就可以近似拟合出有任意个折点的折线，进而近似拟合出任意函数。

实验数据表明，**在一定限度内，对参数数量相等的模型来说，层数多的模型表现比层数少的优秀**。换句话说，深度模型能够更有效率（即参数更少）地描述拟合函数。这一结论可以感性理解：比如编程工程中，我们都会分成若干子模块，子模块下面还有别的一些模块，形成一个图一样的依赖关系，就像 DL 的多层学习一样。

再比如，如果我们要拟合函数 $f(x) = |x \bmod 2 - 1|$，其中 $x \in [0, 128]$，使用一层 hidden layer 就需要 $128$ 个神经元（直接分段），但用多层 hidden layer 的话只需要 $7$ 层，其中每层只需要 $2$ 个神经元（因为 $f(x)$ 是周期性函数，可以用函数嵌套和倍增来逼近）。

因此，**对复杂且有规律的问题（如语音、图像等），DL 比朴素实现表现更优**。

### 选修：Spatial Transformer Layer
实现图像的平移、放大、缩小或旋转，最暴力的想法就是套一个全连接层。但这样的开销太大，参数过多，不是很优秀。

注意到平移、放大、缩小和旋转可以视为对坐标的线性变换，即

$$\begin{bmatrix}x'\\y'\end{bmatrix} = \begin{bmatrix}a&b\\c&d\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix} + \begin{bmatrix}e\\f\end{bmatrix}$$

其中 $x, y$ 是原坐标，$x', y'$ 是变换后的坐标，$a, b, c, d, e, f$ 是 STL 根据上一层的所有输入而给出的输出，即六个与 $x, y$ 无关的常数。

需要注意的是，$x', y'$ 极可能不是整数。粗暴地四舍五入会导致梯度为 $0$（稍微改变 $a, b, c, d, e, f$ 都不会引起改变），所以此时我们会修改四个格子的值，其中

$$\Delta A_{i, j} = |x' - i| \cdot |y' - j| \cdot A_{x, y} \quad (\lfloor x' \rfloor \leq i \leq \lceil x' \rceil \land \lfloor y' \rfloor \leq j \leq \lceil y' \rceil)$$

这一结构称为 Transformer layer（变换层），可以置于图像预处理或卷积层后。也可以在同一位置使用两个 Transformer layer，变化出两个不同的图像（如鸟类检测，变化出一个鸟头和鸟身）。

## 自注意力机制（Self-attention）
### 引入
有些问题给出的输入不是一个向量，而是若干个向量。例如 NLP，如果将一句话的每个单词视为一个向量，就可以称为 **Sophisticated input**（复杂输入）。或者再比如，语音识别如果把每 20 ms 视为一个向量，或图论问题把每个点当成一个向量，都是 Sophisticated input。

在复杂 NLP 场景下，对每一个词都进行 One-hot 编码是不现实的，也无法揭示单词的相关程度。另一个方法叫 Word embedding，会根据单词语义程度给予一个向量作为输出，类似的单词距离应该会较小。本节课重点不在这里，就不展开了。

对于这样的输入，算法产生的输出有如下 3 种类型：

* 对每个向量都会输出一个标量（例如 NLP 中的词性分析）
* 对整个向量集输出一个标量（例如 NLP 中的情绪识别）
* 输出若干个向量，输出个数由算法决定（例如 NLP 中的翻译）

接下来先着重介绍第一种。对于第一种问题，我们可以用“通用的”全连接层和 Self-attention 层来解决，如下：

![一个模型实例](/img/post/ml-intro-ntu/38-1.jpg)

其中：

* 同一层经过的 FC（Fully-connected，全连接层）都是相同的
* Self-attention 层会将 $k$（$k$ 为任意正整数）个向量作为输入，输出为 $k$ 个处理后向量

有关 Self-attention 最知名的 paper 是 *Attention Is All You Need*（[Link](https://arxiv.org/abs/1706.03762)），该 paper 首次提出了 Transformer 模型，并将 Self-attention 发扬广大。

### 实现
记 Self-attention 以 $\{\boldsymbol a_1, \boldsymbol a_2, \ldots, \boldsymbol a_n\}$ 作为输入，以 $\{\boldsymbol b_1, \boldsymbol b_2, \ldots, \boldsymbol b_n\}$ 作为输出。

首先，该算法需要注明如何判断两个向量的相似度 $\alpha$。其中一种方式称为 **Dot-product**，$\alpha = (\boldsymbol W_q \boldsymbol a_i) \cdot (\boldsymbol W_k \boldsymbol a_j)$，

Transformer 等模型使用的是 Dot-product 法。算法流程是这样的：

* 对 $\boldsymbol a_i$，计算 $\alpha(i, 1), \alpha(i, 2), \ldots, \alpha(i, n)$。
* 把 $\{\alpha(i, j)\}$ 做 Softmax 得到 $\{\alpha'(i, j)\}$。
* $\boldsymbol b_i = \sum_j \alpha'(i, j) \boldsymbol W_v \boldsymbol a_j$。

用向量和矩阵表示，就是

* $\boldsymbol q_i = \boldsymbol W_q \boldsymbol a_i \Leftrightarrow \boldsymbol Q = \boldsymbol W_q \boldsymbol A$
* $\boldsymbol k_i = \boldsymbol W_k \boldsymbol a_i \Leftrightarrow \boldsymbol K = \boldsymbol W_k \boldsymbol A$
* $\boldsymbol v_i = \boldsymbol W_v \boldsymbol a_i \Leftrightarrow \boldsymbol V = \boldsymbol W_v \boldsymbol A$
* $\boldsymbol G = \boldsymbol K^T \boldsymbol Q$（这里 $G_{i, j} = \alpha(j, i)$，注意是反着的）
* $\boldsymbol G' = \text{softmax}(\boldsymbol G)$（这里的 $\text{softmax}$ 是对每一列单独做的）
* $\boldsymbol B = \boldsymbol V \boldsymbol G'$

其中 $\boldsymbol W_q, \boldsymbol W_k, \boldsymbol W_v$ 是需要调整的参数。

### 改进
#### Multi-head Self-attention
以 2 个 head 为例。令 $\boldsymbol q_{i, 1} = \boldsymbol W_{q, 1}\boldsymbol q_i$，$\boldsymbol q_{i, 2} = \boldsymbol W_{q, 2}\boldsymbol q_i$，对 $\boldsymbol k_{i, 1}, \boldsymbol v_{i, 1}$ 也进行类似操作。然后根据 $\boldsymbol q_{:, 1}, \boldsymbol k_{:, 1}, \boldsymbol v_{:, 1}$ 算出 $\boldsymbol b_{:, 1}$（注意这里和第 2 个 head **完全独立，互不影响**），最后由 $\boldsymbol b = \boldsymbol W^\circ \begin{bmatrix}\boldsymbol b_{:, 1} & \boldsymbol b_{:, 2} \end{bmatrix}$ 计算出结果。

#### Positional Encoding
为了保留位置信息，为第 $i$ 个位置 $\boldsymbol a_i$ 额外加上 $\boldsymbol e_i$。$\boldsymbol e_i$ 的值是人为规定的，目前还没有表现极其突出的算法。

#### Truncated Self-attention
Attention matrix 大小是 $n \times n$ 的，当输入向量数量过多时会导致计算量过大。

可以用滑动窗口的方式，对于 $\boldsymbol b_i$，只会由 $\boldsymbol a_{i - k}, \boldsymbol a_{i - k + 1}, \ldots, \boldsymbol a_{i + k}$ 决定，这样大小就是 $2k \times n$ 的了。

### 其他应用
Self-attention 也可以用于语音分析（语音作为输入是不定长的），甚至可以用来处理图像，即把每个点视为三维向量（RGB）处理。

可以粗糙地理解为：Self-attention 相当于复杂的 CNN，它会自动学习决定感知区域的大小；CNN 则相当于简化版的 Self-attention。

*On the Relationship between Self-Attention and Convolutional Layers*（https://arxiv.org/abs/1911.03584 ）概述了二者的关系，从数学上说明 CNN 是 Self-attention 的严格子集。

*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*（https://arxiv.org/abs/2010.11929 ）的实验表明，数据较少时 CNN 可能优于 Self-attention，数据多时 Self-attention 可能优于 CNN。

若输入是一个图，则我们可将 Self-attention 稍作修改，限定对任意 $i, j$ 且 $(i, j) \notin E$，则 $\alpha(i, j) = 0$。这样会获得一个 Graph Neural Network（GNN）模型。

## RNN
### 基础
**Recurrent Neural Network**（循环神经网络，简称 RNN）是一种特殊的神经网络，在每次输入后都会保存一部分的神经元的输出（称为 Memory，记忆），这些输出会在下次计算时影响神经元计算。

RNN 可以用于长句子的 **Slot filling**，即对于每一个单词，决定它是什么种类的信息。如“I am going to Taipei at 14:00 on 5/21”，其中 Taipei 是目的地信息，5/21 和 14:00 是时间信息，其他算入 other 信息。在这种情况下，可以用同一个 RNN 从左到右输入每个单词，虽然 RNN 架构相同，但因为上次的神经元输出不同会影响到目前的输出。

**Elman Network** 是一种 RNN，它会在每次计算后保留所有 Hidden layer 的神经元的输出。**Jordan Network** 是另一种 RNN，只保留上一次的输出。据说后者表现更优。

### Bidirectional RNN
同时训练两个 RNN，一个正向读取，一个反向读取，然后把这两个网络的 Hidden layer 都接给一个 Output layer，就可以达到上下文推断的效果。

### Long Short-term Memory（LSTM）
上面提到的 Memory 实现其实是 RNN 的最简单的一个版本。

现在的 RNN 用的记忆方式称之为 **Long short-term memory**（长短期记忆），这个记忆池由 Memory cell（记忆细胞）、Input gate（输入阀门）、Forget gate（遗忘阀门）、Output gate（输出阀门）组成。输入输出阀门控制 Memory 能否被修改或读取，Forget gate 控制 Memory 是否要被清除。它们是由 RNN 自己学习何时打开何时关闭的。

![RNN 结构示意图](/img/post/ml-intro-ntu/40-1.jpg)

如上图，整个 Block 接受四个标量 $z, z_i, z_f, z_o$ 作为输入，其中 $z$ 为输入，$z_i, z_f, z_o$ 分别控制 input gate、forget gate 和 output gate。流程如下：

1. 输入 $z$ 经过函数 $g$ 变为 $g(z)$，然后与 $f(z_i)$ 相乘变为 $f(z_i)g(z)$。此处 $f$ 为激活函数，一般取 sigmoid。可见 $f(z_i) \in [0, 1]$ 的值相当于控制 $g(z)$ 是否能通过。
2. 存在记忆里的 $c$ 与 $f(z_f)$ 相乘变为 $f(z_f)c$，再加上输入的 $f(z_i)g(z)$ 作为新的 $c'$ 并更新，即
$$c' = f(z_i)g(z) + f(z_f)c$$  
3. 输出 $f(z_o)h(c')$。

在实践中，我们可以把神经元直接替换成 LSTM block，每个 block 的四个输入由上一层输出 $\boldsymbol x$ 点乘四个向量 $\boldsymbol w, \boldsymbol w_i, \boldsymbol w_f, \boldsymbol w_o$ 得到，相当于拥有的参数个数是神经元的 4 倍。

### Standard RNN（LSTM）
标准的 RNN（LSTM）就是在采用 LSTM 的前提下，把前一次的 memory 向量 $\boldsymbol c^{(t - 1)}$、前一次的输出向量 $\boldsymbol h^{(t - 1)}$ 和当前输入 $\boldsymbol x^{(t)}$ 连接起来，作为真正的输入。

### 难以训练
实际训练时，RNN 的训练效果可能不佳，甚至 cost 函数会不断上升下降，难以稳定。RNN 的发明者经过研究，发现 RNN 的 cost 图并不稳定，在一些地方极其平坦，在一些地方又极其陡峭。为了避免模型在陡峭时梯度过分变大，可以采用 **Clipping**，即让梯度与某个阈值（比如 $15$）取 $\min$，避免梯度过大。

![RNN 的 cost 函数示意图](/img/post/ml-intro-ntu/41-1.jpg)

为什么会有这种情况呢？考虑一个最简单的 RNN，其中 $F(x) = x$，每次记忆取出时都会乘一个常数 $w$。做 $1000$ 次输入，第一次输入为 $1$，其他输入都是 $0$。

![RNN 的 cost 函数示意图](/img/post/ml-intro-ntu/41-2.jpg)

此时可以注意到，$y^{(1000)} = w^{999}$。可以注意到：

* $w = 1$ 时 $y = 1$，但 $w = 1.01$ 时 $y \approx 2000$。此时正向偏导数非常非常大。
* $w = 1$ 时 $y = 1$，但 $w = 0.99$ 或 $w = 0.01$ 时 $y \approx 0$。此时反向偏导数又很小。

这代表着在 RNN 的偏导数会时大时小，难以有一个合理的学习率。更进一步地，对于这个 memory 的转移倍数 $w$ 会在训练数据时被反复乘积，导致它影响过大。

#### 解决方法
* LSTM：可以解决梯度消失，但不太能解决梯度爆炸
  * 因为输入对 memory 的影响等等都是求和，而不是乘积，且所有影响都会保留（除非 forget gate 关闭）
* GRU：基本约等于 LSTM，但是少了 forget gate，而是由 input gate 同步实现 forget gate 的功能，参数个数相当于 LSTM 的 $\frac 34$ 倍。

### 其他
RNN 也有其他改版：
* Clockwise RNN [Jan Koutnik, JMLR'14]
* Structurally Constrained Recurrent Network（SCRN） [Tomas Mikolov, ICLR'15]
* 朴素 RNN（无 LSTM），以单位矩阵为参数，ReLU 作为激活参数 [Quoc V. Le, arXiv'15]
  * 在 4 个不同的任务上表现和 LSTM 持平甚至更强 

以及除了等长输出（输入向量和输出向量一样多），也可以调整为只输出一个向量，或不固定输出数量个数。调整方法是比较简单的，这里不赘述。

## GAN
### 最基础的模型
记生成网络为 $G$，判别网络为 $D$，生成的数据分布为 $P_G$，原始数据分布为 $P_{data}$。

再记 $D : \R^n \rightarrow [0, 1]$，$D(\boldsymbol x)$ 表示判别网络认为 $\boldsymbol x$ 为原始数据的概率。$E_{\boldsymbol x \sim P_G}[F(\boldsymbol x)]$ 表示随机从 $P_G$ 取样一个 $\boldsymbol x$，$F(\boldsymbol x)$ 的期望值。

训练过程如下：

1. 随机初始化 $G, D$。
2. 固定 $G$ 不变，训练 $D$，使 $E_{\boldsymbol x \sim P_{data}}[\log D(\boldsymbol x)] + E_{\boldsymbol \sim P_G}[\log(1 - D(\boldsymbol x))]$ 最大。
3. 固定 $D$ 不变，训练 $G$，使 $E_{\boldsymbol x \sim P_G}[\log(1 - D(\boldsymbol x))]$ 最小。

用数学语言来说，设 $V(D, G) = E_{\boldsymbol x \sim P_{data}}[\log D(\boldsymbol x)] + E_{\boldsymbol \sim P_G}[\log(1 - D(\boldsymbol x))]$，则这是一个 minimax 博弈问题，需要求解的是

$$\min_G \max_D V(D, G)$$

### WGAN
基础模型中使用的 $V(D, G)$ 其实就是 $P_G$ 和 $P_{data}$ 的 JS divergence（关于信息熵相关可以查阅相关资料或《[杂文：什么是信息熵](/learning-note/misc-entropy)》）。也可以使用别的 Divergence，可参考 [arXiv:1606.00709](https://arxiv.org/abs/1606.00709)。

实际上，即使将 JS divergence 求解出来，GAN 的表现仍然不优（GAN 的特点：难以训练）。接下来解释原因。

1. 分布本身的特性：$P_{data}$ 是高维空间里极其狭小的一块区域（low-dim manifold in high-dim space），导致二者空间难以重合，容易区分。
2. 取样过程导致的损失：如果取样不足够多，即使二者空间有一定重合，也容易导致判别器可以求出足够好（甚至 overfitting）的判别平面，使得二者被区分。

极端地，若 $P_{data}$ 和 $P_G$ 严格不重合，则 JS divergence 恒为 $\log 2$。这也可以从分类器的角度上解释：如果两者严格不重合，则分类器收敛时正确率为 $100\%$。

后人改进 GAN，使用了 Wasserstein distance 替代 JS divergence。Wasserstein distance 可以认为是：将 $P_{data}$ 的数据任意移动，使得分布变为 $P_G$ 的最小距离之和。改进后的 GAN 称为 **WGAN**，其求解最优化问题为

$$\max_{D \in \text{1-Lipschitz}} \{E_{x \sim P_{data}}[D(x)] - E_{x \sim P_G}[D(x)]\}$$

其中 $D \in \text{1-Lipschitz}$ 表示 $D$ 需要是 One-Lipschitz 的。这保证 $D$ 需要足够平滑，否则当 $P_{data}$ 和 $P_G$ 不重合时，训练出的判别网络可能会使得 $D(x) \rightarrow \infty$ 而无法收敛。

早期为了保证 One-Lipschitz，采用的是对判别网络的参数绝对值做常数限制（即令 $w_{clip} \leftarrow \max\{\min\{w, C\}, -C\}$）。后来在 [arXiv:1704.00028](https://arxiv.org/abs/1704.00028) 中改进为 Gradient Penalty（无需了解），近期则是采用 [arXiv:1802.05957](https://arxiv.org/abs/1802.05957) 的 Spectral Normalization，保证每一处的梯度都小于 $1$。

### 更多提示
* Soumith 的提示：https://github.com/soumith/ganhacks
* DCGAN 的提示：关于图像生成问题的网络构造教程 [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
* 训练 GAN 的进阶方式：[arXiv:1606.03498](https://arxiv.org/abs/1606.03498)
* BigGAN 的提示：[arXiv:1089.11096](https://arxiv.org/abs/1089.11096)

### ScratchGAN
序列生成 GAN 不能微分，因而不能用梯度下降训练。同时因为 RL 和 GAN 都很难训练，因而也不适合用 RL 训练生成网络。

[arXiv:1905.09922](https://arxiv.org/abs/1905.09922) 提出了 ScratchGAN，从最基础的 SequenceGAN 经过大量的超参数微调产生表现较为优秀的序列生成 GAN。

### 可选：其他生成模型
除此之外，还有 VAE（Variational Autoencoder）和 FLOW-Based Model 等模型。有兴趣可以看看，但实际 GAN 还是比较优秀的。

还有另一种生成的方式，将图片 $x$ 转化为向量 $f(x)$，作为输入提供给网络，输出为原图片，然后做监督式学习，也就是找出一种可能的反函数 $f^{-1}(y)$。训练得到反函数后，我们只需要取图片 $x$ 和随机噪音 $\gamma$，计算 $f^{-1}(f(x + \gamma))$ 就可以得到新图片 $x'$。这里构造 $f(x)$ 的方式也是需要一定技巧的，可以参考 [arXiv:1707.05776](https://arxiv.org/abs/1707.05776) 和 [arXiv:2007.02798](https://arxiv.org/abs/2007.02798)。

### 衡量生成器的好坏
如果让人来分辨，则可能会受到主观因素影响，也不稳定。

考虑提前训练一个图像分类网络，将生成的图像丢给它，若产生的概率分布极为集中（即，分类网络的自信度高），则认为生成器是好的，**质量**高。

然而这样做可能引发 Mode collapse（模式崩塌），即生成分布单一地围绕某个数据，使得生成出来的数据都极为相似，此时生成网络能通过图像分类网络和 GAN 里分类器的测试，但生成效果很单一。目前暂时没有较好的解决方式，只能在训练时发现 Mode collapse 时停止训练。

更难被侦测到的问题是 Mode dropping（模式丢失），即生成分布只拟合了一部分实际分布，而丢失了部分分布。例如，人脸生成表现优秀，但生成的肤色只有白色，丢失了黄色和黑色的生成。

为了缓解这两个问题，可以将所有图像分类网络的输出做一个平均，即，求解分类为第 $i$ 种的期望概率 $E(P(\text{class}_i))$，若 $|E(P(\text{class}_i)) - E(P(\text{class}_j))|$ 较小，则认为输出图像的分类多，也就是**多样性**高。综合多样性和质量，可以使用 **Inception score**（IS）。多样性越高，质量越好，IS 越高。

当输出分类区分不够明显时，可考虑使用 **Fréchet 
inception distance**（FID），取进入 softmax 层前的输出向量代表图片，再将分布假定为正态分布，计算两个分布之间的 Fréchet distance。

然而还有一种情况，就是当模型表现过分优秀时，生成的图片和训练集的图片几乎一模一样，但模型不会生成其他新的图片。关于更进一步的评估方式，可以参考 [arXiv:1802.03446](https://arxiv.org/abs/1802.03446)。

### 条件 GAN
普通的随机生成 GAN 满足不了我们自定义的需求。

考虑更改生成网络。在将随机高斯噪音 $\boldsymbol z$ 映射到输出 $\boldsymbol y$ 的基础上，增加一个条件输入 $\boldsymbol x$，使得生成器根据条件 $\boldsymbol x$ 和噪音 $\boldsymbol z$ 生成我们需要的 $\boldsymbol y$。例如做文本转图像，则 $\boldsymbol x$ 就是给定的文本，$\boldsymbol y$ 是生成的图像，而 $\boldsymbol z$ 是使得生成图像随机发生一定变化的随机“噪音”。

相对应地，也需更改判别网络，否则生成网络只会学习如何生成真实输出，而完全忽略条件输入。我们需要提供原条件 $\boldsymbol x$ 和输出数据 $\boldsymbol y$ 给判别网络，此时判别网络认为一个数据是原数据，当且仅当输出 $\boldsymbol y$ 看起来是真实的，且输出 $\boldsymbol y$ 和条件 $\boldsymbol x$ 是对应的。为了让判别网络学习到这点，需要放入原数据（标为真实的），以及输出和条件都正确但不相符（随机打乱）的原数据（标为虚假的）。

条件 GAN 不仅可以用于文字转图片，图片转图片（图片处理）、
