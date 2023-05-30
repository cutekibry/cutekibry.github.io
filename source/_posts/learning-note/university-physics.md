---
title: 《大学物理》笔记
date: 2023-04-17
category: 
- 学习
excerpt: 讨厌物理。
---

# 大学物理（上）
## 第 5 章 狭义相对论
**原时**：将在一个惯性系中测得的，发生在该惯性系同一地点的两个事件之间的时间间隔称为原时。

**原长**：相对于测量的长度相对静止的观测者所测量的长度为原长。

**洛伦兹变换**：记 $\beta = \frac vc$，收缩因子 $\gamma = \frac 1{\sqrt {1 - \beta^2}} > 1$，则有坐标变换

$$x' = \gamma(x - vt)$$

$$t' = \gamma(t - \beta \frac xc)$$

逆变换

$$x = \gamma(x' + vt')$$

$$t = \gamma(t' + \beta \frac {x'}c)$$




**时间延长**：若在原惯性系 $S$ 中同一地点间隔 $\Delta t$ 时间发生了两个事件，则

$$\Delta t' = \gamma(\Delta t - \beta \frac {\Delta x}c)$$

因为地点相同所以 $\Delta x = 0$，即

$$\Delta t' = \gamma\Delta t$$

**长度缩短**：考虑原惯性系 $S$ 中同一时间测得两个静止的地点距离为 $\Delta x$，现在观测者在另一惯性系中 $S'$ 测量其长度。注意到，此时观测者身处 $S'$，应当在相对 $S'$ 来说的同一时间点测量长度才有意义，也就是 $\Delta t' = 0$。进行洛伦兹逆变换，得到

$$
\begin{aligned}
    \Delta t' &= \gamma(\Delta t - \beta \frac {\Delta x}c) \\
    0 &= \gamma(\Delta t - \beta \frac {\Delta x}c) \\
    0 &= \Delta t - \beta \frac {\Delta x}c \\
    \Delta t &= \beta \frac {\Delta x}c \\
\end{aligned}
$$

因此

$$
\begin{aligned}
    \Delta x' &= \gamma(\Delta x - v\Delta t) \\
    &= \gamma(\Delta x - v\beta\frac {\Delta x}c) \\
    &= \gamma(1 - \beta^2)\Delta x \\
    &= \frac {\Delta x}\gamma \\
\end{aligned}
$$

上面的一些代换只是将 $\beta = \frac vc, \gamma = \frac 1{\sqrt{1 - \beta^2}}$ 代入。

**速度变换**：

$$u_x' = \frac {u_x \mp  v}{1 \mp \frac {u_xv}{c^2}}$$

$$u_y' = \frac {u_y}{\gamma(1 \mp \frac {u_xv}{c^2})}$$

当 $u_x, v$ 同向时取负号。

----

动力学

* 质量：$m = \gamma m_0$，速度越快质量越大
* 总能量 $E$，静止能量 $E_0$ 和动能 $E_k$：$E = E_0 + E_k$，可知 $E_k = (m - m_0)c^2$
* 动量 $p = mv = m_0 \gamma v$
* $E^2 = p^2c^2 + E_0^2$

## 第 6 章 分子动理论
### 第 1 节 热力学系统与平衡态
概念：

* **孤立系**：无能量和物质交换
* **封闭系**：有能量，无物质交换
* **开放系**：有能量和物质交换
* **平衡态**：处于动态平衡的**孤立系**
  * 对两端分别放在热水和冰中的铁棒，即使达到动态平衡也不是平衡态，因为铁棒不是孤立系
* **物态方程**：$T= T(p, V)$ 等
* **状态参量空间**：以独立的状态参量为坐标可构成空间
* **准静态过程和非静态过程**：前者无限缓慢，任意时刻都在平衡态；后者不一定在平衡态
* **热接触**：两个系统通过导热板接触
* **复合系统**：热接触的两个系统

**热力学第零定律**：若 A=C，B=C，则 A=B。

理想气体方程：

$$pV = vRT$$

其中 $v$ 是**总物质的量**，$R = 8.31 \text{J} / (\text{mol} \cdot \text{K})$ 为**摩尔气体常量**。

根据**阿伏伽德罗常量** $N_A = 6.022 \times 10^{23} \text{mol}^{-1}$ 可以引入**玻尔兹曼常量**

$$k = \frac R{N_A} = 1.38 \times 10^{-23} \text{J} / \text{K}$$

则有

$$p = nkT$$

其中 $n = \frac NV$ 是**单位体积的分子数**，也叫**分子数密度**。据此可得标态下分子数密度 $n_0 = \frac p{kT} = 2.69 \times 10^{25} \text{m}^{-3}$，称为**洛施密特常量**。

**混合气体平均摩尔质量**：混合气体平均摩尔质量 $\overline M$ 满足

$$\frac 1{\overline M} = \sum_i \frac {m_i}m \frac 1{M_i}$$

#### 理想气体微观模型
结论：

$$p = \frac 13nm_f \overline {v^2}$$

其中 $m_f$ 为单个分子质量。

记单个分子平均**平动**动能 $\overline \epsilon_t = \frac 12 m_f \overline {v^2}$，上式写作

$$p = \frac 23n\overline \epsilon_t$$

可推得

$$\overline \epsilon_t = \frac 32 kT$$

### 第 3 节 能量均分定理
自由度：

| 分子种类 | $t$ | $r$ | $s$ | $i = t + r + s$ |
| :-: | :-: | :-: | :-: | :-: |
| 单原子 | $3$ | $0$ | $0$ | $3$ |
| 刚性双原子 | $3$ | $2$ | $0$ | $5$ |
| 非刚性双原子 | $3$ | $2$ | $1$ | $6$ |
| 刚性多原子 | $3$ | $3$ | $0$ | $6$ |
| 非刚性多原子 | $3$ | $3$ | $3n - 6$ | $3n$ |

**能量均分定理**：**每个自由度的平均动能都等于 $\frac {kT}2$**，即可知理想气体的分子平均**总**动能 $\overline \epsilon_k$ 为

$$\overline \epsilon_k = \frac i2kT$$

其中 $\overline \epsilon_k = \overline \epsilon_t + \overline \epsilon_r + \overline \epsilon_s$。

分子平均总**能量**为

$$\overline \epsilon = \overline \epsilon_k + \overline \epsilon_p$$

其中 $\overline \epsilon_p$ 为平均势能。根据振动理论知 $\overline \epsilon_p = \overline \epsilon_s$，因此

$$\overline \epsilon = \frac 12 (t + r + 2s)kT$$

气体内能为

$$
\begin{aligned}
    E = N\overline \epsilon &= \frac 12 (t + r + 2s)NkT \\
    &= \frac 12 (t + r + 2s)vRT \\
\end{aligned}
$$

### 麦克斯韦速率分布
$$f(v) = \frac {\text d N}{N \text{d} v}$$