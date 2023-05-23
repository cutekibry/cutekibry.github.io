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
