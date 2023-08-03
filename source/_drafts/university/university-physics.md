---
title: 《大学物理》笔记
date: 2023-04-17
category: 
- 学习
tags:
- 文化课
excerpt: 讨厌物理。
---

## 微积分
$$\int \frac 1{\sqrt {x^2 \pm r^2}} = \ln(x + \sqrt {x^2 \pm r^2})$$

## 第 1 章 质点运动学
当 $t = 0$ 时 $x = x_0, v = v_0$。若 $a = kx$，求 $v(x)$。

解：

$$a = \frac {\text dv}{\text dt} = \frac {\text dv}{\text dx} \frac {\text dx}{\text dt} = v\frac {\text dv}{\text dx} =kx$$

因此

$$\int_{v_0}^v v\text dv = \int_{x_0}^x kx \text dx$$

$$\frac {v^2 - v_0^2}2 = k\frac {x^2 - x_0^2}2$$

$$v = \sqrt {k(x^2 - x_0^2) + v_0^2}$$

## 第 2 章 牛顿运动定律
### 角动量的变化和守恒
**角动量 / 动量矩** $\boldsymbol L$：

$$\boldsymbol L = \boldsymbol r \times \boldsymbol p = \boldsymbol r \times m\boldsymbol v$$

**力矩** $\boldsymbol M$：

$$\boldsymbol M = \boldsymbol r \times \boldsymbol F = \frac {\text d\boldsymbol L}{\text dt}$$

角动量定理和动量定理其实是类似的，只是在前面乘了一个 $\boldsymbol r$ 而已。

角动量定理可以用来描述较有规律的圆周运动。

## 第 3 章 刚体的定轴转动
**相对力矩**：$\boldsymbol M_z = \boldsymbol r \times \boldsymbol F_\perp$ 称为 $\boldsymbol F$ 相对转轴的力矩。

**转动惯量**：$J = mr^2$ 称为转动惯量。特别地，$\boldsymbol L = J\boldsymbol w$。

**刚体转动定理**：

$$\boldsymbol M_z = J\boldsymbol \beta$$

可以与

$$\boldsymbol F = m\boldsymbol a$$

比较。$m$ 代表平动惯性大小，$J$ 代表转动惯性大小。

**平行轴定理**：轴 $A \parallel$ 轴 $C$，且轴 $C$ 过质点，距离为 $d$，则

$$J_A = J_C + md^2$$

## 第 4 章 流体运动简介
### 理想流体
**伯努利方程**：

$$p + \rho g h + \frac 12 \rho v^2 = C$$

### 黏性流体
**速度梯度**：$\frac {\text dv}{\text dx}$ 表示垂直于速度方向的，相邻单位距离液层间的速度差。

**牛顿粘性定律**：粘性力 $F$ 满足

$$F = \eta\frac {\text dv}{\text dx}S$$

其中 $\eta$ 称为**黏性系数**。

**雷诺数**：

$$\text {Re} = \frac {\rho vd}\eta$$

当 $\text{Re} < 2000$ 时为层流，$\text{Re} > 3000$ 为湍流，介于两者之间为过渡流。

**粘性流体伯努利方程**：

$$\Delta (p + \rho g f + \frac 12 \rho v^2) = \frac {\Delta E_{内}}{\Delta V}$$

**泊肃叶定律**：

$$Q = \frac {\pi R^4}{8\eta L}(p_1 - p_2)$$

其中 $R_f = \frac {8\eta L}{\pi R^4}$ 称为流阻，和电阻一样满足串联并联定律。

**斯克托斯定律**：

$$f = 6\pi \eta rv$$

## 第 5 章 狭义相对论
**原时**：将在一个惯性系中测得的，发生在该惯性系同一地点的两个事件之间的时间间隔称为原时。

**原长**：相对于测量的长度相对静止的观测者所测量的长度为原长。

**洛伦兹变换**：记 $\beta = \frac vc$，收缩因子 $\gamma = \frac 1{\sqrt {1 - \beta^2}} > 1$，则有坐标变换（其中带 $'$ 为变换后的，不带的为原长和原时），且 $v$ 向右则 $> 0$：

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

## 第 7 章 热力学基础
* **内能**：对于封闭系统的某个过程，过程中对外做功 $A$ 和吸收的热量 $Q$ 满足 $Q - A$ 为定值，则可以定义内能 $E$ 满足 $\Delta E = Q - A$。

$$Q = A + \Delta E$$

称为**热力学第一定律**。

准静态过程对外做功 $A$ 满足

$$A = \int_{V_1}^{V_2} p\text{d} V$$

* **热容**：表示某系统升高 1 度需要吸收的热量，记为 $C$。有 $Q = \int_{T_1}^{T_2} C \text{d}T$。
  * **摩尔热容**：表示 $1\ \text{mol}$ 某物质的热容，记为 $C_m$。
  * **比热容 / 质量热容**：表示 $1\ \text{kg}$ 某物质的热容，记为 $c$。
* **定压热容 / 定容热容**：在压强不变 / 体积不变情况下的热容，记为 $C_p$ 和 $C_V$。

### 第 2 节 理想气体热容
注意到

$$C_{V, m} = \frac 1v \left(\frac {\text dQ}{\text dT}\right)_V = \frac 1v \left(\frac {\text dE + p\text dV}{\text dT}\right)_V = \frac 1v \left(\frac {\text dE}{\text dT}\right)_V = \frac 1v \left(\frac {\text dE}{\text dT}\right) = \frac {\text dE_m}{\text dT}$$

$$C_{p, m} = \frac 1v \left(\frac {\text dQ}{\text dT}\right)_p = \frac 1v \left(\frac {\text dE + p\text dV}{\text dT}\right)_p = \frac 1v \left(\frac {\text dE}{\text dT}\right)_p + \frac pv\left(\frac {\text dV}{\text dT}\right)_p = \frac {\text dE_m}{\text dT} + R$$

注意，因为理想气体内能只和温度有关，所以 $\left(\frac {\text dE}{\text dT}\right)_V = \left(\frac {\text dE}{\text dT}\right)_p = v\text dE_m$。

$$C_{p, m} = C_{V, m} + R$$

称为**迈耶公式**。

定义**摩尔热容比**为

$$\gamma = \frac {C_{p, m}}{C_{V, m}} = 1 + \frac R{C_{V, m}} > 1$$

有时热量交换不导致温度变化，而是导致其他状态参量变化。如固体融化成液体吸热，但温度不变。

定义相变过程中吸收热量叫**潜热**。融化时叫**熔化热**，汽化叫**汽化热**。

**特别注意，只有理想气体时内能才是温度的单值函数。题目未提则不可当作当然。**

### 第 3 节 热力学第一定律对理想气体的应用
三个基本方程：

任意过程都满足的：

<!-- $$\Delta E = vC_{V, m}T$$ -->

$$Q = v\int C_m \text d{T}$$

非真空内：

$$A = \int p\text dV$$

#### 绝热过程
**绝热过程**：不和外界发生热交换的过程。**注意：绝热代表 $Q = 0$，但不代表 $\Delta T = 0$**。

**绝热方程**：

$$pV^\gamma = C_1$$

也可以用 $pV = vRT$ 换成 $TV$ 或 $pT$ 相关的方程。

**注意**：气体绝热自由膨胀不做功（外面是真空），$A = 0$，所以 $\Delta E = Q - A = 0$。但它不是等温过程，也不是准静态过程，所以**不能使用绝热方程，绝热方程只对理想气体有用**。

#### 统一过程
可以用统一的公式来刻画**理想**气体的各种等值过程：

$$pV^n = \text{Constant}$$

其中 $n$ 是常数。$n = 0$ 对应等压，$n = 1$ 对应等温，$n = \gamma$ 对应绝热，$n \rightarrow \infty$ 对应等容。此外，$1 < n < \gamma$ 对应介于等温到绝热中间的某种过程，称为**多方过程**。

### 第 4 节 循环过程 卡诺循环
#### 热机
**热机的效率**：让净功与吸收热量的比最大，即

$$\eta = \frac A{Q_1} = 1 - \frac{Q_2}{Q_1}$$

**卡诺循环**：利用一个高温热源、一个低温热源设计的热机，过程如下：

1. 高温吸热：设热机温度和高温热源相同，等温膨胀，吸热 $Q_1$，不做功
2. 绝热膨胀：热机离开热源，继续膨胀，不吸热，对外做功 $A_1$，温度降低
3. 低温放热：设热机温度降低至和低温热源相同，等温压缩，放热 $Q_2$，不做功
4. 绝热压缩：热机离开热源，继续压缩，不放热，外界对气体做功 $A_2$，温度升高

对外界做的净功

$$A = A_1 - A_2 = Q_1 - Q_2$$

效率为

$$\eta = 1 - \frac {Q_2}{Q_1} = 1 - \frac {T_2\ln \frac {V_3}{V_4}}{T_1 \ln \frac {V_2}{V_1}}$$

又根据 $T_1V_2^{\gamma - 1} = T_2V_3^{\gamma - 1}, T_1V_1^{\gamma - 1} = T_2V_4^{\gamma - 1}$ 得 $V_2V_4 = V_1V_3$，因此

$$\eta = 1 - \frac {T_2}{T_1}$$

这是相同热源之间效率最高的热机。

#### 制冷机
**制冷系数**：让吸收的热量与净功的比最大，即

$$w = \frac {Q_2}A = \frac {Q_2}{Q_1 - Q_2}$$

**卡诺制冷机**：卡诺热机逆过程，类似地有

$$w = \frac {T_2}{T_1 - T_2}$$

### 第 6 节 熵
对于卡诺热机，由

$$\eta = 1 + \frac {Q_2}{Q_1} = 1 - \frac {T_2}{T_1}$$

得

$$\frac {Q_2}{T_2} + \frac {Q_1}{T_1} = 0$$

可以看成任意多个小卡诺循环之和，其中大部分过程因为同时包含在两个卡诺循环中方向相反的过程里而相互抵消，只有边界上的还保留。因此有

$$\sum_i \frac {Q_i}{T_i} = 0$$

即

$$\oint \frac {\text dQ}T = 0$$

推论：$\int_{a \text{ to } b} \frac {\text dQ}T$ 和路径无关。

定义**熵** $S$ 使得

$$S_2 - S_1 = \int_1^2 \frac {\text dQ}T$$

**熵增加原理**：对于任意热机有

$$\eta = 1 + \frac {Q_2}{Q_1} \leq \eta_{\text 可} = 1 - \frac {T_2}{T_1}$$

得

$$\frac {Q_2}{T_2} + \frac {Q_1}{T_1} \leq 0$$

即

$$\oint \frac {\text dQ}T \leq 0$$

拆成过程 $1a2$ 和可逆过程 $2b1$，则有

$$
\begin{aligned}
  \int_{1a2} \frac {\text dQ}T + \int_{2b1} \frac {\text dQ}T &\leq 0 \\
  \int_{1a2} \frac {\text dQ}T + S_1 - S_2 &\leq 0 \\
  \int_{1a2} \frac {\text dQ}T &\leq S_2 - S_1 \\
  \frac {\text dQ}T &\leq \text dS \\
\end{aligned}
$$

称为**克劳修斯不等式**。当过程绝热，有

$$\text dS \geq 0$$

表明**绝热系统熵不减少**，称为**熵增加原理**。

**注意**：无法从非可逆过程本身计算出熵变，因为熵是定义在可逆过程上的。例如上面，我们不能认为不可逆过程 $1a2$ 的熵变大于可逆过程 $2b1$ 的熵变。

但只要知道起点终点，我们就可以构造一个可逆过程将其连接起来，计算熵变。

## 第 8 章 静电场
$$\epsilon_0 = 8.8542 \times 10^{-12} \text C^2 / (\text N \cdot \text m^2)$$

圆环电场：

$$E = \frac {Qx}{4\pi \epsilon_0 (R^2 + x^2)^{3/2}}$$

---

圆盘电场：

$$E = \frac {\sigma}{2\epsilon_0}\left(1 - \frac x{\sqrt {R^2 + x^2}} \right)$$

---

球壳：取任意带状圆环，其距离 $x = R\cos \theta$，半径 $r = R\sin \theta$，宽度 $\text dl = R\text d\theta$，计算电场影响。

结论：球面内 $E = 0$，球面外 $E = \frac Q{4\pi \epsilon_0 x^2}$。

---

**重要结论**：无限长带电直线满足

$$E = \frac {\lambda}{2\pi \epsilon_0 r}$$

无限大带电平面满足

$$E = \frac {\sigma}{2\epsilon_0}$$

空心球面满足

$$V_{内} = \frac {q}{4\pi\epsilon_0R_{球}}$$

$$V_{外} = \frac {q}{4\pi\epsilon_0r}$$

### 第 3 节 静电场的高斯定理
电场强度通量 

$$\phi = \int_S E \cdot \text dS$$

**高斯定理**：闭合曲面 $\phi$ 等于曲面内 $q_i$ 之和的 $\frac 1{\epsilon_0}$ 倍。

### 第 4 节 静电场的环路定理
**保守力场**：静电场做功只和起终点有关，和路径无关。

**无旋场**：静电场环路做功为 $0$。

### 第 5 节 电势差和电势
**电势**：$V_P$ 等于把单位正电荷从点 $P$ 移到 $V = 0$ 处做的功。电势能记为 $W$，且 $A_{ab} = q(V_a - V_b)$。

当场源电荷分布在有限空间，可选无限远处 $V = 0$。否则若在无限空间，则选在有限远处某点 $V = 0$。

计算电势时，选择点 $P$ 到零点的一条路径进行积分即可。

**电势梯度**：即 $\text{grad} V = \nabla V = -E$。

### 第 6 节 静电场中的导体
**静电平衡**：导体内部自由电子在静电场力作用下移动并重新分布，直到分布产生的电场恰与静电场抵消，此时达成静电平衡，其充要条件为

* 导体内部 $E = 0$，且
* 导体表面上 $E$ 垂直于表面。

或

* 导体是等势体，且导体表面是等势面，即整个导体内部和表面都满足 $V = \text{Const}$。

---

**静电平衡的性质**：

1. 内部无净电荷，电荷只能分布在表面上。
2. **对孤立导体**，表面曲率越大，电荷面密度 $\sigma_P$ 越大。
3. $E_P = \frac {\sigma_P}{\epsilon_0}$。

**尖端放电**：尖端处表面曲率大，电场强，当超过空气的击穿场强时，空气被电离，异号离子吸引至尖端，同号离子被排斥（“喷射”）。

---

**静电屏蔽**：内部无带电体，则导体壳内表面无电荷，$E = 0$。

若有带电体，则内表面附着电荷 $q_{内} = -q$，外表面 $q_{外} = Q + q$。若将壳接地，则 $q_{外} = 0$，达到静电屏蔽。

### 第 7 节 静电场中的电介质
**电介质**：即绝缘体。

**电介质的极化**：电偶极子在外界电场力作用下发生分离。

**电极化强度矢量** $\boldsymbol{P}$：电介质内某点附近，单位体积的分子电矩的矢量和，即

$$\boldsymbol{P} = \frac {\sum \boldsymbol{p}_i}{\Delta V}$$

实验证明，

$$\boldsymbol{P} = \chi_c \epsilon_0 \boldsymbol{E}$$

其中 $\chi_c = \epsilon_r - 1$ 是电介质的**极化率**。

**极化电荷** $q_p$：在电介质表面产生的电荷，有 $\oint \boldsymbol{P} \cdot \text d\boldsymbol{S} = -q_p$。


**电位移矢量 $\boldsymbol{D}$ 的高斯定理**：

$$\oint \boldsymbol{E} \cdot \text d\boldsymbol{S} = \frac 1{\epsilon_0}(q_f + q_p)$$

或

$$\oint \boldsymbol{D} \cdot \text d\boldsymbol{S} = q_f$$

其中 $\boldsymbol{D} = \epsilon_0\boldsymbol{E} + \boldsymbol{P} = (1 + \chi_c)\epsilon_0\boldsymbol{E} = e_r\epsilon_0\boldsymbol{E}$。

## 第 9 章 恒定磁场
### 磁性与磁场
磁场力 $\boldsymbol F$ 满足：

$$\boldsymbol F = q\boldsymbol v \times \boldsymbol B$$

$$\boldsymbol F = I\int_L \text d\boldsymbol l \times \boldsymbol B$$

**毕奥-萨法尔定律**：

$$\boldsymbol B = \frac {\mu_0}{4\pi} \int_L \frac {I\text d\boldsymbol l \times \boldsymbol e_r}{r^2}$$

$\mu_0 = 4\pi \times 10^{-7}\ \text T \cdot \text m/\text A$ 为真空磁导率。

**重要结论**：对于定长的直线电流 $I$ 在空间激发的磁场：

$$B_x = \frac {\mu_0}{4\pi} \frac Ix(\cos \theta_1 - \cos \theta_2)$$

**磁场高斯定理**：

$$\oint_S B \text dS = 0$$

**安倍环路定理**：

$$\oint_L B \text dl = \mu_0 \sum_S I_i$$