---
title: 《微积分》（下）笔记
date: 2023-06-02
category: 
- 学习
tags:
- 文化课
---

多元函数连续：

$$\lim_{P \rightarrow P_0} f(P) = f(P_0)$$

多元函数可微：需要在点 $P_0$ 连续，且

$$\lim_{(\rho = \sqrt {x^2 + y^2}) \rightarrow 0} \frac {f(x_0 + x, y_0 + y) - f(x_0, y_0) - f_x(x_0, y_0)x - f_y(x_0, y_0)y}\rho = 0$$


### 9.4
若 $F(x, y, z) = 0$，则 $\bold{grad}\ F = \{F_x, F_y, F_z\}$ 是法向量。

### 9.5 极值
若 $AC - B^2 < 0$，则不是极值点。

否则当 $A > 0$ 或 $C > 0$ 时极小，$A < 0$ 时极大。

拉格朗日乘数法？

### 11.1 第一型曲线积分
$$\int_L f\text ds = \int_a^b f(x(t), y(t), z(t)) \sqrt {x'(t)^2 + y'(t)^2 + z'(t)^2}\text dt$$

### 11.2 第二型曲线积分
$$
\begin{aligned}
    \int_L \boldsymbol{F} \cdot \text d\boldsymbol{r} &= \int_L P\text dx+Q\text dy+R\text dz \\
    &= \int_a^b [Px'(t) + Qy'(t) + Rz'(t)]\text dt \\
\end{aligned}
$$

Green 公式：$xy$ 平面上，若在 $D$（**包括边界**）有连续偏导数，则

$$\oint_L P\text dx + Q \text dy = \iint_D (Q_x - P_y)\text dx\text dy$$

当 $Q_x = P_y$ 时积分与路径无关。

### 11.3 第一型曲面积分
$$\iint_S f\text S\sigma = \iint_D f(x, y, z(x, y))\sqrt {1 + z_x^2 + z_y^2} \text dx \text dy$$

### 11.4 第二型曲面积分
$$\iint_S \boldsymbol{F} \cdot \boldsymbol{n} \text dS = \iint_S P\text dy\text dz+Q\text dz\text dx+R\text dx\text dy$$

### 11.5 Gauss 公式与 Stokes 公式
分别称

$$\text {div}\ \boldsymbol{F} = P_x + Q_y + R_z$$

$$\bold {rot}\ \boldsymbol{F} = \{R_y - Q_z, P_z - R_x, Q_x - P_y\}$$

为 $\boldsymbol{F}$ 的**散度**和**旋度**。

引入 **Hamiliton 算子** $\nabla$

$$\nabla = \{\frac \partial{\partial x}, \frac \partial{\partial y}, \frac \partial{\partial z}\}$$

则可记为

$$\text {div}\ \boldsymbol{F} = \nabla \cdot \boldsymbol{F}$$

$$\bold {rot}\ \boldsymbol{F} = \nabla \times \boldsymbol{F}$$

$$\bold {grad}\ \boldsymbol{F} = \nabla \boldsymbol{F}$$

以及记 $\Delta = \nabla^2$ 为 **Laplace 算子**。

**Gauss 公式**：

$$\oiint_S \boldsymbol{F} \cdot \boldsymbol{n} \text dS = \iiint_V (P_x + Q_y + R_z)\text dv$$

**Stroke 公式**：

$$\oint_L \boldsymbol{F} \cdot \text d\boldsymbol{r} = \iint_S \bold{rot}\ \boldsymbol{F} \cdot \boldsymbol{n} \text dS$$

同样地，当 $\bold{rot}\ \boldsymbol{F} = \boldsymbol{0}$，则积分与路径无关。

### 12.1 数项级数
基本方法：比值法、根值法

**Liebniz 判别法**

### 12.3 幂级数
收敛区间：记

$$\rho = \lim_{n \rightarrow \infty}\left|\frac {a_{n + 1}}{a_n}\right|$$

则 $R = \frac 1{\rho}$。