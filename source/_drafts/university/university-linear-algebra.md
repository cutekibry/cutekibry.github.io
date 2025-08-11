---
title: 《线性代数》笔记
date: 2023-05-09
category: 
- 学习
tags:
- 文化课
excerpt: 三天速通线代！
---

## 行列式
练习一 4. (1) 计算行列式

$$\left |\begin{matrix}
    3 & 1 & 1 & 1 \\
    1 & 3 & 1 & 1 \\
    1 & 1 & 3 & 1 \\
    1 & 1 & 1 & 3 \\
\end{matrix} \right |$$

方法：累加法。

$$\left |\begin{matrix}
    3 & 1 & 1 & 1 \\
    1 & 3 & 1 & 1 \\
    1 & 1 & 3 & 1 \\
    1 & 1 & 1 & 3 \\
\end{matrix} \right | = \left |\begin{matrix}
    6 & 6 & 6 & 6 \\
    1 & 3 & 1 & 1 \\
    1 & 1 & 3 & 1 \\
    1 & 1 & 1 & 3 \\
\end{matrix} \right | = 6\left |\begin{matrix}
    1 & 1 & 1 & 1 \\
    1 & 3 & 1 & 1 \\
    1 & 1 & 3 & 1 \\
    1 & 1 & 1 & 3 \\
\end{matrix} \right | = 48$$

练习二 4. (2) 求

$$\left |\begin{matrix}
    a+b+c & 1 & 1 \\
    a^2+b^2+c^2 & b & c \\
    3abc & ca & ab \\
\end{matrix} \right |$$

方法：列相减法。

练习三 3. (1) 求

$$\begin{bmatrix}
    \lambda & 1 & 0 \\
    0 & \lambda & 1 \\
    0 & 0 & \lambda \\
\end{bmatrix}^n$$

方法：数学归纳法。

## 矩阵
矩阵的秩满足

$$r(A) + r(B) - n \leq r(AB) \leq \min\{r(A), r(B)\}$$

$$r(A + B) \leq r(A) + r(B)$$

乘可逆矩阵，秩不变。

## 向量组
若向量组 $T_1$ 中的每个向量都可被 $T_2$ 线性表示，称 **$T_1$ 可被 $T_2$ 线性表示**。若可互相线性表示，称为**向量组等价**。

也可以认为存在矩阵 $\boldsymbol K$ 使得

$$T_1 = T_2\boldsymbol K$$

## 向量空间
若向量集合 $V$ 中 $\vec a + \vec b$ 和 $k\vec a$ 封闭，则 $V$ 是**向量空间**。

若 $U \subseteq V$ 且 $U, V$ 是向量空间，则 $U$ 是 $V$ 的**子空间**。

$Sp\{\vec a_1, \ldots, \vec a_r\}$ 称为其**张成的向量空间**。

若向量空间 $V$ 可被线性无关的向量组 $\vec a_r \subseteq V$ 表示，则 $\{\vec a_1, \ldots, \vec a_r\}$ 是一组**基**，$r$ 是**维数**，记为 $\text{dim }V$。

特别地，$\text{dim }\{0\} = 0$。

---

对于基 $\{\vec a_1, \ldots, \vec a_r\}$ 和向量 $\vec \beta \in V$，存在 $\vec x$ 使得

$$\vec \beta = \sum_{i = 1}^r x_i\vec a_i$$

则 $\vec x$ 称为 **$\beta$ 关于基 $\{a_1, \ldots, a_r\}$ 的坐标向量**，简称**坐标**。

对于同一向量空间 $V$ 的两个基 $\{\vec a_1, \ldots, \vec a_r\}$ 和 $\{\vec b_1, \ldots, \vec b_r\}$，若存在矩阵 $\boldsymbol C_{r \times r}$ 使得

$$[\vec a_1 \ \ldots \ \vec a_r] = [\vec b_1 \ \ldots \ \vec b_r] \boldsymbol C$$

则 $\boldsymbol C$ 为**从基 $\{\vec a_1, \ldots, \vec a_r\}$ 到 $\{\vec b_1, \ldots, \vec b_r\}$ 的过渡矩阵（基变换矩阵）**。

对向量 $\vec a \in V$，设在两组基下的坐标分别为 $\vec x, \vec x$，则

$$\vec x = \boldsymbol C\vec y$$

注意这里的乘积顺序。
 
## 线性方程组
由 $AX = 0$，不妨设 $A$ 的列向量极大线性无关组为 $\{A_1, \ldots, A_r\}$，并设 $A_{r + i} = \sum_{j = 1}^r k_{i, j} A_j$，那么可见

$$a_1 = \begin{bmatrix}
    -k_{11} \\ -k_{12} \\ \vdots \\ -k_{1r} \\ 1 \\ 0 \\ \vdots \\ 0
\end{bmatrix}$$

是一个解。