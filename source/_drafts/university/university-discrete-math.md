---
title: 《离散数学》笔记
date: 2023-05-22
category: 
- 学习
tags:
- 文化课
excerpt: 三天速通离散！
---

## 关系.pptx
* **笛卡尔积**：$A \times B = \{(a, b) \mid a \in A, b \in B\}$
* **恒等关系 / 对角关系**：$\Delta A = I_A = A^2$ 称为恒等关系 / 对角关系
* **关系数据库**：$n$ 元关系可以画成一张表，每一行是关系的一个元素，称为关系数据库
* **整除关系**：$xRy \Leftrightarrow x \mid y$
  * **倍数关系**：整除关系的逆
* **复合**：$R \circ S = \{(a, c) \mid \exists b, aRb \land bRc\}$
* **等价关系**：自反、对称且传递的关系
* **反自反**：$(a, a) \not \in R$
* **反对称**：$a\neq b, (a, b) \in R \Leftrightarrow (b, a) \not \in R$
* **闭包**：$R$ 的闭包就是 $R$ 的超集
  * 自反、对称、传递闭包记为 $r(R), s(R), t(R)$
  * 自反 $r(R) = R \cup I$，对称 $s(R) = R \cup R^{-1}$，传递 $t(R) = \bigcup_{k = 1}^{r_R} R^k$
* **沃舍尔算法**：Floyd 算法
* **等价类**：对**等价关系** $R$，元素 $a \in A$，与 $a$ 等价的所有元素称为 $a$ 的等价类，记为 $[a]_R$
  * 模 $m$ 同余类记为 $[a]_m$，如 $[3]_7$
  * **代表**：若 $b \in [a]_R$，称 $b$ 为 $[a]_R$ 的一个代表
* **划分**：$S$ 的划分为一组子集 $\{A_i\}$，满足非空、不重、不漏
* **商映射**：对**等价关系** $R$，定义 $f_R(x) = [x]_R : A \rightarrow 2^A$ 称为商映射
* **偏序**：同时满足自反、反对称、传递的关系叫偏序关系
  * **全序 / 线性序**：所有元素均可比较
  * **良序**：所有非空子集都有最小元
  * **哈塞图**：下小上大
  * **最大元 / 极大元**：最大必须能和所有比较，极大不需要
  * **上下确界**：对偏序集 $(S, \leq)$ 和 $A \subseteq S$，若 $\exists m \in S, \forall x \in S, x \leq m$，称 $m$ 是 $S$ 的上确界，记作 $\text{lub}(A)$；下确界记作 $\text{glb}(A)$。
  * **格**：若偏序集 $(S, \leq)$ 满足对于任意 $\{x, y\}$ 均有上下确界，称该偏序集为格
    * 例：$(N, \mid)$ 是格，因为 $\text{lub}(\{x, y\}) = \text{lcm}(x, y), \text{glb}(\{x, y\}) = \gcd(x, y)$

## 集合论.pptx
* 函数的记号
  * $f(A) = \{f(x) \mid x \in A\}$
  * $B^A = \{f \mid f : A \rightarrow B\}$ 是所有 $A$ 到 $B$ 的函数的集合
* **单射 / 内射**：$a \neq b \Rightarrow f(a) \neq f(b)$
* **满射**：$\forall b, \exists a, f(a) = b$
* **双射**：满足上面两个性质
* **复合**：$(g \circ f)(x) = g(f(x))$
* **反函数的分支**：当 $f : A \rightarrow B$ 不可逆时，可缩小 $B$ 使之满射，再划分 $A$ 成为多个子集 $A_1', A_2', \ldots, A_k'$，得到若干反函数 $f^{-1}_i : B' \rightarrow A_i'$，每个称为反函数的一个分支
  * 例：$f : \R \rightarrow \R, f(x) = x^2$ 不是满射，改为 $f : \R \rightarrow [0, +\infty)$，再将 $\R$ 划分为 $(-\infty, 0]$ 和 $(0, +\infty)$ 得到反函数 $f_1^{-1}(y) = \sqrt y$ 和 $f_2^{-1}(y) = -\sqrt y$
* **作为记号的 $f^{-1}$**：当 $f$ 不可逆时，可以定义 $f^{-1}(\{y\}) = \{x \mid f(x) = y\}$，此时 $f^{-1}$ 不代表反函数
* **同基**：记为 $|A| = |B|$ 或 $A \sim B$
  * 奇怪的例题：证明 $(0, 1) \sim (0, 1]$：令 $A = \{0.5, 0.25, \ldots\}$，定义函数 $f$ 使得 $\forall x \in A, f(x) = 2x$，$\forall x \not \in A, f(x) = x$，可见 $f : (0, 1) \sim (0, 1]$ 是双射

## 图论.pptx
* 基本术语
  * **邻域**：$N(u)$ 为与 $u$ 相邻的点的集合，$N(S) = \bigcup_{u \in S} N(u)$
  * **度**：出 / 入度记为 $\text{deg}^+(u), \text{deg}^-(u)$；度为 $0$ 称为**孤立点**，度为 $1$ 称为**悬挂点**
  * 完全图 $K_n$，圈图 $C_n$，轮图 $W_n$，$n$ 立方图 $Q_n$，二分图 $K_{n, n}$
  * **关联矩阵**：$M_G = (m_{i, j})_{n \times m}$ 定义为当且仅当 $v_i, e_j$ 关联 $m_{i, j} = 1$，否则为 $0$
  * **简单路**：边不重复
  * **真路**：点不重复
  * **连通分支 / 子图**：极大连通子图
* **霍尔定理**：二分图有完全匹配，当且仅当 $\forall S \subseteq V, |N(S)| \geq |S|$
* 有向图连通性
  * **弱连通**：对应无向图连通
  * **强连通**：对任意 $u, v$ 都存在 $u$ 到 $v$ 的路径
  * **单向连通**：对任意 $u, v$，$u$ 到 $v$ 的路径或 $v$ 到 $u$ 的路径中至少存在一条
* **连通度**：点割集最小大小称为点连通度，记为 $\kappa(G)$；边记为 $\lambda(G)$
  * $\kappa(G) \leq \lambda(G)$
  * 特别地，$\kappa(K_n) = n - 1$
  * 若 $\kappa(G) \geq k$，称 $G$ 是 **$k$ 点连通的**。
* **欧拉路径算法**

```python
def dfs(u):
  for v in range(n):
    if e[u][v]:
      e[u][v] = e[v][u] = 0
      dfs(v)
      print(u, v)
```

* 哈密顿图的必要条件
  * $\forall S \subseteq V$，有 $w(G - S) \leq |S|$，$w(G - S)$ 表示连通分支数量
* 哈密顿图的充分条件
  * 若 $G$ 是简单图，$n \geq 3$，且对于任意 $u, v$ 有 $\text{deg}(u) + \text{deg}(v) \geq n$，则 $G$ 是哈密顿图
* 平面图
  * **欧拉公式**：简单连通平面图 $v - e + f = 2$
  * 推论 1：简单连通平面图若 $v \geq 3$，则 $e \leq 3v - 6$
  * 推论 2：简单连通平面图无长 $3$ 的环，$v \geq 3$，则 $e \leq 2v - 4$
  * 推论证明：每个面至少由 $3$ 条边围成，每条边恰好参与围两个面或参与围一个面两次，因此 $3f \leq 2e$，代入可得
  * 例题：证明若 $G$ 简单连通平面图，则 $\min \text{deg}(u) \leq 5$
  * **同胚**：对 $G_1$ 进行操作，操作为在一条边上新增一个点变为两条边，或删掉度为 $2$ 的点并合并为一条边，若能变为与 $G_2$ 同构，则称两图同胚
  * **库拉托斯基定理**：图是平面图当且仅当有子图与 $K_{3, 3}$ 或 $K_5$ **同胚**。
  * 图着色：最少着色数记为 $\chi(G)$
* 树
  * **外点**：即树叶
  * **$m$ 元树**：每个点儿子数 $\leq m$
    * 二叉树和二元树的区别：二叉树区分左右
    * **满 $m$ 元树**：每个点儿子数 $= m$ 
  * **平衡树**：树叶都在 $h$ 或 $h - 1$ 层
  * **中序遍历**：先遍历第一棵子树，再遍历根，最后遍历其他子树