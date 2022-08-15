---
title: 牛客
---

# 题面
## A
给出 $n$ 个字符串 $S_i$，$S_i = \text{open}$ 或 $S_i = \text{close}$。

问有多少 $i$ 满足 $S_i = \text{close}$ 且 $S_{i + 1} = \text{open}$。

对于 $100\%$ 的数据，$1 \leq n \leq 1000$。

## B
小 A 有 $n$ 张牌，第 $i$ 张牌的点数为正整数 $A_i$。

小 B 有 $m$ 张牌，第 $i$ 张牌的点数为正整数 $B_i$。

小 A 每次打出一张牌，小 B 可以选择打出一张点数**严格大于**小 A 的牌，或不出牌。如此往复直至小 A 牌全部出完为止。

问小 B 打出牌点数总和的可能的最大值。

对于 $100\%$ 的数据，$1 \leq n, m \leq 10^5$，$1 \leq A_i, B_i \leq 10^9$。

## C
给出一棵 $n$ 个节点的树，点 $i$（$i \geq 2$）的父亲为 $p_i$，到父亲的边的非负整数边权为 $w_i$。

$q$ 个询问，每次给出 $x$，询问是否存在一条路径 $s \rightarrow t$ 使得 $s$ 为 $t$ 的祖先，且路径边权异或和为 $x$。

对于 $100\%$ 的数据，$1 \leq n \leq 5000$，$1 \leq q \leq 10^5$，$0 \leq w_i \leq 10^6$。

## D
令 $\sigma(n)$ 表示 $n$ 的所有因子之和，如 $\sigma(12) = 1 + 2 + 3 + 4 + 6 + 12 = 28$。

给出正整数 $l, r, k$，求

$$\max_{l \leq i \leq r}\{1000i - k\sigma(i)\}$$

对于 $100\%$ 的数据，$1 \leq l \leq r \leq 10^{10}$，$1 \leq k \leq 1000$。

# 题解
## A
略。

## B
可以发现，小 A 的出牌顺序并不影响我们的最优决策，我们只需要提前决定好对小 A 的某张牌我们要出哪张牌（或不出）即可。

贪心思想，让小 A 点数从大到小出牌，若小 B 剩余牌中点数最大的大于小 A 打出的这张牌，那么打出，否则不打出。

不难证明这样最优。

注意使用 `int64` 或 `long long int`。

时间复杂度：$O(n \log n + m \log m)$。

## C
注意到 $(s, t)$ 只有 $O(n^2)$ 对，对 $O(n^2)$ 条合法路径，计算出它们的边权异或和，用计数数组统计即可。

时间复杂度：$O(n^2 + q)$。

## D
令 $f(i) = 1000i - k\sigma(i)$。

感性地想，是不是质数处取到的值会比较大？

可以发现，若 $p$ 为质数，则对于 $2 \leq i < p$，必定有 $f(i) \leq f(p)$。

证明：

首先，对于任意**大于** $1$ 的整数 $n$，我们有 $\sigma(n) \geq n + 1$。又因为 $1 \leq k \leq 1000$，因此

$$f(n) = 1000n - k\sigma(n) \leq 1000n - k(n + 1) = (1000 - k)n - k$$

即 $f(n) \leq (1000 - k)n - k$，取等当且仅当 $n$ 为质数。

则根据题设，有

$$f(i) \leq (1000 - k)i - k \leq (1000 - k)p - k = f(p)$$

得证。

---

接下来我们得到一个好像很暴力的做法：从大到小枚举 $i$，计算 $f(i)$，直到 $i$ 为质数或 $i < l$ 就停止。当然最后还要特判一下 $f(1)$。该做法在实践中表现优秀，可以通过所有数据。

为什么跑得这么快？实际上，我们有一个比较超纲的关于质数分布的定理：不超过 $n$ 的质数约有 $O(\frac n{\log n})$ 个。可以认为，每 $O(\log n)$ 个整数里就大概率有一个质数，因此枚举的 $i$ 的个数也大概是 $O(\log n)$ 的。

可以用质因数分解在 $O(\sqrt n)$ 时间内计算单个 $f(n)$。

时间复杂度：$O(\sqrt n \log n)$。