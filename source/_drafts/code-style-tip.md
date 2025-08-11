---
title: ACM 代码风格提示
date: 2022-10-12
category: 
- OI
---

关于代码的一些有用的建议。

<!-- More -->

## 另存为文件或注释原代码，而非删除原代码
当你需要换别的算法时，不要急着把原来的代码删掉，之后可能还会用到（~~甚至你有可能后面写到一半发现之前那个是对的~~）。

## 开新题时注意删掉原来的 `const`
因为上一题 `const int MOD = 1e9 + 7` 没删掉，直接在原来的代码上写，然而这道题的模数是 `998244353`；或者是 `const int n` 不一样。这样的事情已经发生过很多次了……

## 注意读题和看样例
CCPC 预选赛：把题面的周长看成面积，算了半个小时定积分后发现样例过不去。

## 变量声明位置放在函数开头
方便查看这些函数的变量和类型，便于之后修改。

## For 的 Define
```cpp
#define For(i, l, r) for (i = l; i <= r; i++)
#define Fo(i, n) For(i, 1, n)
#define Rof(i, r, l) for (i = r; i >= l; i--)
#define Ro(i, n) Rof(i, n, 1)
```

可以使你的代码更加简洁，**最重要的是可以避免 `for (j = 1; j <= n; i++)`**。

## 树形结构的 Define
```cpp
#define lp (p << 1)
#define rp (p << 1 | 1)
#define mid ((l + r) >> 1)
#define lson lp, l, mid
#define rson rp, mid + 1, r
#define root 1, 1, n
```

那么就可以简化出下面的代码：

```cpp
void add(int p, int l, int r, int a, int b, int x) {
    if (a <= l and r <= b) {
        addnode(p, x);
        return;
    }
    if (a <= mid)
        add(lson, a, b, x); // 注意这里的简写
    if (b > mid)
        add(rson, a, b, x);
    up(p);
}

int main() {
    // do something
    add(root, 3, 5, 2);
}
```

**需要时记得用 `#undef mid` 取消 Define，避免污染命名**。

## BIT，zkw 线段树，线段树
遵循奥卡姆剃刀原则，能简就简。

能写 BIT 不写 zkw，能写 zkw 不写线段树。

可以减少代码时间和常数。

## 更长的变量名
在表达式极其复杂的情况下，考虑用 `cnt0, cnt1, cnt2` 取代 `i, j, k`。

这样的命名比较明显，可以减少出错概率。

——当然相对地打的时间会长一点。在一般情况下可以不用。

## 变量名命名规范
此外，可以思考一些（对自己的）规范，避免出问题，例如：

* 枚举子集（二进制表示）一般用 `s` 和 `t`；
* 枚举普通整型用 `i j k`；
* 变量全小写；结构体（`struct`）命名仅首字母大写；常量全大写。

**这些规范不是定死的，只要你觉得舒服即可**。

```cpp
full = (1 << n) - 1;
Fo(s, full)
    For(i, 0, n - 1)
        if (s >> i & 1)
            // do something
```