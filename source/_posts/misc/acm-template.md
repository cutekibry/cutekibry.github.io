---
title: ACM 模板
date: 2022-10-22
excerpt: 一些容易忘记的算法。会打的我就不记了。
---

## 字符串
#### KMP
```cpp
j = 0;
For(i, 2, m) {
    while (pat[i] ^ pat[j + 1] and j)
        j = nxt[j];
    if (pat[i] == pat[j + 1]) j++;
    nxt[i] = j;
}
j = 0;
Fo(i, n) {
    while (s[i] ^ pat[j + 1] and j)
        j = nxt[j];
    if (s[i] == pat[j + 1]) j++;
    if (j == m) ans++;
}
```

#### Z 算法
```cpp
// 求 z
Fo(i, n) z[i] = 0;
z[1] = n;
for (i = 2, l = 0, r = 0; i <= n; i++) {
    if (i <= r)
        z[i] = min(z[i - l + 1], r - i + 1);
    while (i + z[i] <= n && pat[i + z[i]] == pat[z[i] + 1])
        z[i]++;
    if (i + z[i] - 1 > r)
        l = i, r = i + z[i] - 1;
}

Fo(i, n) p[i] = 0;
for (int i = 1, l = 0, r = 0; i <= n; i++) {
    if (i <= r)
        lcp[i] = min(z[i - l + 1], r - i + 1);
    while (i + lcp[i] <= n && s[i + lcp[i]] == pat[lcp[i] + 1])
        lcp[i]++;
    if (i + lcp[i] - 1 > r)
        l = i, r = i + lcp[i] - 1;
}
```

#### SAM
```cpp
void extend(char c) {
	int cur, clone, p, q;
	
	cur = ++nodecnt;
	len[cur] = len[last] + 1;
	sum[cur] = 1;
	
	p = last;
	while(~p and !ch[p][c]) {
		ch[p][c] = cur;
		p = link[p];
	}
	if(p == -1)
		last = cur;
	else {
		q = ch[p][c];
		if(len[p] + 1 == len[q]) 
			link[cur] = q;
		else {
			clone = ++nodecnt;
			len[clone] = len[p] + 1;
			cpy(ch[clone], ch[q]);
			link[clone] = link[q];
			while(~p and ch[p][c] == q) {
				ch[p][c] = clone;
				p = link[p];
			}
			link[cur] = link[q] = clone;
		}
	}
	last = cur;
}

link[0] = -1;
Fo(i, n)
    extend(s[i] -'a');
```

#### SA
```cpp
// h[i] = |lcp(suf[sa[i]], suf[sa[i - 1])|
void initsa() {
    static int sb[N], brnk[N << 1], sum[N];
    int i, j, k, m;

    Fo(i, n) sum[s[i]]++;
    Fo(i, 128) sum[i] += sum[i - 1];
    Fo(i, n) sa[sum[s[i]]--] = i;
    m = 0;
    Fo(i, n) rnk[sa[i]] = (s[sa[i]] ^ s[sa[i - 1]]) ? ++m : m;
    for (k = 1; m < n; k <<= 1) {
        clr(sum);
        Fo(i, n) sum[rnk[i]]++;
        Fo(i, m) sum[i] += sum[i - 1];
        Ro(i, n) if (sa[i] - k >= 1) sb[sum[rnk[sa[i] - k]]--] = sa[i] - k;
        For(i, n - k + 1, n) sb[sum[rnk[i]]--] = i;
        m = 0;
        Fo(i, n) brnk[sb[i]] = (rnk[sb[i]] ^ rnk[sb[i - 1]] or 
                                rnk[sb[i] + k] ^ rnk[sb[i - 1] + k]) ? ++m : m;
        cpy(rnk, brnk);
        cpy(sa, sb);
    }

    k = 0;
    Fo(i, n) {
        if (rnk[i] == 1)
            continue;
        j = sa[rnk[i] - 1];
        while (k and s[i + k - 1] ^ s[j + k - 1])
            k--;
        while (s[i + k] == s[j + k])
            k++;
        h[rnk[i]] = k;
    }
}
```

#### PAM
```cpp
int getpre(int p, int n) {
	while(s[n] != s[n - len[p] - 1])
		p = link[p];
	return p;
}
int extend(int i) {
	int p;
	
	p = getpre(last, i);
	if(!ch[p][s[i] - 'a']) {
		len[++nodecnt] = len[p] + 2;
		link[nodecnt] = ch[getpre(link[p], i)][s[i] - 'a'];
		dep[nodecnt] = dep[link[nodecnt]] + 1;
		ch[p][s[i] - 'a'] = nodecnt;
	}
	last = ch[p][s[i] - 'a'];
	return dep[last];
}

len[1] = -1;
link[0] = 1;

其中点 1 代表奇数串，点 0 代表偶数串
状态应当从下往上读再从上往下读，和 Trie 相反
奇数串中最顶边只能读一次
```

## 数学
### 第二类斯特林数
$\begin{Bmatrix} n \\m \end{Bmatrix}$ 表示 $n$ 个不同元素划分为 $m$ 个相同集合的方案数。

$$\begin{Bmatrix} n \\m \end{Bmatrix} = \begin{Bmatrix} n - 1 \\m - 1 \end{Bmatrix} + m\begin{Bmatrix} n - 1 \\m \end{Bmatrix}$$ 

$$\begin{Bmatrix} n \\m \end{Bmatrix}={\frac 1 {m!}}\sum_{k=0}^m (-1)^k C(m,k)(m-k)^n$$

$$n^m=\sum_{k=0}^m \begin{Bmatrix} m \\k \end{Bmatrix} n^{\underline k}$$

#### DFT
```cpp
inline void initwn() {
    const int W = qpow(G, (MOD - 1) / LEN);

    int i;

    wn[LEN >> 1] = 1;
    For(i, (LEN >> 1) + 1, LEN - 1) wn[i] = 1LL * wn[i - 1] * W % MOD;
    Ro(i, (LEN >> 1) - 1) wn[i] = wn[i << 1];
}
inline void initrev(int len) {
    int i, k;

    k = 1;
    while (len >> k >> 1) k++;
    Fo(i, len - 1) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (k - 1));
}

void dft(Expr a, int len) {
    int i, j, k, t;

    Fo(i, len - 1) if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    for (k = 1; k < len; k <<= 1) {
        for (i = 0; i < len; i += k << 1) {
            for (j = 0; j < k; j++) {
                t = 1LL * wn[k + j] * a[i + j + k] % MOD;
                qmod(a[i + j + k] = a[i + j] + MOD - t);
                qmod(a[i + j] += t);
            }
        }
    }
}
inline void idft(Expr a, int len) {
    int i, inv = MOD - MOD / len;
    dft(a, len);
    std::reverse(a + 1, a + len);
    For(i, 0, len - 1) a[i] = 1LL * a[i] * inv % MOD;
}
```

###  全家桶
```cpp
void exprinv(Expr res, Expr a, int lim) {
    static Expr b, a0;
    int n, len, i;

    assert(a[0]);
    b[0] = qpow(a[0], MOD - 2);
    n = 1;
    while (n < lim) {
        len = n << 2;
        copy(a0, a, n << 1);

        initrev(len);

        dft(b, len);
        dft(a0, len);
        for (i = 0; i < len; i++)
            b[i] = b[i] * (2 - 1ll * a0[i] * b[i] % MOD + MOD) % MOD;
        idft(b, len);

        for (i = n << 1; i < len; i++)
            b[i] = 0;

        memset(a0, 0, len << 2);

        n <<= 1;
    }
    memcpy(res, b, lim << 2);
    memset(b, 0, len << 2);
}
void exprsqrt(Expr res, Expr a, int lim) {
    static Expr b, a0, invb;
    int n, len, i;

    b[0] = modsqrt(a[0]);
    n = 1;
    while (n < lim) {
        len = n << 2;
        copy(a0, a, n << 1);
        exprinv(invb, b, n << 1);

        initrev(len);

        dft(b, len);
        dft(invb, len);
        dft(a0, len);
        for (i = 0; i < len; i++)
            b[i] = (b[i] + 1ll * a0[i] * invb[i]) % MOD;
        idft(b, len);

        for (i = 0; i < len; i++)
            b[i] = 1ll * b[i] * INV2 % MOD;
        for (i = n << 1; i < len; i++)
            b[i] = 0;

        n <<= 1;
    }
    memcpy(res, b, lim << 2);
    memset(b, 0, len << 2);
    memset(a0, 0, len << 2);
    memset(invb, 0, len << 2);
}
void exprdet(Expr res, Expr a, int lim) {
    int i;

    for (i = 1; i < lim; i++)
        res[i - 1] = 1ll * a[i] * i % MOD;
    res[lim - 1] = 0;
}
void exprint(Expr res, Expr a, int lim) {
    int i;

    for (i = lim - 1; ~i; i--)
        res[i] = 1ll * a[i - 1] * qpow(i, MOD - 2) % MOD;
    res[0] = 0;
}
void exprln(Expr res, Expr a, int lim) {
    static Expr inva;
    int i, len = lim << 1;

    exprinv(inva, a, lim);
    exprdet(res, a, lim);
    initrev(len);
    dft(res, len);
    dft(inva, len);
    for (i = 0; i < len; i++)
        res[i] = 1ll * res[i] * inva[i] % MOD;
    idft(res, len);
    exprint(res, res, len);
    for (i = lim; i < len; i++)
        res[i] = 0;

    memset(inva, 0, len << 2);
}
void exprexp(Expr res, Expr a, int lim) {
    static Expr b, lnb;
    int n, len, i;

    assert(a[0] == 0);
    b[0] = 1;
    n = 1;
    while (n < lim) {
        len = n << 2;
        exprln(lnb, b, n << 1);
        for(i=0; i<n<<1; i++)
            qmod(lnb[i] = MOD + a[i] - lnb[i]);
        qmod(lnb[0] = lnb[0] + 1);

        initrev(len);

        dft(b, len);
        dft(lnb, len);
        for (i = 0; i < len; i++)
            b[i] = 1ll * b[i] * lnb[i] % MOD;
        idft(b, len);

        for (i = n << 1; i < len; i++)
            b[i] = 0;

        n <<= 1;
    }
    memcpy(res, b, lim << 2);
    memset(b, 0, len << 2);
    memset(lnb, 0, len << 2);
}
```

#### Miller-Rabin
```cpp
bool miller_rabin(int64 p) {
    int64 x;
    int i, t, test;
    int64 b, b2;

    if (p == 1)
        return false;

    x = p - 1;
    t = 0;
    while (~x & 1) {
        x >>= 1;
        t++;
    }

    Fo(test, TIMES) {
        b = qpow(randint() % (p - 1) + 1, x, p);
        Fo(i, t) {
            b2 = qmul(b, b, p);
            if (b2 == 1 and b != 1 and b != p - 1)
                return false;
            b = b2;
        }
        if (b != 1)
            return false;
    }
    return true;
}
```

#### Exact Division
```cpp
uint inv(uint n) {
    uint x = n;
    for (int i = 0; i < 5; i++) x *= 2 - n * x;
    return x;
}

bool isdiv(uint a, uint b) { return a * inv(b) <= uint(-1) / b; }
```

#### 拉格朗日插值
$$f(x) = \sum_{i = 0}^n y_i \prod_{j = 0, j \neq i}^n \frac {x - x_j}{x_i - x_j}$$

$$f(x) = \left(\prod_{i = 0}^n (x - x_i)\right)\left(\sum_{i = 0}^n y_i \frac {w_i}{x - x_i}\right)$$

## 数据结构
#### 虚树
```cpp
void build() {
    static int stk[N];
    int stop;
    int i, x, z;

    stop = 1;
    stk[1] = 1;
    for (i = 1; i <= idn; i++) {
        x = id[i];
        while (stop > 1 and lca(stk[stop - 1], x) ^ stk[stop - 1]) {
            addedge(stk[stop - 1], stk[stop]);
            stop--;
        }

        z = lca(stk[stop], x);
        if (z ^ stk[stop]) {
            addedge(z, stk[stop]);
            stop--;
            if (z ^ stk[stop]) 
                stk[++stop] = z;
        }
        stk[++stop] = x;
    }
    while (stop > 1) {
        addedge(stk[stop - 1], stk[stop]);
        stop--;
    }
}
```