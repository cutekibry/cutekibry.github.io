---
title: NOIP2018 游记
date: 2018-11-14
updated: 2019-02-23
tags:
- NOIP
category:
- OI
- 游记
excerpt: 或许这就是青春感伤文学吧。
---

## Day 0
#### 在动车上
龙实众坐在第 7 车厢，金实众坐在第 2 车厢。

我的票不知为什么，是在第 2 车厢的。钟老师为了我和龙实众一起住，和我换了座位。

12 点左右的时候，我便去找 zn 玩了。zn 很热情地让我坐在他旁边，说：这旁边有个空座位，一直没有人坐，你可以先坐这里。

然后，他便给我一杯方便面，让我冲了去吃。

我感谢他的好意，便拿去加了热水，吃完了方便面。

那个牌子的方便面我没有吃过，但那一次，我觉得实在非常美味。

而当钟老师把我的车票给我看的时候，我才发觉：原来，我就是那个空座位的主人。

#### 在酒店里游荡
晚上的时候，我们三个初三（莫队在休息）和金实众一起在酒店里散步。

说是散步，但实际上是为了找某个同学。

于是，十几个人气势汹汹地在酒店里扫荡来扫荡去，游荡了各个房间。

到 zc 房间的时候，我们都躲了起来，关掉电灯，然后当 zc 进来的时候，全部人跳出去吓他一跳。zc 虽然有所发觉，但也似乎并没有想到会有这么多人。

到 cxk 房间的时候，cxk 也加入了队伍。

走着走着，突然遇到了钟老师。我被她叫去处理龙实初一初二的同学，开会，便离队了。

我虽然不知道为什么要走来走去，但莫名地觉得很开心。也许，这就是集体活动的乐趣吧。

#### 小插曲
在我开完会之后，钟老师突然问起一个问题——酒店之间的楼层问题。

这个酒店有些特别，只有刷卡才可以乘坐电梯前往对应的楼层——而且只能是宾客居住的那一层。zn 和金实众基本都住在 36 楼，我也跟着他住在 36 楼；但龙实众却住在 39 楼。

于是，她问我：

既然你是龙实的，要不你还是回去和莫队一起住，如何？

...

愣了许久，我说道：

让我想想吧。

沉思许久，我突然想到：防火通道的楼梯是可以走的！

终于，我还是没有离开 zn。

#### 晚上的聚会
处理完同学们的事情之后，我就到了隔壁的房间去参加金实众的聚会了。

玩 Cultris II，slay.one，Malody，...

一直玩了很久，玩得很开心。

玩得累了，我趴在 zn 的身旁，看着 zn 玩 Cultris II。zn 也许误以为我累了，对我说道：你这样趴着，不如去睡觉。

再过一会儿，我就回到了房间。

#### 回到房间
我煮了壶水，想给 zn 倒水。

我还是第一次煮水，随便乱搞一下，水就开始煮了。我害怕该不该切掉开关，于是就拍了张照片，问了问 zn。

zn 说：不知道。

我最后想了一下，还是决定不关掉开关。

最后，水壶如意料之中的停了下来。

... 可惜的是，zn 并不想要我为他倒水。

不过 zn 还是很善意地问我，要不要帮我往我的水壶里面倒水。我点头了。

## Day 1
#### 和 zn 玩 Cultris II
zn 起床了。我看到他起床，也起床了，一看手机，发现才 6 点出头。

既然还有些时间，不如做点什么吧。没什么心情刷题，就和 zn 一起玩 Cultris II 了。

玩到 7 点就下去吃饭了。

书包又忘在隔壁房间了，没有书包，只好抱着电脑走。

点的早餐吃不完，只好拿着豆浆一边走一边喝。

zn 看不下去，就帮我把电脑和手机放到他的书包里去了。

... 感觉 zn 还是很温柔呢。

钟老师开玩笑说，zn 真是挺操心的。

我感觉对不起 zn，本来还想帮忙 zn 做些事情的，结果反而给他带来了不少麻烦...

#### 比赛
8:30 开始比赛。

开始的时候先 5 分钟扫了一眼题目。

T1，NOIP 原题，我做过，但是结论完了。

T2，水题一道，说得好像很玄乎，看懂题面就能秒正解。

T3，题面太长，不看。

先去做 T2 水题，5 分钟敲完，一次过大样例。

做 T1，花了 15 分钟找了个规律，重新推了结论。

25 分钟打完 T1 和 T2，感觉自己像在打普及组。

然后看了一遍 T3，毫无头绪。

算了，敲部分分吧。

暴力不会打，只会找技巧。

$m=1$ ？树上最长链。

$b_i=a_i+1$ ？退化成序列问题，贪心二分。

$a_i=1$ ？所有点和根节点连接，找出尽可能多组 $(u, v)$ 使得 $(w[1][u]+w[1][v] \geq lbound$ ，然后二分 $lbound$ 。

分支不超过 3？二叉树，树形 DP，设 $f[i][j]$ 为 $i$ 子树内链尾权值为 $j$ 时的链的数量， $O(n\max\{v_i\})$ ，然后套个二分。

如果全部写对的话可以过第 $1 \ldots 15$ 测试点，万一写炸就完了。

最高 $100+100+75=275$ ，期望 $100+100+35=235$ 。

#### 看电影
话说回来，就在 Day 0 的晚上，cxk 突然问我们：要不要去看电影？

上映的电影有两部：《毒液》和《名侦探柯南：零的追随者》。对我来说，这两部电影都是挺不错的，于是我说：zn 去我就去。

zn 听完，表示不去。

cxk 说道，如果 zn 不去我就不去，如果我不去 hjw 就不去，zn 的选择是很关键的！

于是 zn 同意了。

回到 Day 1 的下午。

14:55，从六中回来，我就和 zn 一起去看电影了。

cxk 众人坐在第三排，我和 zn 独自坐在第二排。我有些紧张，不过终究是被激动所盖掉了。

前半部分有些枯燥，zn 似乎睡着了。到中后部分的时候，剧情非常有趣、惊险，我感到非常激动、非常刺激。zn 也很认真地看着。

电影放 ED 的时候，我本要走，看到其他人都坐在原位，就又回来等彩蛋了。ED 过后是一些有趣的小日常，再接着是 19 年电影的预告。

预告上，出现了一个熟悉无比的人影——怪盗基德。

电影落幕。

#### 在六中
为了去六中迎接普及组出来的各位选手，我和 zn 告别了。在那之后，钟老师带着我和 hjw 去了六中，等了半个小时，终于看到了各位普及选手。

今年的普及题目非常难，几乎大家都写不出 C 题。洛谷上 kkksc03 说了大概的题意后，我也没什么头绪。

走出学校后，我们决定在附近的餐馆吃饭。

不知道为什么，我突然非常想念 zn，加上感觉自己有点头晕，便没了什么食欲。好不容易写了份冰镇奶茶的菜单，到了前台，等了许久都没有人，于是我又走了回去，把纸团放到口袋里。

我想到 zn 和 lg。虽然我知道他们并没有那个意思，可是我一想到他们正在开心地玩耍，想到他们天天聊天、无话不说，又想到我和 zn 若有若无的朋友关系，就感到很嫉妒。

...

感觉自己精神恍惚。

我趴在桌子上，什么也不做。

我想 zn 了。

我想快点回去。

...

wjd 他们去麦当劳的时候，帮我买了一份汉堡和可乐。想到他们比较辛苦，也感激他们的善良，于是我就走到酒店的大堂，等他们到来。到了之后，我就拿着汉堡和可乐，和他们一起坐电梯上去。

没什么心情。

我走到楼梯间，关上了门。

...

直到 zn 刚好在楼梯间上去的时候，我才和他一起走回房间。

## Day 2
#### 比赛
心态爆炸。

三道题都不会写。

最高 $60+50+44=154$ ，期望 $60+30+44=134$ 。

总分的话，最高 $100+100+75+60+50+44=275+154=429$ ，期望 $100+100+35+60+30+44=235+134=369$ 。

可以说是最糟糕的一次 NOIP 提高组了——D1 过易，D2 过难。

#### 回家
坐上动车，我又坐在 zn 旁边的位置。在检票口，我无意中听见 lxq 说：

> 你和 zn 的票都是同时买的呀。

我看着 zn 玩 Cultris II，他一遍又一遍地玩着 Flying Cheese（最底下 9 层有随机生成的 1x1 小方块，任务是要将所有方块消除），却又一遍一遍地 GAME OVER。

我不禁为他感到可怜。

他是多么好的一个人啊，又勤奋，又温柔，又坚强，又执着。只是命运的不公，使他最终落到了这个境地。

我又想到我刚认识 zn 的时候。

那时候的 zn，在我眼里就是一个神一般的存在。他会很多我不会的算法，诸如 splay，fhq treap，好多好多我都没听过。

我和他开了很过分的玩笑，他便对我说：lmh 你 tm 和我很熟么？

我在空间连续发了很多淘宝的说说，他便对我说：我真 tm 羡慕你的闲。

他是一个会关心我的人。其他人看到这些，无非只是点个赞，或者当作没看到，便这样离开了——虽说也是正常的；但是 zn 看到我颓废的时候，看到我骄傲的时候，看到我做了错事的时候，会骂我，会说我，会关心我。

他是那样温柔，那样善良，就算明知道我喜欢他，也不害怕我，关心着我，为我煮水，和我一起玩，一起聊天——哪怕那只是他对一个普通朋友的招待。

想到这里，眼泪又摇摇欲坠——尽管我从来都是个很坚强又有些冷漠的人。

...

但愿那样努力的他，最终能有一个好的结局。

## 后续
#### 关于 NOIP
D2T1 居然可以暴力枚举删边，这样暴力而简单的做法我们居然都想不到。

此外，在洛谷自测打代码的时候，发现了一些细节问题。最高分应该是没有的了，估计在最高分和期望之间吧。

成绩听说比赛完的后周一（8 天后）就会出来了，等成绩出来。

嘛，虽然没有发挥出实力，但起码还是有一等的。

#### 关于文化课
NOIP 刚回来，没什么精力，总是会觉得很困。

周二的物理课上睡觉，又因为没写作业，被阿廖赶了出去。我竟不觉得难受，只是站在外面补作业。

想了一下，阿廖说的是对的。我的确最近有些鬼迷心窍了。

周四就要考试了。政治 9 张提纲都没背，也没给一两天休息应考，只能随便抽一些背背了。

...

满目萧然，不知所言。