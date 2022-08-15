---
title: 中学回忆录
date: 2019-02-14
updated: 2020-02-20
category:
- 生活
- 随想
---

喝了点无酒精的酒，突然想写点总结了。

那么就直接开始吧。

## 初一
### OI
我是 LSOI（汕头市龙湖实验中学的 OI）的第一届 OIer。

我们的教练是钟老师——一位负责了三个班级数学的数学老师。她先前并没有 OI 教学经验，不过还是根据金中姚老给的建议买了几本《信息学奥赛一本通》，教了我们基础语句。

然而这不过是兴趣班，每周只上一节；教了一个学期也才把基础语句教完。我当时根本没把 OI 放在心上，也以为 OI 的知识不过就《一本通》里面的内容，所以总是摸鱼——虽然完全能够应付乌龟似的教学进度。

下学期去了 STOI（汕头市选）普及组，kpm 出题。kpm 出了四道题，记忆清晰的是 T1 是道迷惑的送分题：

> 给出 $10$ 组正整数 $a_i, b_i$，要求输出 $a_i + b_i$ 的值。  
> 正整数之和**不大于** $2^{2^{2 \times 2^2}}$。

在场所有选手都敲了高精度，只有我一个分析发现 $a_i + b_i \leq 2^{64}$，然后写了 `uint64` 以及特判 $a_i + b_i = 2^{64}$ 的情况。最后出成绩，毒瘤 kpm 居然给这道题的所有数据加了无数个前导零，最后我和 hjw 成了唯二通过这道题的人（hjw 的高精度居然可以处理前导零……），我靠着这 $100$ 分作为第八名压线进了普及组市队，参加 GDOI2017。

理所当然地，GDOI2017 我考出了 $0$ / $800$ 的好成绩。不过我从此认识了金实众，他们虽然并不常教我知识，但自己也认识到了 OI 的广度和深度远超我的想象；在他们的带领下，我进入洛谷开始刷题。

这里需要提一个人：zn，我的一位不知名的学长。他的 OI 实力并不很强，且运气还破天荒地差，没有拿过一次一等，后来竟连金中也没有考上；但他仍是很努力的，因而即使在一中，也总能取得很前的成绩。他是个很善良的人，在 OI 和人际交往等方面帮了我很多很多，我非常感激他对我的帮助。

那个时候是 2017 年 5 月。然后我不断刷题，也通过洛谷的试炼场提升了自己，逐渐变得强大起来。其实这段时间刷了很多橙题以及一些黄题，每道题基本都没什么差别；不过这确实使我后面发挥相对稳定了一些。

现在想来，如果不是我踩线进了市队，不仅是我，连带着我下面所有的 LSOIer 全部都要荒废掉，成为普及三等的选手。

### 文化课
这时候我基本多数精力都放在文化课上，平时成绩大概能排在级五十名左右。自己也没有多加用心，所以并没有什么进步。

## 初二
### OI
NOIP2017 普及组拿了普及一等，勉勉强强。

在那之后就开始尝试蓝题了，结果发现蓝题也并没有那么困难。然后就慢慢学了挺多基础的算法，堆、线段树、ST 表、状压 DP、分块分治、字符串 Hash、最大流、矩阵加速、树形 DP、二分图最大匹配、主席树、平衡树、左偏树、树链剖分……

这个时段也稍微了解到了初一的同学。我很担心他们浪费掉初一的时间，所以不时会给他们讲一些东西，给他们一点压力。

然后就是 GDOI2018，水了个一般般的成绩。由于比赛经验不够，成绩不理想，水了个初中组的二等而已。

之后又学了 cdq 分治、树套树、斜率优化、线段树合并、LCT、单调队列之类的东西。

到了暑假的时候，整了个 LSSC（Summer Camp 夏令营），把初一的学弟过来塞了一堆算法知识给他们讲。整了一套难度有点高的普及+比赛，打算让他们失掉信心然后决心认真学 OI。这期间还给迷途之家供了一道非常恶心的模拟题，让你模拟 Linux 下的 rm/cd/cat/mv/mkdir 操作，标程挂了十几次，导致自己以后再也不敢出大模拟。

### 文化课
初二没有完全搁置文化课，因而文化课的成绩仍然马马虎虎地保持在年级五十左右的水平。初二下学期的期末考考了年级十三名，和初一开学考是一样的名次，——也是我整个初中考得最好的两次考试。有始有终，大概还是不错的。

### slay.one
顺便一提，这个时候的我玩上了 slay.one。LSSC 后的 JZSC（金中夏令营）上，czllgzmzl（一本清华爷）回来给我们讲课，中午也和我们一起玩 slay.one。印象里非常深刻的是他玩小精灵异常强大，激光手雷异常恶心，并且玩游戏居然还能做到面无神色（~~像有人欠了他几十亿没还~~）；结果一上去讲课的时候神飞色舞，还会幽默地和我们互动。

我也在 slay.one 上结识了很多好友，他们虽然大多 OI 水平不强（甚至有的像 hex 根本不是 OIer），但也都成为了我的 QQ 好友，也成为了清列时一直没有清掉的怀念和回忆。

## 初三
### OI
大概这个时候接触到了 LibreOJ，开始了在 LibreOJ 的艰难的刷题。除了历年 NOIP 的题目基本刷过一遍外，也开始写 100 开头的模板题和 2000 开头的省选题。

NOIP2018 提高组考挂了，虽然也还是水了个一等。

接下来由于成天刷 LibreOJ，所以学了一堆算法：SA，线性基，莫反，杜教筛，高斯消元，矩阵树定理，Lucas 定理，FFT，NTT，点分治，exLucas，范德蒙德卷积，区间众数，李超线段树，01 分数规划，min_max 容斥，FWT，多项式求逆、多项式开根、多项式 ln，拉格朗日插值，线段树分裂，洲阁筛，虚树，线段树分治，图上随机游走，Polya 定理，BSGS，决策单调性，之类的。

GDSOI2019 由于初中组 NOIP 分数限制奇高，所以没得去。

我自己在初二初三的时候，应该可以说是努力的。是，我没有 ntf、cmd 他们那样强，但我觉得我初二初三作为一名没有学长教练指导的，基本自学的选手，能够做到那样自己已经满意了。这段时间把大多数的基础算法全学完了，LibreOJ 也刷了三四百道题。

*你不是刚刚才[在日记里说](https://www.luogu.com.cn/blog/tsukimaru/otter-diary-san)感觉自己像阿 Q 一样成天和别人说“我曾经初二的时候，也是像上官老爷那样多么阔气”吗？怎么又开始扯你初中的事情了？*

我并不打算又旧事重提，毕竟过去的已经过去了；不过是为了交代自己的背景罢了。STOI（汕头 OI）本就不强，初中生能像我这样下定决心天天刷题的选手几乎没有，大多都是在老师、家长的软硬兼施下初三退役了。毕竟大家都觉得“既然都拿到普及一等了，那肯定高一再认真学也来得及”嘛，他们虽然并不是 OIer，但说的话肯定都是很可信的。

### 背后的故事
这一年我过得很艰难。

我的家长、老师都说，我拿了普及一等、GDOI 初中组二等、提高一等，已经达到金中自招的要求三次了（金中自招要求普及一等、GDOI 初中组二等、提高二等至少取得其中的一个），没必要再学 OI。现在好好准备中考，等上了高中再恢复 OI 学习也来得及；金中自然会有办法的。

但我没有听从他们的话。

自从我刷了 LibreOJ 之后，刷的题目越多，越发感觉自己的无能。那一段时间我刷省选题几乎道道不会做（除了一些非常模板的题目），经常去学新的算法或是看题解；越看，越学，就越觉得自己无能。我总觉得自己还是非常非常弱，也很害怕高中的环境还是没什么太大的变化；所以就很勤奋地刷题，刷题。

初三，「时间还很多，上了高中再学吧，你们以前也没有这个先例」。

高一，「我们又不像广州深圳那边，实力比不过他们也是正常的」。

高二，「努力过就好了，好好调整心态准备文化课吧」。

这可不相当正确么？

可**人之所以为人，是因为有猿猴直立起来，成为了别人不愿做的“先例”，才成为了人**。那些劝解我放弃 OI 的人是善良的，可他们并不比我自己更了解我；自己总该知道对自己而言什么是好的，什么是不好的，然后抱着必死的心态走下去，完成给予的任务后活着回来。

当然，STOI 里是没几个人如此的，我甚至举不出一两个例子；我也相信，初中认真准备中考的占了绝大多数。只是我自知自己的懦弱和慵懒，不愿把希望延后到未来罢了；如果人人都能像 hjw，ccz 他们那么勤奋努力，或许早就人人清北了。

于是整个初三，我基本没有写过作业（除了相当严厉的语文老师布置的作业），也没有怎么复习；政治提纲也只背了一部分。成绩自然一落千丈；掉了，稍微学习一段时间又升了；升了，又掉了。级排名从 15 掉到 151，然后是 337、331、225、135、225，成绩起起落落，最后中考考挂了，比金中线低了一二十分。万幸还是因自招被录取了。

上课大抵还是有听一点的，尤其是语文课；唯独数学课基本用来补觉。平日都会带两个书包，一个大书包装课本，另一个小书包装笔记本电脑。自己常常趁课间的十分钟到讲台电脑上看看题目和题解，课上就在草稿纸上演练，午休和放学到机房写代码，总之是比较勤奋的一段时间。

翁老师表面反对，实际上仍然默认同意我跑去机房写题。

家长一开始也不太赞同，经过我几次固执的争论之后，也渐渐不反对了。

现在想来，我的初三是对「梦想」的最好的诠释。之后也再没有像那时的自己一样，将无限的动力和热爱毫无保留地投入到「梦想」里去了。

### 初三（14）班
在这里我想趁着记忆淡去前，记录一下我在初三（14）班的老师们。

首先是老翁。翁金辉老师是我们初三（14）班——全校最强重点班的班主任，兼任我们班的数学老师。他幽默风趣但不失严肃冷静，是个不疯但也能和学生一起玩得很好的老师。在他的管理下班风一直很良好，我们的同学即勤奋刻苦又不木讷呆板——清晨和大课间能听到同学们相互扯皮的欢声笑语，晚上六点后也能看到整栋教学楼只剩我们班灯火通明。

对于我，他一向是关爱的；初一初二时把我视为班级的希望，初三时也很关注我。我初三搁置了文化课，他为了维护班级风气，明面上时不时“讥讽”“批斗”我，暗地里又默认同意我天天跑去机房上机、数学课睡觉、上课写题。若是他真的反对我，大可以不让我去上机；事实证明，他还是支持我弄 OI 的，不过是明面上没有说出来罢了。这一年，我的 OI 事业没有受到他的阻挠，自己的心境也变得坚强了很多很多；班级风气没有变差，反倒是他在几个同学心里背负上了“恶人”的称号（玩笑话）。

我对他也一向很敬重，又有些愧疚，觉得他花了很多时间在我身上，我却没能在中考发挥良好。至今仍能记得他的很多很多名言，记得他和我发生过的一些冲突；记得傍晚六点半灯火通明的初三（14）班教室，记得他班会时给我们放的各种心灵鸡汤。他是我成长路上最感激的一名老师，如果不是他，想必我也不会有今天的成就。

> 「lmh，你这样弄下去相当于精神残废啊！孤注一掷怎么行？」  
> 「你这样子，以后的路会越走越窄！」  
> 「wyl 怎么又在扯皮了？试卷写完了吗？」  
> 「今天的每日一题可能有点难啊，要是做不出来明天再讲。」  
> 「你干嘛！你干嘛！不要对答案啊！」  
> 「这个 lmh，我现在说他草包他肯定听不见，你看他现在还在睡觉。」  
> 「哎哟，萝卜这次考得不错啊！」  
> 「有的黑暗，只能你一个人自己度过。」  

毕业班会上我擦着眼泪递给他一张明信片，他笑着签了名，还写了那句他曾无数次说过我的话；只不过这次不是“讥讽”，而是祈愿——

**「愿路越走越宽」。**

![IMG_20210417_122109.jpg](https://i.loli.net/2021/04/18/TQiWSUzNj1M3DfL.jpg)

---

然后是阿廖，我们的物理老师，广西人。他的个子不过一米六五，终年戴着薄薄的方框眼镜，身着墨绿色的 T 衬衫；眉头微皱，眼睛微眯，浑身萦绕着农村而来的乡土气息。

阿廖的性格相当幽默风趣。阿廖说话的时候，总习惯用手捂着嘴巴，眯着眼睛，说出一段带着口音的、抑扬顿挫的普通话；感情激烈的时候还会说出夹杂着土味浓厚的粗话。他的有趣和老翁是不同的，老翁的有趣是是活泼书生般的幽默风趣；而阿廖呢，他更像是你在任何一个乡村都能见到的那种土生土长的年轻人——俗气但不下流，全身上下散发着这方水土所养育的自然的活力。他身高不高，但很喜欢和体育老师们打篮球；这也成了我们的饭后谈资之一。

他常常给我们讲他在农村的所见所闻：他夏天骑着水牛到水库里游泳，邻居的小孩子爬上屋顶碰电线触电一天不能动弹，小学时自己鼓捣收音机把收音机拆坏又修好……他有着各种各样稀奇古怪的故事，而这些是我们在高楼林立的城市里所不知道的。因为他，我很喜欢听物理课（虽然也还是有时会睡觉）。

毕业之后某次重聚的时候我和 cjr 一起去办公室找他，他依旧是戴着薄薄的方框眼镜，矮矮的个子下透露出活力与朴实的气息。阿廖见到我们，用力拍了拍我们的肩膀，说道：「考不上清华不要回来！」

---

之后是“微微”——我们的语文老师，赵微老师。她是所有老师里最严厉的，我一向有些怕她；我被她训斥的场景如同昨日事一样记忆清晰。她很年轻，长得也很好看，因而大家总称她“微微”老师。记得临近中考的时候，她加班加点给我们赶了很多很多的名著资料，全部都是她一手整理的；有时在家里工作到凌晨一点才睡下，白天又抖擞地带着她治学严谨的精神来上课。那年中考语文的名著题恰好被她压中，一开始还觉得运气挺好，后来想了想，大抵是她完全一点不漏整理的名著资料的缘故。

**我非常敬佩她作为老师的，令人敬畏甚至可怖的称职和勤奋，她教书育人、鞠躬尽瘁的崇高精神，是我一生望尘莫及的。**

---

我对其他的老师（老周，思佳老师，程老师，……）也有些许的记忆，可或许是限于篇幅，又或是记忆淡去了，此处不再展开记叙。

在这初中三年里，无论发生了什么事情，我都从来没有掉过一滴泪；难过的时候也是有的，可无论如何都哭不出来。唯独毕业班会那天，当我举着手机录着阿廖和老翁的临别赠言的时候，泪水突然就控制不住地流了下来，仿佛三年来的所有欢乐、苦涩、遗憾，一瞬间如水库决堤般奔涌而出。甚至于今日再回顾毕业班会的时候，视线仍会不由自主地模糊起来，仿佛自己又回到了坐在教室里看着讲台上老翁拿着根铁棒指着大屏幕给我们讲每日一题的时光。

上了高中忙于竞赛，没有多少时间呆在班里了；班级也常常换，因此对高中的班级并没有什么太大的感情。

**在初三（14）班生活过的回忆，是我一生永远的珍藏。**

> 真的，一直到现在，我实在再没有吃到那夜似的好豆，——也不再看到那夜似的好戏了。

![IMG_20190629_111347.jpg](https://i.loli.net/2021/04/26/Se74DOf3xEdybpo.jpg)

---

> 本来啊，我觉得说，跟你们开个班会课，上上下下至少都有八十节。平时也从来不打稿，但是到这个时候呢，还是比较特殊，我怕我待会语无伦次啊。跟其他老师的那种感情不一样，你们是我这样一点点带大的，就像我六十几个孩子一样，那种感情呢，不一样。所以呢，我还是自己拿个稿子来说一下（从裤袋里掏出演讲稿），要不然待会说不出话（笑）（全场笑）。
>
> 这个……不经意间，大家呢，在龙实的日子，已经慢慢到了要说再见的时候了，是吧。所以呢，老师希望呢，这初中三年的时光，能够给你们留下美好的回忆。然后呢，你们呐，对我来说，很重要。嗯……这种感觉是什么感觉呢，是高兴，但是又很……很焦虑，焦虑。
>
> 呃……（较长时间的停顿）呃……我那种感觉是，很想和你们一起毕业，然后……再教你们！是吧，但是事实上，不可能。咱们赵微老师讲了，我们，原地返回，是吧。所以……因为你们的关系，我也希望说明年，我还是回到初一去，是吧，把这种感情，转接到新的小孩身上。当然你不要说老师啊那那不会就忘记我们，是吧？诶，是吧。呃……应该来讲，来到这里，到现在有十三个年头了，但是呢，你们是我最真情付出的一届，是吧。（语气严肃）但只能发在我们班里面啊，啊不能给其他以前的那些同学听到了，是吧，这是私爱，私爱。（全场笑）
>
> 然后呢……感觉，你看我我一下，一下子就语无伦次，说不上来了，哈。往时我我从来不用打稿，我今天呢，比较激动。那，感觉现在纵有千言万语，也不知道从何说起。那么，希望呢，你们能够过得快乐，过得开心，啊。这应该来讲，是作为班主任我最期望的，到这个时候，什么成绩都不重要了。关键，你看，我每天准时七点十五分到，一进来，唠唠叨叨，是吧。现在我站在这里，让我想起来过往很多很多的回忆。每天进来教室我第一件事，是吧，（走到讲台旁的椅子并拍了拍它）就要……坐这里啊，是吧。（全场笑）（把椅子拖过来坐在上面）然后……我做什么？我干什么？（方达：吃包子！）（全场笑）（方达：剖鸡蛋！）应该是……（笑）（方达：每日一题！）哦每日一题！应该是说：“收作业！”“俊熙不要对答案！”是吧。“舒可，坐好来！”（方达：你干嘛！）（全场笑）这种感情是……无可比拟的，是吧。所以我希望，呃，经常也要说一下李明翰呐，是吧，李明翰每天早上我都要在这说他“李明翰不要睡觉！”（方达：李明翰不要搞电脑！）李明翰，呵呵（笑）……李明翰现在，现在没事啦，现在李明翰，是我们班的一个荣誉啊，是吧。（方达：对！）（全场笑）（方达：你之前不是说是狗屎吗）（全场大笑）
>
> 所以……你们带给我很多美好的回忆啊。那么……（站起）还是要送给大家一些毕业的赠言，啊。班主任忙，总是说说书，啊这个……还是要跟，中考一样！怎么办？（方达：思羽站起来！）对！思羽来！思羽！来这边，来换一个麦克风。是吧，还要干什么？**凝心聚力！继续前行**！是吧？不要输在起跑线上。这个放假……除了……放放松，我觉得说……还是要，吃吃苦的。所以呢……没有随随便便上了高中就能成功，我觉得现在获取了一张高中的只是一张门票，所以暑假呢，还是要早做准备，争取说，三年后啊，高考放榜的时候，（用手指轻轻按了一下鼻子）我们，能够再次听到你们的好消息，是吧。那么，你看，有些同学，很棒啊，王梓玲啊，去了新加坡，刚才问她，她说考过了一个排名第三的学校。（全场惊叹）这是我作为班主任的一个荣誉。还有，你看力方啊，优先被金中录取，啊。我们今年的目标是什么？（全场：五十！）啊，多过五十是吧，所以希望说，希望三号到五号那天，我们……收到这个消息的时候哈，我们能凑够这五十个啊。
>
> 那么……第二点，那……以前总是跟大家讲说，要享受你该享受的，付出你该付出的，是吧。那现在我要倒回来说了啊，既然你已付出你该付出的，那么这个暑假，就应该享受你该享受的，是吧。那么怎么办呢？我觉得……（停顿，看演讲稿）第一个！和父母进行一场，短途或长途的旅行是必要的，是吧。世界那么大，我们总要去看一看吧，哈。所以呢，利用这暑假的时间，好好调整一下，呃……（看演讲稿）读万卷书嘛，不如走万里路，我也是经常出去旅游的，所以我希望说大家，欸能够凑一凑啊，特别是如果有几个家庭的，一起出去，我觉得更开心一点啊。
>
> 最后呢……最关键一个点的啊，**要有情**。什么情啊？（方达：三情。）对，三情，是吧？跟父母的，父母情，是吧，跟老师的师生情，跟同学的这个，同学情，是吧。暑假，我们呢……十四班呢，可能是没完没了了啊，有很多活动。慢一点我要组织大家去唱唱 K 啊，是吧，我刚刚讲了，要有同学情。那，跟父母的情谊在哪里呢？我给大家算过一个事情啊，高中三年，三乘三百六十五，多少天啊？一千零多少？（全场：一千零九十五）一千零九十五，是吧。都是学霸，都算得比我快。一千零九十五，什么意思？这个一零九五对大家的意义非常重大，就像龙应台，书中所提到的，是吧，父母是看着你们的背影，渐行渐远。所以这一千多天呢，应该是你们陪父母，时间最多的时间了，是吧，过后读大学，放假，你们要去……学习啊，慢一点要考研啊，再后来成家，有小孩，可能也就渐行渐远，是吧，把爱慢慢地分出去。所以这一百零多天，要好好跟父母相处，然后呢，放假多给父母分分担，分担一些家务，我觉得，这是一件非常快乐的事。所以这个叫，父母情。师生情呢？师生情很简单，是吧。我是不是，像呆在这个巢里的鸟一样，是吧，总是等待着你们回来。所以呢……没事，多回来坐坐，多回来看看，好。
>
> （看演讲稿）感谢大家……三年生涯，好，老师感觉说，希望说你们往后能够过得越来越优秀，越来越开心，哈（用手指轻轻按了一下鼻子）（九十度鞠躬）（全场掌声）。

### zn
还有关于 zn 的事情。

zn 一直都很关心我，自己又没有怎么和别人交流，因而初三的时候自己以为自己喜欢上了 zn，想要和他处对象。当然 zn 也拒绝了，不过一开始我们关系还好，仍然像朋友一样时不时地聊天。

NOIP2018 的时候由于某些原因，我校要有一个同学去和别校的同学凑合住；我和 zn 自然就被安排了。我很开心可以和 zn 住在一起，他看上去也没有很排斥我；我们仍然一起面基玩 CultrisII，甚至连比赛日的早晨早起都要一起打。说实话，自己那时甚至想过晚上去和 zn 一起同床共枕，或者手牵着手睡着也好（当然是经得同意的前提下——虽然不可能同意）；不过当时并没有记起这个，回家之后才想起自己有这个想法。

写了 NOIP2018 游记，在游记里写了一些很矫情的东西。大概是因为自己写的游记吧，也因为自己后来的一些迷惑操作，zn 开始对我反感了，有时会删掉我的 QQ 好友，甚至于说不希望和我住一个房间，否则他就不去 CSP2019。

现在想来，我那时只是因为自己以前没有被多少同龄人关心过的经历，因而才会被他的关心冲昏头脑；其实那不过是一般的朋友间的关心罢了。现在自己也觉得他其实和我并没有多少精神上的共通点，并没有那么适合当我的对象。

即使自己做了很多不好的事情，zn 仍然没有真的对我恶语相向或憎恶我，只是用沉默和拒绝的方式让我自我反省。现在我们和好了，这些也就成了过往云烟。我很感激他的善良，也抱歉自己在他比较艰难的时候还去烦扰他。希望他高考能够顺顺利利吧，——至少能稳中大。

### 兽文化
*等等这个偏题了吧？*

~~但是自己确实非常非常喜欢兽文化，所以希望在这里引流，让更多的 OIer 众的萌二，感受到兽文化的魅力（并不）。~~

自己也是在初三的时候了解到了兽文化。为了避免模糊，下面将欧美的兽文化称 Furry（英语里“毛茸茸”的意思），日本的兽文化称 Kemono（「ケモノ」，日语里“野兽”的意思）。

其实一开始是从宝可梦了解到 Furry 文化的；但后来又了解到 Kemono，发现自己更喜欢 Kemono，尤其是比较可爱的幼兽、兽化程度比较高的正太，以及 TDM（Teitoshin Deformed Mascot，低头身比 Q 版吉祥物，例如日本的大多数吉祥物都可以算作 TDM）。

至于国内的兽圈，说实话，自己其实归属感不是很强，因为对主流的 Furry 文化（尤其是带有强壮 / 成熟特征的 Furry 角色）并不感兴趣；幼兽和 TDM 又比较小众，所以基本上是不混国内兽圈的。

即使如此，自己仍然非常热爱 Kemono/TDM 文化，自己也关注了很多可爱的幼兽画师，每次看到他们的画的可爱的小动物，总会觉得非常开心。

**温馨提示：最好不要在百度上搜索“Furry”“Kemono”等词，否则会冒出不适宜的结果。**

说了这么多，各位如果有兴趣的也可以去看看这些可爱的小动物（尤其是喜欢可爱的宝可梦的）。兽圈信息良莠不齐，如果担心浏览到不合适的信息，当然也可以不混兽圈，关注几个画师或者 Pixiv 上搜搜看看可爱的插图就好了；哪怕不关注 Kemono，看点伊布皮卡丘基拉祈之类的宝可梦也很有益身心健康。下面放上几张个人喜欢的画师的插画，以更形象地形容我所喜欢的 Kemono 类型。

**为避免大量流量消耗，下面的图片为缩略图，如果需要原图请去 Pixiv 源地址寻找。顺便一提，最好不要用他人的原创角色当头像。**

![85361814_p0.png](https://i.loli.net/2021/04/18/8QcghyYL2DjeFRl.png)

↑ pixiv id=85361814，画师ななほし。宝可梦也是 Kemono/Furry 文化的一环。

![87802244_p0.jpg](https://i.loli.net/2021/04/18/DJ8HdLoIgBA263f.jpg)

↑ pixiv id=87802244，画师白鳥ぱんだ。自己非常喜欢吉祥物和可爱的幼兽，即使没有人类的外貌特征也很可爱。

![58657670_p4_master1200.jpg](https://i.loli.net/2021/04/18/15nY9HbiZom8Kfw.jpg)

↑ pixiv id=58657670，画师なかにし。这张就比较贴近 Kemono 文化了（比较典型的兽耳正太），看起来像是一半兽化了的人类，自己也很喜欢这种。

自己非常非常喜欢兽文化，可爱的 Kemono / TDM / 宝可梦丰富了我的精神生活，成为了自己的一个动力；想到可爱的小动物，自己也会觉得开心不少。能活在有它们的世界上是件很幸福的事情。

## 高一
### 文化课和 OI
高一上学期的时候被分去了重点班（据说自主招生招进来的都去了重点班）。文化课学得还算认真，期末考考了年级两百名左右。

准备 NOIP 停课了三四周，考了个很菜的成绩。省里排名一百开外，连省选都去不了。万幸 cxk 学长把他的名额让给了我（觉得自己没有进队的希望），我才得以参加之后的 GDOI。

下学期选科重新分班，我这个成绩自然就流入了普通班。又遇到疫情，于是全部人在家上网课。然而这个网课没有打卡要求，所以上了几天之后，也理所当然地咕了剩下的网课。

### 迷茫与坚定
还是把破壳日记里的那一段搬过来吧：

> 高一下学期，我陷入了无边的迷茫。其一是因为 CSP2019 的严重失利，其二是逐渐失去了兴趣和信心，产生了放弃 OI 的念头。“回去学文化课，轻轻松松度过我的高中，考个中大以上的学校，难道不比整天在这里希望渺茫地硬磨 OI 要好吗？”我那时是这么想的。况且自己确有对音 MAD、BMS、绘画、钢琴的喜爱，而 OI 的热血也业已消失殆尽。姚老师和我的母亲劝说我继续下去，毕竟学了那么久，初三撑了那么久，现在就放弃太可惜了；我也就没有直接放弃——也只是没有直接放弃而已。
>
> 这种迷茫在五月回校之后仍然持续着。我有时坐在体育馆，看着眼前许许多多的人打球，就这么无所事事地看了很久很久。我真的觉得很累了。我真的觉得学 OI 没有意义了。后来考完 GDOI，考出的成绩也不过是可以聊以自慰的程度。
>
> 直到后来的某一天，我翻看博客 Leancloud 评论系统的时候，看到了 iot 的留言。
>
> > 比克提尼，是 OIer。由于所在学校省选名额被大量扣除，可能几个星期后就要告别 OI 了。  
> > 一个人的命运，要靠自我奋斗，但也要考虑到历史的进程。我就是在历史的滚滚车轮下，一颗微小的沙砾。  
> > 月丸是我目前所知道的 OIer 中，唯一一个宝可梦玩家。这是我最后的文字了，你收下吧。  
> > When there is a Pokemon, there is a dream.  
> ——Isaac



## 高二
### 

## 高三
### 文化课

### 高三（13）班