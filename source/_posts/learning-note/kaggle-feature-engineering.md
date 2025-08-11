---
title: Kaggle Feature Engineering 学习笔记
date: 2023-06-29
category: 
- 学习
tag:
- Kaggle
---

Kaggle 的 [Feature Engineering](https://www.kaggle.com/learn/feature-engineering) 的笔记。

<!-- more -->

## 2. Mutual Information
Mutual Information（互信息）是概率论里的一个名词，它可被用来衡量两个随机变量之间相关程度的大小；换句话说，就是衡量某个随机变量能减少另一个随机变量的不确定性多少。

互信息的最小值为 $0$，代表两个变量互相独立。互信息没有最大值，但一般很少超过 $2$。

互信息可以用来衡量特征和目标之间的关联性，但不能检测不同特征间的关联性。

要计算 MI，需要把字符串变量转化为整数变量。

```python
X = df.copy()
y = X.pop("price")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int

mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
```

为了方便，可以用图表显示：

```python
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
```

查看 `curb_weight` 和 `price` 的关联图：

```python
import seaborn as sns
sns.relplot(x="curb_weight", y="price", data=df);
```

查看 `horsepower` 和 `price` 的关联图，同时根据 `fuel_type` 的分类来对点染色：

```python
sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);
sns.lmplot(x="horsepower", y="price", hue="fuel_type", col="fuel_type", data=df); # 根据 fuel_type 划分成多个图
```

使用这样的方式，就可以检测特征之间的相关性。

## 3. Creating Features
关于创建新特征的小提示：

* 尝试充分理解原有的特征。
* 了解相关领域的研究。
* 参考前人工作。
* 善用数据可视化。

对每个记录，计算并保存 `Neighborhood` 分类对应的 `GrLivArea` 中位数：

```python
X_5["MedNhbdArea"] = X.groupby('Neighborhood').GrLivArea.transform('median')
```

这里和直接使用 `.median()` 不同的是，`.median()` 会返回一个以 `Neighborhood` 为索引的统计信息，而 `.transform()` 则返回与原记录数长度相同的列表，更方便合并。

## 4. Clustering With K-Means
将聚类添加到特征中，可以“分而治之”地解决不同区域的拟合。

![](https://storage.googleapis.com/kaggle-media/learn/images/rraXFed.png)

**K-均值聚类**（K-means Clustering）的思想是，设置 $K$ 个中心点，让每个数据归属于离其最近的中心点；同时让距离尽可能近。

算法是：每次迭代时，让中心点移动，使得归属其的点离它距离最小，全部移动完后重新计算归属。

scikit-learn 提供的 K-均值聚类有三个参数：`n_clusters`（中心点个数）、`max_iter`（单次运行迭代次数）和 `n_init`（需要运行的种子个数）。

```python
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")
```

**注意**：K-均值聚类对取值范围很敏感，因此在聚类前需要考虑是否需要进行放缩。

也可以使用 `kmeans.fit_transform(X)` 获取点到各个中心点的距离。

## 5. Principal Component Analysis
**主成分分析**（Principal Component Analysis，PCA）是另一种用来划分数据的方式。K-均值聚类是根据近似性划分的，而 PCA 是根据差异来划分的。

若提供 $k$ 个特征，则 PCA 会在对应的 $k$ 维空间中进行正交变换，即转化为 $k$ 个线性无关变量的值，这些变量称为**主成分**（Principal Component），转化系数称为 **Loadings**。在此基础上，让每维的方差尽可能大。

主成分可以：

* 用作特征
* 用于降维（删去一些方差接近于 $0$ 的主成分）
* 用于异常检测、减少噪音和去相关性（即，合并一些高度相关的特征）

此外，需要注意：

* PCA 只对有数值意义的特征（numberic features）起效
* PCA 对缩放很敏感，一般都要先标准化再 PCA
* 视情况需要去除部分异常值

```python
from sklearn.decomposition import PCA

# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
loadings

# Look at explained variance
plot_variance(pca);

print(df[features].corrwith(df.SalePrice))


# You can change PC1 to PC2, PC3, or PC4
component = "PC1"

idx = X_pca[component].sort_values(ascending=False).index
df.loc[idx, ["SalePrice", "Neighborhood", "SaleCondition"] + features]
```

## 6. Target Encoding
**目标编码**（Target Encoding）是一种将分类转化为数值的编码。

一种简单的方法是，直接用分类下目标的某种统计数值（如同类的平均价格）来替代这个分类。

```python
autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")
```

如果某类别数据不够多，甚至测试集出现了训练集没有的分类，都会出现问题。一种解决方式是**平滑化**（Smoothing），思路是将同类统计值和全体统计值做一个混合，即

$$\text{encoding} = w \times \text{category} + (1 - w) \times \text{overall}$$

$w \in [0, 1]$ 是与类别出现频率相关的一个实数。一般地，可以使用 **m-估计**（m-estimate）计算 $w$：

$$w = \frac n{n + m}$$

其中 $m$ 是一个常数，称为**平滑因子**（Smoothing factor）。

![](https://storage.googleapis.com/kaggle-media/learn/images/1uVtQEz.png)

目标编码可以解决：

* 分类种数很多的特征：One-hot 编码会导致产生非常多的特征，而目标编码可以解决这一问题。
* MI 较低的特征。

**注意**：为了避免 overfitting，我们需要再从训练集中分离一些数据出来，单独作为目标编码来使用。

```python
from category_encoders import MEstimateEncoder

# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)

# Fit the encoder on the encoding split.
encoder.fit(X_encode, y_encode)

# Encode the Zipcode column to create the final training data
X_train = encoder.transform(X_pretrain)
```