---
title: Panda 学习笔记
date: 2023-02-15
category: 
- 学习
---

Kaggle 的 [Panda 微课](https://www.kaggle.com/learn/pandas)的笔记。

<!-- more -->

## 1. Creating, Reading and Writing
### Panda 基本数据类型
#### DataFrame
DataFrame：表格。

```python
pd.DataFrame({'Bob': [50, 131], 'Sue': ['Ok', 'Next']})
pd.DataFrame({'Bob': [50, 131], 'Sue': ['Ok', 'Next']}, index=['PA', 'PB'])
```
|      | Bob  | Sue  |
| :--- | :--- | :--- |
| 0    | 50   | 131  |
| 1    | Ok   | Next |


|      | Bob  | Sue  |
| :--- | :--- | :--- |
| PA   | 50   | 131  |
| PB   | Ok   | Next |

#### Series
Series：数据序列。Series 没有像上面“Bob”、“Sue”一样的列名，最多只有一个统称（`name`）。

```python
pd.Series([1, 2, 3, 4, 5])
```

```plain
0    1
1    2
2    3
3    4
4    5
dtype: int64
```

```python
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
```

```plain
2015 Sales    30
2016 Sales    35
2017 Sales    40
Name: Product A, dtype: int64
```

### 基本操作
```python
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv") # 读取 csv
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0) # 忽略第一列

wine_reviews.shape # (129971, 14)
wine_reviews.head() # 显示表格前 5 行

animals.to_csv('cows_and_goats.csv') # 写入 csv
```

## 2. Indexing, Selecting & Assigning
### 基于下标的选择
用 `iloc` 来进行基于下标的选择。

注意，`iloc` 和下文的 `loc` 第一个参数都是选择行（也就是选择 
record），第二个参数才选择列（指定想要查询的具体数据类型），与原生 Python 的下表查询不一样。

```python
reviews.iloc[0] # 选择第一行
reviews.iloc[:, 0] # 选择第一列
reviews.iloc[:3, 0] # 选择第一列的前三行
reviews.iloc[[0, 2], 0] # 选择第一列的第一行和第三行
df.iloc[0:1000] # 选择前 1000 列
```

### 基于标签的选择
```python
reviews.loc[0, 'country'] # 选择 country 一列的第一行
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']] # 选择三列
reviews.loc[:, 'Apples':'Potatoes'] # 选择字典序在 [Apples, Potatoes] 之间
df.loc[0:1000] # 选择前 1001 列
```

**注意：`loc` 列参数选择的区间为闭区间，`iloc` 列参数选择的区间为左闭右开区间（与 Python 原生的 `range()` 相同）**。

### 更改索引
```python
reviews.set_index("title") # 将 title 列的内容作为索引（index）
```

### 条件选择
```python
reviews.country == 'Italy' # 返回一个 bool Series，内容为每行的 country 是否恰好为 Italy
reviews.loc[reviews.country == 'Italy'] # 选择 country 为 Italy 的行
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
reviews.loc[reviews.country.isin(['Italy', 'France'])] # 也可以用 notin
reviews.loc[reviews.price.isnull()] # 也可以用 notnull
```

### 赋值
```python
reviews['critic'] = 'everyone' # 将 critic 列所有数据赋值为 everyone
reviews['index_backwards'] = range(len(reviews), 0, -1) 
```

## 3. Summary Functions and Maps
### 获取某一列的统计信息
```python
reviews.points.describe() # 显示该列的统计信息
reviews.points.mean() # 平均数
reviews.taster_name.unique() # 去重后的列表
reviews.taster_name.value_counts() # 计数
```

### 映射
```python
reviews.points.map(lambda x: x * 100) # 返回一个处理后的 Series
reviews.points * 100 # 与上面等价，但因为使用内建函数所以更快
reviews.country + " - " + reviews.region_1 # 返回一个 Series，内容格式是 "country - region_1"

def remean_points(row):
    row.points = row.points * 100
    return row

reviews.apply(remean_points, axis='columns') # 对每一行作如上处理，返回处理后的 DataFrame
reviews.apply(remean_points, axis='index') # 对每一列处理
```

若 `apply()` 提供的函数返回为单值，则 `apply` 返回的将会是 Series 而非 DataFrame。

注意：这两个函数都不会修改原表的内容。

## 4. Grouping and Sorting
