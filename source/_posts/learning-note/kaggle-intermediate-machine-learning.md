---
title: Kaggle Intermediate Machine Learning 学习笔记
date: 2023-02-22
category: 
- 学习
tag:
- Kaggle
---

Kaggle 的 [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) 的笔记。

<!-- more -->

## 2. Missing Values
```python
# 方法 1：删去有 NA 的特征
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)

# 方法 2：SimpleImputer（使用平均值进行 impute，填充）
# 实践中不一定比方法 1 更好
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# 方法 3：新增一列记录是否 NA，再填充
# 当丢失信息很少的时候可能劣于方法 2
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# 之后再进行填充
```

## 3. Categorical Variables
```python
# 方法 1：删去类别特征
drop_X_train = X_train.select_dtypes(exclude=['object'])

# 方法 2：序数编码（Ordinal Encoding）
# 一般在类别确实可以排名（如 Never < Rarely < Most days < Every day），且 / 或使用的模型为树模型（决策树等）时比较有用
# 要特别小心验证集 / 测试集中有可能出现训练集没有的数据！
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

# 方法 3：Onehot 编码（Onehot Encoding）
from sklearn.preprocessing import OneHotEncoder
# handle_unknown='ignore' 会使得验证集中未出现的类别编码为全 0，而不会抛出错误
# sparse=False 会使返回的为 numpy array（而不是稀疏矩阵）
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

```