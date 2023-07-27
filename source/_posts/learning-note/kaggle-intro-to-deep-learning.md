---
title: Kaggle Intro to Deep Learning 学习笔记
date: 2023-02-16
updated: 2023-02-18
category: 
- 学习
tag:
- Kaggle
---

Kaggle 的 [Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning) 的笔记。

<!-- more -->

## 1. A Single Neuron
### 创建单个神经元
```python
from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
```

其中 `layers.Dense()` 表示一个稠密层，`units` 表示该层输出元素个数，`input_shape=[height, width, channels]` 描述该层输入大小。

用 `model.weights` 获取参数。

## 2. Deep Neural Networks
### 创建网络
```python
model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
```

也可以使用 `layers.Activation('relu')`。

### 其他激活函数
ReLU：

$$ReLU(x) = \max\{0, x\}$$

ELU：

$$
ELU(x, \alpha) =
\begin{cases}
    x, &x \geq 0 \\
    \alpha(e^x - 1), &x < 0    
\end{cases}
$$

SeLU：

$$
SeLU(x) = 
\begin{cases}
    \lambda_{selu}x, &x \geq 0 \\
    \lambda_{selu}\alpha_{selu}(e^x - 1), &x < 0 \\    
\end{cases}
$$

其中 $\alpha_{selu} \approx 1.6733, \lambda_{selu} \approx 1.0507$。

## 3. Stochastic Gradient Descent
* RMSE（Root Mean Square Error，均方根误差）
* MAE（Mean Absolute Error，平均绝对误差）

```python
# 钦定模型所用的 optimizer 和损失函数
model.compile(
    optimizer="adam",
    loss="mae",
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot();
```

![](https://www.kaggleusercontent.com/kf/94853433/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..yQ6-AuSYkX5KS_f7osU1uA.claTNB4uwbDWq2zpl_AtHyPaKofrUEK1rrcsLlSC4ZGiBHpDDr_z91sTRQhPSFJ0dicLSP3_0A45nvsmoo3iZ1Nqzzwe2zhZ0QZIr8DOaeWJt89SHEkeI_jLPxdfX72QpZoOHxEULiViysAH1zOQJuxkfDeVznQkv_QFnSf1TYXPWo0y0-ZMMMKhev8ijSM2KI6TunfNkiyEszQ833yPhibR8NTLBTXgeB5EDU-tpcMkG6ki8Wlx41JlrhjzZfLTedE8DRU4W-BWvAnECSm7d6L16YpOkFJxpoXx3IN2Y8ku0-sjJajMLfzGEWVTfnlVanF1HK6pnrMnK2oYk1doENmboNZPxhrW3S5oHoTXVRVhUvYnKst6PvBa_GY34SiLCtdOR5QH7eenJM7o16UcZliSxjyFnl3hPPxrbW3SqEjY8AXs_27sFK03XgAaOZPttiQmG743__X_V4Msq809qmcI99SOILypY1_HS_2SaLrAXddu2-SIRLCzX3gocBczOrydi-xJ0N7e4FETqQu8j7jKZ1TfIsOxohE0DVN_gpdKjzlTsU_SgYRQwNtS6WEQgKtwjwN_0-AbZcoPin-QvI8fb6ZSEeCy__dlHf5o9jv43lAXumiVUz302YLH20qLFdfA5W4pL1TBlg28-NTyz0_HVs-VGO5x5SWgEobQkNE.7FccNx2hBZxUcfrBwbdMZw/__results___files/__results___12_0.png)

## 4. Overfitting and Underfitting
![Underfitting](/img/post/kaggle-intro-to-deep-learning/underfitting.png)

![Overfitting](/img/post/kaggle-intro-to-deep-learning/overfitting.png)

### 提前停止（Early stopping）
Keras 中，Early stopping 是一种回调（Callback）函数，每次迭代后都会执行。

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
)
# 当 20 次迭代内都没有使得验证集的损失函数值减小至少 0.001 时，停止训练

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], 
    verbose=0,  # 关闭日志输出
)

history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
```

## 5. Dropout and Batch Normalization
### Dropout
`layers.Dropout(rate=0.3)`

### Batch Normalization（BN）
BN 层拥有两个可训练的参数 $\mu, \beta$。首先，BN 会对输入参数进行正则化（$\mu = 0, \sigma = 1$），即 $x_i \leftarrow \frac {x_i - \mu}{\sqrt {\sigma^2 + \epsilon}}$；之后再让 $x_i \leftarrow \mu x_i + \beta$。这样可以在正则化数据的同时，又用 $\mu, \beta$ 作为还原参数，一定程度上保留原数据的分布。

BN 层一般可以缓解梯度爆炸或梯度消失的问题，也能使训练变得更快。

`layers.BatchNormalization()`

示例：

```python
model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])
```

## 6. Binary Classification
### Cross-Entropy（交叉熵）
分类问题使用的损失函数，即 $-\ln p_x$。

### 示例
```python
model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),    
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```