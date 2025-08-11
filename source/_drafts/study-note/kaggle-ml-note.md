## Intro to Machine Learning
### 2 Basic Data Exploration
```python
data = pd.read_csv("data.csv")
data.describe()
```

### 3 Basic Data Exploration
```python
data.columns
data = data.dropna(axis=0)

y = data.Price
X = data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]
X.head()

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)
model.predict(X)
```

### 4 
```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, predicted_home_prices)

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=3)
```

### 5 Underfitting and Overfitting
```python
model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
```

生产环境中使用时，直接把所有数据拿去测试。

### 6 Random Forests
```python
from sklearn.ensemble import RandomForestRegressor
```