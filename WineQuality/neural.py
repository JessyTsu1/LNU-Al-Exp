import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# from sklearn import preprocessing
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("./winequality-white.csv", sep=';')
# print (data.head(5))
y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=113)

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=256,
    epochs=100,
)

history_df = pd.DataFrame(history.history)
history_df['loss'].plot();
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

## 如果连续30次迭代，每次的loss下降都不足0.001，则训练终止
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001,  # minimium amount of change to count as an improvement
    patience=30,  # how many epochs to wait before stopping
    restore_best_weights=True,
)
# 预测结果
y_pred = model.predict(X_test)
print(y_pred[:10])
print(y_test[:10])

# 模型评估
score = model.evaluate(X_test, y_test, verbose=1)
# socre的两个值分别代表损失(loss)和精准度(accuracy)
print(score)
