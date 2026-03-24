"""
Deep Learning Model for Dissolved Oxygen Prediction

"""

# Imports necessary Python Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Bidirectional, SimpleRNN
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K


# Custom Metrics
def r_squared(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


# Noise Augmentation
def add_noise(data, noise_factor=0.01):
    noise = np.random.randn(*data.shape) * noise_factor
    return data + noise


class TimeSeriesGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, noise_factor=0.01):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.noise_factor = noise_factor

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_X_aug = add_noise(batch_X, self.noise_factor)
        return batch_X_aug, batch_y


# Load Dataset
file_path = "combine_data.csv"  
data = pd.read_csv(file_path)

data['SAMPLE_DATE'] = pd.to_datetime(
    data['Date'],
    format='%Y-%m-%dT%H:%M:%S.%fZ',
    errors='coerce'
)

data = data.drop(columns=['Date'])

print("Dataset loaded:")
print(data.head())


# Data Cleaning
numeric_cols = ['Temperature', 'Pressure', 'Salinity', 'Oxygen']

print("\nChecking NaNs...")
print(data[numeric_cols].isnull().sum())

data = data.dropna()


# Feature Scaling
variables = ['Salinity', 'Pressure', 'Temperature']

scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(data[variables])

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(data['Oxygen'].values.reshape(-1, 1))


# Sliding Window
time_steps = 10

X_series, y_series = [], []

for i in range(len(X_scaled) - time_steps):
    X_series.append(X_scaled[i:i + time_steps])
    y_series.append(y_scaled[i + time_steps])

X_series = np.array(X_series)
y_series = np.array(y_series)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_series, y_series, test_size=0.2, random_state=42
)


# Model Definition
model = Sequential([
    Bidirectional(SimpleRNN(units=16, return_sequences=True, activation='relu', input_shape=(time_steps, X_train.shape[2]))), # Replace look_back with time_steps
    Conv1D(filters=16, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(50, activation='relu', return_sequences=False)),
    Dropout(0.2),
    Dense(50, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='mse',
    metrics=['mae', r_squared]
)

model.summary()


# Training Setup
train_gen = TimeSeriesGenerator(X_train, y_train, batch_size=64)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-5,
    verbose=1
)


# Train Model
history = model.fit(
    train_gen,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)


# Evaluation
loss, mae, r2 = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss}")
print(f"Test MAE: {mae}")
print(f"Test R2: {r2}")


# Predictions
# =========================
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)


# Plot Training History
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.grid()
plt.show()


# Plot Predictions
plt.figure()
plt.plot(y_test_actual[:100])
plt.plot(y_pred[:100])
plt.xlabel('Time Steps')
plt.ylabel('Oxygen (ml/L)')
plt.title('Actual vs Predicted Oxygen')
plt.grid()
plt.show()
