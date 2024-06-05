import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 데이터 로드
data = pd.read_csv('coefficients.csv', parse_dates=['date'], index_col='date')

# 데이터 전처리
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# LSTM 입력 형식으로 데이터 변환
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        Y.append(data[i + time_step, :])
    return np.array(X), np.array(Y)

# 하이퍼파라미터 설정
time_step = 1
X, Y = create_dataset(scaled_data, time_step)

# LSTM 모델 설계
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(X.shape[2]))  # a, b, c, risk_free_rate의 수

model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X, Y, epochs=50, batch_size=64, verbose=1)

# 다음 date 예측
last_sequence = scaled_data[-time_step:]  # 가장 최근 time_step 만큼의 데이터
last_sequence = np.expand_dims(last_sequence, axis=0)  # 입력 형식에 맞게 변환
next_prediction_scaled = model.predict(last_sequence)
next_prediction = scaler.inverse_transform(next_prediction_scaled)

# 예측 결과 출력
print(f"Next date prediction for a, b, c, risk_free_rate: {next_prediction[0]}")
