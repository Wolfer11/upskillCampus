from data_preprocessing import preprocess_data
from model_training import create_sequences, build_model

df, scaler = preprocess_data('E:/Smart city forecasting/traffic_data.csv')

seq_length = 24
X, y = create_sequences(df['Vehicles'].values, seq_length)  # Updated column name

X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = build_model(seq_length)
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

model.save('traffic_forecast.h5')
