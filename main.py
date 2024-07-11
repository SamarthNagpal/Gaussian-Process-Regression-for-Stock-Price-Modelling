import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
import scipy.stats as stats

stock = 'TCS'           # symbol of the stock
period = '1y'           # '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'
interval = '1d'         # '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
window_size = 150      # number of intervals used for predicting

# this function gets the price data from yfinance
def fetch_data(stock, period, interval):

    ticker = yf.Ticker(stock)
    return ticker.history(period=period, interval=interval)

# prepares data and splits it into training and testing datasets
def prepare_data(stock, period, interval, window_size):
    df = fetch_data(stock, period, interval)

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    

    selected_columns = ['Open', 'High', 'Low', 'Close', 'MA20', 'MA50']
    data = df[selected_columns]
    data.dropna(inplace=True)
    data = np.asarray(data)
    
    X_list = []
    y_list = []
    for i in range(len(data) - window_size):
        X_window = data[i:i+window_size]
        y_value = data[i+window_size][3]
        X_list.append(X_window)
        y_list.append(y_value)

    X = np.array(X_list)
    y = np.array(y_list)
    y = y.reshape(-1, 1) 
    X_flat = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=20, random_state=42)
    return X_train, X_test, y_train, y_test

# calculates the accuracy of prediction
def accuracy(y_test, y_pred, sigma, probability):

    lower_bounds = np.zeros_like(y_pred)
    upper_bounds = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        ci = stats.norm.interval(probability, loc=y_pred[i], scale=sigma[i])
        lower_bounds[i] = ci[0]
        upper_bounds[i] = ci[1]
    
    count = 0
    for i in range(len(y_test)):
        if y_test[i] >= lower_bounds[i] and y_test[i] <= upper_bounds[i]:
            count += 1
    
    return f'{count/len(y_test)*100}%'

# plots the training closing prices, testing closing prices and the predicted closing prices
def plot(y_train, y_test, y_pred, sigma, probability):

    plt.scatter(range(len(y_train)), y_train, color='blue', label='Training Data', s=10)
    plt.scatter(range(len(y_train), len(y_train) + len(y_test)), y_test, color='green', label='Testing Data', s=10)
    plt.scatter(range(len(y_train), len(y_train) + len(y_test)), y_pred, color='red', label='Predicted Data', s=10)

    lower_bounds = np.zeros_like(y_pred)
    upper_bounds = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        ci = stats.norm.interval(probability, loc=y_pred[i], scale=sigma[i])
        lower_bounds[i] = ci[0]
        upper_bounds[i] = ci[1]
    plt.vlines(range(len(y_train), len(y_train) + len(y_test)), lower_bounds, upper_bounds, colors='red', label=f'{probability*100}% Confidence Interval')
    
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Closing Price')

    x_min = -1
    x_max = len(y_train) + len(y_test) + 1
    y_min = min(y_test.min(), y_pred.min()) - 1
    y_max = max(y_test.max(), y_pred.max()) + 1
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.show()


X_train, X_test, y_train, y_test = prepare_data(stock, period, interval, window_size)

# training and testing

# some other kernels that can also be used instead
# kernel = kernels.Matern(length_scale=1)
# kernel = kernels.RationalQuadratic(length_scale=1.0, alpha=1.0)
kernel = kernels.RBF(length_scale=1)      # length_scale parameter can be used to optimize the model for different datsets
gaussian_process = GaussianProcessRegressor(kernel=kernel)

gaussian_process.fit(X_train, y_train)
y_pred, sigma = gaussian_process.predict(X_test, return_std=True) 

probability = 0.99   # to define the confidence interval

print(accuracy(y_test, y_pred, sigma, probability))

plot(y_train, y_test, y_pred, sigma, probability)


