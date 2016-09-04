import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as skm
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor

#Function to import SPY data for 2 years
def data_import():
    df = pd.read_csv("SPY.csv", index_col = 'Date', parse_dates=True)
    return df.iloc[::-1]

#Function to get rolling mean of price
def getRollingMean(values, window):
    return pd.rolling_mean(values, window= window)

#Function to get rolling Standard Deviation of price
def getRollingSD(values, window):
    return pd.rolling_std(values, window= window)

#Function to get daily returns
def getDailyReturns(df):
    dailyReturns = df.copy()
    dailyReturns[:-1] = (df[1:] / df[:-1].values) - 1
    dailyReturns[0] = 0
    return dailyReturns

#Function to get rolling mean of daily returns
def getRollingDailyReturnMean(values, window):
    return pd.rolling_mean(values, window= window)

#Function to get rolling standard deviation of daily returns
def getRollingDailyReturnSTD(values, window):
    return pd.rolling_std(values, window= window)

#Function to get last week's momentum
def getMomentumLastWeek(df):
    momentum = df.copy()
    momentum[5:] = (df[5:] / df[:-5].values) - 1
    momentum[0:4] = 0
    return momentum

#Function to get weekly returns
def getWeeklyReturns(df):
    weeklyReturn = df.copy()
    weeklyReturn[:-5] = (df[5:] / df[:-5].values) - 1
    weeklyReturn[-5:] = 0
    return weeklyReturn

#Function to perform elastic net regression
def doLinearReg(X_train, Y_train, X_test, Y_test, l1_ratio):
    enet = ElasticNetCV(l1_ratio=l1_ratio, max_iter=1000000)
    enet_pred = enet.fit(X_train,Y_train).predict(X_test)
    result = np.sum(np.sign(enet_pred) * np.sign(Y_test.values))
    return result

#Function to do K-nearest neighbor
def doKNN(X_train, Y_train, X_test, Y_test, n_neighbors):
    knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_pred = knn.fit(X_train,Y_train).predict(X_test)
    result = np.sum(np.sign(knn_pred) * np.sign(Y_test.values))
    return result

#Function to do kernel ridge regression
def doKernelRidge(X_train, Y_train, X_test, Y_test, gamma):
    kRidge = KernelRidge(alpha=1.0, gamma = gamma)
    kRidge_pred = kRidge.fit(X_train,Y_train).predict(X_test)
    result = np.sum(np.sign(kRidge_pred) * np.sign(Y_test.values))
    return result

#Function to do Random Forest regression
def doRF(X_train, Y_train, X_test, Y_test):
    RF = RandomForestRegressor(n_estimators=500)
    RF_pred = RF.fit(X_train,Y_train).predict(X_test)
    result = np.sum(np.sign(RF_pred) * np.sign(Y_test.values))
    return result

#Function to make ensemble of elastic net regression
def doEnsemble(X_train, Y_train, X_test, Y_test):
    enet1 = ElasticNetCV(l1_ratio= 1, max_iter=1000000)
    enet_avg = ElasticNetCV(l1_ratio=0.5, max_iter=1000000)
    enet1_pred = enet1.fit(X_train, Y_train).predict(X_test)
    enet_avg_pred = enet_avg.fit(X_train, Y_train).predict(X_test)
    enet_pred = (enet1_pred + enet_avg_pred)/2
    result = np.sum(np.sign(enet_pred) * np.sign(Y_test.values))
    return result

#Function to predict random forests
def predictRF(X_train, Y_train, X_test):
    RF = RandomForestRegressor(n_estimators=500)
    RF_pred = RF.fit(X_train,Y_train).predict(X_test)
    return np.sign(RF_pred)


if __name__ == "__main__":
    spy = data_import()
    ax = spy['Adj Close'].plot()
    #Calculating different features for regression
    #Calculating Rolling mean of price for last 3 months
    rm_SPY = getRollingMean(spy['Adj Close'], window=63)

    #Calculating Rolling standard deviation of price for last 3 months
    rsd_SPY = getRollingSD(spy['Adj Close'], window=63)

    #Calculating Bollinger value for last 3 months
    bollinger = (rm_SPY - spy['Adj Close'])/(2 * rsd_SPY)

    #calculating price / Rolling mean
    priceRMRatio = (spy['Adj Close'] / rm_SPY)

    #Calculating daily returns
    dailyReturns = getDailyReturns(spy['Adj Close'])

    #Calculating rolling daily returns mean for last 3 months
    rm_DailyReturn = getRollingDailyReturnMean(dailyReturns, window=63)

    #Calculating volatility for last 3 months
    rsd_DailyReturn = getRollingDailyReturnSTD(dailyReturns, window=63)

    #Calculating Sharpe ratio for last 3 months
    sharpeRatio = rm_DailyReturn / rsd_DailyReturn

    #Calculating moment for last week
    momentumLastWeek = getMomentumLastWeek(spy['Adj Close'])

    #Creating a dataframe with all the attributes
    spy = spy.join(rm_SPY, rsuffix = ' RM')
    spy = spy.join(rsd_SPY, rsuffix=' RSD')
    spy = spy.join(bollinger, rsuffix=' bollinger')
    spy = spy.join(priceRMRatio, rsuffix='/RM')
    spy = spy.join(sharpeRatio, rsuffix=' Sharpe Ratio')
    spy = spy.join(momentumLastWeek, rsuffix=' Momentum')
    spy = spy.join((dailyReturns*100), rsuffix=' Daily Return')
    spy.drop(spy.columns[[0,1,2,3]], inplace=True, axis=1)
    spy = spy[62:-1]

    #Normalizing
    spy.ix[:, 0:8] = (spy.ix[:, 0:8] - spy.ix[:, 0:8].min()) / (spy.ix[:, 0:8].max() - spy.ix[:, 0:8].min())

    #Linear Regression: Ridge/Lasso/Elastic net

    for i in range(1,7):
        X_train = spy.ix[63*(i - 1):(63*i), 0:8]
        Y_train = spy.ix[63*(i - 1):(63*i), -1]
        X_test = spy.ix[(63*i):((63*i) + 5), 0:8]
        Y_test = spy.ix[(63*i):((63*i) + 5), -1]
        lin_model_score1 = doLinearReg(X_train, Y_train, X_test, Y_test, 1)
        lin_model_score0 = doLinearReg(X_train, Y_train, X_test, Y_test, 0.001)
        lin_model_score_avg = doLinearReg(X_train, Y_train, X_test, Y_test, 0.5)
        print('For LR, backtest no.', i, ', lin_model_score1 is', lin_model_score1, 'lin_model_score0 is', lin_model_score0,\
        'lin_model_score_avg is', lin_model_score_avg)


    #KNN

    for i in range(1,7):
        X_train = spy.ix[63*(i - 1):(63*i), 0:8]
        Y_train = spy.ix[63*(i - 1):(63*i), -1]
        X_test = spy.ix[(63*i):((63*i) + 5), 0:8]
        Y_test = spy.ix[(63*i):((63*i) + 5), -1]
        knn_model_score3 = doKNN(X_train, Y_train, X_test, Y_test, 3)
        knn_model_score5 = doKNN(X_train, Y_train, X_test, Y_test, 5)
        knn_model_score7 = doKNN(X_train, Y_train, X_test, Y_test, 7)
        knn_model_score9 = doKNN(X_train, Y_train, X_test, Y_test, 9)
        print('For KNN, backtest no.', i, ', KNN3 score is', knn_model_score3, 'KNN5 score is', knn_model_score5,\
        'KNN7 score is', knn_model_score7, 'KNN9 score is', knn_model_score9)

    #Kernel regression
    spyDist = pd.DataFrame(squareform(pdist(spy, 'euclidean')))
    sRange = range(0,3)
    for i in range(1,7):
        X_train = spy.ix[63*(i - 1):(63*i), 0:8]
        Y_train = spy.ix[63*(i - 1):(63*i), -1]
        X_test = spy.ix[(63*i):((63*i) + 5), 0:8]
        Y_test = spy.ix[(63*i):((63*i) + 5), -1]
        kRidge1 = doKernelRidge(X_train, Y_train, X_test, Y_test, sRange[0])
        kRidge2 = doKernelRidge(X_train, Y_train, X_test, Y_test, sRange[1])
        kRidge3 = doKernelRidge(X_train, Y_train, X_test, Y_test, sRange[2])
        print('For kRidge, backtest no.', i, ', kRidge1 score is', kRidge1, 'kRidge2 score is', kRidge2,\
        'kRidge3 score is', kRidge3)

    #RandomForest Regression
    for i in range(1,7):
        X_train = spy.ix[63*(i - 1):(63*i), 0:8]
        Y_train = spy.ix[63*(i - 1):(63*i), -1]
        X_test = spy.ix[(63*i):((63*i) + 5), 0:8]
        Y_test = spy.ix[(63*i):((63*i) + 5), -1]
        RF_model_score = doRF(X_train, Y_train, X_test, Y_test)
        print('For RF, backtest no.', i, ', RF score is', RF_model_score)

    #Creating ENSEMBLE (LR1 + LR_avg): Linear models with l1_ratio = 1 and 0.5

    for i in range(1,7):
        X_train = spy.ix[63*(i - 1):(63*i), 0:8]
        Y_train = spy.ix[63*(i - 1):(63*i), -1]
        X_test = spy.ix[(63*i):((63*i) + 5), 0:8]
        Y_test = spy.ix[(63*i):((63*i) + 5), -1]
        ensembleScore = doEnsemble(X_train, Y_train, X_test, Y_test)
        print('For Ensemble, backtest no.', i, ', Ensemble score is', ensembleScore)

    #Resultant model worse than our best model (LR = 0.5), We stick to LR = 0.5

    #Testing the strategy
    #Strategy is to predict for next week and take positions at the beginning of the week. If daily return positive then long, otherwise short it
    spy['State'] = spy['Adj Close']
    spy.ix[0:63, -1] = 0
    i = 63
    step = 5

    #Simulating the strategy 
    while (i + 5) < len(spy.index):
        X_train = spy.ix[(i - 63):i, 0:8]
        Y_train = spy.ix[(i - 63):i, -2]
        X_test = spy.ix[i : (i + 5), 0:8]
        final = ElasticNetCV(l1_ratio=0.5, max_iter=10000)
        final_pred = final.fit(X_train, Y_train).predict(X_test)
        spy.ix[i : (i + 5),-1] = np.sign(final_pred)
        i = i + step

    spy['Predicted Return'] = spy['Adj Close Daily Return'] * spy['State']
    result = spy[['Adj Close Daily Return', 'Predicted Return']]
    result = result.ix[63:-4,:]
    result['SPY'] = result['Adj Close Daily Return']
    result['Strategy'] = result['Predicted Return']
    result.ix[0,2] = 100
    result.ix[0, 3] = 100

    for i in range(1,len(result.index)):
        result.ix[i, 2] = ((result.ix[i - 1, 2])/100) * (100 + result.ix[i, 0])
        result.ix[i, 3] = ((result.ix[i - 1, 3]) / 100) * (100 + result.ix[i, 1])

    ax = result[['SPY', 'Strategy']].plot(title = 'Strategy Comparison')
    plt.show()










