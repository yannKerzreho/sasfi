"""
models package — univariate financial time series forecasters.

All models implement the BaseForecaster protocol (models.base):
    fit(history, horizons)   — calibrate on 1-D numpy array (T,)
    update(x)                — advance state with one new scalar (no refit)
    predict(h)               — scalar h-step-ahead point forecast

Category overview
-----------------
Stat  : HARForecaster (linear.py), GARCHForecaster (garch.py)
ML    : HARForecaster(ridge=True)  (linear.py)
RC    : SASForecaster (sas.py)
DL    : LSTMForecaster (rnn.py)
"""
