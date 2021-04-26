#This is not made by Hansen1015, I just edit a little bit of the code that makes it more reliable.
#the author is https://github.com/Bturan19
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import xgboost
import catboost
import sklearn
import pickle
from numba import jit
from scipy import signal

from freqtrade.strategy import IStrategy
import technical.indicators as ftt
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# This class is a sample. Feel free to customize it.
class Hansen_Prediction_Strategy(IStrategy):

    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 100
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -99

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 240

    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):

        return []

    

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        verbose = False
        col_use = [
                    'volume','smadiff_3','smadiff_5','smadiff_8','smadiff_13',
                    'smadiff_21','smadiff_34','smadiff_55','smadiff_89',
                    'smadiff_120','smadiff_240','maxdiff_3','maxdiff_5','maxdiff_8',
                    'maxdiff_13','maxdiff_21','maxdiff_34','maxdiff_55','maxdiff_89',
                    'maxdiff_120','maxdiff_240','std_3','std_5','std_8','std_13',
                    'std_21','std_34','std_55','std_89','std_120','std_240',
                    'ma_3','ma_5','ma_8','ma_13','ma_21','ma_34','ma_55','ma_89',
                    'ma_120','ma_240','z_score_120','time_hourmin','time_dayofweek','time_hour' ]


        with open('user_data/notebooks/model_portfolio.pkl', 'rb') as f:
            model = pickle.load(f)
        model = model[0]

        # Starting create features
        #sma diff
        for i in [3,5,8,13,21,34,55,89,120,240]:
            dataframe[f"smadiff_{i}"] = (dataframe['close'].rolling(i).mean() - dataframe['close'])
        #max diff
        for i in [3,5,8,13,21,34,55,89,120,240]:
            dataframe[f"maxdiff_{i}"] = (dataframe['close'].rolling(i).max() - dataframe['close'])
        #min diff
        for i in [3,5,8,13,21,34,55,89,120,240]:
            dataframe[f"maxdiff_{i}"] = (dataframe['close'].rolling(i).min() - dataframe['close'])
        #volatiliy
        for i in [3,5,8,13,21,34,55,89,120,240]:
            dataframe[f"std_{i}"] = dataframe['close'].rolling(i).std()
        
        #Return
        for i in [3,5,8,13,21,34,55,89,120,240]:
            dataframe[f"ma_{i}"] = dataframe['close'].pct_change(i).rolling(i).mean()
        
        dataframe['z_score_120'] = ((dataframe.ma_13 - dataframe.ma_13.rolling(21).mean() + 1e-9) 
                            / (dataframe.ma_13.rolling(21).std() + 1e-9))
        
        dataframe["date"] = pd.to_datetime(dataframe["date"], unit='ms')
        dataframe['time_hourmin'] = dataframe.date.dt.hour * 60 + dataframe.date.dt.minute
        dataframe['time_dayofweek'] = dataframe.date.dt.dayofweek
        dataframe['time_hour'] = dataframe.date.dt.hour

        #Model predictions
        preds = pd.DataFrame(model.predict_proba(dataframe[col_use]))
        preds.columns = [f"pred{i}" for i in range(5)]
        dataframe = dataframe.reset_index(drop=True)
        dataframe = pd.concat([dataframe, preds], axis=1)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['pred4'] > .52) & 
                #(dataframe["time_hour"].isin([23,2,5,8,11,14,17,20])) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['pred4'] < .29) 
                #& (dataframe["time_hour"].isin([23,2,5,8,11,14,17,20]))
                #& (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe

"""
============================================================= BACKTESTING REPORT ============================================================
|       Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Wins |   Draws |   Losses |
|------------+--------+----------------+----------------+-------------------+----------------+----------------+--------+---------+----------|
|   WIN/USDT |     23 |           9.43 |         216.91 |         78817.667 |        7881.77 |       10:18:00 |     20 |       0 |        3 |
|  DOGE/USDT |     24 |           6.71 |         161.11 |         28707.677 |        2870.77 |       11:55:00 |     20 |       0 |        4 |
|   BTT/USDT |     23 |           4.85 |         111.65 |         27576.467 |        2757.65 |        9:52:00 |     17 |       0 |        6 |
|   CHZ/USDT |     32 |           4.38 |         140.02 |         20703.234 |        2070.32 |       13:28:00 |     25 |       0 |        7 |
|   SXP/USDT |     28 |           3.60 |         100.80 |         19689.706 |        1968.97 |       10:28:00 |     23 |       0 |        5 |
| MATIC/USDT |     47 |           7.60 |         357.25 |         18187.929 |        1818.79 |       13:50:00 |     38 |       0 |        9 |
|   SOL/USDT |     22 |           6.71 |         147.67 |         17335.265 |        1733.53 |        9:33:00 |     19 |       0 |        3 |
|   VET/USDT |     49 |           6.69 |         327.63 |         10754.929 |        1075.49 |       11:18:00 |     40 |       0 |        9 |
|  CAKE/USDT |      3 |           6.75 |          20.24 |         10182.337 |        1018.23 |        9:40:00 |      3 |       0 |        0 |
|   ADA/USDT |     28 |           6.88 |         192.76 |          7314.169 |         731.42 |       12:45:00 |     26 |       0 |        2 |
|   XRP/USDT |     25 |           4.06 |         101.56 |          6837.751 |         683.78 |       13:24:00 |     20 |       0 |        5 |
|   DOT/USDT |      7 |           4.95 |          34.63 |          5659.805 |         565.98 |        9:34:00 |      6 |       0 |        1 |
|   SRM/USDT |     21 |           3.13 |          65.71 |          4615.968 |         461.60 |       10:06:00 |     16 |       0 |        5 |
|   TRX/USDT |     22 |           5.35 |         117.71 |          2641.038 |         264.10 |       10:41:00 |     19 |       0 |        3 |
|   ETH/USDT |     27 |           4.89 |         132.05 |          1517.090 |         151.71 |        9:00:00 |     24 |       0 |        3 |
|   LTC/USDT |     21 |           3.61 |          75.90 |          1053.731 |         105.37 |        7:31:00 |     16 |       0 |        5 |
|   BTC/USDT |     14 |           3.28 |          45.97 |           360.400 |          36.04 |        7:34:00 |     13 |       0 |        1 |
|      TOTAL |    416 |           5.65 |        2349.57 |        261955.162 |       26195.52 |       11:08:00 |    345 |       0 |       71 |
====================================================== SELL REASON STATS =======================================================
|   Sell Reason |   Sells |   Wins |   Draws |   Losses |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|---------------+---------+--------+---------+----------+----------------+----------------+-------------------+----------------|
|   sell_signal |     415 |    344 |       0 |       71 |           5.65 |        2342.83 |         257725    |         585.71 |
|    force_sell |       1 |      1 |       0 |        0 |           6.74 |           6.74 |           4230.37 |           1.68 |
========================================================= LEFT OPEN TRADES REPORT =========================================================
|     Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Wins |   Draws |   Losses |
|----------+--------+----------------+----------------+-------------------+----------------+----------------+--------+---------+----------|
| WIN/USDT |      1 |           6.74 |           6.74 |          4230.371 |         423.04 |       15:00:00 |      1 |       0 |        0 |
|    TOTAL |      1 |           6.74 |           6.74 |          4230.371 |         423.04 |       15:00:00 |      1 |       0 |        0 |
=============== SUMMARY METRICS ===============
| Metric                | Value               |
|-----------------------+---------------------|
| Backtesting from      | 2018-01-11 00:00:00 |
| Backtesting to        | 2021-04-26 10:00:00 |
| Max open trades       | 4                   |
|                       |                     |
| Total trades          | 416                 |
| Starting balance      | 1000.000 USDT       |
| Final balance         | 262955.162 USDT     |
| Absolute profit       | 261955.162 USDT     |
| Total profit %        | 26195.52%           |
| Trades per day        | 0.35                |
| Avg. stake amount     | 12037.355 USDT      |
| Total trade volume    | 5007539.733 USDT    |
|                       |                     |
| Best Pair             | MATIC/USDT 357.25%  |
| Worst Pair            | CAKE/USDT 20.24%    |
| Best trade            | WIN/USDT 110.5%     |
| Worst trade           | DOGE/USDT -26.18%   |
| Best day              | 48303.112 USDT      |
| Worst day             | -6460.807 USDT      |
| Days win/draw/lose    | 176 / 985 / 40      |
| Avg. Duration Winners | 10:32:00            |
| Avg. Duration Loser   | 14:02:00            |
|                       |                     |
| Min balance           | 987.296 USDT        |
| Max balance           | 262955.162 USDT     |
| Drawdown              | 36.0%               |
| Drawdown              | 13085.976 USDT      |
| Drawdown high         | 219536.987 USDT     |
| Drawdown low          | 206451.011 USDT     |
| Drawdown Start        | 2021-04-20 22:00:00 |
| Drawdown End          | 2021-04-23 11:00:00 |
| Market change         | 1444.02%            |
===============================================
"""
