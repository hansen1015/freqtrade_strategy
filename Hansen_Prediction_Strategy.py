# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
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
|   WIN/USDT |     25 |           4.42 |         110.43 |          9452.601 |         945.26 |       12:10:00 |     18 |       0 |        7 |
|   CHZ/USDT |     40 |           4.85 |         193.93 |          7447.155 |         744.72 |       13:42:00 |     33 |       0 |        7 |
| MATIC/USDT |     42 |           3.62 |         152.05 |          5905.788 |         590.58 |       14:53:00 |     30 |       0 |       12 |
|   SOL/USDT |     24 |           5.73 |         137.53 |          5129.971 |         513.00 |       10:42:00 |     21 |       0 |        3 |
|   ADA/USDT |     46 |           4.19 |         192.57 |          4282.307 |         428.23 |       12:56:00 |     38 |       0 |        8 |
|  DOGE/USDT |     22 |           3.65 |          80.33 |          3998.002 |         399.80 |       11:35:00 |     17 |       0 |        5 |
|   SXP/USDT |     29 |           3.35 |          97.27 |          3682.674 |         368.27 |       12:29:00 |     22 |       0 |        7 |
|   BTT/USDT |     18 |           4.32 |          77.85 |          3324.979 |         332.50 |        8:40:00 |     15 |       0 |        3 |
|   VET/USDT |     55 |           3.88 |         213.13 |          3239.094 |         323.91 |       14:20:00 |     40 |       0 |       15 |
|   TRX/USDT |     31 |           2.67 |          82.80 |          2549.038 |         254.90 |       14:39:00 |     25 |       0 |        6 |
|  CAKE/USDT |      5 |           4.33 |          21.65 |          1934.075 |         193.41 |       17:24:00 |      5 |       0 |        0 |
|   ETH/USDT |     37 |           4.60 |         170.30 |          1672.903 |         167.29 |       11:13:00 |     28 |       0 |        9 |
|   SRM/USDT |     18 |           2.05 |          36.90 |          1573.149 |         157.31 |       10:23:00 |      9 |       0 |        9 |
|   DOT/USDT |      9 |           2.24 |          20.20 |          1458.994 |         145.90 |        8:33:00 |      5 |       0 |        4 |
|   BTC/USDT |     14 |           3.20 |          44.85 |           552.987 |          55.30 |        7:17:00 |     11 |       0 |        3 |
|   XRP/USDT |     32 |           1.61 |          51.42 |           198.890 |          19.89 |       12:15:00 |     25 |       0 |        7 |
|   LTC/USDT |     20 |           1.97 |          39.34 |           148.471 |          14.85 |       10:36:00 |     14 |       0 |        6 |
|      TOTAL |    467 |           3.69 |        1722.55 |         56551.078 |        5655.11 |       12:27:00 |    356 |       0 |      111 |
====================================================== SELL REASON STATS =======================================================
|   Sell Reason |   Sells |   Wins |   Draws |   Losses |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|---------------+---------+--------+---------+----------+----------------+----------------+-------------------+----------------|
|   sell_signal |     465 |    354 |       0 |      111 |           3.69 |        1716.21 |         55652.3   |         429.05 |
|    force_sell |       2 |      2 |       0 |        0 |           3.17 |           6.34 |           898.799 |           1.58 |
========================================================= LEFT OPEN TRADES REPORT ==========================================================
|      Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Wins |   Draws |   Losses |
|-----------+--------+----------------+----------------+-------------------+----------------+----------------+--------+---------+----------|
|  WIN/USDT |      1 |           5.50 |           5.50 |           780.392 |          78.04 |       16:00:00 |      1 |       0 |        0 |
| DOGE/USDT |      1 |           0.84 |           0.84 |           118.406 |          11.84 |        1:00:00 |      1 |       0 |        0 |
|     TOTAL |      2 |           3.17 |           6.34 |           898.799 |          89.88 |        8:30:00 |      2 |       0 |        0 |
=============== SUMMARY METRICS ===============
| Metric                | Value               |
|-----------------------+---------------------|
| Backtesting from      | 2018-01-11 00:00:00 |
| Backtesting to        | 2021-04-26 10:00:00 |
| Max open trades       | 4                   |
|                       |                     |
| Total trades          | 467                 |
| Starting balance      | 1000.000 USDT       |
| Final balance         | 57551.078 USDT      |
| Absolute profit       | 56551.078 USDT      |
| Total profit %        | 5655.11%            |
| Trades per day        | 0.39                |
| Avg. stake amount     | 3221.648 USDT       |
| Total trade volume    | 1504509.398 USDT    |
|                       |                     |
| Best Pair             | VET/USDT 213.13%    |
| Worst Pair            | DOT/USDT 20.2%      |
| Best trade            | VET/USDT 84.48%     |
| Worst trade           | MATIC/USDT -29.48%  |
| Best day              | 2989.536 USDT       |
| Worst day             | -1351.322 USDT      |
| Days win/draw/lose    | 175 / 974 / 52      |
| Avg. Duration Winners | 11:16:00            |
| Avg. Duration Loser   | 16:15:00            |
|                       |                     |
| Min balance           | 988.322 USDT        |
| Max balance           | 57551.078 USDT      |
| Drawdown              | 63.64%              |
| Drawdown              | 2087.273 USDT       |
| Drawdown high         | 51065.169 USDT      |
| Drawdown low          | 48977.895 USDT      |
| Drawdown Start        | 2021-04-22 13:00:00 |
| Drawdown End          | 2021-04-23 11:00:00 |
| Market change         | 1444.02%            |
===============================================
"""
