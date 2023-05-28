# Start hyperopt with the following command:
# freqtrade hyperopt --config config.json --hyperopt-loss SharpeHyperOptLoss --strategy RsiStrat -e 500 --spaces  buy sell --random-state 8711

# --- Do not remove these libs ---
import numpy as np  # noqa
from freqtrade.persistence import Trade, Order
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd  # noqa
from functools import reduce
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter)

# --- Add your lib to import here ---
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --- Generic strategy settings ---


class justinstrategy(IStrategy):
    INTERFACE_VERSION = 2

    # Determine timeframe and # of candles before strategysignals becomes valid
    timeframe = '1d'
    startup_candle_count: int = 25

    # Determine roi take profit and stop loss points
    minimal_roi = {
        "0": 100
    }
    #    buy_dema_long = IntParameter(3, 150, default=123)
    #    buy_dema_short = IntParameter(15, 200, default=148)
    # buy_adx_enabled = BooleanParameter(default=True, space="buy")

    stoploss = -0.10
    # use_custom_stoploss = True
    trailing_stop = False
    position_adjustment_enable = True
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0
    ignore_roi_if_entry_signal = False
    can_short = True

    selltd9 = 0
    selltd13 = 15
    selltd15 = 30
    selltd16 = 0

    # --- Define spaces for the indicators ---

    #    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
    #                            proposed_stake: float, min_stake: float, max_stake: float,
    #                            **kwargs) -> float:
    #        return self.wallet.get_total_stake_amount()
    # --- Used indicators of strategy code ----
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate all indicators used by the strategy
        """
        dataframe['DEMA123'] = ta.DEMA(dataframe, timeperiod=123)
        dataframe['DEMA148'] = ta.DEMA(dataframe, timeperiod=148)

        dataframe['DEMA117'] = ta.DEMA(dataframe, timeperiod=117)
        dataframe['DEMA149'] = ta.DEMA(dataframe, timeperiod=149)
        dataframe['DEMA75'] = ta.DEMA(dataframe, timeperiod=32)
        dataframe['DEMA23'] = ta.DEMA(dataframe, timeperiod=191)
        #        for val in self.buy_dema_short.range:
        #            dataframe[f'dema{val}_short'] = ta.DEMA(dataframe, timeperiod=val)
        #
        #        for val in self.buy_dema_long.range:
        #            dataframe[f'dema{val}_long'] = ta.DEMA(dataframe, timeperiod=val)

        dataframe['sptrend'] = self.supertrend(dataframe, 3, 42)['Supertrend']  # True:lowerband, False:higherband
        dataframe['td9'] = self.TD9(dataframe)  # 9:highest, -9:lowest
        pd.set_option('display.max_rows', None)

        return dataframe

    # --- Buy settings ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = []
        short_conditions = []

        #        upcross123 = qtpylib.crossed_above(dataframe['close'], dataframe['DEMA123'])
        #        upcross148 = qtpylib.crossed_above(dataframe['close'], dataframe['DEMA148'])
        #        downcross123 = qtpylib.crossed_below(dataframe['close'], dataframe['DEMA123'])
        #        downcross148 = qtpylib.crossed_below(dataframe['close'], dataframe['DEMA148'])

        upcross123 = qtpylib.crossed_above(dataframe['close'], dataframe['DEMA117'])
        upcross148 = qtpylib.crossed_above(dataframe['close'], dataframe['DEMA149'])
        downcross123 = qtpylib.crossed_below(dataframe['close'], dataframe['DEMA75'])
        downcross148 = qtpylib.crossed_below(dataframe['close'], dataframe['DEMA23'])

        buy_sigt = dataframe['sptrend']
        buy_sigf = buy_sigt.shift(axis=0, periods=1)  # False --> True
        buy_sig = (buy_sigt ^ buy_sigf) & buy_sigt

        sell_sigt = dataframe['sptrend']
        sell_sigf = sell_sigt.shift(axis=0, periods=1)  # True --> False
        sell_sig = (sell_sigt ^ sell_sigf) & sell_sigf

        upcross = upcross123 | upcross148
        downcross = downcross123 | downcross148

        long_conditions.append(upcross)
        long_conditions.append(buy_sig)
        short_conditions.append(downcross)
        short_conditions.append(sell_sig)

        if long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, long_conditions),
                'enter_long'] = 1
        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_conditions),
                'enter_short'] = 1

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        if last_candle['td9'] == 9:
            return -(trade.stake_amount * ((self.selltd9) / 100))
        elif last_candle['td9'] == 13:
            return -(trade.stake_amount * ((self.selltd13) / 100))
        elif last_candle['td9'] == 15:
            return -(trade.stake_amount * ((self.selltd15) / 100))
        elif last_candle['td9'] == 16:
            return -(trade.stake_amount * ((self.selltd16) / 100))

        if last_candle['td9'] == -9:
            return -(trade.stake_amount * ((self.selltd9) / 100))
        elif last_candle['td9'] == -13:
            return -(trade.stake_amount * ((self.selltd13) / 100))
        elif last_candle['td9'] == -15:
            return -(trade.stake_amount * ((self.selltd15) / 100))
        elif last_candle['td9'] == -16:
            return -(trade.stake_amount * ((self.selltd16) / 100))

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = []
        exit_short_conditions = []
        print('-----------------------------------------------')

        buy_sigt = dataframe['sptrend']
        buy_sigf = buy_sigt.shift(axis=0, periods=1)  # False --> True
        buy_sig = ((buy_sigt ^ buy_sigf) & buy_sigt)  # | (dataframe['td9'] == -15)

        sell_sigt = dataframe['sptrend']
        sell_sigf = sell_sigt.shift(axis=0, periods=1)  # True --> False
        sell_sig = ((sell_sigt ^ sell_sigf) & sell_sigf)  # or ntd15

        exit_long_conditions.append(sell_sig)
        exit_short_conditions.append(buy_sig)

        if exit_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, exit_long_conditions),
                'exit_long'] = 1

        if exit_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, exit_short_conditions),
                'exit_short'] = 1

        return dataframe

    def supertrend(self, df: DataFrame, multiplier, atr_period):
        high = df['high']
        low = df['low']
        close = df['close']

        price_diffs = [high - low, high - close.shift(), close.shift() - low]
        true_range = pd.concat(price_diffs, axis=1)
        true_range = true_range.abs().max(axis=1)
        atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()

        hl2 = (high + low) / 2
        final_upperband = hl2 + (multiplier * atr)
        final_lowerband = hl2 - (multiplier * atr)

        st = [True] * len(df)

        for i in range(1, len(df.index)):
            curr, prev = i, i - 1

            if close[curr] > final_upperband[prev]:
                st[curr] = True
            elif close[curr] < final_lowerband[prev]:
                st[curr] = False
            else:
                st[curr] = st[prev]
                if st[curr] and final_lowerband[curr] < final_lowerband[prev]:
                    final_lowerband[curr] = final_lowerband[prev]
                if not st[curr] and final_upperband[curr] > final_upperband[prev]:
                    final_upperband[curr] = final_upperband[prev]

            if st[curr]:
                final_upperband[curr] = np.nan
            else:
                final_lowerband[curr] = np.nan
        return pd.DataFrame({
            'Supertrend': st,
            'Lowerband': final_lowerband,
            'Upperband': final_upperband
        }, index=df.index)

    def TD9(self, df: DataFrame):
        close_np = [i for i in df['close']]
        close_shift = np.full_like(close_np, np.nan)
        close_shift[:4] = 0
        close_shift[4:] = close_np[:-4]
        compare_array = close_np > close_shift
        result = np.empty(len(close_np), int)
        counting_number: int = 0
        for i in range(len(close_np)):
            if np.isnan(close_shift[i]):
                result[i] = 0
            else:
                compare_bool = compare_array[i]
                if compare_bool:
                    if counting_number >= 0:
                        counting_number += 1
                    else:
                        counting_number = 1
                else:
                    if counting_number <= 0:
                        counting_number -= 1
                    else:
                        counting_number = -1
                result[i] = counting_number
        return result

    def ema(self, df: DataFrame, period):
        return df.ewm(alpha=1 / period, min_periods=period).mean()
    
    def rsi(self, df: DataFrame, period):
        return ta.RSI(df, timeperiod=period)

    def macd(self, df: DataFrame, fastperiod, slowperiod, signalperiod):
        return ta.MACD(df, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)


