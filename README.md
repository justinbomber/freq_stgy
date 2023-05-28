# justinstrategy

This is a strategy implemented in Freqtrade, a cryptocurrency trading bot. The strategy is based on various indicators and conditions for entering and exiting trades.

## Installation

To use this strategy, you need to have Freqtrade installed. Please refer to the [Freqtrade documentation](https://www.freqtrade.io/en/latest/) for installation instructions.

## Strategy Overview

The `justinstrategy` is a trend-following strategy that uses several indicators and signals to determine entry and exit points for trades. It incorporates the following features:

- Timeframe: 1 day
- Startup candle count: 25
- Minimal ROI: 100
- Stop loss: -0.10
- Trailing stop: False
- Position adjustment enabled: True
- Use exit signal: True
- Exit profit only: False
- Exit profit offset: 0.0
- Ignore ROI if entry signal: False
- Can short: True
- Sell thresholds for TD9: 0, 15, 30, 0

## Indicators

The strategy uses the following indicators:

- DEMA123: Double Exponential Moving Average with a time period of 123
- DEMA148: Double Exponential Moving Average with a time period of 148
- DEMA117: Double Exponential Moving Average with a time period of 117
- DEMA149: Double Exponential Moving Average with a time period of 149
- DEMA75: Double Exponential Moving Average with a time period of 75
- DEMA23: Double Exponential Moving Average with a time period of 23
- Supertrend: Supertrend indicator with a multiplier of 3 and an ATR period of 42
- TD9: TD Sequential indicator

## Entry and Exit Conditions

The strategy has the following entry and exit conditions:

- Entry Long:
  - Cross above DEMA117 or DEMA149
  - Supertrend change from False to True

- Entry Short:
  - Cross below DEMA75 or DEMA23
  - Supertrend change from True to False

- Exit Long:
  - Supertrend change from True to False

- Exit Short:
  - Supertrend change from False to True

## Adjusting Trade Position

The strategy adjusts the trade position based on the TD9 indicator. If the TD9 value is 9, 13, 15, or 16, the trade position is adjusted with a sell threshold percentage.

## Usage

To use this strategy, you need to configure it in your Freqtrade configuration file (`config.json`). Add the following lines under the `strategies` section:

```json
"strategies": {
  "justinstrategy": {
    "minimal_roi": {
      "0": 100
    },
    "stoploss": -0.10,
    "trailing_stop": false,
    "position_adjustment_enable": true,
    "use_exit_signal": true,
    "exit_profit_only": false,
    "exit_profit_offset": 0.0,
    "ignore_roi_if_entry_signal": false,
    "can_short": true,
    "selltd9": 0,
    "selltd13": 15,
    "selltd15": 30,
    "selltd16": 0
  }
}
```



You can customize the parameters according to your preferences.

## Disclaimer

This strategy is provided for educational and informational purposes only. Use it at your own risk. The author and OpenAI do not guarantee any profits or outcomes from using this strategy. Cryptocurrency trading involves risk
