import pandas as pd
import yfinance as yf
from finta import TA
import math


def get_data(ticker):
    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()
    tsla = yf.Ticker("{}".format(ticker_upper))
    tsla_df = tsla.history(period="max")
    #change to lowercase for use with finta
    tsla_df = tsla_df.rename(columns={"Open":"open"})
    tsla_df = tsla_df.rename(columns={"Close":"close"})
    tsla_df = tsla_df.rename(columns={"High":"high"})
    tsla_df = tsla_df.rename(columns={"Low":"low"})
    tsla_df = tsla_df.rename(columns={"Volume":"volume"})

    #make copy with all labels and indicators
    tsla_df_ind = tsla_df.copy()
    #calculate indicators
    rsi = TA.RSI(tsla_df)
    stochrsi = TA.STOCHRSI(tsla_df)
    stochk = TA.STOCH(tsla_df)
    stochd = TA.STOCHD(tsla_df)
    williamsR = TA.WILLIAMS(tsla_df)
    ultimate = TA.UO(tsla_df)
    awesome = TA.AO(tsla_df)
    macd = TA.MACD(tsla_df)
    percent_b = TA.PERCENT_B(tsla_df)
    dmi = TA.DMI(tsla_df, 14)
    cci = TA.CCI(tsla_df)
    sma10 = TA.SMA(tsla_df, 10)
    sma20 = TA.SMA(tsla_df, 20)
    #sma50 = TA.SMA(tsla_df, 50)
    #sma200 = TA.SMA(tsla_df, 200)
    ema9 = TA.EMA(tsla_df, 9)
    ema12 = TA.EMA(tsla_df, 12)
    ema26 = TA.EMA(tsla_df, 26)
    obv = TA.OBV(tsla_df)
    #add indicators to table
    #tsla_df_ind["obv"] = obv
    tsla_df_ind["rsi"] = rsi
    #tsla_df_ind["rsi hit"] = "no"
    tsla_df_ind["macd"] = macd["MACD"]
    tsla_df_ind["macd signal"] = macd["SIGNAL"]
    tsla_df_ind["macd histogram"] = macd["MACD"] - macd["SIGNAL"]
    #tsla_df_ind["macd cross"] = "no"
    tsla_df_ind["percent b"] = percent_b
    #tsla_df_ind['percent b hit'] = "no"
    tsla_df_ind["di+"] = dmi["DI+"]
    tsla_df_ind["di-"] = dmi["DI-"]
    tsla_df_ind['di cross'] = dmi["DI+"] - dmi["DI-"]
    #tsla_df_ind["dmi cross"] = "no"
    tsla_df_ind["cci"] = cci
    #tsla_df_ind["cci hit"] = "no"
    #tsla_df_ind['rays cci hit'] = 'no'
    tsla_df_ind["stoch rsi"] = stochrsi
    #tsla_df_ind["stoch rsi hit"] = "no"
    tsla_df_ind["williamsR"] = williamsR
    #tsla_df_ind["williamsR hit"] = "no"
    #tsla_df_ind['rays williamsR hit'] = "no"
    tsla_df_ind["ultimate"] = ultimate
    #tsla_df_ind["ultimate hit"] = "no"
    tsla_df_ind["awesome"] = awesome
    #tsla_df_ind["awesome hit"] = "no"
    #tsla_df_ind['rays awesome hit'] = 'no'
    #tsla_df_ind['rays awesome hit2'] = 'no'
    tsla_df_ind['stoch slow'] = stochd
    tsla_df_ind['stoch fast'] = stochk
    tsla_df_ind['stoch cross'] = stochk - stochd
    #tsla_df_ind['stoch hit'] = 'no'
    tsla_df_ind["sma10"] = sma10
    tsla_df_ind["sma20"] = sma20
    #tsla_df_ind["sma50"] = sma50
    #tsla_df_ind["sma200"] = sma200
    tsla_df_ind["ema9"] = ema9
    tsla_df_ind["ema12"] = ema12
    tsla_df_ind["ema26"] = ema26
    #add price change from close between each day
    tsla_df_ind['change'] = tsla_df_ind['close'].diff()
    tsla_df_ind['pct change'] = tsla_df_ind['close'].pct_change()

    #remove NaN values from indicator data frame
    tsla_df_ind.dropna(inplace=True)
    #remove non helpful columns that came with the table
    tsla_df_ind = tsla_df_ind.drop(columns="Dividends")
    tsla_df_ind = tsla_df_ind.drop(columns="Stock Splits")
    tsla_df_ind = tsla_df_ind.drop(columns="open")
    tsla_df_ind = tsla_df_ind.drop(columns="high")
    tsla_df_ind = tsla_df_ind.drop(columns="low")
    tsla_df_ind = tsla_df_ind.drop(columns="stoch rsi")

    #export entire dataframe to excel file
    tsla_df_ind.to_excel("excel_files/{}.xlsx".format(ticker_lower))
    return tsla_df_ind




