import pandas as pd
import numpy as np
import os
from config import DATA_FILES

def add_technical_indicators(df):
    df = df.sort_values(by=['Ticker']).reset_index(drop=True)
    
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Price_Change_Pct'] = df.groupby('Ticker')['Close'].pct_change() * 100
    df['Price_Change_Abs'] = df.groupby('Ticker')['Close'].diff().abs()
    
    df['RSI'] = df.groupby('Ticker')['Price_Change_Pct'].transform(
        lambda x: x.rolling(window=14, min_periods=1).mean()
    )
    df['RSI_Std'] = df.groupby('Ticker')['Price_Change_Pct'].transform(
        lambda x: x.rolling(window=14, min_periods=1).std()
    )
    
    df['SMA_Fast'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=12, min_periods=1).mean()
    )
    df['SMA_Slow'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=26, min_periods=1).mean()
    )
    df['MACD'] = df['SMA_Fast'] - df['SMA_Slow']
    df['MACD_Signal'] = df.groupby('Ticker')['MACD'].transform(
        lambda x: x.rolling(window=9, min_periods=1).mean()
    )
    
    df.rename(columns={'Volatility': 'Spread_HL'}, inplace=True)
    df['ATR'] = df.groupby('Ticker')['Spread_HL'].transform(
        lambda x: x.rolling(window=14, min_periods=1).mean()
    )
    df['ATR_Ratio'] = df['ATR'] / (df['Close'] + 1e-8)
    
    df['Volume_MA5'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df['Volume_MA20'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    
    df['High_Low_Ratio'] = df['High'] / (df['Low'] + 1e-8)
    df['Close_Range_Pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8) * 100
    
    df['Price_Momentum_5'] = df.groupby('Ticker')['Close'].transform(
        lambda x: (x / x.shift(5) - 1) * 100
    )
    df['Price_Momentum_20'] = df.groupby('Ticker')['Close'].transform(
        lambda x: (x / x.shift(20) - 1) * 100
    )
    
    df.drop(columns=['SMA_Fast', 'SMA_Slow'], inplace=True)
    
    impute_cols = [col for col in [
        'RSI', 'RSI_Std', 'MACD', 'MACD_Signal', 'ATR', 'ATR_Ratio',
        'Volume_MA5', 'Volume_MA20', 'Price_Range',
        'Price_Change_Pct', 'Price_Change_Abs', 'High_Low_Ratio',
        'Close_Range_Pct', 'Price_Momentum_5', 'Price_Momentum_20'
    ] if col in df.columns]
    
    print(f"Технические индикаторы добавлены. {len(impute_cols)} признаков")
    return df

# def add_time_features(df):
#     df['Date'] = pd.to_datetime(df['Date'])
#     df['DayOfWeek'] = df['Date'].dt.dayofweek
#     df['Month'] = df['Date'].dt.month
#     df['Quarter'] = df['Date'].dt.quarter
    
#     df['DayOfWeek'] = df['DayOfWeek'].astype('category')
#     df['Month'] = df['Month'].astype('category')
#     df['Quarter'] = df['Quarter'].astype('category')
    
#     return df

def add_text_features(df):
    positive_words = ['good', 'great', 'excellent', 'best', 'positive', 'gain', 'profit', 
                      'up', 'bullish', 'strong', 'buy', 'rally', 'surge', 'boom', 'rise']
    negative_words = ['bad', 'poor', 'worst', 'negative', 'loss', 'down', 'bearish', 
                      'fall', 'decline', 'weak', 'sell', 'crash', 'plunge', 'slump', 'dump']
    neutral_words = ['neutral', 'stable', 'flat', 'sideways', 'consolidation', 'range', 'hold']
    
    def count_sentiment(text, word_list):
        if pd.isna(text) or text == '':
            return 1
        text_lower = str(text).lower()
        count = sum(1 for word in word_list if word in text_lower)
        return max(1, count)
    
    text_col = 'Combined_Tweets'
    if text_col not in df.columns:
        text_col = [col for col in df.columns if 'tweet' in col.lower()][0] if any('tweet' in col.lower() for col in df.columns) else 'Text_Feature'
    
    df['Sentiment_Positive'] = df[text_col].apply(lambda x: count_sentiment(x, positive_words))
    df['Sentiment_Negative'] = df[text_col].apply(lambda x: count_sentiment(x, negative_words))
    df['Sentiment_Neutral'] = df[text_col].apply(lambda x: count_sentiment(x, neutral_words))
    
    df['Text_Length'] = df[text_col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 1)
    median_length = df[df['Text_Length'] > 0]['Text_Length'].median()
    df['Text_Length'] = df['Text_Length'].replace(0, median_length if pd.notna(median_length) else 10)
    
    df['Sentiment_Score'] = (df['Sentiment_Positive'] - df['Sentiment_Negative']) / (df['Sentiment_Positive'] + df['Sentiment_Negative'] + 1)
    
    df['Sentiment_Ratio'] = df['Sentiment_Positive'] / (df['Sentiment_Negative'] + 1)
    
    df['Sentiment_Total'] = df['Sentiment_Positive'] + df['Sentiment_Negative'] + df['Sentiment_Neutral']
    
    df['Text_Feature'] = df[text_col]
    
    text_features_created = ['Sentiment_Positive', 'Sentiment_Negative', 'Sentiment_Neutral',
                             'Text_Length', 'Sentiment_Score', 'Sentiment_Ratio', 'Sentiment_Total']
    print(f"Текстовые признаки добавлены: {text_features_created}")
    return df

def main():
    print("=" * 60)
    print("ДОБАВЛЕНИЕ ПРИЗНАКОВ")
    print("=" * 60)
    
    df = pd.read_csv(DATA_FILES["raw"])
    print(f"Загружено {len(df)} строк")
    
    df = add_technical_indicators(df)
    # df = add_time_features(df)
    df = add_text_features(df)
    
    print(f"\nФинальный размер датасета: {len(df)} строк")
    print("Колонки:")
    print(df.columns.tolist())
    
    df.to_csv(DATA_FILES["features"], index=False)
    print(f"\nПризнаки сохранены в {DATA_FILES["features"]}")

if __name__ == "__main__":
    main()
