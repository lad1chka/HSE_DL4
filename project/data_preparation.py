import pandas as pd
import numpy as np
import os
import json
from glob import glob
from config import DATASET_PATH, STOCKS, DATA_FILES

def load_price_data(stock):
    price_file = os.path.join(DATASET_PATH, "price", "preprocessed", f"{stock}.txt")
    
    try:
        df = pd.read_csv(
            price_file, 
            sep='\t', 
            header=None, 
            skiprows=1,
            names=['Index', 'Date', 'Movement_Pct', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        df = df.drop(columns=['Index'])
        df['Date'] = pd.to_datetime(df['Date'])
        df['Ticker'] = stock
        print(f"Загружено {len(df)} ценовых записей для {stock}")
        return df
    except Exception as e:
        print(f"Ошибка загрузки цен для {stock}: {e}")
        return pd.DataFrame()

def load_tweet_data(stock):
    tweet_dir = os.path.join(DATASET_PATH, "tweet", "preprocessed", stock)
    all_tweets = []
    
    if not os.path.exists(tweet_dir):
        print(f"Директория твитов не найдена для {stock}: {tweet_dir}")
        return pd.DataFrame()

    for date_file in glob(os.path.join(tweet_dir, "*")):
        try:
            with open(date_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        tweet_data = json.loads(line)
                        tweet_data['text'] = ' '.join(tweet_data['text'])
                        tweet_data['Ticker'] = stock
                        all_tweets.append(tweet_data)
        except Exception as e:
            print(f"Ошибка чтения твитов из {date_file}: {e}")
            continue
    return pd.DataFrame(all_tweets)

def calculate_volatility_target(df):
    df['Volatility'] = df['High'] - df['Low']
    df['Next_Volatility'] = df.groupby('Ticker')['Volatility'].shift(-1)
    df['Target_Volatility'] = (df['Next_Volatility'] > df['Next_Volatility'].median()).astype(int)
    return df

def main():
    print("=" * 60)
    print("ПОДГОТОВКА ДАННЫХ")
    print("=" * 60)
    
    all_price_data = []
    all_tweet_data = []
    
    for stock in STOCKS:
        price_df = load_price_data(stock)
        all_price_data.append(price_df)
            
        tweet_df = load_tweet_data(stock)
        all_tweet_data.append(tweet_df)

    if not all_price_data:
        print("Ошибка: финансовые данные не загружены.")
        return

    df_prices = pd.concat(all_price_data, ignore_index=True)
    df_prices = calculate_volatility_target(df_prices.copy())
    
    df_tweets = pd.concat(all_tweet_data, ignore_index=True)
    df_tweets = df_tweets.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_prices = len(df_prices)
    n_tweets = len(df_tweets)
    
    if n_tweets < n_prices:
        repeat_factor = (n_prices // n_tweets) + 1
        df_tweets_repeated = pd.concat([df_tweets] * repeat_factor, ignore_index=True)
        df_tweets = df_tweets_repeated.head(n_prices)
    elif n_tweets > n_prices:
        df_tweets = df_tweets.head(n_prices)
    
    df_prices = df_prices.reset_index(drop=True)
    df_tweets = df_tweets.reset_index(drop=True)
    
    final_df = df_prices.copy()
    final_df['Tweet_Text'] = df_tweets['text']
    final_df['Tweet_ID'] = df_tweets.index
    
    print(f"\nРазмер финального датасета: {len(final_df)} строк")
    final_df.to_csv(DATA_FILES["raw"], index=False)
    print(f"\nДанные сохранены в {DATA_FILES['raw']}")

if __name__ == "__main__":
    main()