import os

DATASET_PATH = "stocknet-dataset"
STOCKS = ["AAPL", "GOOG", "MSFT"]

DATA_FILES = {
    "raw": "multimodal_volatility_data.csv",
    "features": "final_features_data.csv"
}

MODEL_DIR = "models"
MODEL_PATHS = {
    "catboost": os.path.join(MODEL_DIR, "catboost_model.cbm"),
    "tabnet": os.path.join(MODEL_DIR, "tabnet_model"),
}

ARTIFACTS = {
    "feature_importance_catboost": os.path.join(MODEL_DIR, "feature_importance.csv"),
    "feature_importance_tabnet": os.path.join(MODEL_DIR, "tabnet_feature_importance.csv"),
    "shap_values": os.path.join(MODEL_DIR, "shap_values.csv"),
}

TARGET_COLUMN = "Target_Volatility"
TEXT_COLUMN = "Combined_Tweets"

NUMERICAL_FEATURES = [
    'Spread_HL', 
    'RSI', 
    'MACD', 
    'ATR', 
    'Volume_MA5',
    'Volume_Change',
    'Price_Range',
    'Price_Change_Pct',
    
    'Week_Avg_Close',
    'Week_Std_Close',
    'Week_Max_High',
    'Week_Min_Low',
    'Week_Avg_Volume',
    'Week_Avg_Volatility',
    'Week_Std_Volatility',
    'Week_Price_Trend',
    'Week_Volatility_Trend',
    'Week_Data_Completeness',
    
    'Sentiment_Positive', 
    'Sentiment_Negative', 
    'Sentiment_Neutral',
    'Sentiment_Score',
    'Text_Length',
    'Weekly_Tweet_Count',
]

LAG_FEATURES = []
for lag in range(1, 8):
    LAG_FEATURES.extend([
        f'Close_Lag{lag}',
        f'High_Lag{lag}',
        f'Low_Lag{lag}',
        f'Open_Lag{lag}',
        f'Volume_Lag{lag}',
        f'Volatility_Lag{lag}',
        f'Movement_Lag{lag}'
    ])

CATEGORICAL_FEATURES = [
    'Ticker', 
    'DayOfWeek', 
    'Month', 
    'Quarter'
]

TEXT_FEATURES = ['Text_Feature']

POSITIVE_WORDS = [
    'good', 'great', 'excellent', 'best', 'positive', 'gain', 'profit', 
    'up', 'bullish', 'strong', 'buy', 'rally', 'surge', 'rise', 'boom'
]

NEGATIVE_WORDS = [
    'bad', 'poor', 'worst', 'negative', 'loss', 'down', 'bearish', 'fall', 
    'decline', 'weak', 'sell', 'crash', 'plunge', 'dump', 'slump'
]

NEUTRAL_WORDS = [
    'neutral', 'stable', 'flat', 'sideways', 'consolidation', 'range'
]

CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_strength": 1,
    "bagging_temperature": 0.5,
    "border_count": 128,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": 100,
    "early_stopping_rounds": 50,
    "use_best_model": True,
    "random_seed": 42,
    "subsample": 0.8,
    "colsample_bylevel": 0.8,
}

TABNET_PARAMS = {
    "n_steps": 5,
    "n_independent": 3,
    "n_shared": 2,
    "lambda_sparse": 1e-4,
    "gamma": 1.8,
    "momentum": 0.95,
    "device_name": "cpu",
    "seed": 42,
}

TABNET_FIT_PARAMS = {
    "max_epochs": 15,
    "patience": 15,
    "batch_size": 128,
    "num_workers": 2,
}

TRAIN_TEST_SPLIT = 0.8

RANDOM_STATE = 42
