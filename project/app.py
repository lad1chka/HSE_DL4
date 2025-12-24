import gradio as gr
import pandas as pd
import numpy as np
import os
import shap
from catboost import CatBoostClassifier
from config import MODEL_PATHS, ARTIFACTS, STOCKS, DATA_FILES

model = None
feature_importance_df = None
data_df = None
explainer = None

try:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATHS["catboost"])
    print("Модель CatBoost успешно загружена.")
    
    if os.path.exists(ARTIFACTS["feature_importance_catboost"]):
        feature_importance_df = pd.read_csv(ARTIFACTS["feature_importance_catboost"])
    
    if os.path.exists(DATA_FILES["features"]):
        data_df = pd.read_csv(DATA_FILES["features"])
        STOCKS = sorted(data_df["Ticker"].unique().tolist())
    
    if model is not None:
        try:
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            print(f"Error: {e}")
    
except Exception as e:
    print(f"Error: {e}")

def predict_volatility(ticker):
    if data_df is not None:
        try:
            ticker_data = data_df[data_df["Ticker"] == ticker]
            if ticker_data.empty:
                return (f"Нет данных для тикера {ticker}", "N/A", "N/A")
            
            last_row = ticker_data.iloc[-1]
            
            numerical_features = [
                "Spread_HL", "RSI", "MACD", "Volume_MA5",
                "Price_Range", "Price_Change_Pct", "Price_Change_Abs", "RSI_Std", "MACD_Signal",
                "Volume_MA20", "High_Low_Ratio", "Close_Range_Pct",
                "Price_Momentum_5", "Price_Momentum_20",
                "Sentiment_Positive", "Sentiment_Negative", "Sentiment_Neutral",
                "Text_Length", "Sentiment_Score", "Sentiment_Ratio", "Sentiment_Total"
            ]
            categorical_features = ["Ticker"]
            text_features = ["Text_Feature"]
            
            all_features = numerical_features + categorical_features + text_features
            available_features = [col for col in all_features if col in last_row.index]
            
            X_pred = pd.DataFrame([last_row[available_features]])
            
            for col in categorical_features:
                if col in X_pred.columns:
                    X_pred[col] = X_pred[col].astype(str).fillna('missing')
            
            for col in numerical_features:
                if col in X_pred.columns:
                    median_val = data_df[col].median()
                    fill_val = median_val if pd.notna(median_val) else 0
                    X_pred[col] = X_pred[col].fillna(fill_val)
            
            for col in text_features:
                if col in X_pred.columns:
                    X_pred[col] = X_pred[col].astype(str).fillna('')
            
            prediction = model.predict(X_pred)[0]
            probabilities = model.predict_proba(X_pred)[0]
            
            pred_text = "ВЫСОКАЯ волатильность" if prediction == 1 else "НИЗКАЯ волатильность"
            
            return (
                pred_text,
                f"{probabilities[0]*100:.2f}%",
                f"{probabilities[1]*100:.2f}%"
            )
            
        except Exception as e:
            return (f"Error: {str(e)}", "N/A", "N/A")
    
    return ("Error: нет данных", "N/A", "N/A")

def show_feature_importance():
    if feature_importance_df is not None and not feature_importance_df.empty:
        top_features = feature_importance_df.head(10)
        return top_features.to_string(index=False)
    return "error"

with gr.Blocks(title="Прогнозирование волатильности") as interface:
    gr.Markdown("# Система прогнозирования волатильности акций")
    gr.Markdown("Многомодальное машинное обучение для предсказания волатильности")
    
    with gr.Row():
        with gr.Column():
            ticker_input = gr.Dropdown(
                choices=STOCKS,
                label="Выберите тикер",
                value=STOCKS[0] if STOCKS else "AAPL"
            )
            
            predict_btn = gr.Button("Прогнозировать", size="lg")
        
        with gr.Column():
            prediction_output = gr.Textbox(
                label="Прогноз",
                interactive=False
            )
            prob_low = gr.Textbox(
                label="Вероятность низкой волатильности",
                interactive=False
            )
            prob_high = gr.Textbox(
                label="Вероятность высокой волатильности",
                interactive=False
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Важность признаков")
            importance_output = gr.Textbox(
                label="Feature Importance",
                interactive=False,
                lines=10
            )
            
            importance_btn = gr.Button("Показать важность признаков")
    
    def on_predict(ticker):
        return predict_volatility(ticker)
    
    predict_btn.click(
        on_predict,
        inputs=[ticker_input],
        outputs=[prediction_output, prob_low, prob_high]
    )
    
    importance_btn.click(
        show_feature_importance,
        outputs=importance_output
    )

if __name__ == "__main__":
    interface.launch()
