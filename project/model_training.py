import pandas as pd
import numpy as np
import os
import shap
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from config import (
    TARGET_COLUMN, MODEL_DIR, MODEL_PATHS, ARTIFACTS, DATA_FILES,
    CATBOOST_PARAMS, TABNET_PARAMS, TABNET_FIT_PARAMS, TRAIN_TEST_SPLIT
)

def prepare_data(df):
    df = df.sort_values(by="Date").reset_index(drop=True)
    
    train_size = int(TRAIN_TEST_SPLIT * len(df))
    
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    print(f"Размер обучающей выборки: {len(df_train)}")
    print(f"Размер тестовой выборки: {len(df_test)}")
    
    return df_train, df_test

def define_features(df):
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
    
    return numerical_features, categorical_features, text_features

def train_catboost(df_train, df_test, numerical_features, categorical_features, text_features):
    print("\n" + "=" * 60)
    print("CATBOOST")
    print("=" * 60)
    
    X_train = df_train[numerical_features + categorical_features + text_features].copy()
    y_train = df_train[TARGET_COLUMN]
    X_test = df_test[numerical_features + categorical_features + text_features].copy()
    y_test = df_test[TARGET_COLUMN]
    
    for col in categorical_features:
        X_train[col] = X_train[col].astype(str).fillna('missing')
        X_test[col] = X_test[col].astype(str).fillna('missing')
    
    for col in numerical_features:
        median_val = X_train[col].median()
        fill_val = median_val if pd.notna(median_val) else 0
        X_train[col] = X_train[col].fillna(fill_val)
        X_test[col] = X_test[col].fillna(fill_val)
    
    for col in text_features:
        X_train[col] = X_train[col].astype(str).fillna('')
        X_test[col] = X_test[col].astype(str).fillna('')
    
    cat_features = categorical_features + text_features
    
    print(f"\nРазмер признаков:")
    print(f"  - Числовые: {len(numerical_features)}")
    print(f"  - Категориальные: {len(categorical_features)}")
    print(f"  - Текстовые: {len(text_features)}")
    print(f"  - ИТОГО: {len(numerical_features + categorical_features + text_features)}")
    
    model = CatBoostClassifier(
        iterations=CATBOOST_PARAMS.get('iterations', 200),
        learning_rate=CATBOOST_PARAMS.get('learning_rate', 0.05),
        loss_function=CATBOOST_PARAMS.get('loss_function', 'Logloss'),
        random_seed=CATBOOST_PARAMS.get('random_seed', 42),
        verbose=CATBOOST_PARAMS.get('verbose', 10),
        cat_features=cat_features,
        early_stopping_rounds=CATBOOST_PARAMS.get('early_stopping_rounds', 30),
        depth=CATBOOST_PARAMS.get('depth', 6),
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test)
    )
    
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_test, y_pred)
    print(f"\nТочность на тестовой выборке: {acc:.4f} ({acc*100:.2f}%)")
    print("\nclassification_report:")
    print(classification_report(y_test, y_pred))
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model.save_model(MODEL_PATHS["catboost"])
    print(f"Модель сохранена: {MODEL_PATHS["catboost"]}")
    
    feature_importance = pd.DataFrame({
        "Feature": numerical_features + categorical_features + text_features,
        "Importance": model.get_feature_importance()
    }).sort_values(by="Importance", ascending=False)
    
    feature_importance.to_csv(ARTIFACTS["feature_importance_catboost"], index=False)
    print(f"Feature Importance сохранена: {ARTIFACTS["feature_importance_catboost"]}")
    print("\nТоп-10 признаков по важности:")
    print(feature_importance.head(10).to_string(index=False))
    
    try:
        print("\nРасчет SHAP значений...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        if isinstance(shap_values, list):
            shap_df = pd.DataFrame(shap_values[1], columns=X_test.columns)
        else:
            shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
        
        shap_df.to_csv(ARTIFACTS["shap_values"], index=False)
        print(f"SHAP значения сохранены: {ARTIFACTS["shap_values"]}")
    except Exception as e:
        print(f"Не удалось рассчитать SHAP: {e}")
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nТочность CatBoost:")
    print(f"  - На обучающей выборке: {train_score:.4f}")
    print(f"  - На тестовой выборке: {test_score:.4f}")
    
    return model, test_score

def train_tabnet(df_train, df_test, numerical_features, categorical_features):
    print("\n" + "=" * 60)
    print("TABNET")
    print("=" * 60)
    
    try:
        tabnet_features = numerical_features + categorical_features
        
        X_train = df_train[tabnet_features].copy()
        y_train = df_train[TARGET_COLUMN].values
        X_test = df_test[tabnet_features].copy()
        y_test = df_test[TARGET_COLUMN].values
        
        label_encoders = {}
        
        for col in categorical_features:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
        
        for col in numerical_features:
            median_val = X_train[col].median()
            fill_val = median_val if pd.notna(median_val) else 0
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col] = X_test[col].fillna(fill_val)
        
        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)

        print(f"\nРазмер признаков:")
        print(f"  - Числовые: {len(numerical_features)}")
        print(f"  - Категориальные: {len(categorical_features)}")
        print(f"  - Всего: {len(tabnet_features)}")
        
        model = TabNetClassifier(
            n_steps=TABNET_PARAMS.get('n_steps', 5),
            n_independent=TABNET_PARAMS.get('n_independent', 3),
            n_shared=TABNET_PARAMS.get('n_shared', 2),
            lambda_sparse=TABNET_PARAMS.get('lambda_sparse', 1e-4),
            gamma=TABNET_PARAMS.get('gamma', 1.8),
            momentum=TABNET_PARAMS.get('momentum', 0.95),
            device_name=TABNET_PARAMS.get('device_name', 'cpu'),
            seed=TABNET_PARAMS.get('seed', 42)
        )
        
        model.fit(
            X_train_np, y_train,
            eval_set=[(X_test_np, y_test)],
            eval_metric=["auc"],
            max_epochs=TABNET_FIT_PARAMS.get('max_epochs', 100),
            patience=TABNET_FIT_PARAMS.get('patience', 15),
            batch_size=TABNET_FIT_PARAMS.get('batch_size', 128),
            num_workers=TABNET_FIT_PARAMS.get('num_workers', 0)
        )
        
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        model.save_model(MODEL_PATHS["tabnet"])
        print(f"Модель сохранена: {MODEL_PATHS["tabnet"]}")
        
        from sklearn.metrics import accuracy_score, classification_report
        test_pred = model.predict(X_test_np)
        test_score = accuracy_score(y_test, test_pred)
        
        print(f"\nТочность TabNet:")
        print(f"  - На тестовой выборке: {test_score:.4f} ({test_score*100:.2f}%)")
        print("\nКлассификационный отчет:")
        print(classification_report(y_test, test_pred))
        
        feature_importance_tabnet = pd.DataFrame({
            "Feature": tabnet_features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        
        tabnet_importance_path = os.path.join(MODEL_DIR, "tabnet_feature_importance.csv")
        feature_importance_tabnet.to_csv(tabnet_importance_path, index=False)
        print(f"✓ TabNet Feature Importance сохранена: {tabnet_importance_path}")
        print("\nТоп-10 признаков по важности:")
        print(feature_importance_tabnet.head(10).to_string(index=False))
        
        return model, test_score
        
    except Exception as e:
        print(f"Ошибка при обучении TabNet: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
    print("=" * 60)    
    
    df = pd.read_csv(DATA_FILES["features"])
    print(f"Загружено {len(df)} строк")
    
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    
    df_train, df_test = prepare_data(df)
    
    numerical_features, categorical_features, text_features = define_features(df)
    
    results = {}
    
    catboost_model, catboost_score = train_catboost(
        df_train, df_test, 
        numerical_features, categorical_features, text_features
    )
    results["CatBoost"] = catboost_score
    
    tabnet_model, tabnet_score = train_tabnet(
        df_train, df_test, 
        numerical_features, categorical_features
    )
    results["TabNet"] = tabnet_score
    
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    
    print("\nРезультаты на тестовой выборке:")
    for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name:12} - Accuracy: {score:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\nЛучшая модель: {best_model[0]} с точностью {best_model[1]:.4f}")

if __name__ == "__main__":
    main()
