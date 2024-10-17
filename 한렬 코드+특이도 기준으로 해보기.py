import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from optuna.samplers import TPESampler
import optuna


# 데이터 로드
df = pd.read_csv("../data/data_week3.csv")

df['unknown2'].value_counts().sort_index(ascending=False) # 294451, 1
df = df[df['unknown2'] != 294451]

df['unknown5'].value_counts().sort_index(ascending=False) # 27, 1
df = df[df['unknown2'] != 294451]

df['unknown6'].value_counts().sort_index(ascending=False) # 2398, 1
df = df[df['unknown6'] != 2398]

df['unknown8'].value_counts().sort_index(ascending=False) # 31706, 1
df = df[df['unknown8'] != 31706]

df['unknown10'].value_counts().sort_index(ascending=False) # 877, 1
df = df[df['unknown10'] != 877]

df['unknown14'].value_counts().sort_index(ascending=False) # 403, 1
df = df[df['unknown14'] != 403]

df['unknown16'].value_counts().sort_index(ascending=False) # 2840.5, 1
df = df[df['unknown16'] != 2840.5]



# 범주형 변수를 One-Hot 인코딩 (get_dummies)
df_encoded = pd.get_dummies(df, columns=['unknown1'])

# 피처 선택 (인코딩된 피처 사용)
features = [col for col in df_encoded.columns if col != 'target']  # target을 제외한 모든 피처 사용
X = df_encoded[features]
y = df_encoded['target']

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 20%, stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 1. 랜덤 오버샘플링 적용
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# LightGBM용 Optuna objective 함수
def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }

    model = LGBMClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

# LightGBM 최적화 실행
sampler = TPESampler(seed=42)
lgbm_study = optuna.create_study(direction='maximize', sampler=sampler)
lgbm_study.optimize(lgbm_objective, n_trials=50)

# 최적 하이퍼파라미터와 성능 출력
print("Best LightGBM parameters:", lgbm_study.best_params)
print("Best LightGBM recall:", lgbm_study.best_value)

# 최적 파라미터를 사용해 최적 모델 생성
best_lgbm_model = LGBMClassifier(**lgbm_study.best_params)
best_lgbm_model.fit(X_train_resampled, y_train_resampled)


# RandomForest 최적화 함수
def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'random_state': 42
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

# RandomForest 최적화 실행
rf_study = optuna.create_study(direction='maximize', sampler=sampler)
rf_study.optimize(rf_objective, n_trials=50)

# 최적 파라미터를 사용해 최적 모델 생성
best_rf_model = RandomForestClassifier(**rf_study.best_params)
best_rf_model.fit(X_train_resampled, y_train_resampled)


# XGBoost 최적화 함수
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }

    model = XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

# XGBoost 최적화 실행
xgb_study = optuna.create_study(direction='maximize', sampler=sampler)
xgb_study.optimize(xgb_objective, n_trials=50)

# 최적 파라미터를 사용해 최적 모델 생성
best_xgb_model = XGBClassifier(**xgb_study.best_params)
best_xgb_model.fit(X_train_resampled, y_train_resampled)


# CatBoost 최적화 함수
def cat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 200),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'random_state': 42
    }

    model = CatBoostClassifier(**params, verbose=0)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

# CatBoost 최적화 실행
cat_study = optuna.create_study(direction='maximize', sampler=sampler)
cat_study.optimize(cat_objective, n_trials=50)

# 최적 파라미터를 사용해 최적 모델 생성
best_cat_model = CatBoostClassifier(**cat_study.best_params, verbose=0)
best_cat_model.fit(X_train_resampled, y_train_resampled)


# 최적 파라미터를 사용해 최적 모델 생성
best_lgbm_model = LGBMClassifier(**lgbm_study.best_params, random_state=42)
best_rf_model = RandomForestClassifier(**rf_study.best_params, random_state=42)
best_xgb_model = XGBClassifier(**xgb_study.best_params, random_state=42)
best_cat_model = CatBoostClassifier(**cat_study.best_params, random_state=42, verbose=0)

# 모델 목록과 모델 이름 설정 (최적화된 모델 포함)
models = [
    best_lgbm_model,
    best_rf_model,
    best_xgb_model,
    best_cat_model,
    LGBMClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    XGBClassifier(random_state=42),
    CatBoostClassifier(random_state=42, verbose=0),
    LogisticRegression(random_state=42, max_iter=1000)
]
model_names = ["Optimized LightGBM", "Optimized RandomForest", "Optimized XGBoost", 
               "Optimized CatBoost",
               "LightGBM", "RandomForest", "XGBoost", "CatBoost", "Logistic Regression"]

# 각 모델 학습 (오버샘플링된 데이터로)
for model in models:
    model.fit(X_train_resampled, y_train_resampled)

# 특이도와 재현율 계산 함수
def specificity_recall_thresholds(model, X_test, y_test):
    # 확률 예측
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.0, 1.05, 0.05)  # 임계값 범위 설정
    specificity_scores = []
    recall_scores = []

    for threshold in thresholds:
        # 임계값 적용하여 예측
        y_pred_threshold = (y_proba >= threshold).astype(int)
        
        # 혼동 행렬로 특이도(Specificity)와 재현율(Recall) 계산
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        specificity_scores.append(specificity)
        recall_scores.append(recall)

    return thresholds, specificity_scores, recall_scores

# 특이도와 재현율을 그리는 함수 확장
def plot_specificity_recall_curve(models, model_names, X_test, y_test):
    for model, name in zip(models, model_names):
        thresholds, specificity_scores, recall_scores = specificity_recall_thresholds(model, X_test, y_test)

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, specificity_scores, label="Specificity")
        plt.plot(thresholds, recall_scores, label="Recall")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.legend()
        plt.title(f"Specificity and Recall vs Threshold ({name})")
        plt.grid(True)
        plt.show()

# 각 모델에 대한 Specificity-Recall vs Threshold 그래프 그리기
plot_specificity_recall_curve(models, model_names, X_test, y_test)

# 특이도 0.9 이상에서 최대 재현율을 찾는 함수
def find_best_model_for_specificity(models, model_names, X_test, y_test, min_specificity=0.9):
    best_model_name = None
    best_recall = -1
    best_threshold = -1
    best_specificity = -1
    
    for model, name in zip(models, model_names):
        thresholds, specificity_scores, recall_scores = specificity_recall_thresholds(model, X_test, y_test)
        
        for i, specificity in enumerate(specificity_scores):
            if specificity >= min_specificity:  # 특이도 조건을 만족하는 경우
                if recall_scores[i] > best_recall:  # 최대 재현율을 가진 모델 찾기
                    best_model_name = name
                    best_recall = recall_scores[i]
                    best_threshold = thresholds[i]
                    best_specificity = specificity

    return best_model_name, best_recall, best_specificity, best_threshold

# 특이도 0.9 이상에서 가장 높은 재현율을 가진 모델 찾기
best_model_name, best_recall, best_specificity, best_threshold = find_best_model_for_specificity(
    models, model_names, X_test, y_test, min_specificity=0.9
)

# 결과 출력
print(f"Best model: {best_model_name}")
print(f"Best recall: {best_recall}")
print(f"Specificity: {best_specificity}")
print(f"Threshold: {best_threshold}")

# 베스트 모델로 혼동 행렬을 그리는 함수
def plot_best_model_confusion_matrix(best_model, X_test, y_test, best_threshold):
    # 베스트 모델로 확률 예측
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # 베스트 임계값을 사용하여 최종 예측 생성
    y_pred_best_threshold = (y_proba >= best_threshold).astype(int)
    
    # 혼동 행렬 계산
    conf_matrix_best = confusion_matrix(y_test, y_pred_best_threshold)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Best Model: {best_model_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# 베스트 모델로 혼동 행렬 그리기
best_model_index = model_names.index(best_model_name)
best_model = models[best_model_index]

plot_best_model_confusion_matrix(best_model, X_test, y_test, best_threshold)


# 모든 모델의 특이도, 재현율, 임계값을 한번에 출력하는 함수
def get_model_specificity_recall_thresholds(models, model_names, X_test, y_test, min_specificity=0.9):
    results = []  # 결과 저장 리스트

    for model, name in zip(models, model_names):
        thresholds, specificity_scores, recall_scores = specificity_recall_thresholds(model, X_test, y_test)
        
        best_recall = -1
        best_threshold = -1
        best_specificity = -1
        
        for i, specificity in enumerate(specificity_scores):
            if specificity >= min_specificity:
                if recall_scores[i] > best_recall:
                    best_recall = recall_scores[i]
                    best_specificity = specificity_scores[i]
                    best_threshold = thresholds[i]

        # 결과를 리스트에 추가
        results.append({
            'Model': name,
            'Specificity': best_specificity,
            'Recall': best_recall,
            'Threshold': best_threshold
        })

    return results

# 모든 모델의 특이도, 재현율, 임계값을 한번에 출력
model_results = get_model_specificity_recall_thresholds(models, model_names, X_test, y_test, min_specificity=0.9)

# 결과 출력
for result in model_results:
    print(f"Model: {result['Model']}")
    print(f"Specificity: {result['Specificity']}")
    print(f"Recall: {result['Recall']}")
    print(f"Threshold: {result['Threshold']}")
    print("-" * 40)
