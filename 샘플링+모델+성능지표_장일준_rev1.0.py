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

# 데이터 로드
df = pd.read_csv("../data/data_week3.csv")

# 범주형 변수를 One-Hot 인코딩 (get_dummies)
df_encoded = pd.get_dummies(df, columns=['unknown1'])

# X,y 나누기
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 20%, stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Before Sampling: {Counter(y_train)}")
# {0: 10313, 1: 962}

# 모델 및 샘플링 하기
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from collections import Counter # 샘플링 갯수 결과 보기 위해 필요한 모듈

### 0. 기본 데이터(샘플링 안 함)
# Original Data
X_train
y_train


## Logistic Regression (로지스틱 회귀 분석)
model_Original_LogisticRegression = LogisticRegression()
model_Original_LogisticRegression.fit(X_train, y_train)

y_pred_Original_LogisticRegression = model_Original_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_Original_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_Original_DecisionTreeClassifier.fit(X_train, y_train)

y_pred_Original_DecisionTreeClassifier = model_Original_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_Original_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_Original_RandomForest.fit(X_train, y_train)

y_pred_Original_RandomForest = model_Original_RandomForest.predict(X_test)

## XG Boost
model_Original_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_Original_XGBoost.fit(X_train, y_train)

y_pred_Original_XGBoost = model_Original_XGBoost.predict(X_test)

### Light GBM
model_Original_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0) # verbose=0 하면 셀 실행시 결과 보여주는 창에 글자들 없어짐.
model_Original_LightGBM.fit(X_train, y_train)

y_pred_Original_LightGBM = model_Original_LightGBM.predict(X_test)

### Cat Boost
model_Original_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_Original_CatBoost.fit(X_train, y_train)

y_pred_Original_CatBoost = model_Original_CatBoost.predict(X_test)



### 1. 토멕링크
# Tomek Links
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(sampling_strategy='majority')
X_train_TomekLinks, y_train_TomekLinks = tl.fit_resample(X_train,y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, Tomek Links: {Counter(y_train_TomekLinks)}")
# {0: 9911, 1: 962}


## Logistic Regression (로지스틱 회귀 분석)
model_Tomek_LogisticRegression = LogisticRegression()
model_Tomek_LogisticRegression.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_LogisticRegression = model_Tomek_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_Tomek_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_Tomek_DecisionTreeClassifier.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_DecisionTreeClassifier = model_Tomek_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_Tomek_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_Tomek_RandomForest.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_RandomForest = model_Tomek_RandomForest.predict(X_test)

## XG Boost
model_Tomek_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_Tomek_XGBoost.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_XGBoost = model_Tomek_XGBoost.predict(X_test)

### Light GBM
model_Tomek_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0)
model_Tomek_LightGBM.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_LightGBM = model_Tomek_LightGBM.predict(X_test)

### Cat Boost
model_Tomek_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_Tomek_CatBoost.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_CatBoost = model_Tomek_CatBoost.predict(X_test)



### 2. 스모트
# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, SMOTE: {Counter(y_train_SMOTE)}")
# {0: 10313, 1: 10313}

## Logistic Regression (로지스틱 회귀 분석)
model_SMOTE_LogisticRegression = LogisticRegression()
model_SMOTE_LogisticRegression.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_LogisticRegression = model_SMOTE_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_SMOTE_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_SMOTE_DecisionTreeClassifier.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_DecisionTreeClassifier = model_SMOTE_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_SMOTE_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_SMOTE_RandomForest.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_RandomForest = model_SMOTE_RandomForest.predict(X_test)

## XG Boost
model_SMOTE_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_SMOTE_XGBoost.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_XGBoost = model_SMOTE_XGBoost.predict(X_test)

### Light GBM
model_SMOTE_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0)
model_SMOTE_LightGBM.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_LightGBM = model_SMOTE_LightGBM.predict(X_test)

### Cat Boost
model_SMOTE_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_SMOTE_CatBoost.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_CatBoost = model_SMOTE_CatBoost.predict(X_test)



### 3. 스모트 토멕
# SMOTE TOMEK
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_train_SmoteTomek, y_train_SmoteTomek = smote_tomek.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, SMOTE TOMEK: {Counter(y_train_SmoteTomek)}")
# {0: 10020, 1: 10020}


## Logistic Regression (로지스틱 회귀 분석)
model_SmoteTomek_LogisticRegression = LogisticRegression()
model_SmoteTomek_LogisticRegression.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_LogisticRegression = model_SmoteTomek_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_SmoteTomek_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_SmoteTomek_DecisionTreeClassifier.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_DecisionTreeClassifier = model_SmoteTomek_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_SmoteTomek_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_SmoteTomek_RandomForest.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_RandomForest = model_SmoteTomek_RandomForest.predict(X_test)

## XG Boost
model_SmoteTomek_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_SmoteTomek_XGBoost.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_XGBoost = model_SmoteTomek_XGBoost.predict(X_test)

### Light GBM
model_SmoteTomek_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0)
model_SmoteTomek_LightGBM.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_LightGBM = model_SmoteTomek_LightGBM.predict(X_test)

### Cat Boost
model_SmoteTomek_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_SmoteTomek_CatBoost.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_CatBoost = model_SmoteTomek_CatBoost.predict(X_test)


### recall_score
from sklearn.metrics import recall_score

recall_Original_LogisticRegression = recall_score(y_test, y_pred_Original_LogisticRegression)
recall_Original_DecisionTreeClassifier = recall_score(y_test, y_pred_Original_DecisionTreeClassifier)
recall_Original_RandomForest = recall_score(y_test, y_pred_Original_RandomForest)
recall_Original_XGBoost = recall_score(y_test, y_pred_Original_XGBoost)
recall_Original_LightGBM = recall_score(y_test, y_pred_Original_LightGBM)
recall_Original_CatBoost = recall_score(y_test, y_pred_Original_CatBoost)

recall_Tomek_LogisticRegression = recall_score(y_test, y_pred_Tomek_LogisticRegression)
recall_Tomek_DecisionTreeClassifier = recall_score(y_test, y_pred_Tomek_DecisionTreeClassifier)
recall_Tomek_RandomForest = recall_score(y_test, y_pred_Tomek_RandomForest)
recall_Tomek_XGBoost = recall_score(y_test, y_pred_Tomek_XGBoost)
recall_Tomek_LightGBM = recall_score(y_test, y_pred_Tomek_LightGBM)
recall_Tomek_CatBoost = recall_score(y_test, y_pred_Tomek_CatBoost)

recall_SMOTE_LogisticRegression = recall_score(y_test, y_pred_SMOTE_LogisticRegression)
recall_SMOTE_DecisionTreeClassifier = recall_score(y_test, y_pred_SMOTE_DecisionTreeClassifier)
recall_SMOTE_RandomForest = recall_score(y_test, y_pred_SMOTE_RandomForest)
recall_SMOTE_XGBoost = recall_score(y_test, y_pred_SMOTE_XGBoost)
recall_SMOTE_LightGBM = recall_score(y_test, y_pred_SMOTE_LightGBM)
recall_SMOTE_CatBoost = recall_score(y_test, y_pred_SMOTE_CatBoost)

recall_SmoteTomek_LogisticRegression = recall_score(y_test, y_pred_SmoteTomek_LogisticRegression)
recall_SmoteTomek_DecisionTreeClassifier = recall_score(y_test, y_pred_SmoteTomek_DecisionTreeClassifier)
recall_SmoteTomek_RandomForest = recall_score(y_test, y_pred_SmoteTomek_RandomForest)
recall_SmoteTomek_XGBoost = recall_score(y_test, y_pred_SmoteTomek_XGBoost)
recall_SmoteTomek_LightGBM = recall_score(y_test, y_pred_SmoteTomek_LightGBM)
recall_SmoteTomek_CatBoost = recall_score(y_test, y_pred_SmoteTomek_CatBoost)

recall_score_table = pd.DataFrame({
    'sampling' : ['Original', 'Tomek', 'SMOTE', 'SmoteTomek'],
    'Logistic Regression' : [recall_Original_LogisticRegression, recall_Tomek_LogisticRegression, recall_SMOTE_LogisticRegression, recall_SmoteTomek_LogisticRegression],
    'Decision Tree Regressor' : [recall_Original_DecisionTreeClassifier, recall_Tomek_DecisionTreeClassifier, recall_SMOTE_DecisionTreeClassifier, recall_SmoteTomek_DecisionTreeClassifier],
    'Random Forest' : [recall_Original_RandomForest, recall_Tomek_RandomForest, recall_SMOTE_RandomForest, recall_SmoteTomek_RandomForest],
    'XG Boost' : [recall_Original_XGBoost, recall_Tomek_XGBoost, recall_SMOTE_XGBoost, recall_SmoteTomek_XGBoost],
    'Light GBM' : [recall_Original_LightGBM, recall_Tomek_LightGBM, recall_SMOTE_LightGBM, recall_SmoteTomek_LightGBM],
    'Cat Boost' : [recall_Original_CatBoost, recall_Tomek_CatBoost, recall_SMOTE_CatBoost, recall_SmoteTomek_CatBoost]
})
recall_score_table
# SMOTE - Logistic Regression 세트가 0.605809으로 가장 좋다.


coefficients = model_SMOTE_LogisticRegression.coef_[0]
model_SMOTE_LogisticRegression.intercept_[0]
# 특성 이름과 함께 계수를 출력
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)
feature_importance