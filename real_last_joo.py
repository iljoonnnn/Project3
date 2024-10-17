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
df = pd.read_csv("data_week3.csv")

df['unknown2'].value_counts().sort_index(ascending=False) # 294451, 1
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

# 최적의 모델 생성
best_model = LGBMClassifier(
    colsample_bytree=0.1185260448662222,
    learning_rate=0.030834348179355788,
    max_depth=18,
    min_child_weight=20,
    n_estimators=64,
    random_state=42,
    subsample=0.737265320016441
)

# 모델 훈련 
best_model.fit(X_train, y_train)

y_pred_prob = best_model.predict_proba(X_test)[:, 1]
y_pred_prob

### 이득도표
result = pd.DataFrame({
    'y_test': y_test,
    'y_pred_prob': y_pred_prob
})

result = result.sort_values('y_pred_prob',ascending=False).reset_index(drop=True)
result

thresholds = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.5,0]
thresholds

grades_threshold = list(range(1, len(thresholds) + 1))

# y_pred_prob에 따른 grade 할당
def assign_grade(prob):
    for threshold, grade in zip(thresholds, grades_threshold):
        if prob >= threshold:
            return grade
    return grades[-1]  # 가장 낮은 등급 할당

# grade 열 생성
result['grade_th'] = result['y_pred_prob'].apply(assign_grade)
result

grouped_result = result.groupby(['grade_th'],as_index=False).agg(count=('grade_th','count'),Y_1=('y_test','sum'))
grouped_result

##########

# 각 등급별 행 수
grade_counts = [28, 28, 28, 28, 28, 28, 28, 28, 28, 30, 282, 282, 282, 282, 282, 282, 282, 282, 280]
grades_ratio = []

# 등급별로 구간을 나누어 grade 열에 값 추가
grade = 1
start_idx = 0
for count in grade_counts:
    end_idx = start_idx + count
    grades_ratio.extend([grade] * count)
    grade += 1
    start_idx = end_idx

# grade 열 추가
result['grade_ratio'] = grades_ratio
result

# 각 등급의 마지막 행의 y_pred_prob 값을 가져와서 새로운 열로 추가
last_proba_by_grade = result.groupby('grade_ratio')['y_pred_prob'].transform('last')
result['last_y_pred_prob'] = last_proba_by_grade

# 결과 출력
print(result)
result.loc[26:32]

grouped_grade_th = result.groupby(['grade_th'],as_index=False).agg(count=('grade_th','count'),Y_1=('y_test','sum'))
grouped_grade_th


grouped_grade_ratio = result.groupby(['grade_ratio'],as_index=False).agg(count=('grade_ratio','count'),Y_1=('y_test','sum'))
grouped_grade_ratio['thresholds'] = result['last_y_pred_prob'].unique()
grouped_grade_ratio