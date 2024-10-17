'''
파라미터 조정에 빡친 장일준은 로지스틱회귀 분류 모델 성능 올리는거에 집중하기로 하는데...
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter # 샘플링 갯수 결과 보기 위해 필요한 모듈

# 데이터 로드
df = pd.read_csv("../data/data_week3.csv")

# 범주형 변수를 One-Hot 인코딩 (get_dummies)
df = pd.get_dummies(df, columns=['unknown1'])

##### 극단값 하나씩 삭제. 예솔누나 아이디어
df.shape # (14094, 18)

df['unknown2'].value_counts().sort_index(ascending=False) # 294451
df = df[df['unknown2'] != 294451]

df['unknown6'].value_counts().sort_index(ascending=False) # 2398
df = df[df['unknown6'] != 2398]

df['unknown8'].value_counts().sort_index(ascending=False) # 31706
df = df[df['unknown8'] != 31706]

df['unknown10'].value_counts().sort_index(ascending=False) # 877
df = df[df['unknown10'] != 877]

df['unknown14'].value_counts().sort_index(ascending=False) # 403
df = df[df['unknown14'] != 403]

df['unknown16'].value_counts().sort_index(ascending=False) # 2840.5, 1029.9
df = df[df['unknown16'] != 2840.5]
df = df[df['unknown16'] != 1029.9]

df.shape # (14087, 18) 7개 지워짐.


# X,y 나누기
X = df.drop('target', axis=1)
y = df['target']

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 20%, stratify 적용)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Before Sampling: {Counter(y_train)}")
# {0: 10313, 1: 962}

# 모델 및 샘플링 하기
from sklearn.linear_model import LogisticRegression
# 모델 설정
model = LogisticRegression()

### 1. 기본 데이터(샘플링 안 함)
# Original Data
X_train
y_train

# 모델 핏, 예측
model.fit(X_train, y_train)
y_pred_Original_LogisticRegression = model.predict(X_test)


### 2. 랜덤 언더 샘플링
# Random Under Sampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_train_RandomUnderSample, y_train_RandomUnderSample = rus.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}") # 결과 확인
print(f"After Sampling, Random Under Sampling: {Counter(y_train_RandomUnderSample)}")
# {0: 962, 1: 962}

# 모델 핏, 예측
model.fit(X_train_RandomUnderSample, y_train_RandomUnderSample)
y_pred_Under_LogisticRegression = model.predict(X_test)


### 3. 토멕링크
# Tomek Links
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(sampling_strategy='majority')
X_train_TomekLinks, y_train_TomekLinks = tl.fit_resample(X_train,y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, Tomek Links: {Counter(y_train_TomekLinks)}")
# {0: 9889, 1: 962}

# 모델 핏, 예측
model.fit(X_train_TomekLinks, y_train_TomekLinks)
y_pred_Tomek_LogisticRegression = model.predict(X_test)


### 4. 랜덤 오버 샘플링
# Random Over Sampling
from imblearn.over_sampling import RandomOverSampler # 랜덤오버샘플링 해주는거.
ros = RandomOverSampler(random_state=42)
X_train_RandomOverSample, y_train_RandomOverSample = ros.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, Random Over Sampling: {Counter(y_train_RandomOverSample)}")
# {0: 10307, 1: 10307}

# 모델 핏, 예측
model.fit(X_train_RandomOverSample, y_train_RandomOverSample)
y_pred_Over_LogisticRegression = model.predict(X_test)


### 5. 스모트
# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, SMOTE: {Counter(y_train_SMOTE)}")
# {0: 10307, 1: 10307}

# 모델 핏, 예측
model.fit(X_train_SMOTE, y_train_SMOTE)
y_pred_SMOTE_LogisticRegression = model.predict(X_test)


### 6. 스모트 토멕
# SMOTE TOMEK
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_train_SmoteTomek, y_train_SmoteTomek = smote_tomek.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, SMOTE TOMEK: {Counter(y_train_SmoteTomek)}")
# {0: 10024, 1: 10024}

# 모델 핏, 예측
model.fit(X_train_SmoteTomek, y_train_SmoteTomek)
y_pred_SmoteTomek_LogisticRegression = model.predict(X_test)


### 7. 토멕링크 여러번 + 랜덤언더셈플
# Tomek Links
from imblearn.under_sampling import TomekLinks
tl_2 = TomekLinks(sampling_strategy='majority')
X_train_TomekLinks2, y_train_TomekLinks2 = X_train, y_train

# Tomek Links를 반복적으로 적용
while True:
    X_train_TomekLinks_new, y_train_TomekLinks_new = tl_2.fit_resample(X_train_TomekLinks2, y_train_TomekLinks2)
    
    # 더 이상 변화가 없으면 종료
    if len(y_train_TomekLinks_new) == len(y_train_TomekLinks2):
        break
    
    # 업데이트
    X_train_TomekLinks2, y_train_TomekLinks2 = X_train_TomekLinks_new, y_train_TomekLinks_new

# 달라진거 보기.
print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, Tomek Links: {Counter(y_train_TomekLinks_new)}")
# {0: 9676, 1: 962}
# 한 600개 정도 줄었구만.

# 랜덤언더셈플
rus_2 = RandomUnderSampler(random_state=42)
X_train_TomekUnder, y_train_TomekUnder = rus_2.fit_resample(X_train_TomekLinks_new, y_train_TomekLinks_new)

print(f"Before RandomUnderSampling: {Counter(y_train_TomekLinks_new)}") # 결과 확인
print(f"After Sampling, Random Under Sampling: {Counter(y_train_TomekUnder)}")
# {0: 962, 1: 962}

# 모델 핏, 예측
model.fit(X_train_TomekUnder, y_train_TomekUnder)
y_pred_TomekUnder_LogisticRegression = model.predict(X_test)




### recall_score
from sklearn.metrics import recall_score
recall_Original_LogisticRegression = recall_score(y_test, y_pred_Original_LogisticRegression)
recall_Under_LogisticRegression = recall_score(y_test, y_pred_Under_LogisticRegression)
recall_Tomek_LogisticRegression = recall_score(y_test, y_pred_Tomek_LogisticRegression)
recall_Over_LogisticRegression = recall_score(y_test, y_pred_Over_LogisticRegression)
recall_SMOTE_LogisticRegression = recall_score(y_test, y_pred_SMOTE_LogisticRegression)
recall_SmoteTomek_LogisticRegression = recall_score(y_test, y_pred_SmoteTomek_LogisticRegression)
recall_TomekUnder_LogisticRegression = recall_score(y_test, y_pred_TomekUnder_LogisticRegression)

recall_score_table = pd.DataFrame({
    'sampling' : ['Original', 'RandomUnder', 'Tomek', 'RandomOver', 'SMOTE', 'SmoteTomek', 'TomekUnder'],
    'Logistic Regression' : [recall_Original_LogisticRegression, recall_Under_LogisticRegression, recall_Tomek_LogisticRegression, recall_Over_LogisticRegression, recall_SMOTE_LogisticRegression, recall_SmoteTomek_LogisticRegression, recall_TomekUnder_LogisticRegression]
    })

recall_score_table