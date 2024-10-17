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
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import PartialDependenceDisplay
import shap


# 한글 폰트 깨짐 방지
plt.rc('font', family='Malgun Gothic')

df = pd.read_csv("../data/data_week3.csv")

# 데이터 확인
df
df.info()

# 기초 통계량
df.describe(include='all')

# 결측치 확인
df.isna().sum().sum()

# 각 변수에서 0의 개수를 계산
zero_counts = (df == 0).sum()
zero_counts

# 고윳값 갯수 확인하기
for i in df.columns:
    my_unique_len = len(df[f"{i}"].unique())
    print(f"{i}" + " = " + f"{my_unique_len}")

# unknown 1의 분포
sns.countplot(x='unknown1', data=df)

# unknown 4의 분포
sns.countplot(x='unknown4', data=df)

# 범주형으로 바꿀만한 변수 찾기
df.nunique()
df['unknown1'].unique()
df['unknown4'].unique()

# Unknown1, 4를 범주형으로 변환
df['unknown1'] = df['unknown1'].astype('category')
df['unknown4'] = df['unknown4'].astype('category')

# 확인
df.info()

# 범주형 변수와 숫자형 변수 구분
df_numeric = df.select_dtypes(include='number')
df_category = df.select_dtypes(include= 'category')

# 확인
df_numeric
df_category

######################################
######################################
######################################

# EDA

# 타겟 변수 히스토그램 

# 타겟 변수에서 1의 비율 계산
target_1_ratio = sum(df['target'] == 1)/len(df['target']) * 100 

# countplot에서 '1'에 해당하는 막대 위치에 비율을 표시
plt.figure(figsize=(6, 4))  # 플롯의 크기를 적절히 설정
sns.countplot(x='target', data=df, palette=['skyblue', 'salmon'])
# countplot에서 '1'에 해당하는 막대 위치에 비율을 표시
count_1 = df['target'].value_counts()[1]  # '1'에 해당하는 개수
plt.text(1, count_1 + 100, f'{target_1_ratio:.2f}%', ha='center', fontsize=12, color='red', fontweight='bold')  # 텍스트를 '1' 막대 위에 표시
# 그래프 추가 설정
plt.title('Target Variable Distribution (0: 정상, 1: 이상)')
plt.xlabel('Target (0: Normal, 1: Abnormal)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# KDE plot을 통해 각 변수의 분포 시각화
plt.figure(figsize=(15, 12))
for i, column in enumerate(df_numeric.columns[:-1]):  # target 열 제외
    plt.subplot(4, 4, i+1)  # 4 * 4 서브플롯
    sns.kdeplot(data=df_numeric, x=column, fill=True)
    plt.title(f'Distribution of {column}')
    
plt.tight_layout()
plt.show()
plt.clf()

# KDE plot 보니까 잘 안 보이기 때문에 (최대값이 너무 커서)
# 상위 3% 값 무시하고 그리기

df_pic = df.select_dtypes(include='number').drop(['unknown5', 'unknown15'], axis=1)

# 이상한 애들
sns.histplot(data=df, x= 'unknown5', binrange= [0, 5]) 
sns.histplot(data=df, x= 'unknown4', binrange= [0, 5]) 
sns.histplot(data=df, x= 'unknown15', binrange= [0, 5]) 

# 상위 3% 값 무시하고 분포
for column in df_pic.columns:
    upper_97_percentile = df_pic[column].quantile(0.97)
    
    # binrange 설정
    sns.kdeplot(data=df_pic, x=column, fill=True)
    plt.title(f'Histogram of {column} (97% percentile binrange)')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()


ordinal_cols = ['unknown4']  # 순서형 범주형 변수
categorical_cols = ['unknown1', 'unknown4']
# 이산형 변수
discrete_cols = ['unknown4', 'unknown5', 'unknown15']
# 연속형 변수
continuous_cols = ['unknown2', 'unknown3', 'unknown6', 'unknown7', 'unknown8', 'unknown9', 
                   'unknown10', 'unknown11', 'unknown12', 'unknown13', 'unknown14',
                   'unknown16', 'unknown17']

# 4,5,15는 이산형이라 제외하고 따로 그림

# 위는 단순 변수들의 히스토그램이고, 밑은 타겟에 따른 분포

# 4, 5, 15

# 이산형 변수 시각화 - barplot 및 포인트 플롯 사용
fig, axes = plt.subplots(2, len(discrete_cols), figsize=(5 * len(discrete_cols), 10))
if len(discrete_cols) == 1:
    axes[0] = [axes[0]]
    axes[1] = [axes[1]]
for i, col in enumerate(discrete_cols):
    # Bar Plot
    sns.barplot(x=col, y='target', data=df, ax=axes[0][i], ci=None)
    axes[0][i].set_title(f'Bar Plot of {col} by Target')

    # Point Plot
    sns.pointplot(x=col, y='target', data=df, ax=axes[1][i],ci=None)
    axes[1][i].set_title(f'Point Plot of {col} by Target')

# 범주형 변수 시각화 - countplot 및 스택형 막대그래프 사용
fig, axes = plt.subplots(2, len(categorical_cols), figsize=(5 * len(categorical_cols), 10))
if len(categorical_cols) == 1:
    axes[0] = [axes[0]]
    axes[1] = [axes[1]]
for i, col in enumerate(categorical_cols):
    # Count Plot
    sns.countplot(x=col, hue='target', data=df, ax=axes[0][i])
    axes[0][i].set_title(f'Count Plot of {col} by Target')

    # 스택형 막대그래프
    crosstab = pd.crosstab(df[col], df['target'], normalize='index')
    crosstab.plot(kind='bar', stacked=True, ax=axes[1][i], color=['#1f77b4', '#ff7f0e'])
    axes[1][i].set_title(f'Stacked Bar Plot of {col} by Target')
    axes[1][i].set_xlabel(col)
    axes[1][i].set_ylabel("Proportion")

# 연속형 변수 시각화 - boxplot과 kdeplot 사용
fig, axes = plt.subplots(2, len(continuous_cols), figsize=(5 * len(continuous_cols), 10))
if len(continuous_cols) == 1:
    axes[0] = [axes[0]]
    axes[1] = [axes[1]]
for i, col in enumerate(continuous_cols):
    # Box Plot
    sns.boxplot(x='target', y=col, hue='target', data=df, ax=axes[0][i])
    axes[0][i].set_title(f'Box Plot of {col} by Target')

    # KDE Plot
    sns.kdeplot(data=df[df['target'] == 0][col], label='target=0', ax=axes[1][i], fill=True)
    sns.kdeplot(data=df[df['target'] == 1][col], label='target=1', ax=axes[1][i], fill=True)
    axes[1][i].set_title(f'KDE Plot of {col} by Target')

plt.tight_layout()
plt.show()

# CrossTab

# 각 범주형 변수와 target 간의 관계를 히트맵으로 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# x1과 target 간의 관계
cross_tab1 = pd.crosstab(df['unknown1'], df['target'], normalize='index')
sns.heatmap(cross_tab1, annot=True, cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title("Relationship between unknown1 and Target")

# x4와 target 간의 관계
cross_tab2 = pd.crosstab(df['unknown4'], df['target'], normalize='index')
sns.heatmap(cross_tab2, annot=True, cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title("Relationship between unknown4 and Target")

# x5와 target 간의 관계
cross_tab3 = pd.crosstab(df['unknown15'], df['target'], normalize='index')
sns.heatmap(cross_tab3, annot=True, cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title("Relationship between unknown5 and Target")

# x15와 target 간의 관계
cross_tab4 = pd.crosstab(df['unknown15'], df['target'], normalize='index')
sns.heatmap(cross_tab4, annot=True, cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title("Relationship between unknown15 and Target")

plt.tight_layout()
plt.show()

# 값의 평균

# y가 0과 1일 때 변수 x들의 평균 계산
mean_values = df_numeric.groupby('target').mean().transpose()  # 열과 행을 전치하여 그래프에 적합하게 변환

# 0과 1 모두에서 평균값이 100 이하인 변수들
mean_values_below_100 = mean_values[(mean_values[0] <= 100) & (mean_values[1] <= 100)]

# 0과 1 모두에서 평균값이 100 초과인 변수들
mean_values_above_100 = mean_values[(mean_values[0] > 100) & (mean_values[1] > 100)]

# 그래프 설정 (100 이하인 변수들)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

mean_values_below_100.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title("y=0과 y=1에 따른 변수들의 평균 비교 (100 이하)")
ax1.set_xlabel("Variables")
ax1.set_ylabel("Mean values")
ax1.legend(title='Target', labels=['y=0', 'y=1'])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# 각 막대 위에 평균값 표시 (100 이하)
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.2f', label_type="edge")  # 소수점 두 자리까지 표시

# 그래프 설정 (100 초과인 변수들)
mean_values_above_100.plot(kind='bar', ax=ax2, width=0.8)
ax2.set_title("y=0과 y=1에 따른 변수들의 평균 비교 (100 초과)")
ax2.set_xlabel("Variables")
ax2.set_ylabel("Mean values")
ax2.legend(title='Target', labels=['y=0', 'y=1'])
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

# 각 막대 위에 평균값 표시 (100 초과)
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.2f', label_type="edge")  # 소수점 두 자리까지 표시

plt.tight_layout()
plt.show()

##############################################
##############################################

#### 1. 스피어만 상관계수

# unknown4, unknown5, unknown15 변수를 순서형으로 인코딩
label_encoder = LabelEncoder()
df['unknown4_encoded'] = label_encoder.fit_transform(df['unknown4'])
df['unknown5_encoded'] = label_encoder.fit_transform(df['unknown5'])
df['unknown15_encoded'] = label_encoder.fit_transform(df['unknown15'])

# 이산형 및 연속형 변수 설정 (unknown1은 제외)
discrete_cols = ['unknown4_encoded', 'unknown5_encoded', 'unknown15_encoded']
continuous_cols = ['unknown2', 'unknown3', 'unknown6', 'unknown7', 'unknown8', 'unknown9', 'unknown10', 'unknown11', 'unknown12', 'unknown13', 'unknown14', 'unknown16', 'unknown17']

# 스피어만 상관계수를 구할 변수들 설정 (unknown1 제외)
spearman_cols = discrete_cols + continuous_cols + ['target']

# 스피어만 상관계수 계산
spearman_corr = df[spearman_cols].corr(method='spearman')

# 스피어만 상관계수 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Spearman Correlation Matrix (Discrete, Continuous, and Ordinal variables)')
plt.show()

# 상관계수 행렬에서 대각선 제외한 상관계수 추출
corr_pairs = spearman_corr.unstack()  # 행렬을 쌍으로 변환
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]  # 대각선 제거

# 중복된 순서쌍 제거 (첫 번째 변수의 이름이 두 번째 변수보다 작은 경우만 남기기)
corr_pairs = corr_pairs.groupby(corr_pairs.index.map(frozenset)).first()

# 상관계수 절대값 기준으로 내림차순 정렬
sorted_corr_pairs = corr_pairs.abs().sort_values(ascending=False)

# 상위 5개 상관관계 쌍 출력
top_5_corr_pairs = sorted_corr_pairs.head(5)
top_5_corr_pairs

scatter_pairs = [
    ('unknown13', 'unknown8'),
    ('unknown8', 'unknown10'),
    ('unknown6', 'unknown10'),
    ('unknown13', 'unknown10'),
    ('unknown6', 'unknown8')
]

# 산점도 시각화
plt.figure(figsize=(15, 10))
for i, (x_var, y_var) in enumerate(scatter_pairs, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=df[x_var], y=df[y_var])
    plt.title(f'Scatter Plot of {x_var} vs {y_var}')
plt.tight_layout()
plt.show()




#### 2. 피어슨 상관계수

# 연속형 변수 unknown2부터 unknown17까지 선택
continuous_cols_only = [f'unknown{i}' for i in range(2, 18)]  # unknown2 ~ unknown17

# unknown1과 target 변수를 제외한 나머지 연속형 변수들에 대한 피어슨 상관계수 계산
pearson_corr_continuous = df[continuous_cols_only].corr(method='pearson')

# 피어슨 상관계수 히트맵 그리기 (연속형 변수인 unknown2~unknown17)
plt.figure(figsize=(14, 10))
sns.heatmap(pearson_corr_continuous, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Heatmap (Continuous Variables: unknown2 to unknown17)')
plt.show()


################ Violin Plot

# 숫자형 변수만 선택
numeric_columns = df_numeric.columns

# 변수 개수만큼 서브플롯을 만들기 위한 행과 열 계산
num_columns = len(numeric_columns)
num_rows = (num_columns // 3) + (num_columns % 3 > 0)

# 서브플롯 생성
fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

# 각 숫자형 변수별 바이올린 플롯 그리기
for i, column in enumerate(numeric_columns):
    row = i // 3
    col = i % 3
    sns.violinplot(ax=axes[row, col], y=df[column], hue=df['target'])
    axes[row, col].set_title(f'Violin Plot of {column}')
    axes[row, col].set_xlabel(column)

# 불필요한 빈 서브플롯 제거
for j in range(i + 1, num_rows * 3):
    fig.delaxes(axes[j // 3, j % 3])

plt.tight_layout()
plt.show()


# 타겟 분리
# 타입별 타겟 변수 분포 시각화
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x='unknown1', hue='target', palette='Set2')

# unknown1에서 각각의 값에 대해 1의 비율을 계산
total_counts = df['unknown1'].value_counts()  # unknown1 값의 전체 개수
target1_counts = df[df['target'] == 1]['unknown1'].value_counts()  # target=1인 unknown1 값의 개수

# 비율 계산
target1_ratio = (target1_counts / total_counts * 100).reindex(total_counts.index).fillna(0)

# 막대 위에 수치 및 비율 표시
for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(f'{count}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                fontsize=10, color='black',
                xytext=(0, 5), textcoords='offset points')

# 1의 비율을 추가로 표시
for idx, val in enumerate(total_counts.index):
    # 해당 unknown1 값에 대해 target=1의 비율
    ratio = target1_ratio[val]
    
    # target=1인 막대 위에 비율 표시
    bar = ax.patches[idx + len(total_counts)]  # 두 번째 그룹의 막대에 접근
    ax.annotate(f'{ratio:.2f}%',
                (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                ha='center', va='bottom',
                fontsize=10, color='red',
                xytext=(0, 15), textcoords='offset points')

plt.title('Distribution of Target Variable by Type in unknown1')
plt.xlabel('Type')
plt.ylabel('Count')
plt.legend(title='Target', labels=['Normal (0)', 'Abnormal (1)'])
plt.xticks(rotation=0)  # x축 레이블 회전
plt.show()


######################################
######################################
######################################

# 통계 검정
# 범주형 변수는 카이제곱

# df_category 내 모든 변수와 target 간의 카이제곱 검정
chi2_results = {}

for col in df_category.columns:
    # 교차표 생성 (각 변수와 target 간의 분포)
    contingency_table = pd.crosstab(df[col], df['target'])
    
    # 카이제곱 검정 수행
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # 결과 저장
    chi2_results[col] = {'chi2_statistic': chi2, 'p_value': p, 'degrees_of_freedom': dof}

# 카이제곱 검정 결과 출력
chi2_results_df = pd.DataFrame(chi2_results).transpose()
print(chi2_results_df)


           #statistic  p_value  dof
# unknown1  10.851714  0.012555  3.0
# unknown4  3.723956  0.292856    3.0


# 연속형 변수는 T-test

continuous_cols = ['unknown2', 'unknown3', 'unknown6', 'unknown7', 'unknown8', 'unknown9', 
                   'unknown10', 'unknown11', 'unknown12', 'unknown13', 'unknown14',
                   'unknown16', 'unknown17']

# t-검정 결과 저장할 딕셔너리
t_test_results = {}

# 각 연속형 변수에 대해 t-검정 수행
for col in continuous_cols:
    group0 = df[df['target'] == 0][col]
    group1 = df[df['target'] == 1][col]
    t_stat, p_val = ttest_ind(group0, group1)
    t_test_results[col] = {'T-statistic': t_stat, 'p-value': p_val}

# t-검정 결과를 데이터프레임으로 변환하여 출력
t_test_results_df = pd.DataFrame(t_test_results).T
t_test_results_df 

# 소수점 자리수를 지정하여 p-value와 T-statistic을 표현
t_test_results_df_formatted = t_test_results_df.copy()
t_test_results_df_formatted['T-statistic'] = t_test_results_df_formatted['T-statistic'].map('{:.6f}'.format)
t_test_results_df_formatted['p-value'] = t_test_results_df_formatted['p-value'].map('{:.6f}'.format)

t_test_results_df_formatted # 6,9,11이 유의하지 않음



# ==========================================================================================
# ==================================    EDA 끝    ==========================================
# ==========================================================================================

# 데이터 로드
df = pd.read_csv("../data/data_week3.csv")

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
    model.fit(X_train, y_train)
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
best_lgbm_model.fit(X_train, y_train)


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
    model.fit(X_train, y_train)
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
best_rf_model.fit(X_train, y_train)


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
    model.fit(X_train, y_train)
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
best_xgb_model.fit(X_train, y_train)


# CatBoost 최적화 함수
def cat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 200),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'random_state': 42
    }

    model = CatBoostClassifier(**params, verbose=0)
    model.fit(X_train, y_train)
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
best_cat_model.fit(X_train, y_train)


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
    model.fit(X_train, y_train)

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


######### 중간에 그림 ###########
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
#########           ###########


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


# 모든 모델의 특이도, 재현율, 임계값, target을 1로 예측한 개수와 비율을 출력하는 함수
def get_model_specificity_recall_thresholds(models, model_names, X_test, y_test, min_specificity=0.9):
    results = []  # 결과 저장 리스트

    for model, name in zip(models, model_names):
        thresholds, specificity_scores, recall_scores = specificity_recall_thresholds(model, X_test, y_test)
        
        best_recall = -1
        best_threshold = -1
        best_specificity = -1
        target_1_count = 0
        target_1_ratio = 0
        
        for i, specificity in enumerate(specificity_scores):
            if specificity >= min_specificity:
                if recall_scores[i] > best_recall:
                    best_recall = recall_scores[i]
                    best_specificity = specificity_scores[i]
                    best_threshold = thresholds[i]
                    
                    # 해당 임계값에서 target을 1로 예측한 개수와 비율 계산
                    y_pred_best_threshold = (model.predict_proba(X_test)[:, 1] >= best_threshold).astype(int)
                    target_1_count = sum(y_pred_best_threshold)
                    target_1_ratio = target_1_count / len(y_test)

        # 결과를 리스트에 추가
        results.append({
            'Model': name,
            'Specificity': best_specificity,
            'Recall': best_recall,
            'Threshold': best_threshold,
            'Predicted target=1 Count': target_1_count,
            'Predicted target=1 Ratio': target_1_ratio
        })

    return results

# 모든 모델의 특이도, 재현율, 임계값, target을 1로 예측한 개수와 비율을 한번에 출력
model_results = get_model_specificity_recall_thresholds(models, model_names, X_test, y_test, min_specificity=0.9)

# 결과 출력
for result in model_results:
    print(f"Model: {result['Model']}")
    print(f"Specificity: {result['Specificity']}")
    print(f"Recall: {result['Recall']}")
    print(f"Threshold: {result['Threshold']}")
    print(f"Predicted target=1 Count: {result['Predicted target=1 Count']}")
    print(f"Predicted target=1 Ratio: {result['Predicted target=1 Ratio']:.2%}")
    print("-" * 40)

# 노샘플링 + (최적화 안 한)LightGBM 선택.
'''
Model: LightGBM
Specificity: 0.9538222739619713
Recall: 0.12448132780082988
Threshold: 0.2
Predicted target=1 Count: 149
Predicted target=1 Ratio: 5.29%
'''

### 제일 좋았던 LightGBM 모델 다시 정의.
classic_model = LGBMClassifier(random_state=42)
classic_model.fit(X_train, y_train)

# 테스트 데이터에 대한 확률 예측
y_pred_classic = classic_model.predict_proba(X_test)[:, 1]  # 클래스 1에 대한 확률
# 임계값 설정 (예: 0.3)
threshold = 0.2
# 임계값을 기준으로 클래스 예측
y_pred_threshold_classic = (y_pred_classic >= threshold).astype(int)

# 혼동행렬 그리기
conf_matrix = confusion_matrix(y_test, y_pred_threshold_classic)
# 혼동행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('classic LightGBM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ================================================================
# ================================================================
#                          모델링 끝.
# ================================================================
# ================================================================

# 기본 파라미터 LightGBM의 이득도표

### 이득도표
result = pd.DataFrame({
    'y_test': y_test,
    'y_pred_prob': y_pred_threshold_classic
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
    return grades_threshold[-1]  # 가장 낮은 등급 할당

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

# 특정 구간 출력
print(result.loc[26:32])

# grade_th와 grade_ratio별 집계
grouped_grade_th = result.groupby(['grade_th'], as_index=False).agg(count=('grade_th', 'count'), Y_1=('y_test', 'sum'))
print(grouped_grade_th)

grouped_grade_ratio = result.groupby(['grade_ratio'], as_index=False).agg(count=('grade_ratio', 'count'), Y_1=('y_test', 'sum'))
grouped_grade_ratio['thresholds'] = result.drop_duplicates('grade_ratio')['last_y_pred_prob'].reset_index(drop=True)
grouped_grade_ratio


########## ROC 커브 시각화
# ROC 커브 계산
fpr_classic, tpr_classic, thresholds_classic = roc_curve(y_test, y_pred_classic)
roc_auc_classic = auc(fpr_classic, tpr_classic)
#fpr_optimization, tpr_optimization, thresholds_optimization = roc_curve(y_test, y_pred_optimization)
#roc_auc_optimization = auc(fpr_optimization, tpr_optimization)

# ROC 커브 시각화
plt.figure()
plt.plot(fpr_classic, tpr_classic, color='blue', lw=2, label=f'ROC curve classic lgbm (area = {roc_auc_classic:.2f})')
#plt.plot(fpr_optimization, tpr_optimization, color='red', lw=2, label=f'ROC curve optimization lgbm (area = {roc_auc_optimization:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 대각선 기준선
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate(1-특이도)')
plt.ylabel('True Positive Rate(민감도)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()



########### 변수 중요도 추출
feature_importances = classic_model.feature_importances_
features = X_train.columns

# 변수 중요도 시각화
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 변수 중요도 막대 그래프 그리기
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))  # 상위 10개 피처만 시각화
plt.title('Top 10 Feature Importances')
plt.show()



# unknown2 변수의 PDP 그리기
X_train
feature_to_plot = 0  # unknown2 변수의 인덱스 (변수 번호에 맞게 조정)
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(classic_model, X_train, features=[feature_to_plot], ax=ax)
plt.title("'unknown2' 변수 값 변화에 따른 불량 확률")
plt.xlabel("unknown2")
plt.ylabel("Partial Dependence")
plt.tight_layout()
plt.show()

# unknown17 변수의 PDP 그리기
feature_to_plot = 15  # unknown17 변수의 인덱스 (변수 번호에 맞게 조정)
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(classic_model, X_train, features=[feature_to_plot], ax=ax)
plt.title("'unknown17' 변수 값 변화에 따른 불량 확률")
plt.xlabel("unknown17")
plt.ylabel("Partial Dependence")
plt.tight_layout()
plt.show()

# unknown16 변수의 PDP 그리기
feature_to_plot = 14  # unknown16 변수의 인덱스 (변수 번호에 맞게 조정)
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(classic_model, X_train, features=[feature_to_plot], ax=ax)
plt.title("'unknown16' 변수 값 변화에 따른 불량 확률")
plt.xlabel("unknown16")
plt.ylabel("Partial Dependence")
plt.tight_layout()
plt.show()


### 예솔 누나가 요청한 혼돈 행렬
y_pred_classic = classic_model.predict_proba(X_test)[:, 1]  # 클래스 1에 대한 확률
# 임계값 설정 (예: 0.3)
threshold = 0.157509
# 임계값을 기준으로 클래스 예측
y_pred_threshold_classic = (y_pred_classic >= threshold).astype(int)

# 혼동행렬 그리기
conf_matrix = confusion_matrix(y_test, y_pred_threshold_classic)
# 혼동행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('classic LightGBM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

### 밑에는 안 쓰는거
'''
### 비교용으로 파라미터 최적화한 LightGBM 모델 정의.
# 최적의 모델 생성
optimization_model = LGBMClassifier(
    colsample_bytree=0.1185260448662222,
    learning_rate=0.030834348179355788,
    max_depth=18,
    min_child_weight=20,
    n_estimators=64,
    random_state=42,
    subsample=0.737265320016441
)
optimization_model.fit(X_train, y_train)

# 테스트 데이터에 대한 확률 예측
y_pred_optimization = optimization_model.predict_proba(X_test)[:, 1]  # 클래스 1에 대한 확률
# 임계값 설정 (예: 0.3)
threshold = 0.15
# 임계값을 기준으로 클래스 예측
y_pred_threshold_optimization = (y_pred_optimization >= threshold).astype(int)

# 혼동행렬 그리기
conf_matrix = confusion_matrix(y_test, y_pred_threshold_optimization)
# 혼동행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('optimization LightGBM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
'''