import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind


# 한글 폰트 깨짐 방지
plt.rc('font', family='Malgun Gothic')

df = pd.read_csv("data/data_week3.csv")

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
    sns.histplot(data=df_pic, x=column, binrange=[0, upper_97_percentile])
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

# 산점도 (어따 배치할지 모름)

# 5x4 서브플롯 설정 (20칸)
fig, axes = plt.subplots(5, 4, figsize=(16, 20))

# 각 서브플롯에 scatterplot 추가
for i, col in enumerate(df.columns[:-1]):
    row = i // 4  # 행 인덱스 계산
    col_idx = i % 4  # 열 인덱스 계산
    sns.scatterplot(data=df, x=col, y='target', hue='target', ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'Scatterplot of {col}')

# 남는 서브플롯이 있으면 삭제 대신 비우기
total_columns = len(df.columns[:-1])
if total_columns < 20:
    for j in range(total_columns, 20):
        axes[j // 4, j % 4].set_visible(False)  # 남는 서브플롯을 비활성화

plt.tight_layout()
plt.show()

##############################################
##############################################

# 상관계수

# 스피어만

# unknown4를 순서형으로 인코딩
label_encoder = LabelEncoder()
df['unknown4_encoded'] = label_encoder.fit_transform(df['unknown4'])

# 이산형 변수와 연속형 변수, 인코딩된 unknown4 합치기 (명목형 범주인 unknown1은 제외)
# spearman_cols 설정할 때만 unknown4를 unknown4_encoded로 대체
spearman_cols = [col for col in discrete_cols if col != 'unknown4'] + continuous_cols + ['unknown4_encoded', 'target']

# Spearman 상관계수 계산
spearman_corr = df[spearman_cols].corr(method='spearman')

# Spearman 상관계수 히트맵 시각화
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
print("Top 5 correlation pairs:")
print(top_5_corr_pairs)

# (8,13), (10,8), (10,6), (10,13), (8,6)
# 0.813607  0.806507 0.676573 0.657419  0.654546

# 상관관계 Top 5 연속형 변수 쌍 시각화
scatter_pairs = [
    ('unknown8', 'unknown13'),
    ('unknown8', 'unknown10'),
    ('unknown8', 'unknown6'),
    ('unknown6', 'unknown10'),
    ('unknown10', 'unknown13')
]


# 서브플롯 설정
fig, axes = plt.subplots(1, len(scatter_pairs), figsize=(5 * len(scatter_pairs), 5))

for i, (x_col, y_col) in enumerate(scatter_pairs):
    sns.scatterplot(x=x_col, y=y_col, data=df, ax=axes[i])
    axes[i].set_title(f'Scatter Plot of {x_col} vs {y_col}')

plt.tight_layout()
plt.show() 

#########################################################
#########################################################


# 연속형 x변수끼리 피어슨 상관행렬

## 선형관계

# 피어슨 상관계수 계산
pearson_corr = df_numeric.corr(method='pearson')

# 피어슨 상관계수 히트맵
plt.figure(figsize=(14, 10))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Heatmap')
plt.show()

# 상관계수 행렬에서 대각선 제외한 상관계수 추출
corr_pairs_pearson = pearson_corr.unstack()  # 행렬을 쌍으로 변환
corr_pairs_pearson = corr_pairs_pearson[corr_pairs_pearson.index.get_level_values(0) != corr_pairs_pearson.index.get_level_values(1)]  # 대각선 제거

# 중복된 순서쌍 제거 (첫 번째 변수의 이름이 두 번째 변수보다 작은 경우만 남기기)
corr_pairs_pearson = corr_pairs_pearson.groupby(corr_pairs_pearson.index.map(frozenset)).first()

# 상관계수 절대값 기준으로 내림차순 정렬
sorted_corr_pairs_pearson = corr_pairs_pearson.abs().sort_values(ascending=False)

# 상위 5개 상관관계 쌍 출력
top_5_corr_pairs_pearson = sorted_corr_pairs_pearson.head(5)
print("Top 5 correlation pairs:")
print(top_5_corr_pairs_pearson)

# (15, 14), (10, 8), (10, 14), (8, 12), (8, 5)
# 0.677759  0.622413  0.585047 0.583321 0.536765

# 상관관계 Top 5 연속형 변수 쌍 시각화
scatter_pairs_pearson = [
    ('unknown15', 'unknown14'),
    ('unknown8', 'unknown10'),
    ('unknown10', 'unknown14'),
    ('unknown8', 'unknown12'),
    ('unknown8', 'unknown5')
]

# 서브플롯 설정
fig, axes = plt.subplots(1, len(scatter_pairs_pearson), figsize=(5 * len(scatter_pairs_pearson), 5))

for i, (x_col, y_col) in enumerate(scatter_pairs_pearson):
    sns.scatterplot(x=x_col, y=y_col, data=df, ax=axes[i])
    axes[i].set_title(f'Scatter Plot of {x_col} vs {y_col}')

plt.tight_layout()
plt.show() 


# 변수간 관계, 선형성 확인
# PairPlot
sns.pairplot(df_numeric)
plt.show()

# Violin Plot

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