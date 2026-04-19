import numpy as np
import pandas as pd
import oracledb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib
from ott.models import PredictionModel
import shap

oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\instantclient_19_25")

USER = "scott"
PASSWORD = "tiger"
DSN = "localhost:1521/xe"

model = PredictionModel(user=USER, password=PASSWORD, dsn=DSN)

df = model.get_training_data()

# 데이터 전처리

# USED_LAST_WEEK = 2 , NULL -> 0
df['AVG_MIN_WEEKDAY'] = df['AVG_MIN_WEEKDAY'].fillna(0)
df['AVG_MIN_WEEKEND'] = df['AVG_MIN_WEEKEND'].fillna(0)

# 이진화
df['WATCH_ORIGINAL'] = (df['WATCH_ORIGINAL'] > 0).astype(int)
df['WATCH_MOVIE'] = (df['WATCH_MOVIE'] > 0).astype(int)
df['WATCH_SHORTFORM'] = (df['WATCH_SHORTFORM'] > 0).astype(int)

# feature engineering
df['TOTAL_MIN'] = df['AVG_MIN_WEEKDAY'] * 5 + df['AVG_MIN_WEEKEND'] * 2
df['MONTHLY_FEE'] = df['MONTHLY_FEE_CODE'].map({
    1 : 1500, 2 : 4000, 3 : 7000, 4 : 10500, 5 : 13500, 6 : 17500, 7 : 20000
})
df['USED_LAST_WEEK'] = (df['TOTAL_MIN'] > 0).astype(int)

total_min_safe = df['TOTAL_MIN'].replace(0, np.nan)

df['EXPLORE_IDX'] = (df['CONTENT_DIVERSITY'] / total_min_safe * df['SEARCH_VIEW']).fillna(0)
upper_e = df['EXPLORE_IDX'].quantile(0.99)
df['EXPLORE_IDX'] = df['EXPLORE_IDX'].clip(upper=upper_e)

df['CHERRY_PICK_IDX'] = (df['OTT_COUNT'] / total_min_safe).fillna(0)
upper_c = df['CHERRY_PICK_IDX'].quantile(0.99)
df['CHERRY_PICK_IDX'] = df['CHERRY_PICK_IDX'].clip(upper=upper_c)

# 이용 빈도 분류
def classify_freq(x):
    if x <= 2:
        return 2
    elif x <= 4:
        return 1
    else:
        return 0

df['FREQ_GROUP'] = df['USE_FREQUENCY'].apply(classify_freq)


# input, target 데이터 설정
feature_cols = [
    'RECOMMEND_VIEW', 'TOTAL_MIN', 'EXPLORE_IDX',
    'CHERRY_PICK_IDX', 'MONTHLY_FEE', 'FAMILY_TYPE',
    'BINGE_WATCH', 'OTT_COUNT', 'DEVICE_COUNT', 'USED_LAST_WEEK',
    'WATCH_ORIGINAL', 'WATCH_MOVIE', 'WATCH_SHORTFORM'
]

data = df[feature_cols]
target = df['FREQ_GROUP']

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42, stratify=target
)

# Random Forest (GridSearchCV)
# param = {
#     'n_estimators': [200, 300, 400, 500],
#     'max_depth': [5, 7, 10, 20, 30],
#     'min_samples_split': [10, 15, 20],
#     'min_samples_leaf': [1, 2],
#     'class_weight': ['balanced']
# }

# 최적의 parameter 확정

rf = RandomForestClassifier(
    n_estimators = 300,
    max_depth =10,
    min_samples_split = 10,
    min_samples_leaf = 1,
    class_weight = 'balanced',
    random_state=42
)

rf.fit(train_input, train_target)

# print('\n=== Feature Importance ===')
# importance_series = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
# print(importance_series)


# score 출력
print(rf.score(test_input, test_target))

pred = rf.predict(test_input)
print(classification_report(test_target, pred, target_names=['저빈도(이탈위험)', '중빈도', '고빈도']))

# 모델 DB에 저장

df['PRED_GROUP'] = rf.predict(data)

model.insert_predictions(df)

explainer = shap.TreeExplainer(rf)
joblib.dump(rf, 'random_forest.pkl')
joblib.dump(explainer, 'shap_explainer.pkl')
print('모델 저장 완료')