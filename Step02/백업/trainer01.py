# 파일명: trainer.py
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import IsolationForest
import joblib

print("🤖 AI 모델 학습을 시작합니다.")

# 1. 데이터베이스에서 데이터 불러오기
try:
    db_file = 'sinkbot_data.db'
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query("SELECT * FROM displacement", conn)
    conn.close()
    print(f"✅ 데이터베이스에서 {len(df)}개의 데이터를 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"❌ 데이터베이스 로딩 실패: {e}")
    exit()

# 2. 데이터 전처리 및 특징(Feature) 추출
if len(df) < 10:
    print("❌ 학습을 위한 데이터가 너무 적습니다. 최소 10개 이상의 데이터가 필요합니다.")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp').reset_index(drop=True)

# 기준점 설정
reference_point = df.iloc[0]

# 기준점 대비 변위 계산
df['delta_z'] = df['z'] - reference_point['z'] # 수직 침하량
# 3차원 공간에서의 총 변위 거리 계산
df['distance_3d'] = np.sqrt(
    (df['x'] - reference_point['x'])**2 + 
    (df['y'] - reference_point['y'])**2 + 
    (df['z'] - reference_point['z'])**2
)
print("✅ 데이터 전처리 및 특징 추출 완료. (학습에 사용될 특징: delta_z, distance_3d)")

# 3. AI 모델 학습
# 학습에 사용할 특징들만 선택
features = df[['delta_z', 'distance_3d']]

# Isolation Forest 모델 초기화
# contamination: 데이터 중 이상치의 비율을 의미. 'auto'로 두면 알고리즘이 자동으로 결정
model = IsolationForest(contamination='auto', random_state=42)

print("⏳ AI 모델 학습 중...")
# 선택된 특징으로 모델 학습
model.fit(features)
print("✅ 모델 학습 완료!")

# 4. 학습된 모델 파일로 저장
model_filename = 'sinkbot_model.pkl'
joblib.dump(model, model_filename)
print(f"💾 학습된 모델을 '{model_filename}' 파일로 성공적으로 저장했습니다.")
print("🎉 모든 과정이 완료되었습니다.")