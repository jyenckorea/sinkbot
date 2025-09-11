# 파일명: trainer.py (민감도 조절 버전)
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
if len(df) < 20: # 안정적인 학습을 위해 최소 데이터 수를 20개로 늘립니다.
    print(f"❌ 학습을 위한 데이터가 부족합니다. 최소 20개 이상의 데이터가 필요합니다. (현재: {len(df)}개)")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp').reset_index(drop=True)

reference_point = df.iloc[0]
df['delta_z'] = df['z'] - reference_point['z']
df['distance_3d'] = np.sqrt(
    (df['x'] - reference_point['x'])**2 + 
    (df['y'] - reference_point['y'])**2 + 
    (df['z'] - reference_point['z'])**2
)
print("✅ 데이터 전처리 및 특징 추출 완료.")

# 3. AI 모델 학습
features = df[['delta_z', 'distance_3d']]

# --- ⭐️ 이 부분이 핵심적인 변경점입니다 ⭐️ ---
# contamination: 예상되는 이상치 비율. 값을 낮출수록 모델이 둔감해집니다.
# 0.01은 상위 1%를 이상치로 간주하겠다는 의미입니다.
# 데이터에 이상치가 거의 없다고 확신하면 0.005 또는 더 낮게 설정할 수 있습니다.
contamination_level = 0.01 
print(f"모델 민감도(Contamination)를 {contamination_level * 100}%로 설정합니다.")

# Isolation Forest 모델 초기화
model = IsolationForest(contamination=contamination_level, random_state=42)
# --- ⭐️ 변경 끝 ⭐️ ---

print("⏳ AI 모델 학습 중...")
model.fit(features)
print("✅ 모델 학습 완료!")

# 4. 학습된 모델 파일로 저장
model_filename = 'sinkbot_model.pkl'
joblib.dump(model, model_filename)
print(f"💾 학습된 모델을 '{model_filename}' 파일로 성공적으로 저장했습니다.")
print("🎉 모든 과정이 완료되었습니다.")