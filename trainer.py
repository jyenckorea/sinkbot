# trainer.py (NameError 수정 버전)
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
import psycopg2
import sqlite3

# --- 1. 실행 환경 감지 및 DB/모델 경로 설정 ---
IS_CLOUD_ENV = 'DB_HOST' in os.environ

print("🤖 AI 모델 학습을 시작합니다.")

if IS_CLOUD_ENV:
    # Cloudtype 환경 (PostgreSQL)
    print("Cloud PostgreSQL 모드로 실행합니다.")
    dsn = f"host={os.environ.get('DB_HOST')} port={os.environ.get('DB_PORT')} dbname={os.environ.get('DB_NAME')} user={os.environ.get('DB_USER')} password={os.environ.get('DB_PASSWORD')}"
    MODEL_DIR = "/data" # Cloudtype의 영구 디스크 경로
else:
    # 로컬 개발 환경 (SQLite)
    print("Local SQLite 모드 ('sinkbot_data.db')로 실행합니다.")
    DB_FILE = "sinkbot_data.db"
    MODEL_DIR = "." # 현재 폴더

# --- 2. 데이터베이스에서 데이터 불러오기 ---
def load_data():
    """환경에 따라 적절한 DB에서 데이터를 불러옵니다."""
    try:
        if IS_CLOUD_ENV:
            conn = psycopg2.connect(dsn)
        else:
            if not os.path.exists(DB_FILE):
                print(f"❌ 로컬 DB 파일 '{DB_FILE}'을 찾을 수 없습니다. collector.py를 먼저 실행하여 데이터를 수집해주세요.")
                return None
            conn = sqlite3.connect(DB_FILE)
        
        df = pd.read_sql_query("SELECT * FROM displacement ORDER BY timestamp", conn)
        conn.close()
        print(f"✅ 데이터베이스에서 {len(df)}개의 데이터를 성공적으로 불러왔습니다.")
        return df
    except Exception as e:
        print(f"❌ 데이터베이스 로딩 실패: {e}")
        return None

# --- 3. 데이터 전처리 및 특징 추출 함수 ---
def process_data(df):
    """AI 학습에 사용할 특징(feature)을 계산합니다."""
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy = df_copy.sort_values(by='timestamp').reset_index(drop=True)
    reference_point = df_copy.iloc[0]
    
    # 기준점 대비 변위 및 변화량 계산
    df_copy['delta_z'] = df_copy['z'] - reference_point['z']
    df_copy['distance_3d'] = np.sqrt((df_copy['x'] - reference_point['x'])**2 + (df_copy['y'] - reference_point['y'])**2 + (df_copy['z'] - reference_point['z'])**2)
    df_copy['tilt_magnitude'] = np.sqrt(df_copy['tilt_x']**2 + df_copy['tilt_y']**2)
    df_copy['delta_tilt'] = df_copy['tilt_magnitude'] - df_copy.iloc[0]['tilt_magnitude']
    
    return df_copy

# --- 메인 실행 로직 ---
df = load_data()

# 데이터가 성공적으로 로드되었고, 20개 이상인지 확인
if df is not None and len(df) >= 20:
    print("✅ 데이터 전처리 및 특징 추출을 시작합니다.")
    df_processed = process_data(df)

    # 4. AI 모델 학습
    features = df_processed[['delta_z', 'distance_3d', 'delta_tilt']]
    
    contamination_level = 0.01 
    print(f"모델 민감도(Contamination)를 {contamination_level * 100}%로 설정합니다.")
    
    model = IsolationForest(contamination=contamination_level, random_state=42)

    print("⏳ AI 모델 학습 중...")
    model.fit(features)
    print("✅ 모델 학습 완료!")

    # ⭐️ 5. 모델 저장 (학습이 성공했을 때만 실행되도록 if 블록 안으로 이동) ⭐️
    try:
        if not os.path.exists(MODEL_DIR) and IS_CLOUD_ENV:
            os.makedirs(MODEL_DIR)
        
        model_filename = os.path.join(MODEL_DIR, "sinkbot_model.pkl")
        joblib.dump(model, model_filename)
        print(f"💾 학습된 모델을 '{model_filename}' 파일로 성공적으로 저장했습니다.")
        print("🎉 모든 과정이 완료되었습니다.")
    except Exception as e:
        print(f"❌ 모델 파일 저장 실패: {e}")

else:
    if df is not None:
        print(f"❌ 학습을 위한 데이터가 부족합니다. 최소 20개 이상의 데이터가 필요합니다. (현재: {len(df)}개)")
    # df가 None인 경우는 load_data 함수에서 이미 오류 메시지를 출력함

