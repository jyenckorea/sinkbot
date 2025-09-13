# trainer.py (AI 모델을 DB에 저장하도록 수정)
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
import psycopg2
import io

# --- 1. 실행 환경 감지 및 DB 연결 ---
IS_CLOUD_ENV = 'DB_HOST' in os.environ

print("🤖 AI 모델 학습을 시작합니다.")

conn = None
if IS_CLOUD_ENV:
    print("Cloud PostgreSQL 모드로 실행합니다.")
    try:
        dsn = f"host={os.environ.get('DB_HOST')} port={os.environ.get('DB_PORT')} dbname={os.environ.get('DB_NAME')} user={os.environ.get('DB_USER')} password={os.environ.get('DB_PASSWORD')}"
        conn = psycopg2.connect(dsn)
        print("✅ Cloud PostgreSQL에 성공적으로 연결되었습니다.")
    except Exception as e:
        print(f"❌ Cloud PostgreSQL 연결 실패: {e}")
        exit(1)
else:
    print("❌ 이 스크립트는 Cloudtype 배포 전용입니다. 로컬 테스트를 위해서는 SQLite 버전의 trainer를 사용하세요.")
    exit(1)

# --- 2. 데이터베이스에서 데이터 불러오기 ---
def load_data():
    try:
        df = pd.read_sql_query("SELECT * FROM displacement ORDER BY timestamp", conn)
        print(f"✅ 데이터베이스에서 {len(df)}개의 데이터를 성공적으로 불러왔습니다.")
        return df
    except Exception as e:
        print(f"❌ 데이터베이스 로딩 실패: {e}")
        return None

# --- 3. 데이터 전처리 및 특징 추출 함수 ---
def process_data(df):
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy = df_copy.sort_values(by='timestamp').reset_index(drop=True)
    reference_point = df_copy.iloc[0]
    
    df_copy['delta_z'] = df_copy['z'] - reference_point['z']
    df_copy['distance_3d'] = np.sqrt((df_copy['x'] - reference_point['x'])**2 + (df_copy['y'] - reference_point['y'])**2 + (df_copy['z'] - reference_point['z'])**2)
    df_copy['tilt_magnitude'] = np.sqrt(df_copy['tilt_x']**2 + df_copy['tilt_y']**2)
    df_copy['delta_tilt'] = df_copy['tilt_magnitude'] - df_copy.iloc[0]['tilt_magnitude']
    
    return df_copy

# --- 메인 실행 로직 ---
df = load_data()

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

    # ⭐️ 5. 학습된 모델을 파일이 아닌 DB에 저장 ⭐️
    try:
        # joblib을 이용해 모델을 메모리 상의 바이트 데이터로 변환
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_bytes = buffer.read()
        
        # ai_models 테이블에 모델 저장 (UPSERT: 없으면 INSERT, 있으면 UPDATE)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_models (model_name, model_data)
                VALUES (%s, %s)
                ON CONFLICT (model_name) DO UPDATE
                SET model_data = EXCLUDED.model_data,
                    created_at = NOW();
                """,
                ('sinkbot_model', model_bytes) # 모델 이름은 'sinkbot_model'로 고정
            )
        conn.commit()
        print(f"💾 학습된 모델을 데이터베이스 'ai_models' 테이블에 성공적으로 저장(업데이트)했습니다.")
        print("🎉 모든 과정이 완료되었습니다.")

    except Exception as e:
        conn.rollback()
        print(f"❌ 모델 DB 저장 실패: {e}")

else:
    if df is not None:
        print(f"❌ 학습을 위한 데이터가 부족합니다. 최소 20개 이상의 데이터가 필요합니다. (현재: {len(df)}개)")

if conn:
    conn.close()

