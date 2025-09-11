# trainer.py (Hybrid: PostgreSQL for Cloudtype, SQLite for Local Dev)

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
import psycopg2
import sqlite3

print("🤖 AI 모델 학습을 시작합니다.")

# --- ⭐️ 1. 실행 환경 감지 및 DB 설정 ⭐️ ---
IS_CLOUD_ENV = 'DB_HOST' in os.environ

if IS_CLOUD_ENV:
    # Cloudtype 환경 (PostgreSQL)
    dsn = f"host={os.environ.get('DB_HOST')} port={os.environ.get('DB_PORT')} dbname={os.environ.get('DB_NAME')} user={os.environ.get('DB_USER')} password={os.environ.get('DB_PASSWORD')}"
    MODEL_DIR = "/data" 
    print("Cloud PostgreSQL 모드로 실행합니다.")
else:
    # 로컬 개발 환경 (SQLite)
    DB_FILE = "sinkbot_data.db"
    MODEL_DIR = "." # 현재 폴더
    print(f"Local SQLite 모드 ('{DB_FILE}')로 실행합니다.")

model_filename = os.path.join(MODEL_DIR, "sinkbot_model.pkl")

# 2. 데이터베이스에서 데이터 불러오기
try:
    if IS_CLOUD_ENV:
        conn = psycopg2.connect(dsn)
    else:
        conn = sqlite3.connect(DB_FILE)
    
    df = pd.read_sql_query("SELECT * FROM displacement ORDER BY timestamp", conn)
    conn.close()
    print(f"✅ 데이터베이스에서 {len(df)}개의 데이터를 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"❌ 데이터베이스 로딩 실패: {e}")
    exit()

# 3. 데이터 전처리 및 특징 추출
# ... (rest of the trainer logic is the same) ...
if len(df) < 20:
    print(f"❌ 학습을 위한 데이터가 부족합니다. 최소 20개 이상의 데이터가 필요합니다. (현재: {len(df)}개)")
    exit()

# ... (data processing logic) ...

# 4. AI 모델 학습
# ... (model training logic) ...

# 5. 학습된 모델 파일로 저장
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
joblib.dump(model, model_filename)
print(f"💾 학습된 모델을 '{model_filename}' 파일로 성공적으로 저장했습니다.")
print("🎉 모든 과정이 완료되었습니다.")

