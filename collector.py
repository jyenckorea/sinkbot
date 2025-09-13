# collector.py (AI 모델 테이블 생성 기능 추가)
import os
import time
import json
import psycopg2
from flask import Flask, request, jsonify

# --- 1. 실행 환경 감지 및 DB 연결 ---
IS_CLOUD_ENV = 'DB_HOST' in os.environ

print("🤖 SinkBot Data Collector를 시작합니다.")

conn = None
if IS_CLOUD_ENV:
    print("Cloud PostgreSQL 모드로 실행합니다.")
    try:
        dsn = f"host={os.environ.get('DB_HOST')} port={os.environ.get('DB_PORT')} dbname={os.environ.get('DB_NAME')} user={os.environ.get('DB_USER')} password={os.environ.get('DB_PASSWORD')}"
        conn = psycopg2.connect(dsn)
        print("✅ Cloud PostgreSQL에 성공적으로 연결되었습니다.")
    except Exception as e:
        print(f"❌ Cloud PostgreSQL 연결 실패: {e}")
        # 클라우드 환경에서 DB 연결 실패는 심각한 문제이므로, 프로그램을 종료하여 Cloudtype이 재시작하도록 유도
        exit(1)
else:
    print("❌ 이 스크립트는 Cloudtype 배포 전용입니다. 로컬 테스트를 위해서는 SQLite 버전의 collector를 사용하세요.")
    exit(1)


# --- 2. 데이터베이스 테이블 생성 ---
def create_tables():
    """프로그램 시작 시 필요한 테이블들을 생성합니다."""
    with conn.cursor() as cur:
        # 기존 displacement 테이블 생성
        cur.execute("""
            CREATE TABLE IF NOT EXISTS displacement (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                x DOUBLE PRECISION,
                y DOUBLE PRECISION,
                z DOUBLE PRECISION,
                tilt_x DOUBLE PRECISION,
                tilt_y DOUBLE PRECISION
            );
        """)
        # AI 모델 저장을 위한 새로운 테이블 생성
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_models (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                model_name VARCHAR(255) UNIQUE NOT NULL,
                model_data BYTEA NOT NULL
            );
        """)
    conn.commit()
    print("✅ 데이터베이스 테이블이 준비되었습니다 (displacement, ai_models).")

# --- 3. Flask 웹 서버 설정 ---
app = Flask(__name__)

@app.route('/data', methods=['POST'])
def receive_data():
    """컨트롤러로부터 센서 데이터를 받아 DB에 저장합니다."""
    try:
        data = request.get_json()
        if not all(k in data for k in ['x', 'y', 'z', 'tilt_x', 'tilt_y']):
            return jsonify({"status": "error", "message": "Missing required data fields"}), 400

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO displacement (x, y, z, tilt_x, tilt_y)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (data['x'], data['y'], data['z'], data['tilt_x'], data['tilt_y'])
            )
        conn.commit()
        
        return jsonify({"status": "success"}), 201
    
    except Exception as e:
        conn.rollback() # 오류 발생 시 트랜잭션 되돌리기
        print(f"❌ 데이터 저장 오류: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크를 위한 경로입니다."""
    try:
        # DB 연결이 살아있는지 간단히 확인
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return "Healthy", 200
    except Exception:
        # DB 연결에 문제가 생기면 unhealthy 상태 반환
        return "Unhealthy", 503

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    create_tables()
    # Cloudtype 환경에서는 Gunicorn 같은 WSGI 서버를 사용하는 것이 더 안정적입니다.
    # 여기서는 간단하게 Flask 개발 서버를 사용합니다.
    app.run(host='0.0.0.0', port=5000)

