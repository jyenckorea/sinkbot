# collector.py 
import os
import psycopg2
from flask import Flask, request, jsonify

# DB 연결 정보는 환경 변수에서 가져옴
dsn = f"host={os.environ.get('DB_HOST')} port={os.environ.get('DB_PORT')} dbname={os.environ.get('DB_NAME')} user={os.environ.get('DB_USER')} password={os.environ.get('DB_PASSWORD')}"

def get_db_connection():
    return psycopg2.connect(dsn)

def create_tables():
    """프로그램 시작 시 필요한 테이블들을 생성합니다."""
    conn = get_db_connection()
    with conn.cursor() as cur:
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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_models (
                id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW(),
                model_name VARCHAR(255) UNIQUE NOT NULL, model_data BYTEA NOT NULL);
        """)
    conn.commit()
    conn.close()

# Flask 앱을 시작하기 전에 테이블 생성 함수를 맨 처음 한번만 실행합니다.
print("🤖 SinkBot Data Collector를 시작합니다.")
try:
    print("데이터베이스 테이블 생성을 시도합니다...")
    create_tables()
    print("✅ 데이터베이스 테이블이 준비되었습니다.")
except Exception as e:
    print(f"❌ 테이블 생성 중 오류 발생: {e}")

# app = Flask(__name__)
app_v2 = Flask(__name__)

@app.route('/data', methods=['POST'])
def receive_data():
    conn = get_db_connection()
    try:
        data = request.get_json()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO displacement (x, y, z, tilt_x, tilt_y) VALUES (%s, %s, %s, %s, %s)",
                (data['x'], data['y'], data['z'], data['tilt_x'], data['tilt_y'])
            )
        conn.commit()
        return jsonify({"status": "success"}), 201
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()


@app.route('/health', methods=['GET'])
def health_check():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return "Healthy", 200
    except Exception:
        return "Unhealthy", 503
