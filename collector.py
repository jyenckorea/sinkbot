# collector.py (Hybrid: PostgreSQL for Cloudtype, SQLite for Local Dev)

from flask import Flask, request, jsonify
import os
import datetime
import psycopg2
import sqlite3

# --- ⭐️ 1. 실행 환경 감지 및 DB 설정 ⭐️ ---
IS_CLOUD_ENV = 'DB_HOST' in os.environ

if IS_CLOUD_ENV:
    # Cloudtype 환경 (PostgreSQL)
    dsn = f"host={os.environ.get('DB_HOST')} port={os.environ.get('DB_PORT')} dbname={os.environ.get('DB_NAME')} user={os.environ.get('DB_USER')} password={os.environ.get('DB_PASSWORD')}"
else:
    # 로컬 개발 환경 (SQLite)
    DB_FILE = "sinkbot_data.db"

def init_db():
    """서버 시작 시 DB 테이블을 준비합니다."""
    if IS_CLOUD_ENV:
        # PostgreSQL 테이블 준비
        try:
            conn = psycopg2.connect(dsn)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS displacement (
                    id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL,
                    x DOUBLE PRECISION NOT NULL, y DOUBLE PRECISION NOT NULL, z DOUBLE PRECISION NOT NULL,
                    tilt_x DOUBLE PRECISION, tilt_y DOUBLE PRECISION
                );
            """)
            conn.commit()
            cur.close()
            conn.close()
            print("✅ Cloud PostgreSQL database is ready.")
        except Exception as e:
            print(f"❌ Cloud PostgreSQL initialization failed: {e}")
    else:
        # SQLite 테이블 준비
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS displacement (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
                tilt_x REAL, tilt_y REAL
            );
        """)
        conn.commit()
        conn.close()
        print(f"✅ Local SQLite database '{DB_FILE}' is ready.")

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def receive_data():
    """컨트롤러에서 보낸 JSON 데이터를 받아 DB에 저장합니다."""
    try:
        data = request.get_json()
        if not all(k in data for k in ['x', 'y', 'z', 'tilt_x', 'tilt_y']):
            return jsonify({"status": "error", "message": "Missing required data fields"}), 400

        x, y, z, tilt_x, tilt_y = data['x'], data['y'], data['z'], data['tilt_x'], data['tilt_y']
        
        if IS_CLOUD_ENV:
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            conn = psycopg2.connect(dsn)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO displacement (timestamp, x, y, z, tilt_x, tilt_y) VALUES (%s, %s, %s, %s, %s, %s)",
                (timestamp, x, y, z, tilt_x, tilt_y)
            )
        else: # SQLite
            timestamp = datetime.datetime.now().isoformat()
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO displacement (timestamp, x, y, z, tilt_x, tilt_y) VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, x, y, z, tilt_x, tilt_y)
            )

        conn.commit()
        cur.close()
        conn.close()
        
        print(f"✔️ Data received and stored successfully.")
        return jsonify({"status": "success"}), 201

    except Exception as e:
        print(f"❌ Error during data insertion: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

