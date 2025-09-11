# 파일명: collector.py
from flask import Flask, request, jsonify
import datetime
import sqlite3
import os

# 데이터베이스 파일이 있는 경로 확인
db_file = 'sinkbot_data.db'

# 서버 시작 시 DB 및 테이블 생성
if not os.path.exists(db_file):
    print(f"'{db_file}'가 존재하지 않아 새로 생성하고 테이블을 만듭니다.")
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE displacement (
            timestamp TEXT,
            x REAL,
            y REAL,
            z REAL
        )
    ''')
    conn.commit()
    conn.close()

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def receive_data():
    """ SinkBot 컨트롤러에서 보낸 JSON 데이터를 받아 DB에 저장합니다. """
    try:
        # 데이터베이스 연결
        conn = sqlite3.connect(db_file)
        c = conn.cursor()

        # 데이터 수신 및 시간 기록
        data = request.get_json()
        x, y, z = data['x'], data['y'], data['z']
        timestamp = datetime.datetime.now().isoformat()

        # 데이터베이스에 저장
        c.execute("INSERT INTO displacement (timestamp, x, y, z) VALUES (?, ?, ?, ?)",
                  (timestamp, x, y, z))
        conn.commit()
        
        print(f"✔️  데이터 수신 및 저장 완료: {timestamp}, x={x}, y={y}, z={z}")
        
        return jsonify({"status": "success", "data": data}), 201
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    # 외부에서 접근 가능하도록 host='0.0.0.0'으로 설정
    app.run(host='0.0.0.0', port=5000, debug=True)