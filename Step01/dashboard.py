# 파일명: dashboard.py
import streamlit as st
import pandas as pd
import sqlite3
import time

st.set_page_config(layout="wide") # 넓은 화면 사용
st.title("🛰️ SinkBot 실시간 데이터 모니터링 대시보드")

db_file = 'sinkbot_data.db'

# 실시간 데이터 현황을 표시할 영역
placeholder = st.empty()

# 5초마다 루프를 돌면서 데이터베이스를 확인하고 화면을 업데이트
while True:
    try:
        conn = sqlite3.connect(db_file)
        
        # container를 사용해 UI 요소들을 묶음
        with placeholder.container():
            
            # 총 데이터 건수와 최신 데이터 시간 조회
            try:
                total_count = pd.read_sql_query("SELECT COUNT(*) FROM displacement", conn).iloc[0, 0]
                latest_time = pd.read_sql_query("SELECT MAX(timestamp) FROM displacement", conn).iloc[0, 0]
            except (pd.io.sql.DatabaseError, IndexError):
                total_count = 0
                latest_time = "N/A"

            # 2개의 컬럼 생성
            kpi1, kpi2 = st.columns(2)
            kpi1.metric(label="총 수신 데이터 📦", value=f"{total_count} 건")
            kpi2.metric(label="최근 수신 시간 🕒", value=str(latest_time).split('.')[0])
            
            st.markdown("---") # 구분선

            # 최근 10개 데이터 조회 및 표시
            st.subheader("최근 수신 데이터 로그")
            latest_data = pd.read_sql_query("SELECT * FROM displacement ORDER BY timestamp DESC LIMIT 10", conn)
            st.dataframe(latest_data, use_container_width=True)
        
        conn.close()
        time.sleep(5) # 5초 대기

    except FileNotFoundError:
        with placeholder.container():
            st.warning(f"'{db_file}' 파일을 찾을 수 없습니다. 데이터 수집 서버(`collector.py`)를 먼저 실행해 주세요.", icon="⚠️")
        time.sleep(5)
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        time.sleep(5)