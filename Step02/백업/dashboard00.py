# 파일명: dashboard.py (v3.1 - 지도 기능 추가)
import streamlit as st
import pandas as pd
import sqlite3
import time
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🛰️ SinkBot 분석 대시보드 (v3.1)")

db_file = 'sinkbot_data.db'

def process_data(df):
    # (이전과 동일)
    if len(df) < 1: # 데이터가 1개만 있어도 위치 표시는 가능하도록 수정
        st.warning("분석을 위한 데이터가 부족합니다. (최소 1개 필요)", icon="⚠️")
        return None, None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    reference_point = df.iloc[0]
    if len(df) > 1:
        df['delta_x'] = df['x'] - reference_point['x']
        df['delta_y'] = df['y'] - reference_point['y']
        df['delta_z'] = df['z'] - reference_point['z']
    return df, reference_point

# --- 코드 실행 ---
try:
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query("SELECT * FROM displacement", conn)
    conn.close()

    st.header("📈 실시간 변위 분석")
    
    df_processed, reference_point = process_data(df.copy())

    if df_processed is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📍 측정 위치")
            # --- ⭐️ 지도 기능 추가 시작 ⭐️ ---
            latest_location = df_processed.iloc[-1]
            # st.map은 'lat', 'lon' 컬럼 이름이 필요합니다.
            map_data = pd.DataFrame({
                'lat': [latest_location['y']], # Y좌표가 위도(lat)라고 가정
                'lon': [latest_location['x']]  # X좌표가 경도(lon)라고 가정
            })
            st.map(map_data, zoom=15)
            # --- ⭐️ 지도 기능 추가 끝 ⭐️ ---

            st.subheader("📊 데이터 요약")
            st.info(f"""
                **기준점 좌표 (최초 측정값):**
                - 위도(Y): `{reference_point['y']:.5f}`
                - 경도(X): `{reference_point['x']:.5f}`
                - 고도(Z): `{reference_point['z']:.2f}`
            """)
            
            # (이하 생략, 기존 코드와 동일)
            if 'delta_z' in df_processed.columns:
                latest_data = df_processed.iloc[-1]
                st.metric(
                    label="현재 수직 침하량 (기준점 대비)",
                    value=f"{latest_data['delta_z']:.4f} m",
                    delta=f"{latest_data['delta_z'] - df_processed.iloc[-2]['delta_z']:.4f} m (직전 대비)" if len(df_processed) > 1 else "0 m"
                )

        with col2:
            st.subheader("📉 시간에 따른 수직 침하량(Z) 변화")
            if 'delta_z' in df_processed.columns and len(df_processed) > 1:
                fig = px.line(
                    df_processed, x='timestamp', y='delta_z', title='수직 침하량(delta_z) 시계열 그래프',
                    labels={'timestamp': '시간', 'delta_z': '침하량 (m)'}
                )
                fig.update_layout(title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("데이터가 2개 이상 쌓이면 침하량 그래프가 표시됩니다.")
    
    st.markdown("---")
    st.subheader("🗃️ 원본 데이터 로그")
    st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)

except Exception as e:
    st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")

time.sleep(5)
st.rerun()