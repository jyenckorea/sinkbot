# 파일명: dashboard.py (v4.3 - 아이콘 수정)
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import time
import plotly.express as px
import folium
from streamlit_folium import st_folium
import joblib
import os

st.set_page_config(layout="wide")
st.title("🛰️ SinkBot AI 관제 대시_보드 (v4.3)")

# 빈 슬롯(placeholder) 생성
placeholder = st.empty()

# 사이드바
with st.sidebar:
    st.header("⚙️ 모델 관리")
    if st.button("새 AI 모델 불러오기 (Reload)"):
        st.cache_resource.clear()
        st.toast("새로운 AI 모델을 불러왔습니다!", icon="🤖")

db_file = 'sinkbot_data.db'
model_file = 'sinkbot_model.pkl'

# AI 모델 로딩
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path): return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        return None

# 데이터 처리 함수
def process_data(df):
    if len(df) < 1: return None, None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    reference_point = df.iloc[0]
    if len(df) > 0:
        df['delta_z'] = df['z'] - reference_point['z']
        df['distance_3d'] = np.sqrt(
            (df['x'] - reference_point['x'])**2 +
            (df['y'] - reference_point['y'])**2 +
            (df['z'] - reference_point['z'])**2
        )
    return df, reference_point

model = load_model(model_file)

with placeholder.container():
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query("SELECT * FROM displacement", conn)
        conn.close()

        st.header("🚨 시스템 상태")
        if model is None:
            st.warning(f"'{model_file}' AI 모델 파일을 찾을 수 없습니다.", icon="⚠️")
        elif len(df) < 2:
            st.info("데이터가 충분히 쌓이면 AI 예측을 시작합니다.")
        else:
            df_for_pred, _ = process_data(df.copy())
            latest_features = df_for_pred.iloc[-1][['delta_z', 'distance_3d']]
            prediction = model.predict(latest_features.values.reshape(1, -1))
            if prediction[0] == -1:
                st.error("🚨 위험: AI가 이상 신호를 감지했습니다!", icon="🚨")
            else:
                st.success("✔️ 정상: 시스템이 안정적으로 운영 중입니다.", icon="✔️")
        st.markdown("---")

        st.header("📈 실시간 변위 분석")
        df_processed, reference_point = process_data(df.copy())

        if df_processed is not None:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📍 측정 위치")
                latest_location = df_processed.iloc[-1]
                lat, lon = latest_location['y'], latest_location['x']
                m = folium.Map(location=[lat, lon], zoom_start=16)
                folium.Marker(
                    [lat, lon], popup=f"<b>SinkBot</b><br>위도: {lat:.5f}<br>경도: {lon:.5f}",
                    tooltip="현재 측정 위치", 
                    # ⭐️ 아이콘을 'arrows-v'로 변경했습니다 ⭐️
                    icon=folium.Icon(color='red', icon='arrows-v', prefix='fa')
                ).add_to(m)
                st_folium(m, height=350, use_container_width=True)
                st.subheader("📊 데이터 요약")
                st.info(f"기준점: Y={reference_point['y']:.5f}, X={reference_point['x']:.5f}, Z={reference_point['z']:.2f}")
                if 'delta_z' in df_processed.columns and len(df_processed) > 1:
                    st.metric("현재 수직 침하량", f"{df_processed.iloc[-1]['delta_z']:.4f} m", f"{df_processed.iloc[-1]['delta_z'] - df_processed.iloc[-2]['delta_z']:.4f} m")
            with col2:
                st.subheader("📉 시간에 따른 수직 침하량(Z) 변화")
                if 'delta_z' in df_processed.columns and len(df_processed) > 1:
                    fig = px.line(df_processed, x='timestamp', y='delta_z', title='수직 침하량(delta_z) 시계열 그래프')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("데이터가 2개 이상 쌓이면 침하량 그래프가 표시됩니다.")
        
        st.markdown("---")
        st.subheader("🗃️ 원본 데이터 로그")
        st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")

# 루프의 맨 마지막에 위치
time.sleep(5)
st.rerun()