# 파일명: dashboard.py (v1.1 - 최종 안정화 버전)
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
# ⭐️ 버전을 v1.1로 수정
st.title("🛰️ SinkBot AI 관제 대시보드 (v1.1)")

# --- Session State 초기화 ---
# 앱이 처음 실행될 때 각 위젯의 기본 상태를 한 번만 설정합니다.
if 'time_range' not in st.session_state:
    st.session_state.time_range = '전체'
if 'agg_level' not in st.session_state:
    st.session_state.agg_level = '원본 데이터'
if 'manual_axis' not in st.session_state:
    st.session_state.manual_axis = False
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 15

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 제어판")
    # ⭐️ 1. 모든 위젯이 Session State를 직접 읽고 쓰도록 key를 부여합니다.
    st.toggle("자동 새로고침 활성화 ⏱️", key="auto_refresh")
    
    refresh_options = {"5초": 5, "10초": 10, "15초": 15, "30초": 30, "1분": 60, "5분": 300, "10분": 600}
    
    # 현재 session_state 값에 맞는 라벨을 찾아 index로 사용
    option_labels = list(refresh_options.keys())
    option_values = list(refresh_options.values())
    try:
        current_index = option_values.index(st.session_state.refresh_interval)
    except ValueError:
        current_index = 2 # 기본값(15초)의 인덱스

    selected_label = st.selectbox(
        "새로고침 간격",
        options=option_labels,
        index=current_index
    )
    # 선택된 라벨에 해당하는 숫자 값을 session_state에 저장
    st.session_state.refresh_interval = refresh_options[selected_label]

    if st.session_state.auto_refresh:
        st.caption(f"현재 상태: {st.session_state.refresh_interval}초마다 자동 새로고침 중...")
    else:
        st.caption("현재 상태: 수동 새로고침 모드.")
        
    if st.button("데이터 수동 새로고침 🔄"):
        st.cache_data.clear()
        st.toast("최신 데이터를 불러왔습니다!", icon="✅")
        st.rerun()
        
    st.markdown("---")
    st.header("🤖 모델 관리")
    if st.button("새 AI 모델 불러오기 (Reload)"):
        st.cache_resource.clear()
        st.toast("새로운 AI 모델을 불러왔습니다!", icon="🤖")
        st.rerun()

db_file = 'sinkbot_data.db'
model_file = 'sinkbot_model.pkl'

# (함수 정의는 이전과 동일)
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path): return None
    try: return joblib.load(model_path)
    except Exception as e: st.error(f"모델 로딩 중 오류 발생: {e}")
    return None

@st.cache_data(ttl=st.session_state.refresh_interval)
def load_data(db_path):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM displacement", conn)
        conn.close()
        return df
    except Exception: return pd.DataFrame()

def process_data(df):
    if df.empty or len(df) < 1: return None, None
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy = df_copy.sort_values(by='timestamp').reset_index(drop=True)
    reference_point = df_copy.iloc[0]
    if len(df_copy) > 0:
        df_copy['delta_z'] = df_copy['z'] - reference_point['z']
        df_copy['distance_3d'] = np.sqrt((df_copy['x'] - reference_point['x'])**2 + (df_copy['y'] - reference_point['y'])**2 + (df_copy['z'] - reference_point['z'])**2)
    return df_copy, reference_point

model = load_model(model_file)
df = load_data(db_file)

# --- 메인 대시보드 UI ---
try:
    st.header("🚨 시스템 상태")
    if model is None: st.warning(f"'{model_file}' AI 모델 파일을 찾을 수 없습니다.", icon="⚠️")
    elif df.empty or len(df) < 2: st.info("데이터가 충분히 쌓이면 AI 예측을 시작합니다.")
    else:
        df_for_pred, _ = process_data(df)
        latest_features_df = df_for_pred.tail(1)[['delta_z', 'distance_3d']]
        prediction = model.predict(latest_features_df)
        if prediction[0] == -1: st.error("🚨 위험: AI가 이상 신호를 감지했습니다!", icon="🚨")
        else: st.success("✔️ 정상: 시스템이 안정적으로 운영 중입니다.", icon="✔️")
    st.markdown("---")

    st.header("📈 실시간 변위 분석")
    df_processed, reference_point = process_data(df)
    if df_processed is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📍 측정 위치")
            latest_location = df_processed.iloc[-1]
            lat, lon = latest_location['y'], latest_location['x']
            m = folium.Map(location=[lat, lon], zoom_start=16)
            folium.Marker([lat, lon], popup=f"<b>SinkBot</b><br>위도: {lat:.5f}<br>경도: {lon:.5f}", tooltip="현재 측정 위치", icon=folium.Icon(color='red', icon='arrows-v', prefix='fa')).add_to(m)
            st_folium(m, height=350, use_container_width=True, key="folium_map_final")
            st.subheader("📊 데이터 요약")
            st.info(f"기준점: Y={reference_point['y']:.5f}, X={reference_point['x']:.5f}, Z={reference_point['z']:.2f}")
            if 'delta_z' in df_processed.columns and len(df_processed) > 1:
                st.metric("현재 수직 침하량", f"{df_processed.iloc[-1]['delta_z']:.4f} m", f"{df_processed.iloc[-1]['delta_z'] - df_processed.iloc[-2]['delta_z']:.4f} m")
        with col2:
            st.subheader("📉 시간에 따른 수직 침하량(Z) 변화")
            if 'delta_z' in df_processed.columns and len(df_processed) > 1:
                graph_col1, graph_col2 = st.columns(2)
                with graph_col1:
                    # ⭐️ 2. 위젯의 상태를 Session State가 직접 관리하도록 key 부여
                    st.selectbox("시간 범위 선택", options=['전체', '최근 24시간', '최근 6시간', '최근 1시간'], key="time_range")
                with graph_col2:
                    st.selectbox("데이터 집계 방식", options=['원본 데이터', '시간별 평균', '일별 평균'], key="agg_level")
                
                df_to_plot = df_processed.copy()
                now = pd.to_datetime('now', utc=True).tz_convert('Asia/Seoul')
                if st.session_state.time_range == '최근 24시간': df_to_plot = df_to_plot[df_to_plot['timestamp'] >= (now - pd.Timedelta(hours=24))]
                elif st.session_state.time_range == '최근 6시간': df_to_plot = df_to_plot[df_to_plot['timestamp'] >= (now - pd.Timedelta(hours=6))]
                elif st.session_state.time_range == '최근 1시간': df_to_plot = df_to_plot[df_to_plot['timestamp'] >= (now - pd.Timedelta(hours=1))]
                
                if st.session_state.agg_level != '원본 데이터' and not df_to_plot.empty:
                    df_to_plot = df_to_plot.set_index('timestamp')
                    if st.session_state.agg_level == '시간별 평균': df_to_plot = df_to_plot[['delta_z']].resample('H').mean().reset_index()
                    elif st.session_state.agg_level == '일별 평균': df_to_plot = df_to_plot[['delta_z']].resample('D').mean().reset_index()

                st.checkbox("그래프 축 범위 수동 조절", key="manual_axis")
                
                axis_col1, axis_col2 = st.columns(2)
                with axis_col1:
                    st.markdown("**X축 범위 (날짜)**")
                    min_date, max_date = df_to_plot['timestamp'].min().date(), df_to_plot['timestamp'].max().date()
                    user_start_date = st.date_input("시작 날짜:", value=min_date, min_value=min_date, max_value=max_date, disabled=not st.session_state.manual_axis)
                    user_end_date = st.date_input("종료 날짜:", value=max_date, min_value=min_date, max_value=max_date, disabled=not st.session_state.manual_axis)
                with axis_col2:
                    st.markdown("**Y축 범위 (침하량)**")
                    min_val, max_val = float(df_to_plot['delta_z'].min()), float(df_to_plot['delta_z'].max())
                    user_min = st.number_input("최소값:", value=min_val, format="%.4f", disabled=not st.session_state.manual_axis)
                    user_max = st.number_input("최대값:", value=max_val, format="%.4f", disabled=not st.session_state.manual_axis)

                x_axis_range, y_axis_range = None, None
                if st.session_state.manual_axis:
                    if user_start_date <= user_end_date:
                        x_axis_range = [pd.to_datetime(user_start_date), pd.to_datetime(user_end_date) + pd.Timedelta(days=1)]
                    else: st.warning("시작 날짜는 종료 날짜보다 이전이어야 합니다.", icon="⚠️")
                    if user_min < user_max: y_axis_range = (user_min, user_max)
                    else: st.warning("최소값은 최대값보다 작아야 합니다.", icon="⚠️")

                if not df_to_plot.empty:
                    fig = px.line(df_to_plot, x='timestamp', y='delta_z', title=f"{st.session_state.time_range} 데이터 ({st.session_state.agg_level})", labels={'timestamp': '시간', 'delta_z': '침하량 (m)'})
                    if x_axis_range: fig.update_xaxes(range=x_axis_range)
                    if y_axis_range: fig.update_yaxes(range=y_axis_range)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning("선택된 기간에 해당하는 데이터가 없습니다.", icon="⚠️")
            else: st.info("데이터가 2개 이상 쌓이면 침하량 그래프가 표시됩니다.", icon="ℹ️")

    st.markdown("---")
    st.subheader("🗃️ 원본 데이터 로그")
    st.dataframe(df.tail(20).iloc[::-1], height=735)

except Exception as e:
    st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}", icon="🔥")

# --- ⭐️ 3. st.rerun()을 이용한 안정적인 새로고침 복원
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()

