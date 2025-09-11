# dashboard.py (v1.3 - Z Value Precision Fix)
import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import folium
from streamlit_folium import st_folium
import joblib
import os
import psycopg2
import sqlite3

# --- ⭐️ 1. 실행 환경 감지 및 DB/모델 경로 설정 ⭐️ ---
IS_CLOUD_ENV = 'DB_HOST' in os.environ

st.set_page_config(layout="wide")
# 대시보드 제목에 현재 실행 환경 표시
st.title(f"🛰️ SinkBot AI 관제 대시보드 ({'Cloud' if IS_CLOUD_ENV else 'Local'}) v1.3")

if IS_CLOUD_ENV:
    # Cloudtype 환경 (PostgreSQL)
    dsn = f"host={os.environ.get('DB_HOST')} port={os.environ.get('DB_PORT')} dbname={os.environ.get('DB_NAME')} user={os.environ.get('DB_USER')} password={os.environ.get('DB_PASSWORD')}"
    MODEL_DIR = "/data"
else:
    # 로컬 개발 환경 (SQLite)
    DB_FILE = "sinkbot_data.db"
    MODEL_DIR = "." # 현재 폴더

model_file = os.path.join(MODEL_DIR, "sinkbot_model.pkl")

# --- Session State 초기화 ---
if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.auto_refresh = True
    st.session_state.refresh_interval = 10
    st.session_state.time_range = '전체'
    st.session_state.agg_level = '원본 데이터'
    st.session_state.manual_axis = False
    st.session_state.chart_type = '수직 침하량(delta_z)'

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 제어판")
    st.toggle("자동 새로고침 활성화 ⏱️", key="auto_refresh")
    st.select_slider(
        "새로고침 간격 (초)",
        options=[5, 10, 15, 30, 60, 180, 300, 600],
        format_func=lambda x: f"{x}초" if x < 60 else f"{x//60}분",
        key="refresh_interval"
    )
    if st.session_state.auto_refresh:
        st.caption(f"현재 상태: {st.session_state.refresh_interval}초마다 자동 새로고침 중...")
    else:
        st.caption("현재 상태: 수동 새로고침 모드.")
    if st.button("데이터 수동 새로고침 🔄", key="manual_refresh_button"):
        st.cache_data.clear()
        st.toast("최신 데이터를 불러왔습니다!", icon="✅")
    st.markdown("---")
    st.header("🤖 모델 관리")
    if st.button("새 AI 모델 불러오기 (Reload)", key="reload_model_button"):
        st.cache_resource.clear()
        st.toast("새로운 AI 모델을 불러왔습니다!", icon="🤖")

# --- 함수 정의 ---
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path): return None
    try: return joblib.load(model_path)
    except Exception as e: st.error(f"모델 로딩 중 오류 발생: {e}"); return None

@st.cache_data(ttl=st.session_state.refresh_interval)
def load_data():
    """환경에 따라 적절한 DB에서 데이터를 불러옵니다."""
    try:
        if IS_CLOUD_ENV:
            conn = psycopg2.connect(dsn)
        else:
            if not os.path.exists(DB_FILE):
                st.warning(f"로컬 DB 파일 '{DB_FILE}'을 찾을 수 없습니다. collector.py를 먼저 실행해주세요.")
                return pd.DataFrame()
            conn = sqlite3.connect(DB_FILE)
        
        df = pd.read_sql_query("SELECT * FROM displacement ORDER BY timestamp", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"DB 데이터 로딩 실패: {e}")
        return pd.DataFrame()

def process_data(df):
    if df.empty or len(df) < 1: return None, None
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy = df_copy.sort_values(by='timestamp').reset_index(drop=True)
    reference_point = df_copy.iloc[0]
    if len(df_copy) > 0:
        df_copy['delta_z'] = df_copy['z'] - reference_point['z']
        df_copy['distance_3d'] = np.sqrt((df_copy['x'] - reference_point['x'])**2 + (df_copy['y'] - reference_point['y'])**2 + (df_copy['z'] - reference_point['z'])**2)
        df_copy['tilt_magnitude'] = np.sqrt(df_copy['tilt_x']**2 + df_copy['tilt_y']**2)
        df_copy['delta_tilt'] = df_copy['tilt_magnitude'] - df_copy.iloc[0]['tilt_magnitude']
    return df_copy, reference_point

model = load_model(model_file)
df = load_data()

# --- 메인 대시보드 UI ---
try:
    st.header("🚨 시스템 상태")
    if model is None: st.warning(f"AI 모델 파일('{model_file}')을 찾을 수 없습니다. trainer.py를 먼저 실행해주세요.", icon="⚠️")
    elif df.empty or len(df) < 2: st.info("데이터가 충분히 쌓이면 AI 예측을 시작합니다.")
    else:
        df_for_pred, _ = process_data(df)
        latest_features_df = df_for_pred.tail(1)[['delta_z', 'distance_3d', 'delta_tilt']]
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
            st_folium(m, height=280, use_container_width=True, key="folium_map_final")
            st.subheader("📊 데이터 요약")
            # --- ⭐️ 이 부분이 수정되었습니다: Z값 소수점을 네 자리까지 표시 ⭐️ ---
            st.info(f"기준점: Y={reference_point['y']:.5f}, X={reference_point['x']:.5f}, Z={reference_point['z']:.4f}, TiltX={reference_point['tilt_x']:.3f}°, TiltY={reference_point['tilt_y']:.3f}°")
            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                if 'delta_z' in df_processed.columns and len(df_processed) > 1:
                    st.metric("현재 수직 침하량", f"{df_processed.iloc[-1]['delta_z']:.4f} m", f"{df_processed.iloc[-1]['delta_z'] - df_processed.iloc[-2]['delta_z']:.4f} m")
            with summary_col2:
                 if 'delta_tilt' in df_processed.columns and len(df_processed) > 1:
                    st.metric("현재 기울기 변화량", f"{df_processed.iloc[-1]['delta_tilt']:.3f}°", f"{df_processed.iloc[-1]['delta_tilt'] - df_processed.iloc[-2]['delta_tilt']:.3f}°")
        with col2:
            st.subheader("📉 시간에 따른 변화")
            if 'delta_z' in df_processed.columns and len(df_processed) > 1:
                CHART_OPTIONS = {
                    "수직 침하량(delta_z)": "delta_z",
                    "기울기 변화량(delta_tilt)": "delta_tilt"
                }
                
                st.selectbox("표시할 그래프 선택:", list(CHART_OPTIONS.keys()), key="chart_type")
                
                selected_column = CHART_OPTIONS[st.session_state.chart_type]
                
                graph_col1, graph_col2 = st.columns(2)
                with graph_col1:
                    st.selectbox("시간 범위 선택", options=['전체', '최근 24시간', '최근 6시간', '최근 1시간'], key="time_range")
                with graph_col2:
                    st.selectbox("데이터 집계 방식", options=['원본 데이터', '시간별 평균', '일별 평균'], key="agg_level")
                
                df_to_plot = df_processed.copy()
                now = pd.to_datetime('now', utc=True)
                if st.session_state.time_range == '최근 24시간': df_to_plot = df_to_plot[df_to_plot['timestamp'] >= (now - pd.Timedelta(hours=24))]
                elif st.session_state.time_range == '최근 6시간': df_to_plot = df_to_plot[df_to_plot['timestamp'] >= (now - pd.Timedelta(hours=6))]
                elif st.session_state.time_range == '최근 1시간': df_to_plot = df_to_plot[df_to_plot['timestamp'] >= (now - pd.Timedelta(hours=1))]
                
                if st.session_state.agg_level != '원본 데이터' and not df_to_plot.empty:
                    df_to_plot = df_to_plot.set_index('timestamp')
                    if st.session_state.agg_level == '시간별 평균': df_to_plot = df_to_plot[[selected_column]].resample('H').mean().reset_index()
                    elif st.session_state.agg_level == '일별 평균': df_to_plot = df_to_plot[[selected_column]].resample('D').mean().reset_index()

                st.checkbox("그래프 축 범위 수동 조절", key="manual_axis")
                axis_col1, axis_col2 = st.columns(2)
                with axis_col1:
                    st.markdown("**X축 범위 (날짜)**")
                    min_date, max_date = df_to_plot['timestamp'].min().date(), df_to_plot['timestamp'].max().date()
                    user_start_date = st.date_input("시작 날짜:", value=min_date, min_value=min_date, max_value=max_date, disabled=not st.session_state.manual_axis, key="start_date")
                    user_end_date = st.date_input("종료 날짜:", value=max_date, min_value=min_date, max_value=max_date, disabled=not st.session_state.manual_axis, key="end_date")
                with axis_col2:
                    st.markdown(f"**Y축 범위 ({st.session_state.chart_type})**")
                    min_val, max_val = float(df_to_plot[selected_column].min()), float(df_to_plot[selected_column].max())
                    user_min = st.number_input("최소값:", value=min_val, format="%.4f", disabled=not st.session_state.manual_axis, key="ymin")
                    user_max = st.number_input("최대값:", value=max_val, format="%.4f", disabled=not st.session_state.manual_axis, key="ymax")

                x_axis_range, y_axis_range = None, None
                if st.session_state.manual_axis:
                    if user_start_date <= user_end_date:
                        x_axis_range = [pd.to_datetime(user_start_date), pd.to_datetime(user_end_date) + pd.Timedelta(days=1)]
                    else: st.warning("시작 날짜는 종료 날짜보다 이전이어야 합니다.", icon="⚠️")
                    if user_min < user_max: y_axis_range = (user_min, user_max)
                    else: st.warning("최소값은 최대값보다 작아야 합니다.", icon="⚠️")

                if not df_to_plot.empty:
                    fig = px.line(df_to_plot, x='timestamp', y=selected_column, title=f"시간에 따른 {st.session_state.chart_type} 변화", labels={'timestamp': '시간', selected_column: '변화량'})
                    if x_axis_range: fig.update_xaxes(range=x_axis_range)
                    if y_axis_range: fig.update_yaxes(range=y_axis_range)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning("선택된 기간에 해당하는 데이터가 없습니다.", icon="⚠️")
            else: st.info("데이터가 2개 이상 쌓이면 그래프가 표시됩니다.", icon="ℹ️")

    st.markdown("---")
    st.subheader("🗃️ 원본 데이터 로그")
    st.dataframe(df.tail(20).iloc[::-1], height=740)

except Exception as e:
    st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}", icon="🔥")

if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()

