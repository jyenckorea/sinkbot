# 파일명: dashboard.py (v1.5 - X축/Y축 범위 수동 조절 통합)
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
st.title("🛰️ SinkBot AI 관제 대시보드 (v1.5)")

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 제어판")
    auto_refresh = st.toggle("자동 새로고침 활성화 ⏱️", value=True)
    if auto_refresh:
        st.caption("현재 상태: 5초마다 자동 새로고침 중...")
    else:
        st.caption("현재 상태: 수동 새로고침 모드.")
    if st.button("데이터 수동 새로고침 🔄"):
        st.toast("최신 데이터를 불러왔습니다!", icon="✅")
    st.markdown("---")
    st.header("🤖 모델 관리")
    if st.button("새 AI 모델 불러오기 (Reload)"):
        st.cache_resource.clear()
        st.toast("새로운 AI 모델을 불러왔습니다!", icon="🤖")

db_file = 'sinkbot_data.db'
model_file = 'sinkbot_model.pkl'

# (AI 모델 로딩 및 데이터 처리 함수는 이전과 동일)
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path): return None
    try: return joblib.load(model_path)
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        return None

def process_data(df):
    if len(df) < 1: return None, None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    reference_point = df.iloc[0]
    if len(df) > 0:
        df['delta_z'] = df['z'] - reference_point['z']
        df['distance_3d'] = np.sqrt((df['x'] - reference_point['x'])**2 + (df['y'] - reference_point['y'])**2 + (df['z'] - reference_point['z'])**2)
    return df, reference_point

model = load_model(model_file)

# --- 메인 대시보드 UI ---
try:
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query("SELECT * FROM displacement", conn)
    conn.close()

    # (시스템 상태, 지도, 데이터 요약 등 상단 UI는 이전과 동일)
    st.header("🚨 시스템 상태")
    if model is None: st.warning(f"'{model_file}' AI 모델 파일을 찾을 수 없습니다.", icon="⚠️")
    elif len(df) < 2: st.info("데이터가 충분히 쌓이면 AI 예측을 시작합니다.")
    else:
        df_for_pred, _ = process_data(df.copy())
        latest_features = df_for_pred.iloc[-1][['delta_z', 'distance_3d']]
        prediction = model.predict(latest_features.values.reshape(1, -1))
        if prediction[0] == -1: st.error("🚨 위험: AI가 이상 신호를 감지했습니다!", icon="🚨")
        else: st.success("✔️ 정상: 시스템이 안정적으로 운영 중입니다.", icon="✔️")
    st.markdown("---")

    st.header("📈 실시간 변위 분석")
    df_processed, reference_point = process_data(df.copy())

    if df_processed is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            # (지도 및 요약 정보 UI는 이전과 동일)
            st.subheader("📍 측정 위치")
            latest_location = df_processed.iloc[-1]
            lat, lon = latest_location['y'], latest_location['x']
            m = folium.Map(location=[lat, lon], zoom_start=16)
            folium.Marker([lat, lon], popup=f"<b>SinkBot</b><br>위도: {lat:.5f}<br>경도: {lon:.5f}", tooltip="현재 측정 위치", icon=folium.Icon(color='red', icon='arrows-v', prefix='fa')).add_to(m)
            st_folium(m, height=350, use_container_width=True)
            st.subheader("📊 데이터 요약")
            st.info(f"기준점: Y={reference_point['y']:.5f}, X={reference_point['x']:.5f}, Z={reference_point['z']:.2f}")
            if 'delta_z' in df_processed.columns and len(df_processed) > 1:
                st.metric("현재 수직 침하량", f"{df_processed.iloc[-1]['delta_z']:.4f} m", f"{df_processed.iloc[-1]['delta_z'] - df_processed.iloc[-2]['delta_z']:.4f} m")
        with col2:
            st.subheader("📉 시간에 따른 수직 침하량(Z) 변화")
            if 'delta_z' in df_processed.columns and len(df_processed) > 1:
                # (시간 범위, 집계 방식 선택 UI는 이전과 동일)
                graph_col1, graph_col2 = st.columns(2)
                with graph_col1:
                    time_range = st.selectbox("시간 범위 선택", options=['전체', '최근 24시간', '최근 6시간', '최근 1시간'], key='time_range_select')
                with graph_col2:
                    agg_level = st.selectbox("데이터 집계 방식", options=['원본 데이터', '시간별 평균', '일별 평균'], key='agg_level_select')
                
                # (데이터 필터링 및 가공 로직은 이전과 동일)
                df_to_plot = df_processed.copy()
                now = pd.to_datetime('now')
                if time_range == '최근 24시간': df_to_plot = df_to_plot[df_to_plot['timestamp'] >= (now - pd.Timedelta(hours=24))]
                elif time_range == '최근 6시간': df_to_plot = df_to_plot[df_to_plot['timestamp'] >= (now - pd.Timedelta(hours=6))]
                elif time_range == '최근 1시간': df_to_plot = df_to_plot[df_to_plot['timestamp'] >= (now - pd.Timedelta(hours=1))]
                if agg_level != '원본 데이터' and not df_to_plot.empty:
                    df_to_plot = df_to_plot.set_index('timestamp')
                    if agg_level == '시간별 평균': df_to_plot = df_to_plot[['delta_z']].resample('H').mean().reset_index()
                    elif agg_level == '일별 평균': df_to_plot = df_to_plot[['delta_z']].resample('D').mean().reset_index()

                # --- ⭐️ X축/Y축 수동 조절 기능 통합 ⭐️ ---
                y_axis_range, x_axis_range = None, None
                manual_axis_control = st.checkbox("그래프 축 범위 수동 조절")
                if manual_axis_control and not df_to_plot.empty:
                    axis_col1, axis_col2 = st.columns(2)
                    # X축 (시간) 범위 조절
                    with axis_col1:
                        st.markdown("**X축 범위 (날짜)**")
                        min_date = df_to_plot['timestamp'].min().date()
                        max_date = df_to_plot['timestamp'].max().date()
                        user_start_date = st.date_input("시작 날짜:", value=min_date, min_value=min_date, max_value=max_date, key='start_date')
                        user_end_date = st.date_input("종료 날짜:", value=max_date, min_value=min_date, max_value=max_date, key='end_date')
                        if user_start_date <= user_end_date:
                            x_axis_range = [pd.to_datetime(user_start_date), pd.to_datetime(user_end_date) + pd.Timedelta(days=1)]
                        else:
                            st.warning("시작 날짜는 종료 날짜보다 이전이어야 합니다.")
                    # Y축 (침하량) 범위 조절
                    with axis_col2:
                        st.markdown("**Y축 범위 (침하량)**")
                        min_val = float(df_to_plot['delta_z'].min())
                        max_val = float(df_to_plot['delta_z'].max())
                        user_min = st.number_input("최소값:", value=min_val, format="%.4f", key='y_min')
                        user_max = st.number_input("최대값:", value=max_val, format="%.4f", key='y_max')
                        if user_min < user_max:
                            y_axis_range = (user_min, user_max)
                        else:
                            st.warning("최소값은 최대값보다 작아야 합니다.")
                # --- ⭐️ 기능 통합 끝 ⭐️ ---

                if not df_to_plot.empty:
                    fig = px.line(df_to_plot, x='timestamp', y='delta_z', title=f"{time_range} 데이터 ({agg_level})", labels={'timestamp': '시간', 'delta_z': '침하량 (m)'})
                    # ⭐️ 수동 조절 값이 있으면 그래프에 적용
                    if x_axis_range: fig.update_xaxes(range=x_axis_range)
                    if y_axis_range: fig.update_yaxes(range=y_axis_range)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("선택된 기간에 해당하는 데이터가 없습니다.")
            else:
                st.info("데이터가 2개 이상 쌓이면 침하량 그래프가 표시됩니다.")

    st.markdown("---")
    st.subheader("🗃️ 원본 데이터 로그")
    st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)

except Exception as e:
    st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")

if auto_refresh:
    time.sleep(5)
    st.rerun()