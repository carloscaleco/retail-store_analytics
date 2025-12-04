import streamlit as st
import cv2
import csv
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# 0. EXECUÇÃO INICIAL DO STREAMLIT NO TERMINAL
# streamlit run app.py
# ============================================

# ============================================
# 1. CONFIGURAÇÃO E CSS
# ============================================
st.set_page_config(page_title="Retail Analytics AI", page_icon="🏪", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
        [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: bold; }
        .stProgress > div > div > div > div { background-color: inherit; }
        img { margin-bottom: 0px; }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 2. INICIALIZAÇÃO DE VARIÁVEIS DE SESSÃO
# ============================================
if 'count_in' not in st.session_state: st.session_state['count_in'] = 0
if 'count_out' not in st.session_state: st.session_state['count_out'] = 0
if 'occupancy' not in st.session_state: st.session_state['occupancy'] = 0
if 'video_frame_index' not in st.session_state: st.session_state['video_frame_index'] = 0
if 'status_green_ids' not in st.session_state: st.session_state['status_green_ids'] = set()
if 'status_cyan_ids' not in st.session_state: st.session_state['status_cyan_ids'] = set()
if 'last_frame_rgb' not in st.session_state: st.session_state['last_frame_rgb'] = None
if 'track_history' not in st.session_state: st.session_state['track_history'] = {}

# Constantes
VIDEO_PATH = "Videos/OxfordTownCentre/TownCentreXVID.mp4"
MODEL_NAME = 'yolov8n.pt'
LOG_FILE = 'occupancy_log.csv'
TRAJ_FILE = 'trajectories.csv'
TRACKER_FILE = "my_tracker.yaml"

# ============================================
# 3. SIDEBAR
# ============================================
st.title("🏪 Retail Store Analytics")
st.sidebar.header("Definições")

run_system = st.sidebar.toggle("Ligar Sistema", value=False)
enable_trajectory = st.sidebar.checkbox("Mostrar Trajetórias", value=False)
conf_threshold = st.sidebar.slider("Confiança", 0.0, 1.0, 0.25)
line_position = st.sidebar.slider("Posição Linha Entrada", 0, 600, 240)
max_capacity = st.sidebar.number_input("Capacidade Máx", min_value=1, value=20)

if st.sidebar.button("⚠️ Reset Total"):
    st.cache_resource.clear()
    st.session_state.clear()
    if os.path.exists(TRAJ_FILE): os.remove(TRAJ_FILE)
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    st.rerun()

# ============================================
# 4. FUNÇÕES
# ============================================  
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

def get_status_html(pct):
    if pct <= 0: return "#ffffff", "" 
    if pct < 50: return "#28a745", "(LOW)"
    if pct < 85: return "#ffc107", "(MODERATE)"
    return "#dc3545", "(FULL)"

def custom_progress_bar(pct, color):
    pct = min(max(pct, 0), 100)
    return f"""<div style="background-color:#e0e0e0;border-radius:5px;height:15px;width:100%;margin-top:5px">
    <div style="background-color:{color};width:{pct}%;height:100%;border-radius:5px"></div></div>"""

def load_data():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        except: return pd.DataFrame()
    return pd.DataFrame()

# --- ATUALIZAÇÃO AQUI: Dashboard Simplificado ---
def render_dashboard():
    df = load_data()
    if not df.empty:
        total_visitors = len(df[df['Direction'] == 'IN'])
        total_exits = len(df[df['Direction'] == 'OUT'])
        avg_occupancy = int(df['Occupancy'].mean())
        
        hourly_counts = df[df['Direction'] == 'IN'].groupby('Hour').size()
        peak_hour = hourly_counts.idxmax() if not hourly_counts.empty else "N/A"
        peak_count = hourly_counts.max() if not hourly_counts.empty else 0

        # Métricas no topo
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Entradas", total_visitors)
        c2.metric("Total Saídas", total_exits)
        c3.metric("Média Ocupação", avg_occupancy)
        c4.metric("Hora de Ponta", f"{peak_hour}h", f"{peak_count} pessoas")

        st.divider()

        # Apenas Gráfico de Linha e Barras (Removemos Pie e Heatmap)
        g1, g2 = st.columns([2, 1])
        with g1:
            fig_line = px.line(df, x='Timestamp', y='Occupancy', 
                               title='📈 Evolução da Ocupação em Tempo Real',
                               labels={'Occupancy': 'Pessoas na Loja', 'Timestamp': 'Horário'})
            fig_line.update_traces(line_color='#4763ff', line_width=3)
            st.plotly_chart(fig_line, use_container_width=True)
        
        with g2:
            if not hourly_counts.empty:
                df_hour = hourly_counts.reset_index(name='Count')
                fig_bar = px.bar(df_hour, x='Hour', y='Count', 
                                 title='📊 Entradas por Hora',
                                 color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("A aguardar dados...")

    else:
        st.info("⚠️ Sem dados recolhidos ainda. Ligue o sistema para gerar estatísticas.")

# ============================================
# 5. LAYOUT FIXO (ABAS)
# ============================================
tab1, tab2 = st.tabs(["📹 Monitorização", "📈 Dashboard Inteligente"])

with tab1:
    col_vid, col_stat = st.columns([3, 1])
    
    with col_vid:
        image_spot = st.empty()
        
        # --- LÓGICA DE VISUALIZAÇÃO ESTÁTICA ---
        if st.session_state['last_frame_rgb'] is not None:
            display_frame = st.session_state['last_frame_rgb'].copy()
            
            if enable_trajectory:
                 for tid, points in st.session_state['track_history'].items():
                    if len(points) > 1:
                        pts_array = np.array(points, dtype=np.int32)
                        cv2.polylines(display_frame, [pts_array.reshape((-1, 1, 2))], False, (255, 255, 255), 1)

            image_spot.image(display_frame, use_container_width=True)
    
    with col_stat:
        kpi_occ = st.empty()
        status_spot = st.empty()
        st.divider()
        kpi_in = st.empty()
        kpi_out = st.empty()

        occ = st.session_state['occupancy']
        occ_pct = (occ / max_capacity) * 100
        c_hex, s_txt = get_status_html(occ_pct)

        kpi_occ.markdown(f"""
            <div style="line-height: 1;">
                <span style="font-size: 14px; color: #808495;">Ocupação Atual</span>
                <div style="font-size: 32px; font-weight: bold; margin-top: 5px;">
                    {occ} / {max_capacity} <span style="font-size: 20px; color: {c_hex}; vertical-align: middle; margin-left: 5px;">{s_txt}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        kpi_in.metric("Entradas", st.session_state['count_in'])
        kpi_out.metric("Saídas", st.session_state['count_out'])
        
        status_spot.markdown(f"**Nível:** <span style='color:{c_hex}; font-weight:bold'>{s_txt}</span>", unsafe_allow_html=True)
        status_spot.markdown(custom_progress_bar(occ_pct, c_hex), unsafe_allow_html=True)

with tab2:
    if st.button("🔄 Atualizar Relatório"):
        render_dashboard()
    else:
        render_dashboard()

# ============================================
# 6. LÓGICA DE PROCESSAMENTO (LOOP)
# ============================================
if run_system:
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Hour", "Person_ID", "Direction", "Occupancy", "X", "Y"])
    
    if not os.path.exists(TRAJ_FILE):
        with open(TRAJ_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Frame", "ID", "X", "Y"])

    model = load_model(MODEL_NAME)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    start_frame = st.session_state.get('video_frame_index', 0)
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if not cap.isOpened():
        st.error(f"Erro: Não foi possível abrir {VIDEO_PATH}")
        st.stop()

    ids_seen_bottom = set()
    ids_seen_top = set()
    
    status_green_ids = st.session_state['status_green_ids']
    status_cyan_ids = st.session_state['status_cyan_ids']

    while True:
        success, frame = cap.read()
        
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            st.session_state['video_frame_index'] = 0
            continue
        
        st.session_state['video_frame_index'] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.resize(frame, (854, 480))
        vis_frame = frame.copy()

        try:
            results = model.track(frame, persist=True, classes=0, tracker=TRACKER_FILE, conf=conf_threshold, verbose=False)
        except:
            results = model.track(frame, persist=True, classes=0, conf=conf_threshold, verbose=False)

        cv2.line(vis_frame, (0, line_position), (1020, line_position), (0, 255, 255), 2)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().tolist()
            ids = results[0].boxes.id.int().cpu().tolist()

            with open(TRAJ_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                
                for box, track_id in zip(boxes, ids):
                    x, y, w, h = box
                    cx, cy = int(x), int(y + (h/3))
                    
                    writer.writerow([st.session_state['video_frame_index'], track_id, cx, cy])
                    
                    if track_id not in st.session_state['track_history']:
                        st.session_state['track_history'][track_id] = []
                    
                    history = st.session_state['track_history'][track_id]
                    if len(history) > 0:
                         last_x, last_y = history[-1]
                         dist = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
                         if dist > 100:
                             st.session_state['track_history'][track_id] = [] 

                    st.session_state['track_history'][track_id].append([cx, cy])

                    if cy > line_position: ids_seen_bottom.add(track_id)
                    elif cy < line_position: ids_seen_top.add(track_id)

                    if cy > line_position and track_id in ids_seen_top:
                        if track_id not in status_cyan_ids:
                            st.session_state['count_out'] += 1
                            st.session_state['occupancy'] -= 1
                            status_cyan_ids.add(track_id)
                            st.session_state['status_cyan_ids'] = status_cyan_ids
                            if track_id in status_green_ids: 
                                status_green_ids.remove(track_id)
                                st.session_state['status_green_ids'] = status_green_ids
                            
                            now = datetime.now()
                            with open(LOG_FILE, 'a', newline='') as f_log:
                                csv.writer(f_log).writerow([now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%H"), track_id, "OUT", st.session_state['occupancy'], cx, cy])

                    elif cy < line_position and track_id in ids_seen_bottom:
                        if track_id not in status_green_ids:
                            st.session_state['count_in'] += 1
                            st.session_state['occupancy'] += 1
                            status_green_ids.add(track_id)
                            st.session_state['status_green_ids'] = status_green_ids
                            if track_id in status_cyan_ids: 
                                status_cyan_ids.remove(track_id)
                                st.session_state['status_cyan_ids'] = status_cyan_ids

                            now = datetime.now()
                            with open(LOG_FILE, 'a', newline='') as f_log:
                                csv.writer(f_log).writerow([now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%H"), track_id, "IN", st.session_state['occupancy'], cx, cy])

                    color = (128, 128, 128)
                    if track_id in status_cyan_ids: color = (255, 255, 0)
                    elif track_id in status_green_ids: color = (50, 255, 50)
                    elif cy > line_position: color = (71, 99, 255)

                    cv2.rectangle(vis_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)
                    cv2.putText(vis_frame, f"{track_id}", (int(x-w/2), int(y-h/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Visualização
        frame_rgb_clean = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        st.session_state['last_frame_rgb'] = frame_rgb_clean.copy()

        display_frame = frame_rgb_clean
        if enable_trajectory:
            display_frame = frame_rgb_clean.copy()
            for tid, points in st.session_state['track_history'].items():
                if len(points) > 1:
                    pts_array = np.array(points, dtype=np.int32)
                    cv2.polylines(display_frame, [pts_array.reshape((-1, 1, 2))], False, (255, 255, 255), 1)

        image_spot.image(display_frame, use_container_width=True)
        
        # Atualização KPI
        occ = st.session_state['occupancy']
        occ_pct = (occ / max_capacity) * 100
        c_hex, s_txt = get_status_html(occ_pct)
        
        kpi_occ.markdown(f"""
            <div style="line-height: 1;">
                <span style="font-size: 14px; color: #808495;">Ocupação Atual</span>
                <div style="font-size: 32px; font-weight: bold; margin-top: 5px;">
                    {occ} / {max_capacity} <span style="font-size: 20px; color: {c_hex}; vertical-align: middle; margin-left: 5px;">{s_txt}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        kpi_in.metric("Entradas", st.session_state['count_in'])
        kpi_out.metric("Saídas", st.session_state['count_out'])
        
        status_spot.markdown(f"**Nível:** <span style='color:{c_hex}; font-weight:bold'>{s_txt}</span>", unsafe_allow_html=True)
        status_spot.markdown(custom_progress_bar(occ_pct, c_hex), unsafe_allow_html=True)

    cap.release()

