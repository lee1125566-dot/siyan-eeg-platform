import streamlit as st
import mne
import plotly.graph_objects as go
import os
import numpy as np
import pandas as pd
from openai import OpenAI
from datetime import datetime
from docx import Document
from io import BytesIO

# --- 1. 配置与安全 ---
try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    DEEPSEEK_API_KEY = "sk-c31f71dfba3f4531b230e266c420867a"

BASE_URL = "https://api.deepseek.com"

# --- UI 风格 ---
st.set_page_config(page_title="思言的脑电工作台", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.main { background-color: #fcfcfc; } .stButton>button { border-radius: 8px; font-weight: 500; transition: all 0.3s; } div[data-testid="stExpander"] { border: 1px solid #f0f2f6; box-shadow: 0 4px 12px rgba(0,0,0,0.03); background: white; border-radius: 12px; } .report-box { padding: 20px; background-color: #f9f9f9; border-radius: 10px; border-left: 5px solid #4a90e2; }</style>""", unsafe_allow_html=True)

# 状态初始化
if "messages" not in st.session_state: st.session_state.messages = []
if "raw_obj" not in st.session_state: st.session_state.raw_obj = None
if "cleaned_raw" not in st.session_state: st.session_state.cleaned_raw = None
if "report_content" not in st.session_state: st.session_state.report_content = ""

# --- 2. 侧边栏 ---
with st.sidebar:
    st.header("📂 资源管理")
    uploaded_files = st.file_uploader("导入数据", type=["vhdr", "vmrk", "eeg", "edf", "set", "fif"], accept_multiple_files=True)
    st.divider()
    do_resample = st.toggle("自动降采样", value=True)
    target_fs = st.number_input("目标采样率", value=250)

TEMP_DIR = "temp_eeg_processing"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

# --- 3. 核心功能函数 ---
def create_word_report(content):
    doc = Document()
    doc.add_heading('思言脑电工作台 - 实验处理报告', 0)
    doc.add_paragraph(content)
    bio = BytesIO(); doc.save(bio)
    return bio.getvalue()

def export_to_excel(raw):
    data, times = raw[:, :int(raw.info['sfreq'] * 15)]
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    df.insert(0, 'Time(s)', times)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False)
    return output.getvalue()

# --- 4. 业务逻辑 ---
st.title("✨ 思言的脑电工作台")

if uploaded_files:
    main_file = None
    for f in uploaded_files:
        path = os.path.join(TEMP_DIR, f.name)
        with open(path, "wb") as b: b.write(f.getbuffer())
        if f.name.lower().endswith(('.vhdr', '.edf', '.set', '.fif')): main_file = path
    if main_file and st.session_state.raw_obj is None:
        raw_data = mne.io.read_raw(main_file, preload=True)
        if do_resample and raw_data.info['sfreq'] > target_fs: raw_data.resample(target_fs)
        st.session_state.raw_obj = raw_data

if st.session_state.raw_obj:
    with st.expander("🛠️ 数据处理与多格式导出", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🚀 执行全自动清洗"):
                with st.spinner("处理中..."):
                    raw = st.session_state.raw_obj.copy()
                    raw.filter(0.1, 45.0, verbose=False)
                    # 简单坏道识别
                    data_check = raw.get_data(); stds = np.std(data_check, axis=1)
                    bads = [raw.ch_names[i] for i, s in enumerate(stds) if s > 4 * np.median(stds) or s < 1e-7]
                    if bads: raw.info['bads'] = bads; raw.interpolate_bads(reset_bads=True)
                    # ICA
                    ica = mne.preprocessing.ICA(n_components=min(len(raw.ch_names)-1, 15), random_state=97)
                    ica.fit(raw, verbose=False); ica.exclude = [0, 1]
                    st.session_state.cleaned_raw = ica.apply(raw, verbose=False)
                    st.success("清洗完成")
            if st.button("📝 生成 AI 报告"):
                if st.session_state.cleaned_raw:
                    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
                    res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": "请生成脑电预处理报告。"}])
                    st.session_state.report_content = res.choices[0].message.content
        with c2:
            if st.session_state.cleaned_raw:
                st.session_state.cleaned_raw.save("temp.fif", overwrite=True, verbose=False)
                with open("temp.fif", "rb") as f: st.download_button("💾 下载 FIF", f, "clean.fif")
                st.download_button("📊 导出 Excel (样本)", export_to_excel(st.session_state.cleaned_raw), "sample.xlsx")
        with c3:
            if st.session_state.report_content:
                st.download_button("📄 导出 MD", st.session_state.report_content, "report.md")
                st.download_button("📘 导出 Word", create_word_report(st.session_state.report_content), "report.docx")

    # --- 5. 交互式可视化看板 (恢复此处) ---
    st.divider()
    display_raw = st.session_state.cleaned_raw if st.session_state.cleaned_raw else st.session_state.raw_obj
    st.subheader("📈 数据可视化看板")
    
    # 时域图
    start_sec = st.slider("时间窗口定位 (秒)", 0.0, max(0.0, display_raw.times[-1]-10.0), 0.0)
    idx_start, idx_stop = display_raw.time_as_index([start_sec, start_sec + 10.0])
    data = display_raw.get_data(start=idx_start, stop=idx_stop)
    times = display_raw.times[idx_start:idx_stop]
    wave_fig = go.Figure()
    for i in range(min(12, len(display_raw.ch_names))):
        wave_fig.add_trace(go.Scatter(x=times, y=data[i] + (i * 200e-6), name=display_raw.ch_names[i]))
    wave_fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white", dragmode="pan")
    st.plotly_chart(wave_fig, use_container_width=True, config={'scrollZoom': True})

    # PSD 图
    psds, freqs = display_raw.compute_psd(fmax=50, verbose=False).get_data(return_freqs=True)
    psd_fig = go.Figure()
    for i in range(min(8, len(display_raw.ch_names))):
        psd_fig.add_trace(go.Scatter(x=freqs, y=10*np.log10(psds[i]), name=display_raw.ch_names[i]))
    psd_fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white")
    st.plotly_chart(psd_fig, use_container_width=True)

    # --- 6. 问我 ---
    st.divider(); st.subheader("💬 问我")
    chat_box = st.container(height=350)
    for msg in st.session_state.messages:
        with chat_box:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if p := st.chat_input("输入问题..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with chat_box:
            with st.chat_message("user"): st.markdown(p)
            with st.chat_message("assistant"):
                client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
                res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": p}])
                st.markdown(res.choices[0].message.content)
                st.session_state.messages.append({"role": "assistant", "content": res.choices[0].message.content})
else:
    st.info("👋 请在侧边栏上传文件。注意：大文件上传时请务必开启侧边栏的‘自动降采样’。")