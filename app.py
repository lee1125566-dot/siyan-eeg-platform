import streamlit as st
import mne
import plotly.graph_objects as go
import os
import numpy as np
from openai import OpenAI
from datetime import datetime

# ==========================================
# 1. 核心配置区 (Axzo, 别忘了填你的 Key)
# ==========================================
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"] 
BASE_URL = "https://api.deepseek.com"

# --- 页面 UI 风格配置 ---
st.set_page_config(page_title="思言的脑电工作台", layout="wide", initial_sidebar_state="collapsed")

# 视觉样式增强
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stButton>button { border-radius: 8px; font-weight: 500; transition: all 0.3s; }
    div[data-testid="stExpander"] { border: 1px solid #f0f2f6; box-shadow: 0 4px 12px rgba(0,0,0,0.03); background: white; border-radius: 12px; }
    .report-box { padding: 20px; background-color: #f9f9f9; border-radius: 10px; border-left: 5px solid #4a90e2; }
    </style>
    """, unsafe_allow_html=True)

# 初始化状态
if "messages" not in st.session_state: st.session_state.messages = []
if "raw_obj" not in st.session_state: st.session_state.raw_obj = None
if "cleaned_raw" not in st.session_state: st.session_state.cleaned_raw = None
if "report_content" not in st.session_state: st.session_state.report_content = ""

# --- 2. 顶栏：品牌 ---
st.title("✨ 思言的脑电工作台")
st.caption("专业的脑电数据自动化处理、分析与报告生成平台")

with st.sidebar:
    st.header("📂 资源管理")
    uploaded_files = st.file_uploader("导入脑电数据", type=["vhdr", "vmrk", "eeg", "edf", "set", "fif"], accept_multiple_files=True)
    st.divider()
    do_resample = st.toggle("自动降采样", value=True)
    target_fs = st.number_input("目标采样率", value=250)

TEMP_DIR = "temp_eeg_processing"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

# --- 3. 数据处理区 (缩小并收纳) ---
with st.expander("🛠️ 数据处理与报告生成", expanded=True):
    col_proc, col_report, col_dl = st.columns([1, 1, 2])
    
    # 文件加载逻辑
    if uploaded_files and st.session_state.raw_obj is None:
        main_file = None
        for f in uploaded_files:
            path = os.path.join(TEMP_DIR, f.name)
            with open(path, "wb") as b: b.write(f.getbuffer())
            if f.name.lower().endswith(('.vhdr', '.edf', '.set', '.fif')): main_file = path
        if main_file:
            raw = mne.io.read_raw(main_file, preload=True)
            if do_resample and raw.info['sfreq'] > target_fs: raw.resample(target_fs)
            st.session_state.raw_obj = raw

    # 坏道逻辑
    def auto_find_bads(raw_data):
        data = raw_data.copy().filter(1, 40, verbose=False).get_data()
        stds = np.std(data, axis=1)
        return [raw_data.ch_names[i] for i, s in enumerate(stds) if s > 4 * np.median(stds) or s < 1e-7]

    if st.session_state.raw_obj:
        raw = st.session_state.raw_obj
        
        with col_proc:
            if st.button("🚀 执行全自动深度清洗"):
                with st.spinner("处理中..."):
                    raw.filter(0.1, 45.0, verbose=False)
                    bads = auto_find_bads(raw)
                    if bads:
                        raw.info['bads'] = bads
                        raw.interpolate_bads(reset_bads=True)
                    ica = mne.preprocessing.ICA(n_components=min(len(raw.ch_names)-1, 15), random_state=97)
                    ica.fit(raw, verbose=False)
                    ica.exclude = [0, 1]
                    raw = ica.apply(raw, verbose=False)
                    raw.set_eeg_reference('average', projection=True, verbose=False)
                    st.session_state.cleaned_raw = raw
                    st.success("清洗完毕")

        with col_report:
            if st.button("📝 自动生成实验报告"):
                if st.session_state.cleaned_raw:
                    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
                    with st.spinner("AI 正在撰写报告..."):
                        info = st.session_state.cleaned_raw.info
                        prompt = f"""
                        你是一个严谨的认知神经科学专家助理。请为以下脑电实验数据生成一份标准的中文实验报告摘要。
                        数据指标：
                        - 通道数：{len(info['ch_names'])}
                        - 采样率：{info['sfreq']} Hz
                        - 坏道情况：已自动识别并插值修复了 {info['bads']}
                        - 处理流程：0.1-45Hz带通滤波、ICA伪迹剔除、全脑平均重参考。
                        
                        报告要求包含：
                        1. 数据基本信息概览。
                        2. 详细的预处理步骤描述（符合学术论文规范）。
                        3. 给研究者的后续分析建议。
                        4. 结语。
                        """
                        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
                        st.session_state.report_content = res.choices[0].message.content
                else:
                    st.error("请先执行‘深度清洗’后再生成报告。")

        with col_dl:
            if st.session_state.report_content:
                # 合并数据与报告下载
                st.download_button(
                    label="💾 导出完整实验报告 (.md)",
                    data=st.session_state.report_content,
                    file_name=f"脑电实验报告_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )

# --- 4. 报告预览区 (仅在生成后显示) ---
if st.session_state.report_content:
    st.markdown("### 📄 自动生成的实验报告预览")
    st.markdown(f'<div class="report-box">{st.session_state.report_content}</div>', unsafe_allow_html=True)
    st.divider()

# --- 5. 核心交互看板 (上下分布) ---
if st.session_state.raw_obj:
    display_raw = st.session_state.cleaned_raw if st.session_state.cleaned_raw else st.session_state.raw_obj
    
    st.subheader("📈 时域波形可视化")
    start_sec = st.slider("时间窗口定位 (秒)", 0.0, max(0.0, display_raw.times[-1]-10.0), 0.0)
    
    start_idx, stop_idx = display_raw.time_as_index([start_sec, start_sec + 10.0])
    data = display_raw.get_data(start=start_idx, stop=stop_idx)
    times = display_raw.times[start_idx:stop_idx]
    
    wave_fig = go.Figure()
    # 增加显示的通道数到 15 个，并优化间距
    for i in range(min(15, len(display_raw.ch_names))):
        wave_fig.add_trace(go.Scatter(x=times, y=data[i] + (i * 200e-6), name=display_raw.ch_names[i], mode='lines', line=dict(width=1.2)))
    
    wave_fig.update_layout(
        height=600, margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="时间 (s)", dragmode="pan",
        template="plotly_white", showlegend=True,
        xaxis=dict(showgrid=True, zeroline=False), yaxis=dict(showticklabels=False)
    )
    st.plotly_chart(wave_fig, use_container_width=True, config={'scrollZoom': True})

    st.divider()
    st.subheader("📊 频率特征分析 (PSD)")
    psds, freqs = display_raw.compute_psd(fmax=50).get_data(return_freqs=True)
    psd_fig = go.Figure()
    for i in range(min(10, len(display_raw.ch_names))):
        psd_fig.add_trace(go.Scatter(x=freqs, y=10*np.log10(psds[i]), name=display_raw.ch_names[i], line=dict(width=1.5)))
    
    psd_fig.update_layout(
        height=400, margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="频率 (Hz)", yaxis_title="功率 (dB/Hz)",
        template="plotly_white", hovermode="x unified"
    )
    st.plotly_chart(psd_fig, use_container_width=True)

# --- 6. AI 对话窗口 (问我) ---
st.divider()
st.subheader("💬 问我")
chat_container = st.container(height=350)
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input("您可以继续追问报告中的细节..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
                res = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": "你是一个名为‘思言’的脑电专家助手。"}, 
                              {"role": "user", "content": f"数据信息：{st.session_state.get('raw_obj', '暂无')}。问题：{prompt}"}]
                )
                answer = res.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e: st.error(f"对话出现微小波动: {e}")