# app_min.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt  # matplotlib

st.set_page_config(page_title="Aceleração 6Hz • 100Hz", layout="wide")
st.title("Carregar • Detrend • LP 6 Hz • Interpolar 100 Hz • Marcar ROI")

uploaded = st.file_uploader(
    "Arquivo .txt ou .csv com 4 colunas e cabeçalho (tempo, ax, ay, az)",
    type=["txt", "csv"]
)
if not uploaded:
    st.info("Carregue um arquivo para começar.")
    st.stop()

# ===== Leitura robusta =====
seps = [None, ";", "\t", r"\s+", ","]
df = None
for sep in seps:
    try:
        uploaded.seek(0)
        df_try = pd.read_csv(uploaded, sep=sep, engine="python")
        if df_try.shape[1] >= 4:
            df = df_try
            break
    except Exception:
        continue

if df is None or df.shape[1] < 4:
    st.error("Não consegui ler 4 colunas com cabeçalho. Verifique separador e cabeçalho.")
    st.stop()

# Padroniza 4 primeiras colunas
orig_cols = list(df.columns[:4])
df = df.rename(columns=dict(zip(orig_cols, ["time", "ax", "ay", "az"])))[["time", "ax", "ay", "az"]]
for c in ["time", "ax", "ay", "az"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

# (Opcional) Pré-visualização sem 'use_container_width'
st.caption("Pré-visualização (primeiras linhas):")
st.dataframe(df.head(10), width='stretch')

if len(df) < 10:
    st.error("Poucos dados após limpeza (menos de 10 linhas).")
    st.stop()

# Ordena tempo e remove duplicatas
df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
t = df["time"].to_numpy()
ax = df["ax"].to_numpy()
ay = df["ay"].to_numpy()
az = df["az"].to_numpy()

# Checa tempo válido
if t[-1] <= t[0]:
    st.error(f"Tempo inválido: t_min={t[0]:.6f} ≥ t_max={t[-1]:.6f}. Verifique a coluna de tempo.")
    st.stop()

dt = np.diff(t)
if np.any(dt <= 0):
    st.error("Existem Δt ≤ 0 após ordenação. Confira a coluna de tempo.")
    st.stop()

med_dt = float(np.median(dt))
fs_est = 1.0 / med_dt
st.info(f"Amostragem estimada: ~{fs_est:.2f} Hz | t_min={t[0]:.3f}s, t_max={t[-1]:.3f}s, N={len(t)}")

# ===== Parâmetros =====
colp = st.sidebar
fc = colp.number_input("Corte passa-baixa (Hz)", 0.5, 40.0, 6.0, 0.5)
target_fs = colp.number_input("Frequência de saída (Hz)", 20, 500, 100, 10)
order = colp.slider("Ordem Butterworth", 2, 8, 4, 1)
detrend_mode = colp.selectbox("Detrend", ["linear", "constant"], 0)
irreg_thresh = colp.slider("Limiar irregularidade (desv/med Δt)", 0.00, 1.00, 0.10, 0.01)

# ===== Detrend =====
ax_dt = signal.detrend(ax, type=detrend_mode)
ay_dt = signal.detrend(ay, type=detrend_mode)
az_dt = signal.detrend(az, type=detrend_mode)

# ===== Regular x Irregular =====
std_dt = float(np.std(dt))
irregularity = std_dt / med_dt if med_dt > 0 else np.inf
is_irregular = irregularity > irreg_thresh

# ===== Interpolação para grade uniforme =====
Ts = 1.0 / target_fs
t_uni = np.arange(t[0], t[-1], Ts)
if t_uni.size < 2:
    st.error("Grade uniforme muito curta (menos de 2 pontos). Aumente duração ou diminua a fs alvo.")
    st.stop()
if t_uni[-1] < t[-1]:
    t_uni = np.append(t_uni, t[-1])

def interp(y):
    return np.interp(t_uni, t, y)

ax_dt_i = interp(ax_dt)
ay_dt_i = interp(ay_dt)
az_dt_i = interp(az_dt)

def lp(y, fs, fc, order):
    nyq = 0.5 * fs
    wn = min(max(fc / nyq, 1e-6), 0.999999)
    b, a = signal.butter(order, wn, btype="low")
    return signal.filtfilt(b, a, y, method="gust")

if not is_irregular:
    # Filtra no original e depois interpola
    ax_f = lp(ax_dt, fs_est, fc, order)
    ay_f = lp(ay_dt, fs_est, fc, order)
    az_f = lp(az_dt, fs_est, fc, order)
    ax_p = np.interp(t_uni, t, ax_f)
    ay_p = np.interp(t_uni, t, ay_f)
    az_p = np.interp(t_uni, t, az_f)
    st.success("Pipeline: detrend → filtro 6 Hz → interpolação 100 Hz (amostragem regular).")
else:
    # Interpola e depois filtra
    ax_p = lp(ax_dt_i, target_fs, fc, order)
    ay_p = lp(ay_dt_i, target_fs, fc, order)
    az_p = lp(az_dt_i, target_fs, fc, order)
    st.warning(
        f"Amostragem irregular (desv/med Δt = {irregularity:.3f} > {irreg_thresh:.3f}). "
        "Pipeline: detrend → interpolação 100 Hz → filtro 6 Hz."
    )

# ===== ROI com number_input =====
t0_def, t1_def = float(np.round(t_uni[0], 2)), float(np.round(t_uni[-1], 2))
col1, col2 = st.columns(2)
with col1:
    t0 = st.number_input("Início da ROI (s)", min_value=t0_def, max_value=t1_def,
                         value=t0_def, step=0.01)
with col2:
    t1 = st.number_input("Fim da ROI (s)", min_value=t0_def, max_value=t1_def,
                         value=t1_def, step=0.01)

if t1 <= t0:
    st.warning("⚠️ O tempo final deve ser maior que o tempo inicial.")
roi = (t_uni >= t0) & (t_uni <= t1)

# ===== Plot com matplotlib (st.pyplot não usa use_container_width) =====
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 7))
series = [("ax (g)", ax_p), ("ay (g)", ay_p), ("az (g)", az_p)]
for ax_subplot, (label, y) in zip(axes, series):
    ax_subplot.plot(t_uni, y, linewidth=1.0)
    ax_subplot.axvline(t0, linestyle="--", linewidth=1.5)
    ax_subplot.axvline(t1, linestyle="--", linewidth=1.5)
    ax_subplot.axvspan(t0, t1, alpha=0.2)
    ax_subplot.set_ylabel(label)
axes[-1].set_xlabel("Tempo (s)")
plt.tight_layout()
st.pyplot(fig, clear_figure=True)  # sem 'use_container_width'

# ===== Exporta processado =====
proc = pd.DataFrame({"time_s": t_uni, "ax_g": ax_p, "ay_g": ay_p, "az_g": az_p})
st.download_button(
    "⬇️ Baixar processado (CSV)",
    proc.to_csv(index=False).encode("utf-8"),
    file_name="acel_100Hz_lp6Hz.csv",
    mime="text/csv"
)
