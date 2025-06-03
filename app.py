import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
import numpy as np

@st.cache_data
def load_data():
    return pd.read_csv("Penilaian_Kinerja.csv")

df = load_data()

st.title("Distribusi Skor KPI Pegawai (dengan μ dan σ)")

nipps = df['NIPP_Pekerja'].dropna().unique()
selected_nipp = st.selectbox("Pilih NIPP Pegawai:", sorted(nipps))

df_valid = df[['NIPP_Pekerja', 'Skor_KPI_Final']].dropna()
df_sorted = df_valid.sort_values(by='Skor_KPI_Final').reset_index(drop=True)
mean = df_sorted['Skor_KPI_Final'].mean()
std = df_sorted['Skor_KPI_Final'].std()
skew_val = skew(df_sorted['Skor_KPI_Final'])

selected_row = df_sorted[df_sorted['NIPP_Pekerja'] == selected_nipp]
if selected_row.empty:
    st.warning("NIPP tidak ditemukan.")
else:
    selected_score = selected_row.iloc[0]['Skor_KPI_Final']
    selected_rank = df_sorted[df_sorted['Skor_KPI_Final'] >= selected_score].shape[0]

    # ========== GRAFIK 1: SELURUH PEGAWAI ==========
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    bar_colors = ['orange' if n == selected_nipp else 'skyblue' for n in df_sorted['NIPP_Pekerja']]
    ax1.bar(df_sorted['Skor_KPI_Final'], df_sorted['Skor_KPI_Final'], color=bar_colors, label='Skor KPI Pegawai', alpha=0.8)
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    y = norm.pdf(x, loc=mean, scale=std)
    y_scaled = (y / y.max()) * (df_sorted['Skor_KPI_Final'].max() - df_sorted['Skor_KPI_Final'].min()) * 0.8
    # Area shading ±σ
    for i, color, label in zip([1, 2, 3], ['green', 'blue', 'red'], ['μ±1σ', 'μ±2σ', 'μ±3σ']):
        x_fill = np.linspace(mean - i*std, mean + i*std, 1000)
        y_fill = norm.pdf(x_fill, mean, std)
        y_fill_scaled = (y_fill / y.max()) * (df_sorted['Skor_KPI_Final'].max() - df_sorted['Skor_KPI_Final'].min()) * 0.8
        ax1.fill_between(x_fill, y_fill_scaled, alpha=0.12, color=color, label=f'Area {label}')
    ax1.plot(x, y_scaled, color='black', linewidth=2, label='Kurva Normal (μ ± σ)')
    for i in range(-3, 4):
        x_line = mean + i*std
        ax1.axvline(x_line, linestyle='--', color='gray')
        ax1.text(x_line, ax1.get_ylim()[1]*0.97, f'μ{"" if i==0 else f"{i:+d}σ"}', rotation=90,
                 verticalalignment='center', fontsize=9, color='gray')
    ax1.axhline(selected_score, color='orange', linestyle='--', label=f'NIPP {selected_nipp}: {selected_score:.2f}')
    ax1.axhline(mean, color='blue', linestyle='--', label=f'Rata-rata (μ): {mean:.2f}')
    ax1.set_xlabel("Skor KPI")
    ax1.set_ylabel("Frekuensi Skor (Skala Visualisasi)")
    ax1.set_title("1. Distribusi Seluruh Pegawai (Bar + Kurva Normal)")
    ax1.legend()
    st.pyplot(fig1)

    # ========== GRAFIK 2: KELOMPOK DI BAWAH ATASAN YANG SAMA ==========
    selected_atasan = df[df['NIPP_Pekerja'] == selected_nipp]['NIPP_Atasan'].values[0] if selected_nipp in df['NIPP_Pekerja'].values else None
    if selected_atasan:
        local_df = df[df['NIPP_Atasan'] == selected_atasan][['NIPP_Pekerja', 'Skor_KPI_Final']].dropna()
        if not local_df.empty:
            local_sorted = local_df.sort_values(by='Skor_KPI_Final').reset_index(drop=True)
            mean_local = local_sorted['Skor_KPI_Final'].mean()
            std_local = local_sorted['Skor_KPI_Final'].std()
            fig2, ax2 = plt.subplots(figsize=(14, 5))
            bar_colors_local = ['orange' if n == selected_nipp else 'skyblue' for n in local_sorted['NIPP_Pekerja']]
            ax2.bar(local_sorted['Skor_KPI_Final'], local_sorted['Skor_KPI_Final'], color=bar_colors_local, label='Skor KPI Pegawai', alpha=0.8)
            x_local = np.linspace(mean_local - 4*std_local, mean_local + 4*std_local, 1000)
            y_local = norm.pdf(x_local, loc=mean_local, scale=std_local)
            y_local_scaled = (y_local / y_local.max()) * (local_sorted['Skor_KPI_Final'].max() - local_sorted['Skor_KPI_Final'].min()) * 0.8
            ax2.plot(x_local, y_local_scaled, color='black', linewidth=2, label='Kurva Normal (μ ± σ)')
            for i in range(-3, 4):
                x_line = mean_local + i*std_local
                ax2.axvline(x_line, linestyle='--', color='gray')
                ax2.text(x_line, ax2.get_ylim()[1]*0.97, f'μ{"" if i==0 else f"{i:+d}σ"}', rotation=90,
                         verticalalignment='center', fontsize=9, color='gray')
            ax2.axhline(mean_local, color='blue', linestyle='--', label=f'Rata-rata (μ): {mean_local:.2f}')
            ax2.axhline(selected_score, color='orange', linestyle='--', label=f'NIPP {selected_nipp}: {selected_score:.2f}')
            ax2.set_xlabel("Skor KPI")
            ax2.set_ylabel("Frekuensi Skor (Skala Visualisasi)")
            ax2.set_title("2. Distribusi Pegawai di Bawah Atasan yang Sama (Bar + Kurva Normal)")
            ax2.legend()
            st.pyplot(fig2)

    # ========== GRAFIK 3: HANYA KURVA NORMAL STATISTIK (ALL DATA) ==========
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    # Area shading ±σ
    for i, color, label in zip([1, 2, 3], ['green', 'blue', 'red'], ['μ±1σ', 'μ±2σ', 'μ±3σ']):
        x_fill = np.linspace(mean - i*std, mean + i*std, 1000)
        y_fill = norm.pdf(x_fill, mean, std)
        y_fill_scaled = (y_fill / y.max()) * (df_sorted['Skor_KPI_Final'].max() - df_sorted['Skor_KPI_Final'].min()) * 0.8
        ax3.fill_between(x_fill, y_fill_scaled, alpha=0.12, color=color, label=f'Area {label}')
    ax3.plot(x, y_scaled, color='black', linewidth=2, label='Kurva Normal (μ ± σ)')
    for i in range(-3, 4):
        x_line = mean + i*std
        ax3.axvline(x_line, linestyle='--', color='gray')
        ax3.text(x_line, ax3.get_ylim()[1]*0.97, f'μ{"" if i==0 else f"{i:+d}σ"}', rotation=90,
                 verticalalignment='center', fontsize=9, color='gray')
    ax3.set_xlabel("Skor KPI")
    ax3.set_ylabel("Frekuensi Skor (Skala Visualisasi)")
    ax3.set_title("3. Kurva Distribusi Normal Statistik Pegawai (μ, σ, area shaded)")
    ax3.legend()
    st.pyplot(fig3)

    # === Info tambahan
    st.markdown(f"**Skor Pegawai (NIPP {selected_nipp})**: {selected_score:.2f}")
    st.markdown(f"**Peringkat (semakin tinggi semakin baik)**: {len(df_sorted) - selected_rank + 1} dari {len(df_sorted)}")
    st.markdown(f"**Rata-rata skor KPI (μ)**: {mean:.2f}")
    st.markdown(f"**Standard deviasi (σ)**: {std:.2f}")
    st.markdown(f"**Skewness distribusi (seluruh data)**: {skew_val:.2f}")

