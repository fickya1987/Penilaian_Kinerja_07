import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
import numpy as np

@st.cache_data
def load_data():
    return pd.read_csv("Penilaian_Kinerja.csv")

df = load_data()

st.title("Distribusi Normal Skor KPI Pegawai")

# Pilih NIPP
nipps = df['NIPP_Pekerja'].dropna().unique()
selected_nipp = st.selectbox("Pilih NIPP Pegawai:", sorted(nipps))

# Data valid
df_valid = df[['NIPP_Pekerja', 'Skor_KPI_Final']].dropna()
df_sorted = df_valid.sort_values(by='Skor_KPI_Final').reset_index(drop=True)

selected_row = df_sorted[df_sorted['NIPP_Pekerja'] == selected_nipp]
if selected_row.empty:
    st.warning("NIPP tidak ditemukan.")
else:
    selected_score = selected_row.iloc[0]['Skor_KPI_Final']
    selected_rank = df_sorted[df_sorted['Skor_KPI_Final'] >= selected_score].shape[0]

    mean = df_sorted['Skor_KPI_Final'].mean()
    std = df_sorted['Skor_KPI_Final'].std()
    skew_val = skew(df_sorted['Skor_KPI_Final'])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_colors = ['orange' if n == selected_nipp else 'skyblue' for n in df_sorted['NIPP_Pekerja']]
    ax.bar(df_sorted['Skor_KPI_Final'], df_sorted['Skor_KPI_Final'], color=bar_colors, label='Skor KPI Pegawai')

    # Kurva normal
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    y = norm.pdf(x, loc=mean, scale=std)
    y_scaled = (y / y.max()) * (df_sorted['Skor_KPI_Final'].max() - df_sorted['Skor_KPI_Final'].min()) * 0.8

    # Area shading ±σ
    for i, color, label in zip([1, 2, 3], ['green', 'blue', 'red'], ['μ±1σ', 'μ±2σ', 'μ±3σ']):
        x_fill = np.linspace(mean - i*std, mean + i*std, 1000)
        y_fill = norm.pdf(x_fill, mean, std)
        y_fill_scaled = (y_fill / y.max()) * (df_sorted['Skor_KPI_Final'].max() - df_sorted['Skor_KPI_Final'].min()) * 0.8
        ax.fill_between(x_fill, y_fill_scaled, alpha=0.1, color=color, label=f'Area {label}')

    # Plot kurva
    ax.plot(x, y_scaled, color='black', linewidth=2, label='Kurva Normal (μ ± σ)')

    # Garis vertikal di μ dan ±σ
    for i in range(-3, 4):
        x_line = mean + i*std
        ax.axvline(x_line, linestyle='--', color='gray')
        ax.text(x_line, ax.get_ylim()[1]*0.95, f'μ{"" if i==0 else f"{i:+d}σ"}', rotation=90,
                verticalalignment='center', fontsize=9, color='gray')

    # Garis horizontal KPI pegawai terpilih
    ax.axhline(selected_score, color='orange', linestyle='--', label=f'NIPP {selected_nipp}: {selected_score:.2f}')
    ax.axhline(mean, color='blue', linestyle='--', label=f'Rata-rata (μ): {mean:.2f}')
    ax.set_xlabel("Skor KPI")
    ax.set_ylabel("Frekuensi Skor (Skala Visualisasi)")
    ax.set_title("Distribusi Skor KPI dan Posisi Pegawai terhadap μ dan σ")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"**Skor Pegawai (NIPP {selected_nipp})**: {selected_score:.2f}")
    st.markdown(f"**Peringkat (semakin tinggi semakin baik)**: {len(df_sorted) - selected_rank + 1} dari {len(df_sorted)}")
    st.markdown(f"**Rata-rata skor KPI (μ)**: {mean:.2f}")
    st.markdown(f"**Standard deviasi (σ)**: {std:.2f}")
    st.markdown(f"**Skewness distribusi**: {skew_val:.2f}")
