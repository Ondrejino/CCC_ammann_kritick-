import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Bodová analýza", layout="wide")
st.title("CCC Detektor: Heatmapa a Anomálie")
st.caption("Čistá data, inverzní detekce chyb a statistické rozložení (Opraveno o outliery a nulové dělení).")

# --- 2. POMOCNÉ FUNKCE ---
@st.cache_data(show_spinner="Načítám a parsuji data...")
def nacti_surova_data(file_bytes):
    sample_text = file_bytes[:10000].decode("utf-8", errors="ignore")
    lines = sample_text.splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if "latitude" in line.lower() or "time" in line.lower():
            header_idx = i
            break
            
    header_line = lines[header_idx]
    sep = ';' if header_line.count(';') > header_line.count(',') else ','
    
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, skiprows=header_idx, on_bad_lines='skip', dtype=str)
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    return df

def najdi_vychozi_sloupec(columns, klicova_slova):
    for col in columns:
        for slovo in klicova_slova:
            if slovo in col.lower():
                return col
    return columns[0] if len(columns) > 0 else None

# --- 3. BOČNÍ PANEL ---
with st.sidebar:
    st.header("1. Nahrání dat")
    uploaded_file = st.file_uploader("Vložte CSV z válce", type=['csv'])
    
    if uploaded_file is not None:
        df_raw = nacti_surova_data(uploaded_file.getvalue())
        
        st.header("2. Ověření sloupců")
        col_time = st.selectbox("Sloupec s ČASEM", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['time'])))
        col_lat = st.selectbox("Sloupec LATITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['latitude'])))
        col_lon = st.selectbox("Sloupec LONGITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['longitude'])))
        col_stiff = st.selectbox("Sloupec TUHOSTI (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['stiff', 'kb'])))
        col_dir = st.selectbox("Sloupec SMĚRU", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['direction', 'dir'])))
        
        speed_options = ["Vypočítat z GPS (Záložní)"] + list(df_raw.columns)
        speed_guess = najdi_vychozi_sloupec(df_raw.columns, ['speed', 'rychlost'])
        default_speed_idx = speed_options.index(speed_guess) if speed_guess else 0
        col_speed = st.selectbox("Sloupec s RYCHLOSTÍ", speed_options, index=default_speed_idx)
        
        st.header("3. Očištění dat a Geometrie")
        offset_m = st.number_input("Posun anténa -> běhoun (m)", value=2.0, step=0.1)
        forward_val = st.text_input("Hodnota jízdy VPŘED", value="1")
        min_speed_kmh = st.slider("Minimální rychlost (km/h)", 0.0, 5.0, 1.0, 0.5)
        
        st.header("4. Nastavení očekávané kvality")
        st.markdown("Definuj, jaké hodnoty $K_b$ na stavbě **očekáváš** (Tvůj cíl). Mapa zvýrazní to, co leží mimo toto pásmo.")
        target_min = st.number_input("Očekávané minimum (Kb):", value=20.0, step=1.0)
        target_max = st.number_input("Očekávané maximum (Kb):", value=45.0, step=1.0)
        
        st.header("5. Vizuální nastavení Heatmapy")
        colormap = st.selectbox("Paleta", ['Turbo', 'Viridis', 'Plasma', 'Inferno', 'Jet'], index=0)
        point_opacity = st.slider("Průhlednost bodů", 0.1, 1.0, 0.6, 0.1)
        decimation = st.slider("Decimace map (pouze pro pozadí)", 1, 20, 2, 1)

# --- 4. HLAVNÍ VÝPOČETNÍ LOGIKA ---
if uploaded_file is not None:
    df = df_raw.copy()
    
    # Převody
    for col in [col_lat, col_lon, col_stiff]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.split(' GMT').str[0], errors='coerce')
    
    if col_speed != "Vypočítat z GPS (Záložní)":
        df['speed_kmh'] = pd.to_numeric(df[col_speed].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df['speed_kmh'] = 0.0

    df = df.dropna(subset=[col_lat, col_lon, col_stiff, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    if len(df) < 5:
        st.error("Málo platných dat.")
    else:
        geod = Geod(ellps="WGS84")
        
        # Geometrie
        window = 5
        df['smooth_lon'] = df[col_lon].rolling(window=window, min_periods=1, center=True).mean()
        df['smooth_lat'] = df[col_lat].rolling(window=window, min_periods=1, center=True).mean()
        
        lons_s, lats_s = df['smooth_lon'].values, df['smooth_lat'].values
        lons, lats = df[col_lon].values, df[col_lat].values
        
        step = 3
        fwd_az = np.zeros(len(df))
        az, _, _ = geod.inv(lons_s[:-step], lats_s[:-step], lons_s[step:], lats_s[step:])
        fwd_az[:-step] = az
        fwd_az[-step:] = az[-1] if len(az) > 0 else 0
        
        is_forward = (df[col_dir].astype(str) == str(forward_val)).values
        machine_heading = np.where(is_forward, fwd_az, (fwd_az + 180) % 360)
        new_lons, new_lats, _ = geod.fwd(lons, lats, machine_heading, np.full(len(lons), offset_m))
        df['corr_lon'], df['corr_lat'] = new_lons, new_lats
        
        # OPRAVA 2: Rychlost - ošetření dělení nulou
        if col_speed == "Vypočítat z GPS (Záložní)":
            _, _, dist_step = geod.inv(df['corr_lon'].shift(), df['corr_lat'].shift(), df['corr_lon'], df['corr_lat'])
            # Pokud je časový krok 0, nahradíme ho 0.001s, aby nedošlo k dělení nulou a generování 'inf'
            time_step = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.001)
            df['speed_kmh'] = (dist_step / time_step) * 3.6
            df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1).mean().bfill() 
        
        # Očištěná data
        df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
        
        if not df_valid.empty:
            avg_lat = df_valid['corr_lat'].mean()
            cos_correction = 1 / np.cos(np.radians(avg_lat))
            
            tab1, tab2, tab3 = st.tabs(["1️⃣ Surová Heatmapa", "2️⃣ Mapa Anomálií (Mimo cíl)", "3️⃣ Histogram (Statistika)"])
            
            # Decimace POUZE pro pozadí
            df_plot_background = df_valid.iloc[::decimation]
            
            with tab1:
                st.subheader(f"Surová bodová Heatmapa (Paleta: {colormap})")
                fig_raw = go.Figure()
                fig_raw.add_trace(go.Scatter(
                    x=df_plot_background['corr_lon'], y=df_plot_background['corr_lat'], mode='markers',
                    marker=dict(size=5, color=df_plot_background[col_stiff], colorscale=colormap, showscale=True, opacity=point_opacity, colorbar=dict(title="Kb [-]")),
                    name="Naměřená tuhost", hovertext=df_plot_background[col_stiff].round(1).astype(str) + ' Kb'
                ))
                fig_raw.update_layout(yaxis=dict(scaleanchor="x", scaleratio=cos_correction), height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_raw, use_container_width=True)
                
            with tab2:
                st.subheader(f"Anomálie: Hodnoty MIMO očekávané pásmo ({target_min} – {target_max} Kb)")
                fig_anom = go.Figure()
                
                # Šedý podkres (To co je v limitu = nudné, nezajímá nás). Zde decimace nevadí.
                df_ok_bg = df_plot_background[(df_plot_background[col_stiff] >= target_min) & (df_plot_background[col_stiff] <= target_max)]
                fig_anom.add_trace(go.Scatter(
                    x=df_ok_bg['corr_lon'], y=df_ok_bg['corr_lat'], mode='markers',
                    marker=dict(size=4, color='#E5E7EB', opacity=0.3), name="V cílovém pásmu (Pozadí)", hoverinfo='none'
                ))
                
                # OPRAVA 1: Podmírák a Tvrdé anomálie se VŽDY tahají z plného datasetu (df_valid), žádná decimace!
                df_under = df_valid[df_valid[col_stiff] < target_min]
                if not df_under.empty:
                    fig_anom.add_trace(go.Scatter(
                        x=df_under['corr_lon'], y=df_under['corr_lat'], mode='markers',
                        marker=dict(size=6, color='red'), name=f"Nedohutněno (< {target_min})",
                        hovertext="Měkké: " + df_under[col_stiff].round(1).astype(str) + " Kb"
                    ))
                
                df_over = df_valid[df_valid[col_stiff] > target_max]
                if not df_over.empty:
                    fig_anom.add_trace(go.Scatter(
                        x=df_over['corr_lon'], y=df_over['corr_lat'], mode='markers',
                        marker=dict(size=6, color='blue'), name=f"Tvrdá anomálie (> {target_max})",
                        hovertext="Tvrdé: " + df_over[col_stiff].round(1).astype(str) + " Kb"
                    ))
                
                fig_anom.update_layout(yaxis=dict(scaleanchor="x", scaleratio=cos_correction), height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_anom, use_container_width=True)

            with tab3:
                st.subheader("Statistické rozložení tuhosti (Histogram)")
                st.markdown("Ukazuje četnost všech naměřených hodnot na stavbě. Zelená oblast reprezentuje tvé cílové pásmo.")
                
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Histogram(
                    x=df_valid[col_stiff], nbinsx=50, marker_color='gray', name='Počet bodů'
                ))
                
                # Vyznačení cílového pásma
                fig_hist.add_vrect(
                    x0=target_min, x1=target_max,
                    fillcolor="green", opacity=0.2, layer="below", line_width=0,
                    annotation_text="Cílové pásmo", annotation_position="top left"
                )
                
                # OPRAVA 3: Ořez extrémních outlierů (Osa X končí na 99. percentilu + malá rezerva)
                max_val_99 = df_valid[col_stiff].quantile(0.99)
                safe_max = max(target_max * 1.2, max_val_99) # Zajišťuje, že minimálně cílové pásmo je vždy bezpečně vidět
                
                fig_hist.update_layout(
                    xaxis_title="Hodnota Kb [-]", yaxis_title="Počet naměřených bodů",
                    xaxis=dict(range=[0, safe_max]), # Ochrana proti roztáhnutí kvůli chybě senzoru
                    height=500, bargap=0.1, hovermode="x"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Jednoduchá procentuální statistika (Z nezdecimovaných dat)
                total = len(df_valid)
                pct_under = len(df_under) / total * 100
                pct_ok = len(df_valid[(df_valid[col_stiff] >= target_min) & (df_valid[col_stiff] <= target_max)]) / total * 100
                pct_over = len(df_over) / total * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Pod limitem (Nedohutněno)", f"{pct_under:.1f} %")
                c2.metric("🟢 V cílovém pásmu (OK)", f"{pct_ok:.1f} %")
                c3.metric("🔵 Nad limitem (Příliš tvrdé)", f"{pct_over:.1f} %")

        else:
            st.warning("Po odfiltrování stání a nulové rychlosti nezbyla žádná platná data.")
else:
    st.info("👋 Nahrajte CSV.")
