import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Ultimátní Detektor v2", layout="wide")
st.title("CCC Detektor: Komplexní analýza zhutnění")
st.caption("Interaktivní přehrávání stavby vrstvu po vrstvě se sledováním úvodního i finálního statického žehlení.")

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

# --- CACHOVANÉ GEOPROSTOROVÉ ZPRACOVÁNÍ ---
@st.cache_data(show_spinner="Zpracovávám geodata, azimuty a pojezdy (počítá se jen jednou)...")
def zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_m, offset_transverse_m, min_speed_kmh):
    df = df_raw.copy()
    
    # Převody s ochranou
    for col in [col_lat, col_lon, col_vib]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    df[col_stiff] = pd.to_numeric(df[col_stiff].astype(str).str.replace(',', '.'), errors='coerce')
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.split(' GMT').str[0], errors='coerce')
    
    # Detekce vibrace vs. statiky
    df['is_vibrating'] = df[col_vib].fillna(0) > 0.1
    
    if col_speed != "Vypočítat z GPS (Záložní)":
        df['speed_kmh'] = pd.to_numeric(df[col_speed].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df['speed_kmh'] = 0.0

    df = df.dropna(subset=[col_lat, col_lon, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    if len(df) <= 5:
        return pd.DataFrame()

    geod = Geod(ellps="WGS84")
    
    # Geometrie & Azimut
    df['smooth_lon'] = df[col_lon].rolling(5, min_periods=1, center=True).mean()
    df['smooth_lat'] = df[col_lat].rolling(5, min_periods=1, center=True).mean()
    
    step = 3
    fwd_az = np.zeros(len(df))
    az, _, _ = geod.inv(df['smooth_lon'].values[:-step], df['smooth_lat'].values[:-step], 
                        df['smooth_lon'].values[step:], df['smooth_lat'].values[step:])
    fwd_az[:-step] = az
    fwd_az[-step:] = az[-1] if len(az) > 0 else 0
    
    is_forward = (df[col_dir].astype(str) == "1").values
    machine_heading = np.where(is_forward, fwd_az, (fwd_az + 180) % 360)
    
    # PŘIDÁNO: Uložení azimutu do datasetu pro rotaci čtverců
    df['heading'] = machine_heading
    
    # --- MATEMATIKA L-tvaru ---
    temp_lons, temp_lats, _ = geod.fwd(df[col_lon].values, df[col_lat].values, machine_heading, np.full(len(df), offset_m))
    transverse_heading = (machine_heading + 90) % 360
    new_lons, new_lats, _ = geod.fwd(temp_lons, temp_lats, transverse_heading, np.full(len(df), offset_transverse_m))
    
    df['corr_lon'], df['corr_lat'] = new_lons, new_lats
    
    if col_speed == "Vypočítat z GPS (Záložní)":
        _, _, dist_step = geod.inv(df['corr_lon'].shift(), df['corr_lat'].shift(), df['corr_lon'], df['corr_lat'])
        time_step = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.001)
        df['speed_kmh'] = (dist_step / time_step) * 3.6
        df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1).mean().bfill() 
    
    # Detekce pojezdů
    df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
    if not df_valid.empty:
        time_gap_cond = df_valid['parsed_time'].diff().dt.total_seconds() > 30
        dir_cond = df_valid[col_dir] != df_valid[col_dir].shift().bfill()
        df_valid['pass_id'] = (time_gap_cond | dir_cond).cumsum() + 1
        
    return df_valid


# --- 3. BOČNÍ PANEL ---
with st.sidebar:
    st.header("1. Nahrání dat")
    uploaded_file = st.file_uploader("Vložte CSV z válce", type=['csv'])
    
    if uploaded_file is not None:
        df_raw = nacti_surova_data(uploaded_file.getvalue())
        
        st.header("2. Nastavení mapování")
        col_time = st.selectbox("Sloupec ČAS", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['time'])))
        col_lat = st.selectbox("Sloupec LATITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['latitude'])))
        col_lon = st.selectbox("Sloupec LONGITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['longitude'])))
        col_stiff = st.selectbox("Sloupec TUHOSTI (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['stiff', 'kb'])))
        col_dir = st.selectbox("Sloupec SMĚRU", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['direction', 'dir'])))
        
        vib_guess = najdi_vychozi_sloupec(df_raw.columns, ['amplitude', 'amp', 'frequency', 'freq', 'vibration', 'vibrace'])
        col_vib = st.selectbox("Sloupec VIBRACE (Amp/Freq)", df_raw.columns, index=df_raw.columns.get_loc(vib_guess) if vib_guess else 0)
        
        speed_options = ["Vypočítat z GPS (Záložní)"] + list(df_raw.columns)
        speed_guess = najdi_vychozi_sloupec(df_raw.columns, ['speed', 'rychlost'])
        col_speed = st.selectbox("Sloupec RYCHLOSTI", speed_options, index=speed_options.index(speed_guess) if speed_guess else 0)
        
        st.header("3. Parametry stroje")
        offset_m = st.number_input("Podélný posun anténa -> běhoun (m)", value=2.0, step=0.1)
        offset_transverse_m = st.number_input("Příčný posun anténa -> běhoun (m) [kladné = vpravo, záporné = vlevo]", value=0.20, step=0.05)
        min_speed_kmh = st.slider("Minimální rychlost (km/h)", 0.0, 5.0, 1.0, 0.5)
        
        st.header("4. Nastavení Limitů a Mřížky")
        target_min = st.number_input("Cílové minimum (Kb):", value=20.0, step=1.0)
        target_max = st.number_input("Cílové maximum (Kb):", value=45.0, step=1.0)
        grid_size_m = st.slider("Velikost mřížky (m)", 0.5, 3.0, 1.0, 0.5)
        
        st.header("5. Vizuál")
        colormap = st.selectbox("Paleta Heatmapy", ['Turbo', 'Viridis', 'Plasma', 'Inferno', 'Jet'], index=0)
        track_size = st.slider("Šířka vykresleného pásu v Mapách 2 a 5", min_value=10, max_value=50, value=22, step=1)

# --- 4. HLAVNÍ VÝPOČETNÍ LOGIKA ---
if uploaded_file is not None:
    
    df_valid = zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_m, offset_transverse_m, min_speed_kmh)
    
    if not df_valid.empty:
        max_pass_val = df_valid['pass_id'].max(skipna=True)
        max_pass = int(max_pass_val) if pd.notna(max_pass_val) else 1
        slider_max = max_pass if max_pass > 1 else 2
        
        avg_lat = df_valid['corr_lat'].mean()
        cos_correction = 1 / np.cos(np.radians(avg_lat))

        st.markdown("### ⏱️ Stroj času: Přehrávač hutnění")
        selected_pass = st.slider("Zobrazit stav pojezdu (Vrstvy) do čísla:", min_value=1, max_value=slider_max, value=max_pass)
        
        df_current = df_valid[df_valid['pass_id'] <= selected_pass].copy()
        
        lat_step = grid_size_m / 111320
        lon_step = grid_size_m / (111320 * np.cos(np.radians(avg_lat)))
        df_current['lat_bin'] = (df_current['corr_lat'] // lat_step) * lat_step + (lat_step / 2)
        df_current['lon_bin'] = (df_current['corr_lon'] // lon_step) * lon_step + (lon_step / 2)
        
        df_vib = df_current[(df_current['is_vibrating'] == True) & (df_current[col_stiff].notna())].copy()
        
        df_current_sorted = df_current.sort_values('parsed_time')

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🕹️ 1. Simulace (Vibrace)", 
            "🏁 2. Finální Kb", 
            "🔴 3. Mapa Anomálií", 
            "📊 4. Histogram",
            "🧊 5. Kontrola Žehlení"
        ])
        
        with tab1:
            st.subheader(f"Vývoj tuhosti (Pojezd 1 až {selected_pass}) - Vibrační běhy")
            st.caption("Čistá data reprezentující aktuální stav podkladu. Body zachovány pro detail.")
            fig_raw = go.Figure()
            if not df_vib.empty:
                fig_raw.add_trace(go.Scatter(
                    x=df_vib['corr_lon'], y=df_vib['corr_lat'], mode='markers',
                    marker=dict(size=6, color=df_vib[col_stiff], colorscale=colormap, showscale=True, opacity=0.7, colorbar=dict(title="Kb [-]")),
                    hovertext="Pojezd: " + df_vib['pass_id'].astype(str) + " | Kb: " + df_vib[col_stiff].round(1).astype(str)
                ))
            fig_raw.update_layout(
                xaxis=dict(tickformat=".7f", hoverformat=".7f"),
                yaxis=dict(scaleanchor="x", scaleratio=cos_correction, tickformat=".7f", hoverformat=".7f"), 
                height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_raw, use_container_width=True)

        with tab2:
            st.subheader("Mapa finální kvality (Poslední platný pojezd na daném místě)")
            st.caption("Zobrazuje se orientovaný obrys válce a středová tečka pro finální průjezd.")
            fig_final = go.Figure()
            if not df_vib.empty:
                df_vib_sorted = df_vib.sort_values('parsed_time')
                # Místo agregace průměru na mřížku vytáhneme přesný INDEX posledního záznamu v každé buňce
                idx_last = df_vib_sorted.groupby(['lat_bin', 'lon_bin'])['parsed_time'].idxmax()
                df_final = df_vib.loc[idx_last].copy()
                
                # Vrstva 1: Široký barevný pás (Natočený čtverec podle azimutu)
                # Odezva azimutu v Plotly je CCW, takže dáváme minus
                fig_final.add_trace(go.Scatter(
                    x=df_final['corr_lon'], y=df_final['corr_lat'], mode='markers',
                    marker=dict(
                        symbol='square', size=track_size, opacity=0.7,
                        angle=-df_final['heading'], # Kouzlo natočení
                        color=df_final[col_stiff], colorscale=colormap,
                        showscale=True, colorbar=dict(title="Finální Kb [-]")
                    ),
                    hoverinfo='skip', showlegend=False
                ))
                
                # Vrstva 2: Středová tečka jako základ
                fig_final.add_trace(go.Scatter(
                    x=df_final['corr_lon'], y=df_final['corr_lat'], mode='markers',
                    marker=dict(symbol='circle', size=4, color='black', opacity=0.8),
                    hovertext="Finální Kb: " + df_final[col_stiff].round(1).astype(str) + " | Pojezd: " + df_final['pass_id'].astype(str),
                    name='Střed běhounu'
                ))

            fig_final.update_layout(
                xaxis=dict(tickformat=".7f", hoverformat=".7f"),
                yaxis=dict(scaleanchor="x", scaleratio=cos_correction, tickformat=".7f", hoverformat=".7f"), 
                height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0), showlegend=False
            )
            st.plotly_chart(fig_final, use_container_width=True)

        with tab3:
            # Původní nezměněný kód
            st.subheader(f"Mapa Anomálií (Po pojezdu {selected_pass})")
            fig_anom = go.Figure()
            
            if not df_vib.empty:
                df_ok_bg = df_vib[(df_vib[col_stiff] >= target_min) & (df_vib[col_stiff] <= target_max)]
                fig_anom.add_trace(go.Scatter(
                    x=df_ok_bg['corr_lon'], y=df_ok_bg['corr_lat'], mode='markers',
                    marker=dict(size=4, color='#E5E7EB', opacity=0.3), name="V cílovém pásmu (OK)", hoverinfo='none'
                ))
                
                df_under = df_vib[df_vib[col_stiff] < target_min]
                if not df_under.empty:
                    fig_anom.add_trace(go.Scatter(
                        x=df_under['corr_lon'], y=df_under['corr_lat'], mode='markers',
                        marker=dict(size=6, color='red'), name=f"Nedohutněno (< {target_min})",
                        hovertext="Kb: " + df_under[col_stiff].round(1).astype(str)
                    ))
                
                df_over = df_vib[df_vib[col_stiff] > target_max]
                if not df_over.empty:
                    fig_anom.add_trace(go.Scatter(
                        x=df_over['corr_lon'], y=df_over['corr_lat'], mode='markers',
                        marker=dict(size=6, color='blue'), name=f"Tvrdá anomálie (> {target_max})",
                        hovertext="Kb: " + df_over[col_stiff].round(1).astype(str)
                    ))
            
            fig_anom.update_layout(
                xaxis=dict(tickformat=".7f", hoverformat=".7f"),
                yaxis=dict(scaleanchor="x", scaleratio=cos_correction, tickformat=".7f", hoverformat=".7f"), 
                height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_anom, use_container_width=True)

        with tab4:
            # Původní nezměněný kód
            st.subheader(f"Statistický vývoj (Do pojezdu {selected_pass})")
            fig_hist = go.Figure()
            if not df_vib.empty:
                fig_hist.add_trace(go.Histogram(x=df_vib[col_stiff], nbinsx=50, marker_color='gray', name='Počet bodů (Vib.)'))
                fig_hist.add_vrect(x0=target_min, x1=target_max, fillcolor="green", opacity=0.2, layer="below", line_width=0, annotation_text="Cílové pásmo", annotation_position="top left")
                
                max_val_99 = df_vib[col_stiff].quantile(0.99)
                safe_max = max(target_max * 1.2, max_val_99) if not pd.isna(max_val_99) else target_max * 1.5
                fig_hist.update_layout(xaxis_title="Hodnota Kb [-]", yaxis_title="Počet bodů", xaxis=dict(range=[0, safe_max]), height=500, bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            total_vib = len(df_vib)
            if total_vib > 0:
                pct_under = len(df_under) / total_vib * 100
                pct_ok = len(df_ok_bg) / total_vib * 100
                pct_over = len(df_over) / total_vib * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Pod limitem (Nedohutněno)", f"{pct_under:.1f} %")
                c2.metric("🟢 V cílovém pásmu (OK)", f"{pct_ok:.1f} %")
                c3.metric("🔵 Nad limitem (Příliš tvrdé)", f"{pct_over:.1f} %")

        with tab5:
            st.subheader("Analýza statického žehlení vrstvy")
            st.caption("Filtrujeme přesné hraniční body a zakreslujeme je po směru jízdy.")
            
            ironing_mode = st.radio(
                "Vyberte typ technologické kontroly:",
                ["Finální přežehlení (Uzavření povrchu proti srážkové vodě)", "Úvodní přežehlení (Stabilizace měkkého podkladu proti zaboření)"],
                horizontal=True
            )
            
            if not df_current_sorted.empty:
                if "Finální" in ironing_mode:
                    # Chceme ÚPLNĚ POSLEDNÍ průjezd na daném místě
                    idx_iron = df_current_sorted.groupby(['lat_bin', 'lon_bin'])['parsed_time'].idxmax()
                    label_true = "Finálně přežehleno (Zavřeno)"
                    label_false = "Nepřežehleno na závěr (Zůstalo otevřené)"
                    metric_title = "Plocha úspěšně finálně uzavřena staticky"
                else:
                    # Chceme ÚPLNĚ PRVNÍ průjezd na daném místě
                    idx_iron = df_current_sorted.groupby(['lat_bin', 'lon_bin'])['parsed_time'].idxmin()
                    label_true = "Úvodně přežehleno (Stabilizováno staticky)"
                    label_false = "Započato rovnou s vibrací (Riziko)"
                    metric_title = "Plocha úvodně ošetřena před vibrací"
                
                df_ironing = df_current_sorted.loc[idx_iron].copy()
                df_ironing['Selected_Status'] = df_ironing['is_vibrating'] == False
                
                colors_iron = np.where(df_ironing['Selected_Status'], '#22c55e', '#ef4444')
                labels_iron = np.where(df_ironing['Selected_Status'], label_true, label_false)

                fig_iron = go.Figure()
                
                # Vrstva 1: Široký pás (Natočený obrys válce)
                fig_iron.add_trace(go.Scatter(
                    x=df_ironing['corr_lon'], y=df_ironing['corr_lat'], mode='markers',
                    marker=dict(
                        symbol='square', size=track_size, opacity=0.7,
                        angle=-df_ironing['heading'],
                        color=colors_iron
                    ),
                    hoverinfo='skip'
                ))
                
                # Vrstva 2: Středová tečka 
                fig_iron.add_trace(go.Scatter(
                    x=df_ironing['corr_lon'], y=df_ironing['corr_lat'], mode='markers',
                    marker=dict(symbol='circle', size=4, color='black', opacity=0.8),
                    hovertext=labels_iron
                ))
                
                fig_iron.update_layout(
                    xaxis=dict(tickformat=".7f", hoverformat=".7f"),
                    yaxis=dict(scaleanchor="x", scaleratio=cos_correction, tickformat=".7f", hoverformat=".7f"), 
                    height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0), showlegend=False
                )
                st.plotly_chart(fig_iron, use_container_width=True)
                
                total_area_cells = len(df_ironing)
                ironed_cells = df_ironing['Selected_Status'].sum()
                pct_ironed = (ironed_cells / total_area_cells * 100) if total_area_cells > 0 else 0
                st.metric(metric_title, f"{pct_ironed:.1f} %")

    else:
        st.error("Po odfiltrování nezbyla žádná data.")
else:
    st.info("👋 Nahrajte CSV. Následně použijte slider pro přehrávání dat.")
