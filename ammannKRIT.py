import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Ultimátní Detektor", layout="wide")
st.title("CCC Detektor: Komplexní analýza zhutnění")
st.caption("Interaktivní přehrávání stavby vrstvu po vrstvě.")

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
        
        st.header("2. Nastavení mapování")
        col_time = st.selectbox("Sloupec ČAS", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['time'])))
        col_lat = st.selectbox("Sloupec LATITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['latitude'])))
        col_lon = st.selectbox("Sloupec LONGITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['longitude'])))
        col_stiff = st.selectbox("Sloupec TUHOSTI (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['stiff', 'kb'])))
        col_dir = st.selectbox("Sloupec SMĚRU", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['direction', 'dir'])))
        
        speed_options = ["Vypočítat z GPS (Záložní)"] + list(df_raw.columns)
        speed_guess = najdi_vychozi_sloupec(df_raw.columns, ['speed', 'rychlost'])
        col_speed = st.selectbox("Sloupec RYCHLOSTI", speed_options, index=speed_options.index(speed_guess) if speed_guess else 0)
        
        st.header("3. Parametry analýzy")
        offset_m = st.number_input("Posun anténa -> běhoun (m)", value=2.0, step=0.1)
        min_speed_kmh = st.slider("Minimální rychlost (km/h)", 0.0, 5.0, 1.0, 0.5)
        
        st.header("4. Nastavení Limitů a Mřížky")
        target_min = st.number_input("Cílové minimum (Kb):", value=20.0, step=1.0)
        target_max = st.number_input("Cílové maximum (Kb):", value=45.0, step=1.0)
        grid_size_m = st.slider("Velikost mřížky pro Diferenční mapu (m)", 0.5, 3.0, 1.0, 0.5)
        
        st.header("5. Vizuál")
        colormap = st.selectbox("Paleta Heatmapy", ['Turbo', 'Viridis', 'Plasma', 'Inferno', 'Jet'], index=0)

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
    
    if len(df) > 5:
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
        new_lons, new_lats, _ = geod.fwd(df[col_lon].values, df[col_lat].values, machine_heading, np.full(len(df), offset_m))
        df['corr_lon'], df['corr_lat'] = new_lons, new_lats
        
        # Rychlost (Záložní)
        if col_speed == "Vypočítat z GPS (Záložní)":
            _, _, dist_step = geod.inv(df['corr_lon'].shift(), df['corr_lat'].shift(), df['corr_lon'], df['corr_lat'])
            time_step = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.001)
            df['speed_kmh'] = (dist_step / time_step) * 3.6
            df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1).mean().bfill() 
        
        # Očištění o stání
        df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
        
        if not df_valid.empty:
            # --- DETEKCE POJEZDŮ ---
            time_gap_cond = df_valid['parsed_time'].diff().dt.total_seconds() > 30
            dir_cond = df_valid[col_dir] != df_valid[col_dir].shift().bfill()
            df_valid['pass_id'] = (time_gap_cond | dir_cond).cumsum() + 1
            max_pass = int(df_valid['pass_id'].max())
            
            avg_lat = df_valid['corr_lat'].mean()
            cos_correction = 1 / np.cos(np.radians(avg_lat))

            # --- GLOBÁLNÍ STROJ ČASU ---
            st.markdown("### ⏱️ Stroj času: Přehrávač hutnění")
            selected_pass = st.slider("Zobrazit stav pojezdu (Vrstvy) do čísla:", min_value=1, max_value=max_pass, value=max_pass)
            
            # Filtrujeme data POUZE do zvoleného pojezdu
            df_current = df_valid[df_valid['pass_id'] <= selected_pass].copy()

            # ZÁLOŽKY
            tab1, tab2, tab3, tab4 = st.tabs([
                "🕹️ 1. Simulace (Surová Heatmapa)", 
                "🗺️ 2. Diferenční mapa (Δ Kb)", 
                "🔴 3. Mapa Anomálií", 
                "📊 4. Histogram"
            ])
            
            with tab1:
                st.subheader(f"Vývoj tuhosti (Pojezd 1 až {selected_pass})")
                st.caption("Čistá data reprezentující aktuální stav podkladu tak, jak ho válec zanechal po zvoleném pojezdu.")
                fig_raw = go.Figure()
                fig_raw.add_trace(go.Scatter(
                    x=df_current['corr_lon'], y=df_current['corr_lat'], mode='markers',
                    marker=dict(size=6, color=df_current[col_stiff], colorscale=colormap, showscale=True, opacity=0.7, colorbar=dict(title="Kb [-]")),
                    hovertext="Pojezd: " + df_current['pass_id'].astype(str) + " | Kb: " + df_current[col_stiff].round(1).astype(str)
                ))
                fig_raw.update_layout(yaxis=dict(scaleanchor="x", scaleratio=cos_correction), height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_raw, use_container_width=True)

            with tab2:
                st.subheader(f"Diferenční mapa: Čistý přírůstek tuhosti (Δ Kb)")
                st.caption(f"Ukazuje, co s pláním reálně udělaly pojezdy. Zelená = roste, Červená = materiál se rozvolňuje (dekompakce). Mřížka: {grid_size_m}x{grid_size_m}m.")
                
                # Výpočet Mřížky pro diferenci
                lat_step = grid_size_m / 111320
                lon_step = grid_size_m / (111320 * np.cos(np.radians(avg_lat)))
                df_current['lat_bin'] = (df_current['corr_lat'] // lat_step) * lat_step + (lat_step / 2)
                df_current['lon_bin'] = (df_current['corr_lon'] // lon_step) * lon_step + (lon_step / 2)
                
                df_current = df_current.sort_values('parsed_time')
                df_diff = df_current.groupby(['lat_bin', 'lon_bin']).agg(
                    First_Kb=(col_stiff, 'first'),
                    Last_Kb=(col_stiff, 'last'),
                    Pass_Count=('pass_id', 'nunique')
                ).reset_index()
                
                # Výpočet přírůstku
                df_diff['Delta_Kb'] = df_diff['Last_Kb'] - df_diff['First_Kb']
                # Body s pouze 1 pojezdem nemají přírůstek
                df_diff.loc[df_diff['Pass_Count'] == 1, 'Delta_Kb'] = 0 
                
                fig_diff = go.Figure()
                
                # Diverging colormap: Červená (záporné), Bílá/Šedá (Nula), Modrá/Zelená (Kladné)
                fig_diff.add_trace(go.Scatter(
                    x=df_diff['lon_bin'], y=df_diff['lat_bin'], mode='markers',
                    marker=dict(
                        symbol='square', size=15, opacity=0.8,
                        color=df_diff['Delta_Kb'], 
                        colorscale='RdBu', # Red to Blue (Zero is white/gray)
                        cmin=-15, cmax=15, # Symetrický rozsah kolem nuly
                        showscale=True, colorbar=dict(title="Δ Kb")
                    ),
                    hovertext="Δ Kb: " + df_diff['Delta_Kb'].round(1).astype(str) + " (Průjezdů: " + df_diff['Pass_Count'].astype(str) + ")"
                ))
                fig_diff.update_layout(yaxis=dict(scaleanchor="x", scaleratio=cos_correction), height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_diff, use_container_width=True)

            with tab3:
                st.subheader(f"Mapa Anomálií (Po pojezdu {selected_pass})")
                st.caption(f"Kontrola limitů. Očekávaný cíl: {target_min} až {target_max} Kb. Šedá = V pořádku.")
                fig_anom = go.Figure()
                
                df_ok_bg = df_current[(df_current[col_stiff] >= target_min) & (df_current[col_stiff] <= target_max)]
                fig_anom.add_trace(go.Scatter(
                    x=df_ok_bg['corr_lon'], y=df_ok_bg['corr_lat'], mode='markers',
                    marker=dict(size=4, color='#E5E7EB', opacity=0.3), name="V cílovém pásmu (OK)", hoverinfo='none'
                ))
                
                df_under = df_current[df_current[col_stiff] < target_min]
                if not df_under.empty:
                    fig_anom.add_trace(go.Scatter(
                        x=df_under['corr_lon'], y=df_under['corr_lat'], mode='markers',
                        marker=dict(size=6, color='red'), name=f"Nedohutněno (< {target_min})",
                        hovertext="Kb: " + df_under[col_stiff].round(1).astype(str)
                    ))
                
                df_over = df_current[df_current[col_stiff] > target_max]
                if not df_over.empty:
                    fig_anom.add_trace(go.Scatter(
                        x=df_over['corr_lon'], y=df_over['corr_lat'], mode='markers',
                        marker=dict(size=6, color='blue'), name=f"Tvrdá anomálie (> {target_max})",
                        hovertext="Kb: " + df_over[col_stiff].round(1).astype(str)
                    ))
                
                fig_anom.update_layout(yaxis=dict(scaleanchor="x", scaleratio=cos_correction), height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_anom, use_container_width=True)

            with tab4:
                st.subheader(f"Statistický vývoj (Do pojezdu {selected_pass})")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=df_current[col_stiff], nbinsx=50, marker_color='gray', name='Počet bodů'))
                fig_hist.add_vrect(x0=target_min, x1=target_max, fillcolor="green", opacity=0.2, layer="below", line_width=0, annotation_text="Cílové pásmo", annotation_position="top left")
                
                max_val_99 = df_valid[col_stiff].quantile(0.99)
                safe_max = max(target_max * 1.2, max_val_99) 
                
                fig_hist.update_layout(xaxis_title="Hodnota Kb [-]", yaxis_title="Počet bodů", xaxis=dict(range=[0, safe_max]), height=500, bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                total = len(df_current)
                pct_under = len(df_under) / total * 100
                pct_ok = len(df_ok_bg) / total * 100
                pct_over = len(df_over) / total * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Pod limitem (Nedohutněno)", f"{pct_under:.1f} %")
                c2.metric("🟢 V cílovém pásmu (OK)", f"{pct_ok:.1f} %")
                c3.metric("🔵 Nad limitem (Příliš tvrdé)", f"{pct_over:.1f} %")
        else:
            st.error("Po odfiltrování nezbyla žádná data.")
else:
    st.info("👋 Nahrajte CSV. Následně použijte slider pro přehrávání dat.")
