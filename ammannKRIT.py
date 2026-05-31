import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from pyproj import Geod
import io
import csv

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Detektor: Finální Geodézie", layout="wide")
st.title("🚜 CCC Detektor: Přesná geodetická analýza (Kód 5)")
st.caption("Aplikace optimalizovaná pro korelaci s terénními zkouškami (WGS84, vrstvení polygonů, chronologie).")

# --- 2. DATA PARSER ---
@st.cache_data(show_spinner="Analyzuji hlavičky a načítám surová data...")
def nacti_surova_data(file_bytes):
    sample_text = file_bytes[:50000].decode("utf-8", errors="ignore")
    lines = sample_text.splitlines()
    
    header_idx = 0
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if "latitude" in line_lower or "lat" in line_lower or "time" in line_lower or "gps:" in line_lower:
            header_idx = i
            break
            
    header_line = lines[header_idx]
    try:
        sep = csv.Sniffer().sniff(header_line).delimiter
    except:
        sep = ';' if header_line.count(';') > header_line.count(',') else ','
    
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, skiprows=header_idx, on_bad_lines='skip', dtype=str, low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.replace('"', '').str.replace("'", "")
    return df

def najdi_vychozi_sloupec(columns, klicova_slova):
    for col in columns:
        col_lower = str(col).lower()
        for slovo in klicova_slova:
            if slovo in col_lower:
                return col
    return columns[0] if len(columns) > 0 else None

# --- 3. GEODETICKÉ JÁDRO A VEKTORIZACE ---
def vytvor_geometrii_pasu(df_geom, width_m):
    """Vektorizovaný výpočet rohů válce na elipsoidu WGS84 s dynamickou délkou."""
    geod = Geod(ellps="WGS84")
    
    lon = df_geom['drum_lon'].values
    lat = df_geom['drum_lat'].values
    heading = df_geom['heading'].values
    length_array = df_geom['step_dist'].values
    
    fwd_az = heading
    bck_az = (heading + 180) % 360
    fwd_lon, fwd_lat, _ = geod.fwd(lon, lat, fwd_az, length_array / 2)
    bck_lon, bck_lat, _ = geod.fwd(lon, lat, bck_az, length_array / 2)
    
    right_az = (heading + 90) % 360
    left_az = (heading - 90) % 360
    
    c1_lon, c1_lat, _ = geod.fwd(fwd_lon, fwd_lat, right_az, np.full(len(lon), width_m / 2))
    c2_lon, c2_lat, _ = geod.fwd(bck_lon, bck_lat, right_az, np.full(len(lon), width_m / 2))
    c3_lon, c3_lat, _ = geod.fwd(bck_lon, bck_lat, left_az, np.full(len(lon), width_m / 2))
    c4_lon, c4_lat, _ = geod.fwd(fwd_lon, fwd_lat, left_az, np.full(len(lon), width_m / 2))
    
    return c1_lon, c1_lat, c2_lon, c2_lat, c3_lon, c3_lat, c4_lon, c4_lat

@st.cache_data(show_spinner="Počítám exaktní kinematiku a offsety...")
def zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, offset_right, min_speed_kmh):
    df = df_raw.copy()
    
    for col in [col_lat, col_lon, col_stiff, col_vib]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.replace(' GMT', ''), utc=True, format='mixed', errors='coerce')
    df = df.dropna(subset=[col_lat, col_lon, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    if len(df) < 5: return pd.DataFrame()

    geod = Geod(ellps="WGS84")
    
    # 1. Vyhlazení mikrootřesů RTK GPS
    df['smooth_lon'] = df[col_lon].rolling(5, min_periods=1, center=True).mean()
    df['smooth_lat'] = df[col_lat].rolling(5, min_periods=1, center=True).mean()
    
    # 2. Směrový vektor (Heading)
    step = 2
    fwd_az, _, _ = geod.inv(
        df['smooth_lon'].shift(step).bfill().values, df['smooth_lat'].shift(step).bfill().values,
        df['smooth_lon'].values, df['smooth_lat'].values
    )
    df['heading'] = fwd_az % 360
    
    # 3. KOREKCE ANTÉNY VŮČI STŘEDU BĚHOUNU
    # A: Posun podélný
    temp_lon, temp_lat, _ = geod.fwd(df[col_lon].values, df[col_lat].values, df['heading'].values, np.full(len(df), offset_fwd))
    
    # B: Posun příčný (Bezpečná logika pro záporné hodnoty)
    if offset_right < 0:
        heading_right = (df['heading'] - 90) % 360
        offset_r_val = abs(offset_right)
    else:
        heading_right = (df['heading'] + 90) % 360
        offset_r_val = offset_right
        
    df['drum_lon'], df['drum_lat'], _ = geod.fwd(temp_lon, temp_lat, heading_right, np.full(len(df), offset_r_val))

    # 4. Rychlost a dynamická délka stopy
    df['dt'] = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.01).bfill()
    _, _, dist = geod.inv(df['drum_lon'].shift().bfill().values, df['drum_lat'].shift().bfill().values, df['drum_lon'].values, df['drum_lat'].values)
    df['step_dist'] = np.clip(dist, 0.1, 2.0)
    
    if col_speed != "Vypočítat z GPS":
        df['speed_kmh'] = pd.to_numeric(df[col_speed].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df['speed_kmh'] = (dist / df['dt']) * 3.6
        df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1, center=True).mean()

    df['is_vibrating'] = df[col_vib].fillna(0) > 0.1
    df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
    
    if not df_valid.empty:
        # Vracení změny směru do výpočtu pass_id
        if col_dir in df_valid.columns:
            dir_cond = df_valid[col_dir] != df_valid[col_dir].shift().bfill()
        else:
            dir_cond = False
            
        time_gap = df_valid['parsed_time'].diff().dt.total_seconds() > 30
        df_valid['pass_id'] = (time_gap | dir_cond).cumsum() + 1
        
    return df_valid

# --- 4. BOČNÍ PANEL (UI) ---
with st.sidebar:
    st.header("📂 1. Data")
    uploaded_file = st.file_uploader("Nahrát CSV", type=['csv'])
    
    if uploaded_file:
        df_raw = nacti_surova_data(uploaded_file.getvalue())
        
        st.header("⚙️ 2. Senzory")
        col_time = st.selectbox("Čas", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['time', 'cas'])))
        col_lat = st.selectbox("Latitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['lat'])))
        col_lon = st.selectbox("Longitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['lon'])))
        col_stiff = st.selectbox("Tuhost (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['stiff', 'kb', 'cmv'])))
        col_dir = st.selectbox("Směr (Vpřed/Vzad)", [None] + list(df_raw.columns), index=list(df_raw.columns).index(najdi_vychozi_sloupec(df_raw.columns, ['dir', 'smer'])) + 1 if najdi_vychozi_sloupec(df_raw.columns, ['dir', 'smer']) else 0)
        col_vib = st.selectbox("Vibrace (Amp/Freq)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['amp', 'freq', 'vib'])))
        col_speed = st.selectbox("Rychlost", ["Vypočítat z GPS"] + list(df_raw.columns), index=0)

        st.header("📐 3. Stroj a offsety")
        offset_fwd = st.number_input("Posun antény podélně (m)", value=2.0, step=0.1)
        offset_right = st.number_input("Posun antény příčně (m)", value=0.0, step=0.1, help="Kladné = doprava, Záporné = doleva")
        roller_width = st.number_input("Šířka běhounu (m)", value=2.13, step=0.01)
        min_speed_kmh = st.number_input("Filtr stání (km/h)", value=0.5, step=0.1)

        st.header("🎯 4. Cíle")
        target_min = st.number_input("Minimální Kb:", value=25.0, step=1.0)
        target_max = st.number_input("Maximální Kb:", value=50.0, step=1.0)
        colormap = st.selectbox("Paleta", ['Turbo', 'Viridis', 'Jet'], index=0)

# --- 5. HLAVNÍ LOGIKA A VYKRESLOVÁNÍ ---
def generuj_optimalizovany_polygon_trace(df_subset, width, color_val, color_scale, zmin, zmax, name):
    if df_subset.empty: return None
    
    c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y = vytvor_geometrii_pasu(df_subset, width)
    
    x_vals = np.empty((len(c1x), 6))
    x_vals[:, 0], x_vals[:, 1], x_vals[:, 2], x_vals[:, 3], x_vals[:, 4] = c1x, c2x, c3x, c4x, c1x
    x_vals[:, 5] = np.nan
    x_flat = x_vals.flatten()
    
    y_vals = np.empty((len(c1y), 6))
    y_vals[:, 0], y_vals[:, 1], y_vals[:, 2], y_vals[:, 3], y_vals[:, 4] = c1y, c2y, c3y, c4y, c1y
    y_vals[:, 5] = np.nan
    y_flat = y_vals.flatten()
    
    if isinstance(color_val, str):
        fill_color = color_val
    else:
        norm = np.clip((color_val - zmin) / (zmax - zmin), 0, 1)
        fill_color = sample_colorscale(color_scale, [norm])[0]

    return go.Scatter(
        x=x_flat, y=y_flat, fill='toself', mode='lines',
        line=dict(width=0), fillcolor=fill_color, opacity=0.8,
        name=name, hoverinfo='skip'
    )

if uploaded_file is not None:
    df_valid = zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, offset_right, min_speed_kmh)
    
    if not df_valid.empty:
        max_pass = int(df_valid['pass_id'].max())
        selected_pass = st.slider("Časová osa: Přehrát pojezdy do:", 1, max_pass, max_pass)
        
        df_current = df_valid[df_valid['pass_id'] <= selected_pass].copy()
        df_vib = df_current[df_current['is_vibrating'] == True].copy()
        
        avg_lat = df_current['drum_lat'].mean()
        cos_corr = 1 / np.cos(np.radians(avg_lat))
        map_layout = dict(scaleanchor="x", scaleratio=cos_corr, tickformat=".7f", hoverformat=".7f")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ 1. Bodová dráha", "🟦 2. Plocha překryvů", "🔴 3. Anomálie", "📊 4. Histogram", "🧊 5. Kontrola žehlení"
        ])

        with tab1:
            st.subheader("Přesná WGS84 dráha")
            fig1 = go.Figure()
            if not df_vib.empty:
                fig1.add_trace(go.Scattergl(
                    x=df_vib['drum_lon'], y=df_vib['drum_lat'], mode='markers',
                    marker=dict(size=4, color=df_vib[col_stiff], colorscale=colormap, showscale=True),
                    hovertext="Kb: " + df_vib[col_stiff].round(1).astype(str) + "<br>Lat: " + df_vib['drum_lat'].round(7).astype(str) + "<br>Lon: " + df_vib['drum_lon'].round(7).astype(str)
                ))
            fig1.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            st.subheader("Plošná mapa (Správné chronologické vrstvení)")
            fig2 = go.Figure()
            if not df_vib.empty:
                bins = 15
                zmin, zmax = df_vib[col_stiff].min(), df_vib[col_stiff].max()
                if zmin == zmax: zmax += 0.1
                df_vib['color_bin'] = np.clip(np.floor((df_vib[col_stiff] - zmin) / (zmax - zmin) * bins), 0, bins - 1)
                
                # Iterace primárně přes pass_id pro zachování fyzického překrývání v čase
                for p_id in sorted(df_vib['pass_id'].unique()):
                    df_pass = df_vib[df_vib['pass_id'] == p_id]
                    for b in range(bins):
                        df_bin = df_pass[df_pass['color_bin'] == b]
                        if df_bin.empty: continue
                        val_center = zmin + (b + 0.5) * ((zmax - zmin) / bins)
                        trace = generuj_optimalizovany_polygon_trace(df_bin, roller_width, val_center, colormap, zmin, zmax, f"Pass {p_id} - Kb {val_center:.0f}")
                        if trace: fig2.add_trace(trace)
                
                fig2.add_trace(go.Scattergl(
                    x=df_vib['drum_lon'], y=df_vib['drum_lat'], mode='markers',
                    marker=dict(color='black', size=2, opacity=0.1),
                    hovertext="Kb: " + df_vib[col_stiff].round(1).astype(str) + "<br>Lat: " + df_vib['drum_lat'].round(7).astype(str) + "<br>Lon: " + df_vib['drum_lon'].round(7).astype(str)
                ))

            fig2.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.subheader("Interaktivní extrémy")
            fig3 = go.Figure()
            if not df_vib.empty:
                # Iterace opět přes pass_id pro správné z-index překrytí na hranách anomálií
                for p_id in sorted(df_vib['pass_id'].unique()):
                    df_pass = df_vib[df_vib['pass_id'] == p_id]
                    df_under = df_pass[df_pass[col_stiff] < target_min]
                    df_over = df_pass[df_pass[col_stiff] > target_max]
                    df_ok = df_pass[(df_pass[col_stiff] >= target_min) & (df_pass[col_stiff] <= target_max)]
                    
                    t_ok = generuj_optimalizovany_polygon_trace(df_ok, roller_width, '#E5E7EB', colormap, 0, 1, "V normě")
                    t_under = generuj_optimalizovany_polygon_trace(df_under, roller_width, 'rgba(239, 68, 68, 0.8)', colormap, 0, 1, "Nedohutněno")
                    t_over = generuj_optimalizovany_polygon_trace(df_over, roller_width, 'rgba(59, 130, 246, 0.8)', colormap, 0, 1, "Přezhutněno")
                    
                    for t in [t_ok, t_under, t_over]:
                        if t: fig3.add_trace(t)

                fig3.add_trace(go.Scattergl(
                    x=df_vib['drum_lon'], y=df_vib['drum_lat'], mode='markers',
                    marker=dict(color='black', size=2, opacity=0.05),
                    hovertext="Kb: " + df_vib[col_stiff].round(1).astype(str) + "<br>Lat: " + df_vib['drum_lat'].round(7).astype(str) + "<br>Lon: " + df_vib['drum_lon'].round(7).astype(str)
                ))

            fig3.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            st.subheader("Statistika naměřených Kb")
            if not df_vib.empty:
                fig4 = go.Figure(go.Histogram(x=df_vib[col_stiff], nbinsx=60, marker_color='slategray'))
                fig4.add_vrect(x0=target_min, x1=target_max, fillcolor="green", opacity=0.2)
                st.plotly_chart(fig4, use_container_width=True)

        with tab5:
            st.subheader("Chronologická kontrola uzavření povrchu (11 cm přesnost)")
            fig5 = go.Figure()
            if not df_current.empty:
                df_hist = df_current.copy()
                
                # Spatial index: Exaktní WGS84 na 6 desetinných míst
                df_hist['spatial_idx'] = df_hist['drum_lat'].round(6).astype(str) + "_" + df_hist['drum_lon'].round(6).astype(str)
                
                # Nalezení maxima času vibrace pro každou fyzickou buňku
                vib_times = df_hist[df_hist['is_vibrating'] == True].groupby('spatial_idx')['parsed_time'].max()
                
                # Získání úplně posledního pojezdu pro každou fyzickou buňku
                last_idx = df_hist.groupby('spatial_idx')['parsed_time'].idxmax()
                df_last = df_hist.loc[last_idx].copy()
                
                df_last = df_last.set_index('spatial_idx')
                df_last['max_vib_time'] = vib_times
                df_last = df_last.reset_index()
                
                # Zjištění, zda v dané buňce vůbec někdy proběhla vibrace
                df_last['ever_vibrated'] = df_last['max_vib_time'].notna()
                
                # Zelená: Poslední pojezd je statický A někdy předtím tam byla vibrace
                df_green = df_last[(df_last['is_vibrating'] == False) & (df_last['ever_vibrated'] == True)]
                # Červená: Poslední pojezd byl vibrační
                df_red = df_last[df_last['is_vibrating'] == True]
                
                t_green = generuj_optimalizovany_polygon_trace(df_green, roller_width, 'rgba(34, 197, 94, 0.75)', colormap, 0, 1, "Uzavřeno")
                t_red = generuj_optimalizovany_polygon_trace(df_red, roller_width, 'rgba(239, 68, 68, 0.75)', colormap, 0, 1, "Otevřeno")
                
                if t_green: fig5.add_trace(t_green)
                if t_red: fig5.add_trace(t_red)
                
                fig5.add_trace(go.Scattergl(
                    x=df_last['drum_lon'], y=df_last['drum_lat'], mode='markers',
                    marker=dict(color='black', size=2, opacity=0.05),
                    hovertext="Lat: " + df_last['drum_lat'].round(7).astype(str) + "<br>Lon: " + df_last['drum_lon'].round(7).astype(str)
                ))

            fig5.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

    else:
        st.error("Žádná data neprošla filtry.")
