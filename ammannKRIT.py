import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from pyproj import Geod
from matplotlib.path import Path
import io
import csv

# GLOBÁLNÍ KONSTANTY
METERS_PER_DEGREE = 111320.0  # Přibližný převod stupňů na metry pro rovníkovou vzdálenost

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Detektor", layout="wide")
st.title("CCC Detektor")
st.caption("Aplikace navržená pro analýzu CCC pro účely diplomové práce. (Geometricky a korelačně optimalizováno)")

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
    except Exception:  
        sep = ';' if header_line.count(';') > header_line.count(',') else ','
    
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, skiprows=header_idx, on_bad_lines='skip', dtype=str, low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.replace('"', '').str.replace("'", "")
    return df

def najdi_vychozi_sloupec(columns, klicova_slova):
    for col in columns:
        for slovo in klicova_slova:
            if slovo in str(col).lower(): return col
    return columns[0] if len(columns) > 0 else None

# --- 3. GEODETICKÉ JÁDRO & POLYGONY ---
def vytvor_geometrii_pasu(df_geom, width_m):
    geod = Geod(ellps="WGS84")
    lon, lat, heading = df_geom['drum_lon'].values, df_geom['drum_lat'].values, df_geom['heading'].values
    length_array = df_geom['step_dist'].values
    
    fwd_az = heading
    bck_az = (heading + 180) % 360
    fwd_lon, fwd_lat, _ = geod.fwd(lon, lat, fwd_az, length_array / 2)
    bck_lon, bck_lat, _ = geod.fwd(lon, lat, bck_az, length_array / 2)
    
    right_az, left_az = (heading + 90) % 360, (heading - 90) % 360
    
    c1_lon, c1_lat, _ = geod.fwd(fwd_lon, fwd_lat, right_az, np.full(len(lon), width_m / 2))
    c2_lon, c2_lat, _ = geod.fwd(bck_lon, bck_lat, right_az, np.full(len(lon), width_m / 2))
    c3_lon, c3_lat, _ = geod.fwd(bck_lon, bck_lat, left_az, np.full(len(lon), width_m / 2))
    c4_lon, c4_lat, _ = geod.fwd(fwd_lon, fwd_lat, left_az, np.full(len(lon), width_m / 2))
    
    return c1_lon, c1_lat, c2_lon, c2_lat, c3_lon, c3_lat, c4_lon, c4_lat

@st.cache_data(show_spinner="Počítám geodetickou kinematiku a polygony...")
def zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, offset_right, min_speed_kmh, roller_width):
    df = df_raw.copy()
    for col in [col_lat, col_lon, col_stiff, col_vib]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.replace(' GMT', ''), utc=True, format='mixed', errors='coerce')
    df = df.dropna(subset=[col_lat, col_lon, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    if len(df) < 5: return pd.DataFrame()

    geod = Geod(ellps="WGS84")
    df['smooth_lon'] = df[col_lon].rolling(5, min_periods=1, center=True).mean()
    df['smooth_lat'] = df[col_lat].rolling(5, min_periods=1, center=True).mean()
    
    step = 2
    fwd_az, _, _ = geod.inv(df['smooth_lon'].shift(step).bfill().values, df['smooth_lat'].shift(step).bfill().values, df['smooth_lon'].values, df['smooth_lat'].values)
    df['heading'] = fwd_az % 360
    
    if col_dir in df.columns:
        is_reverse = df[col_dir].astype(str).str.strip().str.startswith('2')
        df.loc[is_reverse, 'heading'] = (df.loc[is_reverse, 'heading'] + 180) % 360
    
    temp_lon, temp_lat, _ = geod.fwd(df[col_lon].values, df[col_lat].values, df['heading'].values, np.full(len(df), offset_fwd))
    
    heading_right = (df['heading'] + 90) % 360 if offset_right < 0 else (df['heading'] - 90) % 360
    df['drum_lon'], df['drum_lat'], _ = geod.fwd(temp_lon, temp_lat, heading_right, np.full(len(df), abs(offset_right)))

    df['dt'] = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.01).bfill()
    
    _, _, dist_initial = geod.inv(df['drum_lon'].shift().bfill().values, df['drum_lat'].shift().bfill().values, df['drum_lon'].values, df['drum_lat'].values)
    
    if col_speed != "Vypočítat z GPS":
        df['speed_kmh'] = pd.to_numeric(df[col_speed].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df['speed_kmh'] = (dist_initial / df['dt']) * 3.6
        df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1, center=True).mean()

    df['is_vibrating'] = pd.to_numeric(df[col_vib].astype(str).str.replace(',', '.'), errors='coerce').fillna(0) > 0.1
    
    df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
    
    if not df_valid.empty:
        _, _, dist_valid = geod.inv(df_valid['drum_lon'].shift().bfill().values, df_valid['drum_lat'].shift().bfill().values, df_valid['drum_lon'].values, df_valid['drum_lat'].values)
        df_valid['step_dist'] = np.clip(dist_valid, 0.1, 10.0) 
        
        dir_cond = df_valid[col_dir] != df_valid[col_dir].shift().bfill() if col_dir in df_valid.columns else False
        vib_cond = df_valid['is_vibrating'] != df_valid['is_vibrating'].shift().bfill()
        time_gap = df_valid['parsed_time'].diff().dt.total_seconds() > 30
        
        df_valid['pass_id'] = (time_gap | dir_cond | vib_cond).cumsum() + 1
        
        c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y = vytvor_geometrii_pasu(df_valid, roller_width)
        df_valid['c1x'], df_valid['c1y'] = c1x, c1y
        df_valid['c2x'], df_valid['c2y'] = c2x, c2y
        df_valid['c3x'], df_valid['c3y'] = c3x, c3y
        df_valid['c4x'], df_valid['c4y'] = c4x, c4y
        
    return df_valid

# --- 4. EXAKTNÍ RASTERIZACE PŘES MATPLOTLIB PATH ---
@st.cache_data(show_spinner="Rasterizuji polygony a otisky šířky běhounu...")
def rasterizuj_do_mrizky(df, grid_size, avg_lat, col_stiff):
    df_work = df[['c1x', 'c1y', 'c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y', 'pass_id', 'is_vibrating', col_stiff, 'parsed_time']].copy()
    df_work.columns = ['c1x', 'c1y', 'c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y', 'pass_id', 'is_vib', 'kb', 'time']
    
    lat_f = METERS_PER_DEGREE
    lon_f = METERS_PER_DEGREE * np.cos(np.radians(avg_lat))
    
    c1x_m, c1y_m = df_work['c1x'].values * lon_f, df_work['c1y'].values * lat_f
    c2x_m, c2y_m = df_work['c2x'].values * lon_f, df_work['c2y'].values * lat_f
    c3x_m, c3y_m = df_work['c3x'].values * lon_f, df_work['c3y'].values * lat_f
    c4x_m, c4y_m = df_work['c4x'].values * lon_f, df_work['c4y'].values * lat_f
    
    pass_ids = df_work['pass_id'].values
    is_vibs = df_work['is_vib'].values
    kbs = df_work['kb'].values
    times = df_work['time'].values
    
    raster_records = []
    
    for i in range(len(df_work)):
        xs = [c1x_m[i], c2x_m[i], c3x_m[i], c4x_m[i]]
        ys = [c1y_m[i], c2y_m[i], c3y_m[i], c4y_m[i]]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        start_gx = (min_x // grid_size) * grid_size
        end_gx = (max_x // grid_size) * grid_size + grid_size
        start_gy = (min_y // grid_size) * grid_size
        end_gy = (max_y // grid_size) * grid_size + grid_size
        
        poly_path = Path(np.column_stack((xs, ys)))
        
        g_xs = np.arange(start_gx, end_gx + grid_size, grid_size)
        g_ys = np.arange(start_gy, end_gy + grid_size, grid_size)
        
        xx, yy = np.meshgrid(g_xs + grid_size/2, g_ys + grid_size/2)
        pts = np.column_stack((xx.flatten(), yy.flatten()))
        
        mask = poly_path.contains_points(pts)
        inside_pts = pts[mask]
        
        for pt in inside_pts:
            raster_records.append({
                'grid_x_m': pt[0] - grid_size/2,
                'grid_y_m': pt[1] - grid_size/2,
                'pass_id': pass_ids[i],
                'is_vib': is_vibs[i],
                'kb': kbs[i],
                'time': times[i]
            })
            
    df_raster = pd.DataFrame(raster_records)
    
    if not df_raster.empty:
        df_pass_avg = df_raster.groupby(['grid_x_m', 'grid_y_m', 'pass_id']).agg({
            'kb': 'mean',
            'is_vib': 'max',
            'time': 'max'
        }).reset_index()
        
        df_pass_avg['cell_lon'] = (df_pass_avg['grid_x_m'] + grid_size/2) / lon_f
        df_pass_avg['cell_lat'] = (df_pass_avg['grid_y_m'] + grid_size/2) / lat_f
        return df_pass_avg
    return pd.DataFrame()

# --- 5. BOČNÍ PANEL (UI) ---
with st.sidebar:
    st.header("📂 1. Data")
    uploaded_file = st.file_uploader("Nahrát CSV stroje", type=['csv'], key="machine_upload")
    
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

        st.header("📐 3. Stroj a Rastrování")
        offset_fwd = st.number_input("Posun antény podélně (m)", value=2.65, step=0.05)
        offset_right = st.number_input("Posun antény příčně (m)", value=0.26, step=0.01, help="Kladné = doprava, Záporné = doleva")
        roller_width = st.number_input("Šířka běhounu (m)", value=2.13, step=0.01)
        grid_size = st.slider("Přesnost Mřížky/Rasteru (m)", 0.2, 1.0, 0.5, 0.1)
        min_speed_kmh = st.number_input("Filtr stání (km/h)", value=0.5, step=0.1)

        st.header("🎯 4. Cíle a Vizualizace")
        target_min = st.number_input("Minimální Kb:", value=25.0, step=1.0)
        target_max = st.number_input("Maximální Kb:", value=50.0, step=1.0)
        colormap = st.selectbox("Paleta Kb", ['Turbo', 'Viridis', 'Jet'], index=0)
        
        st.header("📍 5. Kontrolní zkoušky")
        st.caption("Korelace s reálnými testy na stavbě z CSV")
        zobrazit_krize = st.checkbox("Vykreslit zkoušky do mapy a zjistit Kb", value=False)
        
        kontrolni_body = []
        uploaded_zkousky = st.file_uploader("Nahrát CSV se zkouškami", type=['csv'], key="zk_upload")
        
        if uploaded_zkousky is not None:
            try:
                df_zkousky = pd.read_csv(uploaded_zkousky, sep=None, engine='python')
                
                col_nazev = next((c for c in df_zkousky.columns if any(k in str(c).lower() for k in ['nazev', 'název', 'id', 'bod'])), df_zkousky.columns[0])
                col_lat_zk = next((c for c in df_zkousky.columns if any(k in str(c).lower() for k in ['lat', 'y'])), None)
                col_lon_zk = next((c for c in df_zkousky.columns if any(k in str(c).lower() for k in ['lon', 'x'])), None)
                
                if col_lat_zk and col_lon_zk:
                    for _, row in df_zkousky.iterrows():
                        lat_val = float(str(row[col_lat_zk]).replace(',', '.').strip())
                        lon_val = float(str(row[col_lon_zk]).replace(',', '.').strip())
                        
                        if pd.notna(lat_val) and pd.notna(lon_val) and lat_val != 0.0:
                            kontrolni_body.append({
                                "id": str(row[col_nazev]),
                                "lat": lat_val,
                                "lon": lon_val
                            })
                    st.success(f"✅ Úspěšně načteno {len(kontrolni_body)} zkoušek.")
                else:
                    st.error("❌ Nepodařilo se najít sloupce pro souřadnice. Hledám názvy s 'lat' a 'lon'.")
            except Exception as e:
                st.error(f"❌ Chyba při zpracování CSV se zkouškami: {e}")

# --- 6. RENDER SÍŤOVÝCH BUNĚK V PLOTLY ---
def generuj_mrizku_trace(df_grid, cell_size_m, avg_lat, color_val, color_scale, zmin, zmax, name):
    if df_grid.empty: return None
    
    lat_f = METERS_PER_DEGREE
    lon_f = METERS_PER_DEGREE * np.cos(np.radians(avg_lat))
    dx = (cell_size_m / 2) / lon_f
    dy = (cell_size_m / 2) / lat_f
    
    lons, lats = df_grid['cell_lon'].values, df_grid['cell_lat'].values
    
    c1x, c1y = lons + dx, lats + dy
    c2x, c2y = lons + dx, lats - dy
    c3x, c3y = lons - dx, lats - dy
    c4x, c4y = lons - dx, lats + dy
    
    x_vals = np.empty((len(c1x), 6))
    x_vals[:, 0], x_vals[:, 1], x_vals[:, 2], x_vals[:, 3], x_vals[:, 4] = c1x, c2x, c3x, c4x, c1x
    x_vals[:, 5] = np.nan
    x_flat = x_vals.flatten()
    
    y_vals = np.empty((len(c1y), 6))
    y_vals[:, 0], y_vals[:, 1], y_vals[:, 2], y_vals[:, 3], y_vals[:, 4] = c1y, c2y, c3y, c4y, c1y
    y_vals[:, 5] = np.nan
    y_flat = y_vals.flatten()
    
    if isinstance(color_val, str): fill_color = color_val
    else: fill_color = sample_colorscale(color_scale, [np.clip((color_val - zmin) / (zmax - zmin) if zmax > zmin else 0, 0, 1)])[0]

    return go.Scatter(x=x_flat, y=y_flat, fill='toself', mode='lines', line=dict(width=0), fillcolor=fill_color, opacity=0.9, name=name, hoverinfo='skip', showlegend=False)

# --- 7. HLAVNÍ LOGIKA ---
if uploaded_file is not None:
    df_valid = zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, offset_right, min_speed_kmh, roller_width)
    
    if not df_valid.empty:
        avg_lat = df_valid['drum_lat'].mean()
        df_raster = rasterizuj_do_mrizky(df_valid, grid_size, avg_lat, col_stiff)
        
        max_pass = int(df_valid['pass_id'].max())
        selected_pass = st.slider("Časová osa: Přehrát pojezdy do:", 1, max_pass, max_pass)
        
        df_current_raster = df_raster[df_raster['pass_id'] <= selected_pass].copy()
        
        cos_corr = 1 / np.cos(np.radians(avg_lat))
        map_layout = dict(scaleanchor="x", scaleratio=cos_corr, tickformat=".7f", hoverformat=".7f")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗺️ 1. Raw Trasa", "🔥 2. Překryvy", "🟩 3. Finální Tuhost", "🔴 4. Historická Anomálie", "📊 5. Statistika", "🧊 6. Žehlení"
        ])

        with tab1:
            st.subheader("Přesná dráha stroje (Střed běhounu)")
            fig1 = go.Figure()
            df_v = df_valid[df_valid['pass_id'] <= selected_pass]
            if not df_v.empty:
                step = max(1, len(df_v) // 4000)
                df_v_render = df_v.iloc[::step]
                
                fig1.add_trace(go.Scattergl(
                    x=df_v_render['drum_lon'], y=df_v_render['drum_lat'], mode='markers',
                    marker=dict(size=4, color=df_v_render[col_stiff], colorscale=colormap, showscale=True, colorbar=dict(title="Kb [-]")),
                    hovertext="Kb: " + df_v_render[col_stiff].round(1).astype(str)
                ))
            fig1.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0))
            fig1.update_xaxes(autorange="reversed")
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            st.subheader("Mapa překryvů (Pass Count)")
            fig2 = go.Figure()
            if not df_current_raster.empty:
                df_counts = df_current_raster.groupby(['cell_lon', 'cell_lat'])['pass_id'].nunique().reset_index(name='pass_count')
                max_count = df_counts['pass_count'].max()
                for c in range(1, max_count + 1):
                    df_c = df_counts[df_counts['pass_count'] == c]
                    trace = generuj_mrizku_trace(df_c, grid_size, avg_lat, c, 'hot', 0, max_count + 1, f"{c} Přejezdů")
                    if trace: 
                        trace.showlegend = True
                        fig2.add_trace(trace)
                        
                fig2.add_trace(go.Scattergl(
                    x=df_counts['cell_lon'], y=df_counts['cell_lat'], mode='markers', marker=dict(size=2, opacity=0.01, color='black'),
                    hovertext="Počet přejezdů: " + df_counts['pass_count'].astype(str) + "<br>Lat: " + df_counts['cell_lat'].round(7).astype(str) + "<br>Lon: " + df_counts['cell_lon'].round(7).astype(str),
                    showlegend=False
                ))
            fig2.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0), showlegend=True)
            fig2.update_xaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.subheader("Finální povrchová tuhost (Kb)")
            fig3 = go.Figure()
            
            df_final_for_calc = pd.DataFrame()
            
            if not df_current_raster.empty:
                valid_kb_mask = (df_current_raster['is_vib'] == True) & (df_current_raster['kb'] > 0)
                df_vib_raster = df_current_raster[valid_kb_mask]
                
                if not df_vib_raster.empty:
                    idx_last = df_vib_raster.groupby(['cell_lon', 'cell_lat'])['time'].idxmax()
                    df_final = df_vib_raster.loc[idx_last].copy().reset_index(drop=True)
                    df_final_for_calc = df_final
                    
                    bins = 15
                    zmin, zmax = target_min - 5, target_max + 5
                    df_final['color_bin'] = np.clip(np.floor((df_final['kb'] - zmin) / (zmax - zmin) * bins), 0, bins - 1)
                    
                    for b in range(bins):
                        df_bin = df_final[df_final['color_bin'] == b]
                        val_center = zmin + (b + 0.5) * ((zmax - zmin) / bins)
                        trace = generuj_mrizku_trace(df_bin, grid_size, avg_lat, val_center, colormap, zmin, zmax, f"Kb ~{val_center:.0f}")
                        if trace: fig3.add_trace(trace)
                    
                    fig3.add_trace(go.Scatter(
                        x=[None], y=[None], mode='markers',
                        marker=dict(colorscale=colormap, cmin=zmin, cmax=zmax, showscale=True, colorbar=dict(title="Kb [-]")),
                        showlegend=False, hoverinfo='none'
                    ))
                        
                    fig3.add_trace(go.Scattergl(
                        x=df_final['cell_lon'], y=df_final['cell_lat'], mode='markers', marker=dict(size=2, opacity=0.01, color='black'),
                        hovertext="Kb (Vyhlazeno): " + df_final['kb'].round(1).astype(str) + "<br>Lat: " + df_final['cell_lat'].round(7).astype(str) + "<br>Lon: " + df_final['cell_lon'].round(7).astype(str),
                        showlegend=False
                    ))
                    
                    # --- VYKRESLENÍ KŘÍŽKŮ V MAPĚ ---
                    if zobrazit_krize:
                        for pt in kontrolni_body:
                            if pt["lat"] != 0.0 and pt["lon"] != 0.0:
                                fig3.add_trace(go.Scattergl(
                                    x=[pt["lon"]], y=[pt["lat"]],
                                    mode='markers+text',
                                    marker=dict(
                                        symbol='cross', size=16, color='black', 
                                        line=dict(color='white', width=2)
                                    ),
                                    text=[str(pt["id"])],
                                    textposition="top right",
                                    textfont=dict(color="white", size=14, weight="bold"),
                                    name=str(pt['id']),
                                    hovertext=f"📍 {pt['id']}<br>Lat: {pt['lat']}<br>Lon: {pt['lon']}",
                                    showlegend=False
                                ))

            fig3.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            fig3.update_xaxes(autorange="reversed")
            st.plotly_chart(fig3, use_container_width=True)
            
            # --- AUTOMATICKÁ KORELAČNÍ TABULKA ---
            if zobrazit_krize and not df_final_for_calc.empty and kontrolni_body:
                st.markdown("### 📊 Korelace: Hodnoty válce v místech zkoušek (včetně historie)")
                
                lat_f = METERS_PER_DEGREE
                lon_f = METERS_PER_DEGREE * np.cos(np.radians(avg_lat))
                
                vysledky_zkousek = []
                
                for pt in kontrolni_body:
                    if pt["lat"] != 0.0 and pt["lon"] != 0.0:
                        dy = (df_final_for_calc['cell_lat'] - pt["lat"]) * lat_f
                        dx = (df_final_for_calc['cell_lon'] - pt["lon"]) * lon_f
                        vzdalenosti = np.sqrt(dx**2 + dy**2)
                        
                        if not vzdalenosti.empty:
                            nejblizsi_idx = vzdalenosti.idxmin()
                            min_vzdalenost = vzdalenosti[nejblizsi_idx]
                            
                            target_lon = df_final_for_calc.loc[nejblizsi_idx, 'cell_lon']
                            target_lat = df_final_for_calc.loc[nejblizsi_idx, 'cell_lat']
                            
                            history_cell = df_vib_raster[(df_vib_raster['cell_lon'] == target_lon) & (df_vib_raster['cell_lat'] == target_lat)].sort_values('pass_id')
                            
                            zaznam = {
                                "Zkouška (Název)": pt['id'],
                                "Zadaná Lat": pt["lat"],
                                "Zadaná Lon": pt["lon"],
                                "Odchylka od středu (m)": round(min_vzdalenost, 2),
                                "Finální Kb": round(df_final_for_calc.loc[nejblizsi_idx, 'kb'], 1)
                            }
                            
                            for _, row in history_cell.iterrows():
                                pass_num = int(row['pass_id'])
                                zaznam[f"Kb (Přejezd {pass_num})"] = round(row['kb'], 1)
                                
                            vysledky_zkousek.append(zaznam)
                
                if vysledky_zkousek:
                    df_vysledky = pd.DataFrame(vysledky_zkousek)
                    
                    zakladni_sloupce = ["Zkouška (Název)", "Zadaná Lat", "Zadaná Lon"]
                    # Bezpečné seřazení dynamických sloupců podle čísla přejezdu
                    prejezdy_sloupce = sorted([col for col in df_vysledky.columns if "Kb (Přejezd" in col], 
                                              key=lambda x: int(x.replace("Kb (Přejezd ", "").replace(")", "")))
                    konec_sloupce = ["Finální Kb", "Odchylka od středu (m)"]
                    
                    final_cols = zakladni_sloupce + prejezdy_sloupce + konec_sloupce
                    df_vysledky = df_vysledky[final_cols]
                    
                    st.dataframe(df_vysledky, use_container_width=True)
                    
                    max_odchylka = df_vysledky["Odchylka od středu (m)"].max()
                    if max_odchylka > (grid_size * 2):
                        st.warning(f"⚠️ Pozor: Některé body leží docela daleko (max {max_odchylka} m) od nejbližší projeté trasy válce. Zkontroluj si souřadnice!")

        with tab4:
            st.subheader("Analýza historických rizik a krusty (Bodově)")
            st.caption("Odhaluje místa s vytvořenou povrchovou krustou. Oranžové body = Finální pojezd v normě, ale v historii měřeno pod limitem.")
            fig4 = go.Figure()
            if not df_current_raster.empty:
                valid_kb_mask = (df_current_raster['is_vib'] == True) & (df_current_raster['kb'] > 0)
                df_vib_raster = df_current_raster[valid_kb_mask]
                
                if not df_vib_raster.empty:
                    df_hist_min = df_vib_raster.groupby(['cell_lon', 'cell_lat'])['kb'].min().reset_index(name='min_kb_history')
                    idx_last = df_vib_raster.groupby(['cell_lon', 'cell_lat'])['time'].idxmax()
                    df_final = df_vib_raster.loc[idx_last].copy()
                    df_anom = df_final.merge(df_hist_min, on=['cell_lon', 'cell_lat'])
                    
                    df_active_under = df_anom[df_anom['kb'] < target_min]
                    df_over = df_anom[df_anom['kb'] > target_max]
                    df_ok = df_anom[(df_anom['kb'] >= target_min) & (df_anom['kb'] <= target_max) & (df_anom['min_kb_history'] >= target_min)]
                    df_healed = df_anom[(df_anom['kb'] >= target_min) & (df_anom['kb'] <= target_max) & (df_anom['min_kb_history'] < target_min)]
                    
                    if not df_ok.empty:
                        fig4.add_trace(go.Scattergl(x=df_ok['cell_lon'], y=df_ok['cell_lat'], mode='markers', marker=dict(color='#E5E7EB', size=5), name="V normě (Trvale)",
                            hovertext="Lat: " + df_ok['cell_lat'].round(7).astype(str) + "<br>Lon: " + df_ok['cell_lon'].round(7).astype(str) + "<br>Finální Kb: " + df_ok['kb'].round(1).astype(str) + "<br>Min. historie: " + df_ok['min_kb_history'].round(1).astype(str)))
                    if not df_healed.empty:
                        fig4.add_trace(go.Scattergl(x=df_healed['cell_lon'], y=df_healed['cell_lat'], mode='markers', marker=dict(color='orange', size=8, symbol='diamond'), name="Vyléčené nedohutnění (Riziko krusty)",
                            hovertext="Lat: " + df_healed['cell_lat'].round(7).astype(str) + "<br>Lon: " + df_healed['cell_lon'].round(7).astype(str) + "<br>Finální Kb: " + df_healed['kb'].round(1).astype(str) + "<br>Min. historie: " + df_healed['min_kb_history'].round(1).astype(str)))
                    if not df_over.empty:
                        fig4.add_trace(go.Scattergl(x=df_over['cell_lon'], y=df_over['cell_lat'], mode='markers', marker=dict(color='rgba(59, 130, 246, 0.9)', size=7), name="Aktivní přezhutnění",
                            hovertext="Lat: " + df_over['cell_lat'].round(7).astype(str) + "<br>Lon: " + df_over['cell_lon'].round(7).astype(str) + "<br>Finální Kb: " + df_over['kb'].round(1).astype(str)))
                    if not df_active_under.empty:
                        fig4.add_trace(go.Scattergl(x=df_active_under['cell_lon'], y=df_active_under['cell_lat'], mode='markers', marker=dict(color='rgba(239, 68, 68, 0.9)', size=7), name="Aktivní nedohutnění",
                            hovertext="Lat: " + df_active_under['cell_lat'].round(7).astype(str) + "<br>Lon: " + df_active_under['cell_lon'].round(7).astype(str) + "<br>Finální Kb: " + df_active_under['kb'].round(1).astype(str)))

            fig4.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0), showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            fig4.update_xaxes(autorange="reversed")
            st.plotly_chart(fig4, use_container_width=True)

        with tab5:
            st.subheader("Plošná distribuce tuhosti (Kb)")
            if not df_current_raster.empty:
                valid_kb_mask = (df_current_raster['is_vib'] == True) & (df_current_raster['kb'] > 0)
                df_vib_raster = df_current_raster[valid_kb_mask]
                
                if not df_vib_raster.empty:
                    idx_last = df_vib_raster.groupby(['cell_lon', 'cell_lat'])['time'].idxmax()
                    df_final = df_vib_raster.loc[idx_last].copy()
                    fig5 = go.Figure(go.Histogram(x=df_final['kb'], nbinsx=60, marker_color='slategray'))
                    fig5.add_vrect(x0=target_min, x1=target_max, fillcolor="green", opacity=0.2)
                    fig5.update_layout(xaxis_title="Kb", yaxis_title="Rozloha (Počet buněk mřížky)")
                    st.plotly_chart(fig5, use_container_width=True)

        with tab6:
            st.subheader("Chronologická kontrola uzavření povrchu")
            fig6 = go.Figure()
            if not df_current_raster.empty:
                vib_times = df_current_raster[df_current_raster['is_vib'] == True].groupby(['cell_lon', 'cell_lat'])['time'].max()
                idx_all_last = df_current_raster.groupby(['cell_lon', 'cell_lat'])['time'].idxmax()
                df_last_any = df_current_raster.loc[idx_all_last].copy().set_index(['cell_lon', 'cell_lat'])
                
                df_last_any['max_vib_time'] = vib_times
                df_last_any['ever_vibrated'] = df_last_any['max_vib_time'].notna()
                df_last_any = df_last_any.reset_index()
                
                df_green = df_last_any[(df_last_any['is_vib'] == False) & (df_last_any['ever_vibrated'] == True)]
                df_red = df_last_any[(df_last_any['is_vib'] == True)]
                
                t_green = generuj_mrizku_trace(df_green, grid_size, avg_lat, 'rgba(34, 197, 94, 0.85)', colormap, 0, 1, "Uzavřeno (Statika na závěr)")
                t_red = generuj_mrizku_trace(df_red, grid_size, avg_lat, 'rgba(239, 68, 68, 0.85)', colormap, 0, 1, "Riziko (Zůstalo po vibraci)")
                
                if t_green:
                    t_green.showlegend = True
                    fig6.add_trace(t_green)
                if t_red:
                    t_red.showlegend = True
                    fig6.add_trace(t_red)
                
                fig6.add_trace(go.Scattergl(
                    x=df_last_any['cell_lon'], y=df_last_any['cell_lat'], mode='markers', marker=dict(size=2, opacity=0.01, color='black'),
                    hovertext="Stav uzavření<br>Lat: " + df_last_any['cell_lat'].round(7).astype(str) + "<br>Lon: " + df_last_any['cell_lon'].round(7).astype(str),
                    showlegend=False
                ))

            fig6.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0), showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            fig6.update_xaxes(autorange="reversed")
            st.plotly_chart(fig6, use_container_width=True)

    else:
        st.error("Žádná data neprošla filtry.")
