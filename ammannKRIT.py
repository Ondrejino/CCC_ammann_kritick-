import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from pyproj import Geod
import io

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Ultimátní Detektor v4.0", layout="wide")
st.title("CCC Detektor: Komplexní analýza zhutnění")
st.caption("Profesionální plošné zobrazení stavby (reálná šířka běhounu 2,1 m).")

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
@st.cache_data(show_spinner="Vytvářím přesné 2,1m polygony jízdy a analyzuji data...")
def zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_m, offset_transverse_m, min_speed_kmh):
    df = df_raw.copy()
    
    for col in [col_lat, col_lon, col_vib]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    df[col_stiff] = pd.to_numeric(df[col_stiff].astype(str).str.replace(',', '.'), errors='coerce')
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.split(' GMT').str[0], errors='coerce')
    
    df['is_vibrating'] = df[col_vib].fillna(0) > 0.1
    
    if col_speed != "Vypočítat z GPS (Záložní)":
        df['speed_kmh'] = pd.to_numeric(df[col_speed].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df['speed_kmh'] = 0.0

    df = df.dropna(subset=[col_lat, col_lon, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    if len(df) <= 5: return pd.DataFrame()

    geod = Geod(ellps="WGS84")
    
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
    
    # 1. Přenos z GPS antény na střed válce
    temp_lons, temp_lats, _ = geod.fwd(df[col_lon].values, df[col_lat].values, machine_heading, np.full(len(df), offset_m))
    transverse_heading = (machine_heading + 90) % 360
    new_lons, new_lats, _ = geod.fwd(temp_lons, temp_lats, transverse_heading, np.full(len(df), offset_transverse_m))
    
    df['corr_lon'], df['corr_lat'] = new_lons, new_lats
    
    # 2. TVORBA POLYGONŮ (Tvůj nápad +- 1.05m od osy středu pro plošný pás jízdy)
    lon_L, lat_L, _ = geod.fwd(df['corr_lon'].values, df['corr_lat'].values, (machine_heading - 90) % 360, np.full(len(df), 1.05))
    lon_R, lat_R, _ = geod.fwd(df['corr_lon'].values, df['corr_lat'].values, (machine_heading + 90) % 360, np.full(len(df), 1.05))
    df['lon_L'] = lon_L
    df['lat_L'] = lat_L
    df['lon_R'] = lon_R
    df['lat_R'] = lat_R
    
    if col_speed == "Vypočítat z GPS (Záložní)":
        _, _, dist_step = geod.inv(df['corr_lon'].shift(), df['corr_lat'].shift(), df['corr_lon'], df['corr_lat'])
        time_step = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.001)
        df['speed_kmh'] = (dist_step / time_step) * 3.6
        df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1).mean().bfill() 
    
    df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
    if not df_valid.empty:
        time_gap_cond = df_valid['parsed_time'].diff().dt.total_seconds() > 30
        dir_cond = df_valid[col_dir] != df_valid[col_dir].shift().bfill()
        df_valid['pass_id'] = (time_gap_cond | dir_cond).cumsum() + 1
        
    return df_valid


# --- KRESLÍCÍ JÁDRO PRO PLOŠNÉ PÁSY ---
def vykresli_pasy(fig, df_subset, val_col, colormap_name, vmin, vmax, default_color=None):
    if df_subset.empty: return
    df_s = df_subset.sort_values('parsed_time').reset_index(drop=True)
    
    from collections import defaultdict
    color_traces = defaultdict(lambda: {'x': [], 'y': []})
    
    for pass_id, group in df_s.groupby('pass_id'):
        group = group.reset_index(drop=True)
        if len(group) < 2: continue
        
        lL, laL = group['lon_L'].values, group['lat_L'].values
        lR, laR = group['lon_R'].values, group['lat_R'].values
        times = group['parsed_time'].values
        
        if default_color is None and val_col in group.columns:
            vals = group[val_col].values
        else:
            vals = None
        
        for i in range(len(group) - 1):
            # Přemostění malých mezer (např. v horní vrstvě) max do 10 sekund
            time_diff = np.timedelta64(times[i+1] - times[i], 's').astype(float)
            if time_diff > 10: continue
                
            poly_x = [lL[i], lR[i], lR[i+1], lL[i+1], lL[i], None]
            poly_y = [laL[i], laR[i], laR[i+1], laL[i+1], laL[i], None]
            
            if default_color:
                c = default_color
            else:
                val = vals[i] if vals is not None else None
                if pd.isna(val): continue
                rng = vmax - vmin if vmax > vmin else 1
                n_val = max(0.0, min(1.0, (val - vmin) / rng))
                n_val = round(n_val, 2) # Shlukování podobných barev pro brutální zrychlení mapy
                c = pc.sample_colorscale(colormap_name, [n_val])[0]
                
            color_traces[c]['x'].extend(poly_x)
            color_traces[c]['y'].extend(poly_y)
            
    for c, data in color_traces.items():
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'], mode='lines', fill='toself', fillcolor=c,
            line=dict(color='rgba(255,255,255,0)', width=0), hoverinfo='skip', showlegend=False
        ))


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
        offset_m = st.number_input("Podélný posun anténa -> běhoun (m)", value=2.65, step=0.05)
        offset_transverse_m = st.number_input("Příčný posun anténa -> běhoun (m) [kladné = vpravo, záporné = vlevo]", value=0.26, step=0.01)
        min_speed_kmh = st.slider("Minimální rychlost (km/h)", 0.0, 5.0, 1.0, 0.5)
        
        st.header("4. Nastavení Limitů a Analýzy")
        target_min = st.number_input("Cílové minimum (Kb):", value=20.0, step=1.0)
        target_max = st.number_input("Cílové maximum (Kb):", value=45.0, step=1.0)
        grid_size_m = st.slider("Výpočetní rastr překryvů (m)", 0.5, 3.0, 0.5, 0.1, help="Slouží pouze pro výpočet 'co je nahoře', mapy využívají plošné pásy.")
        
        st.header("5. Vizuál")
        colormap = st.selectbox("Paleta Heatmapy", ['Turbo', 'Viridis', 'Plasma', 'Inferno', 'Jet'], index=0)

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
        
        # Mřížka nutná k nalezení finální (povrchové) vrstvy pod sebou poskládaných přejezdů
        lat_step = grid_size_m / 111320
        lon_step = grid_size_m / (111320 * np.cos(np.radians(avg_lat)))
        df_current['lat_bin'] = (df_current['corr_lat'] // lat_step) * lat_step + (lat_step / 2)
        df_current['lon_bin'] = (df_current['corr_lon'] // lon_step) * lon_step + (lon_step / 2)
        
        df_current['max_pass_in_bin'] = df_current.groupby(['lat_bin', 'lon_bin'])['pass_id'].transform('max')
        df_top_surface_all = df_current[df_current['pass_id'] == df_current['max_pass_in_bin']].copy()
        
        df_vib = df_current[(df_current['is_vibrating'] == True) & (df_current[col_stiff].notna())].copy()
        if not df_vib.empty:
            df_vib['max_vib_pass_in_bin'] = df_vib.groupby(['lat_bin', 'lon_bin'])['pass_id'].transform('max')
            df_top_surface_vib = df_vib[df_vib['pass_id'] == df_vib['max_vib_pass_in_bin']].copy()
            vmin_kb = df_vib[col_stiff].min()
            vmax_kb = df_vib[col_stiff].max()
        else:
            df_top_surface_vib = pd.DataFrame()
            vmin_kb, vmax_kb = 0, 100

        # Logika žehlení přes historii daného místa
        df_current_sorted = df_current.sort_values('parsed_time')
        def analyze_ironing(group):
            pass_vib = group.groupby('pass_id')['is_vibrating'].max()
            is_vib_array = pass_vib.values
            total_passes = len(is_vib_array)
            if not is_vib_array.any(): 
                return pd.Series({'Initial_Static': total_passes, 'Final_Static': total_passes, 'Total_Passes': total_passes})
            first_vib = is_vib_array.argmax()
            last_vib = len(is_vib_array) - 1 - is_vib_array[::-1].argmax()
            return pd.Series({'Initial_Static': first_vib, 'Final_Static': len(is_vib_array) - 1 - last_vib, 'Total_Passes': total_passes})

        df_ironing = df_current_sorted.groupby(['lat_bin', 'lon_bin']).apply(analyze_ironing).reset_index()
        df_ironing['Is_Initial_Ironed'] = df_ironing['Initial_Static'] > 0
        df_ironing['Is_Final_Ironed'] = df_ironing['Final_Static'] > 0

        df_top_surface_all = df_top_surface_all.merge(df_ironing, on=['lat_bin', 'lon_bin'], how='left')
        if not df_top_surface_vib.empty:
            pass_counts = df_current.groupby(['lat_bin', 'lon_bin'])['pass_id'].nunique().reset_index(name='Total_Pass_Count')
            df_top_surface_vib = df_top_surface_vib.merge(pass_counts, on=['lat_bin', 'lon_bin'], how='left')

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🕹️ 1. Simulace (Všechny vrstvy)", 
            "🏁 2. Finální Kb (Povrch)", 
            "🔴 3. Mapa Anomálií", 
            "📊 4. Histogram",
            "🧊 5. Kontrola Žehlení"
        ])
        
        with tab1:
            st.subheader(f"Vývoj tuhosti (Pojezd 1 až {selected_pass}) - Vibrační běhy")
            st.caption("Čistá data reprezentovaná reálnými polygony 2,1m širokého běhounu. Barvy přes sebe přepisují historii.")
            fig_raw = go.Figure()
            if not df_vib.empty:
                vykresli_pasy(fig_raw, df_vib, col_stiff, colormap, vmin_kb, vmax_kb)
                
                # Neviditelné body pro zachování Hover-textu
                fig_raw.add_trace(go.Scatter(
                    x=df_vib['corr_lon'], y=df_vib['corr_lat'], mode='markers',
                    marker=dict(size=12, color='rgba(0,0,0,0)'),
                    hovertext="Pojezd: " + df_vib['pass_id'].astype(str) + " | Kb: " + df_vib[col_stiff].round(1).astype(str),
                    showlegend=False
                ))
                # Falešný bod pro zobrazení barevné škály vedle mapy
                fig_raw.add_trace(go.Scatter(x=[df_vib['corr_lon'].mean()], y=[df_vib['corr_lat'].mean()], mode='markers', marker=dict(size=0, opacity=0, color=[vmin_kb, vmax_kb], colorscale=colormap, showscale=True, colorbar=dict(title="Kb [-]")), hoverinfo='skip', showlegend=False))
            
            fig_raw.update_layout(xaxis=dict(tickformat=".7f", hoverformat=".7f"), yaxis=dict(scaleanchor="x", scaleratio=cos_correction, tickformat=".7f", hoverformat=".7f"), height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_raw, use_container_width=True)

        with tab2:
            st.subheader("Mapa finální kvality (Finální povrchová plocha)")
            fig_final = go.Figure()
            if not df_top_surface_vib.empty:
                vykresli_pasy(fig_final, df_top_surface_vib, col_stiff, colormap, vmin_kb, vmax_kb)
                
                fig_final.add_trace(go.Scatter(
                    x=df_top_surface_vib['corr_lon'], y=df_top_surface_vib['corr_lat'], mode='markers', marker=dict(size=12, color='rgba(0,0,0,0)'),
                    hovertext="Finální Kb: " + df_top_surface_vib[col_stiff].round(1).astype(str) + " (Celkem pojezdů: " + df_top_surface_vib['Total_Pass_Count'].fillna(1).astype(int).astype(str) + ")", showlegend=False
                ))
                fig_final.add_trace(go.Scatter(x=[df_top_surface_vib['corr_lon'].mean()], y=[df_top_surface_vib['corr_lat'].mean()], mode='markers', marker=dict(size=0, opacity=0, color=[vmin_kb, vmax_kb], colorscale=colormap, showscale=True, colorbar=dict(title="Finální Kb [-]")), hoverinfo='skip', showlegend=False))
                
            fig_final.update_layout(xaxis=dict(tickformat=".7f", hoverformat=".7f"), yaxis=dict(scaleanchor="x", scaleratio=cos_correction, tickformat=".7f", hoverformat=".7f"), height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_final, use_container_width=True)

        with tab3:
            st.subheader(f"Mapa Anomálií povrchu (Po pojezdu {selected_pass})")
            fig_anom = go.Figure()
            if not df_top_surface_vib.empty:
                df_ok_bg = df_top_surface_vib[(df_top_surface_vib[col_stiff] >= target_min) & (df_top_surface_vib[col_stiff] <= target_max)]
                df_under = df_top_surface_vib[df_top_surface_vib[col_stiff] < target_min]
                df_over = df_top_surface_vib[df_top_surface_vib[col_stiff] > target_max]
                
                vykresli_pasy(fig_anom, df_ok_bg, col_stiff, colormap, vmin_kb, vmax_kb, default_color='rgba(229, 231, 235, 0.4)')
                vykresli_pasy(fig_anom, df_under, col_stiff, colormap, vmin_kb, vmax_kb, default_color='rgba(239, 68, 68, 0.9)')
                vykresli_pasy(fig_anom, df_over, col_stiff, colormap, vmin_kb, vmax_kb, default_color='rgba(59, 130, 246, 0.9)')
                
                # Sjednocené hover body
                fig_anom.add_trace(go.Scatter(x=df_top_surface_vib['corr_lon'], y=df_top_surface_vib['corr_lat'], mode='markers', marker=dict(size=12, color='rgba(0,0,0,0)'), hovertext="Kb: " + df_top_surface_vib[col_stiff].round(1).astype(str), showlegend=False))
                
            fig_anom.update_layout(xaxis=dict(tickformat=".7f", hoverformat=".7f"), yaxis=dict(scaleanchor="x", scaleratio=cos_correction, tickformat=".7f", hoverformat=".7f"), height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_anom, use_container_width=True)

        with tab4:
            st.subheader(f"Statistický vývoj povrchu (Do pojezdu {selected_pass})")
            fig_hist = go.Figure()
            if not df_top_surface_vib.empty:
                fig_hist.add_trace(go.Histogram(x=df_top_surface_vib[col_stiff], nbinsx=50, marker_color='gray', name='Počet bodů (Vib.)'))
                fig_hist.add_vrect(x0=target_min, x1=target_max, fillcolor="green", opacity=0.2, layer="below", line_width=0, annotation_text="Cílové pásmo", annotation_position="top left")
                max_val_99 = df_top_surface_vib[col_stiff].quantile(0.99)
                safe_max = max(target_max * 1.2, max_val_99) if not pd.isna(max_val_99) else target_max * 1.5
                fig_hist.update_layout(xaxis_title="Hodnota Kb [-]", yaxis_title="Počet bodů na povrchu", xaxis=dict(range=[0, safe_max]), height=500, bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            if len(df_top_surface_vib) > 0:
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Pod limitem (Nedohutněno)", f"{len(df_under) / len(df_top_surface_vib) * 100:.1f} %")
                c2.metric("🟢 V cílovém pásmu (OK)", f"{len(df_ok_bg) / len(df_top_surface_vib) * 100:.1f} %")
                c3.metric("🔵 Nad limitem (Příliš tvrdé)", f"{len(df_over) / len(df_top_surface_vib) * 100:.1f} %")

        with tab5:
            st.subheader("Analýza statického žehlení vrstvy (Plošné pokrytí)")
            ironing_mode = st.radio("Vyberte typ technologické kontroly:", ["Finální přežehlení (Uzavření povrchu)", "Úvodní přežehlení (Stabilizace podkladu)"], horizontal=True)
            
            if "Finální" in ironing_mode:
                df_top_surface_all['Selected_Status'] = df_top_surface_all['Is_Final_Ironed']
                label_true, label_false, metric_title = "Finálně přežehleno", "Nepřežehleno na závěr", "Plocha úspěšně finálně uzavřena staticky"
            else:
                df_top_surface_all['Selected_Status'] = df_top_surface_all['Is_Initial_Ironed']
                label_true, label_false, metric_title = "Úvodně přežehleno", "Započato s vibrací", "Plocha úvodně ošetřena před vibrací"

            fig_iron = go.Figure()
            if not df_top_surface_all.empty:
                df_iron_ok = df_top_surface_all[df_top_surface_all['Selected_Status'] == True]
                df_iron_bad = df_top_surface_all[df_top_surface_all['Selected_Status'] == False]
                
                vykresli_pasy(fig_iron, df_iron_ok, 'pass_id', colormap, 0, 1, default_color='#22c55e')
                vykresli_pasy(fig_iron, df_iron_bad, 'pass_id', colormap, 0, 1, default_color='#ef4444')
                
                labels_iron = np.where(df_top_surface_all['Selected_Status'], label_true, label_false)
                hover_text = [f"{lbl}<br>Úvodní statika (pojezdů): {int(init)}<br>Finální statika (pojezdů): {int(fin)}<br>Celkem pojezdů: {int(tot)}" if pd.notna(init) else "" for lbl, init, fin, tot in zip(labels_iron, df_top_surface_all['Initial_Static'], df_top_surface_all['Final_Static'], df_top_surface_all['Total_Passes'])]
                fig_iron.add_trace(go.Scatter(x=df_top_surface_all['corr_lon'], y=df_top_surface_all['corr_lat'], mode='markers', marker=dict(size=12, color='rgba(0,0,0,0)'), hovertext=hover_text, showlegend=False))
            
            fig_iron.update_layout(xaxis=dict(tickformat=".7f", hoverformat=".7f"), yaxis=dict(scaleanchor="x", scaleratio=cos_correction, tickformat=".7f", hoverformat=".7f"), height=700, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_iron, use_container_width=True)
            
            total_points = len(df_top_surface_all)
            st.metric(metric_title, f"{(df_top_surface_all['Selected_Status'].sum() / total_points * 100) if total_points > 0 else 0:.1f} %")

    else:
        st.error("Po odfiltrování nezbyla žádná data.")
else:
    st.info("👋 Nahrajte CSV. Následně použijte slider pro přehrávání dat.")
