import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from pyproj import Geod
import io
import csv

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Detektor: Exaktní WGS84", layout="wide")
st.title("🚜 CCC Detektor: Geodetická a fyzikální analýza (Kód 4)")
st.caption("Fyzikálně věrná vizualizace CCC dat ve WGS84 projekci s dynamickými polygony a přesností 1 cm.")

# --- 2. DATA PARSER (Odolný vůči reálným strojům) ---
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

# --- 3. GEODETICKÉ JÁDRO (WGS84, Dynamická Fyzika) ---
def vytvor_geometrii_pasu(df_geom, width_m):
    """Vektorizovaný výpočet rohů válce na elipsoidu WGS84 s dynamickou délkou kroků."""
    geod = Geod(ellps="WGS84")
    
    lon = df_geom['drum_lon'].values
    lat = df_geom['drum_lat'].values
    heading = df_geom['heading'].values
    # Délka polygonu je nyní dynamická (podle toho, kolik stroj ujel)
    length_array = df_geom['step_dist'].values
    
    # Krok 1: Posun vpřed/vzad pro určení osy běhounu
    fwd_az = heading
    bck_az = (heading + 180) % 360
    fwd_lon, fwd_lat, _ = geod.fwd(lon, lat, fwd_az, length_array / 2)
    bck_lon, bck_lat, _ = geod.fwd(lon, lat, bck_az, length_array / 2)
    
    # Krok 2: Posun vlevo/vpravo pro získání 4 rohů (šířka válce)
    right_az = (heading + 90) % 360
    left_az = (heading - 90) % 360
    
    c1_lon, c1_lat, _ = geod.fwd(fwd_lon, fwd_lat, right_az, np.full(len(lon), width_m / 2)) # Přední pravý
    c2_lon, c2_lat, _ = geod.fwd(bck_lon, bck_lat, right_az, np.full(len(lon), width_m / 2)) # Zadní pravý
    c3_lon, c3_lat, _ = geod.fwd(bck_lon, bck_lat, left_az, np.full(len(lon), width_m / 2))  # Zadní levý
    c4_lon, c4_lat, _ = geod.fwd(fwd_lon, fwd_lat, left_az, np.full(len(lon), width_m / 2))  # Přední levý
    
    return c1_lon, c1_lat, c2_lon, c2_lat, c3_lon, c3_lat, c4_lon, c4_lat

@st.cache_data(show_spinner="Počítám exaktní 2D kinematiku, offsety a polygony...")
def zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, offset_right, min_speed_kmh):
    df = df_raw.copy()
    
    for col in [col_lat, col_lon, col_stiff, col_vib]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.replace(' GMT', ''), utc=True, format='mixed', errors='coerce')
    df = df.dropna(subset=[col_lat, col_lon, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    if len(df) < 5: return pd.DataFrame()

    geod = Geod(ellps="WGS84")
    
    # 1. Vyhlazení mikrootřesů RTK GPS (Klouzavý průměr na souřadnicích)
    df['smooth_lon'] = df[col_lon].rolling(5, min_periods=1, center=True).mean()
    df['smooth_lat'] = df[col_lat].rolling(5, min_periods=1, center=True).mean()
    
    # 2. Směrový vektor z vyhlazených dat (Zabraňuje skákání azimutu)
    step = 2
    fwd_az, _, _ = geod.inv(
        df['smooth_lon'].shift(step).bfill().values, df['smooth_lat'].shift(step).bfill().values,
        df['smooth_lon'].values, df['smooth_lat'].values
    )
    df['heading'] = fwd_az % 360
    
    # 3. 2D KOREKCE ANTÉNY (Fwd + Right) do osy běhounu
    # Fáze A: Posun podélný
    temp_lon, temp_lat, _ = geod.fwd(df[col_lon].values, df[col_lat].values, df['heading'].values, np.full(len(df), offset_fwd))
    # Fáze B: Posun příčný (kolmo)
    heading_right = (df['heading'] + 90) % 360
    df['drum_lon'], df['drum_lat'], _ = geod.fwd(temp_lon, temp_lat, heading_right, np.full(len(df), offset_right))

    # 4. Rychlost a dynamická délka stopy (dist)
    df['dt'] = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.01).bfill()
    _, _, dist = geod.inv(df['drum_lon'].shift().bfill().values, df['drum_lat'].shift().bfill().values, df['drum_lon'].values, df['drum_lat'].values)
    
    # Fyzikální ošetření: Stopa nesmí být menší než 10 cm a delší než 2 m (ochrana před výpadky GPS pingů)
    df['step_dist'] = np.clip(dist, 0.1, 2.0)
    
    if col_speed != "Vypočítat z GPS":
        df['speed_kmh'] = pd.to_numeric(df[col_speed].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df['speed_kmh'] = (dist / df['dt']) * 3.6
        df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1, center=True).mean()

    # Filtrace a přiřazení sekvencí pojezdů
    df['is_vibrating'] = df[col_vib].fillna(0) > 0.1
    df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
    
    if not df_valid.empty:
        time_gap = df_valid['parsed_time'].diff().dt.total_seconds() > 30
        df_valid['pass_id'] = time_gap.cumsum() + 1
        
    return df_valid

# --- 4. BOČNÍ PANEL (UI) ---
with st.sidebar:
    st.header("📂 1. Data")
    uploaded_file = st.file_uploader("Nahrát CSV z válce", type=['csv'])
    
    if uploaded_file:
        df_raw = nacti_surova_data(uploaded_file.getvalue())
        
        st.header("⚙️ 2. Senzory")
        col_time = st.selectbox("Čas", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['time', 'cas'])))
        col_lat = st.selectbox("Latitude (WGS84)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['lat'])))
        col_lon = st.selectbox("Longitude (WGS84)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['lon'])))
        col_stiff = st.selectbox("Tuhost (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['stiff', 'kb', 'cmv'])))
        col_dir = st.selectbox("Směr (Vpřed/Vzad)", [None] + list(df_raw.columns), index=0)
        col_vib = st.selectbox("Vibrace (Amp/Freq)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['amp', 'freq', 'vib'])))
        col_speed = st.selectbox("Rychlost", ["Vypočítat z GPS"] + list(df_raw.columns), index=0)

        st.header("📐 3. Stroj a offsety")
        offset_fwd = st.number_input("Posun antény podélně (m)", value=2.0, step=0.1)
        offset_right = st.number_input("Posun antény příčně vpravo (m)", value=0.0, step=0.1, help="Kladné = doprava od středu, Záporné = doleva")
        roller_width = st.number_input("Šířka válce / běhounu (m)", value=2.13, step=0.01)
        min_speed_kmh = st.number_input("Filtr stání (km/h)", value=0.5, step=0.1)

        st.header("🎯 4. Cílové meze")
        target_min = st.number_input("Minimální Kb:", value=25.0, step=1.0)
        target_max = st.number_input("Maximální Kb (Přezhutnění):", value=50.0, step=1.0)
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
        # Striktní formát na 7 desetinných míst pro mapu
        map_layout = dict(scaleanchor="x", scaleratio=cos_corr, tickformat=".7f", hoverformat=".7f")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ 1. Bodová dráha", "🟦 2. Plocha překryvů", "🔴 3. Anomálie", "📊 4. Histogram", "🧊 5. Kontrola žehlení"
        ])

        with tab1:
            st.subheader("Přesná WGS84 dráha (Střed běhounu)")
            fig1 = go.Figure()
            if not df_vib.empty:
                fig1.add_trace(go.Scattergl(
                    x=df_vib['drum_lon'], y=df_vib['drum_lat'], mode='markers',
                    marker=dict(size=4, color=df_vib[col_stiff], colorscale=colormap, showscale=True),
                    hovertext="Kb: " + df_vib[col_stiff].round(1).astype(str) + " | Lat: " + df_vib['drum_lat'].round(7).astype(str) + " | Lon: " + df_vib['drum_lon'].round(7).astype(str)
                ))
            fig1.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            st.subheader("Plošná mapa (Polygony s překryvy)")
            fig2 = go.Figure()
            if not df_vib.empty:
                bins = 15
                zmin, zmax = df_vib[col_stiff].min(), df_vib[col_stiff].max()
                if zmin == zmax: zmax += 0.1
                df_vib['color_bin'] = np.clip(np.floor((df_vib[col_stiff] - zmin) / (zmax - zmin) * bins), 0, bins - 1)
                
                df_sorted = df_vib.sort_values('parsed_time')
                
                for b in range(bins):
                    df_bin = df_sorted[df_sorted['color_bin'] == b]
                    if df_bin.empty: continue
                    val_center = zmin + (b + 0.5) * ((zmax - zmin) / bins)
                    trace = generuj_optimalizovany_polygon_trace(df_bin, roller_width, val_center, colormap, zmin, zmax, f"Kb {val_center:.0f}")
                    if trace: fig2.add_trace(trace)
                
                # Exaktní odečet v mapě na 7 desetinných míst
                fig2.add_trace(go.Scattergl(
                    x=df_sorted['drum_lon'], y=df_sorted['drum_lat'], mode='markers',
                    marker=dict(color='black', size=2, opacity=0.1),
                    hovertext="Kb: " + df_sorted[col_stiff].round(1).astype(str) + "<br>Lat: " + df_sorted['drum_lat'].round(7).astype(str) + "<br>Lon: " + df_sorted['drum_lon'].round(7).astype(str)
                ))

            fig2.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.subheader("Interaktivní extrémy (Přezhutněno / Podhutněno)")
            fig3 = go.Figure()
            if not df_vib.empty:
                df_under = df_vib[df_vib[col_stiff] < target_min]
                df_over = df_vib[df_vib[col_stiff] > target_max]
                df_ok = df_vib[(df_vib[col_stiff] >= target_min) & (df_vib[col_stiff] <= target_max)]
                
                t_ok = generuj_optimalizovany_polygon_trace(df_ok, roller_width, '#E5E7EB', colormap, 0, 1, "V normě")
                t_under = generuj_optimalizovany_polygon_trace(df_under, roller_width, 'rgba(239, 68, 68, 0.8)', colormap, 0, 1, "Nedohutněno")
                t_over = generuj_optimalizovany_polygon_trace(df_over, roller_width, 'rgba(59, 130, 246, 0.8)', colormap, 0, 1, "Přezhutněno")
                
                for t in [t_ok, t_under, t_over]:
                    if t: fig3.add_trace(t)

                # Doplnění interaktivní hover vrstvy (aby šly body nalézt v terénu)
                fig3.add_trace(go.Scattergl(
                    x=df_vib['drum_lon'], y=df_vib['drum_lat'], mode='markers',
                    marker=dict(color='black', size=2, opacity=0.05),
                    hovertext="Kb: " + df_vib[col_stiff].round(1).astype(str) + "<br>Lat: " + df_vib['drum_lat'].round(7).astype(str) + "<br>Lon: " + df_vib['drum_lon'].round(7).astype(str)
                ))

            fig3.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            st.subheader("Statistika naměřených Kb (pouze vibrační data)")
            if not df_vib.empty:
                fig4 = go.Figure(go.Histogram(x=df_vib[col_stiff], nbinsx=60, marker_color='slategray'))
                fig4.add_vrect(x0=target_min, x1=target_max, fillcolor="green", opacity=0.2)
                st.plotly_chart(fig4, use_container_width=True)

        with tab5:
            st.subheader("Finální kontrola povrchu: Uzavření po vibraci")
            st.caption("Fyzikální mřížka zjišťuje, zda na daném 0.5m úseku proběhla nejprve vibrace a následně jako absolutně poslední krok statické žehlení.")
            fig5 = go.Figure()
            if not df_current.empty:
                df_hist = df_current.copy()
                
                # Exaktní "flat earth" aproximace pouze pro účely přiřazení do metrické buňky pro určení historie
                lat_f = 111320.0
                lon_f = 111320.0 * np.cos(np.radians(avg_lat))
                grid_s = 0.5
                df_hist['bin_x'] = (df_hist['drum_lon'] * lon_f // grid_s)
                df_hist['bin_y'] = (df_hist['drum_lat'] * lat_f // grid_s)
                
                # Zjištění, zda v dané buňce NĚKDY proběhla vibrace
                vib_bins = df_hist[df_hist['is_vibrating'] == True][['bin_x', 'bin_y']].drop_duplicates()
                vib_bins['ever_vibrated'] = True
                
                # Extrakce zcela posledního pojezdu pro každou fyzickou buňku
                last_idx = df_hist.groupby(['bin_x', 'bin_y'])['parsed_time'].idxmax()
                df_last = df_hist.loc[last_idx].copy()
                
                # Spojení informací
                df_iron = df_last.merge(vib_bins, on=['bin_x', 'bin_y'], how='left')
                df_iron['ever_vibrated'] = df_iron['ever_vibrated'].fillna(False)
                
                # Logika: 
                # Zelená (Vyžehleno): Někdy tam byla vibrace, ale ten úplně poslední pojezd byl statický
                df_green = df_iron[(df_iron['is_vibrating'] == False) & (df_iron['ever_vibrated'] == True)]
                
                # Červená (Riziko): Úplně poslední pojezd byl s vibrací (otevřený povrch)
                df_red = df_iron[df_iron['is_vibrating'] == True]
                
                t_green = generuj_optimalizovany_polygon_trace(df_green, roller_width, 'rgba(34, 197, 94, 0.75)', colormap, 0, 1, "Uzavřeno (Statika na závěr)")
                t_red = generuj_optimalizovany_polygon_trace(df_red, roller_width, 'rgba(239, 68, 68, 0.75)', colormap, 0, 1, "Otevřeno (Ukončeno vibrací)")
                
                if t_green: fig5.add_trace(t_green)
                if t_red: fig5.add_trace(t_red)
                
                # Hover info
                fig5.add_trace(go.Scattergl(
                    x=df_iron['drum_lon'], y=df_iron['drum_lat'], mode='markers',
                    marker=dict(color='black', size=2, opacity=0.05),
                    hovertext="Lat: " + df_iron['drum_lat'].round(7).astype(str) + "<br>Lon: " + df_iron['drum_lon'].round(7).astype(str)
                ))

            fig5.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig5, use_container_width=True)

    else:
        st.error("Žádná data neprošla filtry.")
