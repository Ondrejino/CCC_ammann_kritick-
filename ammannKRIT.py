import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from pyproj import Geod
import io
import csv

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Detektor: Vědecká Analýza", layout="wide")
st.title("🚜 CCC Detektor: Přesná geodetická analýza (Kód 3)")
st.caption("Fyzikálně a prostorově věrná vizualizace CCC dat s WGS84 projekcí a reálnými polygony.")

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

# --- 3. GEODETICKÉ JÁDRO A VEKTORIZACE (WGS84, Fyzika) ---
def vytvor_geometrii_pasu(df_geom, width_m, length_m):
    """Vektorizovaný výpočet rohů válce na elipsoidu WGS84 pomocí pyproj."""
    geod = Geod(ellps="WGS84")
    
    lon = df_geom['drum_lon'].values
    lat = df_geom['drum_lat'].values
    heading = df_geom['heading'].values
    
    # Krok 1: Posun vpřed/vzad pro určení osy běhounu (podélně)
    fwd_az = heading
    bck_az = (heading + 180) % 360
    fwd_lon, fwd_lat, _ = geod.fwd(lon, lat, fwd_az, np.full(len(lon), length_m / 2))
    bck_lon, bck_lat, _ = geod.fwd(lon, lat, bck_az, np.full(len(lon), length_m / 2))
    
    # Krok 2: Posun vlevo/vpravo pro získání 4 rohů (kolmo na směr jízdy)
    right_az = (heading + 90) % 360
    left_az = (heading - 90) % 360
    
    c1_lon, c1_lat, _ = geod.fwd(fwd_lon, fwd_lat, right_az, np.full(len(lon), width_m / 2)) # Přední pravý
    c2_lon, c2_lat, _ = geod.fwd(bck_lon, bck_lat, right_az, np.full(len(lon), width_m / 2)) # Zadní pravý
    c3_lon, c3_lat, _ = geod.fwd(bck_lon, bck_lat, left_az, np.full(len(lon), width_m / 2))  # Zadní levý
    c4_lon, c4_lat, _ = geod.fwd(fwd_lon, fwd_lat, left_az, np.full(len(lon), width_m / 2))  # Přední levý
    
    return c1_lon, c1_lat, c2_lon, c2_lat, c3_lon, c3_lat, c4_lon, c4_lat

@st.cache_data(show_spinner="Počítám exaktní WGS84 kinematiku a polygony...")
def zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, min_speed_kmh):
    df = df_raw.copy()
    
    for col in [col_lat, col_lon, col_stiff, col_vib]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.replace(' GMT', ''), utc=True, format='mixed', errors='coerce')
    df = df.dropna(subset=[col_lat, col_lon, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    if len(df) < 5: return pd.DataFrame()

    geod = Geod(ellps="WGS84")
    
    # 1. Vyhlazení GPS šumu pro stabilní azimut
    df['smooth_lon'] = df[col_lon].rolling(5, min_periods=1, center=True).mean()
    df['smooth_lat'] = df[col_lat].rolling(5, min_periods=1, center=True).mean()
    
    # 2. Výpočet reálného azimutu (heading) na elipsoidu
    # Shift posune hodnoty, abychom porovnávali bod i s bodem i+step
    step = 2
    fwd_az, _, _ = geod.inv(
        df['smooth_lon'].shift(step).bfill().values, df['smooth_lat'].shift(step).bfill().values,
        df['smooth_lon'].values, df['smooth_lat'].values
    )
    df['heading'] = fwd_az % 360
    
    # Krok vzad/vpřed detekce směru (zjednodušená pojistka proti couvání)
    if col_dir in df.columns:
        is_reverse = (df[col_dir].astype(str) == "0") | (df[col_dir].astype(str) == "-1")
        df['heading'] = np.where(is_reverse, (df['heading'] + 180) % 360, df['heading'])

    # 3. Posun anténa -> střed běhounu
    df['drum_lon'], df['drum_lat'], _ = geod.fwd(df[col_lon].values, df[col_lat].values, df['heading'].values, np.full(len(df), offset_fwd))

    # 4. Rychlost a filtrace stání
    if col_speed != "Vypočítat z GPS":
        df['speed_kmh'] = pd.to_numeric(df[col_speed].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df['dt'] = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.01).bfill()
        _, _, dist = geod.inv(df['drum_lon'].shift().bfill().values, df['drum_lat'].shift().bfill().values, df['drum_lon'].values, df['drum_lat'].values)
        df['speed_kmh'] = (dist / df['dt']) * 3.6
        df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1, center=True).mean()

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

        st.header("📐 3. Stroj a geometrie")
        offset_fwd = st.number_input("Posun anténa -> běhoun podélně (m)", value=2.0, step=0.1)
        roller_width = st.number_input("Šířka válce / běhounu (m)", value=2.13, step=0.01)
        segment_len = st.number_input("Délka vykreslovaného segmentu (m)", value=1.0, step=0.1)
        min_speed_kmh = st.number_input("Filtr stání (km/h)", value=0.5, step=0.1)

        st.header("🎯 4. Cíle")
        target_min = st.number_input("Minimální Kb:", value=25.0, step=1.0)
        target_max = st.number_input("Maximální Kb (Přezhutnění):", value=50.0, step=1.0)
        colormap = st.selectbox("Paleta", ['Turbo', 'Viridis', 'Jet'], index=0)

# --- 5. HLAVNÍ LOGIKA A VYKRESLOVÁNÍ ---
def generuj_optimalizovany_polygon_trace(df_subset, width, length, color_val, hover_text, color_scale, zmin, zmax, name):
    """Zásadní zrychlení: Vektorově sestaví cesty pro stovky polygonů do jednoho grafického objektu."""
    if df_subset.empty: return None
    
    # Získání WGS84 rohů (4 x Lon, 4 x Lat arrays)
    c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y = vytvor_geometrii_pasu(df_subset, width, length)
    
    # Vektorizované vložení np.nan pro přerušení čar (Místo dřívějšího python FOR cyklu .extend())
    x_vals = np.empty((len(c1x), 6))
    x_vals[:, 0], x_vals[:, 1], x_vals[:, 2], x_vals[:, 3], x_vals[:, 4] = c1x, c2x, c3x, c4x, c1x
    x_vals[:, 5] = np.nan
    x_flat = x_vals.flatten()
    
    y_vals = np.empty((len(c1y), 6))
    y_vals[:, 0], y_vals[:, 1], y_vals[:, 2], y_vals[:, 3], y_vals[:, 4] = c1y, c2y, c3y, c4y, c1y
    y_vals[:, 5] = np.nan
    y_flat = y_vals.flatten()
    
    # Určení barvy středu binu (jen pro plné barvy anomálií, pro plynulou paletu řešeno jinde)
    if isinstance(color_val, str):
        fill_color = color_val
    else:
        norm = (color_val - zmin) / (zmax - zmin)
        norm = np.clip(norm, 0, 1)
        fill_color = sample_colorscale(color_scale, [norm])[0]

    return go.Scatter(
        x=x_flat, y=y_flat, fill='toself', mode='lines',
        line=dict(width=0), fillcolor=fill_color, opacity=0.8,
        name=name, hoverinfo='skip'
    )

if uploaded_file is not None:
    df_valid = zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, min_speed_kmh)
    
    if not df_valid.empty:
        max_pass = int(df_valid['pass_id'].max())
        selected_pass = st.slider("Časová osa: Přehrát pojezdy do:", 1, max_pass, max_pass)
        
        df_current = df_valid[df_valid['pass_id'] <= selected_pass].copy()
        df_vib = df_current[df_current['is_vibrating'] == True].copy()
        
        # Kosinusová korekce pro proporční zobrazení map v ČR (aby nebyla mapa zploštělá)
        avg_lat = df_current['drum_lat'].mean()
        cos_corr = 1 / np.cos(np.radians(avg_lat))
        map_layout = dict(scaleanchor="x", scaleratio=cos_corr, tickformat=".7f", hoverformat=".7f")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ 1. Bodová dráha", "🟦 2. Plocha překryvů", "🔴 3. Anomálie", "📊 4. Histogram", "🧊 5. Žehlení (Hist.)"
        ])

        with tab1:
            st.subheader("Přesná WGS84 dráha běhounu")
            fig1 = go.Figure()
            if not df_vib.empty:
                fig1.add_trace(go.Scattergl(
                    x=df_vib['drum_lon'], y=df_vib['drum_lat'], mode='markers',
                    marker=dict(size=4, color=df_vib[col_stiff], colorscale=colormap, showscale=True),
                    hovertext="Kb: " + df_vib[col_stiff].round(1).astype(str)
                ))
            fig1.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            st.subheader("Plošná mapa (Polygony s překryvy)")
            st.caption("Fyzikální modely obdélníků stroje ve WGS84. Vykresleno vektorově pro výkon.")
            fig2 = go.Figure()
            if not df_vib.empty:
                # Roztřídění do 15 barevných košů radikálně snižuje počet DOM elementů, ale zachovává přesnou geometrii
                bins = 15
                zmin, zmax = df_vib[col_stiff].min(), df_vib[col_stiff].max()
                if zmin == zmax: zmax += 0.1
                df_vib['color_bin'] = np.clip(np.floor((df_vib[col_stiff] - zmin) / (zmax - zmin) * bins), 0, bins - 1)
                
                # Vykreslení od nejstarších po nejnovější pojezdy (aby nové překryly staré)
                df_sorted = df_vib.sort_values('parsed_time')
                
                for b in range(bins):
                    df_bin = df_sorted[df_sorted['color_bin'] == b]
                    if df_bin.empty: continue
                    val_center = zmin + (b + 0.5) * ((zmax - zmin) / bins)
                    trace = generuj_optimalizovany_polygon_trace(df_bin, roller_width, segment_len, val_center, "", colormap, zmin, zmax, f"Kb {val_center:.0f}")
                    if trace: fig2.add_trace(trace)
                
                # Dodatečná vrstva teček pro přesný hover (odečet souřadnic a hodnoty)
                fig2.add_trace(go.Scattergl(
                    x=df_sorted['drum_lon'], y=df_sorted['drum_lat'], mode='markers',
                    marker=dict(color='black', size=1, opacity=0.1),
                    hovertext="Lat: " + df_sorted['drum_lat'].round(6).astype(str) + " | Kb: " + df_sorted[col_stiff].round(1).astype(str)
                ))

            fig2.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.subheader("Extrémy a nedohutnění (Reálné polygony)")
            fig3 = go.Figure()
            if not df_vib.empty:
                df_under = df_vib[df_vib[col_stiff] < target_min]
                df_over = df_vib[df_vib[col_stiff] > target_max]
                df_ok = df_vib[(df_vib[col_stiff] >= target_min) & (df_vib[col_stiff] <= target_max)]
                
                t_ok = generuj_optimalizovany_polygon_trace(df_ok, roller_width, segment_len, '#E5E7EB', "OK", colormap, 0, 1, "V normě")
                t_under = generuj_optimalizovany_polygon_trace(df_under, roller_width, segment_len, 'rgba(239, 68, 68, 0.8)', "Pod limit", colormap, 0, 1, "Nedohutněno")
                t_over = generuj_optimalizovany_polygon_trace(df_over, roller_width, segment_len, 'rgba(59, 130, 246, 0.8)', "Přezhutněno", colormap, 0, 1, "Tvrdé")
                
                for t in [t_ok, t_under, t_over]:
                    if t: fig3.add_trace(t)

            fig3.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            st.subheader("Statistika naměřených Kb z vibračních pojezdů")
            if not df_vib.empty:
                fig4 = go.Figure(go.Histogram(x=df_vib[col_stiff], nbinsx=60, marker_color='slategray'))
                fig4.add_vrect(x0=target_min, x1=target_max, fillcolor="green", opacity=0.2)
                st.plotly_chart(fig4, use_container_width=True)

        with tab5:
            st.subheader("Analýza historie: Vibrace -> Následné žehlení")
            st.caption("Algoritmus kontroluje každý bod plochy: Proběhl zde někdy vibrační pojezd? A projel tímto bodem válec staticky ČASOVĚ POZDĚJI?")
            
            fig5 = go.Figure()
            if not df_current.empty:
                # 1. Časoprostorové indexování: Zaokrouhlíme WGS84 na ~0.5m grid pouze pro účely spárování historie místa
                df_hist = df_current.copy()
                df_hist['spatial_index'] = df_hist['drum_lat'].round(5).astype(str) + "_" + df_hist['drum_lon'].round(5).astype(str)
                
                # 2. Zjistíme čas POSLEDNÍHO vibračního pojezdu v daném místě
                vib_times = df_hist[df_hist['is_vibrating'] == True].groupby('spatial_index')['parsed_time'].max()
                
                # 3. Zjistíme čas POSLEDNÍHO statického pojezdu v daném místě
                stat_times = df_hist[df_hist['is_vibrating'] == False].groupby('spatial_index')['parsed_time'].max()
                
                # 4. Logika historie: Je čas statiky > čas vibrace?
                df_iron_check = pd.DataFrame({'last_vib': vib_times, 'last_stat': stat_times}).reset_index()
                
                # Místa, kde se vibrovalo, ale NEžehlilo POTÉ
                df_iron_check['ironed_ok'] = (df_iron_check['last_stat'].notna()) & (df_iron_check['last_stat'] > df_iron_check['last_vib'].fillna(pd.Timestamp.min.tz_localize('UTC')))
                
                # Nyní propojíme tento status zpět k nejnovějšímu plošnému bodu dané oblasti, abychom měli rohy
                idx_last_presence = df_hist.groupby('spatial_index')['parsed_time'].idxmax()
                df_visual = df_hist.loc[idx_last_presence].merge(df_iron_check[['spatial_index', 'ironed_ok']], on='spatial_index', how='left')
                df_visual['ironed_ok'] = df_visual['ironed_ok'].fillna(False)
                
                # Vykreslení reálných polygonů (Zelená = Správně uzavřeno na závěr, Červená = Zůstalo neuzavřené)
                df_good = df_visual[df_visual['ironed_ok'] == True]
                df_bad = df_visual[(df_visual['ironed_ok'] == False) & (df_visual['is_vibrating'] == True)] # Zajímá nás jen nedokončená vibrovaná plocha
                
                t_good = generuj_optimalizovany_polygon_trace(df_good, roller_width, segment_len, 'rgba(34, 197, 94, 0.7)', "Uzavřeno", colormap, 0, 1, "Úspěšně přežehleno")
                t_bad = generuj_optimalizovany_polygon_trace(df_bad, roller_width, segment_len, 'rgba(239, 68, 68, 0.7)', "Riziko - Neuzavřeno", colormap, 0, 1, "Nedokončeno (Bez statiky)")
                
                if t_good: fig5.add_trace(t_good)
                if t_bad: fig5.add_trace(t_bad)
                
            fig5.update_layout(yaxis=map_layout, height=700, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig5, use_container_width=True)

    else:
        st.error("Žádná data neprošla filtry.")
