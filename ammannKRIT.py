import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Detektor anomálií", layout="wide")
st.title("CCC Detektor: Vyhledávání kritických oblastí")
st.caption("Automatická detekce 'měkkých míst' s odfiltrováním okrajových podmínek jízdy.")

# --- 2. POMOCNÉ FUNKCE (Zůstávají stejné) ---
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

def najdi_vychozi_sloupec(columns, klicove_slovo):
    for col in columns:
        if klicove_slovo in col.lower():
            return col
    return columns[0] if len(columns) > 0 else None

# --- 3. BOČNÍ PANEL ---
with st.sidebar:
    st.header("1. Nahrání dat")
    uploaded_file = st.file_uploader("Vložte CSV z válce", type=['csv'])
    
    if uploaded_file is not None:
        df_raw = nacti_surova_data(uploaded_file.getvalue())
        
        st.header("2. Ověření sloupců")
        col_time = st.selectbox("Sloupec s ČASEM", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'time')))
        col_lat = st.selectbox("Sloupec LATITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'latitude')))
        col_lon = st.selectbox("Sloupec LONGITUDE", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'longitude')))
        col_stiff = st.selectbox("Sloupec TUHOSTI (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'stiff')))
        col_dir = st.selectbox("Sloupec SMĚRU", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, 'direction')))
        
        st.header("3. Očištění dat (Filtry)")
        offset_m = st.number_input("Posun anténa -> běhoun (m)", value=2.0, step=0.1)
        forward_val = st.text_input("Hodnota jízdy VPŘED", value="1")
        
        st.markdown("**Filtrace rychlosti (odstranění stání a otáčení):**")
        min_speed_kmh = st.slider("Minimální rychlost (km/h)", 0.0, 5.0, 1.0, 0.5, help="Body s nižší rychlostí budou ignorovány.")
        
        st.header("4. Nastavení kritických zón")
        st.markdown("Definujte hranice pro klasifikaci tuhosti Kb:")
        lim_critical = st.number_input("Kritická hodnota (Méně než:)", value=15.0, step=1.0)
        lim_warning = st.number_input("Varovná hodnota (Méně než:)", value=25.0, step=1.0)
        
        st.header("5. Zobrazení na mapě")
        show_critical = st.checkbox("🔴 Zobrazit Kritická místa", value=True)
        show_warning = st.checkbox("🟠 Zobrazit Varovná místa", value=True)
        show_good = st.checkbox("🟢 Zobrazit Vyhovující (pouze pro kontext)", value=False)
        decimation = st.slider("Decimace mapy (pro rychlost)", 1, 50, 5, 1)

# --- 4. HLAVNÍ VÝPOČETNÍ LOGIKA ---
if uploaded_file is not None:
    df = df_raw.copy()
    
    # Převody datových typů
    for col in [col_lat, col_lon, col_stiff]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.split(' GMT').str[0], errors='coerce')
    df = df.dropna(subset=[col_lat, col_lon, col_stiff, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    if len(df) < 5:
        st.error("Málo platných dat.")
    else:
        geod = Geod(ellps="WGS84")
        
        # --- KOREKCE GPS A VÝPOČET AZIMUTU ---
        window = 5
        df['smooth_lon'] = df[col_lon].rolling(window=window, min_periods=1, center=True).mean()
        df['smooth_lat'] = df[col_lat].rolling(window=window, min_periods=1, center=True).mean()
        
        lons_s, lats_s = df['smooth_lon'].values, df['smooth_lat'].values
        lons, lats = df[col_lon].values, df[col_lat].values
        
        step = 3
        fwd_az = np.zeros(len(df))
        az, _, dists = geod.inv(lons_s[:-step], lats_s[:-step], lons_s[step:], lats_s[step:])
        fwd_az[:-step] = az
        fwd_az[-step:] = az[-1] if len(az) > 0 else 0
        
        is_forward = (df[col_dir].astype(str) == str(forward_val)).values
        machine_heading = np.where(is_forward, fwd_az, (fwd_az + 180) % 360)
        new_lons, new_lats, _ = geod.fwd(lons, lats, machine_heading, np.full(len(lons), offset_m))
        df['corr_lon'], df['corr_lat'] = new_lons, new_lats
        
        # --- VÝPOČET RYCHLOSTI PRO FILTRACI OKRAJŮ ---
        # Spočítáme vzdálenost mezi sousedními body
        _, _, dist_step = geod.inv(df['corr_lon'].shift(), df['corr_lat'].shift(), df['corr_lon'], df['corr_lat'])
        time_step = df['parsed_time'].diff().dt.total_seconds()
        
        # Rychlost v m/s převedená na km/h
        df['speed_kmh'] = (dist_step / time_step) * 3.6
        # Vyhlazení rychlosti, aby nevyhazovala náhodné výkyvy
        df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1).mean().bfill() 
        
        # --- APLIKACE FILTRŮ ---
        # Zahození dat, kde stroj stál nebo jel moc pomalu (otáčení atd.)
        df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
        
        # Klasifikace tuhosti do kategorií
        conditions = [
            (df_valid[col_stiff] < lim_critical),
            (df_valid[col_stiff] >= lim_critical) & (df_valid[col_stiff] < lim_warning),
            (df_valid[col_stiff] >= lim_warning)
        ]
        choices = ['Kritické', 'Varovné', 'Vyhovující']
        df_valid['Kategorie'] = np.select(conditions, choices, default='Neznámé')
        
        # Vizuální korekce mapy
        avg_lat = df_valid['corr_lat'].mean()
        cos_correction = 1 / np.cos(np.radians(avg_lat))
        
        # --- VYKRESLENÍ MAPY ---
        st.subheader("Mapa kritických anomálií")
        fig = go.Figure()
        
        # 1. Podkres celkové trasy (Očištěná od stání, zdecimovaná)
        df_bg = df_valid.iloc[::decimation]
        fig.add_trace(go.Scatter(
            x=df_bg['corr_lon'], y=df_bg['corr_lat'], mode='markers',
            marker=dict(size=3, color='#E5E7EB', opacity=0.5), name='Celková ujetá trasa', hoverinfo='none'
        ))
        
        # 2. Vykreslení kategorií na základě zaškrtávátek
        colors = {'Kritické': 'red', 'Varovné': 'orange', 'Vyhovující': 'green'}
        
        for cat, color in colors.items():
            # Zda má být tato kategorie zobrazena
            if (cat == 'Kritické' and not show_critical) or \
               (cat == 'Varovné' and not show_warning) or \
               (cat == 'Vyhovující' and not show_good):
                continue
                
            df_cat = df_valid[df_valid['Kategorie'] == cat]
            # Pro kritická místa chceme vidět vše, nebudeme tolik decimovat
            if not df_cat.empty:
                step_cat = 1 if cat == 'Kritické' else (2 if cat == 'Varovné' else decimation)
                df_plot = df_cat.iloc[::step_cat]
                
                fig.add_trace(go.Scatter(
                    x=df_plot['corr_lon'], y=df_plot['corr_lat'], mode='markers',
                    marker=dict(size=6, color=color, line=dict(width=1, color='dark'+color)),
                    name=f'{cat} (Kb < {lim_critical if cat == "Kritické" else lim_warning})',
                    hovertext=df_plot[col_stiff].round(1).astype(str) + ' Kb (' + df_plot['speed_kmh'].round(1).astype(str) + ' km/h)'
                ))

        fig.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=cos_correction),
            height=800, dragmode='pan',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.8)")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- STATISTIKA ---
        st.subheader("Statistika po filtraci")
        col1, col2, col3 = st.columns(3)
        col1.metric("Celkem analyzováno bodů", len(df_valid))
        col2.metric("Kritických bodů", len(df_valid[df_valid['Kategorie'] == 'Kritické']))
        col3.metric("Odfiltrováno okrajů (stání)", len(df) - len(df_valid))
        
        # Export dat
        csv_export = df_valid[df_valid['Kategorie'].isin(['Kritické', 'Varovné'])].to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
        st.download_button("📥 Stáhnout pouze chybová místa do CSV", csv_export, "Anomalie.csv", "text/csv")
else:
    st.info("👋 Nahrajte CSV z válce. Skript automaticky odfiltruje stání stroje a najde měkká místa.")