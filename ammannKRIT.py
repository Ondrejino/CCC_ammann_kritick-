import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Plošná analýza", layout="wide")
st.title("CCC Detektor: Historie a rozrušení vrstvy")
st.caption("Chytrá mřížková analýza detekující finální stav i propady tuhosti během hutnění.")

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
        
        st.header("3. Očištění dat (Filtry)")
        offset_m = st.number_input("Posun anténa -> běhoun (m)", value=2.0, step=0.1)
        forward_val = st.text_input("Hodnota jízdy VPŘED", value="1")
        min_speed_kmh = st.slider("Minimální rychlost (km/h)", 0.0, 5.0, 1.0, 0.5)
        
        st.header("4. Analýza chování (Kritéria)")
        lim_critical = st.number_input("Kritická hodnota FINÁLNÍ (Méně než:)", value=15.0, step=1.0)
        # NOVÝ SLIDER PRO DETEKCI DEKOMPAKCE
        max_drop = st.slider("Tolerovaný propad tuhosti (Kb)", 1.0, 20.0, 8.0, 1.0, help="Pokud tuhost mezi pojezdy spadne o více než tuto hodnotu, místo bude oranžové.")
        
        st.header("5. Nastavení Mřížky")
        grid_size_m = st.slider("Velikost buňky sítě (m)", 0.5, 5.0, 2.0, 0.5)
        pixel_size = st.slider("Vizuální velikost čtverců", 5, 30, 15)
        
        show_red = st.checkbox("🔴 Nevyhovující finální stav", value=True)
        show_orange = st.checkbox("🟠 Problém v průběhu (Rozrušení)", value=True)
        show_green = st.checkbox("🟢 Bezproblémové hutnění", value=True)

# --- 4. HLAVNÍ VÝPOČETNÍ LOGIKA ---
if uploaded_file is not None:
    df = df_raw.copy()
    
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
        
        if col_speed == "Vypočítat z GPS (Záložní)":
            _, _, dist_step = geod.inv(df['corr_lon'].shift(), df['corr_lat'].shift(), df['corr_lon'], df['corr_lat'])
            time_step = df['parsed_time'].diff().dt.total_seconds()
            df['speed_kmh'] = (dist_step / time_step) * 3.6
            df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1).mean().bfill() 
        
        df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
        
        if not df_valid.empty:
            avg_lat = df_valid['corr_lat'].mean()
            lat_step = grid_size_m / 111320
            lon_step = grid_size_m / (111320 * np.cos(np.radians(avg_lat)))
            
            df_valid['lat_bin'] = (df_valid['corr_lat'] // lat_step) * lat_step + (lat_step / 2)
            df_valid['lon_bin'] = (df_valid['corr_lon'] // lon_step) * lon_step + (lon_step / 2)
            
            # --- ZDE JE TA CHYTRÁ MAGIE HISTORIE BUŇKY ---
            # 1. Seřadíme absolutně vše podle času
            df_valid = df_valid.sort_values(by=['lat_bin', 'lon_bin', 'parsed_time'])
            
            # 2. Spočítáme rozdíl Kb oproti předchozímu záznamu ve STEJNÉM čtverci
            df_valid['Kb_diff'] = df_valid.groupby(['lat_bin', 'lon_bin'])[col_stiff].diff()
            
            # 3. Agregujeme historii pro každý čtverec
            df_grid = df_valid.groupby(['lat_bin', 'lon_bin']).agg(
                Final_Kb=(col_stiff, 'last'), # Poslední naměřená hodnota
                Max_Propad=(col_stiff, lambda x: x.diff().min() if len(x)>1 else 0), # Největší pokles
                Pocet_Bodu=(col_stiff, 'count')
            ).reset_index()
            
            # 4. Klasifikace (Prioritu má červená - finální průser)
            conditions = [
                (df_grid['Final_Kb'] < lim_critical), 
                (df_grid['Max_Propad'] <= -max_drop) & (df_grid['Final_Kb'] >= lim_critical),
                (df_grid['Final_Kb'] >= lim_critical)
            ]
            choices = ['Nezhutněno (Finální stav)', 'Rozrušení (Propad v průběhu)', 'Stabilní / OK']
            df_grid['Kategorie'] = np.select(conditions, choices, default='Neznámé')
            
            cos_correction = 1 / np.cos(np.radians(avg_lat))
            
            # --- VYKRESLENÍ MAPY ---
            st.subheader(f"Analýza historie zhutňování ({grid_size_m}x{grid_size_m} m)")
            fig = go.Figure()
            
            colors = {'Nezhutněno (Finální stav)': '#EF553B', 'Rozrušení (Propad v průběhu)': '#FFA15A', 'Stabilní / OK': '#00CC96'}
            
            for cat, color in colors.items():
                if (cat == 'Nezhutněno (Finální stav)' and not show_red) or \
                   (cat == 'Rozrušení (Propad v průběhu)' and not show_orange) or \
                   (cat == 'Stabilní / OK' and not show_green):
                    continue
                    
                df_cat = df_grid[df_grid['Kategorie'] == cat]
                
                if not df_cat.empty:
                    # Různé texty do hoveru podle toho, co je za problém
                    if cat == 'Rozrušení (Propad v průběhu)':
                        hover_text = "Finální Kb: " + df_cat['Final_Kb'].round(1).astype(str) + "<br>⚠️ Max propad v historii: " + df_cat['Max_Propad'].round(1).astype(str) + " Kb"
                    else:
                        hover_text = "Finální Kb: " + df_cat['Final_Kb'].round(1).astype(str) + "<br>Běžných bodů: " + df_cat['Pocet_Bodu'].astype(str)

                    fig.add_trace(go.Scatter(
                        x=df_cat['lon_bin'], y=df_cat['lat_bin'], mode='markers',
                        marker=dict(symbol='square', size=pixel_size, color=color, opacity=0.8),
                        name=f'{cat}', hovertext=hover_text
                    ))

            fig.update_layout(
                yaxis=dict(scaleanchor="x", scaleratio=cos_correction),
                height=800, dragmode='pan', legend=dict(title="Analýza buněk:")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- STATISTIKA ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Finálně nezhutněno", len(df_grid[df_grid['Kategorie'] == 'Nezhutněno (Finální stav)']))
            col2.metric("Podezřelé (Propady)", len(df_grid[df_grid['Kategorie'] == 'Rozrušení (Propad v průběhu)']))
            col3.metric("Dobře hutněno", len(df_grid[df_grid['Kategorie'] == 'Stabilní / OK']))
            
        else:
            st.warning("Po odfiltrování nezbyla žádná data.")
else:
    st.info("👋 Nahrajte CSV. Skript zanalyzuje historii zhutňování každé části pláně.")
