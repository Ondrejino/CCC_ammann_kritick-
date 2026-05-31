import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Transformer
import io
import csv

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Ultimátní Detektor v3 (CTU Edition)", layout="wide")
st.title("🚜 CCC Detektor: Pokročilá plošná analýza zhutnění")
st.caption("Akademický nástroj pro analýzu CCC dat (Optimalizováno pro velké datasety a plynulý WebGL render).")

# --- 2. ROBUSTNÍ NAČÍTÁNÍ A CACHOVÁNÍ ---
@st.cache_data(show_spinner="Analyzuji a parsuji CSV soubor...")
def nacti_surova_data(file_bytes):
    # Pokus o inteligentní detekci oddělovače (Sniffer)
    sample_text = file_bytes[:10000].decode("utf-8", errors="ignore")
    try:
        sep = csv.Sniffer().sniff(sample_text.splitlines()[0]).delimiter
    except:
        sep = ';' if sample_text.count(';') > sample_text.count(',') else ','
        
    # Hledání hlavičky (některé stroje mají nahoře meta-data)
    lines = sample_text.splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if "latitude" in line.lower() or "time" in line.lower() or "lon" in line.lower():
            header_idx = i
            break
            
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, skiprows=header_idx, on_bad_lines='skip', low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.replace('"', '').str.replace("'", "")
    return df

def najdi_vychozi_sloupec(columns, klicova_slova):
    for col in columns:
        for slovo in klicova_slova:
            if slovo in str(col).lower():
                return col
    return columns[0] if len(columns) > 0 else None

# --- 3. VEKTORIZOVANÉ GEOPROSTOROVÉ JÁDRO ---
@st.cache_data(show_spinner="Transformuji souřadnice a počítám metrickou síť...")
def zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, offset_right, min_speed_kmh, grid_size):
    df = df_raw.copy()
    
    # 1. Standardizace typů (odstranění desetinných čárek)
    for col in [col_lat, col_lon, col_stiff, col_vib]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
    # Bezpečné zpracování času (zvládne ISO, UTC i lokální stringy)
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.replace(' GMT', ''), utc=True, format='mixed', errors='coerce')
    
    # Filtrace neplatných řádků před těžkými výpočty
    df = df.dropna(subset=[col_lat, col_lon, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    if len(df) < 2:
        return pd.DataFrame()

    # 2. LOKÁLNÍ METRICKÁ PROJEKCE (Azimuthal Equidistant)
    # Vytvoříme matematicky přesnou rovnou plochu (v metrech) vycentrovanou přesně na střed stavby
    mean_lat, mean_lon = df[col_lat].mean(), df[col_lon].mean()
    proj_string = f"+proj=aeqd +lat_0={mean_lat} +lon_0={mean_lon} +datum=WGS84 +units=m"
    transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
    
    # Vektorizovaný převod Lat/Lon na metrické X/Y
    df['x_m'], df['y_m'] = transformer.transform(df[col_lon].values, df[col_lat].values)
    
    # 3. VEKTORIZOVANÁ KINEMATIKA (Směr a Rychlost)
    # Směrový vektor (rozdíl souřadnic)
    df['dx'] = df['x_m'].diff().bfill()
    df['dy'] = df['y_m'].diff().bfill()
    
    # Heading v radiánech (Azimut). arctan2 bere (Y, X), kde osa Y je Sever.
    df['heading_rad'] = np.arctan2(df['dx'], df['dy'])
    
    # Rychlost
    if col_speed != "Vypočítat z GPS":
        df['speed_kmh'] = pd.to_numeric(df[col_speed].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df['dt'] = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.01).bfill()
        df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['speed_kmh'] = (df['dist'] / df['dt']) * 3.6
        df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1, center=True).mean()

    # 4. KOREKCE POZICE BĚHOUNU (Offsety)
    # Převod offsetů antény vůči těžišti/běhounu stroje
    # fwd = podélně (ve směru jízdy), right = příčně (vpravo od osy)
    df['drum_x'] = df['x_m'] + (np.sin(df['heading_rad']) * offset_fwd) + (np.cos(df['heading_rad']) * offset_right)
    df['drum_y'] = df['y_m'] + (np.cos(df['heading_rad']) * offset_fwd) - (np.sin(df['heading_rad']) * offset_right)

    # 5. VYTVOŘENÍ RASTROVÉ MŘÍŽKY (Grid Binning)
    # Tímto přiřadíme každý záznam do přesného čtverce na ploše
    df['grid_x'] = (df['drum_x'] // grid_size) * grid_size + (grid_size / 2)
    df['grid_y'] = (df['drum_y'] // grid_size) * grid_size + (grid_size / 2)

    # 6. FILTRACE POJEZDŮ A VIBRACÍ
    df['is_vibrating'] = df[col_vib].fillna(0) > 0.1
    df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
    
    if not df_valid.empty:
        # Identifikace samostatných pracovních tahů (změna směru nebo časová pauza > 30s)
        time_gap = df_valid['parsed_time'].diff().dt.total_seconds() > 30
        dir_cond = df_valid[col_dir] != df_valid[col_dir].shift().bfill() if col_dir in df_valid.columns else False
        df_valid['pass_id'] = (time_gap | dir_cond).cumsum() + 1
        
    return df_valid

# --- 4. BOČNÍ PANEL (UI) ---
with st.sidebar:
    st.header("📂 1. Vstupní data")
    uploaded_file = st.file_uploader("Nahrát CSV z válce (CCC log)", type=['csv'])
    
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        df_raw = nacti_surova_data(file_bytes)
        
        st.header("⚙️ 2. Senzory a Sloupce")
        col_time = st.selectbox("Čas", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['time', 'cas'])))
        col_lat = st.selectbox("Latitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['lat'])))
        col_lon = st.selectbox("Longitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['lon'])))
        col_stiff = st.selectbox("Tuhost (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['stiff', 'kb', 'cmv'])))
        
        dir_guess = najdi_vychozi_sloupec(df_raw.columns, ['dir', 'smer'])
        col_dir = st.selectbox("Směr (Vpřed/Vzad)", [None] + list(df_raw.columns), index=list(df_raw.columns).index(dir_guess)+1 if dir_guess else 0)
        
        vib_guess = najdi_vychozi_sloupec(df_raw.columns, ['amp', 'freq', 'vib'])
        col_vib = st.selectbox("Vibrace (Amp/Freq)", df_raw.columns, index=df_raw.columns.get_loc(vib_guess) if vib_guess else 0)
        
        speed_guess = najdi_vychozi_sloupec(df_raw.columns, ['speed', 'vel', 'rychlost'])
        speed_options = ["Vypočítat z GPS"] + list(df_raw.columns)
        col_speed = st.selectbox("Rýchlost", speed_options, index=speed_options.index(speed_guess) if speed_guess else 0)

        st.header("📐 3. Kalibrace stroje a mřížky")
        offset_fwd = st.number_input("Posun anténa -> běhoun podélně (m)", value=2.0, step=0.1)
        offset_right = st.number_input("Posun anténa -> běhoun příčně (m)", value=0.0, step=0.1, help="+ vpravo, - vlevo")
        grid_size = st.slider("Rozlišení mřížky / Raster (m)", 0.2, 2.0, 0.5, 0.1, help="Menší hodnota = detailnější výpočet, ale hrubší mapa.")
        min_speed_kmh = st.number_input("Odříznout stání (km/h)", value=0.5, step=0.1)

        st.header("🎯 4. Technologické limity")
        target_min = st.number_input("Minimální požadované Kb:", value=25.0, step=1.0)
        target_max = st.number_input("Horní limit (přezhutnění) Kb:", value=50.0, step=1.0)
        colormap = st.selectbox("Paleta Heatmapy", ['Turbo', 'Viridis', 'Jet'], index=0)

# --- 5. HLAVNÍ VÝPOČTOVÁ ČÁST ---
if uploaded_file is not None:
    df_valid = zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, offset_right, min_speed_kmh, grid_size)
    
    if not df_valid.empty:
        max_pass = int(df_valid['pass_id'].max()) if 'pass_id' in df_valid.columns else 1
        
        st.markdown("### ⏱️ Interaktivní časová osa zhutnění")
        selected_pass = st.slider("Přehrát stavbu do pojezdu (vrstvy) č.:", min_value=1, max_value=max_pass, value=max_pass)
        
        # Omezení datasetu podle časové osy
        df_current = df_valid[df_valid['pass_id'] <= selected_pass]
        df_vib = df_current[df_current['is_vibrating'] == True].copy()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ 1. Pásy (Raw GPS)", 
            "🟩 2. Plošné zhutnění (Heatmapa)", 
            "🔴 3. Nedohutněná místa", 
            "📊 4. Statistika (Histogram)", 
            "🧊 5. Kontrola žehlení"
        ])
        
        # Společný formát os pro všechny grafy (Zajišťuje zobrazení 1:1 v metrech)
        axis_layout = dict(
            scaleanchor="x", scaleratio=1, # 1 Metr X = 1 Metr Y, absolutně klíčové pro správné proporce
            showgrid=False, zeroline=False, visible=False 
        )

        with tab1:
            st.subheader("Raw GPS Stopa (Aktivní vibrace)")
            st.caption("Slouží k ověření dráhy stroje. Použito WebGL (Scattergl) pro zvládnutí milionů bodů.")
            
            fig_raw = go.Figure()
            if not df_vib.empty:
                fig_raw.add_trace(go.Scattergl(
                    x=df_vib['drum_x'], y=df_vib['drum_y'], mode='markers',
                    marker=dict(size=4, color=df_vib[col_stiff], colorscale=colormap, showscale=True, colorbar=dict(title="Kb [-]")),
                    hovertext="Kb: " + df_vib[col_stiff].round(1).astype(str) + " | Pojezd: " + df_vib['pass_id'].astype(str)
                ))
            fig_raw.update_layout(yaxis=axis_layout, height=700, dragmode='pan', margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_raw, use_container_width=True)

        with tab2:
            st.subheader("Plošná mapa finálního zhutnění (Mřížka)")
            st.caption(f"Hodnoty agregovány do rastru {grid_size}x{grid_size} m. Zobrazuje se vždy poslední známá hodnota z každé buňky sítě.")
            
            if not df_vib.empty:
                # AGREGACE: Necháme si jen finální přejezd v každém čtverci mřížky
                df_final = df_vib.drop_duplicates(subset=['grid_y', 'grid_x'], keep='last')
                
                # Vytvoření matice pro Heatmapu
                pivot_kb = df_final.pivot(index='grid_y', columns='grid_x', values=col_stiff)
                
                fig_heat = go.Figure(data=go.Heatmap(
                    z=pivot_kb.values,
                    x=pivot_kb.columns,
                    y=pivot_kb.index,
                    colorscale=colormap,
                    hoverongaps=False,
                    colorbar=dict(title="Finální Kb [-]")
                ))
                fig_heat.update_layout(yaxis=axis_layout, height=700, dragmode='pan', margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_heat, use_container_width=True)

        with tab3:
            st.subheader("Klasifikace plochy: Anomálie a Cíle")
            
            if not df_vib.empty:
                # Použijeme stejná agregovaná data jako v Tab 2
                df_final['Kategorie'] = np.where(
                    df_final[col_stiff] < target_min, 'Nedohutněno (< ' + str(target_min) + ')',
                    np.where(df_final[col_stiff] > target_max, 'Přezhutněno (> ' + str(target_max) + ')', 'V normě (OK)')
                )
                
                color_map_discrete = {'Nedohutněno (< ' + str(target_min) + ')': 'red', 'V normě (OK)': '#E5E7EB', 'Přezhutněno (> ' + str(target_max) + ')': 'blue'}
                
                fig_anom = go.Figure()
                for kateg, barva in color_map_discrete.items():
                    df_sub = df_final[df_final['Kategorie'] == kateg]
                    if not df_sub.empty:
                        # Heatmapa pro diskrétní barvy v pravidelné mřížce
                        pivot_cat = df_sub.pivot(index='grid_y', columns='grid_x', values=col_stiff)
                        # Trik: Nastavíme barvu plně na požadovanou kategorii
                        custom_scale = [[0.0, barva], [1.0, barva]]
                        fig_anom.add_trace(go.Heatmap(
                            z=pivot_cat.values, x=pivot_cat.columns, y=pivot_cat.index,
                            colorscale=custom_scale, showscale=False, hoverongaps=False, name=kateg,
                            hovertemplate=kateg + "<br>Kb: %{z:.1f}<extra></extra>"
                        ))
                        
                fig_anom.update_layout(yaxis=axis_layout, height=700, dragmode='pan', margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_anom, use_container_width=True)

        with tab4:
            st.subheader("Statistika z finální plochy")
            if not df_vib.empty:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=df_final[col_stiff], nbinsx=60, marker_color='slategray', name='Plocha'))
                fig_hist.add_vrect(x0=target_min, x1=target_max, fillcolor="green", opacity=0.2, line_width=0, annotation_text="Pásmo přijatelnosti")
                fig_hist.update_layout(xaxis_title="Hodnota Kb [-]", yaxis_title="Rozloha (Počet buněk mřížky)", height=500, bargap=0.05)
                st.plotly_chart(fig_hist, use_container_width=True)

        with tab5:
            st.subheader("Technologická analýza: Statické žehlení")
            mode = st.radio("Sledovaná fáze:", ["Finální přežehlení vrstvy", "Úvodní statický přejezd podkladu"], horizontal=True)
            
            if not df_current.empty:
                # Rozhodovací logika agregace - bereme první NEBO poslední záznam na čtverci sítě
                if "Finální" in mode:
                    df_iron = df_current.drop_duplicates(subset=['grid_y', 'grid_x'], keep='last').copy()
                    df_iron['Splneno'] = df_iron['is_vibrating'] == False
                else:
                    df_iron = df_current.drop_duplicates(subset=['grid_y', 'grid_x'], keep='first').copy()
                    df_iron['Splneno'] = df_iron['is_vibrating'] == False

                pivot_iron = df_iron.pivot(index='grid_y', columns='grid_x', values='Splneno').astype(float)
                
                # Zelená (1.0) = Splněno (Bylo staticky zataženo), Červená (0.0) = Nesplněno (Zůstalo po vibraci)
                iron_scale = [[0.0, 'rgba(239, 68, 68, 0.8)'], [1.0, 'rgba(34, 197, 94, 0.8)']]
                
                fig_iron = go.Figure(data=go.Heatmap(
                    z=pivot_iron.values, x=pivot_iron.columns, y=pivot_iron.index,
                    colorscale=iron_scale, showscale=False, hoverongaps=False,
                    hovertemplate="Status: %{z}<extra></extra>"
                ))
                fig_iron.update_layout(yaxis=axis_layout, height=700, dragmode='pan', margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='#f3f4f6')
                st.plotly_chart(fig_iron, use_container_width=True)
                
                uspelo = df_iron['Splneno'].sum()
                celkem = len(df_iron)
                st.metric("Plocha obsloužená staticky ve vybrané fázi", f"{(uspelo / celkem * 100):.1f} %" if celkem > 0 else "0 %")

    else:
        st.warning("Při zvolených parametrech filtrace (rychlost, sloupce) nezbyla žádná data k analýze.")
else:
    st.info("👋 Připraveno. Vložte CSV soubor vygenerovaný měřícím válcem v bočním panelu.")
