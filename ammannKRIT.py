import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Transformer
import io
import csv

# ==========================================
# 1. NASTAVENÍ APLIKACE A UI
# ==========================================
st.set_page_config(page_title="CCC Analýza Zhutnění", layout="wide")
st.title("🚜 CCC Detektor: Pokročilá plošná analýza zhutnění")
st.caption("Nástroj optimalizovaný pro zpracování rozsáhlých dat ze stavebních válců. Využívá metrickou AEQD projekci, plnou vektorizaci výpočtů a maticové vykreslování pro plynulý běh v prohlížeči.")

# ==========================================
# 2. ROBUSTNÍ DATA PARSER (ČTENÍ ZMATENÝCH CSV)
# ==========================================
@st.cache_data(show_spinner="Analyzuji strukturu CSV a načítám data...")
def nacti_surova_data(file_bytes):
    """
    Funkce odolná proti nepořádku v hlavičkách (různí výrobci strojů).
    Prohledává soubor řádek po řádku, dokud nenajde skutečná GPS/časová data.
    """
    # Načteme vzorek pro analýzu (prvních 50 000 bytů)
    sample_text = file_bytes[:50000].decode("utf-8", errors="ignore")
    lines = sample_text.splitlines()
    
    header_idx = 0
    # Hledáme klíčová slova určující hlavní datovou hlavičku (přeskakujeme metadata)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if "latitude" in line_lower or "lat" in line_lower or "time" in line_lower or "gps:" in line_lower:
            header_idx = i
            break
            
    header_line = lines[header_idx]
    
    # Detekce separátoru (někdy čárky, někdy středníky)
    try:
        sep = csv.Sniffer().sniff(header_line).delimiter
    except:
        sep = ';' if header_line.count(';') > header_line.count(',') else ','
    
    # Načteme data od nalezené hlavičky. Typ 'str' brání chybám s desetinnými čárkami.
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, skiprows=header_idx, on_bad_lines='skip', dtype=str, low_memory=False)
    
    # Vyčištění názvů sloupců
    df.columns = df.columns.astype(str).str.strip().str.replace('"', '').str.replace("'", "")
    return df

def najdi_vychozi_sloupec(columns, klicova_slova):
    """Pomocná funkce pro automatické přiřazení sloupců v UI."""
    for col in columns:
        col_lower = str(col).lower()
        for slovo in klicova_slova:
            if slovo in col_lower:
                return col
    return columns[0] if len(columns) > 0 else None

# ==========================================
# 3. GEOPROSTOROVÉ JÁDRO A VEKTORIZACE
# ==========================================
@st.cache_data(show_spinner="Přepočítávám kinematiku a metrickou mřížku...")
def zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, offset_right, min_speed_kmh, grid_size):
    df = df_raw.copy()
    
    # 1. Převod na čísla (řeší desetinné čárky z evropských Windows)
    for col in [col_lat, col_lon, col_stiff, col_vib]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
    # 2. Bezpečné zpracování času
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.replace(' GMT', ''), utc=True, format='mixed', errors='coerce')
    df = df.dropna(subset=[col_lat, col_lon, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    if len(df) < 2:
        return pd.DataFrame()

    # 3. MATEMATIKA: LOKÁLNÍ METRICKÁ PROJEKCE (Azimuthal Equidistant - AEQD)
    # Proč AEQD? Vytvoří exaktní rovnou plochu vycentrovanou na střed stavby. 1 jednotka = přesně 1 metr.
    # Tím se zcela vyhneme deformacím WGS84 (na rovníku vs. na pólu) a výpočetně náročné geodézii.
    mean_lat, mean_lon = df[col_lat].mean(), df[col_lon].mean()
    proj_string = f"+proj=aeqd +lat_0={mean_lat} +lon_0={mean_lon} +datum=WGS84 +units=m"
    transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
    
    df['x_m'], df['y_m'] = transformer.transform(df[col_lon].values, df[col_lat].values)
    
    # 4. KINEMATIKA: Vektorizovaný výpočet (Žádné FOR cykly!)
    df['dx'] = df['x_m'].diff().bfill()
    df['dy'] = df['y_m'].diff().bfill()
    
    # Heading v radiánech (Azimut). V matematice: 0 rad = Východ.
    df['heading_rad'] = np.arctan2(df['dy'], df['dx'])
    
    # Rychlost
    if col_speed != "Vypočítat z GPS (Záložní)":
        df['speed_kmh'] = pd.to_numeric(df[col_speed].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df['dt'] = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.01).bfill()
        df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['speed_kmh'] = (df['dist'] / df['dt']) * 3.6
        df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1, center=True).mean()

    # 5. KOREKCE ANTÉNY VŮČI STŘEDU BĚHOUNU
    # Podélný posun (ve směru jízdy) a příčný posun (vpravo/vlevo)
    fwd_x = np.cos(df['heading_rad']) * offset_fwd
    fwd_y = np.sin(df['heading_rad']) * offset_fwd
    right_x = np.sin(df['heading_rad']) * offset_right
    right_y = -np.cos(df['heading_rad']) * offset_right
    
    df['drum_x'] = df['x_m'] + fwd_x + right_x
    df['drum_y'] = df['y_m'] + fwd_y + right_y

    # 6. RASTROVÁNÍ (Základ pro heatmapu)
    # Rozdělí plochu na čtverce o hraně grid_size (např. 1x1 metr).
    df['grid_x'] = (df['drum_x'] // grid_size) * grid_size + (grid_size / 2)
    df['grid_y'] = (df['drum_y'] // grid_size) * grid_size + (grid_size / 2)

    # 7. FILTRACE PRÁCE
    df['is_vibrating'] = df[col_vib].fillna(0) > 0.1
    df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
    
    if not df_valid.empty:
        # Identifikace pracovních tahů (pauza > 30s nebo změna směru)
        time_gap = df_valid['parsed_time'].diff().dt.total_seconds() > 30
        dir_cond = df_valid[col_dir] != df_valid[col_dir].shift().bfill() if col_dir in df_valid.columns else False
        df_valid['pass_id'] = (time_gap | dir_cond).cumsum() + 1
        
    return df_valid

# ==========================================
# 4. SIDEBAR - UŽIVATELSKÉ ROZHRANÍ
# ==========================================
with st.sidebar:
    st.header("📂 1. Vstupní data")
    uploaded_file = st.file_uploader("Nahrát CSV z válce", type=['csv'])
    
    if uploaded_file:
        df_raw = nacti_surova_data(uploaded_file.getvalue())
        
        st.header("⚙️ 2. Senzory a Sloupce")
        col_time = st.selectbox("Čas", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['time', 'cas'])))
        col_lat = st.selectbox("Latitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['lat', 'latitude'])))
        col_lon = st.selectbox("Longitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['lon', 'longitude'])))
        col_stiff = st.selectbox("Tuhost (Kb/CMV)", df_raw.columns, index=df_raw.columns.get_loc(najdi_vychozi_sloupec(df_raw.columns, ['stiffness', 'stiff', 'kb', 'cmv'])))
        
        dir_guess = najdi_vychozi_sloupec(df_raw.columns, ['direction', 'dir', 'smer'])
        col_dir = st.selectbox("Směr (Vpřed/Vzad)", [None] + list(df_raw.columns), index=list(df_raw.columns).index(dir_guess)+1 if dir_guess else 0)
        
        vib_guess = najdi_vychozi_sloupec(df_raw.columns, ['amplitude:amplitude', 'amp', 'freq', 'vib', 'frequency'])
        col_vib = st.selectbox("Vibrace (Amp/Freq)", df_raw.columns, index=df_raw.columns.get_loc(vib_guess) if vib_guess else 0)
        
        speed_guess = najdi_vychozi_sloupec(df_raw.columns, ['speed', 'vel', 'rychlost'])
        speed_options = ["Vypočítat z GPS (Záložní)"] + list(df_raw.columns)
        col_speed = st.selectbox("Rychlost", speed_options, index=speed_options.index(speed_guess) if speed_guess else 0)

        st.header("📐 3. Kalibrace stroje a mřížky")
        offset_fwd = st.number_input("Posun anténa -> běhoun podélně (m)", value=2.0, step=0.1)
        offset_right = st.number_input("Posun anténa -> běhoun příčně (m)", value=0.0, step=0.1, help="+ vpravo, - vlevo")
        grid_size = st.slider("Rozlišení mřížky (m)", 0.2, 2.0, 0.5, 0.1, help="Při milionech řádků nastav 1.0 pro vysoký výkon.")
        min_speed_kmh = st.number_input("Odříznout stání (km/h)", value=0.5, step=0.1)

        st.header("🎯 4. Technologické limity")
        target_min = st.number_input("Minimální požadované Kb:", value=25.0, step=1.0)
        target_max = st.number_input("Horní limit (přezhutnění) Kb:", value=50.0, step=1.0)
        colormap = st.selectbox("Paleta Heatmapy", ['Turbo', 'Viridis', 'Jet'], index=0)

# ==========================================
# 5. HLAVNÍ LOGIKA A VYKRESLOVÁNÍ
# ==========================================
if uploaded_file is not None:
    df_valid = zpracuj_geodata(df_raw, col_lat, col_lon, col_stiff, col_vib, col_time, col_speed, col_dir, offset_fwd, offset_right, min_speed_kmh, grid_size)
    
    if not df_valid.empty:
        max_pass = int(df_valid['pass_id'].max()) if 'pass_id' in df_valid.columns else 1
        st.markdown("### ⏱️ Stroj času: Přehrávač hutnění")
        selected_pass = st.slider("Zobrazit stav pojezdu do čísla:", min_value=1, max_value=max_pass, value=max_pass)
        
        # Datový subset podle slideru
        df_current = df_valid[df_valid['pass_id'] <= selected_pass]
        df_vib = df_current[df_current['is_vibrating'] == True].copy()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ 1. Bodová stopa (Raw)", 
            "🟩 2. Plošné zhutnění (Rastr)", 
            "🔴 3. Mapa Anomálií", 
            "📊 4. Histogram", 
            "🧊 5. Kontrola žehlení"
        ])
        
        # Zajištění poměru stran 1:1 pro všechny mapy (1 metr na ose X = 1 metr na ose Y)
        layout_map = dict(
            scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, visible=False
        )

        with tab1:
            st.subheader("Raw GPS Stopa (Aktivní vibrace)")
            st.caption("Používá WebGL technologii (Scattergl) pro zobrazení reálné dráhy statisíců bodů bez pádu prohlížeče.")
            fig_raw = go.Figure()
            if not df_vib.empty:
                fig_raw.add_trace(go.Scattergl(
                    x=df_vib['drum_x'], y=df_vib['drum_y'], mode='markers',
                    marker=dict(size=5, color=df_vib[col_stiff], colorscale=colormap, showscale=True, opacity=0.8, colorbar=dict(title="Kb [-]")),
                    hovertext="Kb: " + df_vib[col_stiff].round(1).astype(str) + " | Pojezd: " + df_vib['pass_id'].astype(str)
                ))
            fig_raw.update_layout(yaxis=layout_map, height=700, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_raw, use_container_width=True)

        with tab2:
            st.subheader("Finální plošné zhutnění (Maticová Heatmapa)")
            st.caption(f"Hodnoty přepočítány do spojité sítě {grid_size}x{grid_size} m. Extrémně šetří výkon počítače.")
            if not df_vib.empty:
                # Klíčová operace: Získáme pouze úplně poslední přejezd v každém čtverci mřížky
                df_final = df_vib.drop_duplicates(subset=['grid_y', 'grid_x'], keep='last')
                
                # Pivot table převede data na 2D Matici (Ideální vstup pro Plotly Heatmap)
                pivot_kb = df_final.pivot(index='grid_y', columns='grid_x', values=col_stiff)
                
                fig_heat = go.Figure(data=go.Heatmap(
                    z=pivot_kb.values, x=pivot_kb.columns, y=pivot_kb.index,
                    colorscale=colormap, hoverongaps=False, colorbar=dict(title="Finální Kb [-]")
                ))
                fig_heat.update_layout(yaxis=layout_map, height=700, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_heat, use_container_width=True)

        with tab3:
            st.subheader("Klasifikace plochy (Nedohutněno / Přezhutněno)")
            if not df_vib.empty:
                # Rozřazení do 3 kategorií (-1: Nedohutněno, 0: OK, 1: Přezhutněno)
                conditions = [
                    df_final[col_stiff] < target_min,
                    df_final[col_stiff] > target_max
                ]
                choices = [-1, 1]
                df_final['Kategorie_Num'] = np.select(conditions, choices, default=0)
                
                pivot_cat = df_final.pivot(index='grid_y', columns='grid_x', values='Kategorie_Num')
                
                # Custom colorscale: Red (-1), Gray (0), Blue (1)
                cat_scale = [[0.0, 'red'], [0.5, '#E5E7EB'], [1.0, 'blue']]
                
                fig_anom = go.Figure(data=go.Heatmap(
                    z=pivot_cat.values, x=pivot_cat.columns, y=pivot_cat.index,
                    colorscale=cat_scale, zmin=-1, zmax=1, showscale=False, hoverongaps=False,
                    hovertemplate="Status (1=Přezhutněno, 0=OK, -1=Nedohutněno): %{z}<extra></extra>"
                ))
                fig_anom.update_layout(yaxis=layout_map, height=700, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white')
                st.plotly_chart(fig_anom, use_container_width=True)

        with tab4:
            st.subheader("Statistika finální plochy (Počet buněk sítě)")
            if not df_vib.empty:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=df_final[col_stiff], nbinsx=60, marker_color='slategray', name='Plocha'))
                fig_hist.add_vrect(x0=target_min, x1=target_max, fillcolor="green", opacity=0.2, line_width=0, annotation_text="Cílové pásmo")
                fig_hist.update_layout(xaxis_title="Hodnota Kb [-]", yaxis_title="Rozloha (Počet buněk sítě)", height=500, bargap=0.05)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Výpočet procentuálního zastoupení
                total_cells = len(df_final)
                pct_under = (df_final['Kategorie_Num'] == -1).sum() / total_cells * 100
                pct_ok = (df_final['Kategorie_Num'] == 0).sum() / total_cells * 100
                pct_over = (df_final['Kategorie_Num'] == 1).sum() / total_cells * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Nedohutněno", f"{pct_under:.1f} %")
                c2.metric("🟢 V cílovém pásmu (OK)", f"{pct_ok:.1f} %")
                c3.metric("🔵 Tvrdá anomálie", f"{pct_over:.1f} %")

        with tab5:
            st.subheader("Analýza statického žehlení")
            mode = st.radio("Technologická fáze:", ["Finální uzavření (Poslední vrstva)", "Úvodní stabilizace podkladu"], horizontal=True)
            
            if not df_current.empty:
                if "Finální" in mode:
                    df_iron = df_current.drop_duplicates(subset=['grid_y', 'grid_x'], keep='last').copy()
                else:
                    df_iron = df_current.drop_duplicates(subset=['grid_y', 'grid_x'], keep='first').copy()
                
                # Žehlení je definováno jako pojezd bez vibrace
                df_iron['Splneno_Num'] = np.where(df_iron['is_vibrating'] == False, 1, 0)
                pivot_iron = df_iron.pivot(index='grid_y', columns='grid_x', values='Splneno_Num')
                
                iron_scale = [[0.0, 'rgba(239, 68, 68, 0.8)'], [1.0, 'rgba(34, 197, 94, 0.8)']]
                
                fig_iron = go.Figure(data=go.Heatmap(
                    z=pivot_iron.values, x=pivot_iron.columns, y=pivot_iron.index,
                    colorscale=iron_scale, showscale=False, hoverongaps=False,
                    hovertemplate="Žehleno (1=Ano, 0=Ne): %{z}<extra></extra>"
                ))
                fig_iron.update_layout(yaxis=layout_map, height=700, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_iron, use_container_width=True)
                
                uspelo = df_iron['Splneno_Num'].sum()
                celkem = len(df_iron)
                st.metric("Plocha úspěšně obsloužena staticky", f"{(uspelo / celkem * 100):.1f} %" if celkem > 0 else "0 %")

    else:
        st.error("Po aplikaci rychlostního filtru nezbyla žádná data.")
else:
    st.info("👋 Připraveno k analýze. Nahrajte z bočního panelu CSV soubor.")
