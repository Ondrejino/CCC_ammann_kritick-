import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io

# --- 1. NASTAVENÍ APLIKACE ---
st.set_page_config(page_title="CCC Surová data & Trend", layout="wide")
st.title("CCC Detektor: Bodová analýza a Trend pojezdů")
st.caption("Práce s čistými surovými daty (bez mřížkování) a detekce efektivity pojezdů.")

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
        
        st.header("3. Očištění dat a Geometrie")
        offset_m = st.number_input("Posun anténa -> běhoun (m)", value=2.0, step=0.1)
        forward_val = st.text_input("Hodnota jízdy VPŘED", value="1")
        min_speed_kmh = st.slider("Minimální rychlost (km/h)", 0.0, 5.0, 1.0, 0.5, help="Odstraní stání a prokluzy.")
        
        st.header("4. Vizuální nastavení")
        colormap = st.selectbox("Paleta Heatmapy", ['Turbo', 'Viridis', 'Plasma', 'Inferno', 'Jet'], index=0)
        point_opacity = st.slider("Průhlednost bodů", 0.1, 1.0, 0.6, 0.1, help="Nižší hodnota pomůže odhalit překrývání pojezdů.")
        decimation = st.slider("Decimace Heatmapy", 1, 20, 2, 1)

        st.header("5. Definice 'Kritických' bodů")
        st.markdown("Zadej pásmo hodnot, které považuješ za kritické (např. nedosažení cíle nebo naopak extrémní odskoky).")
        crit_min = st.number_input("Kritické OD (Kb):", value=1.0, step=1.0)
        crit_max = st.number_input("Kritické DO (Kb):", value=20.0, step=1.0)

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
    
    if len(df) < 5:
        st.error("Málo platných dat.")
    else:
        geod = Geod(ellps="WGS84")
        
        # --- VYHLAZENÍ A OFFSET ---
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
        
        # --- RYCHLOST ---
        if col_speed == "Vypočítat z GPS (Záložní)":
            _, _, dist_step = geod.inv(df['corr_lon'].shift(), df['corr_lat'].shift(), df['corr_lon'], df['corr_lat'])
            time_step = df['parsed_time'].diff().dt.total_seconds()
            df['speed_kmh'] = (dist_step / time_step) * 3.6
            df['speed_kmh'] = df['speed_kmh'].rolling(3, min_periods=1).mean().bfill() 
        
        # Filtrace stání
        df_valid = df[df['speed_kmh'] >= min_speed_kmh].copy()
        
        if not df_valid.empty:
            avg_lat = df_valid['corr_lat'].mean()
            cos_correction = 1 / np.cos(np.radians(avg_lat))
            
            # --- DETEKCE POJEZDŮ (PRO GLOBÁLNÍ TREND) ---
            # Pojezd definujeme jako souvislou jízdu. Změna nastane při otočení směru nebo delší pauze (např. nad 30s)
            time_gap_cond = df_valid['parsed_time'].diff().dt.total_seconds() > 30
            dir_cond = df_valid[col_dir] != df_valid[col_dir].shift().bfill()
            df_valid['pass_id'] = (time_gap_cond | dir_cond).cumsum() + 1
            
            tab1, tab2 = st.tabs(["🗺️ Mapy", "📈 Analýza 'špatného nárůstu' (Trendy)"])
            
            with tab1:
                # --- MAPA 1: SUROVÁ HEATMAPA ---
                st.subheader(f"Surová bodová Heatmapa (Paleta: {colormap})")
                fig_raw = go.Figure()
                
                df_plot = df_valid.iloc[::decimation]
                
                fig_raw.add_trace(go.Scatter(
                    x=df_plot['corr_lon'], y=df_plot['corr_lat'], mode='markers',
                    marker=dict(
                        size=6, 
                        color=df_plot[col_stiff], 
                        colorscale=colormap, 
                        showscale=True, 
                        opacity=point_opacity,
                        colorbar=dict(title="Kb [-]")
                    ),
                    name="Naměřená tuhost",
                    hovertext=df_plot[col_stiff].round(1).astype(str) + ' Kb'
                ))
                
                fig_raw.update_layout(
                    yaxis=dict(scaleanchor="x", scaleratio=cos_correction),
                    height=600, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_raw, use_container_width=True)
                
                # --- MAPA 2: KRITICKÉ BODY ---
                st.subheader(f"Izolované Kritické body ({crit_min} až {crit_max} Kb)")
                st.caption("Šedé pozadí ukazuje celkovou trasu pro kontext. Červeně svítí jen hodnoty ve zvoleném pásmu.")
                fig_crit = go.Figure()
                
                # Obarvení na šedo pro kontext
                fig_crit.add_trace(go.Scatter(
                    x=df_plot['corr_lon'], y=df_plot['corr_lat'], mode='markers',
                    marker=dict(size=4, color='#E5E7EB', opacity=0.3),
                    name="Ostatní data", hoverinfo='none'
                ))
                
                # Filtrace kritických bodů
                df_crit = df_valid[(df_valid[col_stiff] >= crit_min) & (df_valid[col_stiff] <= crit_max)]
                
                if not df_crit.empty:
                    fig_crit.add_trace(go.Scatter(
                        x=df_crit['corr_lon'], y=df_crit['corr_lat'], mode='markers',
                        marker=dict(
                            size=7, 
                            color='#EF553B', # Výrazná červená
                            line=dict(width=1, color='black')
                        ),
                        name="Kritické body",
                        hovertext="Kb: " + df_crit[col_stiff].round(1).astype(str)
                    ))
                
                fig_crit.update_layout(
                    yaxis=dict(scaleanchor="x", scaleratio=cos_correction),
                    height=600, dragmode='pan', margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_crit, use_container_width=True)

            with tab2:
                # --- GRAF: GLOBÁLNÍ TREND POJEZDŮ ---
                st.subheader("Analýza efektivity hutnění (Nárůst tuhosti za celý úsek)")
                st.markdown("""
                Tento graf ukazuje průměrnou a mediánovou hodnotu $K_b$ pro každý souvislý pojezd válce. 
                Sledujte **přírůstek (Deltu)**. Pokud se křivka oploští nebo začne klesat (Delta je blízko nule nebo negativní), 
                zemina už do sebe další práci nepojme a dochází k přehutňování (tzv. odraz bubnu).
                """)
                
                # Výpočet průměrů za pojezd
                trend_df = df_valid.groupby('pass_id').agg(
                    Průměr_Kb=(col_stiff, 'mean'),
                    Medián_Kb=(col_stiff, 'median'),
                    Počet_bodů=(col_stiff, 'count')
                ).reset_index()
                
                # Odfiltrování "mikro pojezdů" (krátké popojetí)
                trend_df = trend_df[trend_df['Počet_bodů'] > 20].copy()
                trend_df['Pojezd_číslo'] = range(1, len(trend_df) + 1) # Přečíslování
                
                # Výpočet nárůstu (Delty)
                trend_df['Nárůst_oproti_minule'] = trend_df['Průměr_Kb'].diff().fillna(0)
                
                # Vykreslení
                fig_trend = go.Figure()
                
                # Křivka průměru
                fig_trend.add_trace(go.Scatter(
                    x=trend_df['Pojezd_číslo'], y=trend_df['Průměr_Kb'],
                    mode='lines+markers+text',
                    text=trend_df['Průměr_Kb'].round(1),
                    textposition="top center",
                    marker=dict(size=12, color='#19D3F3'), line=dict(width=4),
                    name='Průměrné Kb'
                ))
                
                # Sloupcový graf pro Deltu (Nárůst)
                bar_colors = ['#00CC96' if val >= 0 else '#EF553B' for val in trend_df['Nárůst_oproti_minule']]
                fig_trend.add_trace(go.Bar(
                    x=trend_df['Pojezd_číslo'], y=trend_df['Nárůst_oproti_minule'],
                    marker_color=bar_colors, opacity=0.6,
                    name='Přírůstek Kb', yaxis='y2'
                ))
                
                fig_trend.update_layout(
                    xaxis=dict(title="Pořadí pojezdu", tickmode='linear'),
                    yaxis=dict(title="Hodnota Kb [-]"),
                    yaxis2=dict(title="Přírůstek (Delta)", overlaying='y', side='right', showgrid=False),
                    height=500, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
                st.dataframe(trend_df.drop(columns=['pass_id']).round(2), hide_index=True)
        else:
            st.warning("Po odfiltrování stání a nulové rychlosti nezbyla žádná platná data.")
else:
    st.info("👋 Nahrajte CSV.")
