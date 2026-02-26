import os
import streamlit as st
from googleads import ad_manager
from datetime import datetime, date, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import requests
import time
import gzip

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Dashboard GAM Performance", page_icon="📊", layout="wide")

# --- INITIALISATION CLIENT API ---
@st.cache_resource(show_spinner=False)
def get_ad_manager_client():
    """Charge le client API depuis les fichiers locaux ou recrée les fichiers via les secrets Streamlit."""
    try:
        # --- NOUVEAUTÉ : GESTION DU FICHIER JSON ---
        # Si le fichier JSON n'existe pas en local mais qu'il est dans les secrets, on le crée virtuellement
        if not os.path.exists('service_account.json') and "GCP_SERVICE_ACCOUNT" in st.secrets:
            with open('service_account.json', 'w', encoding='utf-8') as f:
                f.write(st.secrets["GCP_SERVICE_ACCOUNT"])

        # 1. Mode LOCAL (sur ton PC)
        if os.path.exists('googleads.yaml'):
            return ad_manager.AdManagerClient.LoadFromStorage('googleads.yaml')
        
        # 2. Mode CLOUD (sur Streamlit)
        elif "GOOGLEADS_YAML" in st.secrets:
            yaml_string = st.secrets["GOOGLEADS_YAML"]
            return ad_manager.AdManagerClient.LoadFromString(yaml_string)
            
        else:
            st.error("❌ Configuration manquante : ni fichier 'googleads.yaml' local, ni secret 'GOOGLEADS_YAML' trouvé.")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation de l'API : {e}")
        st.stop()

# --- FONCTIONS API ---

@st.cache_data(show_spinner=False)
def fetch_advertisers():
    client = get_ad_manager_client()
    service = client.GetService('CompanyService', version='v202602')
    target_names = ["GSO_TELE SECOURS", "GSO_STUDYRAMA", "GSO_VENTA PEIO SL", "GSO_LEROY MERLIN FRANCE"]
    names_query = ", ".join([f"'{n}'" for n in target_names])
    statement = ad_manager.StatementBuilder(version='v202602').Where(f"name IN ({names_query}) AND type = 'ADVERTISER'")
    
    try:
        response = service.getCompaniesByStatement(statement.ToStatement())
        if 'results' in response:
            return sorted([{"id": c['id'], "name": c['name']} for c in response['results']], key=lambda x: x['name'])
        return []
    except Exception as e:
        st.error(f"Erreur lors de la récupération des annonceurs : {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_all_orders_for_adv(adv_id):
    client = get_ad_manager_client()
    service = client.GetService('OrderService', version='v202602')
    # Historique de 2 ans pour les filtres
    start_history = (date.today() - timedelta(days=730)).strftime('%Y-%m-%d')
    statement = ad_manager.StatementBuilder(version='v202602').Where(f"advertiserId = {adv_id} AND startDateTime >= '{start_history}T00:00:00'")
    
    response = service.getOrdersByStatement(statement.ToStatement())
    orders = []
    if 'results' in response:
        for o in response['results']:
            sdt = o['startDateTime']['date']
            orders.append({
                "id": o['id'], 
                "name": o['name'], 
                "year": sdt['year'], 
                "month": sdt['month'],
                "start_str": f"{sdt['day']:02d}/{sdt['month']:02d}/{sdt['year']}" # Formatage propre JJ/MM/AAAA
            })
    return orders

@st.cache_data(show_spinner=False)
def fetch_order_goal(order_id):
    client = get_ad_manager_client()
    li_service = client.GetService('LineItemService', version='v202602')
    statement = ad_manager.StatementBuilder(version='v202602').Where(f"orderId = {order_id}")
    
    response = li_service.getLineItemsByStatement(statement.ToStatement())
    total = 0
    if 'results' in response:
        for li in response['results']:
            try: 
                total += li['primaryGoal']['units']
            except KeyError: 
                pass
    return total

def fetch_report_stats(order_id):
    client = get_ad_manager_client()
    report_service = client.GetService('ReportService', version='v202602')
    
    today_dt = date.today()
    three_years_ago = today_dt - timedelta(days=1094) # Limite max de Google
    
    report_job = {'reportQuery': {
        'dimensions': ['DATE', 'DEVICE_CATEGORY_NAME', 'CREATIVE_NAME'],
        'columns': ['AD_SERVER_IMPRESSIONS', 'AD_SERVER_CLICKS', 'AD_SERVER_CTR'],
        'dateRangeType': 'CUSTOM_DATE',
        'startDate': {'year': three_years_ago.year, 'month': three_years_ago.month, 'day': three_years_ago.day},
        'endDate': {'year': today_dt.year, 'month': today_dt.month, 'day': today_dt.day},
        'statement': {'query': f'WHERE ORDER_ID = {order_id}'}
    }}
    
    try:
        job = report_service.runReportJob(report_job)
        job_id = job['id']

        # Boucle d'attente simplifiée avec un timeout de sécurité (ex: 60 sec)
        timeout = 60
        while timeout > 0:
            status = report_service.getReportJobStatus(job_id)
            if status == 'COMPLETED':
                break
            elif status == 'FAILED':
                st.error("Le rapport a échoué chez Google.")
                return pd.DataFrame()
            time.sleep(2)
            timeout -= 2
            
        if timeout <= 0:
            st.error("Délai d'attente dépassé pour la génération du rapport.")
            return pd.DataFrame()

        url = report_service.getReportDownloadUrlWithOptions(job_id, {'exportFormat': 'CSV_DUMP'})
        r = requests.get(url)
        
        # Décompression propre
        try:
            content = gzip.decompress(r.content).decode('utf-8')
        except OSError: # OSError est levée si ce n'est pas un vrai gzip
            content = r.content.decode('utf-8')
        
        df = pd.read_csv(io.StringIO(content))
        if df.empty: 
            return df
        
        # Nettoyage rigoureux des colonnes
        df.columns = [c.replace('Dimension.', '').replace('Column.', '').upper().strip() for c in df.columns]
        
        # Suppression des lignes de totalisation
        if 'DATE' in df.columns:
            df = df.dropna(subset=['DATE'])
            
        return df
    except Exception as e:
        st.error(f"Erreur Reporting : {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_creatives(order_id):
    client = get_ad_manager_client()
    lica_service = client.GetService('LineItemCreativeAssociationService', version='v202602')
    creative_service = client.GetService('CreativeService', version='v202602')
    
    lica_resp = lica_service.getLineItemCreativeAssociationsByStatement(
        ad_manager.StatementBuilder(version='v202602').Where(f"orderId = {order_id}").ToStatement())
    
    if 'results' not in lica_resp: 
        return []
        
    c_ids = ",".join([str(l['creativeId']) for l in lica_resp['results']])
    c_resp = creative_service.getCreativesByStatement(
        ad_manager.StatementBuilder(version='v202602').Where(f"id IN ({c_ids})").ToStatement())
    
    data = []
    if 'results' in c_resp:
        for c in c_resp['results']:
            img = None
            if c.__class__.__name__ in ['ImageCreative', 'ImageRedirectCreative']:
                try: 
                    img = c['primaryImageAsset']['assetUrl']
                except (KeyError, AttributeError): 
                    pass
            
            # --- CORRECTION ICI ---
            # On extrait l'URL de preview de manière sécurisée pour les objets Zeep
            preview_url = None
            try:
                preview_url = c['previewUrl']
            except (KeyError, AttributeError):
                pass
                
            data.append({
                "name": c['name'], 
                "image": img, 
                "preview": preview_url
            })
    return data

# --- INTERFACE UTILISATEUR ---

with st.sidebar:
    st.header("⚙️ Sélection Campagne")
    advs = fetch_advertisers()
    adv_map = {a['name']: a['id'] for a in advs}
    sel_adv_name = st.selectbox("1. Annonceur", options=[""] + list(adv_map.keys()))

    sel_order_id = None
    sel_order_name = ""
    
    if sel_adv_name:
        all_orders = fetch_all_orders_for_adv(adv_map[sel_adv_name])
        
        if all_orders:
            years = sorted(list(set([o['year'] for o in all_orders])), reverse=True)
            sel_year = st.selectbox("2. Année de début", options=years)
            
            months_in_year = sorted(list(set([o['month'] for o in all_orders if o['year'] == sel_year])))
            month_names = {1:"Janvier", 2:"Février", 3:"Mars", 4:"Avril", 5:"Mai", 6:"Juin", 
                           7:"Juillet", 8:"Août", 9:"Septembre", 10:"Octobre", 11:"Novembre", 12:"Décembre"}
            
            sel_month = st.selectbox("3. Mois de début", options=months_in_year, format_func=lambda x: month_names[x])
            
            final_options = [o for o in all_orders if o['year'] == sel_year and o['month'] == sel_month]
            order_labels = {o['name']: o['id'] for o in final_options}
            
            st.write("---")
            sel_order_name = st.selectbox(f"4. Campagne ({len(final_options)})", options=list(order_labels.keys()))
            sel_order_id = order_labels.get(sel_order_name)
        else:
            st.warning("Aucune campagne pour cet annonceur.")

# --- ZONE PRINCIPALE ---
if sel_adv_name and sel_order_id:
    if st.button("🚀 Analyser la campagne", type="primary", use_container_width=True):
        with st.spinner("Récupération des données depuis Google Ad Manager..."):
            df = fetch_report_stats(sel_order_id)
            goal = fetch_order_goal(sel_order_id)
            creats = fetch_creatives(sel_order_id)

        if not df.empty:
            st.subheader(f"📊 Performances : {sel_order_name}")
            
            # Détection dynamique des colonnes (sécurité)
            col_imp = next((c for c in df.columns if 'IMPRESSIONS' in c), None)
            col_clk = next((c for c in df.columns if 'CLICKS' in c), None)
            
            if col_imp and col_clk:
                ti, tc = df[col_imp].sum(), df[col_clk].sum()
                ctr = (tc/ti*100) if ti > 0 else 0
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Objectif total (Units)", f"{goal:,}")
                m2.metric("Impressions délivrées", f"{ti:,}")
                m3.metric("Clics", f"{tc:,}")
                m4.metric("CTR", f"{ctr:.2f}%")
                
                # --- ÉVOLUTION TEMPORELLE ---
                st.divider()
                st.write("### 📈 Analyse de la diffusion et de l'engagement")
                df['DateObj'] = pd.to_datetime(df['DATE'])
                df_evo = df.groupby('DateObj')[[col_imp, col_clk]].sum().reset_index().sort_values('DateObj')
                df_evo['DateStr'] = df_evo['DateObj'].dt.strftime('%d %b')

                fig_evo = make_subplots(specs=[[{"secondary_y": True}]])
                fig_evo.add_trace(go.Scatter(x=df_evo['DateStr'], y=df_evo[col_imp], name="Impressions",
                                             fill='tozeroy', mode='lines', line=dict(width=0.5, color='#1f77b4'),
                                             fillcolor='rgba(31, 119, 180, 0.2)'), secondary_y=False)
                fig_evo.add_trace(go.Bar(x=df_evo['DateStr'], y=df_evo[col_clk], name="Clics",
                                         marker_color='#ff7f0e', width=0.4), secondary_y=True)

                fig_evo.update_layout(hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=350)
                fig_evo.update_xaxes(showgrid=False)
                fig_evo.update_yaxes(title_text="Volume Impressions", secondary_y=False, showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                fig_evo.update_yaxes(title_text="Volume Clics", secondary_y=True, showgrid=False)
                st.plotly_chart(fig_evo, use_container_width=True)

                # --- RÉPARTITIONS ---
                st.divider()
                c1, c2 = st.columns(2)
                
                with c1:
                    st.write("### 📱 Impressions / Device")
                    fig_device = px.pie(df, values=col_imp, names='DEVICE_CATEGORY_NAME', hole=0.4)
                    fig_device.update_layout(
                        margin=dict(t=20, b=20, l=0, r=0),
                        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
                    )
                    fig_device.update_traces(textposition='inside', textinfo='percent')
                    st.plotly_chart(fig_device, use_container_width=True)
                    
                with c2:
                    st.write("### 🎨 Impressions / Création")
                    fig_creative = px.pie(df, values=col_imp, names='CREATIVE_NAME', hole=0.4)
                    fig_creative.update_layout(
                        margin=dict(t=20, b=20, l=0, r=0),
                        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
                    )
                    fig_creative.update_traces(textposition='inside', textinfo='percent')
                    st.plotly_chart(fig_creative, use_container_width=True)
                    
                # --- VISUELS ---
                st.divider()
                st.subheader("🖼️ Bibliothèque des visuels")
                if creats:
                    cols = st.columns(3)
                    for i, c in enumerate(creats):
                        with cols[i % 3]:
                            with st.container(border=True): # Ajout d'un cadre esthétique
                                st.markdown(f"**{c['name']}**")
                                if c['image']: 
                                    st.image(c['image'], use_container_width=True)
                                elif c['preview']: 
                                    st.link_button("👁️ Ouvrir la Preview", c['preview'], use_container_width=True)
                else:
                    st.info("Aucun visuel associé à cette campagne n'a été trouvé.")
            else:
                 st.error("Les colonnes d'impressions ou de clics sont introuvables dans le rapport.")
        else:
            st.warning("⚠️ Aucune donnée de diffusion n'est remontée pour cette campagne.")
else:
    st.info("👈 Veuillez commencer par sélectionner un annonceur et une campagne dans le panneau de gauche.")