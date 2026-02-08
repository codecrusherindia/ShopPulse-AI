import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os
import google.generativeai as genai

# 1. Page Config (Must be the first Streamlit command)
st.set_page_config(page_title="ShopPulse AI", layout="wide", page_icon="🛍️")
st.title("🛍️ ShopPulse AI: Advanced Shopper Intelligence")

# 2. Setup Gemini Client (Robust Auto-Discovery)
# This checks Streamlit Secrets (Cloud) AND Environment Variables (Render/Docker)
api_key = None
try:
    api_key = st.secrets.get("GOOGLE_API_KEY")
except Exception:
    pass

if not api_key:
    api_key = os.environ.get("GOOGLE_API_KEY")

gemini_model = None

if api_key:
    try:
        genai.configure(api_key=api_key)
        
        # Smart Model Selection: Try the best available model
        target_models = ['models/gemini-1.5-flash', 'models/gemini-2.0-flash-exp', 'models/gemini-pro']
        selected_model_name = None
        
        # internal check for available models is good, but often fails on limited keys
        # We will default to flash as it's the most widely available free tier model
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')

    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
else:
    st.warning("⚠️ API Key not found. AI insights will be disabled. (Set GOOGLE_API_KEY in Secrets or Environment)")

# 3. Helper Functions
def get_ai_insight(prompt):
    if not gemini_model:
        return "AI analysis is currently unavailable. Please check your GOOGLE_API_KEY configuration."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

@st.cache_data
def generate_dummy_data():
    """Generates sample data so the app works immediately without upload."""
    np.random.seed(42)
    data = {
        'Customer ID': range(1001, 1101),
        'Total Spend': np.random.randint(50, 2000, 100),
        'Items Purchased': np.random.randint(1, 20, 100),
        'Average Rating': np.random.uniform(1, 5, 100).round(1),
        'Days Since Last Purchase': np.random.randint(1, 365, 100),
        'Age': np.random.randint(18, 70, 100),
        'Membership Type': np.random.choice(['Gold', 'Silver', 'Standard'], 100),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 100),
        'Discount_Binary': np.random.choice([0, 1], 100)
    }
    return pd.DataFrame(data)

# 4. Data Loading Logic
st.sidebar.header("📂 Data Setup")
uploaded_file = st.sidebar.file_uploader("Upload Shopper Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File Uploaded Successfully")
else:
    df = generate_dummy_data()
    st.sidebar.info("ℹ️ Using Demo Data (Upload CSV to override)")

# 5. Data Processing & Mapping
if df is not None:
    # Standardize Column Names
    column_mapping = {
        'Purchase Amount (USD)': 'Total Spend',
        'Review Rating': 'Average Rating',
        'Location': 'City',
        'Subscription Status': 'Membership Type',
        'Previous Purchases': 'Items Purchased',
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Ensure crucial columns exist
    if 'Total Spend' not in df.columns and 'Purchase Amount (USD)' not in df.columns:
        st.error("❌ Error: Dataset must contain 'Total Spend' or 'Purchase Amount (USD)'")
        st.stop()
    
    # Fill missing columns with defaults
    if 'City' not in df.columns: df['City'] = 'Unknown'
    if 'Membership Type' not in df.columns: df['Membership Type'] = 'Standard'
    if 'Age' not in df.columns: df['Age'] = 30
    if 'Days Since Last Purchase' not in df.columns: 
        # Synthesize recency if missing
        if 'Frequency of Purchases' in df.columns:
             freq_map = {'Weekly': 7, 'Monthly': 30, 'Annually': 365}
             df['Days Since Last Purchase'] = df['Frequency of Purchases'].map(freq_map).fillna(30)
        else:
            df['Days Since Last Purchase'] = 30
    
    if 'Discount_Binary' not in df.columns: 
        if 'Discount Applied' in df.columns:
             df['Discount_Binary'] = df['Discount Applied'].astype(str).str.upper().apply(lambda x: 1 if x in ['TRUE', 'YES', '1'] else 0)
        else:
             df['Discount_Binary'] = 0

    # Calculate Satisfaction if missing
    if 'Satisfaction Level' not in df.columns:
        if 'Average Rating' in df.columns:
            df['Satisfaction Level'] = df['Average Rating'].apply(
                lambda x: 'Satisfied' if x >= 4 else ('Unsatisfied' if x <= 2 else 'Neutral')
            )
        else:
            df['Satisfaction Level'] = 'Neutral'

    # --- CLUSTERING LOGIC ---
    try:
        # Select features for clustering
        base_features = ['Total Spend', 'Items Purchased', 'Days Since Last Purchase', 'Age']
        available_features = [f for f in base_features if f in df.columns]
        
        # Clean data for clustering
        X_df = df.dropna(subset=available_features)
        
        if len(X_df) > 5:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_df[available_features])
            
            k = st.sidebar.slider("Number of Customer Segments", 2, 6, 4)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            df.loc[X_df.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
        else:
            df['Cluster'] = 0 # Not enough data to cluster
            k = 1
            
    except Exception as e:
        st.error(f"⚠️ Clustering Error: {e}")
        st.stop()

    # --- DASHBOARD UI ---
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", len(df))
    k2.metric("Avg Spend", f"${df['Total Spend'].mean():.2f}")
    if 'Average Rating' in df.columns:
        k3.metric("Avg Rating", f"{df['Average Rating'].mean():.1f}/5")
    k4.metric("Top City", df['City'].mode()[0] if not df['City'].mode().empty else "N/A")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Segmentation Landscape", "🎭 AI Persona Deep Dive", "💡 Strategic Insights"])

    # TAB 1: Visuals
    with tab1:
        st.subheader("Interactive Shopper Clusters")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if 'Cluster' in df.columns:
                fig = px.scatter(
                    df, 
                    x="Total Spend", 
                    y="Days Since Last Purchase", 
                    color=df['Cluster'].astype(str),
                    size="Items Purchased" if "Items Purchased" in df.columns else None,
                    hover_data=["Membership Type", "City"],
                    title="Spending vs. Recency (Colored by Segment)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Cluster' in df.columns:
                st.write("**Segment Distribution**")
                st.plotly_chart(px.pie(df, names="Cluster", hole=0.4), use_container_width=True)

    # TAB 2: AI Personas
    with tab2:
        st.subheader("🤖 AI-Powered Persona Analysis")
        
        if 'Cluster' in df.columns:
            # Get stats for selected cluster
            cluster_ids = sorted(df['Cluster'].unique())
            selected_cluster = st.selectbox("Select Segment to Analyze", cluster_ids)
            cluster_data = df[df['Cluster'] == selected_cluster]
            
            avg_spend = cluster_data['Total Spend'].mean()
            avg_recency = cluster_data['Days Since Last Purchase'].mean()
            common_city = cluster_data['City'].mode()[0] if not cluster_data['City'].mode().empty else "Unknown"
            
            # Display Stats
            st.markdown(f"#### Segment {selected_cluster} Stats")
            s1, s2, s3 = st.columns(3)
            s1.metric("Avg Spend", f"${avg_spend:.2f}")
            s2.metric("Recency (Days)", f"{avg_recency:.0f}")
            s3.metric("Top Location", common_city)
            
            st.divider()
            
            # AI Actions
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Generate Marketing Persona", key="btn_persona"):
                    with st.spinner("Consulting Gemini..."):
                        prompt = (
                            f"Create a marketing persona for a customer segment with these stats: "
                            f"Avg Spend: ${avg_spend}, Avg Days Since Last Buy: {avg_recency}, "
                            f"Location: {common_city}. Include: 1. A catchy name, 2. Buying motivation, 3. Best marketing channel."
                        )
                        st.info(get_ai_insight(prompt))
            
            with c2:
                if st.button("Generate Email Draft", key="btn_email"):
                    with st.spinner("Writing email..."):
                        prompt = (
                            f"Write a short, punchy email subject line and body for a customer segment "
                            f"that hasn't bought in {avg_recency} days and usually spends ${avg_spend}."
                        )
                        st.success(get_ai_insight(prompt))

    # TAB 3: Strategy
    with tab3:
        st.subheader("🛠️ Business Recommendations")
        if st.button("Analyze All Segments"):
            with st.spinner("Analyzing entire dataset..."):
                if 'Cluster' in df.columns:
                    summary = df.groupby('Cluster')[['Total Spend', 'Days Since Last Purchase']].mean().to_string()
                    prompt = (
                        f"Analyze these customer segments (Cluster, Spend, Recency): \n{summary}\n"
                        "Provide 3 specific bullet points on how to increase revenue for the lowest spending segment."
                    )
                    st.markdown(get_ai_insight(prompt))
                else:
                    st.error("Not enough data to analyze segments.")

else:
    st.warning("Please upload a CSV file to begin.")
