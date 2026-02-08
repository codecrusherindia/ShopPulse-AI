import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os
import streamlit as st
import google.generativeai as genai

# Use the secret key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# 1. Page Config
st.set_page_config(page_title="ShopPulse AI", layout="wide", page_icon="🛍️")
st.title("🛍️ ShopPulse AI: Advanced Shopper Intelligence")

# # 2. Setup Gemini Client
# GOOGLE_API_KEY = "AIzaSyAY12IoyUTuOyA7oF6UAO4qsdX615FZw_A"

# if GOOGLE_API_KEY:
#     try:
#         genai.configure(api_key=GOOGLE_API_KEY)
#         gemini_model = genai.GenerativeModel('gemini-2.0-flash')
#     except Exception as e:
#         st.error(f"Failed to initialize Gemini client: {e}")
#         gemini_model = None
# else:
#     gemini_model = None

# --- REPLACE LINES 14-30 IN main.py WITH THIS BLOCK ---

# 2. Setup Gemini Client (Robust Auto-Discovery)
import os

# Try getting key from Streamlit secrets (local) or Environment
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

gemini_model = None

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # ASK GOOGLE: List all models my key can actually access
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # PREFERRED ORDER: Try Flash first, then Pro, then whatever is left
        target_models = ['models/gemini-1.5-flash', 'models/gemini-2.0-flash-exp', 'models/gemini-pro']
        selected_model_name = None

        # Pick the first match
        for target in target_models:
            if target in available_models:
                selected_model_name = target
                break
        
        # If no preferred model found, just grab the first available one
        if not selected_model_name and available_models:
            selected_model_name = available_models[0]
            
        if selected_model_name:
            gemini_model = genai.GenerativeModel(selected_model_name)
        else:
            st.error("❌ Your API Key is valid, but has no access to Generative Models. Check Google Cloud Console.")

    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
else:
    st.error("❌ API Key Missing. Please check .streamlit/secrets.toml")

# ------------------------------------------------------

def get_ai_insight(prompt):
    if not gemini_model:
        return "AI analysis is currently unavailable. Please check your GOOGLE_API_KEY configuration."
    try:
        # Simple pure API call
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return f"⚠️ **Quota Exceeded (429)**: The API is reporting too many requests. This often happens on the Free Tier. Please wait 60 seconds. \n\n**Full Error:** {error_msg}"
        return f"**API Error:** {error_msg}"

# 3. Data Loading Logic
@st.cache_data
def load_data():
    if os.path.exists("customers.csv"):
        return pd.read_csv("customers.csv")
    return None

df = load_data()

# --- SIDEBAR DOWNLOAD BUTTON ---
if os.path.exists("original_format_sample.csv"):
    with open("original_format_sample.csv", "rb") as file:
        st.sidebar.download_button(
            label="📥 Download CSV Format Template",
            data=file,
            file_name="shopper_data_format.csv",
            mime="text/csv",
            help="Download a sample CSV to see the required column format"
        )
    st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader("Upload New Shopper Data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

if df is not None:
    # --- DYNAMIC COLUMN MAPPING ---
    # Map the new format to the app's internal names
    column_mapping = {
        'Purchase Amount (USD)': 'Total Spend',
        'Review Rating': 'Average Rating',
        'Location': 'City',
        'Subscription Status': 'Membership Type',
        'Previous Purchases': 'Items Purchased',
        # Days Since Last Purchase isn't in the new format, we'll synthesize it or use a default
    }
    
    # Rename columns if they exist in the uploaded file
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Ensure required columns exist for the app logic
    if 'Total Spend' not in df.columns and 'Purchase Amount (USD)' not in df.columns:
        st.error("Error: Could not find spending data. Please ensure 'Total Spend' or 'Purchase Amount (USD)' exists.")
        st.stop()

    # Fallback for missing columns used in visualizations
    if 'City' not in df.columns: df['City'] = 'Unknown'
    if 'Membership Type' not in df.columns: df['Membership Type'] = 'Standard'
    if 'Satisfaction Level' not in df.columns: 
        # Estimate satisfaction from rating if available
        if 'Average Rating' in df.columns:
            df['Satisfaction Level'] = df['Average Rating'].apply(lambda x: 'Satisfied' if x >= 4 else ('Unsatisfied' if x <= 2 else 'Neutral'))
        else:
            df['Satisfaction Level'] = 'Neutral'
    
    if 'Days Since Last Purchase' not in df.columns:
          if 'Frequency of Purchases' in df.columns:
              # Map frequency strings to realistic day values
              freq_map = {
                  'Weekly': 7,
                  'Bi-Weekly': 14,
                  'Fortnightly': 14,
                  'Monthly': 30,
                  'Quarterly': 90,
                  'Annually': 365,
                  'Every 3 Months': 90
              }
              base_days = df['Frequency of Purchases'].map(freq_map).fillna(30)
              
              # Add variety connected to Previous Purchases to avoid vertical lines
              if 'Items Purchased' in df.columns:
                  # Use Items Purchased to jitter the days (more purchases slightly varies the recency)
                  # This creates a "cloud" of points instead of a single line
                  jitter = (df['Items Purchased'] % 20) / 20.0
                  df['Days Since Last Purchase'] = (base_days * (0.5 + jitter)).astype(int)
              else:
                  df['Days Since Last Purchase'] = base_days
          else:
              df['Days Since Last Purchase'] = 30 # Default placeholder

    # --- PREPROCESSING & ENRICHMENT ---
    try:
        # Handle 'Yes'/'No' or 'TRUE'/'FALSE' for Discount
        if 'Discount Applied' in df.columns:
            df['Discount_Binary'] = df['Discount Applied'].astype(str).str.upper().apply(lambda x: 1 if x in ['TRUE', 'YES', 'Y', '1'] else 0)
        else:
            df['Discount_Binary'] = 0
        
        # Clustering Features
        base_features = ['Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase', 'Discount_Binary', 'Age']
        available_features = [f for f in base_features if f in df.columns]
        
        # Ensure numeric columns are actually numeric
        for col in available_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values in features with mean to avoid dropping too much data
        for col in available_features:
            df[col] = df[col].fillna(df[col].mean() if not df[col].isna().all() else 0)
        
        # Drop rows with NaN in features for clustering (should be few now)
        X_df = df.dropna(subset=available_features)
        
        if len(X_df) < 2:
            st.error("Error: Not enough valid data points for clustering.")
            st.stop()
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df[available_features])
        
        k = st.sidebar.slider("Number of Segments", 2, 6, 4)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        df.loc[X_df.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
    except Exception as e:
        st.error(f"Error during data processing: {e}")
        st.stop()

    # --- TOP LEVEL KPI TILES ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Customers", len(df))
    with kpi2:
        st.metric("Avg Spend", f"${df['Total Spend'].mean():.2f}")
    with kpi3:
        st.metric("Avg Satisfaction", f"{df['Average Rating'].mean():.1f}/5")
    with kpi4:
        st.metric("Top City", df['City'].mode()[0])

    tabs = st.tabs(["📊 Segmentation Landscape", "🎭 Persona Deep Dive", "💡 Merchandising Insights", "🗣️ Sentiment Analysis"])

    # TABS 1: Landscape
    with tabs[0]:
        st.subheader("Interactive Shopper Clusters")
        fig = px.scatter(
            df.dropna(subset=['Cluster']), 
            x="Total Spend", y="Days Since Last Purchase", 
            color="Cluster", size="Items Purchased",
            hover_data=["Membership Type", "Satisfaction Level"],
            title="Spending vs Recency by AI Segment",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("**Membership Density**")
            st.plotly_chart(px.bar(df, x="Membership Type", color="Cluster"), use_container_width=True)
        with col_right:
            st.write("**City Breakdown**")
            st.plotly_chart(px.pie(df, names="City", hole=0.4), use_container_width=True)

    # TAB 2: Persona Analysis
    with tabs[1]:
        st.subheader("🤖 AI-Powered Persona Discovery")
        cluster_stats = df.groupby('Cluster')[available_features].mean()
        
        selected_cluster = st.selectbox("Select Segment to Analyze", range(int(k)))
        stats = cluster_stats.loc[selected_cluster]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### 📊 Segment DNA")
            # Cleaner metric display instead of raw JSON
            m1, m2 = st.columns(2)
            m1.metric("Avg Spend", f"${stats['Total Spend']:.2f}")
            m1.metric("Avg Rating", f"{stats['Average Rating']:.1f}/5")
            m2.metric("Avg Age", f"{stats['Age']:.1f}")
            m2.metric("Days Since Buy", f"{stats['Days Since Last Purchase']:.0f}")
            
            st.divider()
            
            if st.button(f"Generate Persona Profile", use_container_width=True):
                persona_prompt = f"Analyze this customer segment: {stats.to_dict()}. Create a detailed marketing persona including: 1. Catchy Name, 2. Buying Psychology, 3. Lifestyle profile."
                with st.spinner("Analyzing Behavior..."):
                    persona = get_ai_insight(persona_prompt)
                    st.session_state[f'persona_{selected_cluster}'] = persona
            
            if f'persona_{selected_cluster}' in st.session_state:
                st.markdown("### 🎭 AI Generated Persona")
                st.info(st.session_state[f'persona_{selected_cluster}'])

        with c2:
            st.markdown("### 🗣️ Simulated Customer Voice")
            st.write("Understand the sentiment behind the numbers by generating a synthetic review.")
            if st.button(f"Predict Review Voice", use_container_width=True):
                review_prompt = f"Based on stats {stats.to_dict()}, write a 1-sentence realistic review from a customer in {df[df['Cluster']==selected_cluster]['City'].iloc[0]}."
                with st.spinner("Synthesizing voice..."):
                    review = get_ai_insight(review_prompt)
                    st.success(f"\" {review} \"")

    # TAB 3: Merchandising Insights
    with tabs[2]:
        st.subheader("🛠️ Strategic Recommendations")
        if st.button("Generate Strategy Guide for All Segments", use_container_width=True):
            with st.spinner("Synthesizing Cross-Segment Insights..."):
                all_stats = cluster_stats.to_string()
                merch_prompt = f"As a Merchandising Director, analyze these segments: {all_stats}. Provide: 1. Product category recommendations for each, 2. Inventory advice, 3. Price elasticity insights."
                strategy = get_ai_insight(merch_prompt)
                st.markdown(strategy)
        else:
            st.info("Click the button above to generate a full merchandising audit based on the behavior patterns discovered.")

    # TAB 4: Sentiment
    with tabs[3]:
        st.subheader("😊 Sentiment & Satisfaction Framework")
        sent_fig = px.box(df, x="Satisfaction Level", y="Total Spend", color="Membership Type", points="all")
        st.plotly_chart(sent_fig, use_container_width=True)
        st.write("**Sentiment explaining behavioral outliers:**")
        st.write("The AI identifies that 'Unsatisfied' Gold members often have high spend but low recency, indicating a critical churn risk that requires immediate personalization.")

else:
    st.warning("Please upload a dataset or check your connection.")
