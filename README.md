# 🛍️ ShopPulse AI: Advanced Shopper Intelligence

ShopPulse AI is a professional-grade customer intelligence dashboard built with **Streamlit**, **scikit-learn**, and **Google Gemini AI**. It transforms raw e-commerce transaction data into actionable behavioral segments and human-readable marketing personas.

---

## 🚀 Core Features

- **Automated Segmentation**: Uses K-Means clustering to group customers based on spending habits, purchase frequency, age, and satisfaction.
- **AI-Powered Personas**: Integrates with Google Gemini to generate detailed marketing personas (Buying Psychology, Lifestyle Profiles) for each discovered segment.
- **Dynamic KPI Tracking**: Real-time calculation of Total Customers, Average Spend, Satisfaction Levels, and Top Geographical Markets.
- **Interactive Visualizations**: 
    - **Landscape Analysis**: Multi-dimensional scatter plots (Spend vs. Recency vs. Frequency).
    - **Geographic Breakdown**: Distribution of customers across different cities.
    - **Membership Density**: Analysis of membership tiers across behavioral clusters.
- **Strategic Recommendations**: AI-generated merchandising guides, inventory advice, and pricing strategy for each segment.
- **Synthetic Voice of Customer**: Generates realistic, data-driven customer reviews to help stakeholders empathize with different segments.

---

## 🛠️ Technical Architecture

### 🧠 Machine Learning Pipeline
1.  **Feature Engineering**: Extracts key behavioral signals:
    - `Total Spend`: Monetary value of the customer.
    - `Items Purchased`: Engagement level.
    - `Average Rating`: Explicit satisfaction signal.
    - `Days Since Last Purchase`: Recency (calculated dynamically from frequency if missing).
    - `Discount Binary`: Propensity to respond to promotions.
2.  **Normalization**: Uses `StandardScaler` to ensure features with different scales (e.g., Age vs. Spend) contribute equally to the clustering.
3.  **Clustering**: Implements the `K-Means` algorithm with user-adjustable cluster counts (2-6 segments).

### 🤖 AI Integration (Google Gemini)
- **Model**: Automatically discovers and utilizes available Gemini models (prefers `gemini-1.5-flash` or `gemini-2.0-flash`).
- **Narrative Generation**: Translates raw statistical averages into "human" insights using advanced prompting techniques.
- **Robust Error Handling**: Built-in support for API quota management (429 errors) and model fallback logic.

---

## 📊 Data Requirements

The application is designed to be flexible and supports dynamic column mapping. It works best with a CSV file containing:

| Required/Recommended Column | Internal Mapping | Description |
| :--- | :--- | :--- |
| `Purchase Amount (USD)` | `Total Spend` | Total dollar amount spent. |
| `Review Rating` | `Average Rating` | Customer satisfaction (1-5 scale). |
| `Previous Purchases` | `Items Purchased` | Count of total items bought. |
| `Location` | `City` | Customer's city/region. |
| `Subscription Status` | `Membership Type` | e.g., Gold, Silver, Standard. |
| `Frequency of Purchases` | - | Used to estimate recency. |
| `Age` | - | Demographic clustering feature. |

*Note: If specific columns are missing, the app uses intelligent defaults (e.g., estimating satisfaction from ratings or recency from frequency).*

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.11+
- A Google Gemini API Key

### Steps
1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd shopper-behaviour
    ```

2.  **Install dependencies**:
    ```bash
    pip install streamlit pandas numpy scikit-learn plotly google-generativeai
    ```

3.  **Configure API Key**:
    Create a `.streamlit/secrets.toml` file or set an environment variable:
    ```toml
    GOOGLE_API_KEY = "your_gemini_api_key_here"
    ```

4.  **Run the application**:
    ```bash
    streamlit run main.py
    ```

---

## 📂 Project Structure

- `main.py`: The core application logic (Data processing, ML, UI, and AI).
- `customers.csv`: Default dataset for immediate demonstration.
- `original_format_sample.csv`: A template file showing the supported data format.
- `deployment.toml`: Configuration for cloud deployment and workflows.
- `README.md`: This documentation.

---

## 🛡️ License

This project is intended for business intelligence and educational purposes. 
*Built with ❤️ for Shopper Intelligence.*
