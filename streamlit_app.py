import sys
import subprocess

# --- AUTO-INSTALL MISSING PACKAGES (Streamlit Cloud Recovery) ---
required_packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "plotly": "plotly"
}

for package, import_name in required_packages.items():
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.express as px
import plotly.graph_objects as px_go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bank Churn Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INJECT CUSTOM CSS FOR BRANDING ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #4285F4; color: white; }
    .stButton>button:hover { background-color: #357ae8; }
    h1 { color: #202124; font-family: 'Google Sans', sans-serif; font-weight: 700; }
    h2 { color: #4285F4; font-family: 'Google Sans', sans-serif; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-val { font-size: 32px; font-weight: bold; color: #4285F4; }
    .metric-label { font-size: 14px; color: #5f6368; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load local churn dataset
        df = pd.read_csv("churn_data.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ 'churn_data.csv' not found. Please place it in the same directory as this script.")
        # Fallback empty dataframe with structure
        return pd.DataFrame(columns=[
            "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", 
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
        ])

df = load_data()

# --- TITLE BANNER ---
st.title("🏦 Bank Customer Churn Live Analytics Dashboard")
st.markdown("""
This production-ready dashboard implements predictive churn models, real-time risk scores, 
and what-if calculators based on Google Cloud financial analytics practices.
""")

# --- SIDEBAR: NAVIGATION AND CALCULATOR ---
st.sidebar.image("https://www.gstatic.com/images/branding/googlelogo/svg/google_logo_color_272x92dp.png", width=120)
st.sidebar.markdown("### 🎛️ Navigation & Inputs")
page = st.sidebar.radio("Go to Page:", ["📊 Live Analytics & Model Training", "🔮 Real-Time Churn Predictor", "📄 Executive Summary & Guide"])

# Preprocess & Feature Engineer data for modeling
def preprocess_data(data):
    if data.empty:
        return None, None, None
    
    # Feature Engineering matching the analytical design spec
    processed = data.copy()
    processed['isGermany'] = np.where(processed['Geography'] == 'Germany', 1, 0)
    processed['isSpain'] = np.where(processed['Geography'] == 'Spain', 1, 0)
    processed['isMale'] = np.where(processed['Gender'] == 'Male', 1, 0)
    
    # Derived interaction features
    processed['balanceToSalaryRatio'] = np.where(processed['EstimatedSalary'] > 0, processed['Balance'] / processed['EstimatedSalary'], 0)
    processed['productDensity'] = processed['NumOfProducts'] / (processed['Tenure'] + 1)
    processed['engagementProductInteraction'] = processed['IsActiveMember'] * processed['NumOfProducts']
    processed['ageTenureInteraction'] = processed['Age'] * processed['Tenure']
    
    feature_cols = [
        "CreditScore", "isGermany", "isSpain", "isMale", "Age", "Tenure", 
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", 
        "balanceToSalaryRatio", "productDensity", "engagementProductInteraction", "ageTenureInteraction"
    ]
    
    X = processed[feature_cols]
    y = processed["Exited"]
    
    return X, y, feature_cols

# --- PAGE 1: ANALYTICS & MODEL TRAINING ---
if page == "📊 Live Analytics & Model Training":
    st.header("📈 Bank Customer Portfolio & Predictive Performance")
    
    if df.empty:
        st.stop()

    # Upper stats KPI metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-val'>{}</div><div class='metric-label'>Total Customers Logged</div></div>".format(len(df)), unsafe_allow_html=True)
    with col2:
        churn_rate = (df['Exited'].mean() * 100)
        st.markdown("<div class='metric-card'><div class='metric-val'>{:.1f}%</div><div class='metric-label'>Average Churn Rate</div></div>".format(churn_rate), unsafe_allow_html=True)
    with col3:
        avg_age = df['Age'].mean()
        st.markdown("<div class='metric-card'><div class='metric-val'>{:.1f}</div><div class='metric-label'>Average Customer Age</div></div>".format(avg_age), unsafe_allow_html=True)
    with col4:
        avg_balance = df['Balance'].mean()
        st.markdown("<div class='metric-card'><div class='metric-val'>€{:.0f}</div><div class='metric-label'>Average Balance</div></div>".format(avg_balance), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Explorer & Model config side-by-side
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("🤖 Model Selection & Training Settings")
        model_type = st.selectbox("Select ML Classifier:", ["Logistic Regression (Standardized)", "Random Forest (Ensemble)", "Decision Tree (Interpretability)"])
        test_size = st.slider("Test Partition Split (%)", 10, 40, 20, step=5)
        
        # Prepare data & train model
        X, y, feature_names = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == "Logistic Regression (Standardized)":
            model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
            model.fit(X_train_scaled, y_train)
            importances = np.abs(model.coef_[0])
            raw_coefficients = model.coef_[0]
        elif model_type == "Random Forest (Ensemble)":
            model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train) # Tree-based can train on unscaled
            importances = model.feature_importances_
            raw_coefficients = importances # positive magnitude only
        else:
            model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            importances = model.feature_importances_
            raw_coefficients = importances
            
        # Get metrics
        if model_type == "Logistic Regression (Standardized)":
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_prob)
        
        # Display Trained Metrics
        st.markdown("#### 🎯 Test Set Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Accuracy", f"{acc:.2%}")
        m_col2.metric("Precision", f"{prec:.2%}")
        m_col3.metric("Recall (Sensitivity)", f"{rec:.2%}")
        
        m_col4, m_col5, m_col6 = st.columns(3)
        m_col4.metric("F1-Score", f"{f1:.2%}")
        m_col5.metric("ROC-AUC Score", f"{auc_score:.2%}")
        m_col6.metric("Train/Test Samples", f"{len(X_train)} / {len(X_test)}")
        
        st.info("💡 **Precision Focus:** Reducing false positives (predicting churn when safe) avoids wasteful retention marketing spend. **Recall Focus:** Captures maximum true risk.")

    with right_col:
        st.subheader("🌲 Feature Importance (Drivers of Churn)")
        
        # Plot importances
        importance_df = pd.DataFrame({
            "Feature": [
                "Credit Score", "Region: Germany", "Region: Spain", "Gender: Male", "Age", "Tenure (Years)", 
                "Account Balance", "Number of Products", "Has Credit Card", "Is Active Member", "Estimated Salary",
                "Balance-to-Salary Ratio", "Product Density Indicator", "Engagement-Product Interaction", "Age-Tenure Interaction"
            ],
            "Importance": importances,
            "Sign": ["Positive" if c >= 0 else "Negative" for c in raw_coefficients] if model_type == "Logistic Regression (Standardized)" else ["Positive"]*len(importances)
        }).sort_values(by="Importance", ascending=True)
        
        fig = px.bar(
            importance_df, 
            y="Feature", 
            x="Importance", 
            orientation="h",
            color="Sign" if model_type == "Logistic Regression (Standardized)" else None,
            color_discrete_map={"Positive": "#EA4335", "Negative": "#34A853"},
            title="Relative Churn Drivers (Standardized Coeffs Magnitude)"
        )
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0), showlegend=model_type == "Logistic Regression (Standardized)")
        st.plotly_chart(fig, use_container_width=True)

    # Churn Segment Distributions
    st.subheader("📊 Customer Distributions by Churn Outcome")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        fig_age = px.histogram(df, x="Age", color="Exited", barmode="group", title="Age Group Distribution vs Churn Status", color_discrete_sequence=["#34A853", "#EA4335"])
        st.plotly_chart(fig_age, use_container_width=True)
    with d_col2:
        fig_prod = px.histogram(df, x="NumOfProducts", color="Exited", barmode="group", title="Product Count Distribution vs Churn Status", color_discrete_sequence=["#34A853", "#EA4335"])
        st.plotly_chart(fig_prod, use_container_width=True)


# --- PAGE 2: WHAT-IF CHURN PREDICTOR ---
elif page == "🔮 Real-Time Churn Predictor":
    st.header("🔮 Customer Churn Risk What-If Simulator")
    st.markdown("Modify customer demographics and balance attributes in the sidebar panel to see their real-time risk score.")
    
    if df.empty:
        st.warning("⚠️ 'churn_data.csv' is missing. Please download it from the dashboard and place it in the same folder as this script to enable the churn predictor.")
        st.stop()
        
    # Setup trained model for deployment prediction
    X, y, feature_names = preprocess_data(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    deploy_model = LogisticRegression(C=1.0, class_weight='balanced')
    deploy_model.fit(X_scaled, y)
    
    # Simulator Controls (demographics)
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_age = st.slider("Customer Age", 18, 90, 38)
        sim_geography = st.selectbox("Geography / Region", ["France", "Germany", "Spain"])
        sim_gender = st.selectbox("Gender", ["Female", "Male"])
    with c2:
        sim_balance = st.number_input("Account Balance (€)", min_value=0.0, value=75000.0, step=1000.0)
        sim_salary = st.number_input("Estimated Salary (€)", min_value=1000.0, value=115000.0, step=1000.0)
        sim_products = st.slider("Number of Products", 1, 4, 1)
    with c3:
        sim_credit = st.slider("Credit Score", 300, 850, 620)
        sim_tenure = st.slider("Tenure with Bank (Years)", 0, 10, 3)
        sim_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        sim_card = st.selectbox("Has Credit Card?", ["Yes", "No"])

    # Map simulated customer row
    sim_row = pd.DataFrame([{
        "CreditScore": sim_credit,
        "Geography": sim_geography,
        "Gender": sim_gender,
        "Age": sim_age,
        "Tenure": sim_tenure,
        "Balance": sim_balance,
        "NumOfProducts": sim_products,
        "HasCrCard": 1 if sim_card == "Yes" else 0,
        "IsActiveMember": 1 if sim_active == "Yes" else 0,
        "EstimatedSalary": sim_salary,
        "Exited": 0 # dummy
    }])
    
    # Apply same feature engineering
    sim_row['isGermany'] = np.where(sim_row['Geography'] == 'Germany', 1, 0)
    sim_row['isSpain'] = np.where(sim_row['Geography'] == 'Spain', 1, 0)
    sim_row['isMale'] = np.where(sim_row['Gender'] == 'Male', 1, 0)
    sim_row['balanceToSalaryRatio'] = np.where(sim_row['EstimatedSalary'] > 0, sim_row['Balance'] / sim_row['EstimatedSalary'], 0)
    sim_row['productDensity'] = sim_row['NumOfProducts'] / (sim_row['Tenure'] + 1)
    sim_row['engagementProductInteraction'] = sim_row['IsActiveMember'] * sim_row['NumOfProducts']
    sim_row['ageTenureInteraction'] = sim_row['Age'] * sim_row['Tenure']
    
    sim_X = sim_row[feature_names]
    sim_X_scaled = scaler.transform(sim_X)
    
    # Predict Probability
    prob = deploy_model.predict_proba(sim_X_scaled)[0, 1]
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Visual gauge/radial indicator of risk probability
    g1, g2 = st.columns([1, 2])
    with g1:
        st.subheader("🎯 Risk Score Indicator")
        if prob < 0.3:
            color = "#34A853"
            risk_label = "🟢 LOW RISK"
        elif prob < 0.6:
            color = "#FBBC05"
            risk_label = "🟡 MEDIUM RISK"
        else:
            color = "#EA4335"
            risk_label = "🔴 HIGH RISK"
            
        fig_gauge = px_go.Figure(px_go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{risk_label}"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(52, 168, 83, 0.15)"},
                    {'range': [30, 60], 'color': "rgba(251, 188, 5, 0.15)"},
                    {'range': [60, 100], 'color': "rgba(234, 67, 53, 0.15)"}
                ]
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with g2:
        st.subheader("💡 Churn Driver Insights & retention Advice")
        
        # Explain the prediction
        contrib = sim_X_scaled[0] * deploy_model.coef_[0]
        contrib_df = pd.DataFrame({
            "Driver": [
                "Credit Score", "Germany Region", "Spain Region", "Is Male", "Age", "Tenure", 
                "Balance", "Product Count", "Credit Card Owner", "Active Status", "Salary",
                "Balance/Salary", "Product Density", "Engagement Interact", "Age-Tenure interaction"
            ],
            "Contribution Score": contrib
        }).sort_values(by="Contribution Score", key=abs, ascending=False)
        
        st.markdown(f"**Customer profile is rated at a {prob:.1%} probability of churning.**")
        
        top_driver = contrib_df.iloc[0]
        if top_driver["Contribution Score"] > 0:
            st.warning(f"⚠️ **Key Risk Driver:** The customer's **{top_driver['Driver']}** is significantly pushing up their churn risk score (+{top_driver['Contribution Score']:.2f}).")
        else:
            st.success(f"✅ **Key Retention Factor:** The customer's **{top_driver['Driver']}** is significantly keeping them loyal to the bank (-{abs(top_driver['Contribution Score']):.2f}).")
            
        # Personalized actions based on variables
        st.markdown("#### 🎯 Proactive Retention Actions:")
        if prob >= 0.3:
            actions = []
            if sim_products >= 3:
                actions.append("• **Multi-Product Optimization:** Customer holds too many products, potentially signaling product confusion or billing friction. Schedule a consolidation review.")
            if sim_active == "No":
                actions.append("• **Engagement Offer:** Member is inactive. Send a targeted email campaign highlighting active benefits or cash-back credit card incentives.")
            if sim_balance > 100000 and sim_products == 1:
                actions.append("• **Wealth Management:** High-net-worth customer with only 1 product. Offer a personalized financial advisor session to cross-sell secure investment packages.")
            if sim_age > 45:
                actions.append("• **Premium Support Referral:** Age factor indicates high life-value. Route to executive customer assistance channels.")
                
            if not actions:
                actions.append("• **Customer Engagement:** Provide a standard proactive satisfaction survey and reward point loyalty benefits.")
                
            for action in actions:
                st.write(action)
        else:
            st.write("✨ Customer is currently highly loyal. No active intervention needed. Maintain standardized automated engagement campaigns.")


# --- PAGE 3: GUIDE & EXECUTIVE SUMMARY ---
else:
    st.header("📄 Project Documentation & Deployment Guide")
    st.markdown("Prepared by **Ashutosh Gupta**, Lead Financial Analyst and Advisor.")

    doc_tab1, doc_tab2 = st.tabs(["🚀 Deploying this App to Streamlit", "📈 Executive Summary"])
    
    with doc_tab1:
        st.subheader("How to upload and run this dashboard in Streamlit Community Cloud")
        st.markdown("""
        ### Step 1: Create a GitHub Repository
        1. Log in to [GitHub](https://github.com) and click **New Repository**.
        2. Name it `bank-customer-churn-dashboard`.
        3. Keep it Public (or Private) and check **Add a README file**. Click **Create Repository**.

        ### Step 2: Upload Project Files
        You need exactly **3 files** in your repository:
        1. **`app.py`** (This Python file - copy the script from the code panel).
        2. **`requirements.txt`** (Tells Streamlit what packages to install).
        3. **`churn_data.csv`** (Your parsed dataset).

        You can copy the code and download the CSV directly from the dashboard panel and push them using terminal commands:
        ```bash
        git clone <your-github-repo-url>
        cd bank-customer-churn-dashboard
        # Put app.py, requirements.txt, and churn_data.csv inside the directory
        git add .
        git commit -m "Deploy bank customer churn predictive analytics"
        git push origin main
        ```

        ### Step 3: Deploy to Streamlit Community Cloud
        1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
        2. Click **New App** in the top-right corner.
        3. Choose your Repository, Branch (`main`), and set the main file path to **`app.py`**.
        4. Click **Deploy!** In less than a minute, your interactive churn predictor will be live on a public URL!
        """)

    with doc_tab2:
        st.subheader("📈 Executive Summary for Strategic Bank Stakeholders")
        st.markdown("""
        ### 🏦 Predictive churn Risk Analytics & Retention Strategy
        
        **To the European Central Bank and Board of Directors, Retail Banking Division:**

        This quantitative investigation reframes customer churn at retail banks from a Demographical perspective into an **Engagement and Relationship-Strength perspective**. Traditional banking strategies often react *after* a customer closes their account, incurring immense replacement costs. Our machine learning intelligence model assigns a **Risk Probability Score** to active clients, allowing proactive retention efforts.

        #### 🔍 Key Analytical Findings:
        1. **Age Churn Trajectory:** Customer age is the strongest positive driver of churn. Customers between 40-55 show peak churn probability, representing a critical career/retirement asset transition period where clients are sensitive to service quality and cross-bank offers.
        2. **Product Friction:** Holding 3 or more bank products actually *increases* churn risk. This is counter-intuitive and points to service friction, complex fee structures, or bad bundle fit. 
        3. **Engagement Lift:** Active member indicators and credit card ownership significantly reduce the log-odds of churning. High-touch clients are highly retained.
        4. **Geographic Variance:** Clients based in Germany exhibit more than double the average churn rate of Spain or France, requiring a localized review of competitive fee structures and localized retention offers.

        #### 📈 Strategic Recommendations:
        *   **Implement High-Net-Worth Cross-Selling:** Target high-balance, low-product clients with tailored wealth management consulting instead of broad multi-product email campaigns.
        *   **Germany Specific Care Plan:** Offer exclusive zero-fee premium accounts for regional accounts with a tenure greater than 3 years in Germany.
        *   **Precision Target Campaigns:** Shift marketing budgets from expensive mass loyalty rewards to targeted cash-back or active-member bonuses for predicted medium-to-high risk accounts.
        """)
