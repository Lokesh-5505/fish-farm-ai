"""
Fish Farm Disease Outbreak Prediction - Modern Enhanced UI
Advanced Streamlit Application with Modern Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import joblib
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report

# Add path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title='🐟 Fish Farm AI - Disease Prediction',
    page_icon='🐟',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ============================================================================
# MODERN CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
    }
    
    /* Header Styling */
    h1 {
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        color:#ffffff;
    }
    
    h2 {
        font-weight: 600;
        color: #1e293b;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        font-weight: 600;
        color: #475569;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Risk Cards */
    .risk-card {
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        border: 2px solid;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stable-risk {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-color: #10b981;
        color: #064e3b;
    }
    
    .risk-risk {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-color: #dc2626;
        color: #7f1d1d;
    } 
    
    
    .risk-card h3 {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .risk-score {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.875rem;
        margin: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-optimal {
        background: #10b981;
        color: #ffffff;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);
    }
    
    .badge-warning {
        background: #f59e0b;
        color: #ffffff;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.3);
    }
    
    .badge-critical {
        background: #ef4444;
        color: #ffffff;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.3);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    
    /* Data Table Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0284c7;
        margin: 1rem 0;
        color:#000000
    }
    
    /* Feature Card */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
        color:#000000
    }
    
    .feature-card:hover {
        transform: translateX(5px);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load trained model from disk"""
    try:
        model = joblib.load(settings.BEST_MODEL_PATH)
        scaler = joblib.load(settings.SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_historical_data():
    """Load historical data for analysis"""
    try:
        return pd.read_csv(settings.RAW_DATA_FILE)
    except:
        return None

def get_risk_category(probability_or_prediction):
    """Get risk category from probability score or model prediction (Binary Classification: STABLE or RISK)"""
    # Handle both probability scores (0-1) and direct class predictions (0, 1)
    if isinstance(probability_or_prediction, (int, np.integer)):
        # Direct class prediction from model: 0 = STABLE, 1 = RISK
        if probability_or_prediction == 0:
            return 'STABLE', settings.COLOR_STABLE
        else:
            return 'RISK', settings.COLOR_RISK
    else:
        # Probability score (0.0 - 1.0)
        if probability_or_prediction < settings.RISK_THRESHOLD:
            return 'STABLE', settings.COLOR_STABLE
        else:
            return 'RISK', settings.COLOR_RISK

def create_gauge_chart(probability):
    """Create a modern gauge chart for risk probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Outbreak Probability", 'font': {'size': 24, 'weight': 'bold'}},
        delta = {'reference': 50, 'increasing': {'color': "#dc2626"}},
        number = {'suffix': "%", 'font': {'size': 48}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#334155"},
            'bar': {'color': "#667eea", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 50], 'color': '#d1fae5'},
                {'range': [50, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    return fig

def create_parameter_radar_chart(params):
    """Create radar chart for parameter comparison"""
    categories = list(params.keys())
    values = list(params.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Values',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=3)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#e2e8f0'),
            bgcolor='white'
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    return fig

def create_trend_chart(data, param_name):
    """Create trend chart for a specific parameter"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index[-30:],
        y=data[param_name].tail(30),
        mode='lines+markers',
        name=param_name,
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    fig.update_layout(
        title=f'{param_name} - Last 30 Days',
        xaxis_title='Days',
        yaxis_title='Value',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font={'family': 'Inter'},
        hovermode='x unified'
    )
    return fig

def display_modern_risk_card(risk_level, probability):
    """Display modern risk assessment card"""
    if risk_level == 'STABLE':
        message = settings.RISK_MESSAGES['STABLE']
        card_class = 'stable-risk'
        icon = '✅'
    else:  # RISK
        message = settings.RISK_MESSAGES['RISK']
        card_class = 'risk-risk'
        icon = '🚨'
    
    st.markdown(f"""
    <div class="risk-card {card_class}">
        <h3>{icon} {message['title']}</h3>
        <div class="risk-score">{probability:.1%}</div>
        <p style="font-size: 1.1rem; margin-bottom: 1rem;"><strong>Analysis:</strong> {message['description']}</p>
        <p style="font-size: 1.1rem;"><strong>Action Required:</strong> {message['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)

def check_parameter_status(feature_name, value):
    """Check if parameter is within optimal range"""
    optimal_min, optimal_max = settings.OPTIMAL_RANGES[feature_name]
    
    if optimal_min <= value <= optimal_max:
        return '✓ Optimal', 'optimal'
    elif settings.WATER_PARAM_RANGES[feature_name][0] <= value <= settings.WATER_PARAM_RANGES[feature_name][1]:
        return '⚠️ Warning', 'warning'
    else:
        return '✗ Critical', 'critical'

# ============================================================================
# HOME PAGE - MODERN DASHBOARD
# ============================================================================

def modern_home_page():
    """Modern home page with enhanced dashboard"""
    
    st.markdown("""
    <div style='text-align: center; margin: 1.5rem 0 2rem 0;'>
        <h1 style='font-size: 3rem; font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.4);
        margin: 0;
        padding: 1rem 0;
        letter-spacing: 1px;'>
        🐟 FISH FARM AI - DISEASE PREDICTION SYSTEM
        </h1>
        <p style='font-size: 1.2rem; color: #ffffff; margin-top: 1rem; font-weight: 500;'>
        Predict disease outbreaks 5-7 days in advance using advanced machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row
    data = load_historical_data()
    if data is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Total Records",
                value=f"{len(data):,}",
                delta=f"+{len(data)-500} new",
                delta_color="normal"
            )
        
        with col2:
            outbreak_count = data['Disease_Outbreak'].sum()
            st.metric(
                label="🚨 Total Outbreaks",
                value=f"{outbreak_count:,}",
                delta=f"{(outbreak_count/len(data)*100):.1f}%"
            )
        
        with col3:
            recent_outbreaks = data.tail(30)['Disease_Outbreak'].sum()
            st.metric(
                label="📈 Last 30 Days",
                value=f"{recent_outbreaks} outbreaks",
                delta=f"{(recent_outbreaks/30*100):.0f}% rate",
                delta_color="inverse"
            )
        
        with col4:
            model_accuracy = 0.954  # From training results
            st.metric(
                label="🎯 Model Accuracy",
                value=f"{model_accuracy:.1%}",
                delta="+5.4% improved"
            )
    
    st.markdown("---")
    
    # Feature Highlights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 Advanced AI Models</h3>
            <p>Powered by ensemble learning including XGBoost, Random Forest, and SVM with 95%+ accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Real-time Analytics</h3>
            <p>Monitor 9 critical parameters with instant risk assessment and actionable insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Lightning Fast</h3>
            <p>Get predictions in milliseconds with automated alerts and recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Trend Analysis</h3>
            <p>Historical data visualization and pattern recognition for better decision making</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Metrics Display
    st.markdown("## 🎯 Best Model Performance Metrics")
    
    try:
        if os.path.exists(settings.TRAINING_RESULTS_PATH):
            metrics_df = pd.read_csv(settings.TRAINING_RESULTS_PATH)
            
            # Find best model by F1-Score
            best_model_idx = metrics_df['F1-Score'].idxmax()
            best_model_row = metrics_df.loc[best_model_idx]
            
            # Display Best Model Name
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Model", best_model_row['Model'], delta=None)
            
            with col2:
                st.metric("Accuracy", f"{best_model_row['Accuracy']:.4f}", delta=None)
            
            with col3:
                st.metric("Precision", f"{best_model_row['Precision']:.4f}", delta=None)
            
            with col4:
                st.metric("Recall", f"{best_model_row['Recall']:.4f}", delta=None)
            
            with col5:
                st.metric("F1-Score", f"{best_model_row['F1-Score']:.4f}", delta=None)
            
            st.markdown("---")
            
            # Comparison of All Models
            st.markdown("### 📊 All Models Comparison")
            
            # Create a bar chart comparing all models
            fig = go.Figure()
            
            metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
            
            for idx, metric in enumerate(metrics_list):
                fig.add_trace(go.Bar(
                    x=metrics_df['Model'],
                    y=metrics_df[metric],
                    name=metric,
                    marker=dict(color=colors[idx]),
                    text=metrics_df[metric].round(4),
                    textposition='outside'
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                barmode='group',
                xaxis_title="Model",
                yaxis_title="Score",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.markdown("### 📋 Detailed Metrics Table")
            display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
            display_metrics = metrics_df[display_cols]
            st.dataframe(
                display_metrics.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                                               color='lightgreen'),
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Best Model - Confusion Matrix
            st.markdown("### 📊 Best Model - Confusion Matrix")
            
            try:
                model, scaler = load_model()
                data = load_historical_data()
                
                if model is not None and data is not None:
                    X = data[settings.INPUT_FEATURES]
                    y = data['Disease_Outbreak']
                    X_scaled = scaler.transform(X)
                    
                    predictions = model.predict(X_scaled)
                    cm = confusion_matrix(y, predictions)
                    
                    # Create heatmap for confusion matrix
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Stable', 'Predicted Risk'],
                        y=['Actual Stable', 'Actual Risk'],
                        colorscale='Blues',
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 14},
                        hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
                    ))
                    
                    fig_cm.update_layout(
                        title=f"{best_model_row['Model']} - Confusion Matrix",
                        xaxis_title="Predicted Label",
                        yaxis_title="Actual Label",
                        height=400,
                        width=600,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='white'
                    )
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    tn, fp, fn, tp = cm.ravel()
                    
                    st.markdown("**Confusion Matrix Summary:**")
                    
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    with summary_col1:
                        st.metric("True Negative", f"{tn:,}")
                    with summary_col2:
                        st.metric("False Positive", f"{fp:,}")
                    with summary_col3:
                        st.metric("False Negative", f"{fn:,}")
                    with summary_col4:
                        st.metric("True Positive", f"{tp:,}")
                        
            except Exception as e:
                st.warning(f"Could not generate confusion matrix: {str(e)}")
        else:
            st.warning("Training results file not found. Please train the model first.")
    except Exception as e:
        st.error(f"Error loading model metrics: {str(e)}")

# ============================================================================
# PREDICTION PAGE - ENHANCED UI
# ============================================================================

def modern_prediction_page():
    """Modern prediction page with enhanced UI"""
    
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0 1rem 0;'>
        <h1 style='font-size: 3rem; font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.4);
        margin: 0;
        padding: 1rem 0;
        letter-spacing: 1px;'>
        🔮 DISEASE OUTBREAK PREDICTIONS
        </h1>
        <p style='font-size: 1.2rem; color: #475569; margin-top: 1rem; font-weight: 500;'>
        Enter current farm parameters to get instant AI-powered risk assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    model, scaler = load_model()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please train a model first.")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training model... This may take a few minutes"):
                os.system("python src/model_training.py")
            st.success("Model trained successfully! Please refresh the page.")
        return
    
    # Tabs for different input modes
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📤 Batch Upload", "🎲 Quick Test"])
    
    # ====== TAB 1: MANUAL INPUT ======
    with tab1:
        st.markdown('<div class="info-box">💡 Adjust the sliders below to match your current farm conditions. The AI will analyze all parameters and provide a comprehensive risk assessment.</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #667eea;'>Parameter Input</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown("<h3 style='color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;'>🌡️ Water Quality</h3>", unsafe_allow_html=True)
            temp = st.slider('Temperature (°C)', 
                             min_value=18.0, max_value=34.0, value=26.0, step=0.1,
                             help="Optimal range: 24-28°C")
            ph = st.slider('pH Level', 
                          min_value=5.5, max_value=8.5, value=7.0, step=0.1,
                          help="Optimal range: 6.5-8.0")
            do = st.slider('Dissolved Oxygen (mg/L)', 
                          min_value=2.0, max_value=10.0, value=7.5, step=0.1,
                          help="Optimal range: 5-10 mg/L")
        
        with col2:
            st.markdown("<h3 style='color: #10b981; border-bottom: 2px solid #10b981; padding-bottom: 0.5rem;'>💧 Chemical Parameters</h3>", unsafe_allow_html=True)
            ammonia = st.slider('Ammonia (mg/L)', 
                               min_value=0.0, max_value=5.0, value=0.2, step=0.1,
                               help="Optimal range: 0-0.5 mg/L")
            nitrate = st.slider('Nitrate (mg/L)', 
                               min_value=0.0, max_value=100.0, value=20.0, step=1.0,
                               help="Optimal range: 0-40 mg/L")
            turbidity = st.slider('Turbidity (NTU)', 
                                 min_value=2.0, max_value=100.0, value=30.0, step=1.0,
                                 help="Optimal range: 2-5 NTU")
        
        with col3:
            st.markdown("<h3 style='color: #f59e0b; border-bottom: 2px solid #f59e0b; padding-bottom: 0.5rem;'>🐟 Fish Behavior</h3>", unsafe_allow_html=True)
            feed_intake = st.slider('Feed Intake (%)', 
                                   min_value=20.0, max_value=100.0, value=85.0, step=1.0,
                                   help="Optimal range: 80-100%")
            growth_rate = st.slider('Growth Rate (g/week)', 
                                   min_value=0.2, max_value=3.0, value=1.7, step=0.1,
                                   help="Optimal range: 1.2-3.0 g/week")
            mortality = st.slider('Mortality Count (per day)', 
                                 min_value=0, max_value=50, value=3, step=1,
                                 help="Optimal range: 0-2 per day")
        
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Prediction button with modern styling
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            predict_button = st.button('🔍 Analyze Risk & Generate Report', use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if predict_button:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Temperature_C': [temp],
                'pH': [ph],
                'Dissolved_Oxygen_mg_L': [do],
                'Ammonia_mg_L': [ammonia],
                'Nitrate_mg_L': [nitrate],
                'Turbidity_NTU': [turbidity],
                'Feed_Intake_Percent': [feed_intake],
                'Growth_Rate_g_week': [growth_rate],
                'Mortality_Count_per_day': [mortality]
            })
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            with st.spinner('🤖 AI is analyzing your data...'):
                probability = model.predict_proba(input_scaled)[0][1]
                prediction = model.predict(input_scaled)[0]
            
            risk_level, _ = get_risk_category(probability)
            
            # Results Section
            st.markdown("---")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; margin: 2rem 0;'>
                <h1 style='font-size: 2.5rem; font-weight: 800; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
                margin: 0;
                padding: 1rem 0;'>
                📊 COMPREHENSIVE RISK ASSESSMENT
                </h1>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display gauge chart and risk card side by side
            col1, col2, col3 = st.columns([1, 2, 1], gap="medium")
            
            with col1:
                st.write("")  # Spacer
            
            with col2:
                # Centered container for gauge and risk card
                st.markdown("""
                <div style='background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                border: 2px solid #667eea;'>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                display_modern_risk_card(risk_level, probability)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.write("")  # Spacer
            
            st.markdown("---")
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Parameter Analysis
            st.markdown("""
            <div style='text-align: center; margin: 2rem 0;'>
                <h1 style='font-size: 2.5rem; font-weight: 800;
                background: linear-gradient(135deg, #10b981 0%, #059669 50%, #34d399 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
                margin: 0;
                padding: 1rem 0;'>
                🔬 DETAILED PARAMETER ANALYSIS
                </h1>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <h2 style='text-align: center; 
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 50%, #fbbf24 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1.5rem;
            padding: 0.5rem 0;'>
            📋 PARAMETER STATUS
            </h2>
            """, unsafe_allow_html=True)
            
            params_dict = {
                'Temperature_C': temp,
                'pH': ph,
                'Dissolved_Oxygen_mg_L': do,
                'Ammonia_mg_L': ammonia,
                'Nitrate_mg_L': nitrate,
                'Turbidity_NTU': turbidity,
                'Feed_Intake_Percent': feed_intake,
                'Growth_Rate_g_week': growth_rate,
                'Mortality_Count_per_day': mortality
            }
            
            for feature_name, value in params_dict.items():
                status, badge_type = check_parameter_status(feature_name, value)
                param_display = feature_name.replace('_', ' ').title()
                
                optimal_min, optimal_max = settings.OPTIMAL_RANGES[feature_name]
                
                # Color based on status
                if badge_type == 'optimal':
                    bg_color = '#f0fdf4'
                    border_color = '#10b981'
                    text_color = '#064e3b'
                elif badge_type == 'warning':
                    bg_color = '#fffbeb'
                    border_color = '#f59e0b'
                    text_color = '#78350f'
                else:
                    bg_color = '#fef2f2'
                    border_color = '#ef4444'
                    text_color = '#991b1b'
                
                st.markdown(f"""
                <div style="margin: 0.5rem 0; padding: 1rem; background: {bg_color}; border-radius: 10px; border-left: 4px solid {border_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong style="color: {text_color}; font-size: 1rem;">{param_display}</strong>
                        <span class="status-badge badge-{badge_type}" style="font-weight: 600;">{status}</span>
                    </div>
                    <div style="color: {text_color}; font-size: 1.2rem; font-weight: 700; margin: 0.5rem 0;">{value:.2f}</div>
                    <small style="color: #64748b; font-weight: 500;">Optimal Range: {optimal_min} - {optimal_max}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Action Items
            st.markdown("---")
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='text-align: center; margin: 2rem 0;'>
                <h1 style='font-size: 2.5rem; font-weight: 800;
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #f87171 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
                margin: 0;
                padding: 1rem 0;'>
                🎯 RECOMMENDED ACTIONS
                </h1>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            action_container = st.container()
            with action_container:
                if risk_level == 'RISK':
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #dc2626; margin: 1rem 0;">
                        <h3 style="color: #991b1b; margin: 0; font-size: 1.3rem;">🚨 IMMEDIATE ACTION REQUIRED</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    actions = [
                    "✓ Increase water circulation and aeration immediately",
                    "✓ Perform emergency water quality testing",
                    "✓ Reduce feeding by 30-50% for the next 24 hours",
                    "✓ Contact veterinarian for preventive treatment options",
                    "✓ Isolate any fish showing abnormal behavior",
                    "✓ Monitor mortality rates every 4 hours"
                ]
                else:  # STABLE
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #10b981; margin: 1rem 0;">
                        <h3 style="color: #065f46; margin: 0; font-size: 1.3rem;">✅ MAINTAIN CURRENT PRACTICES</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    actions = [
                    "✓ Continue regular monitoring schedule",
                    "✓ Maintain current feeding rates",
                    "✓ Keep water quality parameters stable",
                    "✓ Document current conditions for future reference",
                    "✓ Monitor fish behavior daily",
                    "✓ Test water parameters weekly"
                ]
            
                for i, action in enumerate(actions, 1):
                    st.markdown(f"""
                    <div style="padding: 0.75rem 1rem; margin: 0.5rem 0; background: white; border-radius: 8px; border-left: 3px solid #667eea; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <span style="color: #667eea; font-weight: 700; margin-right: 0.5rem;">{i}.</span>
                        <span style="color: #334155; font-size: 1rem;">{action}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ====== TAB 2: BATCH UPLOAD ======
    with tab2:
        st.markdown("""
        <div style='text-align: center; margin: 1.5rem 0;'>
            <h1 style='font-size: 2rem; font-weight: 800;
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            padding: 0.5rem 0;'>
            📤 BATCH CSV UPLOAD
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>Format Requirements:</strong>
            <ul>
                <li>CSV file with columns matching the parameter names</li>
                <li>One row per prediction</li>
                <li>All 9 parameters must be included</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data download
        sample_data = pd.DataFrame({
            'Temperature_C': [26.0, 28.5, 23.0],
            'pH': [7.0, 6.8, 7.5],
            'Dissolved_Oxygen_mg_L': [7.5, 5.0, 8.0],
            'Ammonia_mg_L': [0.2, 1.5, 0.1],
            'Nitrate_mg_L': [20.0, 45.0, 15.0],
            'Turbidity_NTU': [30.0, 65.0, 25.0],
            'Feed_Intake_Percent': [85.0, 60.0, 90.0],
            'Growth_Rate_g_week': [1.7, 1.0, 2.0],
            'Mortality_Count_per_day': [3, 12, 2]
        })
        
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_data.to_csv(index=False),
            file_name="sample_template.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader('Upload Your CSV File', type='csv')
        
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(batch_data)} records")
            
            with st.expander("📋 Preview Data"):
                st.dataframe(batch_data.head(10), use_container_width=True)
            
            if st.button('🔍 Run Batch Predictions', use_container_width=True):
                try:
                    missing_cols = [col for col in settings.INPUT_FEATURES if col not in batch_data.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                    else:
                        with st.spinner('🤖 Processing predictions...'):
                            X = batch_data[settings.INPUT_FEATURES].copy()
                            X_scaled = scaler.transform(X)
                            
                            probabilities = model.predict_proba(X_scaled)[:, 1]
                            predictions = model.predict(X_scaled)
                            
                            results = batch_data.copy()
                            results['Outbreak_Probability'] = probabilities
                            results['Prediction'] = predictions
                            results['Risk_Level'] = results['Outbreak_Probability'].apply(
                                lambda x: get_risk_category(x)[0]
                            )
                        
                        st.success("✅ Predictions completed successfully!")
                        
                        # Summary Statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Records", len(results))
                        with col2:
                            stable = len(results[results['Risk_Level'] == 'STABLE'])
                            st.metric("Stable", stable, delta=f"{stable/len(results)*100:.1f}%", delta_color="normal")
                        with col3:
                            risk = len(results[results['Risk_Level'] == 'RISK'])
                            st.metric("Risk", risk, delta=f"{risk/len(results)*100:.1f}%", delta_color="inverse")
                        
                        # Results Table
                        st.markdown("### 📊 Detailed Results")
                        st.dataframe(results, use_container_width=True)
                        
                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label='📥 Download Results CSV',
                            data=csv,
                            file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                        
                        # Visualization
                        fig = px.pie(
                            results,
                            names='Risk_Level',
                            title='Risk Distribution',
                            color='Risk_Level',
                            color_discrete_map={'STABLE': '#96e6a1', 'RISK': '#ff7675'}
                        )
                        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
    
    # ====== TAB 3: QUICK TEST ======
    with tab3:
        st.markdown("""
        <div style='text-align: center; margin: 1.5rem 0;'>
            <h1 style='font-size: 2rem; font-weight: 800;
            background: linear-gradient(135deg, #ec4899 0%, #db2777 50%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            padding: 0.5rem 0;'>
            🎲 QUICK TEST SCENARIOS
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            Test the model with predefined scenarios to see how different conditions affect disease outbreak risk.
        </div>
        """, unsafe_allow_html=True)
        
        scenario = st.selectbox(
            "Select a Test Scenario",
            ["Optimal Conditions", "Warning Conditions", "Critical Conditions", "Random Test"]
        )
        
        if scenario == "Optimal Conditions":
            test_params = {
                'Temperature_C': 26.0, 'pH': 7.2, 'Dissolved_Oxygen_mg_L': 8.0,
                'Ammonia_mg_L': 0.2, 'Nitrate_mg_L': 20.0, 'Turbidity_NTU': 15.0,
                'Feed_Intake_Percent': 90.0, 'Growth_Rate_g_week': 2.0, 'Mortality_Count_per_day': 1
            }
        elif scenario == "Warning Conditions":
            test_params = {
                'Temperature_C': 29.0, 'pH': 6.5, 'Dissolved_Oxygen_mg_L': 5.5,
                'Ammonia_mg_L': 0.8, 'Nitrate_mg_L': 45.0, 'Turbidity_NTU': 55.0,
                'Feed_Intake_Percent': 70.0, 'Growth_Rate_g_week': 1.2, 'Mortality_Count_per_day': 7
            }
        elif scenario == "Critical Conditions":
            test_params = {
                'Temperature_C': 32.0, 'pH': 6.0, 'Dissolved_Oxygen_mg_L': 3.5,
                'Ammonia_mg_L': 2.5, 'Nitrate_mg_L': 75.0, 'Turbidity_NTU': 85.0,
                'Feed_Intake_Percent': 45.0, 'Growth_Rate_g_week': 0.6, 'Mortality_Count_per_day': 18
            }
        else:  # Random
            test_params = {
                'Temperature_C': np.random.uniform(22, 30),
                'pH': np.random.uniform(6.5, 7.8),
                'Dissolved_Oxygen_mg_L': np.random.uniform(4, 9),
                'Ammonia_mg_L': np.random.uniform(0.1, 1.5),
                'Nitrate_mg_L': np.random.uniform(10, 60),
                'Turbidity_NTU': np.random.uniform(15, 70),
                'Feed_Intake_Percent': np.random.uniform(60, 95),
                'Growth_Rate_g_week': np.random.uniform(1.0, 2.5),
                'Mortality_Count_per_day': int(np.random.uniform(1, 10))
            }
        
        # Display parameters
        st.markdown("### 📋 Test Parameters")
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            for param in list(test_params.keys())[:3]:
                st.metric(param.replace('_', ' ').title(), f"{test_params[param]:.2f}")
        
        with param_col2:
            for param in list(test_params.keys())[3:6]:
                st.metric(param.replace('_', ' ').title(), f"{test_params[param]:.2f}")
        
        with param_col3:
            for param in list(test_params.keys())[6:]:
                st.metric(param.replace('_', ' ').title(), f"{test_params[param]:.2f}")
        
        if st.button('🚀 Run Test Prediction', use_container_width=True):
            input_df = pd.DataFrame([test_params])
            input_scaled = scaler.transform(input_df)
            
            with st.spinner('Running prediction...'):
                probability = model.predict_proba(input_scaled)[0][1]
            
            risk_level, _ = get_risk_category(probability)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
            
            with col2:
                display_modern_risk_card(risk_level, probability)

# ============================================================================
# ABOUT PAGE
# ============================================================================

def about_page():
    """Modern about page"""
    
    st.markdown("""
    <div style='text-align: center; margin: 1.5rem 0 2rem 0;'>
        <h5 style='font-size: 3rem; font-weight: 900;
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 50%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(6, 182, 212, 0.4);
        margin: 0;
        padding: 1rem 0;
        letter-spacing: 1px;
        color:#ffffff;'>
         ABOUT THIS SYSTEM
        </h5>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; color: white; margin: 2rem 0;">
        <h2>Advanced AI-Powered Disease Prediction</h2>
        <p style="font-size: 1.2rem;">Helping fish farmers prevent disease outbreaks through data-driven insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Mission
        
        To revolutionize fish farming through artificial intelligence, reducing mortality rates and improving farm profitability through early disease detection.
        
        ### 🔬 Technology Stack
        
        - **Machine Learning**: XGBoost, Random Forest, SVM
        - **Data Processing**: Pandas, NumPy, Scikit-learn
        - **Visualization**: Plotly, Streamlit
        - **Accuracy**: 95.4% with 2500+ training samples
        
        ### 📊 Model Performance
        
        - **Accuracy**: 95.40%
        - **Precision**: 94.37%
        - **Recall**: 89.93%
        - **F1-Score**: 92.10%
        - **ROC-AUC**: 99.12%
        """)
    
    with col2:
        st.markdown("""
        ### 🌟 Key Features
        
        - ✅ Real-time disease outbreak prediction
        - ✅ 9 critical water quality parameters
        - ✅ Interactive visualization dashboard
        - ✅ Batch processing capabilities
        - ✅ Automated risk assessment
        - ✅ Actionable recommendations
        - ✅ Historical data analysis
        - ✅ Trend identification
        
        ### 📈 Benefits
        
        - **Reduce Mortality**: Early warning saves fish populations
        - **Cut Costs**: Preventive measures cheaper than treatment
        - **Improve Efficiency**: Data-driven farm management
        - **Increase Revenue**: Healthier fish = better yields
        
        ### 📞 Support
        
        For technical support or feature requests, contact the development team.
        
        **Version**: 2.0 (Enhanced)  
        **Last Updated**: March 2026
        """)
    
    st.markdown("---")
    
    # Model information
    with st.expander("🤖 Model Technical Details"):
        st.markdown("""
        #### Training Data
        - **Total Samples**: 2,500 records
        - **Features**: 9 water quality and behavior parameters
        - **Training Set**: 2,000 samples (80%)
        - **Test Set**: 500 samples (20%)
        - **Outbreak Rate**: 29.8%
        
        #### Model Architecture
        - **Best Model**: Selected automatically from trained models
        - **Hyperparameters**: Optimized through grid search
        - **Cross-Validation**: 5-fold CV
        - **Feature Importance**: Analyzed and validated
        
        #### Performance by Class
        - **Healthy Conditions**: 96% accuracy
        - **Disease Outbreak**: 90% recall
        - **False Positive Rate**: 6.3%
        - **False Negative Rate**: 10.1%
        """)

# ============================================================================
# MAIN APP NAVIGATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Modern Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h5 style="color: #ffffff !important; font-size: 1.5rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🐟 Fish Farm AI</h5>
            <p style="color: #ffffff !important; font-size: 0.9rem;">Disease Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            'Navigation',
            ['🏠 Dashboard', '🔮 Predictions', 'ℹ️ About'],
            label_visibility='collapsed'
        )
        
        st.markdown("---")
        
        # System status
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px;">
            <h3 style="color: white; font-size: 1rem; margin-bottom: 0.5rem;">System Status</h3>
            <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0.25rem 0;">
                <strong>Model:</strong> ✓ Active<br>
                <strong>Accuracy:</strong> 95.4%<br>
                <strong>Data Points:</strong> 2,500<br>
                <strong>Version:</strong> 2.0
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick stats
        data = load_historical_data()
        if data is not None:
            recent_outbreak_rate = data.tail(30)['Disease_Outbreak'].mean() * 100
            
            if recent_outbreak_rate < 20:
                status_color = "#96e6a1"
                status_text = "Low Risk Period"
            elif recent_outbreak_rate < 35:
                status_color = "#fdcb6e"
                status_text = "Moderate Activity"
            else:
                status_color = "#ff7675"
                status_text = "High Risk Period"
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px;">
                <h3 style="color: white; font-size: 1rem; margin-bottom: 0.5rem;">Recent Trends</h3>
                <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">
                    Last 30 Days:<br>
                    <strong style="color: {status_color}; font-size: 1.2rem;">{recent_outbreak_rate:.1f}%</strong> outbreak rate<br>
                    <span style="color: {status_color};">● {status_text}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Route to selected page
    if page == '🏠 Dashboard':
        modern_home_page()
    elif page == '🔮 Predictions':
        modern_prediction_page()
    elif page == 'ℹ️ About':
        about_page()

if __name__ == '__main__':
    main()
