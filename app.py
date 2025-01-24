import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from PIL import Image  # Import the Image class from PIL to handle images
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import your existing classes
from dummy import EnhancedHealthMetrics  # Replace with actual module name
from history import HealthHistoryAnalyzer

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# App configuration
st.set_page_config(
    page_title="Health Analytics Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the app's appearance
st.markdown("""
<style>
/* General styles for metric boxes */
.metric-box {
    padding: 20px;
    border-radius: 8px; /* Slightly reduced for a sleeker look */
    margin: 10px 0;
    background: #2A2A40; /* Darker background to match dashboard theme */
    color: #F5F5F5; /* Light text color for readability */
    border-left: 4px solid #4CAF50; /* Vibrant green for accent */
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2); /* Subtle shadow for depth */
}

.metric-box:hover {
    background: #3A3A50; /* Slightly lighter on hover for interactivity */
    border-left-color: #388E3C; /* Darker green on hover for accent */
}


/* Styles for buttons */
.stButton>button {
    background-color: #4CAF50; /* Updated to match primary color */
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    border: none;
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #388E3C; /* Darker green on hover */
}

/* Styles for sidebar */
.css-1d391kg {
    background-color: #2A2A40; /* Darker sidebar background */
    padding: 20px;
    border-radius: 10px;
}

/* Styles for headers */
h1, h2, h3, h4, h5, h6 {
    color: #4CAF50; /* Updated to vibrant green */
}

/* Styles for success and error messages */
.stSuccess {
    background-color: #d4edda; /* Light green for success messages */
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #c3e6cb;
}

.stError {
    background-color: #f8d7da; /* Light red for error messages */
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #f5c6cb;
}

/* Styles for input fields */
.stTextInput>div>div>input, .stNumberInput>div>div>input {
    border-radius: 8px;
    border: 1px solid #4CAF50; /* Updated to match primary accent color */
    padding: 10px;
    font-size: 14px;
}

/* Styles for the sidebar title */
.css-1d391kg h1 {
    color: #4CAF50; /* Updated to vibrant green */
    font-size: 24px;
    margin-bottom: 20px;
}

/* Styles for the main content area */
.block-container {
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)




def safe_get(data, keys, default=None):
    """Safely get nested dictionary values"""
    for key in keys:
        try:
            data = data[key]
        except (KeyError, TypeError):
            return default
    return data

def safe_length_check(item):
    """Safely check length of an item"""
    return len(item) if item and isinstance(item, (list, dict, str)) else 0

def display_metric(label, value):
    """Safe metric display with null checks"""
    if value is None:
        value = "N/A"
    st.markdown(f"""
    <div class="metric-box">
    <h3 style="margin: 0 0 10px 0; color: #F5F5F5;">{label}</h3>
    <p style="font-size: 36px; margin: 0; font-weight: 700; color: #4CAF50;">{value}</p>
    </div>
    """, unsafe_allow_html=True)

def validate_report(report):
    """Ensure report has required structure"""
    required_keys = {
        'basic_info': ['age', 'bmi'],
        'health_status': ['current_risk', 'scores', 'trend', 'anomalies'],
        'recommendations': ['do', "don't"]
    }
    
    if not isinstance(report, dict):
        return False
    
    for section, keys in required_keys.items():
        if section not in report:
            return False
        for key in keys:
            if key not in report[section]:
                return False
    return True


def show_insights(person_id):
    analyzer = HealthHistoryAnalyzer(person_id)
    insights_data = analyzer.generate_insights()
    
    # Display Metrics
    st.subheader("Key Health Metrics")
    cols = st.columns(4)
    metrics = insights_data['metrics']
    
    with cols[0]:
        display_metric("Avg Heart Rate", metrics['avg_heart_rate'])
    with cols[1]:
        display_metric("Avg Blood Pressure", metrics['blood_pressure'])
    with cols[2]:
        display_metric("Avg Glucose", metrics['avg_glucose'])
    with cols[3]:
        display_metric("Avg Cholesterol", metrics['total_cholesterol'])

    
    # Show Interactive Plot
    st.plotly_chart(insights_data['plots'], use_container_width=True)
    
    # Display Insights
    st.subheader("Clinical Insights")
    for insight in insights_data['insights']:
        st.info(f"📌 {insight}")
    return metrics

def plot_risk_trends(historical_data):
    """Plot historical risk trends in a 2x2 grid using Plotly"""
    # Create 2x2 subplot grid with individual titles
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Cluster Risk", "Heart Risk", 
                        "Diabetic Risk", "Overall Risk"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Helper function to create smoothed traces
    def create_trace(data, name, color):
        return go.Scatter(
            x=data['date'],
            y=data[name.lower().replace(' ', '_')],
            mode='lines+markers',
            line=dict(shape='spline', width=2.5, color=color),
            marker=dict(size=10, color=color, line=dict(width=1, color='DarkSlateGrey')),
            name=name
        )

    # Add traces to subplots
    traces = [
        (create_trace(historical_data, 'Cluster Risk', '#1f77b4'), 1, 1),
        (create_trace(historical_data, 'Heart Risk', '#ff7f0e'), 1, 2),
        (create_trace(historical_data, 'Diabetic Risk', '#2ca02c'), 2, 1),
        (create_trace(historical_data, 'Overall Risk', '#d62728'), 2, 2)
    ]

    for trace, row, col in traces:
        fig.add_trace(trace, row=row, col=col)

    # Add risk bands to all subplots
    risk_bands = [
        (0, 0.2, "green", "Low Risk"),
        (0.2, 0.55, "orange", "Moderate Risk"),
        (0.55, 1.0, "red", "High Risk")
    ]

    for row in [1, 2]:
        for col in [1, 2]:
            for y0, y1, color, text in risk_bands:
                fig.add_hrect(
                    y0=y0, y1=y1, 
                    fillcolor=color, 
                    opacity=0.1,
                    line_width=0,
                    annotation_text=text if row == 1 and col == 1 else "",
                    annotation_position="inside top left",
                    row=row, col=col
                )

    # Update layout settings
    fig.update_layout(
        title_text="Risk Trends Analysis - Last 5 Days",
        title_x=0.35,
        title_font_size=32,
        title_font_color="#4CAF50",
        showlegend=False,
        margin=dict(t=100),
        width=1400,
        height=900
    )

    # Axis formatting
    fig.update_xaxes(title_text="Date", row=2, col=1, title_font_size=14)
    fig.update_xaxes(title_text="Date", row=2, col=2, title_font_size=14)
    fig.update_yaxes(
        title_text="Risk Score", 
        range=[-0.05, 1.05],  # Extended y-axis range
        title_font_size=14,
        tickfont_size=12,
        row=1, col=1
    )
    fig.update_yaxes(range=[-0.05, 1.05], row=1, col=2)  # Extended y-axis range
    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)  # Extended y-axis range
    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=2)  # Extended y-axis range

    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def display_historical_data(report):
    """Display historical data from the report"""
    st.subheader("Risk Trends for Last 5 instance")
    
    # Convert last 5 instances status to a DataFrame for better visualization
    historical_data = pd.DataFrame(report['last_5_instances_status'])
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    historical_data.set_index('date', inplace=True)
    
    # Display the historical data table
    st.dataframe(historical_data.style.format({
        'cluster_risk': '{:.2%}',
        'heart_risk': '{:.2%}',
        'diabetic_risk': '{:.2%}',
        'overall_risk': '{:.2%}'
    }))
    
    # Plot historical trends using Plotly
    plot_risk_trends(historical_data.reset_index())

def main():
    try:
        st.title("🏥 Health Analytics Dashboard")
        
        # User input
        person_id = st.sidebar.number_input("Patient ID", min_value=1, max_value=6, value=1)
        
        if st.sidebar.button("Analyze"):
            with st.spinner('Analyzing...'):
                try:
                    # Add connection error handling
                    try:
                        analyzer = EnhancedHealthMetrics(person_id)
                    except ConnectionError as conn_err:
                        st.error(f"Connection Error: {conn_err}")
                        st.error("Unable to connect to data source. Please check your network connection.")
                        return
                    except Exception as init_err:
                        st.error(f"Initialization Error: {init_err}")
                        return

                    # Defensive programming for report generation
                    report = analyzer.generate_report()
                    
                    # Enhanced null/empty checks
                    if not report or not isinstance(report, dict):
                        st.error("No valid report generated")
                        return
                        
                    # Core metrics display with additional checks
                    st.subheader("Core Health Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_metric("Age", safe_get(report, ['basic_info', 'age'], 'N/A'))
                        display_metric("BMI", safe_get(report, ['basic_info', 'bmi'], 'N/A'))
                        
                    with col2:
                        display_metric("Current Risk", safe_get(report, ['health_status', 'current_risk'], 'N/A'))
                        trend_value = safe_get(report, ['health_status', 'trend'], 0)
                        display_metric("Risk Trend", f"{trend_value:.1f}%" if trend_value is not None else 'N/A')
                    
                    display_historical_data(report)
                    # Recommendations with expanded null checks
                    st.subheader("Recommendations")
                    recs = safe_get(report, ['recommendations'], {"do": [], "don't": []})
                    
                    # Ensure recs is a dictionary with 'do' and 'don't' keys
                    recs = {
                        'do': recs.get('do', []) if isinstance(recs, dict) else [],
                        "don't": recs.get("don't", []) if isinstance(recs, dict) else []
                    }
                    
                    if safe_length_check(recs['do']) or safe_length_check(recs["don't"]):
                        cols = st.columns(2)
                        with cols[0]:
                            st.write("**Do**")
                            for item in recs.get('do', []):
                                st.success(f"- {item}")
                        with cols[1]:
                            st.write("**Don't**")
                            for item in recs.get("don't", []):
                                st.error(f"- {item}")
                    else:
                        st.info("No specific recommendations available")
                        
                    # Anomaly detection with safe access and length check
                    st.subheader("Anomaly Detection")
                    anomalies = safe_get(report, ['health_status', 'anomalies'], [])
                    if safe_length_check(anomalies):
                        st.warning(f"⚠️ Anomalies detected on: {', '.join(map(str, anomalies))}")
                    else:
                        st.success("✅ No anomalies detected")

                    
                    st.subheader("Insights")
                    # insights = generate_insights(person_id)
                    # st.info(insights)
                    # In your main app code where you display insights:
                    insight_data = show_insights(person_id)
                    

                except Exception as e:
                    logging.error(f"Analysis failed: {str(e)}")
                    st.error(f"Analysis failed: Unexpected error occurred. {str(e)}")

    except Exception as main_e:
        logging.error(f"App crashed: {str(main_e)}")
        st.error("Critical error occurred. Please check logs.")

if __name__ == "__main__":
    main()