import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import pickle

# Set page config
st.set_page_config(
    page_title="Walmart Sales Prediction",
    page_icon="🔮",
    layout="wide"
)

# Initialize session state for model storage
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'model_source' not in st.session_state:
    st.session_state.model_source = None

# ============================================================================
# CORE FUNCTIONS FOR MODEL LOADING AND PREDICTION
# ============================================================================

def load_default_model(model_type):
    """Load default model from models/default/ directory"""
    model_path = f"models/default/{model_type}.pkl"
    
    if not os.path.exists(model_path):
        return None, f"Default {model_type} model not found at {model_path}"
    
    try:
        # Load both model types using joblib for consistency
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def load_uploaded_model(uploaded_file, model_type):
    """Load model from uploaded file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Load both model types using joblib for consistency
        model = joblib.load(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return model, None
    except Exception as e:
        # Clean up if error occurs
        try:
            os.unlink(tmp_path)
        except:
            pass
        return None, f"Invalid model file. Please check format or retrain."

def predict_next_4_weeks(model, model_type):
    """Predict next 4 weeks of sales"""
    # Generate dates for next 4 weeks
    today = datetime.now()
    dates = [today + timedelta(weeks=i) for i in range(1, 5)]
    
    try:
        # Handle both display names and file names for model types
        if model_type in ["Auto ARIMA", "auto_arima"]:
            predictions = model.predict(n_periods=4)
        elif model_type in ["Exponential Smoothing (Holt-Winters)", "exponential_smoothing"]:
            predictions = model.forecast(4)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return predictions, dates, None
    except Exception as e:
        return None, None, f"Error generating predictions: {str(e)}"

# ============================================================================
# STREAMLIT APP INTERFACE
# ============================================================================

def main():
    # App title and description
    st.title("🔮 Walmart Sales Prediction")
    st.markdown("""
    This app generates sales forecasts for the next 4 weeks using trained time series models.
    
    **You can:**
    - Use pre-loaded default models (recommended)
    - Upload your own trained models
    - View interactive forecasts
    - Download prediction results
    """)
    
    # Model selection section
    st.header("🤖 Model Selection")
    
    # Tabs for default vs uploaded models
    tab1, tab2 = st.tabs(["Default Models", "Upload Model"])
    
    with tab1:
        st.subheader("Use Default Models")
        
        # Only showing Exponential Smoothing (Holt-Winters Method) - Auto ARIMA removed
        if st.button("Load Exponential Smoothing (Holt-Winters Method) Model", use_container_width=True):
            model, error = load_default_model("exponential_smoothing")
            if error:
                st.error(error)
            else:
                st.session_state.current_model = model
                st.session_state.model_type = "Exponential Smoothing (Holt-Winters)"
                st.session_state.model_source = "Default"
                st.success("✅ Exponential Smoothing (Holt-Winters Method) model loaded successfully!")
    
    with tab2:
        st.subheader("Upload Custom Model")
        
        # Model type selection for upload
        model_type = st.selectbox(
            "Select model type:",
            ["Auto ARIMA", "Exponential Smoothing (Holt-Winters)"],
            key="upload_model_type"
        )
        
        # Display a more descriptive label for exponential smoothing
        if model_type == "exponential_smoothing":
            st.info("Exponential Smoothing (Holt-Winters Method)")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload model file (.pkl)", 
            type=["pkl"],
            key="model_uploader"
        )
        
        if uploaded_file:
            button_text = "Load Uploaded Model"
            if model_type == "exponential_smoothing":
                button_text = "Load Uploaded Exponential Smoothing (Holt-Winters Method) Model"
            elif model_type == "auto_arima":
                button_text = "Load Uploaded Auto ARIMA Model"
                
                        if st.button(button_text, use_container_width=True):
                model, error = load_uploaded_model(uploaded_file, model_type)
                if error:
                    st.error(error)
                else:
                    st.session_state.current_model = model
                    st.session_state.model_type = model_type
                    st.session_state.model_source = "Uploaded"
                    
                    st.success(f"✅ {model_type} model loaded successfully!")
    
    # Display current model info
    if st.session_state.current_model is not None:
        model_display_name = st.session_state.model_type
        st.info(f"**Current Model:** {model_display_name} ({st.session_state.model_source})")
    else:
        st.warning("No model loaded. Please select a model to make predictions.")
    
    # Prediction section
    st.header("📈 Generate Predictions")
    
    if st.session_state.current_model is not None:
        if st.button("Generate 4-Week Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating predictions..."):
                predictions, dates, error = predict_next_4_weeks(
                    st.session_state.current_model,
                    st.session_state.model_type
                )
                
                if error:
                    st.error(error)
                else:
                    # Create prediction dataframe
                    prediction_df = pd.DataFrame({
                        'Week': [f"Week {i+1}" for i in range(4)],
                        'Date': [date.strftime('%Y-%m-%d') for date in dates],
                        'Weekly_Change': predictions
                    })
                    
                    # Display results
                    st.subheader("📊 Forecast Results")
                    
                    # Create interactive plot
                    fig = go.Figure()
                    
                    # Add bars for positive and negative changes
                    positive_mask = prediction_df['Weekly_Change'] >= 0
                    negative_mask = prediction_df['Weekly_Change'] < 0
                    
                    # Add positive bars
                    fig.add_trace(go.Bar(
                        x=prediction_df.loc[positive_mask, 'Week'],
                        y=prediction_df.loc[positive_mask, 'Weekly_Change'],
                        name='Sales Increase',
                        marker_color='green',
                    ))
                    
                    # Add negative bars
                    fig.add_trace(go.Bar(
                        x=prediction_df.loc[negative_mask, 'Week'],
                        y=prediction_df.loc[negative_mask, 'Weekly_Change'],
                        name='Sales Decrease',
                        marker_color='red',
                    ))
                    
                    fig.update_layout(
                        title='Weekly Sales Change Forecast (Next 4 Weeks)',
                        xaxis_title='Week',
                        yaxis_title='Sales Change ($)',
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )
                    
                    # Add grid
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display data table with formatted values
                    st.subheader("📋 Week-over-Week Sales Changes")
                    
                    # Format the Weekly_Change as currency with sign
                    formatted_df = prediction_df.copy()
                    formatted_df['Weekly_Change'] = formatted_df['Weekly_Change'].apply(
                        lambda x: f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"
                    )
                    
                    # Rename column for clarity
                    formatted_df = formatted_df.rename(columns={
                        'Weekly_Change': 'Sales Change ($)'
                    })
                    
                    # Display the formatted dataframe
                    st.dataframe(formatted_df, use_container_width=True)
                    
                    # Add explanation box
                    st.info("""
                    **Understanding the Forecast:**
                    * These values represent **week-over-week sales changes**, not absolute sales.
                    * **Positive values** indicate sales increases from the previous week.
                    * **Negative values** indicate sales decreases from the previous week.
                    * All values are in dollar amounts.
                    """)
                    
                    # Download section
                    st.subheader("💾 Download Results")
                    
                    # Prepare CSV for download - use original numeric values
                    # But rename column for clarity
                    download_df = prediction_df.rename(columns={
                        'Weekly_Change': 'Sales_Change_$'
                    })
                    csv = download_df.to_csv(index=False).encode('utf-8')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="walmart_sales_changes.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Create JSON version
                        json_str = download_df.to_json(orient='records')
                        st.download_button(
                            label="Download as JSON",
                            data=json_str,
                            file_name="walmart_sales_changes.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        net_change = predictions.sum()
                        st.metric(
                            "Net Sales Change", 
                            f"${net_change:,.2f}" if net_change >= 0 else f"-${abs(net_change):,.2f}",
                            delta=f"{np.sign(net_change) * 100:.1f}%" if net_change != 0 else "0%"
                        )
                    with col2:
                        max_increase = max(predictions.max(), 0)
                        max_week = predictions.argmax() + 1 if max_increase > 0 else None
                        max_label = f"Week {max_week}" if max_week else "None"
                        st.metric("Largest Increase", f"${max_increase:,.2f}", delta=max_label)
                    with col3:
                        max_decrease = min(predictions.min(), 0)
                        min_week = predictions.argmin() + 1 if max_decrease < 0 else None
                        min_label = f"Week {min_week}" if min_week else "None"
                        st.metric("Largest Decrease", f"-${abs(max_decrease):,.2f}", delta=min_label)
                    with col4:
                        positive_weeks = sum(predictions > 0)
                        negative_weeks = sum(predictions < 0)
                        st.metric("Growth Trend", 
                                 f"{positive_weeks} up, {negative_weeks} down",
                                 delta="Positive" if positive_weeks > negative_weeks else 
                                       "Negative" if negative_weeks > positive_weeks else "Neutral")
    
    else:
        st.info("👆 Please load a model first to generate predictions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Walmart Sales Forecasting System © 2025</p>
        <p><small>Built with ❤️ using Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()