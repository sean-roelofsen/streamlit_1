import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# Initialize SQLite engine
engine = create_engine(r'sqlite:///C:/Users/Administrator/HAMN/air_monitoring_HAMN_A.db')

# Define the available parameters for each station
station_pollutants = {
    'COH_MBL1': ['AQHI', 'ATEM', 'BP', 'CO', 'NO', 'NO2', 'NOX', 'O3', 'PM2.5', 'RH', 'Rain', 'SO2', 'WD', 'WS'],
    'COH_MBL2': ['AQHI', 'ATEM', 'BP', 'CO', 'NO', 'NO2', 'NOX', 'O3', 'PM2.5', 'RH', 'Rain', 'SO2', 'WD', 'WS'],
    'GFL Lago': ['TRS'],
    'STN29102': ['NO', 'NO2', 'NOX', 'PM10', 'PM2.5', 'SO2', 'TRS', 'WD', 'WS'],
    'STN29153': ['PM10', 'PM2.5', 'WD', 'WS'],
    'STN29163': ['TRS'],
    'STN29167': ['ATEM', 'BP', 'WD', 'WS'],
    'STN29168': ['PM10', 'PM2.5', 'TRS'],
    'STN29171': ['ATEM', 'RH', 'WD', 'WS'],
    'STN29172': ['WD', 'WS'],
    'STN29180': ['PM10', 'PM2.5', 'WD', 'WS'],
    'STN29565': ['PM10', 'Rain', 'WD', 'WS'],
    'STN29667': ['AT', 'NO', 'NO2', 'NOX', 'PM10', 'PM2.5', 'SO2', 'TRS'],
}

# Function to fetch data from the database
def fetch_data(stations, pollutants, start_date, end_date):
    station_filter = "','".join(stations)
    pollutant_filter = "','".join(pollutants)
    query = f"""
    SELECT * FROM measurements
    WHERE station_id IN ('{station_filter}') AND pollutants IN ('{pollutant_filter}')
    AND timestamp BETWEEN '{start_date}' AND '{end_date}'
    """
    df = pd.read_sql(query, engine)
    return df

# Function for anomaly detection using Z-score
def detect_anomalies(df, threshold=3):
    df['z_score'] = np.abs(stats.zscore(df['value']))
    anomalies = df[df['z_score'] > threshold]
    return anomalies

# Streamlit app
st.title("EDA and Statistical Analysis Dashboard")

# User input for station selection
stations = list(station_pollutants.keys())
selected_stations = st.multiselect("Select Stations", stations)

# Dynamically update pollutant options based on selected stations
if selected_stations:
    available_pollutants = sorted(set(
        param for station in selected_stations for param in station_pollutants[station]
    ))
else:
    available_pollutants = []

selected_pollutants = st.multiselect("Select Pollutants", available_pollutants)

# User input for date range selection
start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
end_date = st.date_input("End Date", value=datetime(2023, 12, 31))

# Fetch data based on user input
if st.button("Analyze"):
    df = fetch_data(selected_stations, selected_pollutants, start_date, end_date)
    
    if not df.empty:
        st.write(f"Data for selected stations and pollutants from {start_date} to {end_date}")

        # Drop rows with missing values
        df = df.dropna(subset=['value', 'timestamp'])

        # Display basic statistics for each station side by side
        st.write("Basic Statistics for Each Station")
        columns = st.columns(len(selected_stations))
        for col, station in zip(columns, selected_stations):
            station_df = df[df['station_id'] == station]
            if not station_df.empty:
                col.write(f"**{station}**")
                col.write(station_df['value'].describe())

        # Define a color sequence for distinct colors
        color_sequence = px.colors.qualitative.Set1

        # Plot data
        st.write("Time Series Plot")
        fig_time_series = px.line(df, x='timestamp', y='value', color='station_id', title='Time Series Data', color_discrete_sequence=color_sequence)
        st.plotly_chart(fig_time_series)

        # Display histogram
        st.write("Histogram")
        fig_histogram = px.histogram(df, x='value', color='station_id', title='Histogram', color_discrete_sequence=color_sequence)
        st.plotly_chart(fig_histogram)
        
        # Calculate and display moving average
        st.write("Moving Average")
        df['moving_avg'] = df['value'].rolling(window=7).mean()
        fig_moving_avg = px.line(df, x='timestamp', y='moving_avg', color='station_id', title='Moving Average', color_discrete_sequence=color_sequence)
        st.plotly_chart(fig_moving_avg)
        
        # Seasonal decomposition
        st.write("Seasonal Decomposition")
        if len(df['pollutants'].unique()) == 1:  # Decomposition requires one time series
            decomposition = seasonal_decompose(df.set_index('timestamp')['value'], model='additive', period=30)
            fig_trend = px.line(decomposition.trend.reset_index(), x='timestamp', y='trend', title='Trend Component', color_discrete_sequence=color_sequence)
            fig_seasonal = px.line(decomposition.seasonal.reset_index(), x='timestamp', y='seasonal', title='Seasonal Component', color_discrete_sequence=color_sequence)
            fig_residual = px.line(decomposition.resid.reset_index(), x='timestamp', y='resid', title='Residual Component', color_discrete_sequence=color_sequence)
            st.plotly_chart(fig_trend)
            st.plotly_chart(fig_seasonal)
            st.plotly_chart(fig_residual)
        
        # Correlation matrix
        if len(selected_pollutants) > 1:
            pivot_table = df.pivot_table(index='timestamp', columns='pollutants', values='value')
            corr_matrix = pivot_table.corr()
            st.write("Correlation Matrix")
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect='auto', title='Correlation Matrix', color_continuous_scale='RdBu')
            st.plotly_chart(fig_corr)
        
        # Additional statistics
        st.write("Skewness and Kurtosis")
        skewness = df['value'].skew()
        kurtosis = df['value'].kurtosis()
        st.write(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

        # Anomaly detection
        st.write("Anomaly Detection")
        anomalies = detect_anomalies(df)
        st.write(f"Number of anomalies detected: {len(anomalies)}")
        st.dataframe(anomalies)

        # Plot anomalies
        if not anomalies.empty:
            fig_anomalies = px.scatter(anomalies, x='timestamp', y='value', color='station_id', title='Anomalies', size='z_score', color_discrete_sequence=color_sequence)
            st.plotly_chart(fig_anomalies)
        
    else:
        st.write("No data available for the selected criteria.")
