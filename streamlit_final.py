import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd 
import os
from scipy import stats

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from IPython.display import clear_output
from scipy.signal import find_peaks
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout

import torch
import random
import time
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyGDataLoader

import json

from algorithms.dif import DIF




# 전류 데이터 전처리
def Clean(df):
    df = df.drop(df.columns[0], axis=1)
    new_column_name = df.iloc[2, :]
    new_column_name = new_column_name.reset_index(drop=True)
    df.columns = new_column_name
    df = df.iloc[3:]
    df = df.drop("TAG DESC", axis=1)
    df =df.rename(columns={'날짜':'Date'})
    # Convert the Date column to string
    df['Date'] = df['Date'].astype(str)
    # Convert the Date column to datetime with the desired format
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H:%M:%S")
    df['VALUE'] = pd.to_numeric(df['VALUE'])  # VALUE 열의 데이터 타입을 숫자로 변환
    df = df.reset_index(drop=True)

    return df

def series_filter(values, kernel_size=3):

    filter_values = np.cumsum(values, dtype=float)

    filter_values[kernel_size:] = filter_values[kernel_size:] - filter_values[:-kernel_size]
    filter_values[kernel_size:] = filter_values[kernel_size:] / kernel_size

    for i in range(1, kernel_size):
        filter_values[i] /= i + 1

    return filter_values


def extrapolate_next(values):

    last_value = values[-1]
    slope = [(last_value - v) / i for (i, v) in enumerate(values[::-1])]
    slope[0] = 0
    next_values = last_value + np.cumsum(slope)

    return next_values


def marge_series(values, extend_num=5, forward=5):

    next_value = extrapolate_next(values)[forward]
    extension = [next_value] * extend_num

    if isinstance(values, list):
        marge_values = values + extension
    else:
        marge_values = np.append(values, extension)
    return marge_values


class Silency(object):
    def __init__(self, amp_window_size, series_window_size, score_window_size):
        self.amp_window_size = amp_window_size
        self.series_window_size = series_window_size
        self.score_window_size = score_window_size

    def transform_silency_map(self, values):

        freq = np.fft.fft(values)
        mag = np.sqrt(freq.real ** 2 + freq.imag ** 2)
        spectral_residual = np.exp(np.log(mag) - series_filter(np.log(mag), self.amp_window_size))

        freq.real = freq.real * spectral_residual / mag
        freq.imag = freq.imag * spectral_residual / mag

        silency_map = np.fft.ifft(freq)
        return silency_map

    def transform_spectral_residual(self, values):
        silency_map = self.transform_silency_map(values)
        spectral_residual = np.sqrt(silency_map.real ** 2 + silency_map.imag ** 2)
        return spectral_residual

    def generate_anomaly_score(self, values, type="avg"):


        extended_series = marge_series(values, self.series_window_size, self.series_window_size)
        mag = self.transform_spectral_residual(extended_series)[: len(values)]

        if type == "avg":
            ave_filter = series_filter(mag, self.score_window_size)
            score = (mag - ave_filter) / ave_filter
        elif type == "abs":
            ave_filter = series_filter(mag, self.score_window_size)
            score = np.abs(mag - ave_filter) / ave_filter
        elif type == "chisq":
            score = stats.chi2.cdf((mag - np.mean(mag)) ** 2 / np.var(mag), df=1)
        else:
            raise ValueError("No type!")
        return score


def sr_time_series(time_series, amp_window_size=5, series_window_size=15, score_window_size=5):
    # Initialize the Silency class with given window sizes
    silency_transformer = Silency(amp_window_size, series_window_size, score_window_size)

    #amp_window_size: 스펙트럼 구성 요소의 진폭을 평활화하거나 필터링하는 데 사용되는 창 크기. 이 값을 증가시키면 진폭의 평활화가 더욱 강조되어 스펙트럼에서의 급격한 변동을 줄일 수 있습니다.
    #series_window_size: 시계열 데이터를 직접 작업할 때 사용되는 창 크기(예: 시계열 병합 또는 세분화)를 제어할 수 있습니다. 정확한 목적은 시계열에 적용되는 특정 방법이나 알고리즘에 따라 달라집니다.
    #SCORE_WINDOW_SIZE: 이는 스펙트럼 잔차의 최종 점수 또는 순위와 관련이 있을 수 있으며, 잔차가 최종 이상치 점수로 변환되는 방식을 제어합니다. 다시 말하지만, 이동 평균 또는 기타 평활화 기법에서 최종 측정값을 도출하는 데 사용될 수 있습니다.

    # Transform the time series to spectral residuals
    spectral_residuals = silency_transformer.transform_spectral_residual(time_series)

    # Return the spectral residuals
    return spectral_residuals

def process_data(dataset, category):
    if dataset.empty:
        raise ValueError("The provided dataset is empty.")
        
    data = dataset[dataset['Category'] == category]['VALUE']

    if len(data) == 0:
        raise ValueError(f"No data found for category: {category}")
    
    # FFT Transformation
    fft_data = np.fft.fft(data)
    fft_data_magnitude = [abs(value) for value in fft_data]

    # Conversion to SR format
    sr_data = sr_time_series(data)

    return data, sr_data, fft_data_magnitude


def visualize_data(data, sr_data, fft_data_magnitude, category):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=(f"{category} Data", f"SR_{category}", f"fft_{category} Magnitude"))

    fig.add_trace(go.Scatter(y=data, mode='lines', name=f'{category} Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=sr_data, mode='lines', name=f'SR_{category}'), row=2, col=1)
    fig.add_trace(go.Scatter(y=np.log(np.abs(fft_data_magnitude) + 1e-10), mode='lines', name=f'fft_{category} Magnitude (Log Scale)'), row=3, col=1)

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text=f"{category} Data Value", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text=f"SR_{category} Value", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text=f"fft_{category} Magnitude (Log Scale)", row=3, col=1)
    fig.update_layout(title_text=f"{category} Data, SR_{category}, and fft_{category} Magnitude Time Series")
    
    return fig


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

def create_lstm_model(window_size):
    model = Sequential()
    
    # 첫 번째 LSTM 계층
    model.add(LSTM(128, input_shape=(window_size, 1), return_sequences=True))
    model.add(Dropout(0.2))  # 첫 번째 드롭아웃 계층
    
    # 두 번째 LSTM 계층
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))  # 두 번째 드롭아웃 계층
    
    # 세 번째 LSTM 계층
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))  # 세 번째 드롭아웃 계층
    
    # 출력 계층
    model.add(Dense(1, activation='linear'))
    
    # Optimizer 설정
    optimizer = Adam(learning_rate=0.001)
    
    # 모델 컴파일
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model






def create_dataset(dataset, window_size):
    dataX, dataY = [], []
    for i in range(len(dataset)-window_size-1):
        a = dataset[i:(i+window_size)]
        dataX.append(a)
        dataY.append(dataset[i + window_size])
    return np.array(dataX), np.array(dataY)



def weighted_section(data, section_threshold, weight):
    # If no section_threshold is provided, return the original data
    if section_threshold is None:
        return data
    
    # Make a copy of the data to avoid modifying the original
    weighted_data = data.copy()
    
    # Find all peak points
    peaks, _ = find_peaks(weighted_data, prominence=4, height=5)
    
    for peak in peaks:
        # Calculate the mean value around each peak (2 points on either side)
        values_around_peak = weighted_data[max(0, peak-2):min(len(weighted_data), peak+3)]
        mean_value = np.mean(values_around_peak)
        
        # If this mean value is below the threshold, apply the weight
        if mean_value < section_threshold:
            weighted_data[max(0, peak-2):min(len(weighted_data), peak+3)] *= weight
    
    return weighted_data


def weighted_peak(data, peak_threshold, weight):
    # Make a copy of the data to avoid modifying the original
    weighted_data = data.copy()
    
    # Find all peak points
    peaks, _ = find_peaks(weighted_data, prominence=4, height=5)
    
    for peak in peaks:
        if weighted_data[peak] < peak_threshold:
            weighted_data[peak] *= weight
    
    return weighted_data


from keras.callbacks import EarlyStopping

def preprocess_and_train_for_category(dataset, category, window_size, epochs, weight, weighting_func, section_threshold=None, peak_threshold=None):
    data_subset = dataset[dataset['Category'] == category]['VALUE'].values
    if len(data_subset) == 0:
        st.write(f"No data found for category {category}")
        return None, None

    # Choosing the weighting function based on the user's selection
    if weighting_func == "weighted_peak":
        operation_data = weighted_peak(data_subset, peak_threshold, weight)
    elif weighting_func == "weighted_section":
        operation_data = weighted_section(data_subset, section_threshold, weight)
    else:
        operation_data = data_subset.copy()

    # SR_data 변환 수행
    SR_data = sr_time_series(operation_data)

    X, y = create_dataset(SR_data, window_size)
    X = X.reshape(X.shape[0], window_size, 1)

    # Split data into train and validation sets
    train_size = int(len(X) * 0.9)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Create and train LSTM model
    lstm_model = create_lstm_model(window_size)
    
    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=64, shuffle=False, callbacks=[early_stop])

    SR_data_dif = SR_data.reshape(-1, 1)
    model_configs = {
        'n_ensemble': 100,
        'n_estimators': 8,
        'max_samples': 'auto',
        'batch_size': 1024,
        'n_processes': 4
    }
    model_dif = DIF(**model_configs)
    model_dif.fit(SR_data_dif)

    return lstm_model, model_dif


def preprocess_and_train(dataset, category, window_size, epochs, weight, weighting_func, section_threshold=None, peak_threshold=None):
    dataset = dataset.dropna(subset=['Category'])
    
    if category == 'EU':
        lstm_model, model_dif = preprocess_and_train_for_category(dataset, 'EU', window_size, epochs, weight, weighting_func, section_threshold, peak_threshold)
    elif category == 'US':
        lstm_model, model_dif = preprocess_and_train_for_category(dataset, 'US', window_size, epochs, weight, weighting_func, section_threshold, peak_threshold)
    else:
        raise ValueError(f"Unsupported category: {category}")

    return lstm_model, model_dif




def visualize_results(real, all_predictions, all_anomaly_scores, all_errors, category, window_size):
    transformed_real = sr_time_series(np.array(real))
    
    titles = [
        f"Original Data ({category})", 
        f"Transformed Data & Predictions ({category})", 
        f"Error ({category})", 
        f"Anomaly Score ({category})"
    ] * 2  # Duplicate for both EU & US

    fig = make_subplots(rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=titles)
    
    starting_row = 1 if category == 'EU' else 5

    fig.add_trace(go.Scatter(y=real, mode='lines+markers', name=f'Original Data ({category})'), row=starting_row, col=1)
    fig.add_trace(go.Scatter(y=transformed_real, mode='lines+markers', name=f'Transformed Data ({category})'), row=starting_row + 1, col=1)
    fig.add_trace(go.Scatter(x=list(range(window_size, len(all_predictions) + window_size)), y=all_predictions, mode='lines+markers', name=f'Predictions ({category})'), row=starting_row + 1, col=1)
    fig.add_trace(go.Scatter(x=list(range(window_size, len(all_errors) + window_size)), y=all_errors, mode='lines+markers', name=f'Error ({category})'), row=starting_row + 2, col=1)
    fig.add_trace(go.Scatter(x=list(range(window_size, len(all_anomaly_scores) + window_size)), y=all_anomaly_scores, mode='lines+markers', name=f'Anomaly Score ({category})'), row=starting_row + 3, col=1)

    fig.update_layout(height=1400, title_text="Visualizations for EU & US Data")
    fig.show()

    return fig


def real_time_visualization(real_data, model, model_dif, window_size, weight, buffer_size, weighting_func, section_threshold=None, peak_threshold=None):
    # Extracting the raw data values
    raw_data_values = real_data['VALUE'].values
    
    # Choosing the weighting function based on the user's selection
    if weighting_func == "weighted_peak":
        weighted_data = weighted_peak(raw_data_values, peak_threshold, weight)
    elif weighting_func == "weighted_section":
        weighted_data = weighted_section(raw_data_values, section_threshold, weight)
    else:
        weighted_data = raw_data_values  # If neither threshold is provided, use raw data
    
    # Convert weighted data to array and then apply the sr_time_series transformation
    weighted_data_array = np.array(weighted_data)
    sr_data = sr_time_series(weighted_data_array)
    
    # Rest of the function remains the same...
    all_windows = []
    all_predictions = [None] * window_size  
    all_anomaly_scores = [None] * window_size
    all_errors = [None] * window_size  
    prediction_buffer = []

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)  # To link x-axes
    plot_placeholder = st.empty()
    plot_placeholder.plotly_chart(fig, use_container_width=True)

    for i in range(len(weighted_data_array) - window_size):
        current_window = sr_data[i:i+window_size].reshape(1, window_size, 1)
        prediction = model.predict(current_window, verbose=0)
        error = abs(prediction[0, 0] - sr_data[i+window_size])
        
        prediction_buffer.append(prediction[0, 0])
        all_windows.append(current_window)
        all_predictions.append(prediction[0, 0])
        all_errors.append(error)
        
        if len(prediction_buffer) == buffer_size or i == (len(weighted_data_array) - window_size - 1):
            buffered_anomalies = model_dif.decision_function(np.array(prediction_buffer).reshape(-1, 1))
            expected_length = buffer_size if len(prediction_buffer) == buffer_size else len(weighted_data_array) - window_size - i
            if len(buffered_anomalies) != expected_length:
                buffered_anomalies = [None] * (expected_length - len(buffered_anomalies)) + list(buffered_anomalies)
            all_anomaly_scores.extend(buffered_anomalies)
            prediction_buffer.clear()

        if len(fig.data) == 0:
            fig.add_trace(go.Scatter(x=list(range(i+window_size)), y=raw_data_values[:i+window_size], mode='lines+markers', name='Raw Data'), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(i+window_size)), y=weighted_data_array[:i+window_size], mode='lines+markers', name='Weighted Data'), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(i+window_size)), y=sr_data[:i+window_size], mode='lines+markers', name='Transformed Data'), row=2, col=1)
            fig.add_trace(go.Scatter(x=list(range(i+2*window_size)), y=all_predictions, mode='lines+markers', name='Predictions'), row=2, col=1)
            fig.add_trace(go.Scatter(x=list(range(i+2*window_size)), y=all_errors, mode='lines+markers', name='Errors'), row=3, col=1)
            fig.add_trace(go.Scatter(x=list(range(i+2*window_size)), y=all_anomaly_scores, mode='lines+markers', name='Anomaly Scores'), row=4, col=1)
        else:
            fig.data[0].x = list(range(i+window_size))
            fig.data[1].x = list(range(i+window_size))
            fig.data[2].x = list(range(i+2*window_size))
            fig.data[3].x = list(range(i+2*window_size))
            fig.data[4].x = list(range(i+2*window_size))
            fig.data[5].x = list(range(i+2*window_size))
            
            fig.data[0].y = raw_data_values[:i+window_size]
            fig.data[1].y = weighted_data_array[:i+window_size]
            fig.data[2].y = sr_data[:i+window_size]
            fig.data[3].y = all_predictions
            fig.data[4].y = all_errors
            fig.data[5].y = all_anomaly_scores

        plot_placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(0.2)

    return all_windows, all_predictions, all_anomaly_scores, all_errors




# The function run_visualization has been provided only once
def run_visualization(real_data, window_size, section_threshold, weight, buffer_size, peak_threshold):
    unique_categories = real_data['Category'].unique()
    
    st.write(f"Unique Categories Found: {unique_categories}")

    result = {}
    for category in unique_categories:
        st.write(f"Processing data for: {category}")

        data_subset = real_data[real_data['Category'] == category]
        st.write(f"Entries for {category}: {len(data_subset)}")

        lstm_model_name = f"{category}_lstm_model"
        model_dif_name = f"{category}_model_dif"

        if lstm_model_name not in st.session_state or model_dif_name not in st.session_state:
            st.session_state[lstm_model_name] = lstm_model
            st.session_state[model_dif_name] = model_dif
        else:
            lstm_model = st.session_state.get(lstm_model_name)
            model_dif = st.session_state.get(model_dif_name)

        windows, predictions, anomaly_scores, errors = real_time_visualization(data_subset, lstm_model, model_dif, window_size, weight, buffer_size, "weighted_section", section_threshold, peak_threshold)
        
        result[category] = {
            'windows': windows,
            'predictions': predictions,
            'anomaly_scores': anomaly_scores,
            'errors': errors
        }
        
    return result





def generate_plot(idx, data_values, data_name):
    start_idx = max(0, idx - 8)
    end_idx = min(len(data_values), idx + 9)  # Ensure it doesn't go out of bounds

    x = list(range(start_idx, end_idx))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=data_values[start_idx:end_idx],
                             mode='lines+markers',
                             name=data_name))
                             
    fig.update_layout(title=f"Visualization of {data_name} around index {idx}",
                      xaxis_title="Index",
                      yaxis_title="Value")
    
    return fig

def generate_combined_plot(idx, data1_values, data2_values, name1, name2):
    start_idx = max(0, idx - 8)
    end_idx = min(len(data1_values), idx + 9)  # Ensure it doesn't go out of bounds

    x = list(range(start_idx, end_idx))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=data1_values[start_idx:end_idx],
                             mode='lines+markers',
                             name=name1))
    fig.add_trace(go.Scatter(x=x, y=data2_values[start_idx:end_idx],
                             mode='lines+markers',
                             name=name2))
                             
    fig.update_layout(title=f"Visualization around index {idx}",
                      xaxis_title="Index",
                      yaxis_title="Value")
    
    return fig


def data_visualization():
    st.title("Data Visualization")

    # Initialize session states
    if 'show_visualizations' not in st.session_state:
        st.session_state.show_visualizations = False
    if 'real_time_visualizations' not in st.session_state:
        st.session_state.real_time_visualizations = False
    if 'lstm_model' not in st.session_state:
        st.session_state.lstm_model = None  # Store the LSTM model in the session state

    dataset = None  # Initialize the dataset variable

    # File uploader for the main dataset with a unique key
    dataset_file = st.file_uploader("Upload the dataset", type=["csv", "xlsx"], key='main_dataset_uploader')
    if dataset_file:
        # Check the file extension and use the appropriate function
        if dataset_file.name.endswith(".csv"):
            dataset = pd.read_csv(dataset_file)
            dataset = dataset.drop(columns=['Unnamed: 0'], errors='ignore') 

        elif dataset_file.name.endswith(".xlsx"):
            dataset = pd.read_excel(dataset_file)
            # Apply the Clean() function to the xlsx file
            dataset = Clean(dataset)
        st.session_state.dataset = dataset 
        st.write("Dataset uploaded successfully!")
        st.write(dataset.head())  # display the first few rows of the dataset

    # File uploader for the real-time data with a unique key
    real_time_file = st.file_uploader("Upload the real-time data", type=["csv", "xlsx"], key='real_time_dataset_uploader')
    if real_time_file:
        # Check the file extension and use the appropriate function
        if real_time_file.name.endswith(".csv"):
            real_time_data = pd.read_csv(real_time_file)
            real_time_data = real_time_data.drop(columns=['Unnamed: 0'], errors='ignore') 
        elif real_time_file.name.endswith(".xlsx"):
            real_time_data = pd.read_excel(real_time_file)
            # Apply the Clean() function to the xlsx file
            real_time_data = Clean(real_time_data)
        st.session_state.real_time_data = real_time_data
        st.write("Real-time data uploaded successfully!")
        st.write(real_time_data.head())  # display the first few rows of the real-time data



    if st.button("Toggle Visualizations"):  # The button's purpose is to toggle the visualization on/off
        st.session_state.show_visualizations = not st.session_state.show_visualizations

    if st.session_state.show_visualizations:
        # Check if the dataset is uploaded and valid
        if dataset is not None:
            # For EU Data
            if 'Category' in dataset.columns and 'EU' in dataset['Category'].unique():
                EU_data, SR_EU, fft_EU_magnitude = process_data(dataset, 'EU')
                fig_eu = visualize_data(EU_data, SR_EU, fft_EU_magnitude, 'EU')
                st.plotly_chart(fig_eu)

            # For US Data
            if 'Category' in dataset.columns and 'US' in dataset['Category'].unique():
                US_data, SR_US, fft_US_magnitude = process_data(dataset, 'US')
                fig_us = visualize_data(US_data, SR_US, fft_US_magnitude, 'US')
                st.plotly_chart(fig_us)
        else:
            st.write("Please upload a dataset first.")


# 재귀적으로 ndarray를 리스트로 변환하는 함수
def ndarray_to_list(obj):
    if isinstance(obj, dict):
        return {key: ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ndarray_to_list(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):  # float32 처리 추가
        return float(obj)
    else:
        return obj


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):  # Handle float32 type
            return float(obj)
        return super(NumpyEncoder, self).default(obj)



def app():
    st.title("Real-Time Visualizations for EU & US Data")

    # Initialize 'results' in session_state if it doesn't exist
    if 'results' not in st.session_state:
        st.session_state['results'] = {}

    # Radio button to select the weighting function
    weighting_func = st.radio("Select a weighting function", ["weighted_peak", "weighted_section"])
    
    # Store the chosen function in session state
    st.session_state["weighting_func"] = weighting_func


    window_size = st.slider('Select window size', 10, 100, 15)  
    buffer_size = st.slider('Select buffer size', 10, 1000, 300)  
    
    epochs = st.text_input('Enter number of epochs for training', '10')
    try:
        epochs = int(epochs)
    except ValueError:
        st.write("Please enter a valid integer for epochs.")
        return

    for category in ['EU', 'US']:
        st.subheader(f"{category} control")

        # Depending on the weighting function selected, display the relevant input field
        if weighting_func == "weighted_section":
            st.session_state[f"{category}_section_threshold"] = st.number_input(f'Enter section threshold value for {category}', value=7.6, step=0.1)
        elif weighting_func == "weighted_peak":
            st.session_state[f"{category}_peak_threshold"] = st.number_input(f'Enter peak threshold value for {category}', value=10.3, step=0.1)

        st.session_state[f"{category}_weight"] = st.number_input(f'Enter additional weight value for {category}', value=1.3, step=0.1)


    dataset = st.session_state.get("dataset", None)
    real_time_data = st.session_state.get("real_time_data", None)

    if st.button("Train LSTM Models for EU & US"):
        # Initialize the progress bar
        progress_bar = st.progress(0)

        for index, category in enumerate(['EU', 'US']):
            category_dataset = dataset[dataset['Category'] == category]
            category_weight = st.session_state[f"{category}_weight"]

            # Apply selected weighting function to the data before preprocessing and training
            if st.session_state["weighting_func"] == "weighted_peak":
                category_peak_threshold = st.session_state.get(f"{category}_peak_threshold", None)
                if category_peak_threshold:
                    weighted_data = weighted_peak(category_dataset['VALUE'].values, category_peak_threshold, category_weight)
                    category_dataset['VALUE'] = weighted_data
                    lstm_model, model_dif = preprocess_and_train(category_dataset, category, window_size, epochs, None, category_weight, category_peak_threshold)
            else:  # weighted_anomaly_peak
                category_section_threshold = st.session_state.get(f"{category}_section_threshold", None)
                if category_section_threshold:
                    weighted_data = weighted_section(category_dataset['VALUE'].values, category_section_threshold, category_weight)
                    category_dataset['VALUE'] = weighted_data
                    lstm_model, model_dif = preprocess_and_train(category_dataset, category, window_size, epochs, category_section_threshold, category_weight, None)
            
            st.session_state[f"{category}_lstm_model"] = lstm_model
            st.session_state[f"{category}_model_dif"] = model_dif

            # Update the progress bar after each model is trained
            progress_bar.progress((index + 1) * 0.5)  # since we have 2 categories, we increment by 0.5 (or 50%) after each training

        st.write("LSTM models trained for both EU and US!")



    # Button to Start Real-Time Visualizations
    if st.button("Start Real-Time Visualizations"):
        if 'EU_lstm_model' in st.session_state and 'US_lstm_model' in st.session_state:
            with st.spinner("Running real-time visualizations..."):
                results = {}
                categories_processed = st.session_state.get('categories_processed', set())

                # Initialize 'results' in session_state if it doesn't exist
                if 'results' not in st.session_state:
                    st.session_state['results'] = {}

                for category in ['EU', 'US']:
                    if category not in categories_processed:
                        category_dataset = real_time_data[real_time_data['Category'] == category].copy()
                        category_section_threshold = st.session_state.get(f"{category}_section_threshold", None)
                        category_peak_threshold = st.session_state.get(f"{category}_peak_threshold", None)
                        category_weight = st.session_state.get(f"{category}_weight", None)

                        # Apply selected weighting function to the data
                        if st.session_state["weighting_func"] == "weighted_peak":
                            weighted_data = weighted_peak(category_dataset['VALUE'].values, category_peak_threshold, category_weight)
                        else:
                            weighted_data = weighted_section(category_dataset['VALUE'].values, category_section_threshold, category_weight)

                        category_dataset['VALUE'] = weighted_data
                        results_data = run_visualization(category_dataset, window_size, category_section_threshold, category_weight, buffer_size, category_peak_threshold)

                        results[category] = results_data[category]

                        # Mark this category as processed
                        categories_processed.add(category)
                        st.session_state['categories_processed'] = categories_processed
                        st.session_state['results'][category] = results_data

                        # Save results for the specific category to a .json file
                        with open(f'{category}.json', 'w') as file:
                            json.dump(results_data, file, cls=NumpyEncoder)

            st.success("Real-time visualization completed!")
        else:
            st.error("Please ensure the LSTM models for both EU and US are trained.")




def check_anomal_point():
    st.title("Check anomaly point")

    # Choose region
    category = st.radio("Choose a category", ["EU", "US"])

    # Safely retrieve results for the chosen category from the saved JSON file
    try:
        with open(f'{category}.json', 'r') as file:
            results = json.load(file)
    except FileNotFoundError:
        st.error(f"No results found for {category}. Please generate the real-time visualization first.")
        return

    # Extract session values
    errors = results[category].get('errors', [])
    anomaly_scores = results[category].get('anomaly_scores', [])


    # Input for threshold values
    error_threshold = st.number_input(f'Enter error threshold value for {category}', value=0.02, step=0.0001, format="%.4f")
    anomaly_score_threshold = st.number_input(f'Enter anomaly score threshold value for {category}', value=0.2, step=0.0001, format="%.4f")

    section_threshold = st.session_state.get(f"{category}_section_threshold", None)
    peak_threshold = st.session_state.get(f"{category}_peak_threshold", None)
    weight = st.session_state.get(f"{category}_weight", None)

    # Check against the thresholds
    if st.button("Check Thresholds"):
        error_indices = [i for i, e in enumerate(errors) if e is not None and e > error_threshold]
        anomaly_indices = [i for i, a in enumerate(anomaly_scores) if a is not None and a > anomaly_score_threshold]
        both_indices = [i for i, (e, a) in enumerate(zip(errors, anomaly_scores)) if e is not None and a is not None and e > error_threshold and a > anomaly_score_threshold]

        st.write(f"Indices with errors above the threshold: {error_indices}")
        st.write(f"Indices with anomaly scores above the threshold: {anomaly_indices}")
        st.write(f"Indices that exceed both thresholds: {both_indices}")

    # Get specific index input from the user
    specific_index = st.number_input("Enter a specific index to visualize", min_value=0, value=100)


    # Button to visualize data around that specific index
    if st.button("Visualize Data around Specified Index"):
        if 'real_time_data' not in st.session_state:
            st.error("Real-time data is missing. Please generate it first.")
            return
        
        predictions = results.get('predictions', [])
        raw_data_values = st.session_state['real_time_data'][st.session_state['real_time_data']['Category'] == category]['VALUE'].values
        
        if st.session_state.get("weighting_func", "") == "weighted_peak" and peak_threshold is not None:
            weighted_data = weighted_peak(raw_data_values, peak_threshold, weight)
        else:
            weighted_data = weighted_section(raw_data_values, section_threshold, weight)

        sr_data = sr_time_series(weighted_data)

        st.write("## Visualization for Raw Data and Weighted Data")
        st.plotly_chart(generate_combined_plot(specific_index, raw_data_values, weighted_data, "Raw Data", "Weighted Data"))

        st.write("## Visualization for Transformed Data and Predictions")
        st.plotly_chart(generate_combined_plot(specific_index, sr_data, predictions, "Transformed Data", "Predictions"))

        st.write("## Visualization for Errors")
        st.plotly_chart(generate_plot(specific_index, errors, "Errors"))

        st.write("## Visualization for Anomaly Scores")
        st.plotly_chart(generate_plot(specific_index, anomaly_scores, "Anomaly Scores"))






def main_page():
    # Check if the session state has 'selected_option'
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None

    st.sidebar.header("Navigation")

    if st.sidebar.button("Data Visualization"):
        st.session_state.selected_option = 'Data Visualization'

    if st.sidebar.button("Real-Time Visualizations for EU & US Data"):
        st.session_state.selected_option = 'Real-Time Visualizations for EU & US Data'

    if st.sidebar.button("Check Anomal Point"):
        st.session_state.selected_option = 'Check Anomal Point'

    # Display based on selected option
    if st.session_state.selected_option == 'Data Visualization':
        data_visualization()

    elif st.session_state.selected_option == 'Real-Time Visualizations for EU & US Data':
        app()

    elif st.session_state.selected_option == 'Check Anomal Point':
        check_anomal_point()

if __name__ == "__main__":
    main_page()
