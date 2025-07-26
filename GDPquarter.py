import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from datetime import datetime
import pytz
import plotly.express as px
import plotly.graph_objects as go
import os
import shap
import matplotlib.pyplot as plt
import joblib
import hashlib
import csv
import time
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initial configuration
st.title("🔍 Quarterly GDP Prediction for 2024")
cet = pytz.timezone('CET')
current_date_time = cet.localize(datetime(2025, 7, 27, 0, 39))
st.write(f"**Current date and time:** {current_date_time.strftime('%d/%m/%Y %H:%M %Z')}")

random.seed(42)
np.random.seed(42)

# Initialize error log
error_log = []

# Function to normalize strings
def normalize_name(name):
    if pd.isna(name) or not isinstance(name, str):
        error_log.append(f"Non-text or NaN value: {name}. Replacing with 'unknown'.")
        return "unknown"
    name = re.sub(r'\s+', ' ', name.strip())
    name = name.replace("_", " ")
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii').strip()
    name = re.sub(r'\s+', ' ', name).lower()
    return name

# Convert quarters to standard format
def convert_roman_to_quarter(index_str):
    match = re.match(r'^Q([1-4])\s*(\d{4})$', index_str, re.IGNORECASE)
    if not match:
        error_log.append(f"Invalid quarter format: {index_str}")
        raise ValueError(f"Invalid quarter format: {index_str}")
    quarter, year = match.groups()
    return f"{year}Q{quarter}"

# Compute file hash for cache validation
def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# Load and preprocess data
@st.cache_data(show_spinner=False, persist=True)
def load_and_preprocess(uploaded_file=None, _cache_key="default"):
    start_time = time.time()
    try:
        if uploaded_file:
            uploaded_file.seek(0)
            raw_content = uploaded_file.read()
            if not raw_content.strip():
                error_log.append("Uploaded file is empty.")
                st.error("Error: The uploaded file is empty. Please check the file.")
                raise ValueError("Empty file.")
            error_log.append(f"Raw content of uploaded file (first 200 bytes): {raw_content[:200].decode('utf-8', errors='ignore')}...")
            
            encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']
            separators = [',', ';', '\t', '|', ':']
            df = None
            for encoding in encodings:
                for sep in separators:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(
                            uploaded_file,
                            thousands=' ', 
                            decimal=',',
                            encoding=encoding,
                            sep=sep,
                            skipinitialspace=True,
                            engine='python',
                            dtype_backend='numpy_nullable'
                        )
                        if not df.empty and len(df.columns) > 1:
                            first_col = df.columns[0].lower().strip()
                            if first_col.startswith('sect') or first_col in ['\ufeffsector', 'sector']:
                                error_log.append(f"File loaded with encoding '{encoding}' and separator '{sep}'.")
                                break
                    except Exception as e:
                        error_log.append(f"Failed to read with encoding '{encoding}' and separator '{sep}': {str(e)}")
                if df is not None:
                    break
            else:
                st.error("Failed to automatically read the CSV. Please specify encoding and separator.")
                encoding = st.selectbox("Select encoding", encodings)
                sep = st.text_input("Enter separator (e.g., ';', ',', '\\t')", value=',')
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file,
                        thousands=' ', 
                        decimal=',',
                        encoding=encoding,
                        sep=sep,
                        skipinitialspace=True,
                        engine='python'
                    )
                    error_log.append(f"File loaded with manual encoding '{encoding}' and separator '{sep}'.")
                except Exception as e:
                    error_log.append(f"Failed to read with manual encoding '{encoding}' and separator '{sep}': {str(e)}")
                    st.error(f"Error: Unable to read the CSV file with the provided parameters: {str(e)}")
                    raise ValueError("Invalid CSV format or incorrect parameters.")

            if df is None:
                error_log.append("No valid DataFrame could be loaded.")
                st.error("Error: Unable to read the CSV file. Please check the file content.")
                raise ValueError("No valid DataFrame loaded.")
        else:
            default_file = "PIB_Trimestrielle.csv"
            if not os.path.exists(default_file):
                error_log.append(f"File '{default_file}' not found.")
                st.error(f"Error: File '{default_file}' not found. Please check the file path.")
                raise FileNotFoundError(f"File '{default_file}' not found.")
            df = pd.read_csv(
                default_file,
                thousands=' ',
                decimal=',',
                encoding='utf-8',
                sep=',',
                engine='python',
                dtype_backend='numpy_nullable'
            )
            error_log.append(f"File loaded as CSV with encoding 'utf-8' and separator ','.")

        if df.empty or len(df.columns) == 0:
            error_log.append("The CSV file contains no valid columns.")
            st.error("Error: The CSV file contains no valid columns.")
            raise ValueError("No columns in the CSV file.")
        
        first_col = df.columns[0].lower().strip()
        if not (first_col.startswith('sect') or first_col in ['\ufeffsector', 'sector']):
            df.columns = ['Sector'] + list(df.columns[1:])
            error_log.append(f"First column renamed to 'Sector' (previously: '{first_col}')")
        
        if 'Sector' not in df.columns:
            st.error(f"Error: The 'Sector' column is not present. Current columns: {list(df.columns)}")
            error_log.append(f"'Sector' column missing. Found columns: {df.columns.tolist()}")
            raise KeyError("'Sector' column missing after renaming.")

        df['Sector'] = df['Sector'].apply(normalize_name)
        df.columns = ['Sector' if col == 'Sector' else normalize_name(col) for col in df.columns]
        for col in df.columns[1:]:
            df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')

        macro_keywords = [
            "unemployment rate", "inflation rate", "interest rate", "public debt",
            "international monetary policy", "regional geopolitical tensions",
            "commodity prices", "drought and climate disasters", "pandemics",
            "social crisis"
        ]
        sectors = [
            "agriculture forestry and fishing", "oil and gas extraction",
            "mining products extraction", "agro food industries",
            "textile clothing and leather industry", "petroleum refining",
            "chemical industry", "construction materials ceramic and glass",
            "mechanical and electrical industries", "miscellaneous industries",
            "electricity and gas production and distribution",
            "water distribution and waste treatment", "construction",
            "commerce maintenance and repair", "transport and storage services",
            "hospitality cafe and restaurant services", "information and communication",
            "financial activities", "public administration and defense services",
            "private and public education", "private and public health and social services",
            "other commercial services", "other household activities",
            "services provided by associations", "commercial activities",
            "non commercial activities", "net taxes on products"
        ]
        macro_rates = [
            "unemployment rate", "inflation rate", "interest rate", "public debt"
        ]
        events = [
            "international monetary policy", "regional geopolitical tensions",
            "commodity prices", "drought and climate disasters", "pandemics",
            "social crisis"
        ]

        macro_keywords = [normalize_name(m) for m in macro_keywords]
        sectors = [normalize_name(s) for s in sectors]
        macro_rates = [normalize_name(m) for m in macro_rates]
        events = [normalize_name(e) for e in events]

        actual_sectors = df['Sector'].tolist()
        error_log.append(f"Sectors in the CSV: {actual_sectors}")

        df_macro = df[df['Sector'].isin(macro_keywords)].copy()
        df_pib = df[df['Sector'] == "gross domestic product"].copy()
        if df_pib.empty:
            possible_gdp_names = ['gross domestic product', 'gdp']  # Updated to include 'gdp'
            for gdp_name in possible_gdp_names:
                if normalize_name(gdp_name) in df['Sector'].values:
                    df_pib = df[df['Sector'] == normalize_name(gdp_name)].copy()
                    df_pib['Sector'] = 'gross domestic product'
                    break
            if df_pib.empty:
                st.error(f"Error: No GDP data found even after searching for variants. Available columns: {df['Sector'].tolist()}")
                error_log.append("No GDP data found in the file.")
                raise ValueError("GDP data missing in the CSV.")

        df_secteurs = df[df['Sector'].isin(sectors) & ~df['Sector'].str.contains("GDP", case=False)].copy()

        missing_sectors = [s for s in sectors if s not in df['Sector'].values]
        missing_macro = [m for m in macro_keywords if m not in df['Sector'].values]
        if missing_sectors:
            st.warning(f"Warning: Missing sectors in the CSV: {missing_sectors}. Using the average of available sectors.")
            error_log.append(f"Missing sectors in the CSV: {missing_sectors}")
        if missing_macro:
            st.warning(f"Warning: Missing macros: {missing_macro}. Using default values (0).")
            error_log.append(f"Missing macros: {missing_macro}")

        df_macro.set_index("Sector", inplace=True)
        df_pib.set_index("Sector", inplace=True)
        df_secteurs.set_index("Sector", inplace=True)

        df_macro_T = df_macro.transpose()
        df_pib_T = df_pib.transpose()
        df_secteurs_T = df_secteurs.transpose()

        try:
            df_macro_T.index = [convert_roman_to_quarter(idx) for idx in df_macro_T.index]
            df_pib_T.index = [convert_roman_to_quarter(idx) for idx in df_pib_T.index]
            df_secteurs_T.index = [convert_roman_to_quarter(idx) for idx in df_secteurs_T.index]
            df_macro_T.index = pd.PeriodIndex(df_macro_T.index, freq='Q').to_timestamp()
            df_pib_T.index = pd.PeriodIndex(df_pib_T.index, freq='Q').to_timestamp()
            df_secteurs_T.index = pd.PeriodIndex(df_secteurs_T.index, freq='Q').to_timestamp()
        except ValueError as e:
            st.error(f"Error converting indices to dates: {str(e)}")
            error_log.append(f"Error converting indices: {str(e)}")
            raise

        X_df = pd.concat([df_secteurs_T, df_macro_T], axis=1).dropna()
        error_log.append(f"Shape of X_df after concatenation: {X_df.shape}")
        error_log.append(f"Columns of X_df after concatenation: {list(X_df.columns)}")

        y_df = df_pib_T.loc[X_df.index]
        if y_df.empty:
            st.error(f"Error: y_df empty after alignment with X_df. X_df indices: {X_df.index.tolist()}. df_pib_T indices: {df_pib_T.index.tolist()}")
            error_log.append("y_df empty after alignment.")
            raise ValueError("GDP data empty after preprocessing.")

        key_sectors = [
            "agriculture forestry and fishing",
            "mechanical and electrical industries",
            "hospitality cafe and restaurant services",
            "information and communication",
            "financial activities"
        ]
        key_sectors = [normalize_name(s) for s in key_sectors]

        for sector in key_sectors:
            col_name = f"{sector}_lag1"
            if sector in X_df.columns:
                X_df[col_name] = X_df[sector].shift(1).fillna(X_df[sector].mean())
            else:
                X_df[col_name] = X_df[sectors].mean(axis=1).shift(1).fillna(X_df[sectors].mean().mean()) if sectors else 0
                error_log.append(f"Lagged feature '{col_name}' added with sector average as '{sector}' is missing.")

        for rate in macro_rates:
            col_name = f"{rate}_lag1"
            if rate in X_df.columns:
                X_df[col_name] = X_df[rate].shift(1).fillna(X_df[rate].mean())
            else:
                X_df[col_name] = 0
                error_log.append(f"Lagged feature '{col_name}' added with value 0 as '{rate}' is missing.")

        X_df['gdp_lag1'] = y_df.shift(1).fillna(y_df.mean())

        expected_features = (
            sectors +
            macro_rates +
            events +
            [f"{s}_lag1" for s in key_sectors] +
            [f"{r}_lag1" for r in macro_rates] +
            ['gdp_lag1']
        )
        error_log.append(f"Expected columns in expected_features: {expected_features} (count: {len(expected_features)})")

        missing_cols = [col for col in expected_features if col not in X_df.columns]
        extra_cols = [col for col in X_df.columns if col not in expected_features]
        if missing_cols:
            existing_cols = [col for col in sectors + macro_rates + events if col in X_df.columns]
            for col in missing_cols:
                if col in sectors and existing_cols:
                    X_df[col] = X_df[existing_cols].mean(axis=1)
                    error_log.append(f"Missing feature '{col}' added with average of available sectors.")
                elif col.endswith('_lag1') and col.replace('_lag1', '') in X_df.columns:
                    X_df[col] = X_df[col.replace('_lag1', '')].shift(1).fillna(X_df[col.replace('_lag1', '')].mean())
                    error_log.append(f"Missing feature '{col}' added with lag.")
                else:
                    X_df[col] = 0
                    error_log.append(f"Missing feature '{col}' added with value 0.")
        if extra_cols:
            st.warning(f"Warning: Extra columns in X_df: {extra_cols}")
            error_log.append(f"Extra columns in X_df: {extra_cols}")
            X_df = X_df.drop(columns=extra_cols, errors='ignore')

        X_df = X_df[expected_features]
        error_log.append(f"Columns in X_df after reordering: {list(X_df.columns)}")
        error_log.append(f"Number of columns in X_df: {X_df.shape[1]} (expected: {len(expected_features)})")

        if list(X_df.columns) != expected_features:
            differences = [(i, a, b) for i, (a, b) in enumerate(zip(X_df.columns, expected_features)) if a != b]
            error_log.append(f"Mismatch in X_df columns: Differences at positions {differences}")
            st.error(f"Error: X_df columns ({len(X_df.columns)}) do not match expected_features ({len(expected_features)}). Differences: {differences}")
            st.stop()

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X_df)
        y = scaler_y.fit_transform(y_df.values.reshape(-1, 1)).flatten()
        quarters = X_df.index

        last_year = int(max(quarters).year)
        logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        return X, y, quarters, X_df, scaler_X, scaler_y, macro_keywords, sectors, macro_rates, events, last_year, y_df, expected_features, df

    except Exception as e:
        error_log.append(f"Error loading file: {str(e)}")
        st.error(f"Error: Error loading file: {str(e)}")
        raise

# File upload
uploaded_file = st.file_uploader("Upload your updated dataset (CSV, optional)", type=["csv"])
if uploaded_file:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.write("### Preview of the uploaded CSV file")
    try:
        uploaded_file.seek(0)
        try:
            df_preview = pd.read_csv(uploaded_file, thousands=' ', decimal=',', encoding='utf-8', sep=',')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df_preview = pd.read_csv(uploaded_file, thousands=' ', decimal=',', encoding='latin-1', sep=',')
        st.write(df_preview)
        
        if st.button("Add a new row"):
            quarter_columns = [col for col in df_preview.columns if re.match(r'^Q[1-4]\s*\d{4}$', col, re.IGNORECASE)]
            if quarter_columns:
                valid_quarters = []
                for col in quarter_columns:
                    try:
                        period = pd.Period(convert_roman_to_quarter(col), freq='Q')
                        valid_quarters.append((col, period))
                    except ValueError as e:
                        error_log.append(f"Ignored column '{col}': invalid quarter format ({str(e)})")
                        continue
                if valid_quarters:
                    last_quarter, last_period = max(valid_quarters, key=lambda x: x[1])
                    new_period = last_period + 1
                    new_quarter = f"Q{new_period.quarter} {new_period.year}"
                else:
                    error_log.append("No valid quarter columns found. Using 'Q1 2024' as default.")
                    new_quarter = "Q1 2024"
            else:
                error_log.append("No quarter columns detected. Using 'Q1 2024' as default.")
                new_quarter = "Q1 2024"
            new_row = pd.DataFrame({col: ['gross domestic product' if col == 'Sector' else 0.0] for col in df_preview.columns})
            if new_quarter not in df_preview.columns:
                new_row[new_quarter] = 0.0
            st.write(f"### Add data for {new_quarter}")
            edited_row = st.data_editor(new_row, num_rows="dynamic")
            
            if st.button("Save new row"):
                for col in df_preview.columns:
                    if col not in edited_row.columns:
                        edited_row[col] = 0.0
                if new_quarter not in df_preview.columns:
                    df_preview[new_quarter] = 0.0
                df_updated = pd.concat([df_preview, edited_row], ignore_index=True)
                output_file = "updated_PIB_Trimestrielle.csv"
                df_updated.to_csv(output_file, sep=',', index=False, encoding='utf-8')
                st.success(f"Success: New row saved to '{output_file}'.")
                csv_buffer = BytesIO()
                df_updated.to_csv(csv_buffer, sep=',', index=False, encoding='utf-8')
                csv_buffer.seek(0)
                uploaded_file = csv_buffer
                uploaded_file.name = output_file
    except Exception as e:
        error_log.append(f"Error reading uploaded file for preview: {str(e)}")
        st.error(f"Error: Error reading uploaded file: {str(e)}")
        st.stop()

# Load data
try:
    cache_key = "default" if uploaded_file is None else hashlib.sha256(uploaded_file.read()).hexdigest()
    if uploaded_file:
        uploaded_file.seek(0)
    X, y, quarters, X_df, scaler_X, scaler_y, macro_keywords, sectors, macro_rates, events, last_year, y_df, expected_features, df = load_and_preprocess(uploaded_file, cache_key)
except (ValueError, FileNotFoundError, KeyError) as e:
    st.error(f"Error: {str(e)}")
    st.stop()

st.write(f"**Last year available in the data:** {last_year}")
st.write(f"**Number of features in X_df:** {X_df.shape[1]} (expected: {len(expected_features)})")

# Cache model structure
@st.cache_resource(show_spinner=False)
def get_model_structure(model_type, _cache_key="default"):
    if model_type == "Ridge":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=mutual_info_regression)),
            ('ridge', Ridge())
        ])
    elif model_type == "ElasticNet":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=mutual_info_regression)),
            ('elasticnet', ElasticNet())
        ])
    elif model_type == "Huber":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=mutual_info_regression)),
            ('huber', HuberRegressor(max_iter=1000))
        ])

# Define models
tscv = TimeSeriesSplit(n_splits=8)
ridge_params = {
    'ridge__alpha': np.logspace(-2, 3, 50),
    'feature_selection__k': [5, 10, 15, 20, 25]
}
ridge_cv = RandomizedSearchCV(
    get_model_structure("Ridge"),
    ridge_params,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_iter=20,
    random_state=42,
    n_jobs=-1
)

elasticnet_params = {
    'elasticnet__alpha': np.logspace(-2, 3, 50),
    'elasticnet__l1_ratio': np.linspace(0.1, 0.9, 9),
    'feature_selection__k': [5, 10, 15, 20, 25]
}
elasticnet_cv = RandomizedSearchCV(
    get_model_structure("ElasticNet"),
    elasticnet_params,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_iter=20,
    random_state=42,
    n_jobs=-1
)

huber_params = {
    'huber__epsilon': np.linspace(1.1, 2.0, 10),
    'huber__alpha': np.logspace(-4, 1, 20),
    'feature_selection__k': [5, 10, 15, 20, 25]
}
huber_cv = RandomizedSearchCV(
    get_model_structure("Huber"),
    huber_params,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_iter=20,
    random_state=42,
    n_jobs=-1
)

# Load or train models
def load_or_train_models(X, y, cache_key):
    default_file = "PIB_Trimestrielle.csv"
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_files = {
        "Ridge": os.path.join(model_dir, f"ridge_model_{cache_key}.joblib"),
        "ElasticNet": os.path.join(model_dir, f"elasticnet_model_{cache_key}.joblib"),
        "Huber": os.path.join(model_dir, f"huber_model_{cache_key}.joblib")
    }
    results = []
    models = {}
    test_maes = {}

    if uploaded_file is None and os.path.exists(default_file):
        file_hash = compute_file_hash(default_file)
        if cache_key == file_hash:
            for name, file_path in model_files.items():
                if os.path.exists(file_path):
                    try:
                        model_cv = joblib.load(file_path)
                        train_pred = model_cv.predict(X)
                        train_pred_unscaled = scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
                        y_unscaled = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
                        train_mae = mean_absolute_error(y_unscaled, train_pred_unscaled)
                        train_r2 = r2_score(y_unscaled, train_pred_unscaled)

                        preds_test = []
                        for tr, te in tscv.split(X):
                            best_model = model_cv.best_estimator_
                            best_model.fit(X[tr], y[tr])
                            preds_test.extend(best_model.predict(X[te]))

                        test_pred_unscaled = scaler_y.inverse_transform(np.array(preds_test).reshape(-1, 1)).flatten()
                        test_mae = mean_absolute_error(y_unscaled[-len(preds_test):], test_pred_unscaled)
                        test_r2 = r2_score(y_unscaled[-len(preds_test):], test_pred_unscaled)

                        st.markdown(f"### 🔍 Results for **{name}** (loaded from disk)")
                        st.write(f"Training MAE: {train_mae:.2f}, Test MAE (TimeSeriesSplit): {test_mae:.2f}")
                        st.write(f"Training R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
                        st.write(f"Best hyperparameters: {model_cv.best_params_}")

                        interpret_results(name, train_mae, test_mae, train_r2, test_r2)
                        results.append({
                            'Model': name,
                            'CV MAE': test_mae,
                            'Train R²': train_r2
                        })
                        models[name] = model_cv
                        test_maes[name] = test_mae
                        error_log.append(f"Model {name} loaded from {file_path}")
                    except Exception as e:
                        error_log.append(f"Failed to load model {name} from {file_path}: {str(e)}")
                        st.warning(f"Warning: Failed to load model {name}. Retraining...")
                        mae, r2, trained_model = eval_and_detect(globals()[f"{name.lower()}_cv"], X, y, name)
                        joblib.dump(trained_model, file_path)
                        results.append({
                            'Model': name,
                            'CV MAE': mae,
                            'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(trained_model.predict(X).reshape(-1, 1)))
                        })
                        models[name] = trained_model
                        test_maes[name] = mae
                else:
                    with st.spinner(f"Training {name}..."):
                        mae, r2, trained_model = eval_and_detect(globals()[f"{name.lower()}_cv"], X, y, name)
                        joblib.dump(trained_model, file_path)
                        results.append({
                            'Model': name,
                            'CV MAE': mae,
                            'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(trained_model.predict(X).reshape(-1, 1)))
                        })
                        models[name] = trained_model
                        test_maes[name] = mae
        else:
            for name, file_path in model_files.items():
                with st.spinner(f"Training {name}..."):
                    mae, r2, trained_model = eval_and_detect(globals()[f"{name.lower()}_cv"], X, y, name)
                    joblib.dump(trained_model, file_path)
                    results.append({
                        'Model': name,
                        'CV MAE': mae,
                        'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(trained_model.predict(X).reshape(-1, 1)))
                    })
                    models[name] = trained_model
                    test_maes[name] = mae
    else:
        for name, file_path in model_files.items():
            with st.spinner(f"Training {name}..."):
                mae, r2, trained_model = eval_and_detect(globals()[f"{name.lower()}_cv"], X, y, name)
                joblib.dump(trained_model, file_path)
                results.append({
                    'Model': name,
                    'CV MAE': mae,
                    'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(trained_model.predict(X).reshape(-1, 1)))
                })
                models[name] = trained_model
                test_maes[name] = mae

    return results, models, test_maes

# Evaluation and interpretation function
def interpret_results(model_name, train_mae, test_mae, train_r2, test_r2):
    rel_error = test_mae / np.mean(scaler_y.inverse_transform(y.reshape(-1, 1)))
    st.markdown("#### 💡 Interpretation")
    st.write(f"**Test R²:** {test_r2:.4f} — indicates generalization quality.")
    st.write(f"**Absolute MAE:** {test_mae:.0f} — for an average GDP ~{np.mean(scaler_y.inverse_transform(y.reshape(-1, 1))):,.0f}, i.e., a relative error of about **{rel_error*100:.1f}%**.")
    diff_r2 = train_r2 - test_r2
    if diff_r2 > 0.15:
        st.error("⚠️ Significant gap between training and test R² → possible overfitting.")
    else:
        st.success("✅ No obvious signs of overfitting.")

    st.markdown("#### ✅ Conclusion")
    if test_r2 >= 0.96 and rel_error < 0.03:
        st.write(f"✔️ **{model_name} delivers excellent results.**")
        st.write("- Can be used as a benchmark.")
        st.write("- Highly reliable for GDP forecasting.")
    elif test_r2 >= 0.90:
        st.write(f"✔️ **{model_name} is a good model,** but could be improved.")
    else:
        st.write(f"❌ **{model_name} shows limitations.** Consider another method or further tuning.")

def eval_and_detect(model_cv, X, y, model_name):
    model_cv.fit(X, y)
    train_pred = model_cv.predict(X)
    train_pred_unscaled = scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
    y_unscaled = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
    train_mae = mean_absolute_error(y_unscaled, train_pred_unscaled)
    train_r2 = r2_score(y_unscaled, train_pred_unscaled)

    preds_test = []
    for tr, te in tscv.split(X):
        best_model = model_cv.best_estimator_
        best_model.fit(X[tr], y[tr])
        preds_test.extend(best_model.predict(X[te]))

    test_pred_unscaled = scaler_y.inverse_transform(np.array(preds_test).reshape(-1, 1)).flatten()
    test_mae = mean_absolute_error(y_unscaled[-len(preds_test):], test_pred_unscaled)
    test_r2 = r2_score(y_unscaled[-len(preds_test):], test_pred_unscaled)

    st.markdown(f"### 🔍 Results for **{model_name}**")
    st.write(f"Training MAE: {train_mae:.2f}, Test MAE (TimeSeriesSplit): {test_mae:.2f}")
    st.write(f"Training R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    st.write(f"Best hyperparameters: {model_cv.best_params_}")

    interpret_results(model_name, train_mae, test_mae, train_r2, test_r2)
    return test_mae, test_r2, model_cv

# Run models
st.header("📊 Model Diagnostics and Interpretation")
results, models, test_maes = load_or_train_models(X, y, cache_key)

if not test_maes:
    st.error("Error: No models were trained or loaded. Please check the input data.")
    st.stop()

# Select best model
best_model_name = min(test_maes, key=test_maes.get)
best_model = models[best_model_name].best_estimator_
st.markdown(f"### 🏆 Selected Model: **{best_model_name}**")
st.write(f"The **{best_model_name}** model was chosen as it has the lowest MAE: {test_maes[best_model_name]:.2f}")

# Verify selected model
st.header("🔎 Verification of the Selected Model")
st.markdown("#### 1. Data Integrity Check")
if X_df.isna().any().any():
    error_log.append("Missing values detected in X_df.")
    st.error("Error: Missing values in input data. Replacing with 0.")
    X_df = X_df.fillna(0)
if y_df.isna().any().any():
    error_log.append("Missing values detected in y_df.")
    st.warning("Warning: Missing values in target data. Replacing with mean.")
    y_df = y_df.fillna(y_df.mean())
if y_df.empty or y_df.shape[0] == 0:
    error_log.append("y_df is empty or has no rows.")
    st.error("Error: Target data (y_df) is empty. Stopping program.")
    st.stop()
st.success(f"Success: No missing values in data after preprocessing. y_df shape: {y_df.shape}")

st.markdown("#### 2. Test Set Verification")
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
best_model.fit(X_train, y_train)
y_pred_test = best_model.predict(X_test)
y_pred_test_unscaled = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
test_mae = mean_absolute_error(y_test_unscaled, y_pred_test_unscaled)
test_r2 = r2_score(y_test_unscaled, y_pred_test_unscaled)
st.write(f"Test set MAE: {test_mae:.2f}")
st.write(f"Test set R²: {test_r2:.4f}")
if test_mae > 1.5 * test_maes[best_model_name]:
    error_log.append(f"Test set MAE ({test_mae:.2f}) significantly higher than CV MAE ({test_maes[best_model_name]:.2f}).")
    st.warning("Warning: Test set performance is worse than expected.")

st.markdown("#### 3. Residual Analysis")
residuals = y_test_unscaled - y_pred_test_unscaled
fig_residuals = px.scatter(
    x=range(len(residuals)),
    y=residuals,
    title="Residuals on Test Set",
    labels={'x': 'Index', 'y': 'Residuals (million TND)'},
    color_discrete_sequence=['#FF6B6B'],
    render_mode='webgl'
)
fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
st.plotly_chart(fig_residuals, use_container_width=True)
if np.abs(residuals).mean() > test_maes[best_model_name]:
    error_log.append(f"Average residuals ({np.abs(residuals).mean():.2f}) are high compared to CV MAE ({test_maes[best_model_name]:.2f}).")
    st.warning("Warning: Residuals show a high average error, indicating possible underperformance.")

st.markdown("#### 4. Prediction Intervals")
n_bootstraps = 50
bootstrap_preds = []
for _ in range(n_bootstraps):
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    best_model.fit(X_train[indices], y_train[indices])
    pred = best_model.predict(X_test)
    bootstrap_preds.append(scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten())
bootstrap_preds = np.array(bootstrap_preds)
lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)
upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)
st.write("95% Prediction Intervals for Test Set:")
for i, (lower, upper, actual) in enumerate(zip(lower_bound, upper_bound, y_test_unscaled)):
    st.write(f"Quarter {i+1}: Predicted = {y_pred_test_unscaled[i]:,.0f}, Interval = [{lower:,.0f}, {upper:,.0f}], Actual = {actual:,.0f}")

# Predict for 2024
if st.button("🔮 Predict GDP for 2024"):
    with st.spinner("Training and predicting..."):
        start_time = time.time()
        quarters_to_predict = ["Q1", "Q2", "Q3", "Q4"]
        base_quarter = max(X_df.index)
        historical_df = pd.DataFrame({'Quarter': [str(q) for q in quarters], 'GDP': scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()})
        pred_df = pd.DataFrame({'Quarter': [f"{q} 2024" for q in quarters_to_predict], 'GDP': [0.0] * 4})
        combined_df = pd.concat([historical_df, pred_df], ignore_index=True)

        quarterly_predictions = []
        feature_vectors = []
        current_base_quarter = base_quarter
        current_base_data = X_df.loc[current_base_quarter][expected_features].copy()

        recent_data = X_df[expected_features].tail(4)
        growth_rates = {}
        for col in sectors + macro_rates:
            if col in recent_data.columns:
                quarter_growth = recent_data[col].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                growth_rates[col] = quarter_growth.mean() * 100 if not quarter_growth.empty else 0.0
            else:
                growth_rates[col] = 0.0
                error_log.append(f"Growth rate for '{col}' not calculated (column missing). Using 0.")
        for event in events:
            if event in recent_data.columns:
                growth_rates[event] = recent_data[event].iloc[-1] if not recent_data[event].empty else 0
            else:
                growth_rates[event] = 0
                error_log.append(f"Value for '{event}' not found. Using 0.")

        for i, q in enumerate(quarters_to_predict):
            feature_vector = pd.DataFrame(0.0, index=[0], columns=expected_features)
            logger.info(f"Creating feature vector for {q} with {len(feature_vector.columns)} columns: {list(feature_vector.columns)}")

            for sector in sectors:
                feature_vector[sector] = (
                    current_base_data[sector] * (1 + growth_rates.get(sector, 0.0) / 100)
                    if sector in X_df.columns else 0.0
                )
                if sector not in X_df.columns:
                    error_log.append(f"Sector '{sector}' not found for {q}. Using 0.")

            for rate in macro_rates:
                feature_vector[rate] = (
                    current_base_data[rate] * (1 + growth_rates.get(rate, 0.0) / 100)
                    if rate in X_df.columns else 0.0
                )
                if rate not in X_df.columns:
                    error_log.append(f"Rate '{rate}' not found for {q}. Using 0.")

            for event in events:
                feature_vector[event] = growth_rates.get(event, 0.0)
                if event not in X_df.columns:
                    error_log.append(f"Event '{event}' not found for {q}. Using 0.")

            for col in expected_features:
                if col.endswith('_lag1'):
                    base_col = col.replace('_lag1', '')
                    feature_vector[col] = (
                        current_base_data.get(base_col, X_df[base_col].mean() if base_col in X_df.columns else 0.0)
                        if base_col in feature_vector.columns else
                        current_base_data.get(col, X_df[col].mean() if col in X_df.columns else 0.0)
                    )
                    if feature_vector[col].iloc[0] == 0.0:
                        error_log.append(f"Lagged feature '{col}' for {q} set to 0.")

            if list(feature_vector.columns) != expected_features:
                error_log.append(f"Mismatch in feature_vector columns for {q}: Expected {len(expected_features)} columns, Got {len(feature_vector.columns)} columns: {list(feature_vector.columns)}")
                st.error(f"Error: feature_vector columns ({len(feature_vector.columns)}) do not match expected_features ({len(expected_features)}).")
                st.stop()

            if feature_vector.isna().any().any():
                error_log.append(f"NaN values for {q}: {feature_vector.columns[feature_vector.isna().any()].tolist()}. Replacing with 0.")
                feature_vector = feature_vector.fillna(0.0)

            X_new = scaler_X.transform(feature_vector)
            feature_vectors.append(X_new)

            predicted_gdp = float(scaler_y.inverse_transform(best_model.predict(X_new).reshape(-1, 1))[0])
            quarterly_predictions.append(predicted_gdp)
            combined_df.loc[combined_df['Quarter'] == f"{q} 2024", 'GDP'] = predicted_gdp

            current_base_data = feature_vector.iloc[0].copy()

        yearly_gdp = sum(quarterly_predictions)
        st.markdown("### 📈 Prediction Results")
        st.write(f"**Model used:** {best_model_name}")
        st.write("**Quarterly predictions:**")
        for i, q in enumerate(quarters_to_predict):
            st.write(f"- **{q} 2024**: {quarterly_predictions[i]:,.0f} million TND")
        st.write(f"**Estimated annual GDP for 2024**: {yearly_gdp:,.0f} million TND")

        y_pred_historical = best_model.predict(X)
        y_pred_historical_unscaled = scaler_y.inverse_transform(y_pred_historical.reshape(-1, 1)).flatten()
        y_historical_unscaled = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
        historical_df = pd.DataFrame({
            'Quarter': [str(q) for q in quarters],
            'Actual GDP': y_historical_unscaled,
            'Predicted GDP': y_pred_historical_unscaled
        })
        pred_df = pd.DataFrame({
            'Quarter': [f"{q} 2024" for q in quarters_to_predict],
            'Actual GDP': [np.nan] * 4,
            'Predicted GDP': quarterly_predictions
        })
        combined_df = pd.concat([historical_df, pred_df], ignore_index=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=combined_df['Quarter'],
            y=combined_df['Actual GDP'],
            mode='lines+markers',
            name='Actual GDP',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=combined_df['Quarter'],
            y=combined_df['Predicted GDP'],
            mode='lines+markers',
            name='Predicted GDP',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title='Historical GDP vs Predictions (incl. 2024)',
            xaxis_title='Quarter',
            yaxis_title='GDP (million TND)',
            xaxis_tickangle=45,
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧠 Prediction Explanation with SHAP")
        st.write("The following plots explain how each feature contributes to the GDP predictions for 2024.")
        best_model.fit(X, y)
        feature_vectors_for_shap = np.vstack(feature_vectors)
        error_log.append(f"Shape of feature_vectors_for_shap: {feature_vectors_for_shap.shape}")
        background_data = scaler_X.transform(X_df[expected_features].iloc[:20])
        error_log.append(f"Shape of background_data: {background_data.shape}")

        try:
            if best_model_name in ["Ridge", "ElasticNet"]:
                explainer = shap.LinearExplainer(
                    best_model,
                    background_data,
                    feature_names=expected_features
                )
            else:
                explainer = shap.KernelExplainer(
                    best_model.predict,
                    background_data,
                    feature_names=expected_features,
                    nsamples=100
                )

            shap_values = explainer.shap_values(feature_vectors_for_shap)
            error_log.append(f"Shape of shap_values: {np.array(shap_values).shape}")

            st.markdown("#### 📊 Global Feature Importance (Summary Plot)")
            plt.figure(figsize=(10, 6), dpi=80)
            shap.summary_plot(shap_values, feature_vectors_for_shap, feature_names=expected_features, show=False)
            st.pyplot(plt)
            plt.close()

            st.markdown("#### 📊 Feature Importance by Quarter")
            for i, q in enumerate(quarters_to_predict):
                st.write(f"**{q} 2024**")
                plt.figure(figsize=(10, 6), dpi=80)
                shap.bar_plot(shap_values[i], feature_names=expected_features, max_display=10, show=False)
                st.pyplot(plt)
                plt.close()

        except Exception as e:
            error_log.append(f"Error calculating SHAP: {str(e)}")
            st.error(f"Error: Unable to generate SHAP explanations: {str(e)}. Please check the data.")

        st.info(f"🧪 Prediction based on the {best_model_name} model with the lowest MAE, using historical trends extrapolated from the last 4 quarters.")
        logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")

        show_errors = st.checkbox("Show log", value=True)
        if show_errors and error_log:
            st.markdown("### Informative Log")
            for error in error_log:
                st.write(error)