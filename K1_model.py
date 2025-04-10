"""
HIV Treatment Adherence Prediction Pipeline
Author: Naveen Malla
Date: 2025-04-10
Description: Predicts appointment adherence and risk levels for HIV patients
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Load and preprocess the feature-engineered dataset
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Processed DataFrame with datetime conversions
    """
    try:
        df = pd.read_csv(file_path, parse_dates=[
            'birthdate', 'Encounter_Date', 
            'Next_clinical_appointment', 'Diagnosis_Date'
        ])
        logging.info(f"Data loaded successfully with {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def calculate_target(df):
    """
    Create target variable for adherence prediction
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with target column added
    """
    try:
        # Ensure proper datetime conversion with error handling
        df['Encounter_Date'] = pd.to_datetime(
            df['Encounter_Date'], 
            errors='coerce',
            format='%Y-%m-%d %H:%M:%S'
        )
        df['Next_clinical_appointment'] = pd.to_datetime(
            df['Next_clinical_appointment'], 
            errors='coerce',
            format='%Y-%m-%d %H:%M:%S'
        )
        
        # Verify datetime conversion
        if df['Encounter_Date'].isna().any() or df['Next_clinical_appointment'].isna().any():
            logging.warning("NaT values detected in datetime columns after conversion")
            
        # Calculate days difference (handling negative values)
        df['days_diff'] = (df['Next_clinical_appointment'] - df['Encounter_Date']).dt.days
        
        # Define target: 1 if >28 days late (IIT), 0 otherwise
        df['adherence_target'] = np.where(df['days_diff'] > 28, 1, 0)
        
        return df.drop(columns=['days_diff'])
    
    except Exception as e:
        logging.error(f"Error in calculate_target: {str(e)}")
        raise


def temporal_split(df, date_col='Encounter_Date'):
    """
    Split data temporally into train/validation/test sets
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Date column for splitting
        
    Returns:
        tuple: (train, val, test) DataFrames
    """
    try:
        train = df[df[date_col].dt.year <= 2020]
        val = df[df[date_col].dt.year == 2021]
        test = df[df[date_col].dt.year >= 2022]
        
        logging.info(f"Temporal split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test
    except Exception as e:
        logging.error(f"Error in temporal split: {str(e)}")
        raise

def create_preprocessor():
    """
    Create preprocessing pipeline for model features
    
    Returns:
        ColumnTransformer: Configured preprocessing pipeline
    """
    numeric_features = [
        'num_past_iits', 'pct_late_arrivals',
        'CD4_Count', 'Viral_Load', 'age_at_scheduled_appointment'
    ]
    
    categorical_features = [
        'Current_WHO_HIV_Stage', 'gender'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def train_adherence_model(X_train, y_train, preprocessor):
    """
    Train logistic regression model for adherence prediction
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        preprocessor (ColumnTransformer): Fitted preprocessor
        
    Returns:
        tuple: (fitted model, fitted preprocessor)
    """
    try:
        # Create pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        logging.info("Adherence model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def calculate_risk_tiers(df):
    """
    Calculate risk tiers based on historical behavior at patient level
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with risk tiers added
    """
    try:
        # Aggregate patient-level features
        patient_stats = df.groupby('person_id').agg({
            'num_past_iits': 'max',        # Total IITs for patient
            'pct_late_arrivals': 'last',   # Latest adherence percentage
            'past_encounters': 'max'       # Total appointment count
        }).reset_index()
        
        # Define risk conditions
        conditions = [
            (patient_stats['num_past_iits'] >= 3) | (patient_stats['pct_late_arrivals'] >= 30),
            (patient_stats['num_past_iits'] >= 1) | (patient_stats['pct_late_arrivals'] >= 15),
            (patient_stats['num_past_iits'] == 0) & (patient_stats['pct_late_arrivals'] < 15)
        ]
        
        choices = ['High', 'Medium', 'Low']
        patient_stats['risk_tier'] = np.select(conditions, choices, default='Medium')
        
        # Merge back to original dataframe
        df = df.merge(
            patient_stats[['person_id', 'risk_tier']],
            on='person_id',
            how='left'
        )
        
        logging.info(f"Unique patients in risk tiers: {patient_stats['risk_tier'].value_counts().to_dict()}")
        return df
        
    except Exception as e:
        logging.error(f"Error calculating risk tiers: {str(e)}")
        raise

def evaluate_model(model, X_val, y_val):
    """
    Evaluate model performance on validation set
    
    Args:
        model: Trained model
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        # Generate predictions
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_val, y_pred, output_dict=True)
        metrics = {
            'auc_roc': roc_auc_score(y_val, y_proba),
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }
        
        logging.info(f"Validation AUC-ROC: {metrics['auc_roc']:.2f}")
        return metrics
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        raise

def save_artifacts(model, preprocessor, file_prefix):
    """
    Save model and preprocessor artifacts
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        file_prefix (str): Path prefix for saving files
    """
    try:
        joblib.dump(model, f"{file_prefix}_model.pkl")
        joblib.dump(preprocessor, f"{file_prefix}_preprocessor.pkl")
        logging.info("Artifacts saved successfully")
    except Exception as e:
        logging.error(f"Error saving artifacts: {str(e)}")
        raise

def main():
    """Main execution pipeline"""
    try:
        # 1. Load and prepare data
        df = load_data("data/AI Predictive Modeling HIV AMPATH Feature Engineering.csv")
        df = calculate_target(df)
        
        # 2. Temporal split
        train, val, test = temporal_split(df)
        
        # 3. Calculate risk tiers
        train = calculate_risk_tiers(train)
        val = calculate_risk_tiers(val)
        
        # 4. Prepare model features
        features = [
            'num_past_iits', 'pct_late_arrivals', 'CD4_Count',
            'Viral_Load', 'age_at_scheduled_appointment',
            'Current_WHO_HIV_Stage', 'gender'
        ]
        
        # 5. Train adherence model
        preprocessor = create_preprocessor()
        model = train_adherence_model(
            train[features], 
            train['adherence_target'],
            preprocessor
        )
        
        # 6. Evaluate model
        metrics = evaluate_model(model, val[features], val['adherence_target'])
        
        # 7. Save artifacts
        save_artifacts(model, preprocessor, "hiv_adherence_model")
        
        return {
            'risk_tier_distribution': val['risk_tier'].value_counts().to_dict(),
            'model_metrics': metrics
        }
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
    print("\nFinal Results:")
    print(f"Risk Tier Distribution: {results['risk_tier_distribution']}")
    print(f"Model Metrics: {results['model_metrics']}")
