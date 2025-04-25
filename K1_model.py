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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

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

def train_random_forest_model(X_train, y_train, preprocessor):
    """
    Train Random Forest model for adherence prediction
    """
    try:
        # Create pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1  # Use all available cores
            ))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        logging.info("Random Forest model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training Random Forest model: {str(e)}")
        raise

def train_xgboost_model(X_train, y_train, preprocessor):
    """
    Train XGBoost model for adherence prediction
    """
    try:
        # Create pipeline with proper XGBoost parameters
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                tree_method='hist',  # More efficient tree method
                random_state=42
            ))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        logging.info("XGBoost model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training XGBoost model: {str(e)}")
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

def temporal_split(df):
    """
    Split patients (not encounters) temporally based on their last encounter
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        tuple: (train, val, test) DataFrames
    """
    try:
        # Get latest encounter per patient
        patient_last_encounter = df.groupby('person_id')['Encounter_Date'].max()
        
        # Split patients based on last encounter year
        train_patients = patient_last_encounter[patient_last_encounter.dt.year <= 2020].index
        val_patients = patient_last_encounter[patient_last_encounter.dt.year == 2021].index
        test_patients = patient_last_encounter[patient_last_encounter.dt.year >= 2022].index
        
        # Split encounters accordingly
        train = df[df['person_id'].isin(train_patients)]
        val = df[df['person_id'].isin(val_patients)]
        test = df[df['person_id'].isin(test_patients)]
        
        logging.info(f"Patient-based split - Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
        return train, val, test
    except Exception as e:
        logging.error(f"Error in temporal split: {str(e)}")
        raise


def create_sample_predictions(model, test_data, sample_size=100, output_path="sample_predictions.csv"):
    """
    Generate sample predictions for demonstration and analysis
    
    Args:
        model: Trained model
        test_data (pd.DataFrame): Test dataset
        sample_size (int): Number of unique patients to sample (one record per patient)
        output_path (str): Path to save sample predictions
        
    Returns:
        pd.DataFrame: Sample predictions with key columns (one row per patient)
    """
    try:
        logging.info("Creating sample predictions...")
        
        # Define required columns that must not have missing values
        required_columns = [
            'Next_clinical_appointment', 'CD4_Count', 'Viral_Load', 
            'num_past_iits', 'pct_late_arrivals'
        ]
        
        # Filter test data to only include rows with no missing values in required columns
        complete_data = test_data.dropna(subset=required_columns)
        
        # Get unique patients from complete data
        unique_patients = complete_data['person_id'].unique()
        
        # Sample patients (if we have enough)
        sample_size = min(sample_size, len(unique_patients))
        sampled_patients = np.random.choice(unique_patients, size=sample_size, replace=False)
        
        # Create empty DataFrame to hold complete records
        sample_data = pd.DataFrame()
        
        # For each sampled patient, get their latest complete encounter
        for patient_id in sampled_patients:
            patient_records = complete_data[complete_data['person_id'] == patient_id]
            
            if not patient_records.empty:
                # Get the latest complete record
                latest_record = patient_records.loc[patient_records['Encounter_Date'].idxmax()]
                sample_data = pd.concat([sample_data, pd.DataFrame([latest_record])], ignore_index=True)
        
        # Check if we have enough complete records
        if len(sample_data) < sample_size:
            logging.warning(f"Only found {len(sample_data)} complete patient records instead of {sample_size}")
        
        # Generate predictions
        features = [
            'num_past_iits', 'pct_late_arrivals', 'CD4_Count',
            'Viral_Load', 'age_at_scheduled_appointment',
            'Current_WHO_HIV_Stage', 'gender'
        ]
        
        # Add predictions to sample data
        sample_data['predicted_iit_prob'] = model.predict_proba(sample_data[features])[:, 1]

        # Select key columns for analysis
        output_cols = [
            'person_id', 'Encounter_Date', 'Next_clinical_appointment', 'overall_appointment_success', 'Current_WHO_HIV_Stage', 
         'risk_tier', 'adherence_target',
            'num_past_iits', 'pct_late_arrivals', 
            'CD4_Count', 'Viral_Load'
        ]

        # Save to CSV
        sample_data[output_cols].to_csv(output_path, index=False)
        logging.info(f"Saved sample predictions for exactly {len(sample_data)} patients to {output_path}")
        
        return sample_data[output_cols]

    except Exception as e:
        logging.error(f"Error creating sample predictions: {str(e)}")
        raise



def main():
    """Main execution pipeline"""
    try:
        # 1. Load and prepare data
        df = load_data("data/AI_Predictive_Modeling_HIV_AMPATH_FeatureEngineering.csv")
        df = calculate_target(df)
        
        # 2. Calculate risk tiers FIRST (patient-level)
        df = calculate_risk_tiers(df)
        
        # 3. Split patients temporally
        train, val, test = temporal_split(df)
        # Save full test data with risk tiers
        test.to_csv(
            "data/AI_Predictive_Modeling_HIV_AMPATH_FeatureEngineering_Risk_Tier_Test.csv",
            index=False,
            encoding='utf-8'
        )
        logging.info("Saved full test data with risk tiers")

        # 4. Prepare model features
        features = [
            'num_past_iits', 'pct_late_arrivals', 
            'CD4_Count', 'Viral_Load', 'age_at_scheduled_appointment',
            'Current_WHO_HIV_Stage', 'gender'
        ]
        
        # 5. Train all models
        preprocessor = create_preprocessor()
        
        # Train Logistic Regression
        lr_model = train_adherence_model(
            train[features], 
            train['adherence_target'],
            preprocessor
        )
        
        # Train XGBoost
        xgb_model = train_xgboost_model(
            train[features], 
            train['adherence_target'],
            preprocessor
        )
        
        # Train Random Forest
        rf_model = train_random_forest_model(
            train[features], 
            train['adherence_target'],
            preprocessor
        )
        
        # 6. Evaluate all models
        metrics = {
            'logistic_regression': evaluate_model(lr_model, val[features], val['adherence_target']),
            'xgboost': evaluate_model(xgb_model, val[features], val['adherence_target']),
            'random_forest': evaluate_model(rf_model, val[features], val['adherence_target'])
        }
        
        # 7. Save all artifacts
        joblib.dump(lr_model, "models/lr_model.pkl")
        joblib.dump(xgb_model, "models/xgb_model.pkl")
        joblib.dump(rf_model, "models/rf_model.pkl")
        joblib.dump(preprocessor, "models/preprocessor.pkl")
        logging.info("All models and preprocessor saved successfully")
        
        # 8. Create sample predictions (using best model based on AUC)
        best_model_name = max(metrics.items(), key=lambda x: x[1]['auc_roc'])[0]
        best_model_dict = {
            'logistic_regression': lr_model,
            'xgboost': xgb_model,
            'random_forest': rf_model
        }
        
        _ = create_sample_predictions(
            model=best_model_dict[best_model_name],
            test_data=test,
            sample_size=100,
            output_path="data/sample_predictions.csv"
        )
        
        return {
            'patient_risk_distribution': df.groupby('person_id')['risk_tier'].first().value_counts().to_dict(),
            'model_metrics': metrics,
            'best_model': best_model_name
        }
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
    print("\nFinal Results:")
    print(f"Risk Tier Distribution: {results['patient_risk_distribution']}")
    print(f"Best Model: {results['best_model']}")
    print("\nModel Metrics:")
    for model_name, metrics in results['model_metrics'].items():
        print(f"\n{model_name.title()}:")
        print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1: {metrics['f1']:.3f}")
