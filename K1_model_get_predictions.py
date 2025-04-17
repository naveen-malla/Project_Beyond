#!/usr/bin/env python3
"""
HIV Treatment Adherence Prediction Script
Author: Naveen Malla
Date: 2025-04-15
Description: Standalone script to make predictions on new feature-engineered data
"""

import pandas as pd
import numpy as np
import joblib
import logging
import argparse
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"prediction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Make predictions on new HIV patient data')
    parser.add_argument('--input', type=str, default='data/appointments_202401_to_202401.csv',
                        help='Path to input feature-engineered data CSV')
    parser.add_argument('--output', type=str, default='data/appointments_202401_to_202401_predictions.csv',
                        help='Path to save predictions CSV')
    parser.add_argument('--model', type=str, default='hiv_adherence_model_model.pkl',
                        help='Path to saved model pickle file')
    parser.add_argument('--preprocessor', type=str, default='hiv_adherence_model_preprocessor.pkl',
                        help='Path to saved preprocessor pickle file')
    return parser.parse_args()

def load_data(file_path):
    """
    Load feature-engineered dataset
    
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

def load_model_artifacts(model_path, preprocessor_path):
    """
    Load saved model and preprocessor
    
    Args:
        model_path (str): Path to model pickle file
        preprocessor_path (str): Path to preprocessor pickle file
        
    Returns:
        tuple: (model, preprocessor)
    """
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logging.info("Model and preprocessor loaded successfully")
        return model, preprocessor
    except Exception as e:
        logging.error(f"Error loading model artifacts: {str(e)}")
        raise

def make_predictions(df, model, features):
    """
    Generate predictions for input data
    
    Args:
        df (pd.DataFrame): Input data
        model: Loaded model
        features (list): Feature columns used by model
        
    Returns:
        pd.DataFrame: DataFrame with predictions added
    """
    try:
        # Check for missing features
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logging.error(f"Missing required features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Make predictions
        df['predicted_iit_prob'] = model.predict_proba(df[features])[:, 1]
        df['adherence_predicted'] = model.predict(df[features])
        
        logging.info(f"Generated predictions for {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise

def main():
    """Main execution pipeline"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # 1. Load data
        df = load_data(args.input)
        
        # 2. Load model and preprocessor
        model, preprocessor = load_model_artifacts(args.model, args.preprocessor)
        
        # 3. Define features (must match those used in training)
        features = [
            'num_past_iits', 'pct_late_arrivals', 
            'CD4_Count', 'Viral_Load', 'age_at_scheduled_appointment',
            'Current_WHO_HIV_Stage', 'gender'
        ]
        
        # 4. Make predictions
        # Note: The model already includes the preprocessor in its pipeline
        df_with_predictions = make_predictions(df, model, features)
        
        # 5. Save results
        output_columns = [
            'person_id', 'gender',  'Encounter_Date', 'Next_clinical_appointment', 'adherence_predicted', 
            'next_actual_visit', 'days_diff', 'adherence_actual', 'overall_appointment_success'
        ]
        
        df_with_predictions[output_columns].to_csv(args.output, index=False)
        logging.info(f"Predictions saved to {args.output}")
        
        # 6. Summary statistics
        prediction_counts = df_with_predictions['adherence_predicted'].value_counts()
        logging.info(f"Prediction summary: {prediction_counts.to_dict()}")
        
        return df_with_predictions
        
    except Exception as e:
        logging.error(f"Prediction pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
