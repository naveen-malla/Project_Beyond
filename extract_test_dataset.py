#!/usr/bin/env python3
"""
HIV Appointment Data Extraction Pipeline

Author: Naveen Malla
Date: 2025-04-17

Description:
    This script extracts patient appointments scheduled within a given date range,
    matches them to the next actual visit, and calculates appointment adherence metrics.
    It ensures temporal consistency per patient and outputs a clean, analysis-ready dataset.

Usage:
    - Set the FILTER_START and FILTER_END variables (format: YYYY-MM)
    - Run the script. Output will be saved with a descriptive filename.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm

# ================== USER CONFIGURATION ==================
# Set your desired date range (inclusive), format: "YYYY-MM"
FILTER_START = "2024-01"
FILTER_END = "2024-01"
# Set your input and output directories
INPUT_FILE = "data/AI_Predictive_Modeling_HIV_AMPATH_FeatureEngineering_Risk_Tier_Test.csv"
OUTPUT_DIR = "data/"
# ========================================================

def setup_logging():
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{OUTPUT_DIR}appointment_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

def load_data(file_path):
    """
    Loads the dataset in chunks for memory efficiency and applies optimal dtypes.
    Returns the full DataFrame.
    """
    dtypes = {
        'person_id': 'category',
        'gender': 'category',
        'Current_WHO_HIV_Stage': 'category',
        'age_group': 'category'
    }
    date_cols = ['birthdate', 'Encounter_Date', 'Next_clinical_appointment', 'Diagnosis_Date']
    logging.info("Loading data in chunks...")
    chunk_iter = pd.read_csv(
        file_path,
        dtype=dtypes,
        parse_dates=date_cols,
        chunksize=100000
    )
    df = pd.concat([chunk for chunk in tqdm(chunk_iter, desc='Loading chunks')])
    logging.info(f"Data loaded. Shape: {df.shape}")
    return df

def validate_chronological_order(df):
    """
    Ensures that each patient's encounters are in chronological order.
    Removes any records that violate this order to maintain data integrity.
    """
    logging.info("Validating temporal sequence per patient...")
    df = df.sort_values(['person_id', 'Encounter_Date'])
    df['valid_sequence'] = (df.groupby('person_id')['Encounter_Date']
                            .diff() >= pd.Timedelta(days=0))
    invalid_records = df[~df['valid_sequence']]
    if not invalid_records.empty:
        logging.warning(f"Removed {len(invalid_records)} temporally inconsistent records")
        df = df[df['valid_sequence']]
    return df.drop(columns=['valid_sequence'])

def filter_by_appointment_date(df, start_month, end_month):
    """
    Filters the DataFrame to include only those appointments whose
    Next_clinical_appointment falls within the specified date range (inclusive).
    """
    logging.info(f"Filtering appointments for {start_month} to {end_month}")
    df['Next_clinical_appointment'] = pd.to_datetime(
        df['Next_clinical_appointment'], errors='coerce'
    )
    start_date = pd.to_datetime(f"{start_month}-01")
    end_date = pd.to_datetime(f"{end_month}-01") + pd.offsets.MonthEnd(1)
    mask = (
        df['Next_clinical_appointment'].notna() &
        (df['Next_clinical_appointment'] >= start_date) &
        (df['Next_clinical_appointment'] <= end_date)
    )
    filtered_df = df[mask]
    logging.info(f"Found {len(filtered_df)} appointments in target range")
    return filtered_df

def calculate_visit_metrics(df):
    """
    For each appointment, finds the next actual visit for the same patient,
    computes the days difference, and classifies adherence using IIT threshold.
    """
    logging.info("Calculating visit metrics (next actual visit, days_diff, adherence)...")
    df['Encounter_Date'] = pd.to_datetime(df['Encounter_Date'], errors='coerce')
    
    # Sort and find next actual visit
    df = df.sort_values(['person_id', 'Encounter_Date'])
    df['next_actual_visit'] = df.groupby('person_id')['Encounter_Date'].shift(-1)
    
    # Filter out appointments without a subsequent visit
    df = df[df['next_actual_visit'].notna()]
    
    # Calculate metrics
    df['days_diff'] = (df['next_actual_visit'] - df['Next_clinical_appointment']).dt.days
    
    # Binary classification for adherence (IIT threshold at 28 days)
    df['adherence_actual'] = np.where(
        df['days_diff'] > 28,  # IIT definition
        "Late/IIT",  # Non-adherent (IIT)
        "On Time"   # Adherent (On Time/Early/Within 28 days)
    )
    
    return df



def get_output_filename(start_month, end_month, output_dir):
    """
    Generates a descriptive output filename based on the date range.
    """
    start_label = start_month.replace("-", "")
    end_label = end_month.replace("-", "")
    filename = f"appointments_{start_label}_to_{end_label}.csv"
    return output_dir + filename

def save_results(df, output_path):
    """
    Saves the processed DataFrame to CSV, including all relevant columns.
    """
    required_cols = [
        'person_id', 'Encounter_Date', 'Next_clinical_appointment',
        'next_actual_visit', 'days_diff', 'adherence_actual'
    ]

    # Add additional columns if present
    additional_cols = [
        'birthdate', 'risk_tier', 'gender', 'Diagnosis_Date', 'Current_WHO_HIV_Stage', 'age_at_scheduled_appointment',
        'overall_appointment_success', 'num_past_iits', 'pct_late_arrivals', 
            'CD4_Count', 'Viral_Load'
    ]
    output_cols = required_cols + [col for col in additional_cols if col in df.columns]
    df[output_cols].to_csv(output_path, index=False)
    logging.info(f"Saved {len(df)} records to {output_path}")

def main():
    """
    Main pipeline:
    - Loads data
    - Validates encounter sequence
    - Filters by appointment date range
    - Calculates visit metrics
    - Saves results with a descriptive filename
    """
    setup_logging()
    try:
        df = load_data(INPUT_FILE)
        df = validate_chronological_order(df)
        df = filter_by_appointment_date(df, FILTER_START, FILTER_END)
        df = calculate_visit_metrics(df)
        if not df.empty:
            output_file = get_output_filename(FILTER_START, FILTER_END, OUTPUT_DIR)
            save_results(df, output_file)
            logging.info(f"Successfully processed and saved {len(df)} appointments.")
        else:
            logging.warning("No valid appointments found in the specified date range.")
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
