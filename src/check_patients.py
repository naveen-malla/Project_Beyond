import pandas as pd

# Read all CSV files
deliveries_df = pd.read_excel('data/Ampath Deliveries.xlsx')
ampath_patients_list_df = pd.read_csv('data/AI_Predictive_Modeling_HIV_AMPATH_Data.csv')
kasha_patients_list_df = pd.read_excel('data/Patient List.xlsx') 

# intersect Customer No. column from deliveries_df and kasha_patients_list_df
deliveries_patients = set(deliveries_df['Customer No.'])
kasha_patients = set(kasha_patients_list_df['Customer No.'])

# Check if there are any patients in deliveries_df that are not in kasha_patients_list_df
missing_patients = deliveries_patients - kasha_patients
if missing_patients:
    print(f"Missing patients in Kasha list: {missing_patients}")
else:  
    print("All patients in deliveries_df are present in kasha_patients_list_df.")

# intersect person_id column from ampath_patients_list_df and EMR Person Id from kasha_patients_list_df
ampath_patients = set(ampath_patients_list_df['person_id'])
kasha_patients = set(kasha_patients_list_df['EMR Person Id'])

# Check if there are any patients in kasha_patients_list_df that are not in ampath_patients_list_df
missing_patients = kasha_patients - ampath_patients
if missing_patients:
    print(f"Missing patients in AMPATH list: {len(missing_patients)} patients out of {len(kasha_patients)}")
else:
    print("All patients in kasha_patients_list_df are present in ampath_patients_list_df.")


# convert Created On column to datetime in kasha_patients_list_df
kasha_patients_list_df['Created On'] = pd.to_datetime(kasha_patients_list_df['Created On'], errors='coerce')

# Find out how many records are present that are created after february 14, 2024 and store the count
after_feb_14 = kasha_patients_list_df[kasha_patients_list_df['Created On'] > '2024-02-14']
after_feb_14_count = after_feb_14.shape[0]
print(f"Number of records created after February 14, 2024: {after_feb_14_count}")
