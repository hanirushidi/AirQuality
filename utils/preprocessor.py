import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Main function to orchestrate preprocessing steps.
    """
    # 1. Combine Date and Time, convert to datetime
    if 'Date' in df.columns and 'Time' in df.columns:
        try:
            # Ensure Date and Time are strings before concatenation
            df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), format='%d/%m/%Y %H.%M.%S', errors='coerce')
            # Drop rows where DateTime conversion failed
            df.dropna(subset=['DateTime'], inplace=True)
            # Set DateTime as index
            df.set_index('DateTime', inplace=True)
            # Drop original Date and Time columns
            df.drop(columns=['Date', 'Time'], inplace=True, errors='ignore')
        except Exception as e:
            print(f"Error processing Date/Time: {e}")
            # Fallback or error handling if columns are not as expected
            pass # Or raise error

    # 2. Handle missing values (example: interpolation for numeric columns)
    # Identify numeric columns for interpolation (excluding GT target variables if needed for specific tasks)
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        # Fix interpolation warning
        df[col] = df[col].interpolate(method='linear', limit_direction='both')  # Remove inplace
        
        # Fix fillna warning
        df[col] = df[col].fillna(df[col].mean())  # Remove inplace
        
    # Convert datetime column
    # df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time']) # <--- REMOVE THIS LINE
    # 3. Aggregate to daily average (example)
    # Ensure all relevant columns are numeric before resampling
    # df_daily = df.resample('D').mean() # This might fail if non-numeric columns are present that can't be averaged
    
    # Select only numeric columns for resampling mean
    numeric_df_for_resample = df.select_dtypes(include=np.number)
    df_daily = numeric_df_for_resample.resample('D').mean()
    
    # Any other specific cleaning steps can be added here.
    
    return df, df_daily # Return both processed hourly and daily aggregated