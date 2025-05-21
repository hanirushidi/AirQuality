import pandas as pd
import os

def load_and_preview_data(file_path, num_rows=10):
    """
    Loads a CSV file into a pandas DataFrame and returns a preview.
    The UCI dataset uses ';' as a delimiter and ',' as decimal.
    Missing values are tagged as -200. <mcreference link="https://archive.ics.uci.edu/ml/datasets/Air+Quality" index="0">0</mcreference>
    """
    try:
        # The dataset specific parsing:
        # - Delimiter is ';'
        # - Decimal is ','
        # - Missing values are -200, replace them with NaN for easier handling
        df = pd.read_csv(file_path, sep=';', decimal=',', na_values=[-200])
        
        # Drop empty columns that might result from trailing semicolons in rows
        df.dropna(axis=1, how='all', inplace=True)
        
        # Preview
        preview_df = df.head(num_rows)
        return preview_df, None
    except FileNotFoundError:
        return None, f"Error: File not found at {file_path}"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def get_full_data(file_path=None):
    """
    Loads the full CSV dataset.
    """
    if file_path is None:
        # Construct default path relative to this file's location if not provided
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Moves up to project root
        file_path = os.path.join(base_dir, 'data', 'AirQualityUCI.csv')

    try:
        df = pd.read_csv(file_path, sep=';', decimal=',', na_values=[-200])
        df.dropna(axis=1, how='all', inplace=True)
        return df, None
    except FileNotFoundError:
        return None, f"Error: File not found at {file_path}"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"