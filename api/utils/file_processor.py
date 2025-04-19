
import pandas as pd
from typing import Dict, List, Union
import logging

def identify_file_type(df: pd.DataFrame) -> str:
    """Identifies the type of GSC export file based on its columns."""
    columns = set(df.columns.str.lower())
    
    file_types = {
        'queries': {'query', 'clicks', 'impressions', 'ctr', 'position'},
        'pages': {'page', 'clicks', 'impressions', 'ctr', 'position'},
        'devices': {'device', 'clicks', 'impressions', 'ctr', 'position'},
        'countries': {'country', 'clicks', 'impressions', 'ctr', 'position'},
        'dates': {'date', 'clicks', 'impressions', 'ctr', 'position'},
        'search_appearance': {'search_appearance_type', 'clicks', 'impressions', 'ctr', 'position'},
        'filters': {'filter_type', 'clicks', 'impressions', 'ctr', 'position'}
    }
    
    for file_type, required_cols in file_types.items():
        if required_cols.issubset(columns):
            return file_type
            
    return 'unknown'

def process_csv_files(files: List[Dict[str, Union[str, bytes]]]) -> Dict[str, pd.DataFrame]:
    """Process multiple CSV files and organize them by type."""
    processed_data = {}
    
    for file_info in files:
        try:
            df = pd.read_csv(file_info['content'])
            file_type = identify_file_type(df)
            if file_type != 'unknown':
                processed_data[file_type] = df
        except Exception as e:
            logging.error(f"Error processing file {file_info['filename']}: {str(e)}")
    
    return processed_data

