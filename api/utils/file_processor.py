
import pandas as pd
import logging
from typing import List, Dict, Union
import re

def identify_file_type(df: pd.DataFrame) -> str:
    """Identify the type of CSV file based on its columns."""
    columns = set(df.columns.str.lower())
    
    if {'query', 'page', 'clicks', 'impressions', 'ctr', 'position'}.issubset(columns):
        if 'device' in columns:
            return 'devices'
        elif 'country' in columns:
            return 'countries'
        elif 'date' in columns:
            return 'dates'
        else:
            return 'queries'
    elif {'page', 'clicks', 'impressions', 'ctr', 'position'}.issubset(columns):
        return 'pages'
    elif {'url', 'title', 'meta_description'}.issubset(columns):
        return 'meta_data'
    elif {'url', 'mobile_usability', 'mobile_friendly_score'}.issubset(columns):
        return 'mobile'
    elif {'url', 'backlinks', 'domain_authority'}.issubset(columns):
        return 'backlinks'
    elif {'feature', 'query', 'position', 'url'}.issubset(columns):
        return 'serp_features'
    else:
        return 'unknown'

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to English and consistent format."""
    # Common column mappings
    column_map = {
        # Query data
        'search query': 'query',
        'keyword': 'query',
        'search term': 'query',
        'term': 'query',
        
        # Page data
        'landing page': 'page',
        'url': 'page',
        'web page': 'page',
        
        # Metrics
        'click': 'clicks',
        'impression': 'impressions',
        'view': 'impressions',
        'average position': 'position',
        'avg. position': 'position',
        'avg position': 'position',
        'rank': 'position',
        
        # Geography
        'country/territory': 'country',
        'region': 'country',
        'location': 'country',
        
        # Device
        'device type': 'device',
        'device category': 'device',
        
        # Date
        'date range': 'date',
        'day': 'date',
        
        # Mobile
        'mobile friendly': 'mobile_friendly',
        'mobile score': 'mobile_friendly_score',
        'usability': 'mobile_usability',
        
        # Backlinks
        'external links': 'backlinks',
        'referring domains': 'referring_domains',
        'authority': 'domain_authority',
        'da': 'domain_authority',
        
        # SERP features
        'serp feature': 'feature',
        'feature type': 'feature'
    }
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Standardize column names (lowercase and strip spaces)
    df_copy.columns = [col.lower().strip() for col in df_copy.columns]
    
    # Replace column names based on mapping
    for old_col, new_col in column_map.items():
        if old_col in df_copy.columns and old_col != new_col:
            df_copy.rename(columns={old_col: new_col}, inplace=True)
    
    return df_copy

def process_csv_files(files: List[Dict[str, Union[str, bytes]]]) -> Dict[str, pd.DataFrame]:
    """Process multiple CSV files and organize them by type."""
    processed_data = {}
    
    for file_info in files:
        try:
            # Read CSV file
            df = pd.read_csv(file_info['content'])
            
            # Determine file type
            file_type = identify_file_type(df)
            
            if file_type != 'unknown':
                # Standardize column names to English
                df = standardize_column_names(df)
                
                # If a file type already exists, append or merge data
                if file_type in processed_data:
                    processed_data[file_type] = pd.concat([processed_data[file_type], df], ignore_index=True)
                else:
                    processed_data[file_type] = df
            else:
                logging.warning(f"Unknown file format: {file_info['filename']}")
        except Exception as e:
            logging.error(f"Error processing file {file_info['filename']}: {str(e)}")
    
    return processed_data
