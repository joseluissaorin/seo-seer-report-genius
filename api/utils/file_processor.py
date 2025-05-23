import pandas as pd
import logging
from typing import List, Dict, Union, IO
import re
from io import BytesIO, StringIO

# Define standard column sets for identification
METRIC_COLS = {'clicks', 'impressions', 'ctr', 'position'}
QUERY_PAGE_COLS = {'query', 'page'}
QUERY_ONLY_COLS = {'query'}
PAGE_ONLY_COLS = {'page'}
DEVICE_COLS = {'device'}
COUNTRY_COLS = {'country'}
DATE_COLS = {'date'}
FEATURE_COLS = {'feature'}
FILTER_COLS = {'filter', 'value'} # Simplified filter check

# Full base sets (dimension + metrics)
QUERY_COLS_BASE = QUERY_ONLY_COLS.union(METRIC_COLS)
PAGE_COLS_BASE = PAGE_ONLY_COLS.union(METRIC_COLS)
DEVICE_COLS_BASE = DEVICE_COLS.union(METRIC_COLS)
COUNTRY_COLS_BASE = COUNTRY_COLS.union(METRIC_COLS)
DATE_COLS_BASE = DATE_COLS.union(METRIC_COLS)
# More flexible SERP feature check (feature + position + metrics)
SERP_COLS_BASE = FEATURE_COLS.union(METRIC_COLS) 

# Define Spanish column equivalents
SPANISH_COLS = {
    'consultas principales': 'query',
    'páginas': 'page',
    'páginas principales': 'page', # Added this
    'clics': 'clicks',
    'impresiones': 'impressions',
    'ctr': 'ctr',
    'posición': 'position',
    'país': 'country',
    'dispositivo': 'device',
    'fecha': 'date',
    'aparición en búsquedas': 'feature',
    'filtrar': 'filter',  # Added this
    'comparación': 'comparison', # Keep in case it appears
    'valor': 'value', # Added this
    'url': 'page' # Needed for SERP feature matching if present
}

def get_standardized_columns(df: pd.DataFrame) -> set:
    """Get a set of standardized column names, handling Spanish variants."""
    cols = set(df.columns.str.lower().str.strip())
    standard_cols = set()
    for col in cols:
        standard_cols.add(SPANISH_COLS.get(col, col)) # Map Spanish to standard, keep others as is
    return standard_cols

def identify_file_type(df: pd.DataFrame) -> str:
    """Identify the type of CSV file based on its columns (handles English & Spanish)."""
    standard_cols = get_standardized_columns(df)
    
    # --- Prioritize Checks ---
    # Check based on presence of key dimension columns + metrics
    # Note: Order matters if a file could match multiple types (e.g., query+page+device)

    # 1. Query & Page present (potentially with other dimensions)
    if QUERY_PAGE_COLS.issubset(standard_cols) and METRIC_COLS.issubset(standard_cols):
        # This is the most complete GSC performance data
        if DEVICE_COLS.issubset(standard_cols):
            return 'devices' # Contains Query, Page, Metrics, Device
        elif COUNTRY_COLS.issubset(standard_cols):
            return 'countries' # Contains Query, Page, Metrics, Country
        elif DATE_COLS.issubset(standard_cols):
            return 'dates' # Contains Query, Page, Metrics, Date
        else:
            return 'queries' # Contains Query, Page, Metrics

    # 2. Dimension-specific types (dimension + metrics, NO query/page assumed)
    # Check these before simpler query/page only checks
    elif DEVICE_COLS_BASE.issubset(standard_cols):
        return 'devices' # Only device + metrics
    elif COUNTRY_COLS_BASE.issubset(standard_cols):
        return 'countries' # Only country + metrics
    elif DATE_COLS_BASE.issubset(standard_cols):
        return 'dates' # Only date + metrics
    elif SERP_COLS_BASE.issubset(standard_cols):
        return 'serp_features' # Only feature + metrics + position
        
    # 3. Query only + Metrics
    elif QUERY_COLS_BASE.issubset(standard_cols):
         return 'queries' # Query + metrics (like Consultas.csv example)
         
    # 4. Page only + Metrics
    elif PAGE_COLS_BASE.issubset(standard_cols):
        return 'pages' # Page + metrics (like Páginas.csv example)
        
    # 5. Filter file check (Simplified)
    elif FILTER_COLS.issubset(standard_cols):
        return 'filters'
         
    # 6. Other specific, non-GSC performance types (add more as needed)
    elif {'url', 'title', 'meta_description'}.issubset(standard_cols):
        return 'meta_data'
    elif {'url', 'mobile_usability', 'mobile_friendly_score'}.issubset(standard_cols):
        return 'mobile'
    elif {'url', 'backlinks', 'domain_authority'}.issubset(standard_cols):
        return 'backlinks'
        
    # 7. Fallback Unknown
    else:
        logging.warning(f"Unknown file type. Standardized columns found: {standard_cols}")
        return 'unknown'

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to English and consistent format (handles English & Spanish)."""
    df_copy = df.copy()
    df_copy.columns = [col.lower().strip() for col in df_copy.columns]
    
    # Combine original mapping with Spanish mappings
    # Prioritize specific mappings defined earlier
    column_map = {
        # Query data
        'search query': 'query',
        'keyword': 'query',
        'search term': 'query',
        'term': 'query',
        'consultas principales': 'query', # Spanish
        
        # Page data
        'landing page': 'page',
        'url': 'page',
        'web page': 'page',
        'páginas': 'page', # Spanish
        
        # Metrics
        'click': 'clicks',
        'clics': 'clicks', # Spanish
        'impression': 'impressions',
        'view': 'impressions',
        'impresiones': 'impressions', # Spanish
        'average position': 'position',
        'avg. position': 'position',
        'avg position': 'position',
        'rank': 'position',
        'posición': 'position', # Spanish
        # 'ctr' is often already 'ctr', but handle potential case differences via lowercasing
        
        # Geography
        'country/territory': 'country',
        'region': 'country',
        'location': 'country',
        'país': 'country', # Spanish
        
        # Device
        'device type': 'device',
        'device category': 'device',
        'dispositivo': 'device', # Spanish
        
        # Date
        'date range': 'date',
        'day': 'date',
        'fecha': 'date', # Spanish
        
        # Mobile (keep existing)
        'mobile friendly': 'mobile_friendly',
        'mobile score': 'mobile_friendly_score',
        'usability': 'mobile_usability',
        
        # Backlinks (keep existing)
        'external links': 'backlinks',
        'referring domains': 'referring_domains',
        'authority': 'domain_authority',
        'da': 'domain_authority',
        
        # SERP features
        'serp feature': 'feature',
        'feature type': 'feature',
        'aparición en búsquedas': 'feature' # Spanish
    }

    # Apply renaming based on the combined map
    df_copy.rename(columns=column_map, inplace=True)
    
    # Optional: Log columns that were not renamed (for debugging)
    # original_cols = set(df.columns.str.lower().str.strip())
    # renamed_cols = set(df_copy.columns)
    # unmapped_cols = original_cols - set(column_map.keys())
    # if unmapped_cols:
    #     logging.debug(f"Columns not explicitly mapped: {unmapped_cols}")
        
    return df_copy

def process_csv_files(files: List[Dict[str, Union[str, bytes, IO[bytes], IO[str]]]]) -> Dict[str, pd.DataFrame]:
    """Process multiple CSV files and organize them by type. Accepts bytes, str, or file-like objects."""
    processed_data = {}
    
    for file_info in files:
        filename = file_info.get('filename', 'unknown_file') # Get filename for logging
        content = file_info.get('content')
        
        if not content:
            logging.warning(f"Skipping file {filename} due to missing content.")
            continue
            
        try:
            # Read CSV directly from content (str, bytes, or file-like)
            # pandas can handle BytesIO, StringIO, or file paths/objects
            df = pd.read_csv(content)
            
            # Standardize column names first (handles Spanish/English variations)
            df_standardized = standardize_column_names(df)

            # Determine file type using standardized columns
            file_type = identify_file_type(df_standardized) # Pass the standardized df here
            
            if file_type != 'unknown':
                # Store the standardized dataframe
                if file_type in processed_data:
                    processed_data[file_type] = pd.concat([processed_data[file_type], df_standardized], ignore_index=True)
                else:
                    processed_data[file_type] = df_standardized
            else:
                # Log using the original filename if available
                logging.warning(f"Unknown file format or insufficient columns for file: {filename}")
                
        except pd.errors.EmptyDataError:
             logging.warning(f"Skipping empty file: {filename}")
        except Exception as e:
            # Log using the original filename if available
            logging.error(f"Error processing file {filename}: {str(e)}")
            # Optionally, re-raise or handle specific exceptions differently
            # Consider potential decoding errors if bytes are passed without proper encoding
            # raise e 
    
    return processed_data

# Example Usage (Optional: for testing)
if __name__ == '__main__':
    # Create dummy dataframes mimicking Spanish exports
    data_consultas = {'Consultas principales': ['keyword 1', 'keyword 2'], 'Clics': [10, 20], 'Impresiones': [100, 200], 'CTR': [0.1, 0.1], 'Posición': [1.5, 2.5]}
    df_consultas = pd.DataFrame(data_consultas)

    data_paginas = {'Páginas': ['/page1', '/page2'], 'Clics': [5, 15], 'Impresiones': [50, 150], 'CTR': [0.1, 0.1], 'Posición': [3.0, 4.0]}
    df_paginas = pd.DataFrame(data_paginas)
    
    data_paises = {'País': ['Spain', 'Mexico'], 'Clics': [8, 12], 'Impresiones': [80, 120], 'CTR': [0.1, 0.1], 'Posición': [1.0, 1.2], 'Consultas principales': ['q1','q2'], 'Páginas':['p1','p2']}
    df_paises = pd.DataFrame(data_paises) # Needs query and page too
    
    data_fechas = {'Fecha': ['2023-01-01', '2023-01-02'], 'Clics': [100, 110], 'Impresiones': [1000, 1100], 'CTR': [0.1, 0.1], 'Posición': [2.0, 2.1], 'Consultas principales': ['q1','q2'], 'Páginas':['p1','p2']}
    df_fechas = pd.DataFrame(data_fechas) # Needs query and page too
    
    data_dispositivos = {'Dispositivo': ['Desktop', 'Mobile'], 'Clics': [7, 13], 'Impresiones': [70, 130], 'CTR': [0.1, 0.1], 'Posición': [1.8, 2.2], 'Consultas principales': ['q1','q2'], 'Páginas':['p1','p2']}
    df_dispositivos = pd.DataFrame(data_dispositivos) # Needs query and page too

    data_serp = {'Aparición en búsquedas': ['Featured Snippet', 'Image Pack'], 'Position': [1, 3], 'URL': ['/page1', '/page-img'], 'Query': ['query serp', 'image query']}
    df_serp = pd.DataFrame(data_serp)

    data_filtros = {'Filter': ['country'], 'Comparison': ['equals'], 'Value': ['ES']}
    df_filtros = pd.DataFrame(data_filtros)

    # Test identification
    print(f"Consultas type: {identify_file_type(df_consultas)}")
    print(f"Páginas type: {identify_file_type(df_paginas)}")
    print(f"Países type: {identify_file_type(df_paises)}")
    print(f"Fechas type: {identify_file_type(df_fechas)}")
    print(f"Dispositivos type: {identify_file_type(df_dispositivos)}")
    print(f"SERP type: {identify_file_type(df_serp)}")
    print(f"Filtros type: {identify_file_type(df_filtros)}")
    print("-" * 20)

    # Test standardization
    std_consultas = standardize_column_names(df_consultas)
    print(f"Standardized Consultas columns: {list(std_consultas.columns)}")

    std_paginas = standardize_column_names(df_paginas)
    print(f"Standardized Páginas columns: {list(std_paginas.columns)}")

    std_paises = standardize_column_names(df_paises)
    print(f"Standardized Países columns: {list(std_paises.columns)}")
    
    std_fechas = standardize_column_names(df_fechas)
    print(f"Standardized Fechas columns: {list(std_fechas.columns)}")
    
    std_dispositivos = standardize_column_names(df_dispositivos)
    print(f"Standardized Dispositivos columns: {list(std_dispositivos.columns)}")

    std_serp = standardize_column_names(df_serp)
    print(f"Standardized SERP columns: {list(std_serp.columns)}")

    std_filtros = standardize_column_names(df_filtros)
    print(f"Standardized Filtros columns: {list(std_filtros.columns)}")
    print("-" * 20)
    
    # Test processing function (requires file-like objects)
    from io import StringIO, BytesIO
    
    files_to_process = [
        {'filename': 'consultas.csv', 'content': df_consultas.to_csv(index=False)},
        {'filename': 'paginas.csv', 'content': df_paginas.to_csv(index=False).encode('utf-8')}, # Test bytes
        {'filename': 'paises.csv', 'content': df_paises.to_csv(index=False)},
        {'filename': 'fechas.csv', 'content': df_fechas.to_csv(index=False)},
        {'filename': 'dispositivos.csv', 'content': df_dispositivos.to_csv(index=False)},
        {'filename': 'serp.csv', 'content': df_serp.to_csv(index=False)},
        {'filename': 'filtros.csv', 'content': df_filtros.to_csv(index=False)},
        {'filename': 'empty.csv', 'content': ''}, # Test empty file
        {'filename': 'malformed.csv', 'content': 'col1,col2\nval1'}, # Test malformed
        {'filename': 'other.csv', 'content': 'ColA,ColB\n1,2'}, # Test unknown
        {'filename': 'no_content.csv', 'content': None} # Test missing content
    ]
    
    processed_data = process_csv_files(files_to_process)
    
    print("Processed Data Summary:")
    for file_type, df in processed_data.items():
        print(f"  Type: {file_type}, Shape: {df.shape}, Columns: {list(df.columns)}")
    
