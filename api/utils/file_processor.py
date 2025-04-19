
import pandas as pd
from typing import Dict, List, Union
import logging

def identify_file_type(df: pd.DataFrame) -> str:
    """Identifies the type of GSC export file based on its columns."""
    columns = set(df.columns.str.lower())
    
    # Spanish to English column name mapping
    spanish_english_mapping = {
        'consulta': 'query',
        'consultas': 'query',
        'página': 'page',
        'pagina': 'page',
        'páginas': 'pages',
        'paginas': 'pages',
        'dispositivo': 'device',
        'dispositivos': 'devices',
        'país': 'country',
        'pais': 'country',
        'países': 'countries',
        'paises': 'countries',
        'fecha': 'date',
        'fechas': 'dates',
        'tipo de aparición en búsquedas': 'search_appearance_type',
        'tipo de aparicion en busquedas': 'search_appearance_type',
        'aparición en búsquedas': 'search_appearance',
        'aparicion en busquedas': 'search_appearance',
        'tipo de filtro': 'filter_type',
        'filtros': 'filters',
        'clics': 'clicks',
        'impresiones': 'impressions',
        'ctr': 'ctr',
        'posición': 'position',
        'posicion': 'position'
    }
    
    # Standardize column names (convert Spanish to English)
    standardized_columns = set()
    for col in columns:
        if col in spanish_english_mapping:
            standardized_columns.add(spanish_english_mapping[col])
        else:
            standardized_columns.add(col)
    
    # Define required columns for each file type
    file_types = {
        'queries': {'query', 'clicks', 'impressions', 'ctr', 'position'},
        'pages': {'page', 'clicks', 'impressions', 'ctr', 'position'},
        'devices': {'device', 'clicks', 'impressions', 'ctr', 'position'},
        'countries': {'country', 'clicks', 'impressions', 'ctr', 'position'},
        'dates': {'date', 'clicks', 'impressions', 'ctr', 'position'},
        'search_appearance': {'search_appearance_type', 'clicks', 'impressions', 'ctr', 'position'},
        'filters': {'filter_type', 'clicks', 'impressions', 'ctr', 'position'}
    }
    
    # Check if the standardized columns match any file type
    for file_type, required_cols in file_types.items():
        if required_cols.issubset(standardized_columns):
            return file_type
            
    return 'unknown'

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
                processed_data[file_type] = df
            else:
                logging.warning(f"Unknown file format: {file_info['filename']}")
        except Exception as e:
            logging.error(f"Error processing file {file_info['filename']}: {str(e)}")
    
    return processed_data

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names from Spanish to English."""
    rename_map = {
        'consulta': 'query',
        'consultas': 'query',
        'página': 'page',
        'pagina': 'page',
        'páginas': 'page',
        'paginas': 'page',
        'dispositivo': 'device',
        'dispositivos': 'device',
        'país': 'country',
        'pais': 'country',
        'países': 'country',
        'paises': 'country',
        'fecha': 'date',
        'fechas': 'date',
        'tipo de aparición en búsquedas': 'search_appearance_type',
        'tipo de aparicion en busquedas': 'search_appearance_type',
        'aparición en búsquedas': 'search_appearance_type',
        'aparicion en busquedas': 'search_appearance_type',
        'tipo de filtro': 'filter_type',
        'filtros': 'filter_type',
        'clics': 'clicks',
        'impresiones': 'impressions',
        'ctr': 'ctr',
        'posición': 'position',
        'posicion': 'position'
    }
    
    # Create a new DataFrame with standardized column names
    df_standardized = df.copy()
    
    # Rename columns (case-insensitive)
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in rename_map:
            df_standardized = df_standardized.rename(columns={col: rename_map[col_lower]})
    
    return df_standardized
