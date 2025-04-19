
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
