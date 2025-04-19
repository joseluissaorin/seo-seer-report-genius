import requests
import argparse
import os
import glob
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_report(folder_path: str, api_key: str, output_file: str, api_url: str):
    """
    Finds CSV files in a folder, sends them to the SEO analysis API,
    and saves the resulting PDF report.
    """
    
    # 1. Validate folder path
    if not os.path.isdir(folder_path):
        logging.error(f"Error: Folder not found at '{folder_path}'")
        return

    # 2. Find CSV files
    csv_pattern = os.path.join(folder_path, '*.csv')
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        logging.error(f"Error: No CSV files found in '{folder_path}'")
        return
        
    logging.info(f"Found {len(csv_files)} CSV files to process: {', '.join([os.path.basename(f) for f in csv_files])}")

    # 3. Prepare files payload for the request
    files_payload = []
    try:
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            # Read file as bytes, let the API handle decoding
            files_payload.append(('files', (filename, open(file_path, 'rb'), 'text/csv'))) 
            logging.debug(f"Prepared file for upload: {filename}")
    except Exception as e:
        logging.error(f"Error opening file {file_path}: {str(e)}")
        # Ensure already opened files are closed in case of error
        for _, file_tuple in files_payload:
            file_tuple[1].close()
        return

    # 4. Prepare data payload
    data_payload = {'api_key': api_key}
    logging.info(f"Sending request to API endpoint: {api_url}")

    # 5. Make the POST request
    try:
        response = requests.post(api_url, files=files_payload, data=data_payload, timeout=300) # Increased timeout for potentially long analysis

        # Ensure files are closed after the request is made
        for _, file_tuple in files_payload:
             file_tuple[1].close()
             
        logging.info(f"Received response with status code: {response.status_code}")

        # 6. Handle the response
        if response.status_code == 200 and response.headers.get('content-type') == 'application/pdf':
            try:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Successfully saved report to '{output_file}'")
            except Exception as e:
                logging.error(f"Error saving PDF file '{output_file}': {str(e)}")
        else:
            logging.error(f"API request failed. Status Code: {response.status_code}")
            try:
                # Try to decode error message from API if possible
                error_detail = response.json().get('detail', response.text) 
                logging.error(f"Error Detail: {error_detail}")
            except Exception: # If response is not JSON or can't be decoded
                logging.error(f"Error Detail: {response.text[:500]}...") # Log first 500 chars

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {str(e)}")
        # Ensure files are closed if the request itself fails
        for _, file_tuple in files_payload:
             if not file_tuple[1].closed:
                 file_tuple[1].close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SEO Analysis Report by sending CSV files to the API.")
    
    parser.add_argument("-f", "--folder", 
                        required=True, 
                        help="Path to the folder containing the CSV files.")
    parser.add_argument("-k", "--api-key", 
                        required=True, 
                        help="Your Google Gemini API Key.")
    parser.add_argument("-o", "--output-file", 
                        default="seo_analysis_report_client.pdf", 
                        help="Name for the output PDF file (default: seo_analysis_report_client.pdf).")
    parser.add_argument("-u", "--api-url", 
                        default="http://localhost:4568/analyze-seo", 
                        help="URL of the running SEO analysis API endpoint (default: http://localhost:4568/analyze-seo).")

    args = parser.parse_args()

    generate_report(args.folder, args.api_key, args.output_file, args.api_url) 