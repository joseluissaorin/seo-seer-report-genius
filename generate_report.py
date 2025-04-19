import argparse
import requests
import os
import sys
from pathlib import Path

def generate_report(csv_file_path: Path, api_key: str, api_endpoint: str, output_dir: Path):
    """Sends a CSV file to the SEO analysis API and saves the resulting PDF report."""
    try:
        with open(csv_file_path, 'rb') as f:
            files = {'file': (csv_file_path.name, f, 'text/csv')}
            data = {'api_key': api_key}
            
            print(f"Sending {csv_file_path.name} to {api_endpoint}...")
            response = requests.post(api_endpoint, files=files, data=data, timeout=300) # 5 min timeout

            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Save the PDF report
            output_filename = output_dir / f"{csv_file_path.stem}_report.pdf"
            with open(output_filename, 'wb') as out_f:
                out_f.write(response.content)
            print(f"Successfully generated report: {output_filename}")
            return True

    except requests.exceptions.RequestException as e:
        print(f"Error sending request for {csv_file_path.name}: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"API Error Detail: {error_detail.get('detail', e.response.text)}", file=sys.stderr)
            except requests.exceptions.JSONDecodeError:
                print(f"API Error Response (non-JSON): {e.response.text}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred while processing {csv_file_path.name}: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate SEO reports by sending CSV files to the SEO Seer API.")
    parser.add_argument("csv_folder", help="Path to the folder containing CSV files.")
    parser.add_argument("api_key", help="Your Google Gemini API Key.")
    parser.add_argument("-e", "--endpoint", default="http://127.0.0.1:4568/analyze-seo", 
                        help="The URL endpoint of the running SEO analysis API.")
    parser.add_argument("-o", "--output", default="reports", 
                        help="Directory to save the generated PDF reports.")

    args = parser.parse_args()

    csv_folder_path = Path(args.csv_folder)
    output_dir_path = Path(args.output)
    api_key = args.api_key
    api_endpoint = args.endpoint

    if not csv_folder_path.is_dir():
        print(f"Error: Folder not found at {csv_folder_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing CSV files from: {csv_folder_path}")
    print(f"Saving reports to: {output_dir_path}")
    print(f"Using API endpoint: {api_endpoint}")

    processed_count = 0
    error_count = 0

    for item in csv_folder_path.iterdir():
        if item.is_file() and item.suffix.lower() == '.csv':
            if generate_report(item, api_key, api_endpoint, output_dir_path):
                processed_count += 1
            else:
                error_count += 1

    print("\nProcessing complete.")
    print(f"Successfully generated reports: {processed_count}")
    print(f"Failed reports: {error_count}")

if __name__ == "__main__":
    main() 