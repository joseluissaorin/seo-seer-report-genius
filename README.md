
# SEO Seer: Advanced SEO Analysis Tool

SEO Seer is a powerful web application that transforms Google Search Console exports into comprehensive SEO reports with actionable insights. The application uses advanced analytics combined with AI to provide meaningful recommendations for improving your website's search engine performance.

## Features

- **CSV Upload**: Easily upload your Google Search Console export data
- **AI-Powered Analysis**: Leverages Gemini AI for intelligent insights
- **Detailed PDF Reports**: Receive comprehensive analysis reports
- **Actionable Recommendations**: Get clear steps to improve your SEO
- **Keyword Opportunities**: Discover new keywords to target
- **Content Suggestions**: Get ideas for new content based on your data

## Project Structure

- **Frontend**: React application with modern UI
- **Backend**: FastAPI Python server for data processing and AI analysis

## Setup Instructions

### Frontend Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. The React application will be available at http://localhost:8080

### Backend Setup

1. Navigate to the API directory:
   ```bash
   cd api
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

5. The backend API will be available at http://localhost:8000

## API Endpoints

- `POST /analyze-seo`: Upload GSC data and generate SEO report
  - Parameters:
    - `file`: CSV file from Google Search Console
    - `api_key`: Gemini API key for AI analysis

## How It Works

1. **Data Upload**: Users upload their Google Search Console export data
2. **Data Processing**: The system analyzes the CSV data to extract key metrics and patterns
3. **AI Analysis**: Gemini AI is used to generate insights and recommendations
4. **Report Generation**: A comprehensive PDF report is created with visualizations and actionable steps
5. **Download**: Users can download and view their complete SEO analysis

## Getting a Gemini API Key

To use SEO Seer, you'll need a Gemini API key:

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and paste it into the application when prompted

## Example Google Search Console Export

For testing, you can export data from Google Search Console:

1. Go to Google Search Console
2. Select your property
3. Navigate to Performance
4. Set your desired date range
5. Click "Export" and choose "CSV"

## Security Notes

- API keys are only used for processing and are not stored on our servers
- Data uploads are processed and then removed to ensure privacy

## Technologies Used

- **Frontend**: React, TypeScript, TailwindCSS
- **Backend**: FastAPI, Python
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **AI Integration**: Google Generative AI (Gemini)
- **Visualization**: Matplotlib
- **Report Generation**: ReportLab
