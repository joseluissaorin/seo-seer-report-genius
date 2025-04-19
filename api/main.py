from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
from typing import List, Dict, Optional, Any
import io
import json
import tempfile
import os
import asyncio
import logging
import uuid
import google.generativeai as genai
from dotenv import load_dotenv

# --- Add basic logging configuration ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Keep old config commented

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create file handler
log_file_path = os.path.join(os.path.dirname(__file__), 'api.log') # Place log file in api directory
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Logging configured. Log file: {log_file_path}") # Log confirmation
# --- End logging configuration ---

from utils.file_processor import process_csv_files, SupportedLocale
from utils.analyzer import (
    analyze_query_patterns, analyze_device_data, analyze_temporal_patterns,
    analyze_geographic_data, analyze_mobile_friendliness, analyze_backlinks,
    detect_keyword_cannibalization, generate_serp_previews
)
from utils.keyword_research import KeywordResearchService
from utils.competitor_analysis import SimplifiedCompetitorAnalysisService
from utils.content_gap import SimplifiedContentGapService
from utils.serp_feature_analyzer import SimplifiedSERPAnalyzerService
from utils.seo_health import SEOHealthAnalyzer
from utils.visualization import (
    create_query_visualizations, create_device_visualizations, create_temporal_visualizations,
    create_geographic_visualizations, create_mobile_visualizations, create_backlink_visualizations,
    create_cannibalization_visualizations, create_seo_health_visualizations, create_serp_preview_visualizations,
    create_keyword_research_visualizations, create_old_competitor_viz, create_old_gap_viz, create_old_serp_viz
)
from utils.ai_analyzer import (
    analyze_keywords_with_ai, analyze_competitors_with_ai,
    analyze_content_gaps_with_ai, analyze_serp_features_with_ai
)
from utils.ai_visualization import (
    create_ai_keyword_visualizations, create_ai_competitor_visualizations,
    create_ai_content_gap_visualizations, create_ai_serp_visualizations
)

# Import report generation components (assuming they are within main.py or separate module)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# --- Model Selection ---
from enum import Enum

class ModelType(str, Enum):
    GEMINI = "gemini" # Use Gemini for analysis
    # Add other models if needed in the future
    # CHATGPT = "chatgpt"


# --- Environment Variables & API Key Handling ---
load_dotenv()
# Define required environment variables
REQUIRED_ENV_VARS = ["GOOGLE_API_KEY"]

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-seo")
async def analyze_seo(
    files: List[UploadFile] = File(...), 
    api_key: str = Form(...), 
    ai_model_selection: ModelType = Form(ModelType.GEMINI) # Default to Gemini
):
    logger.info(f"Received /analyze-seo request with {len(files)} file(s). Using AI Model: {ai_model_selection}")
    processed_files_list = []
    filenames = [] # Keep track of original filenames
    temp_image_paths = [] # List to store paths of generated temp images for cleanup
    
    if not files:
        logger.error("No files uploaded.")
        raise HTTPException(status_code=400, detail="No files uploaded.")

    for file in files:
        try:
            content = await file.read()
            # Attempt to decode as UTF-8, replace errors to avoid crashes
            # Using BytesIO allows pandas to handle encoding detection potentially better
            processed_files_list.append({
                'filename': file.filename,
                'content': io.BytesIO(content) # Pass BytesIO directly
            })
            filenames.append(file.filename)
            logger.info(f"Successfully read file: {file.filename}")
        except Exception as e:
            logger.error(f"Error reading file {file.filename}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading file {file.filename}: {str(e)}")
        finally:
            await file.close() # Ensure file handles are closed

    try:
        logger.info(f"Processing {len(processed_files_list)} files...")
        data_frames = process_csv_files(processed_files_list)
        logger.info(f"Processed data types: {list(data_frames.keys())}")
        
        # --- Log dtypes after processing ---
        for name, df in data_frames.items():
            logger.debug(f"DataFrame '{name}' dtypes after processing:\n{df.dtypes}")
            # --- Attempt basic numeric conversion --- 
            metric_cols_to_convert = ['clicks', 'impressions', 'ctr', 'position']
            for col in metric_cols_to_convert:
                if col in df.columns:
                    original_dtype = str(df[col].dtype)
                    if not pd.api.types.is_numeric_dtype(df[col]):
                         # Check for percentage sign in CTR specifically
                         if col == 'ctr' and df[col].astype(str).str.contains('%').any():
                              logger.info(f"Attempting conversion for percentage CTR column in '{name}'")
                              df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce') / 100.0
                         # Check for commas as thousand separators
                         elif df[col].astype(str).str.contains(',').any():
                             logger.info(f"Attempting conversion for comma-formatted column '{col}' in '{name}'")
                             df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
                         else:
                             df[col] = pd.to_numeric(df[col], errors='coerce')
                             
                         new_dtype = str(df[col].dtype)
                         if original_dtype != new_dtype:
                              logger.info(f"Converted column '{col}' in DataFrame '{name}' from {original_dtype} to {new_dtype}. Errors coerced to NaN.")
                         else:
                              logger.warning(f"Attempted conversion on non-numeric column '{col}' in DataFrame '{name}', but dtype remained {original_dtype}. Check data for non-standard formats.")
            # Log dtypes again after conversion attempt
            logger.debug(f"DataFrame '{name}' dtypes after conversion attempt:\n{df.dtypes}")
        # --- End basic numeric conversion ---
            
        if not data_frames:
             logger.warning("No processable data found in the uploaded files.")
             raise HTTPException(status_code=400, detail="No processable data found in the uploaded CSV files.")
        
        # Initialize services
        keyword_research_service = KeywordResearchService()
        competitor_analysis_service = SimplifiedCompetitorAnalysisService()
        content_gap_service = SimplifiedContentGapService()
        serp_analyzer_service = SimplifiedSERPAnalyzerService()
        seo_health_analyzer = SEOHealthAnalyzer()
        
        # Perform analyses
        analysis_results = {}
        visualization_figures = {} # Store dictionaries of Plotly figures
        ai_visualization_figures = {} # Store AI-specific visualization figures
        
        # --- Conditional Analysis with Logging --- 
        if 'queries' in data_frames or 'pages' in data_frames:
            queries_df = data_frames.get('queries') 
            pages_df = data_frames.get('pages')
            
            if queries_df is not None:
                logger.info("--- Starting Query Analysis --- Dtypes:")
                logger.info(f"\n{queries_df.dtypes}")
                analysis_results['query_analysis'] = analyze_query_patterns(queries_df)
                figs = create_query_visualizations(analysis_results['query_analysis'])
                if figs: visualization_figures['query_visualizations'] = figs
            
            top_queries = queries_df.sort_values('impressions', ascending=False)['query'].head(10).tolist()
            
            logger.info("--- Starting Keyword Research --- ")
            try:
                keyword_results = await keyword_research_service.research_keywords(
                    seed_keyword=top_queries[0] if top_queries else None,
                    api_key=api_key,
                    model=ai_model_selection,
                    fetch_trends=True
                )
                analysis_results['ai_keyword_analysis'] = keyword_results
                figs = create_keyword_research_visualizations(keyword_results)
                if figs: visualization_figures['ai_keyword_visualizations'] = figs
            except Exception as e:
                logger.error(f"AI Keyword Research error: {str(e)}", exc_info=True)
                analysis_results['ai_keyword_analysis'] = {"error": str(e)}
            
            logger.info("--- Starting Competitor Analysis --- ")
            try:
                competitor_results = await competitor_analysis_service.analyze_competitors(
                    seed_keyword=top_queries[0] if top_queries else None,
                    api_key=api_key,
                    model=ai_model_selection
                )
                analysis_results['ai_competitor_analysis'] = competitor_results
                figs = create_old_competitor_viz(competitor_results)
                if figs: visualization_figures['ai_competitor_visualizations'] = figs
            except Exception as e:
                logger.error(f"AI Competitor Analysis error: {str(e)}", exc_info=True)
                analysis_results['ai_competitor_analysis'] = {"error": str(e)}

            logger.info("--- Starting Keyword Cannibalization Detection --- Dtypes:")
            # Log dtypes of all dataframes as this function takes the dict
            for name, df_log in data_frames.items():
                logger.info(f"Dataframe '{name}':\n{df_log.dtypes}")
            try:
                cannibalization_results = detect_keyword_cannibalization(data_frames)
                analysis_results['keyword_cannibalization'] = cannibalization_results
                figs = create_cannibalization_visualizations(cannibalization_results)
                if figs: visualization_figures['cannibalization_visualizations'] = figs
            except Exception as e:
                logger.error(f"Cannibalization analysis error: {str(e)}")
            
            if 'ai_content_gaps' in analysis_results:
                 logger.info("--- Starting AI Content Gap Analysis --- Dtypes:")
                 for name, df_log in data_frames.items(): logger.info(f"Dataframe '{name}':\n{df_log.dtypes}")
                 try:
                     gap_results = await content_gap_service.analyze_content_gaps(
                         user_keyword_clusters=analysis_results['ai_keyword_analysis'].get('ai_analysis', {}).get('clusters', []),
                         niche=top_queries[0] if top_queries else None,
                         api_key=api_key,
                         model=ai_model_selection
                     )
                     analysis_results['ai_content_gaps'] = gap_results
                     figs = create_old_gap_viz(gap_results)
                     if figs: visualization_figures['ai_gap_visualizations'] = figs
                 except Exception as e:
                    logger.error(f"AI Content Gap Analysis error: {str(e)}", exc_info=True)
                    analysis_results['ai_content_gaps'] = {"error": str(e)}

        if 'serp_features' in data_frames:
            serp_features_df = data_frames['serp_features']
            logger.info("--- Starting AI SERP Feature Analysis --- Dtypes:")
            logger.info(f"\n{serp_features_df.dtypes}")
            try:
                serp_results = await serp_analyzer_service.analyze_serp_features(
                    seed_keyword=top_queries[0] if top_queries else None,
                    target_keywords=top_queries[:10],
                    api_key=api_key,
                    model=ai_model_selection
                )
                analysis_results['ai_serp_analysis'] = serp_results
                figs = create_old_serp_viz(serp_results)
                if figs: visualization_figures['ai_serp_visualizations'] = figs
            except Exception as e:
                logger.error(f"AI SERP Feature Analysis error: {str(e)}", exc_info=True)
                analysis_results['ai_serp_analysis'] = {"error": str(e)}

        page_urls_available = False
        if 'pages' in data_frames and not data_frames['pages'].empty:
             page_urls_available = True
        elif 'queries' in data_frames and 'page' in data_frames['queries'].columns and not data_frames['queries'].empty:
             page_urls_available = True
        
        if page_urls_available:
            logger.info("--- Starting SERP Preview Generation --- Dtypes:")
            for name, df_log in data_frames.items(): logger.info(f"Dataframe '{name}':\n{df_log.dtypes}")
            try:
                serp_previews = generate_serp_previews(data_frames)
                analysis_results['serp_previews'] = serp_previews
                figs = create_serp_preview_visualizations(serp_previews)
                if figs: visualization_figures['serp_preview_visualizations'] = figs
            except Exception as e:
                logger.error(f"SERP preview error: {str(e)}")

        if 'devices' in data_frames:
            devices_df = data_frames['devices']
            logger.info("--- Starting Device Analysis --- Dtypes:")
            logger.info(f"\n{devices_df.dtypes}")
            analysis_results['device_analysis'] = analyze_device_data(devices_df)
            figs = create_device_visualizations(analysis_results['device_analysis'])
            if figs: visualization_figures['device_visualizations'] = figs

        if 'mobile' in data_frames:
            mobile_df = data_frames['mobile']
            logger.info("--- Starting Mobile Optimization Analysis --- Dtypes:")
            logger.info(f"\n{mobile_df.dtypes}")
            try:
                mobile_optimization = analyze_mobile_friendliness(mobile_df) 
                analysis_results['mobile_optimization'] = mobile_optimization
                figs = create_mobile_visualizations(mobile_optimization)
                if figs: visualization_figures['mobile_visualizations'] = figs
            except Exception as e:
                logger.error(f"Mobile optimization analysis error: {str(e)}")
        elif 'devices' in data_frames:
             logger.info("Mobile data not found, attempting basic insights from device data.")
             pass 
        
        if 'dates' in data_frames:
            dates_df = data_frames['dates']
            logger.info("--- Starting Temporal Analysis --- Dtypes:")
            logger.info(f"\n{dates_df.dtypes}")
            analysis_results['temporal_analysis'] = analyze_temporal_patterns(dates_df)
            figs = create_temporal_visualizations(analysis_results['temporal_analysis'])
            if figs: visualization_figures['temporal_visualizations'] = figs
        
        if 'countries' in data_frames:
            countries_df = data_frames['countries']
            logger.info("--- Starting Geographic Analysis --- Dtypes:")
            logger.info(f"\n{countries_df.dtypes}")
            analysis_results['geo_analysis'] = analyze_geographic_data(countries_df)
            figs = create_geographic_visualizations(analysis_results['geo_analysis'])
            if figs: visualization_figures['geo_visualizations'] = figs
            
        if 'backlinks' in data_frames:
            backlinks_df = data_frames['backlinks']
            logger.info("--- Starting Backlink Analysis --- Dtypes:")
            logger.info(f"\n{backlinks_df.dtypes}")
            try:
                backlink_data = analyze_backlinks(backlinks_df) 
                analysis_results['backlink_analysis'] = backlink_data
                figs = create_backlink_visualizations(backlink_data)
                if figs: visualization_figures['backlink_visualizations'] = figs
            except Exception as e:
                logger.error(f"Backlink analysis error: {str(e)}")

        if any(key in data_frames for key in ['queries', 'pages', 'devices', 'countries', 'dates', 'mobile', 'backlinks']): # Check based on relevant keys
            logger.info("--- Starting SEO Health Score Calculation --- Dtypes:")
            for name, df_log in data_frames.items(): logger.info(f"Dataframe '{name}':\n{df_log.dtypes}")
            try:
                health_score = seo_health_analyzer.calculate_health_score(data_frames)
                analysis_results['seo_health'] = health_score
                vis_path = create_seo_health_visualizations(health_score)
                if vis_path: visualization_figures['seo_health_visualizations'] = vis_path
            except Exception as e:
                logger.error(f"SEO Health analysis error: {str(e)}")
        
        logger.info(f"Analysis complete. Results keys: {list(analysis_results.keys())}")
        logger.info(f"Visualization paths: {list(visualization_figures.keys())}")

        # Combine analysis text results and visualization paths/figures for the report generator
        report_data = {**analysis_results, **visualization_figures} # visualization_figures now contains dicts of figures

        # Generate PDF report
        logger.info("Generating PDF report...")
        # Pass the temp_image_paths list to the report generator
        pdf_path = await generate_pdf_report(report_data, api_key, ai_model_selection, data_frames, temp_image_paths)
        logger.info(f"PDF report generated at: {pdf_path}")

        # Return PDF report
        return FileResponse(pdf_path, media_type='application/pdf', filename='seo_analysis_report.pdf')

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc # Re-raise HTTP exceptions
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}") # Log full traceback
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        # Clean up temporary image files using the collected list
        logger.info(f"Cleaning up {len(temp_image_paths)} temporary image(s)...")
        for path in temp_image_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up temp image: {path}")
                except Exception as e:
                    logger.error(f"Error cleaning up temp file {path}: {str(e)}")

# --- Report Generation Function ---
async def generate_pdf_report(
    analysis_data: Dict[str, Any], # Combined analysis results (standard + AI)
    visualization_figures: Dict[str, Dict[str, go.Figure]], # Combined figures (standard + AI)
    api_key: str, # Keep for Gemini summary generation
    ai_model_client: Optional[genai.GenerativeModel], # Accept the client object
    data_frames: Dict[str, pd.DataFrame],
    temp_image_paths: List[str] # List to manage temp files
) -> str:
    """Generate a PDF report conditionally based on available analysis data."""
    logger.info(f"Generating PDF report. AI Model: {ai_model_client.model_name if ai_model_client else 'None'}. Data types: {list(data_frames.keys())}")

    # PDF Setup
    # Use BytesIO for in-memory PDF generation
    pdf_buffer = io.BytesIO()
    pdf_filename = f"seo_analysis_report_{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(tempfile.gettempdir(), pdf_filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom Styles
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Code', fontName='Courier', fontSize=9, leading=11))
    styles.add(ParagraphStyle(name='h3_custom', parent=styles['h3'], spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(name='ListBullet', parent=styles['Normal'], leftIndent=18))

    # --- Report Header ---
    story.append(Paragraph("SEO Analysis Report", styles['h1']))
    story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Data Types Included: {', '.join(sorted(data_frames.keys()))}", styles['Normal']))
    story.append(Paragraph(f"AI Model Used: {ai_model_client.model_name if ai_model_client else 'None'}", styles['Normal']))
    story.append(Spacer(1, 12))

    # --- Executive Summary (Using AI if available) ---
    logger.info("Attempting to generate executive summary...")
    summary_text = "Summary generation failed or was skipped (AI client not available)."
    if ai_model_client:
        try:
            # Build prompt for summary
            prompt_parts = [
                f"You are an expert SEO analyst using the {ai_model_client.model_name} model. Analyze the following SEO data overview and detailed analysis snippets.",
                "Provide a concise executive summary (approx. 3-5 paragraphs).",
                "Focus on the most impactful insights, key performance indicators (KPIs), significant trends (positive or negative), and actionable recommendations based *only* on the data provided.",
                "Structure the summary logically, starting with overall performance and then highlighting key areas.",
                "\n--- Available Data Overview ---"
            ]
            # Add data overview (same as before)
            for key, df in data_frames.items():
                prompt_parts.append(f"\n*   **{key.replace('_', ' ').title()} Data:** Present ({df.shape[0]} rows). Key columns: {list(df.columns)}")
                if 'date' in df.columns:
                    try:
                        min_date = pd.to_datetime(df['date']).min().strftime('%Y-%m-%d')
                        max_date = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
                        prompt_parts[-1] += f" (Range: {min_date} to {max_date})"
                    except Exception: pass

            prompt_parts.append("\n--- Detailed Analysis Snippets --- ")
            analysis_summaries_found = False
            # Prioritize AI results from analysis_data
            ai_keys_map = {
                'ai_keyword_analysis': 'AI Keyword Analysis',
                'ai_competitor_analysis': 'AI Competitor Analysis',
                'ai_content_gap_analysis': 'AI Content Gap Analysis',
                'ai_serp_analysis': 'AI SERP Trend Analysis'
            }
            for key, title in ai_keys_map.items():
                if key in analysis_data and isinstance(analysis_data[key], dict) and 'error' not in analysis_data[key]:
                    analysis_summaries_found = True
                    # Extract key insights from the AI analysis structure
                    insight_summary = "(AI insights generated)" # Default
                    data = analysis_data[key]
                    if key == 'ai_keyword_analysis' and 'themes' in data:
                        insight_summary = f"Identified themes: {len(data.get('themes', []))}. Analyzed keywords: {len(data.get('keyword_analysis', []))}. Content ideas: {len(data.get('content_ideas', []))}."
                    elif key == 'ai_competitor_analysis' and 'landscape_summary' in data:
                        insight_summary = f"Landscape: {data.get('landscape_summary', 'N/A')[:100]}... Opportunities: {len(data.get('opportunities', []))}. Threats: {len(data.get('threats', []))}."
                    elif key == 'ai_content_gap_analysis' and 'gap_summary' in data:
                         insight_summary = f"Gaps: {data.get('gap_summary', 'N/A')[:100]}... Priority Gaps: {len(data.get('priority_gaps', []))}. Recommendations: {len(data.get('content_recommendations', []))}."
                    elif key == 'ai_serp_analysis' and 'trend_summary' in data:
                         insight_summary = f"Trends: {data.get('trend_summary', 'N/A')[:100]}... Implications noted. Recommendations: {len(data.get('strategic_recommendations', []))}."
                    prompt_parts.append(f"*   **{title}:** {insight_summary}")

            # Include summaries from non-AI analysis if they exist
            for key, data in analysis_data.items():
                 if key not in ai_keys_map and isinstance(data, dict):
                     summary = None
                     if 'summary' in data and isinstance(data['summary'], str):
                         summary = data['summary']
                     elif 'score' in data: # e.g., SEO Health
                          summary = f"Score calculated: {data['score']:.2f}"
                     # Add other non-AI summary extractions if needed
                     if summary:
                         analysis_summaries_found = True
                         prompt_parts.append(f"*   **{key.replace('_', ' ').title()} Analysis Summary:** {summary[:200]}...")

            if not analysis_summaries_found:
                 prompt_parts.append("(No detailed summaries from analysis modules were found; base summary on the raw data overview.)")

            prompt_parts.append("\n--- End of Data Context ---")
            prompt_parts.append("Please generate the executive summary:")

            full_prompt = "\n".join(prompt_parts)

            # Limit prompt size
            max_prompt_length = 30000
            if len(full_prompt) > max_prompt_length:
                logger.warning(f"Prompt length ({len(full_prompt)}) exceeds limit ({max_prompt_length}), truncating.")
                full_prompt = full_prompt[:max_prompt_length]

            logger.debug(f"Sending final prompt to Gemini (length {len(full_prompt)}):")
            response = await ai_model_client.generate_content_async(full_prompt) # Use async version
            summary_text = response.text
            logger.info("Successfully generated summary using Gemini.")

        except Exception as e:
            logger.error(f"Gemini call for summary failed: {str(e)}", exc_info=True)
            summary_text = f"Error: Could not generate summary using the AI model: {str(e)}"

    story.append(Paragraph("Executive Summary", styles['h2']))
    summary_style = styles.get('Justify', styles['Normal'])
    story.append(Paragraph(summary_text.replace('\n', '<br/>'), summary_style))
    story.append(Spacer(1, 12))

    # --- Data Formatters for AI Results ---
    def format_ai_keyword_data(data: Dict[str, Any]) -> List:
        elements = []
        if not isinstance(data, dict): return [Paragraph("(Invalid AI keyword data format)", styles['Normal'])]
        
        elements.append(Paragraph("Identified Themes:", styles['h3_custom']))
        themes = data.get('themes', [])
        if themes:
            for theme in themes:
                elements.append(Paragraph(f"• {theme}", styles['ListBullet']))
        else:
            elements.append(Paragraph("No specific themes identified.", styles['Normal']))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("Keyword Analysis:", styles['h3_custom']))
        kw_analysis = data.get('keyword_analysis', [])
        if kw_analysis:
            for item in kw_analysis:
                 elements.append(Paragraph(f"<b>Query:</b> {item.get('query', 'N/A')}", styles['ListBullet']))
                 elements.append(Paragraph(f"<i>Analysis:</i> {item.get('analysis', 'N/A')}", styles['ListBullet']))
                 elements.append(Spacer(1, 3))
        else:
            elements.append(Paragraph("No specific keyword analyses provided.", styles['Normal']))
        elements.append(Spacer(1, 6))
        
        elements.append(Paragraph("Content Ideas:", styles['h3_custom']))
        ideas = data.get('content_ideas', [])
        if ideas:
            for idea in ideas:
                elements.append(Paragraph(f"• {idea}", styles['ListBullet']))
        else:
             elements.append(Paragraph("No specific content ideas provided.", styles['Normal']))
        
        return elements

    def format_ai_competitor_data(data: Dict[str, Any]) -> List:
        elements = []
        if not isinstance(data, dict): return [Paragraph("(Invalid AI competitor data format)", styles['Normal'])]

        elements.append(Paragraph("Landscape Summary:", styles['h3_custom']))
        summary = data.get('landscape_summary', 'N/A')
        elements.append(Paragraph(summary, styles['Normal']))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("Opportunities:", styles['h3_custom']))
        opportunities = data.get('opportunities', [])
        if opportunities:
            for item in opportunities:
                 elements.append(Paragraph(f"<b>Opportunity:</b> {item.get('opportunity', 'N/A')}", styles['ListBullet']))
                 elements.append(Paragraph(f"<i>Reasoning:</i> {item.get('reasoning', 'N/A')}", styles['ListBullet']))
                 elements.append(Spacer(1, 3))
        else:
             elements.append(Paragraph("No specific opportunities identified.", styles['Normal']))
        elements.append(Spacer(1, 6))
        
        elements.append(Paragraph("Threats:", styles['h3_custom']))
        threats = data.get('threats', [])
        if threats:
             for item in threats:
                 elements.append(Paragraph(f"<b>Threat:</b> {item.get('threat', 'N/A')}", styles['ListBullet']))
                 elements.append(Paragraph(f"<i>Reasoning:</i> {item.get('reasoning', 'N/A')}", styles['ListBullet']))
                 elements.append(Spacer(1, 3))
        else:
             elements.append(Paragraph("No specific threats identified.", styles['Normal']))
        
        return elements

    def format_ai_content_gap_data(data: Dict[str, Any]) -> List:
        elements = []
        if not isinstance(data, dict): return [Paragraph("(Invalid AI content gap data format)", styles['Normal'])]

        elements.append(Paragraph("Gap Summary:", styles['h3_custom']))
        summary = data.get('gap_summary', 'N/A')
        elements.append(Paragraph(summary, styles['Normal']))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("Priority Gaps:", styles['h3_custom']))
        gaps = data.get('priority_gaps', [])
        if gaps:
            for item in gaps:
                 elements.append(Paragraph(f"<b>Query:</b> {item.get('query', 'N/A')}", styles['ListBullet']))
                 elements.append(Paragraph(f"<i>Reasoning:</i> {item.get('reasoning', 'N/A')}", styles['ListBullet']))
                 elements.append(Spacer(1, 3))
        else:
             elements.append(Paragraph("No priority gaps identified.", styles['Normal']))
        elements.append(Spacer(1, 6))
        
        elements.append(Paragraph("Content Recommendations:", styles['h3_custom']))
        recommendations = data.get('content_recommendations', [])
        if recommendations:
            for item in recommendations:
                 elements.append(Paragraph(f"<b>Idea:</b> {item.get('title_idea', 'N/A')}", styles['ListBullet']))
                 kw_list = item.get('target_keywords', [])
                 kw_str = ", ".join(kw_list) if kw_list else 'N/A'
                 elements.append(Paragraph(f"<i>Target Keywords:</i> {kw_str}", styles['ListBullet']))
                 elements.append(Spacer(1, 3))
        else:
            elements.append(Paragraph("No specific content recommendations provided.", styles['Normal']))
            
        return elements

    def format_ai_serp_data(data: Dict[str, Any]) -> List:
        elements = []
        if not isinstance(data, dict): return [Paragraph("(Invalid AI SERP data format)", styles['Normal'])]

        elements.append(Paragraph("Trend Summary:", styles['h3_custom']))
        summary = data.get('trend_summary', 'N/A')
        elements.append(Paragraph(summary, styles['Normal']))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("Feature Implications:", styles['h3_custom']))
        implications = data.get('feature_implications', 'N/A')
        elements.append(Paragraph(implications, styles['Normal']))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("Strategic Recommendations:", styles['h3_custom']))
        recommendations = data.get('strategic_recommendations', [])
        if recommendations:
            for item in recommendations:
                 elements.append(Paragraph(f"<b>Recommendation:</b> {item.get('recommendation', 'N/A')}", styles['ListBullet']))
                 elements.append(Paragraph(f"<i>Reasoning:</i> {item.get('reasoning', 'N/A')}", styles['ListBullet']))
                 elements.append(Spacer(1, 3))
        else:
            elements.append(Paragraph("No specific strategic recommendations provided.", styles['Normal']))
            
        return elements

    # --- Helper function to add sections (Modified to handle figures correctly) ---
    def add_report_section(title, analysis_key, visualization_fig_key=None, data_formatter=None, is_ai_result=False):
        if analysis_key in analysis_data:
            logger.info(f"Adding section: {title}")
            story.append(Paragraph(title, styles['h2']))
            data = analysis_data[analysis_key]

            # Handle potential errors from AI analysis
            if is_ai_result and isinstance(data, dict) and 'error' in data:
                story.append(Paragraph(f"Error during analysis: {data['error']}", styles['Normal']))
                story.append(Spacer(1, 12))
                return True # Section added, but with error message

            # Add specific formatted data using the provided formatter function
            if data_formatter:
                try:
                    formatted_elements = data_formatter(data)
                    story.extend(formatted_elements)
                except Exception as e:
                    logger.error(f"Error formatting data for {title}: {str(e)}", exc_info=True)
                    story.append(Paragraph(f"Error displaying data for {title}.", styles['Normal']))
            # Fallback for dicts if no formatter
            elif isinstance(data, dict) and not data_formatter:
                 try:
                      # Basic formatting for unknown dicts
                      story.append(Paragraph(json.dumps(data, indent=2, default=str), styles['Code']))
                 except Exception as e:
                      logger.error(f"Error formatting generic dict for {title}: {e}")
                      story.append(Paragraph("(Could not display raw data)", styles['Normal']))
            elif isinstance(data, str): # Handle simple string results if any
                 story.append(Paragraph(data.replace('\n', '<br/>'), summary_style))

            # Add visualization(s) using the visualization_figures dictionary
            if visualization_fig_key and visualization_fig_key in visualization_figures:
                viz_fig_dict = visualization_figures[visualization_fig_key]
                if isinstance(viz_fig_dict, dict):
                    logger.info(f"Processing visualization dictionary '{visualization_fig_key}' for section: {title}")
                    for fig_name, fig in viz_fig_dict.items():
                        if hasattr(fig, 'write_image'):
                            try:
                                # Generate unique temp filename
                                temp_img_filename = f"{title.replace(' ', '_').lower()}_{fig_name}_{uuid.uuid4()}.png"
                                temp_img_path = os.path.join(tempfile.gettempdir(), temp_img_filename)

                                # Save the figure as an image
                                logger.info(f"Saving figure '{fig_name}' to temp path: {temp_img_path}")
                                fig.write_image(temp_img_path)

                                # Add image to PDF story
                                img = Image(temp_img_path, width=500, height=250) # Adjust size as needed
                                img.hAlign = 'CENTER'
                                story.append(img)
                                story.append(Spacer(1, 6)) # Spacer after each image

                                # Add the path to the list for cleanup
                                temp_image_paths.append(temp_img_path)

                            except Exception as e:
                                logger.error(f"Error saving or adding figure '{fig_name}' for section {title}: {str(e)}", exc_info=True)
                                story.append(Paragraph(f"(Error rendering visualization: {fig_name})", styles['Normal']))
                        else:
                             logger.warning(f"Item '{fig_name}' in visualization dict for '{title}' is not a recognized figure object (type: {type(fig)}). Skipping.")
                    story.append(Spacer(1, 6)) # Extra spacer after all images in the dict
                else:
                    logger.warning(f"Visualization data for key '{visualization_fig_key}' is not a dictionary. Skipping. Type: {type(viz_fig_dict)}")

            # Add a spacer after the section content (text/data/images)
            story.append(Spacer(1, 12))
            return True
        return False

    # --- Add Report Sections Conditionally ---
    story.append(PageBreak())
    story.append(Paragraph("Detailed Analysis", styles['h1']))

    # Add standard SEO sections (if data exists)
    add_report_section("Query Performance Analysis", 'query_analysis', 'query_visualizations')
    add_report_section("Keyword Cannibalization", 'keyword_cannibalization', 'cannibalization_visualizations')
    # ... (add calls for device, temporal, geo, mobile, backlinks, health, serp_previews as before)
    add_report_section("Device Performance", 'device_analysis', 'device_visualizations')
    add_report_section("Temporal Trends", 'temporal_analysis', 'temporal_visualizations')
    add_report_section("Geographic Performance", 'geo_analysis', 'geo_visualizations')
    add_report_section("Mobile Friendliness", 'mobile_optimization', 'mobile_visualizations')
    add_report_section("Backlink Overview", 'backlink_analysis', 'backlink_visualizations')
    add_report_section("Overall SEO Health", 'seo_health', 'seo_health_visualizations')
    add_report_section("SERP Previews", 'serp_previews', 'serp_preview_visualizations')

    # Add AI Analysis Sections (if data exists)
    story.append(PageBreak())
    story.append(Paragraph("AI-Powered Insights", styles['h1']))

    add_report_section(
        "AI Keyword Analysis", 
        'ai_keyword_analysis', 
        visualization_fig_key='ai_keyword_visualizations',
        data_formatter=format_ai_keyword_data,
        is_ai_result=True
    )
    add_report_section(
        "AI Competitor Landscape", 
        'ai_competitor_analysis', 
        visualization_fig_key='ai_competitor_visualizations', 
        data_formatter=format_ai_competitor_data,
        is_ai_result=True
    )
    add_report_section(
        "AI Content Gap Opportunities", 
        'ai_content_gap_analysis', 
        visualization_fig_key='ai_content_gap_visualizations', 
        data_formatter=format_ai_content_gap_data,
        is_ai_result=True
    )
    add_report_section(
        "AI SERP Trend Analysis", 
        'ai_serp_analysis', 
        visualization_fig_key='ai_serp_visualizations',
        data_formatter=format_ai_serp_data,
        is_ai_result=True
    )

    # Build the PDF
    try:
        logger.info(f"Building PDF document at path: {pdf_path}")
        doc.build(story)
        logger.info("PDF built successfully.")
    except Exception as e:
        logger.error(f"Error building PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build PDF report: {e}")

    return pdf_path

# --- Simple Table Helper ---
def create_simple_table(df: pd.DataFrame, title: str = None):
    """Creates a basic ReportLab table from a DataFrame."""
    elements = []
    if title:
        elements.append(Paragraph(title, styles['h3']))
    if not df.empty:
        # Limit columns and rows for display
        df_display = df.head(15).copy() # Limit rows
        # Truncate long cell values
        for col in df_display.select_dtypes(include='object').columns:
             df_display[col] = df_display[col].str.slice(0, 100) # Limit string length
             
        # Convert all data to string for table generation to avoid type issues
        table_data = [df_display.columns.to_list()] + df_display.astype(str).values.tolist()
        table = Table(table_data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 6))
    else:
        elements.append(Paragraph("(No data available for this table)", styles['Normal']))
    return elements

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "SEO Seer Report Genius API is running."}

# Optional: Add cleanup for older temp PDF files on startup if needed
