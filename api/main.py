from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
from typing import List, Dict
import io
import json
import tempfile
import os
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import logging

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

from utils.file_processor import process_csv_files
from utils.query_analysis import QueryAnalyzer
from utils.device_analysis import analyze_device_data
from utils.temporal_analysis import analyze_temporal_patterns
from utils.geo_analysis import analyze_geographic_data
from utils.keyword_research import KeywordResearcher
from utils.competitor_analysis import CompetitorAnalyzer
from utils.visualization import (
    create_query_visualizations,
    create_temporal_visualizations,
    create_geographic_visualizations,
    create_device_visualizations,
    create_keyword_research_visualizations,
    create_competitor_visualizations
)
from utils.enhanced_visualization import (
    create_seo_health_visualizations,
    create_mobile_visualizations,
    create_cannibalization_visualizations,
    create_serp_feature_visualizations,
    create_backlink_visualizations,
    create_content_gap_visualizations,
    create_serp_preview_visualizations
)
from utils.seo_health import SEOHealthAnalyzer
from utils.mobile_analyzer import analyze_mobile_friendliness
from utils.keyword_cannibalization import detect_keyword_cannibalization
from utils.serp_feature_analyzer import analyze_serp_features
from utils.serp_preview import generate_serp_previews
from utils.backlink_analyzer import analyze_backlinks
from utils.content_gap_analyzer import analyze_content_gaps

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
async def analyze_seo(files: List[UploadFile] = File(...), api_key: str = Form(...)):
    logger.info(f"Received /analyze-seo request with {len(files)} file(s).")
    processed_files_list = []
    filenames = [] # Keep track of original filenames
    
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
        # Organize files by type
        logger.info(f"Processing {len(processed_files_list)} files...")
        data_frames = process_csv_files(processed_files_list)
        logger.info(f"Processed data types: {list(data_frames.keys())}")
        
        if not data_frames:
             logger.warning("No processable data found in the uploaded files.")
             # Depending on desired behavior, could raise HTTPException or return empty/error report
             raise HTTPException(status_code=400, detail="No processable data found in the uploaded CSV files.")

        # Initialize analyzers
        query_analyzer = QueryAnalyzer()
        keyword_researcher = KeywordResearcher()
        competitor_analyzer = CompetitorAnalyzer()
        seo_health_analyzer = SEOHealthAnalyzer()
        
        # Perform analyses
        analysis_results = {}
        visualization_paths = {} # Store paths to generated images
        
        # --- Conditional Analysis --- 
        # Note: We should review dependencies between analyses carefully.
        # Some analyses might depend on results from others (e.g., content gap needs competitor data)
        
        # Query, Keyword, Competitor, Cannibalization, SERP Features, SERP Preview, Content Gap (often related)
        if 'queries' in data_frames or 'pages' in data_frames: # Broaden trigger slightly?
            queries_df = data_frames.get('queries') # Use .get for safety
            pages_df = data_frames.get('pages')
            
            # Combine queries and pages if both exist for richer context? Requires review of analyzers.
            # combined_df = pd.concat([queries_df, pages_df], ignore_index=True) if queries_df is not None and pages_df is not None else queries_df if queries_df is not None else pages_df

            if queries_df is not None:
                logger.info("Performing Query Analysis...")
                analysis_results['query_analysis'] = query_analyzer.analyze_query_patterns(queries_df)
                vis_path = create_query_visualizations(analysis_results['query_analysis'])
                if vis_path: visualization_paths['query_visualizations'] = vis_path
                
                top_queries = queries_df.sort_values('impressions', ascending=False)['query'].head(10).tolist()
                
                logger.info("Performing Keyword Research...")
                try:
                    keyword_data = {}
                    trends = keyword_researcher.research_keyword_trends(top_queries[:5])
                    keyword_data['trend_data'] = trends
                    if top_queries:
                        suggestions = keyword_researcher.get_keyword_suggestions(top_queries[0])
                        keyword_data['related_queries'] = {top_queries[0]: {'top': suggestions}}
                    difficulties = keyword_researcher.analyze_keyword_difficulty(top_queries[:5])
                    keyword_data['difficulty'] = difficulties
                    analysis_results['keyword_research'] = keyword_data
                    vis_path = create_keyword_research_visualizations(keyword_data)
                    if vis_path: visualization_paths['keyword_visualizations'] = vis_path
                except Exception as e:
                    logger.error(f"Keyword research error: {str(e)}")

                logger.info("Performing Competitor Analysis...")
                try:
                    competitor_data = {}
                    competitors = competitor_analyzer.identify_competitors(top_queries)
                    competitor_data['competitors'] = competitors
                    if competitors:
                        top_competitor = list(competitors.keys())[0]
                        competitor_url = f"https://{top_competitor}"
                        content_analysis = competitor_analyzer.analyze_competitor_content(competitor_url)
                        competitor_data['content_analysis'] = content_analysis
                        ranking_comparison = competitor_analyzer.compare_rankings(top_queries[:5], top_competitor)
                        competitor_data['ranking_comparison'] = ranking_comparison
                    analysis_results['competitor_analysis'] = competitor_data
                    vis_path = create_competitor_visualizations(competitor_data)
                    if vis_path: visualization_paths['competitor_visualizations'] = vis_path
                except Exception as e:
                    logger.error(f"Competitor analysis error: {str(e)}")

                # Cannibalization needs careful review of its input requirements
                logger.info("Performing Keyword Cannibalization Detection...")
                try:
                    # Assuming it primarily needs query/page data
                    cannibalization_results = detect_keyword_cannibalization(data_frames)
                    analysis_results['keyword_cannibalization'] = cannibalization_results
                    vis_path = create_cannibalization_visualizations(cannibalization_results)
                    if vis_path: visualization_paths['cannibalization_visualizations'] = vis_path
                except Exception as e:
                    logger.error(f"Cannibalization analysis error: {str(e)}")
            
            # Content Gap Analysis (Depends on Competitor Analysis results)
            if 'competitor_analysis' in analysis_results:
                 logger.info("Performing Content Gap Analysis...")
                 try:
                     content_gaps = analyze_content_gaps(data_frames, analysis_results.get('competitor_analysis'))
                     analysis_results['content_gaps'] = content_gaps
                     vis_path = create_content_gap_visualizations(content_gaps)
                     if vis_path: visualization_paths['content_gap_visualizations'] = vis_path
                 except Exception as e:
                    logger.error(f"Content gap analysis error: {str(e)}")

        # SERP Features (May exist independently or with queries/pages)
        if 'serp_features' in data_frames:
            logger.info("Performing SERP Feature Analysis...")
            try:
                serp_features_df = data_frames['serp_features']
                serp_analysis = analyze_serp_features(serp_features_df) # Adjust function if it needs more context
                analysis_results['serp_features'] = serp_analysis
                vis_path = create_serp_feature_visualizations(serp_analysis)
                if vis_path: visualization_paths['serp_visualizations'] = vis_path
            except Exception as e:
                logger.error(f"SERP feature analysis error: {str(e)}")

        # SERP Preview (Needs pages/URLs, potentially from multiple types?)
        # Let's assume it primarily uses 'pages' or 'queries' df for URLs
        page_urls_available = False
        if 'pages' in data_frames and not data_frames['pages'].empty:
             page_urls_available = True
        elif 'queries' in data_frames and 'page' in data_frames['queries'].columns and not data_frames['queries'].empty:
             page_urls_available = True
        # Add other potential sources like 'serp_features' if relevant
        
        if page_urls_available:
            logger.info("Generating SERP Previews...")
            try:
                # Pass the combined dict; the function should extract needed URLs
                serp_previews = generate_serp_previews(data_frames)
                analysis_results['serp_previews'] = serp_previews
                vis_path = create_serp_preview_visualizations(serp_previews)
                if vis_path: visualization_paths['serp_preview_visualizations'] = vis_path
            except Exception as e:
                logger.error(f"SERP preview error: {str(e)}")

        # Device analysis
        if 'devices' in data_frames:
            logger.info("Performing Device Analysis...")
            analysis_results['device_analysis'] = analyze_device_data(data_frames['devices'])
            vis_path = create_device_visualizations(analysis_results['device_analysis'])
            if vis_path: visualization_paths['device_visualizations'] = vis_path

        # Mobile Optimization (Often needs 'mobile' specific data, but might relate to device traffic)
        if 'mobile' in data_frames:
            logger.info("Performing Mobile Optimization Analysis...")
            try:
                mobile_df = data_frames['mobile']
                mobile_optimization = analyze_mobile_friendliness(mobile_df) # Assuming it takes the mobile df
                analysis_results['mobile_optimization'] = mobile_optimization
                vis_path = create_mobile_visualizations(mobile_optimization)
                if vis_path: visualization_paths['mobile_visualizations'] = vis_path
            except Exception as e:
                logger.error(f"Mobile optimization analysis error: {str(e)}")
        elif 'devices' in data_frames:
             # Fallback: Maybe infer some mobile insights from device distribution?
             logger.info("Mobile data not found, attempting basic insights from device data.")
             # Add placeholder or basic analysis based on device df if desired
             pass 

        # Temporal analysis
        if 'dates' in data_frames:
            logger.info("Performing Temporal Analysis...")
            analysis_results['temporal_analysis'] = analyze_temporal_patterns(data_frames['dates'])
            vis_path = create_temporal_visualizations(analysis_results['temporal_analysis'])
            if vis_path: visualization_paths['temporal_visualizations'] = vis_path
        
        # Geographic analysis
        if 'countries' in data_frames:
            logger.info("Performing Geographic Analysis...")
            analysis_results['geo_analysis'] = analyze_geographic_data(data_frames['countries'])
            vis_path = create_geographic_visualizations(analysis_results['geo_analysis'])
            if vis_path: visualization_paths['geo_visualizations'] = vis_path
            
        # Backlink Analysis
        if 'backlinks' in data_frames:
            logger.info("Performing Backlink Analysis...")
            try:
                backlinks_df = data_frames['backlinks']
                backlink_data = analyze_backlinks(backlinks_df) # Assuming it takes the backlinks df
                analysis_results['backlink_analysis'] = backlink_data
                vis_path = create_backlink_visualizations(backlink_data)
                if vis_path: visualization_paths['backlink_visualizations'] = vis_path
            except Exception as e:
                logger.error(f"Backlink analysis error: {str(e)}")

        # SEO Health Score (Needs multiple data types? Check function)
        # Let's assume it needs a variety, calculate if any relevant data exists
        if any(key in data_frames for key in ['queries', 'pages', 'devices', 'countries', 'dates', 'mobile', 'backlinks']):
            logger.info("Calculating SEO Health Score...")
            try:
                # Pass the whole dictionary, let the function handle it
                health_score = seo_health_analyzer.calculate_health_score(data_frames)
                analysis_results['seo_health'] = health_score
                vis_path = create_seo_health_visualizations(health_score)
                if vis_path: visualization_paths['seo_health_visualizations'] = vis_path
            except Exception as e:
                logger.error(f"SEO Health analysis error: {str(e)}")
        
        logger.info(f"Analysis complete. Results keys: {list(analysis_results.keys())}")
        logger.info(f"Visualization paths: {list(visualization_paths.keys())}")

        # Combine analysis text results and visualization paths for the report generator
        report_data = {**analysis_results, **visualization_paths}

        # Generate PDF report
        logger.info("Generating PDF report...")
        pdf_path = generate_pdf_report(report_data, api_key, data_frames)
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
        # Clean up temporary image files if visualization_paths was populated
        for path in visualization_paths.values():
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up temp image: {path}")
                except Exception as e:
                    logger.error(f"Error cleaning up temp file {path}: {str(e)}")

# --- Report Generation Function --- 
def generate_pdf_report(report_data: Dict, api_key: str, data_frames: Dict[str, pd.DataFrame]) -> str:
    """Generate a PDF report conditionally based on available analysis data."""
    logger.info("Generating PDF report with available data types: %s", list(data_frames.keys()))
    
    # Configure Gemini (keep existing try/except block)
    try:
        logger.info("Configuring Gemini API...")
        genai.configure(api_key=api_key)
        # Consider allowing model selection via config or request parameter later
        model = genai.GenerativeModel('gemini-1.5-pro-latest') 
        logger.info("Gemini API configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure Gemini API: {str(e)}")

    # PDF Setup
    pdf_buffer = BytesIO()
    # Use a known location for easier debugging if needed, ensure cleanup
    pdf_dir = tempfile.gettempdir()
    # Ensure unique filenames if running concurrently, e.g., using uuid
    import uuid
    pdf_filename = f"seo_analysis_report_{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(pdf_dir, pdf_filename)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom Styles (Optional - enhance appearance)
    styles.add(ParagraphStyle(name='Justify', alignment=4)) # TA_JUSTIFY = 4
    styles.add(ParagraphStyle(name='Code', fontName='Courier', fontSize=9, leading=11))

    # --- Report Header ---
    story.append(Paragraph("SEO Analysis Report", styles['h1']))
    story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Data Types Included: {', '.join(sorted(data_frames.keys()))}", styles['Normal']))
    story.append(Spacer(1, 12))

    # --- Executive Summary (Gemini) ---
    logger.info("Attempting to generate executive summary...")
    summary_text = "Summary generation failed or was skipped."
    try:
        # Build a more informative prompt
        prompt_parts = [
            "You are an expert SEO analyst. Analyze the following SEO data extracted from Google Search Console exports and provide a concise executive summary (approx. 3-5 paragraphs).",
            "Focus on the most impactful insights, key performance indicators (KPIs), significant trends (positive or negative), and actionable recommendations based *only* on the data provided.",
            "Structure the summary logically, starting with overall performance and then highlighting key areas like queries, pages, devices, etc., if data is available.",
            "\n--- Available Data Overview ---"
        ]
        for key, df in data_frames.items():
            prompt_parts.append(f"\n*   **{key.replace('_', ' ').title()} Data:** Present ({df.shape[0]} rows). Key columns: {list(df.columns)}")
            # Optionally add very brief stats like date range if available
            if 'date' in df.columns:
                try:
                    min_date = pd.to_datetime(df['date']).min().strftime('%Y-%m-%d')
                    max_date = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
                    prompt_parts[-1] += f" (Range: {min_date} to {max_date})"
                except Exception:
                    pass # Ignore date parsing errors

        prompt_parts.append("\n--- Detailed Analysis Snippets ---")
        # Include summaries from analysis_results if they exist
        analysis_summaries_found = False
        for key, data in report_data.items():
             # Look for dictionary results containing a 'summary' key
             if isinstance(data, dict) and 'summary' in data and isinstance(data['summary'], str):
                 analysis_summaries_found = True
                 prompt_parts.append(f"\n*   **{key.replace('_', ' ').title()} Analysis Summary:** {data['summary'][:500]}...") # Limit length
             # Could add top N rows from key dataframes here too if needed, carefully manage prompt size

        if not analysis_summaries_found:
             prompt_parts.append("\n(No detailed text summaries from automated analysis were provided; base summary on the raw data overview.)")

        prompt_parts.append("\n--- End of Data Context ---")
        prompt_parts.append("\nPlease generate the executive summary:")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Limit overall prompt size to avoid issues (adjust limit as needed)
        max_prompt_length = 30000 # Example limit for Gemini 1.5
        if len(full_prompt) > max_prompt_length:
            logger.warning(f"Prompt length ({len(full_prompt)}) exceeds limit ({max_prompt_length}), truncating.")
            full_prompt = full_prompt[:max_prompt_length]
            
        logger.debug(f"Sending final prompt to Gemini (length {len(full_prompt)}):")
        # logger.debug(full_prompt) # Potentially log full prompt if debugging needed

        response = model.generate_content(full_prompt)
        # Add error handling for blocked prompts / safety settings
        if response.prompt_feedback and response.prompt_feedback.block_reason:
             logger.error(f"Gemini prompt blocked. Reason: {response.prompt_feedback.block_reason}")
             summary_text = f"Error: Summary generation blocked by safety settings (Reason: {response.prompt_feedback.block_reason})."
        elif not response.text:
             logger.error("Gemini response contained no text.")
             summary_text = "Error: AI model returned an empty response."
        else:
            summary_text = response.text
            logger.info("Successfully received summary from Gemini.")
            
    except Exception as e:
        logger.error(f"Gemini API call failed: {str(e)}", exc_info=True) # Include traceback
        summary_text = f"Error: Could not generate summary using the AI model due to an unexpected error: {str(e)}"

    story.append(Paragraph("Executive Summary", styles['h2']))
    # Use the Justify style if defined, otherwise Normal
    summary_style = styles.get('Justify', styles['Normal'])
    story.append(Paragraph(summary_text.replace('\n', '<br/>'), summary_style))
    story.append(Spacer(1, 12))

    # --- Helper function to add sections --- 
    def add_report_section(title, analysis_key, visualization_key=None, data_formatter=None):
        if analysis_key in report_data:
            logger.info(f"Adding section: {title}")
            story.append(Paragraph(title, styles['h2']))
            data = report_data[analysis_key]
            
            # Add textual summary if exists
            if isinstance(data, dict) and 'summary' in data:
                story.append(Paragraph(data['summary'].replace('\n', '<br/>'), summary_style))
                story.append(Spacer(1, 6))
                
            # Add specific formatted data using the provided formatter function
            if data_formatter:
                 try:
                     formatted_elements = data_formatter(data)
                     story.extend(formatted_elements)
                 except Exception as e:
                     logger.error(f"Error formatting data for {title}: {str(e)}", exc_info=True)
                     story.append(Paragraph(f"Error displaying data for {title}.", styles['Normal']))

            # Add visualization if key exists and path is valid
            if visualization_key and visualization_key in report_data:
                img_path = report_data[visualization_key]
                if img_path and os.path.exists(img_path):
                    logger.info(f"Adding image {img_path} to PDF section {title}.")
                    try:
                        img = Image(img_path, width=500, height=250) # Adjust size
                        img.hAlign = 'CENTER'
                        story.append(img)
                        story.append(Spacer(1, 12))
                    except Exception as e:
                        logger.error(f"Error adding image {img_path}: {str(e)}")
                else:
                    logger.warning(f"Visualization image not found or invalid path: {img_path}")
            story.append(Spacer(1, 12))
            return True
        return False

    # --- Helper function for simple tables --- 
    def create_simple_table(df: pd.DataFrame, title: str = None):
        elements = []
        if title:
            elements.append(Paragraph(title, styles['h3']))
        if not df.empty:
            # Limit columns and rows for display
            df_display = df.head(15).copy() # Limit rows
            # Truncate long cell values
            for col in df_display.select_dtypes(include='object').columns:
                 df_display[col] = df_display[col].str.slice(0, 100) # Limit string length
                 
            table_data = [df_display.columns.to_list()] + df_display.values.tolist()
            # Basic styling
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

    # --- Define Data Formatters for specific sections --- 
    def format_query_data(data):
        elements = []
        if isinstance(data, dict):
            if 'top_queries_by_clicks' in data and isinstance(data['top_queries_by_clicks'], pd.DataFrame):
                 elements.extend(create_simple_table(data['top_queries_by_clicks'], "Top Queries by Clicks"))
            if 'top_queries_by_impressions' in data and isinstance(data['top_queries_by_impressions'], pd.DataFrame):
                 elements.extend(create_simple_table(data['top_queries_by_impressions'], "Top Queries by Impressions"))
            # Add more tables/paragraphs as needed based on query_analyzer output
        return elements
        
    def format_keyword_research(data):
        elements = []
        if isinstance(data, dict):
             if 'trend_data' in data:
                 elements.append(Paragraph("Keyword Trends (Sample)", styles['h3']))
                 # Format trend data - maybe a simple list or small table
                 elements.append(Paragraph(json.dumps(data['trend_data'], indent=2), styles['Code'])) 
             if 'related_queries' in data:
                 elements.append(Paragraph("Related Query Suggestions", styles['h3']))
                 elements.append(Paragraph(json.dumps(data['related_queries'], indent=2), styles['Code']))
             if 'difficulty' in data:
                 elements.append(Paragraph("Keyword Difficulty Analysis", styles['h3']))
                 elements.append(Paragraph(json.dumps(data['difficulty'], indent=2), styles['Code']))
        return elements
        
    def format_competitor_data(data):
         elements = []
         if isinstance(data, dict):
             if 'competitors' in data:
                 elements.append(Paragraph("Identified Competitors", styles['h3']))
                 elements.append(Paragraph(json.dumps(data['competitors'], indent=2), styles['Code']))
             if 'content_analysis' in data:
                 elements.append(Paragraph("Top Competitor Content Analysis", styles['h3']))
                 elements.append(Paragraph(json.dumps(data['content_analysis'], indent=2), styles['Code']))
             if 'ranking_comparison' in data:
                 elements.append(Paragraph("Ranking Comparison", styles['h3']))
                 # Might need a table here if data is structured
                 elements.append(Paragraph(json.dumps(data['ranking_comparison'], indent=2), styles['Code']))
         return elements

    def format_seo_health(data):
        elements = []
        if isinstance(data, dict):
            overall_score = data.get('overall_score', 0)
            elements.append(Paragraph(f"Overall Score: {overall_score:.1f}/100", styles['h3']))
            # Add score explanation
            score_explanation = "Excellent" if overall_score >= 80 else "Good" if overall_score >= 60 else "Needs Improvement" if overall_score >= 40 else "Critical"
            elements.append(Paragraph(f"Assessment: {score_explanation}", styles['Normal']))
            elements.append(Spacer(1,6))
            if 'component_scores' in data:
                 comp_df = pd.DataFrame(list(data['component_scores'].items()), columns=['Component', 'Score'])
                 comp_df['Component'] = comp_df['Component'].str.replace('_', ' ').str.title()
                 comp_df['Score'] = comp_df['Score'].round(1)
                 elements.extend(create_simple_table(comp_df, "Component Scores"))
            if 'recommendations' in data and data['recommendations']:
                 elements.append(Paragraph("Recommendations", styles['h3']))
                 for i, rec in enumerate(data['recommendations'][:5]):
                     elements.append(Paragraph(f"{i+1}. {rec.get('title', 'Recommendation')}", styles['Normal']))
                     elements.append(Paragraph(rec.get('description', ''), styles['Definition'])) # Use Definition style
                     elements.append(Spacer(1, 4))
        return elements
        
    # Add more formatters for device, temporal, geo, mobile, cannibalization, serp features, etc. following the pattern
    # Example for Device Data
    def format_device_data(data):
        elements = []
        if isinstance(data, dict) and 'device_performance' in data and isinstance(data['device_performance'], pd.DataFrame):
            elements.extend(create_simple_table(data['device_performance'], "Performance by Device"))
        # Add more based on device_analysis output structure
        return elements
        
    # Example for Temporal Data
    def format_temporal_data(data):
        elements = []
        if isinstance(data, dict) and 'trend_summary' in data and isinstance(data['trend_summary'], pd.DataFrame):
             elements.extend(create_simple_table(data['trend_summary'], "Overall Trend Summary"))
        # Add more based on temporal_analysis output
        return elements
        
    # Example for Geo Data
    def format_geo_data(data):
        elements = []
        if isinstance(data, dict) and 'performance_by_country' in data and isinstance(data['performance_by_country'], pd.DataFrame):
             elements.extend(create_simple_table(data['performance_by_country'], "Performance by Country (Top 15)"))
        # Add more based on geo_analysis output
        return elements
        
    def format_generic_data(data):
        # Fallback formatter for analysis results that are just dicts/strings
        elements = []
        if isinstance(data, dict):
            elements.append(Paragraph(json.dumps(data, indent=2, default=str), styles['Code']))
        elif isinstance(data, str):
            elements.append(Paragraph(data.replace('\n', '<br/>'), summary_style))
        return elements

    # --- Build Report Sections Conditionally ---
    add_report_section("Query Performance", 'query_analysis', 'query_visualizations', format_query_data)
    add_report_section("Device Performance", 'device_analysis', 'device_visualizations', format_device_data)
    add_report_section("Geographic Performance", 'geo_analysis', 'geo_visualizations', format_geo_data)
    add_report_section("Temporal Trends", 'temporal_analysis', 'temporal_visualizations', format_temporal_data)
    add_report_section("Keyword Research", 'keyword_research', 'keyword_visualizations', format_keyword_research)
    add_report_section("Competitor Analysis", 'competitor_analysis', 'competitor_visualizations', format_competitor_data)
    add_report_section("Content Gaps", 'content_gaps', 'content_gap_visualizations', format_generic_data) # Needs formatter
    add_report_section("Mobile Friendliness", 'mobile_optimization', 'mobile_visualizations', format_generic_data) # Needs formatter
    add_report_section("Keyword Cannibalization", 'keyword_cannibalization', 'cannibalization_visualizations', format_generic_data) # Needs formatter
    add_report_section("SERP Feature Analysis", 'serp_features', 'serp_visualizations', format_generic_data) # Needs formatter
    add_report_section("SERP Previews", 'serp_previews', 'serp_preview_visualizations', format_generic_data) # Needs formatter
    add_report_section("Backlink Analysis", 'backlink_analysis', 'backlink_visualizations', format_generic_data) # Needs formatter
    add_report_section("Overall SEO Health", 'seo_health', 'seo_health_visualizations', format_seo_health)
    
    # --- Build PDF --- 
    logger.info("Building final PDF document...")
    try:
        doc.build(story)
        logger.info(f"PDF document built successfully: {pdf_path}")
    except Exception as e:
         logger.error(f"Error building PDF: {str(e)}", exc_info=True)
         # Clean up the potentially corrupted file
         if os.path.exists(pdf_path):
              os.remove(pdf_path)
         raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")

    return pdf_path

@app.get("/")
async def root():
    return {"message": "SEO Seer Report Genius API is running."}

# Optional: Add cleanup for older temp PDF files on startup if needed
