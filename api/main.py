from fastapi import FastAPI, UploadFile, Form, HTTPException
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
async def analyze_seo(file: UploadFile, api_key: str = Form(...)):
    try:
        # Process uploaded file
        content = await file.read()
        processed_files = [{
            'filename': file.filename,
            'content': io.StringIO(content.decode('utf-8', errors='replace'))
        }]
        
        # Organize files by type
        data_frames = process_csv_files(processed_files)
        
        # Initialize analyzers
        query_analyzer = QueryAnalyzer()
        keyword_researcher = KeywordResearcher()
        competitor_analyzer = CompetitorAnalyzer()
        seo_health_analyzer = SEOHealthAnalyzer()
        
        # Perform analyses
        analysis_results = {}
        
        # Query analysis if available
        if 'queries' in data_frames:
            queries_df = data_frames['queries']
            
            # Base query analysis
            analysis_results['query_analysis'] = query_analyzer.analyze_query_patterns(queries_df)
            analysis_results['query_visualizations'] = create_query_visualizations(
                analysis_results['query_analysis']
            )
            
            # Extract top queries for additional analysis
            top_queries = queries_df.sort_values('impressions', ascending=False)['query'].head(10).tolist()
            
            # Add keyword research
            try:
                keyword_data = {}
                
                # Research trends for top queries
                trends = keyword_researcher.research_keyword_trends(top_queries[:5])
                keyword_data['trend_data'] = trends
                
                # Get keyword suggestions for the top query
                if top_queries:
                    suggestions = keyword_researcher.get_keyword_suggestions(top_queries[0])
                    keyword_data['related_queries'] = {top_queries[0]: {'top': suggestions}}
                
                # Estimate keyword difficulty
                difficulties = keyword_researcher.analyze_keyword_difficulty(top_queries[:5])
                keyword_data['difficulty'] = difficulties
                
                analysis_results['keyword_research'] = keyword_data
                analysis_results['keyword_visualizations'] = create_keyword_research_visualizations(keyword_data)
            except Exception as e:
                print(f"Keyword research error: {str(e)}")
            
            # Add competitor analysis
            try:
                competitor_data = {}
                
                # Identify competitors based on top queries
                competitors = competitor_analyzer.identify_competitors(top_queries)
                competitor_data['competitors'] = competitors
                
                # If we have competitors, analyze the top one
                if competitors:
                    top_competitor = list(competitors.keys())[0]
                    competitor_url = f"https://{top_competitor}"
                    
                    # Analyze competitor content
                    content_analysis = competitor_analyzer.analyze_competitor_content(competitor_url)
                    competitor_data['content_analysis'] = content_analysis
                    
                    # Compare rankings
                    ranking_comparison = competitor_analyzer.compare_rankings(top_queries[:5], top_competitor)
                    competitor_data['ranking_comparison'] = ranking_comparison
                
                analysis_results['competitor_analysis'] = competitor_data
                analysis_results['competitor_visualizations'] = create_competitor_visualizations(competitor_data)
            except Exception as e:
                print(f"Competitor analysis error: {str(e)}")
            
            # Add SEO Health Score
            try:
                health_score = seo_health_analyzer.calculate_health_score(data_frames)
                analysis_results['seo_health'] = health_score
                analysis_results['seo_health_visualizations'] = create_seo_health_visualizations(health_score)
            except Exception as e:
                print(f"SEO Health analysis error: {str(e)}")
            
            # Add Keyword Cannibalization Detection
            try:
                cannibalization_results = detect_keyword_cannibalization(data_frames)
                analysis_results['keyword_cannibalization'] = cannibalization_results
                analysis_results['cannibalization_visualizations'] = create_cannibalization_visualizations(cannibalization_results)
            except Exception as e:
                print(f"Cannibalization analysis error: {str(e)}")
            
            # Add SERP Feature Analysis
            try:
                serp_features = analyze_serp_features(data_frames)
                analysis_results['serp_features'] = serp_features
                analysis_results['serp_visualizations'] = create_serp_feature_visualizations(serp_features)
            except Exception as e:
                print(f"SERP feature analysis error: {str(e)}")
                
            # Add SERP Preview Generator
            try:
                serp_previews = generate_serp_previews(data_frames)
                analysis_results['serp_previews'] = serp_previews
                analysis_results['serp_preview_visualizations'] = create_serp_preview_visualizations(serp_previews)
            except Exception as e:
                print(f"SERP preview error: {str(e)}")
                
            # Add Backlink Analysis
            try:
                backlink_data = analyze_backlinks(data_frames)
                analysis_results['backlink_analysis'] = backlink_data
                analysis_results['backlink_visualizations'] = create_backlink_visualizations(backlink_data)
            except Exception as e:
                print(f"Backlink analysis error: {str(e)}")
                
            # Add Content Gap Analysis
            try:
                content_gaps = analyze_content_gaps(data_frames, analysis_results.get('competitor_analysis', None))
                analysis_results['content_gaps'] = content_gaps
                analysis_results['content_gap_visualizations'] = create_content_gap_visualizations(content_gaps)
            except Exception as e:
                print(f"Content gap analysis error: {str(e)}")
        
        # Device analysis if available
        if 'devices' in data_frames:
            analysis_results['device_analysis'] = analyze_device_data(
                data_frames['devices']
            )
            analysis_results['device_visualizations'] = create_device_visualizations(
                analysis_results['device_analysis']
            )
            
            # Add Mobile Optimization Analysis
            try:
                mobile_optimization = analyze_mobile_friendliness(data_frames)
                analysis_results['mobile_optimization'] = mobile_optimization
                analysis_results['mobile_visualizations'] = create_mobile_visualizations(mobile_optimization)
            except Exception as e:
                print(f"Mobile optimization analysis error: {str(e)}")
        
        # Temporal analysis if available
        if 'dates' in data_frames:
            analysis_results['temporal_analysis'] = analyze_temporal_patterns(
                data_frames['dates']
            )
            analysis_results['temporal_visualizations'] = create_temporal_visualizations(
                analysis_results['temporal_analysis']
            )
        
        # Geographic analysis if available
        if 'countries' in data_frames:
            analysis_results['geographic_analysis'] = analyze_geographic_data(
                data_frames['countries']
            )
            analysis_results['geographic_visualizations'] = create_geographic_visualizations(
                analysis_results['geographic_analysis']
            )
        
        # Generate comprehensive report
        report_path = generate_pdf_report(analysis_results, api_key)
        
        return FileResponse(
            path=report_path,
            filename="seo_analysis_report.pdf",
            media_type="application/pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_pdf_report(analysis_results: Dict, api_key: str) -> str:
    """Generate a comprehensive PDF report from the analysis results."""
    # Create a temporary file for the PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.close()
    
    # Create the PDF document
    doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title
    title_style = styles['Title']
    elements.append(Paragraph("SEO Seer Pro Analysis Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Add introduction
    elements.append(Paragraph("Executive Summary", styles['Heading1']))
    
    # Generate executive summary with Gemini AI
    try:
        # Configure Gemini API with the provided key
        genai.configure(api_key=api_key)
        
        # Set up generation configuration
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 4000,
        }
        
        # Create the model with appropriate parameters
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config
        )
        
        # Prepare a simplified version of the analysis results (to avoid token limits)
        simplified_results = {}
        for key, value in analysis_results.items():
            if key.endswith('_visualizations'):
                continue  # Skip visualization data
            simplified_results[key] = value
            
        # Create prompt for the summary
        summary_prompt = f"""
        Create a concise executive summary for an SEO analysis report based on the following data:
        {json.dumps(simplified_results, default=str)}
        
        Focus on key insights, trends, and actionable recommendations. Keep it under 300 words.
        """
        
        # Generate the summary
        response = model.generate_content(summary_prompt)
        summary_text = response.text
        
        # Add the generated summary to the PDF
        elements.append(Paragraph(summary_text, styles['Normal']))
    except Exception as e:
        # Handle any errors with the Gemini API
        error_message = f"Unable to generate AI summary. Error: {str(e)}"
        print(error_message)
        elements.append(Paragraph("Unable to generate AI summary. Please check your API key or try again later.", styles['Normal']))
    
    elements.append(Spacer(1, 20))
    
    # Add SEO Health Score section if available
    if 'seo_health' in analysis_results:
        elements.append(Paragraph("SEO Health Score", styles['Heading1']))
        
        health_data = analysis_results['seo_health']
        overall_score = health_data.get('overall_score', 0)
        
        # Add overall score
        elements.append(Paragraph(f"Overall Score: {overall_score}/100", styles['Heading2']))
        
        # Add score explanation based on range
        score_explanation = ""
        if overall_score >= 80:
            score_explanation = "Your site has excellent SEO health. Continue maintaining best practices."
        elif overall_score >= 60:
            score_explanation = "Your site has good SEO health with room for improvement in specific areas."
        elif overall_score >= 40:
            score_explanation = "Your site requires significant SEO improvements to compete effectively."
        else:
            score_explanation = "Your site has critical SEO issues that need immediate attention."
        
        elements.append(Paragraph(score_explanation, styles['Normal']))
        elements.append(Spacer(1, 10))
        
        # Add component scores table
        if 'component_scores' in health_data:
            elements.append(Paragraph("Component Scores", styles['Heading2']))
            
            component_scores = health_data['component_scores']
            
            # Create a table for component scores
            data = [["Component", "Score", "Status"]]
            
            for component, score in component_scores.items():
                component_name = component.replace('_', ' ').title()
                
                # Determine status based on score
                status = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement" if score >= 40 else "Critical"
                
                data.append([component_name, f"{score:.1f}/100", status])
            
            t = Table(data, colWidths=[200, 100, 150])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Left-align first column
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t)
            elements.append(Spacer(1, 20))
        
        # Add recommendations
        if 'recommendations' in health_data and health_data['recommendations']:
            elements.append(Paragraph("Health Score Recommendations", styles['Heading2']))
            
            for i, rec in enumerate(health_data['recommendations'][:5]):  # Limit to top 5
                elements.append(Paragraph(f"{i+1}. {rec['title']}", styles['Heading3']))
                elements.append(Paragraph(rec['description'], styles['Normal']))
                elements.append(Spacer(1, 10))
            
            elements.append(Spacer(1, 10))
    
    # ... keep existing code (remaining PDF report generation including all analysis sections)
    
    # Add final recommendations from AI
    elements.append(Paragraph("Strategic Recommendations", styles['Heading1']))
    
    try:
        # Create a simplified version for the final recommendations
        simplified_results = {}
        for key, value in analysis_results.items():
            if key.endswith('_visualizations'):
                continue  # Skip visualization data
            simplified_results[key] = value
            
        # Create the prompt for final recommendations
        final_prompt = f"""
        Based on this comprehensive SEO analysis including keyword research and competitor analysis:
        {json.dumps(simplified_results, default=str)}
        
        Provide 5-7 strategic, actionable recommendations to improve SEO performance.
        Each recommendation should be specific, measurable, and prioritized.
        Format each recommendation with a clear action item followed by the expected benefit.
        Include specific advice about:
        1. Content optimization based on competitor analysis
        2. Keyword targeting opportunities
        3. Device-specific optimizations
        4. Geographic targeting
        5. Technical SEO improvements
        """
        
        # Generate recommendations
        final_insights = model.generate_content(final_prompt)
        elements.append(Paragraph(final_insights.text, styles['Normal']))
    except Exception as e:
        error_message = f"Unable to generate AI recommendations. Error: {str(e)}"
        print(error_message)
        elements.append(Paragraph("Unable to generate AI recommendations. Please check your API key or try again later.", styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    
    return temp_file.name

@app.get("/")
async def root():
    return {"message": "SEO Seer Pro API is running. Use /analyze-seo endpoint to analyze your data."}
