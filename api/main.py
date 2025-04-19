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
from utils.visualization import (
    create_query_visualizations,
    create_temporal_visualizations,
    create_geographic_visualizations,
    create_device_visualizations
)

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
            'content': io.StringIO(content.decode())
        }]
        
        # Organize files by type
        data_frames = process_csv_files(processed_files)
        
        # Initialize analyzers
        query_analyzer = QueryAnalyzer()
        
        # Perform analyses
        analysis_results = {}
        
        # Query analysis if available
        if 'queries' in data_frames:
            analysis_results['query_analysis'] = query_analyzer.analyze_query_patterns(
                data_frames['queries']
            )
            analysis_results['query_visualizations'] = create_query_visualizations(
                analysis_results['query_analysis']
            )
        
        # Device analysis if available
        if 'devices' in data_frames:
            analysis_results['device_analysis'] = analyze_device_data(
                data_frames['devices']
            )
            analysis_results['device_visualizations'] = create_device_visualizations(
                analysis_results['device_analysis']
            )
        
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
    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # Create a temporary file for the PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.close()
    
    # Create the PDF document
    doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title
    title_style = styles['Title']
    elements.append(Paragraph("SEO Analysis Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Add introduction
    elements.append(Paragraph("Executive Summary", styles['Heading1']))
    
    # Generate executive summary with Gemini
    summary_prompt = f"""
    Create a concise executive summary for an SEO analysis report based on the following data:
    {json.dumps(analysis_results, default=str)}
    
    Focus on key insights, trends, and actionable recommendations. Keep it under 300 words.
    """
    
    try:
        summary_response = model.generate_content(summary_prompt)
        summary_text = summary_response.text
    except Exception as e:
        summary_text = "Unable to generate AI summary. Please check your API key or try again later."
    
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Add query analysis section if available
    if 'query_analysis' in analysis_results:
        elements.append(Paragraph("Query Analysis", styles['Heading1']))
        
        # Add query clusters information
        elements.append(Paragraph("Query Clusters", styles['Heading2']))
        clusters = analysis_results['query_analysis']['query_clusters']
        
        for i, cluster in enumerate(clusters[:5]):  # Limit to top 5 clusters
            elements.append(Paragraph(f"Cluster {i+1}", styles['Heading3']))
            
            # Create a table for cluster data
            data = [["Queries", "Avg Position", "Total Clicks", "Total Impressions"]]
            data.append([
                ", ".join(cluster['queries'][:5]) + ("..." if len(cluster['queries']) > 5 else ""),
                f"{cluster['avg_position']:.2f}",
                str(cluster['total_clicks']),
                str(cluster['total_impressions'])
            ])
            
            t = Table(data, colWidths=[300, 70, 70, 70])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t)
            elements.append(Spacer(1, 10))
        
        # Add performance metrics
        elements.append(Paragraph("Performance Metrics", styles['Heading2']))
        metrics = analysis_results['query_analysis']['performance_metrics']
        
        data = [["Metric", "Value"]]
        data.append(["Average Position", f"{metrics['avg_position']:.2f}"])
        data.append(["Total Clicks", str(metrics['total_clicks'])])
        data.append(["Total Impressions", str(metrics['total_impressions'])])
        data.append(["Average CTR", f"{metrics['avg_ctr']:.2f}%"])
        
        t = Table(data, colWidths=[200, 300])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))
        
        # Generate AI insights for query analysis
        query_prompt = f"""
        Analyze this query performance data and provide 3-5 specific, actionable recommendations:
        {json.dumps(analysis_results['query_analysis'], default=str)}
        
        Focus on:
        1. Keyword opportunities
        2. Content optimization suggestions
        3. Specific queries to prioritize
        """
        
        try:
            query_insights = model.generate_content(query_prompt)
            elements.append(Paragraph("AI Insights & Recommendations", styles['Heading2']))
            elements.append(Paragraph(query_insights.text, styles['Normal']))
        except Exception as e:
            pass
    
    # Add device analysis section if available
    if 'device_analysis' in analysis_results:
        elements.append(Paragraph("Device Analysis", styles['Heading1']))
        elements.append(Spacer(1, 10))
        
        device_data = analysis_results['device_analysis']['metrics']
        data = [["Device", "Clicks", "Impressions", "Avg Position", "Avg CTR", "Share"]]
        
        for device, metrics in device_data.items():
            data.append([
                device,
                str(metrics['total_clicks']),
                str(metrics['total_impressions']),
                f"{metrics['avg_position']:.2f}",
                f"{metrics['avg_ctr']:.2f}%",
                f"{metrics['impression_share']:.2f}%"
            ])
        
        t = Table(data, colWidths=[80, 70, 90, 90, 80, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 10))
        
        # Add recommendations
        if 'recommendations' in analysis_results['device_analysis']:
            elements.append(Paragraph("Device Recommendations", styles['Heading2']))
            for rec in analysis_results['device_analysis']['recommendations']:
                elements.append(Paragraph(f"• {rec}", styles['Normal']))
    
    # Add temporal analysis section if available
    if 'temporal_analysis' in analysis_results:
        elements.append(Paragraph("Temporal Analysis", styles['Heading1']))
        elements.append(Spacer(1, 10))
        
        # Add growth metrics
        if 'growth_metrics' in analysis_results['temporal_analysis']:
            elements.append(Paragraph("Growth Trends", styles['Heading2']))
            growth = analysis_results['temporal_analysis']['growth_metrics']
            
            data = [["Metric", "Growth Rate", "Trend", "Volatility"]]
            for metric, values in growth.items():
                data.append([
                    metric.title(),
                    f"{values['growth_rate']:.2f}%",
                    values['trend_direction'].upper(),
                    f"{values['volatility']:.2f}%"
                ])
            
            t = Table(data, colWidths=[100, 100, 100, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t)
            elements.append(Spacer(1, 20))
    
    # Add geographic analysis section if available
    if 'geographic_analysis' in analysis_results:
        elements.append(Paragraph("Geographic Analysis", styles['Heading1']))
        elements.append(Spacer(1, 10))
        
        # Add opportunities
        if 'opportunities' in analysis_results['geographic_analysis']:
            elements.append(Paragraph("Geographic Opportunities", styles['Heading2']))
            opps = analysis_results['geographic_analysis']['opportunities']
            
            for opp in opps[:5]:  # Limit to top 5
                if opp['type'] == 'ctr_improvement':
                    elements.append(Paragraph(
                        f"• {opp['country']}: CTR improvement opportunity. Current: {opp['current_ctr']:.2f}%, " +
                        f"Benchmark: {opp['benchmark_ctr']:.2f}%",
                        styles['Normal']
                    ))
                else:
                    elements.append(Paragraph(
                        f"• {opp['country']}: Ranking improvement opportunity. Current position: {opp['current_position']:.2f}",
                        styles['Normal']
                    ))
            
            elements.append(Spacer(1, 20))
    
    # Add final recommendations from AI
    elements.append(Paragraph("Strategic Recommendations", styles['Heading1']))
    
    final_prompt = f"""
    Based on this comprehensive SEO analysis:
    {json.dumps(analysis_results, default=str)}
    
    Provide 5-7 strategic, actionable recommendations to improve SEO performance.
    Each recommendation should be specific, measurable, and prioritized.
    Format each recommendation with a clear action item followed by the expected benefit.
    """
    
    try:
        final_insights = model.generate_content(final_prompt)
        elements.append(Paragraph(final_insights.text, styles['Normal']))
    except Exception as e:
        elements.append(Paragraph("Unable to generate AI recommendations. Please check your API key or try again later.", styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    
    return temp_file.name

@app.get("/")
async def root():
    return {"message": "SEO Seer API is running. Use /analyze-seo endpoint to analyze your data."}
