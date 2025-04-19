
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
        keyword_researcher = KeywordResearcher()
        competitor_analyzer = CompetitorAnalyzer()
        
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
    elements.append(Paragraph("SEO Seer Pro Analysis Report", title_style))
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
    
    # Add keyword research section if available
    if 'keyword_research' in analysis_results:
        elements.append(Paragraph("Keyword Research & Trends", styles['Heading1']))
        elements.append(Spacer(1, 10))
        
        # Add keyword difficulty information
        if 'difficulty' in analysis_results['keyword_research']:
            elements.append(Paragraph("Keyword Difficulty Analysis", styles['Heading2']))
            
            difficulties = analysis_results['keyword_research']['difficulty']
            data = [["Keyword", "Difficulty Score", "Recommendation"]]
            
            for keyword, score in difficulties.items():
                recommendation = "Easy to rank for" if score < 30 else "Moderate competition" if score < 70 else "Highly competitive"
                data.append([keyword, f"{score:.1f}/100", recommendation])
            
            t = Table(data, colWidths=[200, 100, 200])
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
        
        # Add related queries if available
        if 'related_queries' in analysis_results['keyword_research']:
            elements.append(Paragraph("Related Search Queries", styles['Heading2']))
            
            for keyword, queries in analysis_results['keyword_research']['related_queries'].items():
                elements.append(Paragraph(f"Related to: {keyword}", styles['Heading3']))
                
                if 'top' in queries and queries['top']:
                    data = [["Related Query", "Score", "Type"]]
                    
                    for query_info in queries['top'][:10]:  # Top 10 related queries
                        data.append([
                            query_info['keyword'],
                            str(query_info['score']),
                            query_info['type'].capitalize()
                        ])
                    
                    t = Table(data, colWidths=[250, 100, 100])
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
    
    # Add competitor analysis section if available
    if 'competitor_analysis' in analysis_results:
        elements.append(Paragraph("Competitor Analysis", styles['Heading1']))
        elements.append(Spacer(1, 10))
        
        # Add competitors overview
        if 'competitors' in analysis_results['competitor_analysis']:
            elements.append(Paragraph("Top Competitors", styles['Heading2']))
            
            competitors = analysis_results['competitor_analysis']['competitors']
            data = [["Competitor Domain", "Appearances", "Share (%)"]]
            
            for domain, metrics in competitors.items():
                data.append([
                    domain,
                    str(metrics['appearances']),
                    f"{metrics['share']:.2f}%"
                ])
            
            t = Table(data, colWidths=[200, 100, 150])
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
        
        # Add content analysis if available
        if 'content_analysis' in analysis_results['competitor_analysis']:
            content = analysis_results['competitor_analysis']['content_analysis']
            elements.append(Paragraph("Competitor Content Analysis", styles['Heading2']))
            
            # Content metrics
            data = [["Metric", "Value"]]
            data.append(["Word Count", str(content.get('word_count', 0))])
            data.append(["Internal Links", str(content.get('links', {}).get('internal', 0))])
            data.append(["External Links", str(content.get('links', {}).get('external', 0))])
            
            t = Table(data, colWidths=[200, 250])
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
            
            # Keyword density
            if 'keyword_density' in content and content['keyword_density']:
                elements.append(Paragraph("Top Keywords in Competitor Content", styles['Heading3']))
                
                data = [["Keyword", "Density (%)"]]
                for keyword, density in list(content['keyword_density'].items())[:10]:
                    data.append([keyword, f"{density:.2f}%"])
                
                t = Table(data, colWidths=[200, 100])
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
    Based on this comprehensive SEO analysis including keyword research and competitor analysis:
    {json.dumps(analysis_results, default=str)}
    
    Provide 5-7 strategic, actionable recommendations to improve SEO performance.
    Each recommendation should be specific, measurable, and prioritized.
    Format each recommendation with a clear action item followed by the expected benefit.
    Include specific advice about:
    1. Content optimization based on competitor analysis
    2. Keyword targeting opportunities
    3. Device-specific optimizations
    4. Geographic targeting
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
    return {"message": "SEO Seer Pro API is running. Use /analyze-seo endpoint to analyze your data."}
