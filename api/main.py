
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
            'content': io.StringIO(content.decode())
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
    
    # Add mobile optimization section if available
    if 'mobile_optimization' in analysis_results:
        elements.append(Paragraph("Mobile Optimization Analysis", styles['Heading1']))
        
        mobile_data = analysis_results['mobile_optimization']
        mobile_score = mobile_data.get('mobile_score', 0)
        
        # Add mobile score
        elements.append(Paragraph(f"Mobile Optimization Score: {mobile_score:.1f}/100", styles['Heading2']))
        
        # Add score explanation
        score_explanation = ""
        if mobile_score >= 80:
            score_explanation = "Your site is well optimized for mobile devices."
        elif mobile_score >= 60:
            score_explanation = "Your site has good mobile optimization with some improvements needed."
        elif mobile_score >= 40:
            score_explanation = "Your site needs significant mobile optimization improvements."
        else:
            score_explanation = "Your site has critical mobile usability issues that require immediate attention."
        
        elements.append(Paragraph(score_explanation, styles['Normal']))
        elements.append(Spacer(1, 10))
        
        # Add mobile issues summary if available
        if 'issues_summary' in mobile_data and mobile_data['issues_summary']:
            elements.append(Paragraph("Mobile Usability Issues", styles['Heading2']))
            
            issues = mobile_data['issues_summary']
            
            # Create a table for issues
            data = [["Issue Type", "Count"]]
            
            for issue, count in issues.items():
                issue_name = issue.replace('_', ' ').title()
                data.append([issue_name, str(count)])
            
            t = Table(data, colWidths=[300, 100])
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
        if 'recommendations' in mobile_data and mobile_data['recommendations']:
            elements.append(Paragraph("Mobile Optimization Recommendations", styles['Heading2']))
            
            for i, rec in enumerate(mobile_data['recommendations'][:5]):  # Limit to top 5
                elements.append(Paragraph(f"{i+1}. {rec['issue']}", styles['Heading3']))
                elements.append(Paragraph(rec['description'], styles['Normal']))
                elements.append(Paragraph(f"Impact: {rec['impact'].title()}", styles['Italic']))
                elements.append(Spacer(1, 10))
            
            elements.append(Spacer(1, 10))
    
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
    
    # Add keyword cannibalization section if available
    if 'keyword_cannibalization' in analysis_results:
        elements.append(Paragraph("Keyword Cannibalization Analysis", styles['Heading1']))
        
        cannibalization_data = analysis_results['keyword_cannibalization']
        
        # Add summary
        if 'severity_summary' in cannibalization_data:
            severity = cannibalization_data['severity_summary']
            
            elements.append(Paragraph("Cannibalization Summary", styles['Heading2']))
            
            # Create a table for severity summary
            data = [["Severity", "Count"]]
            data.append(["High", str(severity['high'])])
            data.append(["Medium", str(severity['medium'])])
            data.append(["Low", str(severity['low'])])
            data.append(["Total", str(severity['high'] + severity['medium'] + severity['low'])])
            
            t = Table(data, colWidths=[200, 100])
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
            elements.append(Spacer(1, 10))
        
        # Add top cannibalization issues
        if 'cannibalization_issues' in cannibalization_data and cannibalization_data['cannibalization_issues']:
            elements.append(Paragraph("Top Cannibalization Issues", styles['Heading2']))
            
            issues = cannibalization_data['cannibalization_issues'][:5]  # Top 5 issues
            
            for i, issue in enumerate(issues):
                elements.append(Paragraph(f"Issue {i+1}: {issue['query']}", styles['Heading3']))
                
                # Create a table for issue details
                data = [["Severity", issue['severity'].upper()]]
                data.append(["Primary Page", issue['primary_page']])
                data.append(["Competing Pages", ", ".join(issue['competing_pages'][:2]) + 
                            ("..." if len(issue['competing_pages']) > 2 else "")])
                data.append(["Position Range", f"{issue['metrics']['best_position']:.1f} - {issue['metrics']['worst_position']:.1f}"])
                data.append(["Impressions", str(issue['metrics']['total_impressions'])])
                
                t = Table(data, colWidths=[150, 350])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(t)
                elements.append(Spacer(1, 10))
        
        # Add recommendations
        if 'recommendations' in cannibalization_data and cannibalization_data['recommendations']:
            elements.append(Paragraph("Cannibalization Recommendations", styles['Heading2']))
            
            for i, rec in enumerate(cannibalization_data['recommendations'][:5]):  # Limit to top 5
                elements.append(Paragraph(f"{i+1}. {rec['title']}", styles['Heading3']))
                elements.append(Paragraph(rec['description'], styles['Normal']))
                elements.append(Paragraph(f"Impact: {rec['impact'].title()}", styles['Italic']))
                elements.append(Spacer(1, 10))
            
            elements.append(Spacer(1, 10))
    
    # Add SERP features section if available
    if 'serp_features' in analysis_results:
        elements.append(Paragraph("SERP Feature Analysis", styles['Heading1']))
        
        serp_data = analysis_results['serp_features']
        
        # Add feature summary
        if 'feature_summary' in serp_data and serp_data['feature_summary']:
            elements.append(Paragraph("SERP Feature Distribution", styles['Heading2']))
            
            features = serp_data['feature_summary']
            
            # Create a table for feature summary
            data = [["Feature Type", "Occurrences"]]
            
            for feature, count in sorted(features.items(), key=lambda x: x[1], reverse=True):
                feature_name = feature.replace('_', ' ').title()
                data.append([feature_name, str(count)])
            
            t = Table(data, colWidths=[250, 150])
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
        
        # Add feature opportunities
        if 'feature_opportunities' in serp_data and serp_data['feature_opportunities']:
            elements.append(Paragraph("SERP Feature Opportunities", styles['Heading2']))
            
            opportunities = serp_data['feature_opportunities'][:5]  # Top 5 opportunities
            
            # Create a table for opportunities
            data = [["Query", "Feature Type", "Current Position", "Opportunity Score"]]
            
            for opp in opportunities:
                query = opp['query'] if len(opp['query']) < 30 else opp['query'][:27] + "..."
                data.append([
                    query,
                    opp['feature'],
                    f"{opp['current_position']:.1f}",
                    f"{opp['opportunity_score']:.1f}"
                ])
            
            t = Table(data, colWidths=[180, 120, 100, 100])
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
        if 'recommendations' in serp_data and serp_data['recommendations']:
            elements.append(Paragraph("SERP Feature Recommendations", styles['Heading2']))
            
            for i, rec in enumerate(serp_data['recommendations'][:5]):  # Limit to top 5
                elements.append(Paragraph(f"{i+1}. {rec['title']}", styles['Heading3']))
                elements.append(Paragraph(rec['description'], styles['Normal']))
                elements.append(Paragraph(f"Impact: {rec['impact'].title()}", styles['Italic']))
                elements.append(Spacer(1, 10))
            
            elements.append(Spacer(1, 10))
    
    # Add SERP preview section if available
    if 'serp_previews' in analysis_results:
        elements.append(Paragraph("SERP Preview Analysis", styles['Heading1']))
        
        preview_data = analysis_results['serp_previews']
        
        # Add optimization tips
        if 'optimization_tips' in preview_data and preview_data['optimization_tips']:
            tips = preview_data['optimization_tips']
            
            elements.append(Paragraph("Meta Tag Optimization", styles['Heading2']))
            
            # Create a table for title issues
            if 'title_issues' in tips:
                title_issues = tips['title_issues']
                
                elements.append(Paragraph("Title Tag Issues", styles['Heading3']))
                
                data = [["Issue Type", "Count"]]
                data.append(["Missing Titles", str(title_issues.get('missing', 0))])
                data.append(["Too Short Titles", str(title_issues.get('too_short', 0))])
                data.append(["Too Long Titles", str(title_issues.get('too_long', 0))])
                data.append(["Optimal Titles", str(title_issues.get('optimal', 0))])
                
                t = Table(data, colWidths=[200, 100])
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
                elements.append(Spacer(1, 10))
            
            # Create a table for description issues
            if 'description_issues' in tips:
                desc_issues = tips['description_issues']
                
                elements.append(Paragraph("Meta Description Issues", styles['Heading3']))
                
                data = [["Issue Type", "Count"]]
                data.append(["Missing Descriptions", str(desc_issues.get('missing', 0))])
                data.append(["Too Short Descriptions", str(desc_issues.get('too_short', 0))])
                data.append(["Too Long Descriptions", str(desc_issues.get('too_long', 0))])
                data.append(["Optimal Descriptions", str(desc_issues.get('optimal', 0))])
                
                t = Table(data, colWidths=[200, 100])
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
        
        # Add sample SERP previews
        if 'previews' in preview_data and preview_data['previews']:
            elements.append(Paragraph("Sample SERP Previews", styles['Heading2']))
            
            # Show one best and one worst preview
            previews = preview_data['previews']
            best_preview = max(previews, key=lambda x: x['optimization_score'])
            worst_preview = min(previews, key=lambda x: x['optimization_score'])
            
            # Best preview
            elements.append(Paragraph("Best Optimized Preview", styles['Heading3']))
            
            # Create a styled "preview box"
            best_data = [[best_preview['displayed_title']]]
            best_data.append([best_preview['display_url']])
            best_data.append([best_preview['displayed_description']])
            
            best_table = Table(best_data, colWidths=[450])
            best_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), colors.lightblue),  # Title
                ('BACKGROUND', (0, 1), (0, 1), colors.lightgreen),  # URL
                ('BACKGROUND', (0, 2), (0, 2), colors.white),  # Description
                ('TEXTCOLOR', (0, 0), (0, 0), colors.blue),  # Title
                ('TEXTCOLOR', (0, 1), (0, 1), colors.green),  # URL
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('TOPPADDING', (0, 0), (0, 2), 6),
                ('BOTTOMPADDING', (0, 0), (0, 2), 6),
                ('LEFTPADDING', (0, 0), (0, 2), 10),
                ('RIGHTPADDING', (0, 0), (0, 2), 10),
                ('BOX', (0, 0), (0, 2), 1, colors.grey)
            ]))
            elements.append(best_table)
            elements.append(Paragraph(f"Score: {best_preview['optimization_score']}/100", styles['Italic']))
            elements.append(Spacer(1, 10))
            
            # Worst preview
            elements.append(Paragraph("Preview Needing Improvement", styles['Heading3']))
            
            # Create a styled "preview box"
            worst_data = [[worst_preview['displayed_title']]]
            worst_data.append([worst_preview['display_url']])
            worst_data.append([worst_preview['displayed_description']])
            
            worst_table = Table(worst_data, colWidths=[450])
            worst_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), colors.lightblue),  # Title
                ('BACKGROUND', (0, 1), (0, 1), colors.lightgreen),  # URL
                ('BACKGROUND', (0, 2), (0, 2), colors.white),  # Description
                ('TEXTCOLOR', (0, 0), (0, 0), colors.blue),  # Title
                ('TEXTCOLOR', (0, 1), (0, 1), colors.green),  # URL
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('TOPPADDING', (0, 0), (0, 2), 6),
                ('BOTTOMPADDING', (0, 0), (0, 2), 6),
                ('LEFTPADDING', (0, 0), (0, 2), 10),
                ('RIGHTPADDING', (0, 0), (0, 2), 10),
                ('BOX', (0, 0), (0, 2), 1, colors.grey)
            ]))
            elements.append(worst_table)
            elements.append(Paragraph(f"Score: {worst_preview['optimization_score']}/100", styles['Italic']))
            elements.append(Spacer(1, 20))
        
        # Add recommendations
        if 'recommendations' in preview_data and preview_data['recommendations']:
            elements.append(Paragraph("SERP Preview Recommendations", styles['Heading2']))
            
            for i, rec in enumerate(preview_data['recommendations'][:5]):  # Limit to top 5
                elements.append(Paragraph(f"{i+1}. {rec['title']}", styles['Heading3']))
                elements.append(Paragraph(rec['description'], styles['Normal']))
                elements.append(Paragraph(f"Impact: {rec['impact'].title()}", styles['Italic']))
                elements.append(Spacer(1, 10))
            
            elements.append(Spacer(1, 10))
    
    # Add backlink analysis section if available
    if 'backlink_analysis' in analysis_results:
        elements.append(Paragraph("Backlink Analysis", styles['Heading1']))
        
        backlink_data = analysis_results['backlink_analysis']
        
        # Add backlink summary
        if 'backlink_summary' in backlink_data:
            summary = backlink_data['backlink_summary']
            
            elements.append(Paragraph("Backlink Profile Summary", styles['Heading2']))
            
            # Create a table for backlink summary
            data = [["Metric", "Value"]]
            
            if 'total_backlinks' in summary:
                data.append(["Total Backlinks", f"{summary['total_backlinks']:,}"])
            
            if 'total_referring_domains' in summary:
                data.append(["Total Referring Domains", f"{summary['total_referring_domains']:,}"])
            
            if 'backlinks_per_domain' in summary:
                data.append(["Backlinks per Domain", f"{summary['backlinks_per_domain']:.1f}"])
            
            if 'domain_authority' in backlink_data:
                data.append(["Domain Authority", f"{backlink_data['domain_authority']:.1f}/100"])
            
            t = Table(data, colWidths=[200, 300])
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
        
        # Add anchor text distribution
        if 'anchor_text_analysis' in backlink_data and 'categories' in backlink_data['anchor_text_analysis']:
            anchor_categories = backlink_data['anchor_text_analysis']['categories']
            
            elements.append(Paragraph("Anchor Text Distribution", styles['Heading2']))
            
            # Create a table for anchor text categories
            data = [["Anchor Type", "Percentage"]]
            
            for category, percentage in anchor_categories.items():
                category_name = category.replace('_', ' ').title()
                data.append([category_name, f"{percentage:.1f}%"])
            
            t = Table(data, colWidths=[200, 100])
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
        if 'recommendations' in backlink_data and backlink_data['recommendations']:
            elements.append(Paragraph("Backlink Recommendations", styles['Heading2']))
            
            for i, rec in enumerate(backlink_data['recommendations'][:5]):  # Limit to top 5
                elements.append(Paragraph(f"{i+1}. {rec['title']}", styles['Heading3']))
                elements.append(Paragraph(rec['description'], styles['Normal']))
                elements.append(Paragraph(f"Impact: {rec['impact'].title()}", styles['Italic']))
                elements.append(Spacer(1, 10))
            
            elements.append(Spacer(1, 10))
    
    # Add content gap section if available
    if 'content_gaps' in analysis_results:
        elements.append(Paragraph("Content Gap Analysis", styles['Heading1']))
        
        content_gap_data = analysis_results['content_gaps']
        
        # Add gap summary
        if 'gap_summary' in content_gap_data:
            summary = content_gap_data['gap_summary']
            
            elements.append(Paragraph("Content Gap Summary", styles['Heading2']))
            
            # Create a table for gap summary
            data = [["Metric", "Value"]]
            
            if 'total_gaps' in summary:
                data.append(["Total Content Gaps", str(summary['total_gaps'])])
            
            if 'total_opportunities' in summary:
                data.append(["Content Opportunities", str(summary['total_opportunities'])])
            
            if 'competitor_advantage_topics' in summary:
                data.append(["Competitor Advantage Topics", str(summary['competitor_advantage_topics'])])
            
            t = Table(data, colWidths=[200, 300])
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
        
        # Add topic opportunities
        if 'topic_opportunities' in content_gap_data and content_gap_data['topic_opportunities']:
            elements.append(Paragraph("Content Opportunities", styles['Heading2']))
            
            opportunities = content_gap_data['topic_opportunities'][:5]  # Top 5 opportunities
            
            # Create a table for opportunities
            data = [["Topic", "Related Keywords", "Search Volume", "Opportunity Score"]]
            
            for opp in opportunities:
                topic = opp['topic'] if len(opp['topic']) < 30 else opp['topic'][:27] + "..."
                keywords = ", ".join(opp['related_keywords'][:3])
                if len(keywords) > 40:
                    keywords = keywords[:37] + "..."
                
                data.append([
                    topic,
                    keywords,
                    f"{opp['search_volume']:,}",
                    f"{opp['opportunity_score']:.1f}"
                ])
            
            t = Table(data, colWidths=[120, 180, 100, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Left-align first column
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),  # Left-align keywords column
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t)
            elements.append(Spacer(1, 20))
        
        # Add recommendations
        if 'content_recommendations' in content_gap_data and content_gap_data['content_recommendations']:
            elements.append(Paragraph("Content Gap Recommendations", styles['Heading2']))
            
            for i, rec in enumerate(content_gap_data['content_recommendations'][:5]):  # Limit to top 5
                elements.append(Paragraph(f"{i+1}. {rec['title']}", styles['Heading3']))
                elements.append(Paragraph(rec['description'], styles['Normal']))
                elements.append(Paragraph(f"Impact: {rec['impact'].title()}", styles['Italic']))
                elements.append(Spacer(1, 10))
            
            elements.append(Spacer(1, 10))
    
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
    5. Technical SEO improvements
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
