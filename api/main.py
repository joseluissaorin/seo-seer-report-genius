
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
import io
import json
import os
import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import tempfile
import time
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SEO Seer API", description="API for analyzing SEO data from Google Search Console")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temp directory for files
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Sample Gemini API call function (in a real scenario, you'd use the google-generativeai library)
def generate_seo_insights(data: pd.DataFrame, api_key: str) -> Dict[str, Any]:
    """
    Generate SEO insights using the Gemini API
    In a real implementation, this would call the Gemini API
    """
    # Example response structure
    insights = {
        "overview": "Your website has shown a 12% increase in organic traffic over the last month, with significant improvements in mobile search visibility. However, there are opportunities to improve page load times and enhance content relevance for key search terms.",
        "strengths": [
            "Strong performance for branded keywords",
            "Good click-through rate for product pages",
            "Increasing trend in impression share for target keywords"
        ],
        "weaknesses": [
            "High bounce rate on blog content",
            "Low average position for competitive keywords",
            "Limited mobile optimization for key landing pages"
        ],
        "opportunities": [
            "Create more in-depth content for keywords ranking on page 2",
            "Improve meta descriptions for pages with high impressions but low CTR",
            "Optimize page speed for mobile devices to reduce bounce rate",
            "Add structured data to enhance rich snippet opportunities",
            "Develop content clusters around top-performing topics"
        ],
        "keyword_recommendations": [
            {"keyword": "seo strategies 2023", "volume": "high", "competition": "medium", "opportunity_score": 8},
            {"keyword": "technical seo audit guide", "volume": "medium", "competition": "low", "opportunity_score": 9},
            {"keyword": "ecommerce seo tools", "volume": "high", "competition": "high", "opportunity_score": 7},
            {"keyword": "content optimization tips", "volume": "medium", "competition": "medium", "opportunity_score": 8},
            {"keyword": "mobile seo best practices", "volume": "medium", "competition": "low", "opportunity_score": 9}
        ],
        "content_suggestions": [
            "Ultimate Guide to Technical SEO in 2023",
            "How to Optimize E-commerce Product Pages for SEO",
            "Mobile SEO: Strategies for Improving Mobile Search Rankings",
            "Understanding Core Web Vitals and Their Impact on SEO",
            "Creating Topic Clusters to Boost Organic Traffic"
        ]
    }
    
    return insights

def analyze_gsc_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze Google Search Console data and extract key metrics"""
    
    results = {}
    
    # Overall metrics
    try:
        results["total_clicks"] = int(df["Clicks"].sum())
        results["total_impressions"] = int(df["Impressions"].sum())
        results["avg_ctr"] = float(df["CTR"].str.rstrip("%").astype(float).mean())
        results["avg_position"] = float(df["Position"].mean())
        
        # Top queries by clicks
        top_queries = df.sort_values("Clicks", ascending=False).head(10)
        results["top_queries"] = [
            {
                "query": row["Query"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": row["CTR"],
                "position": float(row["Position"])
            } 
            for _, row in top_queries.iterrows()
        ]
        
        # Queries with high impressions but low CTR (opportunities)
        opportunity_queries = df[
            (df["Impressions"] > df["Impressions"].quantile(0.7)) & 
            (df["CTR"].str.rstrip("%").astype(float) < df["CTR"].str.rstrip("%").astype(float).quantile(0.3))
        ].sort_values("Impressions", ascending=False).head(10)
        
        results["opportunity_queries"] = [
            {
                "query": row["Query"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": row["CTR"],
                "position": float(row["Position"])
            } 
            for _, row in opportunity_queries.iterrows()
        ]
        
        # High-ranking queries (position < 10) with good impressions
        high_ranking = df[
            (df["Position"] < 10) & 
            (df["Impressions"] > df["Impressions"].quantile(0.5))
        ].sort_values("Clicks", ascending=False).head(10)
        
        results["high_ranking_queries"] = [
            {
                "query": row["Query"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": row["CTR"],
                "position": float(row["Position"])
            } 
            for _, row in high_ranking.iterrows()
        ]
        
    except Exception as e:
        logger.error(f"Error analyzing GSC data: {e}")
        results["error"] = str(e)
    
    return results

def generate_visualizations(df: pd.DataFrame) -> Dict[str, str]:
    """Generate visualizations from the data and return file paths"""
    
    visualizations = {}
    
    try:
        # Create visualization directory
        vis_dir = os.path.join(TEMP_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Top Queries by Clicks
        plt.figure(figsize=(10, 6))
        top_queries = df.sort_values("Clicks", ascending=False).head(10)
        plt.barh(top_queries["Query"], top_queries["Clicks"], color="indigo")
        plt.xlabel("Clicks")
        plt.title("Top 10 Queries by Clicks")
        plt.tight_layout()
        top_queries_path = os.path.join(vis_dir, "top_queries.png")
        plt.savefig(top_queries_path)
        plt.close()
        visualizations["top_queries"] = top_queries_path
        
        # 2. CTR vs Position Scatter Plot
        plt.figure(figsize=(10, 6))
        df_sample = df.sample(min(100, len(df)))
        plt.scatter(
            df_sample["Position"], 
            df_sample["CTR"].str.rstrip("%").astype(float), 
            alpha=0.6, 
            color="purple"
        )
        plt.xlabel("Position")
        plt.ylabel("CTR (%)")
        plt.title("CTR vs Position Correlation")
        plt.grid(True, linestyle="--", alpha=0.7)
        ctr_position_path = os.path.join(vis_dir, "ctr_position.png")
        plt.savefig(ctr_position_path)
        plt.close()
        visualizations["ctr_position"] = ctr_position_path
        
        # 3. Clicks vs Impressions for Top Queries
        plt.figure(figsize=(10, 6))
        plt.scatter(
            top_queries["Impressions"], 
            top_queries["Clicks"], 
            alpha=0.7,
            s=100,
            color="blue"
        )
        for i, row in top_queries.iterrows():
            plt.annotate(
                row["Query"], 
                (row["Impressions"], row["Clicks"]),
                xytext=(7, 0),
                textcoords="offset points"
            )
        plt.xlabel("Impressions")
        plt.ylabel("Clicks")
        plt.title("Clicks vs Impressions for Top Queries")
        plt.grid(True, linestyle="--", alpha=0.7)
        clicks_impressions_path = os.path.join(vis_dir, "clicks_impressions.png")
        plt.savefig(clicks_impressions_path)
        plt.close()
        visualizations["clicks_impressions"] = clicks_impressions_path
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        visualizations["error"] = str(e)
    
    return visualizations

def generate_pdf_report(
    data_analysis: Dict[str, Any], 
    insights: Dict[str, Any],
    visualizations: Dict[str, str]
) -> str:
    """Generate a PDF report with the analysis results and insights"""
    
    try:
        # Create a temporary file for the PDF
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(TEMP_DIR, f"seo_report_{timestamp}.pdf")
        
        # Create the PDF
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontName='Helvetica-Bold',
            fontSize=24,
            spaceAfter=24,
            textColor=colors.purple,
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            spaceAfter=12,
            textColor=colors.purple,
        )
        
        subheading_style = ParagraphStyle(
            'Subheading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            spaceAfter=8,
            textColor=colors.darkblue,
        )
        
        # Title
        elements.append(Paragraph("SEO Performance Analysis Report", title_style))
        elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%B %d, %Y')}", styles["Normal"]))
        elements.append(Spacer(1, 24))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", heading_style))
        elements.append(Paragraph(insights["overview"], styles["Normal"]))
        elements.append(Spacer(1, 12))
        
        # Key Metrics
        elements.append(Paragraph("Key Metrics", heading_style))
        data = [
            ["Metric", "Value"],
            ["Total Clicks", f"{data_analysis['total_clicks']}"],
            ["Total Impressions", f"{data_analysis['total_impressions']}"],
            ["Average CTR", f"{data_analysis['avg_ctr']:.2f}%"],
            ["Average Position", f"{data_analysis['avg_position']:.2f}"],
        ]
        
        t = Table(data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lavender),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 24))
        
        # Add visualizations
        if "top_queries" in visualizations:
            elements.append(Paragraph("Top Performing Queries", heading_style))
            elements.append(Image(visualizations["top_queries"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        if "ctr_position" in visualizations:
            elements.append(Paragraph("CTR vs Position Correlation", heading_style))
            elements.append(Image(visualizations["ctr_position"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        if "clicks_impressions" in visualizations:
            elements.append(Paragraph("Clicks vs Impressions Analysis", heading_style))
            elements.append(Image(visualizations["clicks_impressions"], width=450, height=270))
            elements.append(Spacer(1, 24))
        
        # SWOT Analysis
        elements.append(Paragraph("SEO SWOT Analysis", heading_style))
        
        # Strengths
        elements.append(Paragraph("Strengths", subheading_style))
        for strength in insights["strengths"]:
            elements.append(Paragraph(f"• {strength}", styles["Normal"]))
        elements.append(Spacer(1, 12))
        
        # Weaknesses
        elements.append(Paragraph("Weaknesses", subheading_style))
        for weakness in insights["weaknesses"]:
            elements.append(Paragraph(f"• {weakness}", styles["Normal"]))
        elements.append(Spacer(1, 12))
        
        # Opportunities
        elements.append(Paragraph("Opportunities", subheading_style))
        for opportunity in insights["opportunities"]:
            elements.append(Paragraph(f"• {opportunity}", styles["Normal"]))
        elements.append(Spacer(1, 24))
        
        # Actionable Recommendations
        elements.append(Paragraph("Actionable Recommendations", heading_style))
        for i, opportunity in enumerate(insights["opportunities"], 1):
            elements.append(Paragraph(f"{i}. {opportunity}", styles["Normal"]))
            elements.append(Spacer(1, 6))
        elements.append(Spacer(1, 12))
        
        # Keyword Opportunities
        elements.append(Paragraph("Keyword Opportunities", heading_style))
        
        keyword_data = [["Keyword", "Search Volume", "Competition", "Opportunity Score"]]
        for kw in insights["keyword_recommendations"]:
            keyword_data.append([
                kw["keyword"], 
                kw["volume"], 
                kw["competition"],
                f"{kw['opportunity_score']}/10"
            ])
        
        t = Table(keyword_data, colWidths=[200, 100, 100, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 12))
        
        # Content Suggestions
        elements.append(Paragraph("Content Creation Opportunities", heading_style))
        for i, content in enumerate(insights["content_suggestions"], 1):
            elements.append(Paragraph(f"{i}. {content}", styles["Normal"]))
            elements.append(Spacer(1, 6))
        
        # Build the PDF
        doc.build(elements)
        
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        raise e

@app.post("/analyze-seo")
async def analyze_seo(
    file: UploadFile = File(...),
    api_key: str = Form(...)
):
    """
    Process Google Search Console export file and generate SEO report
    """
    try:
        # Validate the API key (in a real app, you'd verify this with Google)
        if not api_key or len(api_key) < 10:
            raise HTTPException(status_code=400, detail="Invalid API key")
        
        # Validate the file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV export from Google Search Console")
        
        # Read the CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Check if it's a valid GSC export
        required_columns = ["Query", "Clicks", "Impressions", "CTR", "Position"]
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file format. Must be a Google Search Console export with Query, Clicks, Impressions, CTR and Position columns."
            )
        
        # Perform data analysis
        analysis_results = analyze_gsc_data(df)
        
        # Generate visualizations
        visualization_paths = generate_visualizations(df)
        
        # Generate insights using Gemini API
        insights = generate_seo_insights(df, api_key)
        
        # Generate PDF report
        report_path = generate_pdf_report(analysis_results, insights, visualization_paths)
        
        # Return the report file
        return FileResponse(
            path=report_path, 
            filename="SEO_Analysis_Report.pdf", 
            media_type="application/pdf"
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "SEO Seer API is running. Use POST /analyze-seo to analyze your GSC data."}

# In a real implementation, you'd run this with:
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
