
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
from matplotlib.patches import Rectangle
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile
import time
import random
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import google.generativeai as genai
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import matplotlib.cm as cm

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

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

# Create temp directories for files
TEMP_DIR = "temp"
VISUALIZATIONS_DIR = os.path.join(TEMP_DIR, "visualizations")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

def setup_gemini_api(api_key: str):
    """Configure Gemini API with the provided API key"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Error setting up Gemini API: {e}")
        return False

def generate_seo_insights_with_gemini(data: pd.DataFrame, api_key: str) -> Dict[str, Any]:
    """
    Generate SEO insights using the Gemini API with actual data analysis
    """
    try:
        # Setup Gemini API
        if not setup_gemini_api(api_key):
            raise Exception("Failed to setup Gemini API")
        
        # Prepare data summary for Gemini
        # Getting top queries by clicks
        top_queries = data.sort_values("Clicks", ascending=False).head(20)
        top_queries_list = [
            {
                "query": row["Query"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": row["CTR"],
                "position": float(row["Position"])
            } 
            for _, row in top_queries.iterrows()
        ]
        
        # Queries with high impressions but low CTR
        opportunity_queries = data[
            (data["Impressions"] > data["Impressions"].quantile(0.7)) & 
            (data["CTR"].str.rstrip("%").astype(float) < data["CTR"].str.rstrip("%").astype(float).quantile(0.3))
        ].sort_values("Impressions", ascending=False).head(10)
        
        opportunity_queries_list = [
            {
                "query": row["Query"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": row["CTR"],
                "position": float(row["Position"])
            } 
            for _, row in opportunity_queries.iterrows()
        ]
        
        # Calculate averages and totals
        total_clicks = int(data["Clicks"].sum())
        total_impressions = int(data["Impressions"].sum())
        avg_ctr = float(data["CTR"].str.rstrip("%").astype(float).mean())
        avg_position = float(data["Position"].mean())
        
        # Prepare prompt for Gemini
        prompt = f"""
        You are an expert SEO consultant analyzing Google Search Console data. Please provide a comprehensive SEO analysis based on the following data:

        Overall Metrics:
        - Total Clicks: {total_clicks}
        - Total Impressions: {total_impressions}
        - Average CTR: {avg_ctr:.2f}%
        - Average Position: {avg_position:.2f}

        Top Performing Queries (by Clicks):
        {json.dumps(top_queries_list, indent=2)}

        Opportunity Queries (High Impressions, Low CTR):
        {json.dumps(opportunity_queries_list, indent=2)}

        Based on this data, please provide:
        1. An executive summary of the website's SEO performance
        2. Identified strengths in the current SEO strategy
        3. Weaknesses that need to be addressed
        4. Specific actionable opportunities for improvement
        5. Strategic keyword recommendations for new content with estimated search volume (high/medium/low), competition level, and opportunity score (1-10)
        6. At least 5 detailed content suggestions based on the data

        Format your response as a structured JSON object with the following keys: 
        "overview", "strengths", "weaknesses", "opportunities", "keyword_recommendations", "content_suggestions".

        For the keyword recommendations, include fields for "keyword", "volume", "competition", and "opportunity_score".
        Be specific, data-driven, and actionable in your recommendations.
        """
        
        # Generate model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Parse the response
        try:
            insights = json.loads(response.text)
            return insights
        except json.JSONDecodeError:
            # If the response is not valid JSON, try to extract JSON from the text
            text = response.text
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                try:
                    insights = json.loads(json_str)
                    return insights
                except:
                    pass
            
            # Fallback to structured response
            return {
                "overview": "Analysis of your SEO data shows opportunities for improvement in organic search visibility.",
                "strengths": [
                    "Some queries are performing well with good click-through rates",
                    "You have significant impression volume for several keywords",
                    "Your site appears in search results for diverse query types"
                ],
                "weaknesses": [
                    "Many high-impression queries have low CTR, indicating potential content or meta description issues",
                    f"The average position of {avg_position:.2f} suggests many keywords aren't ranking on page 1",
                    "There appears to be keyword cannibalization for some search terms"
                ],
                "opportunities": [
                    "Optimize meta titles and descriptions for high-impression, low-CTR queries",
                    "Create more comprehensive content for keywords ranking on page 2",
                    "Implement schema markup to enhance search appearance",
                    "Consolidate content for keywords with potential cannibalization",
                    "Focus on improving page load speed and mobile user experience"
                ],
                "keyword_recommendations": [
                    {"keyword": top_queries_list[0]["query"] if len(top_queries_list) > 0 else "product reviews", "volume": "high", "competition": "medium", "opportunity_score": 8},
                    {"keyword": f"{top_queries_list[1]['query']} guide" if len(top_queries_list) > 1 else "how to guides", "volume": "medium", "competition": "low", "opportunity_score": 9},
                    {"keyword": f"best {top_queries_list[2]['query']}" if len(top_queries_list) > 2 else "product comparisons", "volume": "high", "competition": "high", "opportunity_score": 7},
                    {"keyword": f"{opportunity_queries_list[0]['query']} tips" if len(opportunity_queries_list) > 0 else "industry tips", "volume": "medium", "competition": "medium", "opportunity_score": 8},
                    {"keyword": f"{opportunity_queries_list[1]['query']} solutions" if len(opportunity_queries_list) > 1 else "problem solutions", "volume": "medium", "competition": "low", "opportunity_score": 9}
                ],
                "content_suggestions": [
                    f"Ultimate Guide to {top_queries_list[0]['query'].title() if len(top_queries_list) > 0 else 'Your Industry'}",
                    f"How to Choose the Best {opportunity_queries_list[0]['query'].title() if len(opportunity_queries_list) > 0 else 'Products in Your Niche'}",
                    f"{top_queries_list[1]['query'].title() if len(top_queries_list) > 1 else 'Your Product'} vs Competitors: A Comprehensive Comparison",
                    f"10 Expert Tips to Improve Your {top_queries_list[2]['query'].title() if len(top_queries_list) > 2 else 'Results'}",
                    f"Solving Common Problems with {opportunity_queries_list[1]['query'].title() if len(opportunity_queries_list) > 1 else 'Your Product'}"
                ]
            }
    
    except Exception as e:
        logger.error(f"Error generating insights with Gemini: {e}")
        # Fallback response
        return {
            "overview": "We were unable to generate AI-powered insights at this time. However, our analysis has identified several key opportunities in your SEO data.",
            "strengths": [
                "Multiple queries driving traffic to your site",
                "Established presence in search results for relevant terms",
                "Some keywords showing strong click-through performance"
            ],
            "weaknesses": [
                "Several high-impression keywords with low engagement",
                "Average position suggests many terms aren't on page 1",
                "Potential missed opportunities in related search terms"
            ],
            "opportunities": [
                "Optimize meta descriptions for higher CTR",
                "Create more targeted content for keywords on page 2",
                "Improve internal linking structure",
                "Consider technical SEO improvements for better indexing",
                "Build more comprehensive content around top-performing topics"
            ],
            "keyword_recommendations": [
                {"keyword": data["Query"].iloc[0] if not data.empty else "product reviews", "volume": "medium", "competition": "medium", "opportunity_score": 7},
                {"keyword": data["Query"].iloc[1] if len(data) > 1 else "how to guides", "volume": "medium", "competition": "low", "opportunity_score": 8},
                {"keyword": data["Query"].iloc[2] if len(data) > 2 else "product comparisons", "volume": "high", "competition": "medium", "opportunity_score": 6},
                {"keyword": data["Query"].iloc[3] if len(data) > 3 else "industry tips", "volume": "low", "competition": "low", "opportunity_score": 9},
                {"keyword": data["Query"].iloc[4] if len(data) > 4 else "problem solutions", "volume": "medium", "competition": "medium", "opportunity_score": 7}
            ],
            "content_suggestions": [
                f"Complete Guide to {data['Query'].iloc[0].title() if not data.empty else 'Your Main Topic'}",
                f"How to Maximize Results with {data['Query'].iloc[1].title() if len(data) > 1 else 'Your Product'}",
                f"Common Mistakes to Avoid When Using {data['Query'].iloc[2].title() if len(data) > 2 else 'Your Service'}",
                f"Expert Tips for {data['Query'].iloc[3].title() if len(data) > 3 else 'Your Industry'}",
                f"Troubleshooting Guide for {data['Query'].iloc[4].title() if len(data) > 4 else 'Your Product'}"
            ]
        }

def analyze_gsc_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze Google Search Console data and extract key metrics"""
    
    results = {}
    
    try:
        # Clean the CTR column
        df["CTR_numeric"] = df["CTR"].str.rstrip("%").astype(float)
        
        # Overall metrics
        results["total_clicks"] = int(df["Clicks"].sum())
        results["total_impressions"] = int(df["Impressions"].sum())
        results["avg_ctr"] = float(df["CTR_numeric"].mean())
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
            (df["CTR_numeric"] < df["CTR_numeric"].quantile(0.3))
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
        
        # Performance by position
        position_bins = [1, 3, 5, 10, 20, 50, 100]
        df['position_group'] = pd.cut(df['Position'], position_bins, labels=[f'{position_bins[i]}-{position_bins[i+1]-1}' for i in range(len(position_bins)-1)])
        position_performance = df.groupby('position_group').agg({
            'Clicks': 'sum',
            'Impressions': 'sum',
            'CTR_numeric': 'mean',
            'Query': 'count'
        }).reset_index()
        position_performance['CTR'] = position_performance['CTR_numeric'].apply(lambda x: f'{x:.2f}%')
        
        results["position_performance"] = [
            {
                "position_range": row["position_group"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": row["CTR"],
                "query_count": int(row["Query"])
            } 
            for _, row in position_performance.iterrows()
        ]
        
        # Advanced CTR Analysis
        ctr_threshold = df["CTR_numeric"].quantile(0.75)
        high_ctr_terms = df[df["CTR_numeric"] > ctr_threshold].sort_values("Impressions", ascending=False).head(10)
        
        results["high_ctr_terms"] = [
            {
                "query": row["Query"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": row["CTR"],
                "position": float(row["Position"])
            } 
            for _, row in high_ctr_terms.iterrows()
        ]
        
        # Keyword clustering to identify topics
        # Prepare data
        if len(df) >= 5:  # Need at least a few rows for clustering
            # Simplified clustering
            kmeans = KMeans(n_clusters=min(5, len(df)), random_state=42)
            
            # Use position and CTR for clustering
            X = df[['Position', 'CTR_numeric']].copy()
            X = (X - X.mean()) / X.std()  # Simple normalization
            
            df['cluster'] = kmeans.fit_predict(X)
            
            # Get cluster insights
            cluster_data = df.groupby('cluster').agg({
                'Query': lambda x: list(x)[:3],  # Sample of 3 queries per cluster
                'Clicks': 'sum',
                'Impressions': 'sum',
                'CTR_numeric': 'mean',
                'Position': 'mean'
            }).reset_index()
            
            results["keyword_clusters"] = [
                {
                    "cluster_id": int(row["cluster"]),
                    "sample_queries": row["Query"],
                    "total_clicks": int(row["Clicks"]),
                    "total_impressions": int(row["Impressions"]),
                    "avg_ctr": f"{row['CTR_numeric']:.2f}%",
                    "avg_position": float(row["Position"])
                }
                for _, row in cluster_data.iterrows()
            ]
        
        # Extract query word frequency for content gap analysis
        words = []
        for query in df["Query"]:
            words.extend(word_tokenize(query.lower()))
        
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        word_freq = Counter(filtered_words)
        common_words = word_freq.most_common(20)
        
        results["common_query_terms"] = [
            {"term": word, "frequency": freq}
            for word, freq in common_words
        ]
        
        # Performance trend by search position (normalized)
        position_impact = df.copy()
        positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ctr_by_position = []
        
        for pos in positions:
            pos_data = position_impact[(position_impact["Position"] >= pos) & (position_impact["Position"] < pos+1)]
            if not pos_data.empty:
                avg_ctr = pos_data["CTR_numeric"].mean()
                ctr_by_position.append({"position": pos, "avg_ctr": avg_ctr})
            else:
                ctr_by_position.append({"position": pos, "avg_ctr": None})
        
        results["ctr_by_position"] = ctr_by_position
        
        # Long tail vs. short tail performance
        df['query_length'] = df['Query'].apply(lambda x: len(x.split()))
        df['query_type'] = df['query_length'].apply(lambda x: 'Short (1-2 words)' if x <= 2 else 'Medium (3-4 words)' if x <= 4 else 'Long (5+ words)')
        
        query_type_perf = df.groupby('query_type').agg({
            'Clicks': 'sum',
            'Impressions': 'sum',
            'CTR_numeric': 'mean',
            'Position': 'mean',
            'Query': 'count'
        }).reset_index()
        
        results["query_length_performance"] = [
            {
                "query_type": row["query_type"],
                "clicks": int(row["Clicks"]),
                "impressions": int(row["Impressions"]),
                "avg_ctr": f"{row['CTR_numeric']:.2f}%",
                "avg_position": float(row["Position"]),
                "query_count": int(row["Query"])
            }
            for _, row in query_type_perf.iterrows()
        ]
        
    except Exception as e:
        logger.error(f"Error analyzing GSC data: {e}")
        results["error"] = str(e)
    
    return results

def generate_visualizations(df: pd.DataFrame) -> Dict[str, str]:
    """Generate visualizations from the data and return file paths"""
    
    visualizations = {}
    
    try:
        # Set the style for all plots
        plt.style.use('ggplot')
        sns.set_palette("deep")
        
        # Clean the CTR column
        df["CTR_numeric"] = df["CTR"].str.rstrip("%").astype(float)
        
        # 1. Top Queries by Clicks
        plt.figure(figsize=(10, 6))
        top_queries = df.sort_values("Clicks", ascending=False).head(10)
        plt.barh(top_queries["Query"], top_queries["Clicks"], color="#4B0082")
        plt.xlabel("Clicks")
        plt.ylabel("")
        plt.title("Top 10 Queries by Clicks", fontsize=16, fontweight='bold')
        plt.tight_layout()
        top_queries_path = os.path.join(VISUALIZATIONS_DIR, "top_queries.png")
        plt.savefig(top_queries_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualizations["top_queries"] = top_queries_path
        
        # 2. CTR vs Position Scatter Plot with trendline
        plt.figure(figsize=(10, 6))
        
        # Only take a sample if dataset is very large
        if len(df) > 200:
            df_sample = df.sample(200, random_state=42)
        else:
            df_sample = df.copy()
            
        # Create scatter plot
        sns.regplot(
            x="Position", 
            y="CTR_numeric", 
            data=df_sample,
            scatter_kws={"alpha": 0.6, "color": "purple", "s": 30},
            line_kws={"color": "blue", "lw": 2}
        )
        
        plt.xlabel("Position", fontsize=12)
        plt.ylabel("CTR (%)", fontsize=12)
        plt.title("CTR vs Position Correlation", fontsize=16, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.7)
        ctr_position_path = os.path.join(VISUALIZATIONS_DIR, "ctr_position.png")
        plt.savefig(ctr_position_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualizations["ctr_position"] = ctr_position_path
        
        # 3. Clicks vs Impressions Bubble Chart
        plt.figure(figsize=(10, 6))
        # Calculate bubble sizes based on CTR
        sizes = df_sample["CTR_numeric"] * 10
        
        # Create a colormap based on position (lower position = better rank = greener)
        norm = plt.Normalize(df_sample["Position"].max(), df_sample["Position"].min())
        colors = cm.viridis(norm(df_sample["Position"]))
        
        sc = plt.scatter(
            df_sample["Impressions"], 
            df_sample["Clicks"], 
            s=sizes,
            c=df_sample["Position"],
            cmap='viridis_r',
            alpha=0.7
        )
        
        # Add colorbar to show position scale
        cbar = plt.colorbar(sc)
        cbar.set_label('Position (Lower is Better)', fontsize=10)
        
        plt.xlabel("Impressions", fontsize=12)
        plt.ylabel("Clicks", fontsize=12)
        plt.title("Clicks vs Impressions Relationship", fontsize=16, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.7)
        clicks_impressions_path = os.path.join(VISUALIZATIONS_DIR, "clicks_impressions.png")
        plt.savefig(clicks_impressions_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualizations["clicks_impressions"] = clicks_impressions_path
        
        # 4. CTR Distribution by Position Range
        plt.figure(figsize=(10, 6))
        position_bins = [1, 3, 5, 10, 20, 50, 100]
        df['position_group'] = pd.cut(df['Position'], position_bins, 
                                      labels=[f'{position_bins[i]}-{position_bins[i+1]-1}' 
                                              for i in range(len(position_bins)-1)])
        
        position_perf = df.groupby('position_group').agg({'CTR_numeric': 'mean'}).reset_index()
        
        # Sort by position for better visualization
        position_sort_order = [f'{position_bins[i]}-{position_bins[i+1]-1}' for i in range(len(position_bins)-1)]
        position_perf['position_group'] = pd.Categorical(position_perf['position_group'], categories=position_sort_order)
        position_perf = position_perf.sort_values('position_group')
        
        bars = plt.bar(position_perf['position_group'], position_perf['CTR_numeric'], color='purple')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel("Position Range", fontsize=12)
        plt.ylabel("Average CTR (%)", fontsize=12)
        plt.title("CTR Performance by Position Range", fontsize=16, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        ctr_by_position_path = os.path.join(VISUALIZATIONS_DIR, "ctr_by_position.png")
        plt.savefig(ctr_by_position_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualizations["ctr_by_position"] = ctr_by_position_path
        
        # 5. Query Length Analysis
        plt.figure(figsize=(10, 6))
        df['query_length'] = df['Query'].apply(lambda x: len(x.split()))
        df['query_type'] = df['query_length'].apply(
            lambda x: 'Short (1-2 words)' if x <= 2 else 'Medium (3-4 words)' if x <= 4 else 'Long (5+ words)'
        )
        
        query_type_perf = df.groupby('query_type').agg({
            'Clicks': 'sum',
            'Impressions': 'sum',
            'CTR_numeric': 'mean',
        }).reset_index()
        
        # Create custom sort order
        query_type_order = ['Short (1-2 words)', 'Medium (3-4 words)', 'Long (5+ words)']
        query_type_perf['query_type'] = pd.Categorical(query_type_perf['query_type'], categories=query_type_order)
        query_type_perf = query_type_perf.sort_values('query_type')
        
        # Set up the figure with multiple subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot clicks by query type
        sns.barplot(x='query_type', y='Clicks', data=query_type_perf, ax=axes[0], palette='viridis')
        axes[0].set_title('Clicks by Query Length')
        axes[0].set_xlabel('')
        
        # Plot impressions by query type
        sns.barplot(x='query_type', y='Impressions', data=query_type_perf, ax=axes[1], palette='viridis')
        axes[1].set_title('Impressions by Query Length')
        axes[1].set_xlabel('')
        
        # Plot CTR by query type
        sns.barplot(x='query_type', y='CTR_numeric', data=query_type_perf, ax=axes[2], palette='viridis')
        axes[2].set_title('CTR (%) by Query Length')
        axes[2].set_xlabel('')
        
        plt.tight_layout()
        query_length_path = os.path.join(VISUALIZATIONS_DIR, "query_length_analysis.png")
        plt.savefig(query_length_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualizations["query_length_analysis"] = query_length_path
        
        # 6. Word Cloud of Search Terms
        plt.figure(figsize=(10, 6))
        text = ' '.join(df['Query'])
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            colormap='viridis', 
            max_words=100,
            contour_width=1,
            contour_color='steelblue'
        ).generate(text)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        
        wordcloud_path = os.path.join(VISUALIZATIONS_DIR, "search_terms_wordcloud.png")
        plt.savefig(wordcloud_path, dpi=150, bbox_inches='tight')
        plt.close()
        visualizations["search_terms_wordcloud"] = wordcloud_path
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        visualizations["error"] = str(e)
    
    return visualizations

def generate_pdf_report(
    data_analysis: Dict[str, Any], 
    insights: Dict[str, Any],
    visualizations: Dict[str, str]
) -> str:
    """Generate a comprehensive PDF report with the analysis results and insights"""
    
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
        
        section_title_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=12,
            spaceAfter=6,
            textColor=colors.black,
        )
        
        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=10,
        )
        
        # Title Page
        elements.append(Paragraph("SEO Performance Analysis Report", title_style))
        elements.append(Paragraph(f"Comprehensive Search Engine Optimization Analysis", styles["Italic"]))
        elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%B %d, %Y')}", styles["Normal"]))
        elements.append(Spacer(1, 24))
        
        # Table of Contents
        elements.append(Paragraph("Table of Contents", heading_style))
        toc_items = [
            "1. Executive Summary",
            "2. Key Performance Metrics",
            "3. Search Visibility Analysis",
            "4. Click-Through Rate Analysis",
            "5. Keyword Performance Deep Dive",
            "6. Search Query Analysis",
            "7. SEO SWOT Analysis",
            "8. Actionable Recommendations",
            "9. Content Strategy Recommendations",
            "10. Keyword Opportunities"
        ]
        for item in toc_items:
            elements.append(Paragraph(f"• {item}", body_style))
        
        elements.append(PageBreak())
        
        # Executive Summary
        elements.append(Paragraph("1. Executive Summary", heading_style))
        elements.append(Paragraph(insights["overview"], body_style))
        elements.append(Spacer(1, 12))
        
        # Key Metrics
        elements.append(Paragraph("2. Key Performance Metrics", heading_style))
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
        
        # Search Visibility Analysis
        elements.append(Paragraph("3. Search Visibility Analysis", heading_style))
        elements.append(Paragraph("The following visualizations show the distribution of your search performance and visibility:", body_style))
        
        if "top_queries" in visualizations:
            elements.append(Paragraph("Top Performing Queries", subheading_style))
            elements.append(Paragraph("These are your top search queries by clicks. These terms drive the most traffic to your website.", body_style))
            elements.append(Image(visualizations["top_queries"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        if "clicks_impressions" in visualizations:
            elements.append(Paragraph("Clicks vs Impressions Analysis", subheading_style))
            elements.append(Paragraph("This bubble chart shows the relationship between impressions and clicks. Bubble size indicates CTR, and color indicates position (greener = better position).", body_style))
            elements.append(Image(visualizations["clicks_impressions"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        if "search_terms_wordcloud" in visualizations:
            elements.append(Paragraph("Search Query Word Cloud", subheading_style))
            elements.append(Paragraph("This visualization shows the most common words in your search queries, with size indicating frequency.", body_style))
            elements.append(Image(visualizations["search_terms_wordcloud"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        elements.append(PageBreak())
        
        # Click-Through Rate Analysis
        elements.append(Paragraph("4. Click-Through Rate Analysis", heading_style))
        
        if "ctr_position" in visualizations:
            elements.append(Paragraph("CTR vs Position Correlation", subheading_style))
            elements.append(Paragraph("This scatter plot with regression line shows how your click-through rate varies by position. The downward trend indicates that as position numbers increase (lower ranking), CTR typically decreases.", body_style))
            elements.append(Image(visualizations["ctr_position"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        if "ctr_by_position" in visualizations:
            elements.append(Paragraph("CTR by Position Range", subheading_style))
            elements.append(Paragraph("This chart shows average CTR for different position ranges, helping you understand the performance drop-off as rankings decrease.", body_style))
            elements.append(Image(visualizations["ctr_by_position"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        # Position Performance Table
        elements.append(Paragraph("Position Performance Breakdown", subheading_style))
        if "position_performance" in data_analysis:
            position_data = [["Position Range", "Clicks", "Impressions", "CTR", "Query Count"]]
            for item in data_analysis["position_performance"]:
                position_data.append([
                    item["position_range"],
                    f"{item['clicks']}",
                    f"{item['impressions']}",
                    item["ctr"],
                    f"{item['query_count']}"
                ])
            
            t = Table(position_data, colWidths=[100, 80, 100, 80, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(t)
        else:
            elements.append(Paragraph("Position performance data not available.", body_style))
        
        elements.append(Spacer(1, 12))
        elements.append(PageBreak())
        
        # Keyword Performance Deep Dive
        elements.append(Paragraph("5. Keyword Performance Deep Dive", heading_style))
        
        # High CTR Terms
        elements.append(Paragraph("High-Converting Search Terms", subheading_style))
        elements.append(Paragraph("These search terms have an above-average CTR, indicating strong alignment with user intent:", body_style))
        
        if "high_ctr_terms" in data_analysis:
            high_ctr_data = [["Query", "Position", "CTR", "Clicks", "Impressions"]]
            for item in data_analysis["high_ctr_terms"]:
                high_ctr_data.append([
                    item["query"],
                    f"{item['position']:.1f}",
                    item["ctr"],
                    f"{item['clicks']}",
                    f"{item['impressions']}"
                ])
            
            t = Table(high_ctr_data, colWidths=[180, 60, 60, 60, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(t)
        else:
            elements.append(Paragraph("High CTR terms data not available.", body_style))
        
        elements.append(Spacer(1, 12))
        
        # Opportunity Queries
        elements.append(Paragraph("Opportunity Queries", subheading_style))
        elements.append(Paragraph("These queries have high impressions but low CTR, representing opportunities for optimization:", body_style))
        
        if "opportunity_queries" in data_analysis:
            opportunity_data = [["Query", "Position", "CTR", "Clicks", "Impressions"]]
            for item in data_analysis["opportunity_queries"]:
                opportunity_data.append([
                    item["query"],
                    f"{item['position']:.1f}",
                    item["ctr"],
                    f"{item['clicks']}",
                    f"{item['impressions']}"
                ])
            
            t = Table(opportunity_data, colWidths=[180, 60, 60, 60, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(t)
        else:
            elements.append(Paragraph("Opportunity queries data not available.", body_style))
        
        elements.append(Spacer(1, 24))
        elements.append(PageBreak())
        
        # Search Query Analysis
        elements.append(Paragraph("6. Search Query Analysis", heading_style))
        
        if "query_length_analysis" in visualizations:
            elements.append(Paragraph("Query Length Performance", subheading_style))
            elements.append(Paragraph("This analysis shows how short, medium, and long-tail keywords perform in terms of clicks, impressions, and CTR:", body_style))
            elements.append(Image(visualizations["query_length_analysis"], width=500, height=200))
            elements.append(Spacer(1, 12))
        
        # Query Length Performance Details
        if "query_length_performance" in data_analysis:
            elements.append(Paragraph("Query Length Performance Breakdown", subheading_style))
            query_length_data = [["Query Type", "Avg. Position", "Avg. CTR", "Clicks", "Impressions", "Count"]]
            for item in data_analysis["query_length_performance"]:
                query_length_data.append([
                    item["query_type"],
                    f"{item['avg_position']:.1f}",
                    item["avg_ctr"],
                    f"{item['clicks']}",
                    f"{item['impressions']}",
                    f"{item['query_count']}"
                ])
            
            t = Table(query_length_data, colWidths=[100, 80, 70, 70, 100, 60])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(t)
            elements.append(Spacer(1, 12))
        
        # Common Query Terms
        if "common_query_terms" in data_analysis:
            elements.append(Paragraph("Frequently Used Terms in Search Queries", subheading_style))
            elements.append(Paragraph("These are the most common words found in the search queries driving traffic to your site:", body_style))
            
            # Create a table of common terms and frequencies
            terms_data = [["Term", "Frequency"]]
            for item in data_analysis["common_query_terms"]:
                terms_data.append([item["term"], f"{item['frequency']}"]) 
            
            # Create a multi-column layout for terms
            num_terms = len(data_analysis["common_query_terms"])
            col1 = terms_data[:min(11, num_terms+1)]  # +1 for header
            col2 = terms_data[0:1] + terms_data[11:min(21, num_terms+1)] if num_terms > 10 else []
            
            if col2:
                # Two-column layout
                table_data = []
                for i in range(max(len(col1), len(col2))):
                    row = []
                    row.extend(col1[i] if i < len(col1) else ["", ""])
                    row.extend(col2[i] if i < len(col2) else ["", ""])
                    table_data.append(row)
                
                t = Table(table_data, colWidths=[100, 60, 100, 60])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.lavender),
                    ('BACKGROUND', (2, 0), (3, 0), colors.lavender),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                    ('TEXTCOLOR', (2, 0), (3, 0), colors.black),
                    ('ALIGN', (0, 0), (3, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (1, -1), 1, colors.black),
                    ('GRID', (2, 0), (3, -1), 1, colors.black),
                ]))
            else:
                # Single-column layout
                t = Table(col1, colWidths=[100, 60])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.lavender),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                    ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (1, -1), 1, colors.black),
                ]))
            
            elements.append(t)
        
        elements.append(Spacer(1, 24))
        elements.append(PageBreak())
        
        # SWOT Analysis
        elements.append(Paragraph("7. SEO SWOT Analysis", heading_style))
        
        # Strengths
        elements.append(Paragraph("Strengths", subheading_style))
        for strength in insights["strengths"]:
            elements.append(Paragraph(f"• {strength}", body_style))
        elements.append(Spacer(1, 12))
        
        # Weaknesses
        elements.append(Paragraph("Weaknesses", subheading_style))
        for weakness in insights["weaknesses"]:
            elements.append(Paragraph(f"• {weakness}", body_style))
        elements.append(Spacer(1, 12))
        
        # Opportunities
        elements.append(Paragraph("Opportunities", subheading_style))
        for opportunity in insights["opportunities"]:
            elements.append(Paragraph(f"• {opportunity}", body_style))
        elements.append(Spacer(1, 24))
        
        # Actionable Recommendations
        elements.append(Paragraph("8. Actionable Recommendations", heading_style))
        elements.append(Paragraph("Based on our analysis, we recommend the following specific actions to improve your SEO performance:", body_style))
        
        for i, opportunity in enumerate(insights["opportunities"], 1):
            elements.append(Paragraph(f"{i}. {opportunity}", body_style))
        
        elements.append(Spacer(1, 12))
        elements.append(PageBreak())
        
        # Content Strategy Recommendations
        elements.append(Paragraph("9. Content Strategy Recommendations", heading_style))
        elements.append(Paragraph("Based on your current performance and search data, we recommend creating the following content:", body_style))
        
        for i, content in enumerate(insights["content_suggestions"], 1):
            elements.append(Paragraph(f"{i}. {content}", body_style))
            elements.append(Spacer(1, 6))
        elements.append(Spacer(1, 12))
        
        # Keyword Opportunities
        elements.append(Paragraph("10. Keyword Opportunities", heading_style))
        elements.append(Paragraph("These keywords represent strategic opportunities for your SEO growth:", body_style))
        
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
        # Validate the API key
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
        
        # Clean the data - remove rows with NaN values
        df = df.dropna(subset=["Query", "Clicks", "Impressions", "CTR", "Position"])
        
        # Perform data analysis
        analysis_results = analyze_gsc_data(df)
        
        # Generate visualizations
        visualization_paths = generate_visualizations(df)
        
        # Generate insights using Gemini API
        insights = generate_seo_insights_with_gemini(df, api_key)
        
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

