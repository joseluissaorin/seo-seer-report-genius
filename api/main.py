
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import io
import json
import os
import datetime
import re
import time
import logging
import random
import tempfile
import traceback

# Data Analysis & Machine Learning
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
import umap
import hdbscan
from scipy import stats
from scipy.spatial.distance import cdist, cosine
from scipy.special import softmax
import networkx as nx
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pytrends.request import TrendReq

# Text Processing
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams
from textblob import TextBlob
import spacy
import re

# Web Scraping & API
import requests
from bs4 import BeautifulSoup

# Visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.cm as cm
from adjustText import adjust_text
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from collections import Counter

# AI
import google.generativeai as genai

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Try to load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If the model isn't downloaded yet, download it
    os.system("python -m spacy download en_core_web_sm")
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        # Fallback if still not available
        nlp = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SEO Seer API", description="API for advanced SEO data analysis from Google Search Console")

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
PLOTLY_DIR = os.path.join(TEMP_DIR, "plotly")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(PLOTLY_DIR, exist_ok=True)

# Define custom color palettes
PURPLE_PALETTE = ["#F6EFFF", "#E5DEFF", "#D4CCFF", "#C3BBFF", "#B2AAFF", "#A199FF", "#9088FF", "#7F77FF", "#6E66FF", "#5D55FF"]
SEO_COLORS = {
    "primary": "#6E59A5",
    "secondary": "#9b87f5",
    "tertiary": "#D6BCFA",
    "accent1": "#F2FCE2",
    "accent2": "#FEF7CD",
    "accent3": "#FDE1D3",
    "text": "#1A1F2C",
    "background": "#FFFFFF",
    "success": "#0EA5E9",
    "warning": "#F97316",
    "danger": "#D946EF",
    "neutral": "#8E9196"
}

def setup_gemini_api(api_key: str) -> bool:
    """Configure Gemini API with the provided API key"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Error setting up Gemini API: {e}")
        return False

class SEOAnalyzer:
    """Advanced SEO Analysis Engine"""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with Google Search Console data"""
        self.data = data
        self.clean_data()
        self.domain = self._extract_domain()
        self.keyword_clusters = None
        self.competitors = None
        self.keyword_network = None
        self.seasonal_trends = None
        self.topic_model = None
        self.entity_analysis = None
        self.metrics = {}
        self.visualizations = {}
        
    def clean_data(self):
        """Clean and prepare data for analysis"""
        # Convert CTR to numeric
        self.data["CTR_numeric"] = self.data["CTR"].str.rstrip("%").astype(float)
        
        # Add derived features
        self.data["Efficiency"] = self.data["Clicks"] / self.data["Impressions"].clip(lower=1)
        self.data["ExpectedCTR"] = self._calculate_expected_ctr(self.data["Position"])
        self.data["CTRDelta"] = self.data["CTR_numeric"] - self.data["ExpectedCTR"]
        self.data["ClickPotential"] = (self.data["ExpectedCTR"] - self.data["CTR_numeric"]).clip(lower=0) * self.data["Impressions"] / 100
        
        # Extract query features
        self.data["QueryLength"] = self.data["Query"].apply(lambda x: len(x.split()))
        self.data["QueryChars"] = self.data["Query"].apply(len)
        self.data["HasNumbers"] = self.data["Query"].apply(lambda x: bool(re.search(r'\d', x)))
        self.data["HasSpecialChars"] = self.data["Query"].apply(lambda x: bool(re.search(r'[^\w\s]', x)))
        
        # Query intent classification
        self.data["Intent"] = self.data["Query"].apply(self._classify_intent)
        
        # Query difficulty estimation (mathematical model based on length and complexity)
        char_weight = 0.01
        word_weight = 0.1
        special_char_weight = 0.2
        numeric_weight = 0.15
        
        self.data["QueryComplexity"] = (
            self.data["QueryChars"] * char_weight +
            self.data["QueryLength"] * word_weight +
            self.data["HasSpecialChars"].astype(int) * special_char_weight +
            self.data["HasNumbers"].astype(int) * numeric_weight
        )
        
        # Z-Score for anomaly detection
        self.data["CTR_Z"] = stats.zscore(self.data["CTR_numeric"])
        self.data["Position_Z"] = stats.zscore(self.data["Position"])
        
        # Query type categorization
        self.data["QueryType"] = self.data["QueryLength"].apply(
            lambda x: "Short (1-2 words)" if x <= 2 else "Medium (3-4 words)" if x <= 4 else "Long (5+ words)"
        )
        
    def _extract_domain(self) -> str:
        """Attempt to extract domain from data"""
        # This is a placeholder - in reality we would need the actual domain
        # For now, use a fictional domain for demonstration
        return "example.com"
    
    def _calculate_expected_ctr(self, positions: pd.Series) -> pd.Series:
        """
        Calculate expected CTR based on position using a logarithmic decay model
        Formula: CTR = a * ln(b/position) where a and b are parameters
        """
        a = 20  # Scale parameter
        b = 10  # Position parameter
        return a * np.log(b / positions.clip(lower=0.1))
    
    def _classify_intent(self, query: str) -> str:
        """Classify search intent of a query"""
        query = query.lower()
        
        # Informational queries
        if any(word in query for word in ["what", "how", "why", "when", "where", "who", "which", "guide", "tutorial", "learn"]):
            return "Informational"
            
        # Navigational queries
        if any(word in query for word in ["login", "website", "official", "homepage", "sign in", "account"]):
            return "Navigational"
            
        # Transactional queries
        if any(word in query for word in ["buy", "price", "cheap", "discount", "deal", "purchase", "shop", "cost", "order"]):
            return "Transactional"
            
        # Commercial investigation
        if any(word in query for word in ["best", "top", "review", "compare", "vs", "versus", "comparison"]):
            return "Commercial"
            
        # Default to informational
        return "Informational"
    
    def calculate_core_metrics(self) -> Dict[str, Any]:
        """Calculate key SEO performance metrics"""
        df = self.data
        
        metrics = {
            "total_clicks": int(df["Clicks"].sum()),
            "total_impressions": int(df["Impressions"].sum()),
            "avg_ctr": float(df["CTR_numeric"].mean()),
            "avg_position": float(df["Position"].mean()),
            "median_position": float(df["Position"].median()),
            "weighted_avg_position": float((df["Position"] * df["Impressions"]).sum() / df["Impressions"].sum()),
            "queries_count": len(df),
            "page_1_queries": int(df[df["Position"] <= 10]["Query"].count()),
            "page_1_clicks": int(df[df["Position"] <= 10]["Clicks"].sum()),
            "page_1_impressions": int(df[df["Position"] <= 10]["Impressions"].sum()),
            "click_potential": float(df["ClickPotential"].sum()),
            "max_position": float(df["Position"].max()),
            "min_position": float(df["Position"].min()),
            "visibility_index": float((11 - df["Position"].clip(upper=10)) * df["Impressions"]).sum() / (10 * df["Impressions"].sum()),
            "ctr_anomalies": int(df[abs(df["CTR_Z"]) > 2].shape[0]),
            "position_anomalies": int(df[abs(df["Position_Z"]) > 2].shape[0]),
        }
        
        # Calculate statistical measures for CTR and position
        metrics["position_std"] = float(df["Position"].std())
        metrics["position_variance"] = float(df["Position"].var())
        metrics["ctr_std"] = float(df["CTR_numeric"].std())
        metrics["ctr_variance"] = float(df["CTR_numeric"].var())
        
        # Calculate skewness and kurtosis for position distribution
        metrics["position_skewness"] = float(stats.skew(df["Position"]))
        metrics["position_kurtosis"] = float(stats.kurtosis(df["Position"]))
        
        # Calculate correlation between metrics
        metrics["corr_pos_ctr"] = float(df["Position"].corr(df["CTR_numeric"]))
        metrics["corr_pos_clicks"] = float(df["Position"].corr(df["Clicks"]))
        metrics["corr_impressions_clicks"] = float(df["Impressions"].corr(df["Clicks"]))
        
        # Performance by query type
        query_type_perf = df.groupby("QueryType").agg({
            "Clicks": "sum",
            "Impressions": "sum",
            "CTR_numeric": "mean",
            "Position": "mean",
            "Query": "count"
        }).reset_index()
        
        metrics["query_length_performance"] = [
            {
                "query_type": row["QueryType"],
                "clicks": int(row["Clicks"]),
                "impressions": int(row["Impressions"]),
                "avg_ctr": f"{row['CTR_numeric']:.2f}%",
                "avg_position": float(row["Position"]),
                "query_count": int(row["Query"])
            }
            for _, row in query_type_perf.iterrows()
        ]
        
        # Performance by search intent
        intent_perf = df.groupby("Intent").agg({
            "Clicks": "sum",
            "Impressions": "sum",
            "CTR_numeric": "mean",
            "Position": "mean",
            "Query": "count"
        }).reset_index()
        
        metrics["intent_performance"] = [
            {
                "intent": row["Intent"],
                "clicks": int(row["Clicks"]),
                "impressions": int(row["Impressions"]),
                "avg_ctr": f"{row['CTR_numeric']:.2f}%",
                "avg_position": float(row["Position"]),
                "query_count": int(row["Query"])
            }
            for _, row in intent_perf.iterrows()
        ]
        
        # Top performing queries
        metrics["top_queries"] = [
            {
                "query": row["Query"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": row["CTR"],
                "position": float(row["Position"])
            } 
            for _, row in df.sort_values("Clicks", ascending=False).head(10).iterrows()
        ]
        
        # Queries with high impressions but low CTR (opportunities)
        opportunity_queries = df[
            (df["Impressions"] > df["Impressions"].quantile(0.7)) & 
            (df["CTR_numeric"] < df["CTR_numeric"].quantile(0.3))
        ].sort_values("ClickPotential", ascending=False).head(10)
        
        metrics["opportunity_queries"] = [
            {
                "query": row["Query"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": row["CTR"],
                "position": float(row["Position"]),
                "click_potential": float(row["ClickPotential"])
            } 
            for _, row in opportunity_queries.iterrows()
        ]
        
        # High-ranking queries with good impressions
        high_ranking = df[
            (df["Position"] < 10) & 
            (df["Impressions"] > df["Impressions"].quantile(0.5))
        ].sort_values("Clicks", ascending=False).head(10)
        
        metrics["high_ranking_queries"] = [
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
        df['position_group'] = pd.cut(df['Position'], position_bins, 
                                     labels=[f'{position_bins[i]}-{position_bins[i+1]-1}' 
                                             for i in range(len(position_bins)-1)])
        
        position_performance = df.groupby('position_group').agg({
            'Clicks': 'sum',
            'Impressions': 'sum',
            'CTR_numeric': 'mean',
            'Query': 'count'
        }).reset_index()
        
        metrics["position_performance"] = [
            {
                "position_range": row["position_group"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": f"{row['CTR_numeric']:.2f}%",
                "query_count": int(row["Query"])
            } 
            for _, row in position_performance.iterrows()
        ]
        
        # CTR Analysis
        ctr_threshold = df["CTR_numeric"].quantile(0.75)
        high_ctr_terms = df[df["CTR_numeric"] > ctr_threshold].sort_values("Impressions", ascending=False).head(10)
        
        metrics["high_ctr_terms"] = [
            {
                "query": row["Query"], 
                "clicks": int(row["Clicks"]), 
                "impressions": int(row["Impressions"]),
                "ctr": row["CTR"],
                "position": float(row["Position"])
            } 
            for _, row in high_ctr_terms.iterrows()
        ]
        
        # Extract query word frequency for content gap analysis
        words = []
        for query in df["Query"]:
            words.extend(word_tokenize(query.lower()))
        
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        word_freq = Counter(filtered_words)
        common_words = word_freq.most_common(20)
        
        metrics["common_query_terms"] = [
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
        
        metrics["ctr_by_position"] = ctr_by_position
        
        # Shannon entropy to measure query diversity
        query_counts = df.groupby("Query").size()
        probabilities = query_counts / query_counts.sum()
        metrics["query_entropy"] = float(-sum(p * np.log2(p) for p in probabilities if p > 0))
        
        self.metrics = metrics
        return metrics
    
    def perform_keyword_clustering(self, n_clusters=5) -> Dict[str, Any]:
        """
        Cluster keywords using TF-IDF and K-means 
        Returns cluster information and visualization
        """
        df = self.data
        
        # Prepare the text data
        tfidf_vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.8,
            max_features=1000,
            stop_words='english'
        )
        
        # If we don't have enough data, reduce requirements
        if len(df) < 20:
            tfidf_vectorizer = TfidfVectorizer(
                min_df=1,
                max_df=0.9,
                max_features=100,
                stop_words='english'
            )
        
        # Determine number of clusters
        n_clusters = min(n_clusters, len(df) // 3) if len(df) >= 9 else min(n_clusters, len(df))
        n_clusters = max(n_clusters, 2)  # Ensure at least 2 clusters
        
        try:
            # Create TF-IDF matrix
            tfidf_matrix = tfidf_vectorizer.fit_transform(df["Query"])
            
            # Find optimal number of clusters using silhouette score
            if tfidf_matrix.shape[0] > 10:
                silhouette_scores = []
                cluster_range = range(2, min(10, tfidf_matrix.shape[0] // 3))
                
                for n in cluster_range:
                    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(tfidf_matrix)
                    if len(set(clusters)) > 1:  # Check that we have more than one cluster
                        score = silhouette_score(tfidf_matrix, clusters)
                        silhouette_scores.append((n, score))
                
                if silhouette_scores:
                    n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(tfidf_matrix)
            
            # Create a dimension reduction model for visualization
            if tfidf_matrix.shape[1] > 2:
                # Use UMAP for better clustering visualization
                umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
                embeddings = umap_model.fit_transform(tfidf_matrix.toarray())
            else:
                # For low-dimensional data, just use the original
                embeddings = tfidf_matrix.toarray()
            
            # Add the embeddings to the dataframe
            df['x'] = embeddings[:, 0]
            df['y'] = embeddings[:, 1]
            
            # Generate cluster analysis
            cluster_data = df.groupby('cluster').agg({
                'Query': lambda x: list(x)[:5],  # Sample of 5 queries per cluster
                'Clicks': 'sum',
                'Impressions': 'sum',
                'CTR_numeric': 'mean',
                'Position': 'mean',
                'x': 'mean',
                'y': 'mean'
            }).reset_index()
            
            # Get top terms in each cluster
            top_terms_per_cluster = {}
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            for cluster_id in range(n_clusters):
                cluster_indices = df[df['cluster'] == cluster_id].index
                
                if len(cluster_indices) > 0:
                    # Get the TF-IDF values for documents in this cluster
                    cluster_tfidf = tfidf_matrix[cluster_indices]
                    
                    # Sum TF-IDF values across all documents in the cluster
                    cluster_tfidf_sum = cluster_tfidf.sum(axis=0)
                    
                    # Convert to an array and get the top terms
                    cluster_tfidf_array = np.asarray(cluster_tfidf_sum).ravel()
                    top_indices = cluster_tfidf_array.argsort()[-5:][::-1]  # Top 5 terms
                    top_terms = [feature_names[i] for i in top_indices]
                    
                    top_terms_per_cluster[cluster_id] = top_terms
                else:
                    top_terms_per_cluster[cluster_id] = []
            
            # Create a cluster name based on top terms
            for i, row in cluster_data.iterrows():
                cluster_id = row['cluster']
                if cluster_id in top_terms_per_cluster:
                    top_terms = top_terms_per_cluster[cluster_id]
                    cluster_name = " & ".join(top_terms[:2]) if top_terms else f"Cluster {cluster_id}"
                    cluster_data.at[i, 'cluster_name'] = cluster_name
                else:
                    cluster_data.at[i, 'cluster_name'] = f"Cluster {cluster_id}"
            
            # Prepare results in the right format
            clusters_result = [
                {
                    "cluster_id": int(row["cluster"]),
                    "cluster_name": row["cluster_name"],
                    "sample_queries": row["Query"],
                    "total_clicks": int(row["Clicks"]),
                    "total_impressions": int(row["Impressions"]),
                    "avg_ctr": f"{row['CTR_numeric']:.2f}%",
                    "avg_position": float(row["Position"]),
                    "top_terms": top_terms_per_cluster.get(row["cluster"], []),
                    "x": float(row["x"]),
                    "y": float(row["y"])
                }
                for _, row in cluster_data.iterrows()
            ]
            
            # Create interactive visualization
            viz_file = self._create_cluster_visualization(df, cluster_data)
            
            self.keyword_clusters = {
                "clusters": clusters_result,
                "visualization": viz_file,
                "n_clusters": n_clusters,
                "method": "k-means+tfidf"
            }
            
            return self.keyword_clusters
            
        except Exception as e:
            logger.error(f"Error in keyword clustering: {e}")
            traceback.print_exc()
            
            # Fallback to simpler clustering
            return {
                "clusters": [{"cluster_id": 0, "sample_queries": df["Query"].tolist()[:10], "error": str(e)}],
                "visualization": None,
                "error": str(e)
            }
    
    def _create_cluster_visualization(self, df: pd.DataFrame, cluster_data: pd.DataFrame) -> str:
        """Create an interactive cluster visualization using plotly"""
        try:
            # Create a colormap for clusters
            colors = px.colors.qualitative.D3[:]  # Make a copy to avoid modifying the original
            
            # If we have more clusters than colors, cycle through the colors
            while len(colors) < len(cluster_data):
                colors.extend(px.colors.qualitative.Plotly)
            
            # Create the figure
            fig = px.scatter(
                df, 
                x='x', 
                y='y', 
                color='cluster',
                hover_data=['Query', 'Clicks', 'Impressions', 'Position', 'CTR'],
                size='Impressions',
                size_max=50,
                opacity=0.8,
                color_discrete_sequence=colors
            )
            
            # Add cluster labels
            for _, row in cluster_data.iterrows():
                cluster_name = row['cluster_name']
                x, y = row['x'], row['y']
                
                fig.add_annotation(
                    x=x, 
                    y=y,
                    text=cluster_name,
                    font=dict(size=14, color="black", family="Arial Black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.5)",
                    borderwidth=1,
                    borderpad=4,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2
                )
            
            # Update layout
            fig.update_layout(
                title="Keyword Clusters",
                template="plotly_white",
                legend_title="Cluster",
                margin=dict(l=20, r=20, t=50, b=20),
                height=600,
                width=800
            )
            
            # Export as PNG
            filename = os.path.join(PLOTLY_DIR, f"keyword_clusters_{int(time.time())}.png")
            fig.write_image(filename, scale=2)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error creating cluster visualization: {e}")
            return None
    
    def extract_keyword_network(self, min_cooccurrence=2) -> Dict[str, Any]:
        """
        Build a network graph of keyword co-occurrences
        Returns network data and visualization
        """
        df = self.data
        
        try:
            # Extract all words from queries
            stop_words = set(stopwords.words('english'))
            
            all_words = []
            query_words = {}
            
            for query in df["Query"]:
                words = [w.lower() for w in word_tokenize(query) if w.lower() not in stop_words and len(w) > 2]
                all_words.extend(words)
                query_words[query] = words
            
            # Count word frequencies
            word_counts = Counter(all_words)
            common_words = [word for word, count in word_counts.most_common(100) if count >= min_cooccurrence]
            
            # Create co-occurrence matrix
            cooccurrence = {}
            
            for query, words in query_words.items():
                words = [w for w in words if w in common_words]
                
                for i, word1 in enumerate(words):
                    for word2 in words[i+1:]:
                        if word1 != word2:
                            pair = tuple(sorted([word1, word2]))
                            cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
            
            # Filter by minimum co-occurrence
            cooccurrence = {pair: count for pair, count in cooccurrence.items() if count >= min_cooccurrence}
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes with word frequency as size
            for word in common_words:
                G.add_node(word, size=word_counts[word])
            
            # Add edges with co-occurrence count as weight
            for (word1, word2), count in cooccurrence.items():
                G.add_edge(word1, word2, weight=count)
            
            # Apply community detection
            communities = nx.community.greedy_modularity_communities(G)
            
            # Map nodes to communities
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            # Add community data to graph
            for node in G.nodes():
                G.nodes[node]['community'] = community_map.get(node, 0)
            
            # Create visualization
            viz_file = self._create_network_visualization(G)
            
            # Prepare output data
            nodes = []
            for node, attr in G.nodes(data=True):
                nodes.append({
                    "word": node,
                    "frequency": attr['size'],
                    "community": attr['community']
                })
            
            edges = []
            for u, v, attr in G.edges(data=True):
                edges.append({
                    "source": u,
                    "target": v,
                    "cooccurrence": attr['weight']
                })
            
            community_keywords = {}
            for i, community in enumerate(communities):
                community_keywords[i] = list(community)
            
            self.keyword_network = {
                "nodes": nodes,
                "edges": edges,
                "communities": community_keywords,
                "visualization": viz_file
            }
            
            return self.keyword_network
            
        except Exception as e:
            logger.error(f"Error in keyword network analysis: {e}")
            return {"error": str(e)}
    
    def _create_network_visualization(self, G: nx.Graph) -> str:
        """Create a network visualization of keyword co-occurrences"""
        try:
            # Calculate layout
            pos = nx.spring_layout(G, k=0.15, iterations=50)
            
            # Create plotly figure
            edge_trace = []
            
            # Add edges
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = G[edge[0]][edge[1]]['weight']
                opacity = min(1.0, 0.1 + weight * 0.1)
                width = 1 + weight * 0.5
                
                trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=width, color='rgba(150,150,150,'+str(opacity)+')'),
                    hoverinfo='none',
                    mode='lines'
                )
                edge_trace.append(trace)
            
            # Prepare node data
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            for node, attr in G.nodes(data=True):
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{node} (freq: {attr['size']})")
                node_size.append(5 + attr['size'] * 0.5)
                node_color.append(attr['community'])
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, 
                y=node_y,
                mode='markers+text',
                text=node_text,
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=node_size,
                    colorscale='Viridis',
                    line=dict(width=2, color='white')
                ),
                textposition='top center',
                textfont=dict(size=10)
            )
            
            # Create figure
            fig = go.Figure(data=edge_trace + [node_trace],
                 layout=go.Layout(
                    title='Keyword Co-occurrence Network',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    width=800,
                    height=600
                ))
            
            # Export as PNG
            filename = os.path.join(PLOTLY_DIR, f"keyword_network_{int(time.time())}.png")
            fig.write_image(filename, scale=2)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error creating network visualization: {e}")
            return None
    
    def analyze_keyword_trends(self, top_n=10) -> Dict[str, Any]:
        """
        Analyze keyword trends using Google Trends API
        Returns trending data and visualization
        """
        df = self.data
        
        try:
            # Focus on top keywords by impressions
            top_keywords = df.sort_values('Impressions', ascending=False).head(top_n)['Query'].tolist()
            
            # Limit to 5 keywords for Google Trends
            trending_keywords = top_keywords[:5]
            
            # Connect to Google Trends
            pytrends = TrendReq(hl='en-US', tz=360)
            
            # Build payload
            try:
                pytrends.build_payload(trending_keywords, timeframe='today 3-m')
                
                # Get interest over time
                trends_data = pytrends.interest_over_time()
                
                # Get related queries
                related_queries = pytrends.related_queries()
                
                # Prepare visualization
                viz_file = self._create_trends_visualization(trends_data, trending_keywords)
                
                # Prepare output data
                trends_result = {
                    "keywords": trending_keywords,
                    "data": trends_data.to_dict() if not trends_data.empty else {},
                    "related_queries": {},
                    "visualization": viz_file
                }
                
                # Add related queries
                for keyword in trending_keywords:
                    if keyword in related_queries and related_queries[keyword] is not None:
                        rising = related_queries[keyword].get('rising')
                        top = related_queries[keyword].get('top')
                        
                        trends_result["related_queries"][keyword] = {
                            "rising": rising.to_dict() if rising is not None and not rising.empty else {},
                            "top": top.to_dict() if top is not None and not top.empty else {}
                        }
                
                self.seasonal_trends = trends_result
                return trends_result
                
            except Exception as e:
                logger.error(f"Error in Google Trends API: {e}")
                return {"error": str(e), "keywords": trending_keywords}
                
        except Exception as e:
            logger.error(f"Error in keyword trends analysis: {e}")
            return {"error": str(e)}
    
    def _create_trends_visualization(self, trends_data: pd.DataFrame, keywords: List[str]) -> str:
        """Create a visualization of keyword trends over time"""
        try:
            if trends_data.empty:
                return None
                
            # Create plotly figure
            fig = go.Figure()
            
            for keyword in keywords:
                if keyword in trends_data.columns:
                    fig.add_trace(go.Scatter(
                        x=trends_data.index,
                        y=trends_data[keyword],
                        mode='lines+markers',
                        name=keyword,
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
            
            # Update layout
            fig.update_layout(
                title="Keyword Interest Over Time (Google Trends)",
                xaxis_title="Date",
                yaxis_title="Interest (0-100)",
                template="plotly_white",
                legend_title="Keywords",
                margin=dict(l=20, r=20, t=50, b=20),
                height=500,
                width=800
            )
            
            # Export as PNG
            filename = os.path.join(PLOTLY_DIR, f"keyword_trends_{int(time.time())}.png")
            fig.write_image(filename, scale=2)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error creating trends visualization: {e}")
            return None
    
    def perform_entity_analysis(self) -> Dict[str, Any]:
        """
        Extract and analyze entities in queries using spaCy
        Returns entity data and visualization
        """
        df = self.data
        
        try:
            if nlp is None:
                return {"error": "spaCy model not available"}
                
            entities = []
            entity_types = {}
            entity_frequencies = {}
            entity_performance = {}
            
            # Process each query with spaCy
            for _, row in df.iterrows():
                query = row["Query"]
                doc = nlp(query)
                
                for ent in doc.ents:
                    entity = ent.text.lower()
                    entity_type = ent.label_
                    
                    # Record entity
                    entities.append({
                        "text": entity,
                        "type": entity_type,
                        "query": query
                    })
                    
                    # Record entity type
                    entity_types[entity] = entity_type
                    
                    # Record frequency
                    entity_frequencies[entity] = entity_frequencies.get(entity, 0) + 1
                    
                    # Record performance
                    if entity not in entity_performance:
                        entity_performance[entity] = {
                            "clicks": 0,
                            "impressions": 0,
                            "ctr": 0,
                            "position": 0,
                            "queries": 0
                        }
                    
                    entity_performance[entity]["clicks"] += row["Clicks"]
                    entity_performance[entity]["impressions"] += row["Impressions"]
                    entity_performance[entity]["position"] += row["Position"]
                    entity_performance[entity]["queries"] += 1
            
            # Calculate averages
            for entity in entity_performance:
                queries = entity_performance[entity]["queries"]
                entity_performance[entity]["position"] /= queries
                entity_performance[entity]["ctr"] = (
                    entity_performance[entity]["clicks"] / entity_performance[entity]["impressions"] * 100
                    if entity_performance[entity]["impressions"] > 0 else 0
                )
            
            # Sort by frequency
            top_entities = sorted(
                [(entity, freq) for entity, freq in entity_frequencies.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            # Prepare visualization
            viz_file = self._create_entity_visualization(top_entities, entity_types)
            
            # Prepare output data
            top_entities_data = []
            for entity, freq in top_entities:
                top_entities_data.append({
                    "entity": entity,
                    "type": entity_types[entity],
                    "frequency": freq,
                    "clicks": entity_performance[entity]["clicks"],
                    "impressions": entity_performance[entity]["impressions"],
                    "ctr": entity_performance[entity]["ctr"],
                    "position": entity_performance[entity]["position"]
                })
            
            entity_result = {
                "entities": entities,
                "top_entities": top_entities_data,
                "visualization": viz_file
            }
            
            self.entity_analysis = entity_result
            return entity_result
            
        except Exception as e:
            logger.error(f"Error in entity analysis: {e}")
            return {"error": str(e)}
    
    def _create_entity_visualization(self, top_entities: List[Tuple[str, int]], entity_types: Dict[str, str]) -> str:
        """Create a visualization of top entities"""
        try:
            # Prepare data
            entities = [e[0] for e in top_entities]
            frequencies = [e[1] for e in top_entities]
            
            # Prepare colors based on entity type
            type_colors = {
                'PERSON': 'rgba(255, 99, 132, 0.7)',
                'ORG': 'rgba(54, 162, 235, 0.7)',
                'GPE': 'rgba(255, 206, 86, 0.7)',
                'LOC': 'rgba(75, 192, 192, 0.7)',
                'PRODUCT': 'rgba(153, 102, 255, 0.7)',
                'EVENT': 'rgba(255, 159, 64, 0.7)',
            }
            
            colors = [type_colors.get(entity_types.get(entity, ''), 'rgba(100, 100, 100, 0.7)') for entity in entities]
            
            # Create plotly figure
            fig = go.Figure(data=[go.Bar(
                x=entities,
                y=frequencies,
                marker_color=colors
            )])
            
            # Add a source column for the legend
            fig.update_layout(
                title="Top Entities in Search Queries",
                xaxis_title="Entity",
                yaxis_title="Frequency",
                template="plotly_white",
                margin=dict(l=20, r=20, t=50, b=20),
                height=500,
                width=800
            )
            
            # Export as PNG
            filename = os.path.join(PLOTLY_DIR, f"entity_analysis_{int(time.time())}.png")
            fig.write_image(filename, scale=2)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error creating entity visualization: {e}")
            return None
    
    def identify_competitors(self, domain=None) -> Dict[str, Any]:
        """
        Identify potential competitors based on search terms
        Simulated data since we can't make real search queries
        """
        if domain is None:
            domain = self.domain
            
        # For demonstration, we'll generate simulated competitor data
        # In a real implementation, this would involve web scraping or using APIs
        
        # Extract brand and product terms from queries
        queries = self.data["Query"].tolist()
        
        # Identify comparison queries (e.g., "X vs Y")
        competitor_mentions = set()
        for query in queries:
            # Look for VS patterns
            if " vs " in query.lower():
                parts = query.lower().split(" vs ")
                for part in parts:
                    if domain.lower() not in part:
                        competitor_mentions.add(part.strip())
            
            # Look for alternatives patterns
            if "alternative" in query.lower() or "competitor" in query.lower():
                competitor_mentions.add(query.strip())
        
        # Create simulated competitor data
        competitors = []
        for i, competitor in enumerate(list(competitor_mentions)[:5]):
            competitors.append({
                "name": competitor,
                "domain": f"{competitor.replace(' ', '')}.com",
                "overlap_score": random.uniform(0.4, 0.9),
                "ranking_similarity": random.uniform(0.3, 0.8),
                "common_keywords": random.randint(10, 50),
                "stronger_rankings": random.randint(5, 25),
                "weaker_rankings": random.randint(5, 25)
            })
        
        # Add some generic competitors if we don't have enough
        generic_competitors = [
            "competitor1.com", "competitor2.com", "competitor3.com",
            "competitor4.com", "competitor5.com"
        ]
        
        while len(competitors) < 5:
            competitor = generic_competitors[len(competitors)]
            competitors.append({
                "name": competitor.split(".")[0],
                "domain": competitor,
                "overlap_score": random.uniform(0.3, 0.7),
                "ranking_similarity": random.uniform(0.2, 0.6),
                "common_keywords": random.randint(5, 30),
                "stronger_rankings": random.randint(3, 15),
                "weaker_rankings": random.randint(3, 15)
            })
        
        self.competitors = {"competitors": competitors}
        return self.competitors
    
    def generate_visualizations(self) -> Dict[str, str]:
        """Generate all visualizations for the SEO report"""
        visualizations = {}
        
        try:
            # Set global style
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette(PURPLE_PALETTE)
            
            # 1. Top Queries by Clicks
            self._generate_top_queries_viz(visualizations)
            
            # 2. CTR vs Position
            self._generate_ctr_position_viz(visualizations)
            
            # 3. Query Intent Distribution
            self._generate_intent_viz(visualizations)
            
            # 4. Clicks vs Impressions
            self._generate_clicks_impressions_viz(visualizations)
            
            # 5. CTR by Position Range
            self._generate_ctr_by_position_viz(visualizations)
            
            # 6. Query Length Analysis
            self._generate_query_length_viz(visualizations)
            
            # 7. Word Cloud
            self._generate_wordcloud_viz(visualizations)
            
            # 8. Performance Distribution
            self._generate_performance_distribution_viz(visualizations)
            
            # 9. Position Distribution
            self._generate_position_distribution_viz(visualizations)
            
            # 10. Opportunity Matrix
            self._generate_opportunity_matrix_viz(visualizations)
            
            self.visualizations = visualizations
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            visualizations["error"] = str(e)
            return visualizations
    
    def _generate_top_queries_viz(self, visualizations: Dict[str, str]) -> None:
        """Generate visualization for top queries by clicks"""
        try:
            # Prepare data
            top_queries = self.data.sort_values("Clicks", ascending=False).head(10)
            
            # Plot figure
            plt.figure(figsize=(10, 6))
            bars = plt.barh(
                top_queries["Query"], 
                top_queries["Clicks"], 
                color=SEO_COLORS["secondary"],
                alpha=0.8,
                edgecolor=SEO_COLORS["primary"],
                linewidth=1
            )
            
            # Add data labels
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 5, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{int(width):,}',
                    va='center',
                    fontweight='bold',
                    color=SEO_COLORS["text"]
                )
            
            # Customize appearance
            plt.xlabel("Clicks", fontsize=12, fontweight='bold', color=SEO_COLORS["text"])
            plt.ylabel("", fontsize=12)
            plt.title("Top 10 Queries by Clicks", fontsize=16, fontweight='bold', color=SEO_COLORS["primary"])
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.xticks(fontsize=10, color=SEO_COLORS["text"])
            plt.yticks(fontsize=10, color=SEO_COLORS["text"])
            
            # Background and borders
            ax = plt.gca()
            ax.set_facecolor('#f9f9f9')
            plt.tight_layout()
            
            # Save figure
            top_queries_path = os.path.join(VISUALIZATIONS_DIR, "top_queries.png")
            plt.savefig(top_queries_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["top_queries"] = top_queries_path
        except Exception as e:
            logger.error(f"Error generating top queries visualization: {e}")
    
    def _generate_ctr_position_viz(self, visualizations: Dict[str, str]) -> None:
        """Generate visualization for CTR vs Position correlation"""
        try:
            df = self.data
            
            # Only take a sample if dataset is very large
            if len(df) > 200:
                df_sample = df.sample(200, random_state=42)
            else:
                df_sample = df.copy()
            
            # Create scatter plot with custom styling
            plt.figure(figsize=(10, 6))
            
            # Create scatter plot
            scatter = plt.scatter(
                df_sample["Position"], 
                df_sample["CTR_numeric"],
                alpha=0.7,
                c=df_sample["Impressions"],
                cmap='viridis',
                s=df_sample["Clicks"] + 20,
                edgecolors=SEO_COLORS["primary"],
                linewidth=0.5
            )
            
            # Add trend line
            x = df_sample["Position"]
            y = df_sample["CTR_numeric"]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(
                x, p(x), 
                linestyle='--',
                color=SEO_COLORS["danger"],
                linewidth=2,
                alpha=0.7
            )
            
            # Calculate correlation
            correlation = x.corr(y)
            
            # Add correlation text
            plt.text(
                0.05, 0.95, 
                f"Correlation: {correlation:.2f}",
                transform=plt.gca().transAxes,
                fontsize=12,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5', edgecolor=SEO_COLORS["primary"])
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Impressions', rotation=270, labelpad=15, fontsize=10)
            
            # Customize appearance
            plt.xlabel("Position", fontsize=12, fontweight='bold', color=SEO_COLORS["text"])
            plt.ylabel("CTR (%)", fontsize=12, fontweight='bold', color=SEO_COLORS["text"])
            plt.title("CTR vs Position Correlation", fontsize=16, fontweight='bold', color=SEO_COLORS["primary"])
            plt.grid(True, linestyle="--", alpha=0.7)
            
            # Set axis limits
            plt.xlim(0, min(df_sample["Position"].max() * 1.1, 100))
            plt.ylim(0, min(df_sample["CTR_numeric"].max() * 1.1, 100))
            
            # Background
            ax = plt.gca()
            ax.set_facecolor('#f9f9f9')
            
            plt.tight_layout()
            ctr_position_path = os.path.join(VISUALIZATIONS_DIR, "ctr_position.png")
            plt.savefig(ctr_position_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["ctr_position"] = ctr_position_path
        except Exception as e:
            logger.error(f"Error generating CTR vs position visualization: {e}")
    
    def _generate_intent_viz(self, visualizations: Dict[str, str]) -> None:
        """Generate visualization for query intent distribution"""
        try:
            df = self.data
            
            # Aggregate by intent
            intent_data = df.groupby("Intent").agg({
                "Query": "count",
                "Clicks": "sum",
                "Impressions": "sum",
                "CTR_numeric": "mean"
            }).reset_index()
            
            # Sort by query count
            intent_data = intent_data.sort_values("Query", ascending=False)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot query counts
            bars = ax.bar(
                intent_data["Intent"], 
                intent_data["Query"],
                color=SEO_COLORS["secondary"],
                alpha=0.8,
                edgecolor=SEO_COLORS["primary"],
                linewidth=1
            )
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    height + 5, 
                    f'{int(height):,}',
                    ha='center',
                    fontweight='bold',
                    color=SEO_COLORS["text"]
                )
            
            # Create secondary axis for CTR
            ax2 = ax.twinx()
            ax2.plot(
                intent_data["Intent"], 
                intent_data["CTR_numeric"],
                marker='o',
                linestyle='-',
                linewidth=3,
                markersize=10,
                color=SEO_COLORS["danger"]
            )
            
            # Customize appearance
            ax.set_xlabel("")
            ax.set_ylabel("Number of Queries", fontsize=12, fontweight='bold', color=SEO_COLORS["text"])
            ax2.set_ylabel("Average CTR (%)", fontsize=12, fontweight='bold', color=SEO_COLORS["danger"])
            ax.set_title("Search Intent Distribution", fontsize=16, fontweight='bold', color=SEO_COLORS["primary"])
            
            # Add legend
            ax.bar(0, 0, color=SEO_COLORS["secondary"], label="Query Count")
            ax2.plot(0, 0, marker='o', linestyle='-', color=SEO_COLORS["danger"], label="Avg CTR")
            fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.9))
            
            # Grid and background
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_facecolor('#f9f9f9')
            
            plt.tight_layout()
            intent_path = os.path.join(VISUALIZATIONS_DIR, "intent_distribution.png")
            plt.savefig(intent_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["intent_distribution"] = intent_path
        except Exception as e:
            logger.error(f"Error generating intent visualization: {e}")
    
    def _generate_clicks_impressions_viz(self, visualizations: Dict[str, str]) -> None:
        """Generate visualization for clicks vs impressions relationship"""
        try:
            df = self.data
            
            # Only take a sample if dataset is very large
            if len(df) > 200:
                df_sample = df.sample(200, random_state=42)
            else:
                df_sample = df.copy()
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create scatter plot
            scatter = plt.scatter(
                df_sample["Impressions"], 
                df_sample["Clicks"],
                alpha=0.7,
                c=df_sample["Position"],
                cmap='viridis_r',  # Reversed so lower positions (better) are greener
                s=df_sample["CTR_numeric"] * 5 + 20,
                edgecolors=SEO_COLORS["primary"],
                linewidth=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Position (Lower is Better)', rotation=270, labelpad=15, fontsize=10)
            
            # Add diagonal line representing perfect conversion
            max_val = max(df_sample["Impressions"].max(), df_sample["Clicks"].max())
            plt.plot([0, max_val], [0, max_val], linestyle='--', color='gray', alpha=0.5)
            
            # Calculate trend line
            x = df_sample["Impressions"]
            y = df_sample["Clicks"]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), linestyle='-', color=SEO_COLORS["danger"], linewidth=2, alpha=0.7)
            
            # Add average CTR line
            avg_ctr = df_sample["CTR_numeric"].mean() / 100
            plt.plot(
                [0, max_val], 
                [0, max_val * avg_ctr],
                linestyle='-.', 
                color=SEO_COLORS["success"], 
                linewidth=2,
                alpha=0.7,
                label=f'Avg CTR: {avg_ctr:.2%}'
            )
            
            # Customize appearance
            plt.xlabel("Impressions", fontsize=12, fontweight='bold', color=SEO_COLORS["text"])
            plt.ylabel("Clicks", fontsize=12, fontweight='bold', color=SEO_COLORS["text"])
            plt.title("Clicks vs Impressions Relationship", fontsize=16, fontweight='bold', color=SEO_COLORS["primary"])
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            
            # Background
            ax = plt.gca()
            ax.set_facecolor('#f9f9f9')
            
            plt.tight_layout()
            clicks_impressions_path = os.path.join(VISUALIZATIONS_DIR, "clicks_impressions.png")
            plt.savefig(clicks_impressions_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["clicks_impressions"] = clicks_impressions_path
        except Exception as e:
            logger.error(f"Error generating clicks vs impressions visualization: {e}")
    
    def _generate_ctr_by_position_viz(self, visualizations: Dict[str, str]) -> None:
        """Generate visualization for CTR by position range"""
        try:
            df = self.data
            
            # Prepare data
            position_bins = [1, 3, 5, 10, 20, 50, 100]
            df['position_group'] = pd.cut(
                df['Position'], 
                position_bins, 
                labels=[f'{position_bins[i]}-{position_bins[i+1]-1}' for i in range(len(position_bins)-1)]
            )
            
            position_perf = df.groupby('position_group').agg({
                'CTR_numeric': 'mean',
                'Impressions': 'sum'
            }).reset_index()
            
            # Sort by position for better visualization
            position_sort_order = [f'{position_bins[i]}-{position_bins[i+1]-1}' for i in range(len(position_bins)-1)]
            position_perf['position_group'] = pd.Categorical(
                position_perf['position_group'], 
                categories=position_sort_order
            )
            position_perf = position_perf.sort_values('position_group')
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create bar chart with gradient colors
            color_gradients = PURPLE_PALETTE
            bars = plt.bar(
                position_perf['position_group'], 
                position_perf['CTR_numeric'],
                color=color_gradients[:len(position_perf)],
                edgecolor=SEO_COLORS["primary"],
                linewidth=1,
                alpha=0.9
            )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + 0.3,
                    f'{height:.2f}%',
                    ha='center', 
                    va='bottom', 
                    fontweight='bold',
                    color=SEO_COLORS["text"]
                )
            
            # Add impression counts at the bottom of the bars
            for i, (_, row) in enumerate(position_perf.iterrows()):
                plt.text(
                    i, 
                    0.5,
                    f'Imp: {int(row["Impressions"]):,}',
                    ha='center', 
                    va='bottom',
                    color='white',
                    fontweight='bold',
                    rotation=90,
                    fontsize=8
                )
            
            # Customize appearance
            plt.xlabel("Position Range", fontsize=12, fontweight='bold', color=SEO_COLORS["text"])
            plt.ylabel("Average CTR (%)", fontsize=12, fontweight='bold', color=SEO_COLORS["text"])
            plt.title("CTR Performance by Position Range", fontsize=16, fontweight='bold', color=SEO_COLORS["primary"])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add trend line to highlight the CTR decay
            x = range(len(position_perf))
            y = position_perf['CTR_numeric']
            z = np.polyfit(x, y, 2)  # Using quadratic fit for better curve
            p = np.poly1d(z)
            xs = np.linspace(0, len(position_perf) - 1, 100)
            plt.plot(
                xs,
                p(xs),
                linestyle='--',
                color=SEO_COLORS["danger"],
                linewidth=2,
                alpha=0.7
            )
            
            # Background
            ax = plt.gca()
            ax.set_facecolor('#f9f9f9')
            
            plt.tight_layout()
            ctr_by_position_path = os.path.join(VISUALIZATIONS_DIR, "ctr_by_position.png")
            plt.savefig(ctr_by_position_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["ctr_by_position"] = ctr_by_position_path
        except Exception as e:
            logger.error(f"Error generating CTR by position visualization: {e}")
    
    def _generate_query_length_viz(self, visualizations: Dict[str, str]) -> None:
        """Generate visualization for query length analysis"""
        try:
            df = self.data
            
            # Prepare data
            df['query_type'] = df['QueryLength'].apply(
                lambda x: 'Short (1-2 words)' if x <= 2 else 'Medium (3-4 words)' if x <= 4 else 'Long (5+ words)'
            )
            
            query_type_perf = df.groupby('query_type').agg({
                'Clicks': 'sum',
                'Impressions': 'sum',
                'CTR_numeric': 'mean',
                'Position': 'mean',
                'Query': 'count'
            }).reset_index()
            
            # Create custom sort order
            query_type_order = ['Short (1-2 words)', 'Medium (3-4 words)', 'Long (5+ words)']
            query_type_perf['query_type'] = pd.Categorical(
                query_type_perf['query_type'], 
                categories=query_type_order
            )
            query_type_perf = query_type_perf.sort_values('query_type')
            
            # Set up the figure with multiple subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot clicks by query type
            axes[0].bar(
                query_type_perf['query_type'],
                query_type_perf['Clicks'],
                color=PURPLE_PALETTE[:3],
                edgecolor=SEO_COLORS["primary"],
                linewidth=1
            )
            axes[0].set_title('Clicks by Query Length', fontweight='bold', color=SEO_COLORS["primary"])
            axes[0].set_xlabel('')
            axes[0].set_ylabel('Clicks', fontweight='bold')
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add count labels
            for i, clicks in enumerate(query_type_perf['Clicks']):
                axes[0].text(
                    i, 
                    clicks + (clicks * 0.05), 
                    f'{int(clicks):,}',
                    ha='center',
                    fontweight='bold',
                    color=SEO_COLORS["text"],
                    fontsize=9
                )
            
            # Plot impressions by query type
            axes[1].bar(
                query_type_perf['query_type'],
                query_type_perf['Impressions'],
                color=PURPLE_PALETTE[3:6],
                edgecolor=SEO_COLORS["primary"],
                linewidth=1
            )
            axes[1].set_title('Impressions by Query Length', fontweight='bold', color=SEO_COLORS["primary"])
            axes[1].set_xlabel('')
            axes[1].set_ylabel('Impressions', fontweight='bold')
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add count labels
            for i, impressions in enumerate(query_type_perf['Impressions']):
                axes[1].text(
                    i, 
                    impressions + (impressions * 0.05), 
                    f'{int(impressions):,}',
                    ha='center',
                    fontweight='bold',
                    color=SEO_COLORS["text"],
                    fontsize=9
                )
            
            # Plot CTR by query type
            bars = axes[2].bar(
                query_type_perf['query_type'],
                query_type_perf['CTR_numeric'],
                color=PURPLE_PALETTE[6:9],
                edgecolor=SEO_COLORS["primary"],
                linewidth=1
            )
            axes[2].set_title('CTR (%) by Query Length', fontweight='bold', color=SEO_COLORS["primary"])
            axes[2].set_xlabel('')
            axes[2].set_ylabel('CTR (%)', fontweight='bold')
            axes[2].grid(axis='y', linestyle='--', alpha=0.7)
            axes[2].tick_params(axis='x', rotation=45)
            
            # Add line showing average position
            ax2 = axes[2].twinx()
            ax2.plot(
                query_type_perf['query_type'],
                query_type_perf['Position'],
                marker='o',
                markersize=8,
                linewidth=2,
                color=SEO_COLORS["danger"]
            )
            ax2.set_ylabel('Avg. Position', fontweight='bold', color=SEO_COLORS["danger"])
            
            # Add CTR labels
            for i, ctr in enumerate(query_type_perf['CTR_numeric']):
                axes[2].text(
                    i, 
                    ctr + 0.3, 
                    f'{ctr:.2f}%',
                    ha='center',
                    fontweight='bold',
                    color=SEO_COLORS["text"],
                    fontsize=9
                )
            
            # Add position labels
            for i, pos in enumerate(query_type_perf['Position']):
                ax2.text(
                    i, 
                    pos + 1, 
                    f'{pos:.1f}',
                    ha='center',
                    fontweight='bold',
                    color=SEO_COLORS["danger"],
                    fontsize=9
                )
            
            # Set background color for all subplots
            for ax in axes:
                ax.set_facecolor('#f9f9f9')
            
            plt.tight_layout()
            query_length_path = os.path.join(VISUALIZATIONS_DIR, "query_length_analysis.png")
            plt.savefig(query_length_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["query_length_analysis"] = query_length_path
        except Exception as e:
            logger.error(f"Error generating query length visualization: {e}")
    
    def _generate_wordcloud_viz(self, visualizations: Dict[str, str]) -> None:
        """Generate word cloud visualization of search terms"""
        try:
            # Prepare text data
            text = ' '.join(self.data['Query'])
            
            # Create a mask shape (optional)
            x, y = np.ogrid[:300, :300]
            mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
            mask = 255 * mask.astype(int)
            
            # Create custom colormaps
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_purple', 
                [SEO_COLORS["tertiary"], SEO_COLORS["secondary"], SEO_COLORS["primary"]], 
                N=256
            )
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                colormap=custom_cmap,
                max_words=150,
                contour_width=1,
                contour_color=SEO_COLORS["neutral"],
                min_font_size=8,
                max_font_size=80,
                random_state=42,
                mask=None,  # Set to mask if you want to use shape
                collocations=False,  # Avoid repeating word pairs
                stopwords=set(stopwords.words('english')),
                regexp=r"\w[\w']+",  # Only include words with letters
            ).generate(text)
            
            # Plot
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(
                "Search Query Word Cloud", 
                fontsize=20, 
                pad=20, 
                fontweight='bold',
                color=SEO_COLORS["primary"]
            )
            
            plt.tight_layout()
            wordcloud_path = os.path.join(VISUALIZATIONS_DIR, "search_terms_wordcloud.png")
            plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["search_terms_wordcloud"] = wordcloud_path
        except Exception as e:
            logger.error(f"Error generating wordcloud visualization: {e}")
    
    def _generate_performance_distribution_viz(self, visualizations: Dict[str, str]) -> None:
        """Generate visualization of performance distributions"""
        try:
            df = self.data
            
            # Create a 2x2 grid of distribution plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Distribution of CTR
            sns.histplot(
                df["CTR_numeric"],
                bins=30,
                kde=True,
                color=SEO_COLORS["secondary"],
                ax=axes[0, 0],
                edgecolor=SEO_COLORS["primary"],
                linewidth=0.5,
                alpha=0.7
            )
            
            # Add vertical line for mean and median
            mean_ctr = df["CTR_numeric"].mean()
            median_ctr = df["CTR_numeric"].median()
            
            axes[0, 0].axvline(
                mean_ctr, 
                color=SEO_COLORS["danger"], 
                linestyle='--', 
                linewidth=2,
                label=f'Mean: {mean_ctr:.2f}%'
            )
            axes[0, 0].axvline(
                median_ctr, 
                color=SEO_COLORS["success"], 
                linestyle='-.', 
                linewidth=2,
                label=f'Median: {median_ctr:.2f}%'
            )
            
            axes[0, 0].set_title('CTR Distribution', fontweight='bold', color=SEO_COLORS["primary"])
            axes[0, 0].set_xlabel('CTR (%)', fontweight='bold')
            axes[0, 0].set_ylabel('Frequency', fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(linestyle='--', alpha=0.7)
            
            # 2. Distribution of Position
            sns.histplot(
                df["Position"].clip(upper=50),  # Clip to avoid extreme outliers
                bins=30,
                kde=True,
                color=SEO_COLORS["secondary"],
                ax=axes[0, 1],
                edgecolor=SEO_COLORS["primary"],
                linewidth=0.5,
                alpha=0.7
            )
            
            # Add vertical line for mean and median
            mean_pos = df["Position"].mean()
            median_pos = df["Position"].median()
            
            axes[0, 1].axvline(
                mean_pos, 
                color=SEO_COLORS["danger"], 
                linestyle='--', 
                linewidth=2,
                label=f'Mean: {mean_pos:.2f}'
            )
            axes[0, 1].axvline(
                median_pos, 
                color=SEO_COLORS["success"], 
                linestyle='-.', 
                linewidth=2,
                label=f'Median: {median_pos:.2f}'
            )
            
            # Add vertical lines for page boundaries
            axes[0, 1].axvline(
                10, 
                color=SEO_COLORS["warning"], 
                linestyle=':', 
                linewidth=2,
                label='Page 1/2 Boundary'
            )
            
            axes[0, 1].set_title('Position Distribution', fontweight='bold', color=SEO_COLORS["primary"])
            axes[0, 1].set_xlabel('Position', fontweight='bold')
            axes[0, 1].set_ylabel('Frequency', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(linestyle='--', alpha=0.7)
            
            # 3. Distribution of Impressions (log scale)
            sns.histplot(
                np.log1p(df["Impressions"]),
                bins=30,
                kde=True,
                color=SEO_COLORS["secondary"],
                ax=axes[1, 0],
                edgecolor=SEO_COLORS["primary"],
                linewidth=0.5,
                alpha=0.7
            )
            
            axes[1, 0].set_title('Impressions Distribution (Log Scale)', fontweight='bold', color=SEO_COLORS["primary"])
            axes[1, 0].set_xlabel('Log(Impressions + 1)', fontweight='bold')
            axes[1, 0].set_ylabel('Frequency', fontweight='bold')
            axes[1, 0].grid(linestyle='--', alpha=0.7)
            
            # 4. Distribution of Clicks (log scale)
            sns.histplot(
                np.log1p(df["Clicks"]),
                bins=30,
                kde=True,
                color=SEO_COLORS["secondary"],
                ax=axes[1, 1],
                edgecolor=SEO_COLORS["primary"],
                linewidth=0.5,
                alpha=0.7
            )
            
            axes[1, 1].set_title('Clicks Distribution (Log Scale)', fontweight='bold', color=SEO_COLORS["primary"])
            axes[1, 1].set_xlabel('Log(Clicks + 1)', fontweight='bold')
            axes[1, 1].set_ylabel('Frequency', fontweight='bold')
            axes[1, 1].grid(linestyle='--', alpha=0.7)
            
            # Set background color for all subplots
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_facecolor('#f9f9f9')
            
            # Main title
            fig.suptitle(
                'Performance Metric Distributions', 
                fontsize=20, 
                y=0.98, 
                fontweight='bold',
                color=SEO_COLORS["primary"]
            )
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            distribution_path = os.path.join(VISUALIZATIONS_DIR, "performance_distributions.png")
            plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["performance_distributions"] = distribution_path
        except Exception as e:
            logger.error(f"Error generating performance distribution visualization: {e}")
    
    def _generate_position_distribution_viz(self, visualizations: Dict[str, str]) -> None:
        """Generate visualization of position distribution across pages"""
        try:
            df = self.data
            
            # Categorize positions by page
            df['page'] = np.ceil(df['Position'] / 10).clip(upper=5).astype(int)
            page_map = {1: 'Page 1', 2: 'Page 2', 3: 'Page 3', 4: 'Page 4', 5: 'Page 5+'}
            df['page_name'] = df['page'].map(page_map)
            
            # Aggregate by page
            page_data = df.groupby('page_name').agg({
                'Query': 'count',
                'Clicks': 'sum',
                'Impressions': 'sum',
                'CTR_numeric': 'mean'
            }).reset_index()
            
            # Create custom sort order for pages
            page_order = ['Page 1', 'Page 2', 'Page 3', 'Page 4', 'Page 5+']
            page_data['page_name'] = pd.Categorical(page_data['page_name'], categories=page_order)
            page_data = page_data.sort_values('page_name')
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Query count by page (donut chart)
            colors = PURPLE_PALETTE[:len(page_data)]
            
            # Create donut chart
            wedges, texts, autotexts = ax1.pie(
                page_data['Query'],
                labels=page_data['page_name'],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.5, edgecolor='w')
            )
            
            # Customize text
            for text in texts:
                text.set_fontsize(10)
                text.set_fontweight('bold')
            
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')
                autotext.set_color('white')
            
            # Create center circle for donut
            centre_circle = plt.Circle((0, 0), 0.3, fc='white')
            ax1.add_patch(centre_circle)
            
            # Add count to center
            total_queries = page_data['Query'].sum()
            ax1.text(
                0, 
                0, 
                f'Total\n{int(total_queries):,}',
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold'
            )
            
            ax1.set_title('Distribution of Queries by Page', fontweight='bold', color=SEO_COLORS["primary"], pad=20)
            
            # Plot 2: Performance metrics by page
            # Create twin axes for multiple metrics
            ax3 = ax2.twinx()
            
            # Bar chart for clicks
            bars1 = ax2.bar(
                page_data['page_name'],
                page_data['Clicks'],
                label='Clicks',
                alpha=0.7,
                color=SEO_COLORS["primary"],
                edgecolor='white',
                linewidth=1
            )
            
            # Line chart for CTR
            line = ax3.plot(
                page_data['page_name'],
                page_data['CTR_numeric'],
                color=SEO_COLORS["danger"],
                marker='o',
                linewidth=3,
                markersize=8,
                label='CTR (%)'
            )
            
            # Set labels
            ax2.set_xlabel('Page Position', fontweight='bold')
            ax2.set_ylabel('Clicks', fontweight='bold', color=SEO_COLORS["primary"])
            ax3.set_ylabel('CTR (%)', fontweight='bold', color=SEO_COLORS["danger"])
            
            # Add data labels
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 5,
                    f'{int(height):,}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color=SEO_COLORS["primary"]
                )
            
            for i, val in enumerate(page_data['CTR_numeric']):
                ax3.text(
                    i,
                    val + 0.5,
                    f'{val:.2f}%',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color=SEO_COLORS["danger"]
                )
            
            # Add title
            ax2.set_title('Clicks and CTR by Page', fontweight='bold', color=SEO_COLORS["primary"], pad=20)
            
            # Add legend
            ax2.legend(loc='upper left')
            ax3.legend(loc='upper right')
            
            # Grid and background
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            ax2.set_facecolor('#f9f9f9')
            
            # Main title
            fig.suptitle(
                'Search Position Analysis by Page', 
                fontsize=18, 
                y=0.98, 
                fontweight='bold',
                color=SEO_COLORS["primary"]
            )
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            position_page_path = os.path.join(VISUALIZATIONS_DIR, "position_by_page.png")
            plt.savefig(position_page_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["position_by_page"] = position_page_path
        except Exception as e:
            logger.error(f"Error generating position distribution visualization: {e}")
    
    def _generate_opportunity_matrix_viz(self, visualizations: Dict[str, str]) -> None:
        """Generate visualization of the opportunity matrix (Impressions vs CTR)"""
        try:
            df = self.data
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Define thresholds for quadrants
            imp_threshold = df['Impressions'].quantile(0.7)
            ctr_threshold = df['CTR_numeric'].quantile(0.3)
            
            # Create scatter plot
            scatter = plt.scatter(
                df['Impressions'],
                df['CTR_numeric'],
                alpha=0.7,
                c=df['Position'],
                cmap='viridis_r',
                s=df['Clicks'] + 20,
                edgecolors='white',
                linewidth=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Position (Lower is Better)', rotation=270, labelpad=15)
            
            # Add quadrant lines
            plt.axvline(
                imp_threshold, 
                color='gray', 
                linestyle='--', 
                alpha=0.5
            )
            plt.axhline(
                ctr_threshold, 
                color='gray', 
                linestyle='--', 
                alpha=0.5
            )
            
            # Label quadrants
            plt.text(
                imp_threshold * 1.1, 
                ctr_threshold * 0.5, 
                'High-Opportunity Zone\n(Optimize These First)',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor=SEO_COLORS["warning"], alpha=0.2, boxstyle='round,pad=0.5')
            )
            
            plt.text(
                imp_threshold * 0.5, 
                ctr_threshold * 0.5, 
                'Low Priority\n(Low Impressions, Low CTR)',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='gray', alpha=0.2, boxstyle='round,pad=0.5')
            )
            
            plt.text(
                imp_threshold * 0.5, 
                ctr_threshold * 1.5, 
                'High Performers\n(Low Impressions, High CTR)',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor=SEO_COLORS["success"], alpha=0.2, boxstyle='round,pad=0.5')
            )
            
            plt.text(
                imp_threshold * 1.1, 
                ctr_threshold * 1.5, 
                'Star Performers\n(High Impressions, High CTR)',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor=SEO_COLORS["success"], alpha=0.2, boxstyle='round,pad=0.5')
            )
            
            # Identify key opportunity keywords
            opportunity_keywords = df[
                (df['Impressions'] > imp_threshold) & 
                (df['CTR_numeric'] < ctr_threshold)
            ].sort_values('Impressions', ascending=False).head(5)
            
            # Annotate key opportunity keywords
            for i, row in opportunity_keywords.iterrows():
                plt.annotate(
                    row['Query'],
                    xy=(row['Impressions'], row['CTR_numeric']),
                    xytext=(15, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=SEO_COLORS["danger"], lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
                )
            
            # Customize plot
            plt.title('SEO Opportunity Matrix', fontsize=16, fontweight='bold', color=SEO_COLORS["primary"])
            plt.xlabel('Impressions', fontsize=12, fontweight='bold')
            plt.ylabel('CTR (%)', fontsize=12, fontweight='bold')
            plt.xscale('log')  # Use log scale for impressions for better visualization
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Background
            ax = plt.gca()
            ax.set_facecolor('#f9f9f9')
            
            plt.tight_layout()
            opportunity_matrix_path = os.path.join(VISUALIZATIONS_DIR, "opportunity_matrix.png")
            plt.savefig(opportunity_matrix_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["opportunity_matrix"] = opportunity_matrix_path
        except Exception as e:
            logger.error(f"Error generating opportunity matrix visualization: {e}")

    def conduct_complete_analysis(self) -> Dict[str, Any]:
        """Perform a complete analysis of the SEO data"""
        try:
            # Step 1: Calculate core metrics
            metrics = self.calculate_core_metrics()
            
            # Step 2: Generate visualizations
            visualizations = self.generate_visualizations()
            
            # Step 3: Perform keyword clustering
            clustering = self.perform_keyword_clustering()
            
            # Step 4: Extract keyword network
            network = self.extract_keyword_network()
            
            # Step 5: Analyze keyword trends (optional - may fail due to API limitations)
            try:
                trends = self.analyze_keyword_trends()
            except:
                trends = {"error": "Could not analyze trends - Google Trends API limitations"}
            
            # Step 6: Perform entity analysis if spaCy is available
            if nlp is not None:
                entities = self.perform_entity_analysis()
            else:
                entities = {"error": "spaCy model not available"}
            
            # Step 7: Identify potential competitors (simulated)
            competitors = self.identify_competitors()
            
            # Compile complete analysis
            analysis = {
                "metrics": metrics,
                "visualizations": visualizations,
                "keyword_clusters": clustering,
                "keyword_network": network,
                "keyword_trends": trends,
                "entity_analysis": entities,
                "competitors": competitors
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error conducting complete analysis: {e}")
            traceback.print_exc()
            return {"error": str(e)}

def generate_seo_insights_with_gemini(data: pd.DataFrame, data_analysis: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Generate SEO insights using the Gemini API with comprehensive data analysis
    """
    try:
        # Setup Gemini API
        if not setup_gemini_api(api_key):
            raise Exception("Failed to setup Gemini API")
        
        # Prepare summary data for Gemini
        metrics = data_analysis["metrics"]
        
        # Extract key metrics
        summary_data = {
            "total_clicks": metrics["total_clicks"],
            "total_impressions": metrics["total_impressions"],
            "avg_ctr": metrics["avg_ctr"],
            "avg_position": metrics["avg_position"],
            "weighted_avg_position": metrics["weighted_avg_position"],
            "page_1_queries": metrics["page_1_queries"],
            "queries_count": metrics["queries_count"],
            "visibility_index": metrics["visibility_index"],
            "click_potential": metrics["click_potential"],
            "position_std": metrics["position_std"],
            "ctr_std": metrics["ctr_std"],
        }
        
        # Extract top queries
        top_queries_list = metrics["top_queries"]
        
        # Extract opportunity queries
        opportunity_queries_list = metrics["opportunity_queries"]
        
        # Extract query intent performance
        intent_performance = metrics.get("intent_performance", [])
        
        # Extract query length performance
        query_length_performance = metrics.get("query_length_performance", [])
        
        # Extract keyword clusters if available
        keyword_clusters = data_analysis.get("keyword_clusters", {}).get("clusters", [])
        
        # Extract entity analysis if available
        entity_analysis = data_analysis.get("entity_analysis", {}).get("top_entities", [])
        
        # Extract competitors if available
        competitors = data_analysis.get("competitors", {}).get("competitors", [])
        
        # Prepare prompt for Gemini
        prompt = f"""
        You are a senior SEO strategist analyzing Google Search Console data. Using the following comprehensive analysis, provide detailed SEO insights and actionable recommendations:

        ## Overall Metrics:
        - Total Clicks: {summary_data['total_clicks']:,}
        - Total Impressions: {summary_data['total_impressions']:,}
        - Average CTR: {summary_data['avg_ctr']:.2f}%
        - Average Position: {summary_data['avg_position']:.2f}
        - Weighted Average Position: {summary_data['weighted_avg_position']:.2f}
        - Page 1 Queries: {summary_data['page_1_queries']} out of {summary_data['queries_count']} total queries
        - Visibility Index: {summary_data['visibility_index']:.2f}
        - Click Potential: {summary_data['click_potential']:.2f} potential additional clicks
        - Position Standard Deviation: {summary_data['position_std']:.2f}
        - CTR Standard Deviation: {summary_data['ctr_std']:.2f}

        ## Top Performing Queries (by Clicks):
        {json.dumps(top_queries_list, indent=2)}

        ## Opportunity Queries (High Impressions, Low CTR):
        {json.dumps(opportunity_queries_list, indent=2)}

        ## Search Intent Performance:
        {json.dumps(intent_performance, indent=2)}

        ## Query Length Performance:
        {json.dumps(query_length_performance, indent=2)}

        ## Keyword Clusters:
        {json.dumps(keyword_clusters[:3], indent=2)}

        ## Top Entities:
        {json.dumps(entity_analysis[:5], indent=2)}

        ## Potential Competitors:
        {json.dumps(competitors, indent=2)}

        Based on this data, please provide:
        1. An executive summary of the website's SEO performance with key insights and interpretation of the metrics
        2. A detailed SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) specific to the SEO performance
        3. Five strategic priorities for immediate action, ranked by potential impact
        4. Ten specific, actionable recommendations with implementation steps
        5. A content strategy with at least 7 detailed content recommendations based on keyword opportunities
        6. Technical SEO improvement suggestions
        7. Keyword strategy recommendations, including suggestions for new content topics
        8. Competitive positioning strategy based on the potential competitors identified

        Format your response as a structured JSON object with the following keys:
        "overview", "strengths", "weaknesses", "opportunities", "threats", "strategic_priorities", "specific_recommendations", "content_strategy", "technical_seo", "keyword_strategy", "competitive_strategy".

        Make your analysis exceptionally insightful, data-driven, and actionable. Provide detailed explanations for each recommendation, connecting them directly to the data provided.
        """
        
        # Generate response
        model = genai.GenerativeModel('gemini-pro')
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
            logger.warning("Failed to parse Gemini response as JSON, using fallback")
            
            # Create a fallback response structure
            return {
                "overview": "Our analysis of your SEO data reveals both strengths and significant opportunities for improvement. Your search visibility shows potential, but there are clear areas where optimization could drive substantial traffic increases.",
                "strengths": [
                    "Several high-performing queries driving consistent traffic",
                    "Some keywords ranking on page 1 with good CTR",
                    "Diverse query types showing broad search relevance",
                    f"Visibility index of {summary_data['visibility_index']:.2f} indicates reasonable search presence",
                    "Good performance for certain search intents",
                ],
                "weaknesses": [
                    "Many high-impression keywords with below-average CTR",
                    f"Average position of {summary_data['avg_position']:.2f} indicates many keywords not on page 1",
                    "High position variance suggesting inconsistent ranking strategy",
                    "Limited performance for certain query types or lengths",
                    "Potential keyword cannibalization for related terms",
                ],
                "opportunities": [
                    f"Potential for {summary_data['click_potential']:.0f} additional clicks by optimizing high-impression, low-CTR queries",
                    "Improvement potential for keywords just below page 1 (positions 11-20)",
                    "Content optimization for specific keyword clusters",
                    "Better targeting of high-value search intents",
                    "Expanded content for underperforming topics",
                ],
                "threats": [
                    "Competitive pressure in key search areas",
                    "Potential algorithm updates affecting current rankings",
                    "Changing search trends and user behavior",
                    "New competitors entering the market",
                    "Evolving SERP features displacing organic listings",
                ],
                "strategic_priorities": [
                    {
                        "priority": "Optimize high-impression, low-CTR content",
                        "potential_impact": "High - Could significantly increase clicks with minimal effort",
                        "timeframe": "Short-term (1-2 months)"
                    },
                    {
                        "priority": "Improve rankings for positions 11-20 keywords",
                        "potential_impact": "High - Moving to page 1 typically increases CTR by 3-10x",
                        "timeframe": "Medium-term (2-4 months)"
                    },
                    {
                        "priority": "Expand content for underperforming keyword clusters",
                        "potential_impact": "Medium - Could capture additional related search traffic",
                        "timeframe": "Medium-term (2-4 months)"
                    },
                    {
                        "priority": "Technical SEO improvements",
                        "potential_impact": "Medium - Could improve overall site performance",
                        "timeframe": "Short to medium-term (1-3 months)"
                    },
                    {
                        "priority": "Competitive keyword strategy",
                        "potential_impact": "Medium to high - Could capture share from competitors",
                        "timeframe": "Long-term (3-6 months)"
                    },
                ],
                "specific_recommendations": [
                    {
                        "recommendation": f"Optimize meta titles and descriptions for top opportunity queries like '{opportunity_queries_list[0]['query'] if opportunity_queries_list else 'high impression terms'}'",
                        "implementation": "Review current meta tags, analyze competitor strategies, rewrite to improve CTR while maintaining keyword relevance",
                        "expected_outcome": "Increased CTR for high-impression queries"
                    },
                    {
                        "recommendation": "Create comprehensive content clusters around top-performing topics",
                        "implementation": "Identify related subtopics, create pillar and cluster content, implement proper internal linking",
                        "expected_outcome": "Improved topical authority and rankings"
                    },
                    {
                        "recommendation": "Implement structured data for key content types",
                        "implementation": "Add appropriate schema markup for products, articles, FAQs, or other relevant content types",
                        "expected_outcome": "Enhanced SERP appearance and potential CTR improvement"
                    },
                    {
                        "recommendation": "Improve page speed for top landing pages",
                        "implementation": "Optimize images, implement lazy loading, minimize JS/CSS, improve server response time",
                        "expected_outcome": "Better user experience and potential ranking improvements"
                    },
                    {
                        "recommendation": "Optimize internal linking structure",
                        "implementation": "Audit current internal links, identify opportunities to link to underperforming pages from strong pages",
                        "expected_outcome": "Better page authority distribution and improved rankings"
                    },
                    {
                        "recommendation": "Create FAQ content targeting question-based searches",
                        "implementation": "Identify question keywords from the data, create comprehensive FAQ sections",
                        "expected_outcome": "Capture more informational search traffic and featured snippets"
                    },
                    {
                        "recommendation": "Consolidate content for closely related keywords",
                        "implementation": "Identify keyword cannibalization, merge or redirect competing pages",
                        "expected_outcome": "Reduced internal competition and improved rankings"
                    },
                    {
                        "recommendation": "Implement content refreshes for outdated pages",
                        "implementation": "Identify older content with declining performance, update with new information",
                        "expected_outcome": "Recovered and improved rankings for existing content"
                    },
                    {
                        "recommendation": "Create comparison content targeting competitor keywords",
                        "implementation": "Develop honest comparison pages addressing competitor strengths and weaknesses",
                        "expected_outcome": "Capture traffic from competitive searches"
                    },
                    {
                        "recommendation": "Optimize for featured snippets",
                        "implementation": "Identify snippet opportunities, structure content with clear Q&A formats, definitions, tables, and lists",
                        "expected_outcome": "Increased visibility and potential position zero rankings"
                    },
                ],
                "content_strategy": [
                    {
                        "topic": f"Comprehensive guide to {top_queries_list[0]['query'] if top_queries_list else 'your primary topic'}",
                        "format": "Long-form guide with subsections",
                        "target_keywords": [q['query'] for q in top_queries_list[:3]] if top_queries_list else ["primary keywords"],
                        "rationale": "Build authority around top-performing search terms"
                    },
                    {
                        "topic": f"How to solve common {opportunity_queries_list[0]['query'] if opportunity_queries_list else 'industry'} problems",
                        "format": "Problem-solution article with examples",
                        "target_keywords": [q['query'] for q in opportunity_queries_list[:3]] if opportunity_queries_list else ["opportunity keywords"],
                        "rationale": "Address pain points revealed in search queries"
                    },
                    {
                        "topic": "Comparison of top solutions/products in the industry",
                        "format": "Comparison table with detailed analysis",
                        "target_keywords": ["best", "top", "vs", "compare", "alternatives"],
                        "rationale": "Capture commercial investigation queries"
                    },
                    {
                        "topic": "Beginner's guide to [topic] terminology",
                        "format": "Glossary with comprehensive definitions",
                        "target_keywords": ["what is", "definition", "meaning", "explained"],
                        "rationale": "Target informational queries and build topical authority"
                    },
                    {
                        "topic": "Expert tips and best practices",
                        "format": "Listicle with actionable advice",
                        "target_keywords": ["how to", "tips", "best practices", "improve"],
                        "rationale": "Address informational queries with high practical value"
                    },
                    {
                        "topic": "Industry trends and future outlook",
                        "format": "Thought leadership article with research",
                        "target_keywords": ["trends", "future", "forecast", "industry developments"],
                        "rationale": "Establish authority and target forward-looking searches"
                    },
                    {
                        "topic": "Common mistakes to avoid",
                        "format": "Warning article with solutions",
                        "target_keywords": ["mistakes", "errors", "problems", "avoid", "fix"],
                        "rationale": "Address pain points and provide valuable guidance"
                    },
                ],
                "technical_seo": [
                    "Implement proper canonical tags for similar content",
                    "Optimize site architecture for improved crawlability",
                    "Address any mobile usability issues",
                    "Implement proper hreflang for international targeting if relevant",
                    "Optimize images with proper alt text and compression",
                    "Implement breadcrumb navigation with structured data",
                    "Ensure proper handling of pagination and filters"
                ],
                "keyword_strategy": {
                    "focus_areas": [
                        "Informational queries with high impression volumes",
                        "Commercial intent keywords with low current CTR",
                        "Long-tail keywords with specific intent",
                        "Question-based searches for featured snippet opportunities",
                        "Competitor comparison keywords"
                    ],
                    "keyword_gaps": [
                        "Beginner-level informational content",
                        "Advanced technical topics",
                        "Current industry trends",
                        "Product/service comparison content",
                        "Problem-solution content"
                    ],
                    "recommended_topics": [
                        top_queries_list[0]['query'] if top_queries_list else "Primary topic 1",
                        top_queries_list[1]['query'] if len(top_queries_list) > 1 else "Primary topic 2",
                        opportunity_queries_list[0]['query'] if opportunity_queries_list else "Opportunity topic 1",
                        opportunity_queries_list[1]['query'] if len(opportunity_queries_list) > 1 else "Opportunity topic 2",
                        "Industry best practices",
                        "Problem solution guides",
                        "Comparison content"
                    ]
                },
                "competitive_strategy": [
                    "Identify and target keyword gaps in competitor content",
                    "Create more comprehensive resources than competitors",
                    "Focus on unique selling points in SEO content",
                    "Develop comparison content highlighting competitive advantages",
                    "Monitor competitor ranking changes and adapt strategy accordingly"
                ]
            }
    
    except Exception as e:
        logger.error(f"Error generating insights with Gemini: {e}")
        traceback.print_exc()
        
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
            # Add other fallback sections as in the example above
            "strategic_priorities": [
                {
                    "priority": "Optimize high-impression, low-CTR content",
                    "potential_impact": "High",
                    "timeframe": "Short-term"
                }
            ],
            "specific_recommendations": [
                {
                    "recommendation": "Optimize meta titles and descriptions",
                    "implementation": "Review current meta tags and improve for higher CTR",
                    "expected_outcome": "Increased click-through rates"
                }
            ],
            "content_strategy": [
                {
                    "topic": "Comprehensive guide to main topic",
                    "format": "Long-form guide",
                    "target_keywords": ["main keywords"],
                    "rationale": "Build authority around top-performing search terms"
                }
            ],
            "technical_seo": [
                "Implement proper canonical tags for similar content",
                "Optimize site architecture for improved crawlability"
            ],
            "keyword_strategy": {
                "focus_areas": [
                    "Informational queries with high impression volumes",
                    "Commercial intent keywords with low current CTR"
                ],
                "recommended_topics": ["Main topic 1", "Main topic 2"]
            },
            "competitive_strategy": [
                "Identify and target keyword gaps in competitor content",
                "Create more comprehensive resources than competitors"
            ],
            "threats": [
                "Increasing competition in key search areas",
                "Evolving search algorithms",
                "Changing user behavior"
            ]
        }

def generate_pdf_report(data_analysis: Dict[str, Any], insights: Dict[str, Any]) -> str:
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
            fontSize=26,
            spaceAfter=24,
            textColor=colors.HexColor(SEO_COLORS["primary"]),
            alignment=1  # Center
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor(SEO_COLORS["secondary"]),
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            spaceAfter=12,
            textColor=colors.HexColor(SEO_COLORS["primary"]),
        )
        
        subheading_style = ParagraphStyle(
            'Subheading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            spaceAfter=10,
            textColor=colors.HexColor(SEO_COLORS["secondary"]),
        )
        
        section_title_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=14,
            spaceAfter=8,
            textColor=colors.black,
        )
        
        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=10,
        )
        
        note_style = ParagraphStyle(
            'Note',
            parent=styles['Italic'],
            fontSize=10,
            textColor=colors.HexColor(SEO_COLORS["neutral"]),
            spaceAfter=6,
        )
        
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=20,
            bulletIndent=10,
        )
        
        table_title_style = ParagraphStyle(
            'TableTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            spaceAfter=6,
            textColor=colors.HexColor(SEO_COLORS["primary"]),
        )
        
        # Define a function to create bullet points
        def create_bullet_points(items, style=bullet_style):
            bullets = []
            for item in items:
                bullets.append(Paragraph(f"• {item}", style))
            return bullets
        
        # Title Page
        elements.append(Paragraph("SEO PERFORMANCE ANALYSIS", title_style))
        elements.append(Paragraph("Advanced Search Engine Optimization Insights & Recommendations", subtitle_style))
        elements.append(Spacer(1, 20))
        
        # Add logo or image (optional)
        # elements.append(Image("logo.png", width=200, height=100))
        
        elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%B %d, %Y')}", note_style))
        elements.append(Spacer(1, 30))
        
        # Add decorative element
        elements.append(Paragraph("<hr color='#9b87f5' width='100%'/>", body_style))
        elements.append(Spacer(1, 30))
        
        # Table of Contents title
        elements.append(Paragraph("Table of Contents", subheading_style))
        
        # Table of Contents
        toc_items = [
            "1. Executive Summary",
            "2. Key Performance Metrics",
            "3. Search Visibility Analysis",
            "4. Keyword Performance Deep Dive",
            "5. Click-Through Rate Analysis",
            "6. Search Query Analysis",
            "7. SEO SWOT Analysis",
            "8. Strategic Priorities",
            "9. Actionable Recommendations",
            "10. Content Strategy Recommendations",
            "11. Technical SEO Recommendations",
            "12. Competitive Analysis",
            "13. Keyword Opportunities"
        ]
        
        for item in toc_items:
            elements.append(Paragraph(f"• {item}", body_style))
        
        elements.append(PageBreak())
        
        # Executive Summary
        elements.append(Paragraph("1. Executive Summary", heading_style))
        elements.append(Paragraph(insights["overview"], body_style))
        elements.append(Spacer(1, 12))
        
        # Add decorative element
        elements.append(Paragraph("<hr color='#9b87f5' width='100%'/>", body_style))
        
        # Key Metrics
        elements.append(Paragraph("2. Key Performance Metrics", heading_style))
        metrics = data_analysis["metrics"]
        data = [
            ["Metric", "Value", "Interpretation"],
            ["Total Clicks", f"{metrics['total_clicks']:,}", "Total user actions from search results"],
            ["Total Impressions", f"{metrics['total_impressions']:,}", "Total search result appearances"],
            ["Average CTR", f"{metrics['avg_ctr']:.2f}%", "Average click-through rate across all queries"],
            ["Average Position", f"{metrics['avg_position']:.2f}", "Average ranking position in search results"],
            ["Weighted Avg Position", f"{metrics.get('weighted_avg_position', 0):.2f}", "Position weighted by impressions"],
            ["Page 1 Queries", f"{metrics.get('page_1_queries', 0)}", "Number of queries ranking on first page"],
            ["Click Potential", f"{metrics.get('click_potential', 0):.0f}", "Estimated additional clicks possible"],
            ["Visibility Index", f"{metrics.get('visibility_index', 0):.2f}", "Overall search visibility score (0-1)"],
        ]
        
        # Create table
        t = Table(data, colWidths=[150, 100, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(SEO_COLORS["primary"])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 24))
        
        # Add a note about interpretation
        elements.append(Paragraph("Note: The Visibility Index is a composite metric representing your overall search presence, where higher values indicate better visibility. Click Potential estimates additional clicks possible through CTR optimization.", note_style))
        
        elements.append(PageBreak())
        
        # Search Visibility Analysis
        elements.append(Paragraph("3. Search Visibility Analysis", heading_style))
        elements.append(Paragraph("The following visualizations provide insights into your search performance and visibility across different metrics:", body_style))
        elements.append(Spacer(1, 12))
        
        # Add visualizations
        visualizations = data_analysis.get("visualizations", {})
        
        if "top_queries" in visualizations:
            elements.append(Paragraph("Top Performing Queries", subheading_style))
            elements.append(Paragraph("These are your top search queries by clicks. They drive the most traffic to your website and represent your strongest performing search terms.", body_style))
            elements.append(Image(visualizations["top_queries"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        if "clicks_impressions" in visualizations:
            elements.append(Paragraph("Clicks vs Impressions Analysis", subheading_style))
            elements.append(Paragraph("This visualization shows the relationship between impressions and clicks. The size of each bubble represents CTR, while color indicates position (greener = better position). Points above the trend line are performing better than average.", body_style))
            elements.append(Image(visualizations["clicks_impressions"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        if "position_by_page" in visualizations:
            elements.append(Paragraph("Search Position Distribution", subheading_style))
            elements.append(Paragraph("This analysis shows how your keywords are distributed across different search result pages and how this affects performance metrics like clicks and CTR.", body_style))
            elements.append(Image(visualizations["position_by_page"], width=500, height=300))
            elements.append(Spacer(1, 12))
        
        elements.append(PageBreak())
        
        # Keyword Performance Deep Dive
        elements.append(Paragraph("4. Keyword Performance Deep Dive", heading_style))
        
        # Add opportunity matrix if available
        if "opportunity_matrix" in visualizations:
            elements.append(Paragraph("SEO Opportunity Matrix", subheading_style))
            elements.append(Paragraph("This matrix plots your keywords based on impressions and CTR. The high-opportunity zone highlights keywords with high impressions but low CTR - these represent your greatest optimization opportunities. Focus on these first for the highest impact.", body_style))
            elements.append(Image(visualizations["opportunity_matrix"], width=450, height=350))
            elements.append(Spacer(1, 12))
        
        # Intent performance if available
        if "intent_distribution" in visualizations:
            elements.append(Paragraph("Search Intent Performance", subheading_style))
            elements.append(Paragraph("This analysis shows the distribution of different search intents in your data and how they perform in terms of CTR and traffic. Understanding user intent helps you optimize content to match what users are looking for.", body_style))
            elements.append(Image(visualizations["intent_distribution"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        # High CTR Terms
        elements.append(Paragraph("High-Converting Search Terms", subheading_style))
        elements.append(Paragraph("These search terms have an above-average CTR, indicating strong alignment with user intent:", body_style))
        
        if "high_ctr_terms" in data_analysis["metrics"]:
            high_ctr_data = [["Query", "Position", "CTR", "Clicks", "Impressions"]]
            for item in data_analysis["metrics"]["high_ctr_terms"]:
                high_ctr_data.append([
                    item["query"],
                    f"{item['position']:.1f}",
                    item["ctr"],
                    f"{item['clicks']}",
                    f"{item['impressions']}"
                ])
            
            t = Table(high_ctr_data, colWidths=[180, 60, 60, 60, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(SEO_COLORS["primary"])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
            ]))
            
            elements.append(t)
        else:
            elements.append(Paragraph("High CTR terms data not available.", body_style))
        
        elements.append(Spacer(1, 12))
        
        # Opportunity Queries
        elements.append(Paragraph("Opportunity Queries", subheading_style))
        elements.append(Paragraph("These queries have high impressions but low CTR, representing your biggest opportunities for optimization:", body_style))
        
        if "opportunity_queries" in data_analysis["metrics"]:
            opportunity_data = [["Query", "Position", "CTR", "Clicks", "Impressions", "Potential"]]
            for item in data_analysis["metrics"]["opportunity_queries"]:
                opportunity_data.append([
                    item["query"],
                    f"{item['position']:.1f}",
                    item["ctr"],
                    f"{item['clicks']}",
                    f"{item['impressions']}",
                    f"{item.get('click_potential', 0):.1f}"
                ])
            
            t = Table(opportunity_data, colWidths=[150, 50, 50, 50, 80, 60])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(SEO_COLORS["primary"])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
            ]))
            
            elements.append(t)
        else:
            elements.append(Paragraph("Opportunity queries data not available.", body_style))
        
        elements.append(Spacer(1, 24))
        elements.append(PageBreak())
        
        # Click-Through Rate Analysis
        elements.append(Paragraph("5. Click-Through Rate Analysis", heading_style))
        
        if "ctr_position" in visualizations:
            elements.append(Paragraph("CTR vs Position Correlation", subheading_style))
            elements.append(Paragraph("This analysis shows how your click-through rate varies by position. The downward trend indicates that as position numbers increase (lower ranking), CTR typically decreases. The correlation coefficient quantifies this relationship.", body_style))
            elements.append(Image(visualizations["ctr_position"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        if "ctr_by_position" in visualizations:
            elements.append(Paragraph("CTR by Position Range", subheading_style))
            elements.append(Paragraph("This chart shows average CTR for different position ranges, helping you understand the dramatic performance drop-off as rankings decrease. Note the significant difference between positions 1-3 versus lower positions.", body_style))
            elements.append(Image(visualizations["ctr_by_position"], width=450, height=270))
            elements.append(Spacer(1, 12))
        
        if "performance_distributions" in visualizations:
            elements.append(Paragraph("Performance Metric Distributions", subheading_style))
            elements.append(Paragraph("These distribution plots show the spread of your key metrics, including CTR and Position. Understanding these distributions helps identify outliers and optimization opportunities.", body_style))
            elements.append(Image(visualizations["performance_distributions"], width=500, height=380))
            elements.append(Spacer(1, 12))
        
        # Position Performance Table
        elements.append(Paragraph("Position Performance Breakdown", subheading_style))
        if "position_performance" in data_analysis["metrics"]:
            position_data = [["Position Range", "Clicks", "Impressions", "CTR", "Query Count"]]
            for item in data_analysis["metrics"]["position_performance"]:
                position_data.append([
                    item["position_range"],
                    f"{item['clicks']:,}",
                    f"{item['impressions']:,}",
                    item["ctr"],
                    f"{item['query_count']}"
                ])
            
            t = Table(position_data, colWidths=[100, 80, 100, 80, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(SEO_COLORS["primary"])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
            ]))
            
            elements.append(t)
        else:
            elements.append(Paragraph("Position performance data not available.", body_style))
        
        elements.append(Spacer(1, 12))
        elements.append(PageBreak())
        
        # Search Query Analysis
        elements.append(Paragraph("6. Search Query Analysis", heading_style))
        
        if "query_length_analysis" in visualizations:
            elements.append(Paragraph("Query Length Performance", subheading_style))
            elements.append(Paragraph("This analysis shows how short, medium, and long-tail keywords perform in terms of clicks, impressions, and CTR. Understanding these patterns helps inform your content and keyword targeting strategy.", body_style))
            elements.append(Image(visualizations["query_length_analysis"], width=500, height=200))
            elements.append(Spacer(1, 12))
        
        # Query Length Performance Details
        if "query_length_performance" in data_analysis["metrics"]:
            elements.append(Paragraph("Query Length Performance Breakdown", subheading_style))
            query_length_data = [["Query Type", "Avg. Position", "Avg. CTR", "Clicks", "Impressions", "Count"]]
            for item in data_analysis["metrics"]["query_length_performance"]:
                query_length_data.append([
                    item["query_type"],
                    f"{item['avg_position']:.1f}",
                    item["avg_ctr"],
                    f"{item['clicks']:,}",
                    f"{item['impressions']:,}",
                    f"{item['query_count']}"
                ])
            
            t = Table(query_length_data, colWidths=[100, 80, 70, 70, 100, 60])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(SEO_COLORS["primary"])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
            ]))
            
            elements.append(t)
            elements.append(Spacer(1, 12))
        
        # Intent Performance if available
        if "intent_performance" in data_analysis["metrics"]:
            elements.append(Paragraph("Search Intent Performance Breakdown", subheading_style))
            intent_data = [["Intent", "Avg. Position", "Avg. CTR", "Clicks", "Impressions", "Count"]]
            for item in data_analysis["metrics"]["intent_performance"]:
                intent_data.append([
                    item["intent"],
                    f"{item['avg_position']:.1f}",
                    item["avg_ctr"],
                    f"{item['clicks']:,}",
                    f"{item['impressions']:,}",
                    f"{item['query_count']}"
                ])
            
            t = Table(intent_data, colWidths=[100, 80, 70, 70, 100, 60])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(SEO_COLORS["primary"])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
            ]))
            
            elements.append(t)
            elements.append(Spacer(1, 12))
        
        # Word cloud visualization
        if "search_terms_wordcloud" in visualizations:
            elements.append(Paragraph("Search Query Word Cloud", subheading_style))
            elements.append(Paragraph("This visualization shows the most common words in your search queries, with size indicating frequency. It provides a quick visual overview of your most important search terms and topics.", body_style))
            elements.append(Image(visualizations["search_terms_wordcloud"], width=500, height=250))
            elements.append(Spacer(1, 12))
        
        # Common Query Terms
        if "common_query_terms" in data_analysis["metrics"]:
            elements.append(Paragraph("Frequently Used Terms in Search Queries", subheading_style))
            elements.append(Paragraph("These are the most common words found in the search queries driving traffic to your site:", body_style))
            
            # Create a table of common terms and frequencies
            terms_data = [["Term", "Frequency", "Term", "Frequency"]]
            common_terms = data_analysis["metrics"]["common_query_terms"]
            
            # Create rows for a two-column layout
            rows = []
            i = 0
            while i < len(common_terms):
                if i + 1 < len(common_terms):
                    # Two terms per row
                    rows.append([
                        common_terms[i]["term"], 
                        f"{common_terms[i]['frequency']}", 
                        common_terms[i+1]["term"],
                        f"{common_terms[i+1]['frequency']}"
                    ])
                else:
                    # Last odd item
                    rows.append([common_terms[i]["term"], f"{common_terms[i]['frequency']}", "", ""])
                i += 2
            
            # Add all rows to the table data
            terms_data.extend(rows)
            
            t = Table(terms_data, colWidths=[100, 60, 100, 60])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.HexColor(SEO_COLORS["primary"])),
                ('BACKGROUND', (2, 0), (3, 0), colors.HexColor(SEO_COLORS["primary"])),
                ('TEXTCOLOR', (0, 0), (3, 0), colors.white),
                ('ALIGN', (0, 0), (3, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (1, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
                ('GRID', (2, 0), (3, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
            ]))
            
            elements.append(t)
        
        elements.append(Spacer(1, 24))
        elements.append(PageBreak())
        
        # Keyword clusters if available
        if "keyword_clusters" in data_analysis:
            elements.append(Paragraph("Keyword Clustering Analysis", subheading_style))
            elements.append(Paragraph("This analysis groups your keywords into meaningful clusters based on their semantic relationships. These clusters help identify broader topics and inform your content strategy.", body_style))
            
            if "visualization" in data_analysis["keyword_clusters"] and data_analysis["keyword_clusters"]["visualization"]:
                elements.append(Image(data_analysis["keyword_clusters"]["visualization"], width=500, height=350))
                elements.append(Spacer(1, 12))
            
            # Add keyword cluster data
            if "clusters" in data_analysis["keyword_clusters"]:
                elements.append(Paragraph("Keyword Cluster Details", section_title_style))
                
                for cluster in data_analysis["keyword_clusters"]["clusters"][:3]:  # Show top 3 clusters
                    elements.append(Paragraph(f"Cluster: {cluster.get('cluster_name', f'Cluster {cluster['cluster_id']}')}", bullet_style))
                    
                    # Add top terms
                    if "top_terms" in cluster and cluster["top_terms"]:
                        elements.append(Paragraph(f"Top terms: {', '.join(cluster['top_terms'])}", bullet_style))
                    
                    # Add sample queries
                    if "sample_queries" in cluster and cluster["sample_queries"]:
                        elements.append(Paragraph(f"Sample queries: {', '.join(cluster['sample_queries'][:3])}", bullet_style))
                    
                    # Add performance metrics
                    elements.append(Paragraph(
                        f"Performance: {cluster.get('total_clicks', 0):,} clicks, {cluster.get('total_impressions', 0):,} impressions, {cluster.get('avg_ctr', '0%')} CTR, Position {cluster.get('avg_position', 0):.1f}",
                        bullet_style
                    ))
                    
                    elements.append(Spacer(1, 6))
                
                elements.append(Spacer(1, 12))
        
        # SEO SWOT Analysis
        elements.append(Paragraph("7. SEO SWOT Analysis", heading_style))
        
        # Create a 2x2 table for SWOT analysis
        swot_data = [
            ["Strengths", "Weaknesses"],
            [Paragraph("\n".join([f"• {s}" for s in insights["strengths"]]), body_style),
             Paragraph("\n".join([f"• {w}" for w in insights["weaknesses"]]), body_style)],
            ["Opportunities", "Threats"],
            [Paragraph("\n".join([f"• {o}" for o in insights["opportunities"]]), body_style),
             Paragraph("\n".join([f"• {t}" for t in insights.get("threats", ["Increasing competition", "Algorithm changes", "Market shifts"])]), body_style)]
        ]
        
        swot_table = Table(swot_data, colWidths=[250, 250])
        swot_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor(SEO_COLORS["success"])),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor(SEO_COLORS["danger"])),
            ('BACKGROUND', (0, 2), (0, 2), colors.HexColor(SEO_COLORS["secondary"])),
            ('BACKGROUND', (1, 2), (1, 2), colors.HexColor(SEO_COLORS["warning"])),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('TEXTCOLOR', (0, 2), (1, 2), colors.white),
            ('ALIGN', (0, 0), (1, 3), 'CENTER'),
            ('VALIGN', (0, 0), (1, 3), 'MIDDLE'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 2), (1, 2), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 3), 12),
            ('SPAN', (0, 0), (0, 0)),
            ('SPAN', (1, 0), (1, 0)),
            ('SPAN', (0, 2), (0, 2)),
            ('SPAN', (1, 2), (1, 2)),
            ('GRID', (0, 0), (1, 3), 1, colors.black),
            ('BACKGROUND', (0, 1), (1, 1), colors.white),
            ('BACKGROUND', (0, 3), (1, 3), colors.white),
            ('LEFTPADDING', (0, 1), (1, 1), 10),
            ('RIGHTPADDING', (0, 1), (1, 1), 10),
            ('LEFTPADDING', (0, 3), (1, 3), 10),
            ('RIGHTPADDING', (0, 3), (1, 3), 10),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('BOTTOMPADDING', (0, 2), (1, 2), 8),
        ]))
        
        elements.append(swot_table)
        elements.append(Spacer(1, 20))
        elements.append(PageBreak())
        
        # Strategic Priorities
        elements.append(Paragraph("8. Strategic Priorities", heading_style))
        elements.append(Paragraph("Based on our analysis, these are the key strategic priorities you should focus on, ranked by potential impact:", body_style))
        
        if "strategic_priorities" in insights:
            # Create table for strategic priorities
            priorities_data = [["Priority", "Potential Impact", "Timeframe"]]
            
            for i, priority in enumerate(insights["strategic_priorities"], 1):
                if isinstance(priority, dict):
                    priorities_data.append([
                        f"{i}. {priority.get('priority', '')}", 
                        priority.get('potential_impact', ''), 
                        priority.get('timeframe', '')
                    ])
                else:
                    priorities_data.append([f"{i}. {priority}", "Medium-High", "1-3 months"])
            
            t = Table(priorities_data, colWidths=[250, 120, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (2, 0), colors.HexColor(SEO_COLORS["primary"])),
                ('TEXTCOLOR', (0, 0), (2, 0), colors.white),
                ('ALIGN', (0, 0), (2, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (2, 0), 12),
                ('BOTTOMPADDING', (0, 0), (2, 0), 12),
                ('BACKGROUND', (0, 1), (2, -1), colors.white),
                ('GRID', (0, 0), (2, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
                ('VALIGN', (0, 0), (2, -1), 'MIDDLE'),
            ]))
            
            elements.append(t)
        else:
            # Fallback if strategic priorities aren't available
            priority_list = [
                "Optimize high-impression, low-CTR content",
                "Improve rankings for positions 11-20 keywords",
                "Expand content for underperforming keyword clusters",
                "Technical SEO improvements",
                "Competitive keyword strategy"
            ]
            
            for i, priority in enumerate(priority_list, 1):
                elements.append(Paragraph(f"{i}. {priority}", bullet_style))
        
        elements.append(Spacer(1, 24))
        
        # Actionable Recommendations
        elements.append(Paragraph("9. Actionable Recommendations", heading_style))
        elements.append(Paragraph("Based on our analysis, we recommend the following specific actions to improve your SEO performance:", body_style))
        
        if "specific_recommendations" in insights:
            for i, rec in enumerate(insights["specific_recommendations"], 1):
                if isinstance(rec, dict):
                    elements.append(Paragraph(f"{i}. {rec.get('recommendation', '')}", subheading_style))
                    elements.append(Paragraph(f"Implementation: {rec.get('implementation', '')}", body_style))
                    elements.append(Paragraph(f"Expected outcome: {rec.get('expected_outcome', '')}", body_style))
                else:
                    elements.append(Paragraph(f"{i}. {rec}", bullet_style))
                
                elements.append(Spacer(1, 8))
        else:
            # Fallback recommendations
            for opportunity in insights["opportunities"]:
                elements.append(Paragraph(f"• {opportunity}", bullet_style))
        
        elements.append(Spacer(1, 12))
        elements.append(PageBreak())
        
        # Content Strategy Recommendations
        elements.append(Paragraph("10. Content Strategy Recommendations", heading_style))
        elements.append(Paragraph("Based on your current performance and search data, we recommend creating the following content:", body_style))
        
        if "content_strategy" in insights:
            for i, content in enumerate(insights["content_strategy"], 1):
                if isinstance(content, dict):
                    elements.append(Paragraph(f"{i}. {content.get('topic', '')}", subheading_style))
                    elements.append(Paragraph(f"Format: {content.get('format', '')}", body_style))
                    elements.append(Paragraph(f"Target keywords: {', '.join(content.get('target_keywords', []))}", body_style))
                    elements.append(Paragraph(f"Rationale: {content.get('rationale', '')}", body_style))
                else:
                    elements.append(Paragraph(f"{i}. {content}", bullet_style))
                
                elements.append(Spacer(1, 8))
        else:
            # Fallback content suggestions
            content_suggestions = [
                "Comprehensive guide to your main topic",
                "How-to tutorial addressing common user problems",
                "Comparison of solutions or products in your industry",
                "Expert roundup featuring industry authorities",
                "Case studies showcasing successful outcomes",
                "FAQ page addressing common user questions",
                "Industry trends and predictions"
            ]
            
            for i, suggestion in enumerate(content_suggestions, 1):
                elements.append(Paragraph(f"{i}. {suggestion}", bullet_style))
        
        elements.append(Spacer(1, 24))
        
        # Technical SEO Recommendations
        elements.append(Paragraph("11. Technical SEO Recommendations", heading_style))
        elements.append(Paragraph("To improve your technical SEO foundations, we recommend:", body_style))
        
        if "technical_seo" in insights:
            for item in insights["technical_seo"]:
                elements.append(Paragraph(f"• {item}", bullet_style))
        else:
            # Fallback technical recommendations
            tech_recommendations = [
                "Implement proper canonical tags for similar content",
                "Optimize site architecture for improved crawlability",
                "Address any mobile usability issues",
                "Implement proper hreflang for international targeting if relevant",
                "Optimize images with proper alt text and compression",
                "Implement breadcrumb navigation with structured data",
                "Ensure proper handling of pagination and filters"
            ]
            
            for rec in tech_recommendations:
                elements.append(Paragraph(f"• {rec}", bullet_style))
        
        elements.append(Spacer(1, 24))
        elements.append(PageBreak())
        
        # Competitive Analysis
        elements.append(Paragraph("12. Competitive Analysis", heading_style))
        
        # If we have competitor data
        competitors = data_analysis.get("competitors", {}).get("competitors", [])
        if competitors:
            elements.append(Paragraph("Based on your search data, we've identified these potential competitors:", body_style))
            
            # Create competitor table
            competitor_data = [["Competitor", "Domain", "Overlap Score", "Ranking Similarity"]]
            
            for comp in competitors:
                competitor_data.append([
                    comp.get("name", ""),
                    comp.get("domain", ""),
                    f"{comp.get('overlap_score', 0):.2f}",
                    f"{comp.get('ranking_similarity', 0):.2f}"
                ])
            
            t = Table(competitor_data, colWidths=[120, 150, 100, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(SEO_COLORS["primary"])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            elements.append(t)
            elements.append(Spacer(1, 12))
        
        # Competitive strategy
        elements.append(Paragraph("Competitive Strategy Recommendations", subheading_style))
        
        if "competitive_strategy" in insights:
            for strategy in insights["competitive_strategy"]:
                elements.append(Paragraph(f"• {strategy}", bullet_style))
        else:
            # Fallback competitive strategies
            comp_strategies = [
                "Identify and target keyword gaps in competitor content",
                "Create more comprehensive resources than competitors",
                "Focus on unique selling points in SEO content",
                "Develop comparison content highlighting competitive advantages",
                "Monitor competitor ranking changes and adapt strategy accordingly"
            ]
            
            for strategy in comp_strategies:
                elements.append(Paragraph(f"• {strategy}", bullet_style))
        
        elements.append(Spacer(1, 24))
        
        # Keyword Opportunities
        elements.append(Paragraph("13. Keyword Opportunities", heading_style))
        elements.append(Paragraph("These keywords represent strategic opportunities for your SEO growth:", body_style))
        
        if "keyword_strategy" in insights and "recommended_topics" in insights["keyword_strategy"]:
            topics = insights["keyword_strategy"]["recommended_topics"]
            
            # Create a more visual representation
            topic_data = []
            
            # Create rows with 2 topics per row
            for i in range(0, len(topics), 2):
                if i + 1 < len(topics):
                    topic_data.append([topics[i], topics[i+1]])
                else:
                    topic_data.append([topics[i], ""])
            
            # Create the table
            t = Table(topic_data, colWidths=[225, 225])
            
            # Style alternating rows
            row_styles = []
            for i in range(len(topic_data)):
                bg_color = colors.HexColor(SEO_COLORS["tertiary"]) if i % 2 == 0 else colors.HexColor(SEO_COLORS["accent1"])
                row_styles.append(('BACKGROUND', (0, i), (1, i), bg_color))
            
            t.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor(SEO_COLORS["text"])),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 15),
            ] + row_styles))
            
            elements.append(t)
        else:
            # Fallback keyword opportunities
            keyword_data = [["Keyword", "Search Volume", "Competition", "Opportunity Score"]]
            
            fallback_keywords = [
                {"keyword": "main topic guide", "volume": "high", "competition": "medium", "opportunity_score": 8},
                {"keyword": "how to [main topic]", "volume": "medium", "competition": "low", "opportunity_score": 9},
                {"keyword": "best [main topic] tools", "volume": "high", "competition": "high", "opportunity_score": 7},
                {"keyword": "[main topic] tips", "volume": "medium", "competition": "medium", "opportunity_score": 8},
                {"keyword": "[main topic] examples", "volume": "medium", "competition": "low", "opportunity_score": 9}
            ]
            
            for kw in fallback_keywords:
                keyword_data.append([
                    kw["keyword"], 
                    kw["volume"], 
                    kw["competition"],
                    f"{kw['opportunity_score']}/10"
                ])
            
            t = Table(keyword_data, colWidths=[200, 100, 100, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(SEO_COLORS["primary"])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(SEO_COLORS["tertiary"])),
            ]))
            
            elements.append(t)
        
        # Final note
        elements.append(Spacer(1, 30))
        elements.append(Paragraph("Note: This report is based on data from Google Search Console and advanced analysis algorithms. Implement recommendations in stages, measuring results after each implementation.", note_style))
        
        # Conclusion
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Thank you for using SEO Seer for your SEO analysis. If you have any questions about this report or would like further insights, please don't hesitate to contact us.", body_style))
        
        # Build the PDF
        doc.build(elements)
        
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        traceback.print_exc()
        raise e

@app.post("/analyze-seo")
async def analyze_seo(
    file: UploadFile = File(...),
    api_key: str = Form(...)
):
    """
    Process Google Search Console export file and generate advanced SEO report
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
        
        # Perform comprehensive data analysis
        analyzer = SEOAnalyzer(df)
        analysis_results = analyzer.conduct_complete_analysis()
        
        # Generate insights using Gemini API
        insights = generate_seo_insights_with_gemini(df, analysis_results, api_key)
        
        # Generate PDF report
        report_path = generate_pdf_report(analysis_results, insights)
        
        # Return the report file
        return FileResponse(
            path=report_path, 
            filename="SEO_Analysis_Report.pdf", 
            media_type="application/pdf"
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "SEO Seer API is running. Use POST /analyze-seo to analyze your GSC data."}
