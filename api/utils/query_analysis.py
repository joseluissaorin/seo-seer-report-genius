
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import networkx as nx
from typing import Dict, List, Tuple
import spacy

class QueryAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
    def analyze_query_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze search query patterns and relationships."""
        results = {
            'query_clusters': self._cluster_queries(df),
            'keyword_importance': self._calculate_keyword_importance(df),
            'query_relationships': self._analyze_query_relationships(df),
            'performance_metrics': self._calculate_performance_metrics(df)
        }
        return results
    
    def _cluster_queries(self, df: pd.DataFrame) -> List[Dict]:
        """Cluster similar queries using DBSCAN."""
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(df['query'])
        
        clustering = DBSCAN(eps=0.3, min_samples=2)
        clusters = clustering.fit_predict(X)
        
        unique_clusters = sorted(set(clusters))
        cluster_info = []
        
        for cluster_id in unique_clusters:
            if cluster_id != -1:  # Skip noise points
                cluster_queries = df[clusters == cluster_id]
                cluster_info.append({
                    'id': int(cluster_id),
                    'queries': cluster_queries['query'].tolist(),
                    'avg_position': float(cluster_queries['position'].mean()),
                    'total_clicks': int(cluster_queries['clicks'].sum()),
                    'total_impressions': int(cluster_queries['impressions'].sum())
                })
        
        return cluster_info
    
    def _calculate_keyword_importance(self, df: pd.DataFrame) -> List[Dict]:
        """Calculate keyword importance using TF-IDF and performance metrics."""
        keywords = []
        for query in df['query']:
            doc = self.nlp(query.lower())
            keywords.extend([token.text for token in doc if not token.is_stop and token.is_alpha])
        
        keyword_df = pd.DataFrame({'keyword': keywords})
        keyword_stats = keyword_df['keyword'].value_counts().reset_index()
        keyword_stats.columns = ['keyword', 'frequency']
        
        return keyword_stats.head(20).to_dict('records')
    
    def _analyze_query_relationships(self, df: pd.DataFrame) -> Dict:
        """Analyze relationships between queries using graph theory."""
        G = nx.Graph()
        
        for query in df['query']:
            words = set(w.lower() for w in query.split() if len(w) > 3)
            for w1 in words:
                for w2 in words:
                    if w1 < w2:
                        if G.has_edge(w1, w2):
                            G[w1][w2]['weight'] += 1
                        else:
                            G.add_edge(w1, w2, weight=1)
        
        centrality = nx.degree_centrality(G)
        communities = list(nx.community.greedy_modularity_communities(G))
        
        return {
            'central_terms': dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]),
            'community_count': len(communities),
            'largest_community': len(max(communities, key=len))
        }
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate advanced performance metrics."""
        metrics = {
            'avg_position': float(df['position'].mean()),
            'total_clicks': int(df['clicks'].sum()),
            'total_impressions': int(df['impressions'].sum()),
            'avg_ctr': float((df['clicks'].sum() / df['impressions'].sum()) * 100),
            'position_distribution': df['position'].value_counts().to_dict(),
            'performance_segments': {
                'high_potential': len(df[(df['position'] <= 10) & (df['ctr'] < df['ctr'].mean())]),
                'performing_well': len(df[(df['position'] <= 10) & (df['ctr'] >= df['ctr'].mean())]),
                'needs_improvement': len(df[df['position'] > 10])
            }
        }
        return metrics

