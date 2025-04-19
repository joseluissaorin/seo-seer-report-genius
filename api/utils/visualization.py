
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import pandas as pd
import networkx as nx

def create_query_visualizations(data: Dict) -> Dict:
    """Create visualizations for query analysis."""
    figs = {}
    
    # Query clusters network graph
    G = nx.Graph()
    for cluster in data['query_clusters']:
        for q1 in cluster['queries']:
            for q2 in cluster['queries']:
                if q1 < q2:
                    G.add_edge(q1, q2, weight=cluster['avg_position'])
    
    pos = nx.spring_layout(G)
    fig = go.Figure(data=[
        go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=list(G.nodes()),
            hoverinfo='text',
            marker=dict(size=10)
        )
    ])
    figs['query_network'] = fig
    
    # Performance metrics distribution
    metrics = data['performance_metrics']
    fig = go.Figure(data=[
        go.Bar(
            x=['High Potential', 'Performing Well', 'Needs Improvement'],
            y=[
                metrics['performance_segments']['high_potential'],
                metrics['performance_segments']['performing_well'],
                metrics['performance_segments']['needs_improvement']
            ]
        )
    ])
    figs['performance_distribution'] = fig
    
    return figs

def create_temporal_visualizations(data: Dict) -> Dict:
    """Create visualizations for temporal analysis."""
    figs = {}
    
    # Time series decomposition
    for metric, decomp in data['decomposition'].items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=decomp['trend'], name='Trend'))
        fig.add_trace(go.Scatter(y=decomp['seasonal'], name='Seasonal'))
        fig.add_trace(go.Scatter(y=decomp['resid'], name='Residual'))
        fig.update_layout(title=f'{metric.title()} Decomposition')
        figs[f'{metric}_decomposition'] = fig
    
    return figs

def create_geographic_visualizations(data: Dict) -> Dict:
    """Create visualizations for geographic analysis."""
    figs = {}
    
    # World map of performance
    df = pd.DataFrame.from_dict(data['market_analysis'], orient='index')
    fig = px.choropleth(
        df,
        locations=df.index,
        color='performance_score',
        title='Global Performance Score',
        color_continuous_scale='Viridis'
    )
    figs['world_performance'] = fig
    
    return figs

def create_device_visualizations(data: Dict) -> Dict:
    """Create visualizations for device analysis."""
    figs = {}
    
    # Device distribution
    metrics = data['metrics']
    fig = go.Figure(data=[
        go.Pie(
            labels=list(metrics.keys()),
            values=[d['impression_share'] for d in metrics.values()],
            title='Device Distribution (Impressions)'
        )
    ])
    figs['device_distribution'] = fig
    
    return figs

