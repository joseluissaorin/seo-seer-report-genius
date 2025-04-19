
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import plotly.io as pio
import json
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO

# Configure plotly to generate high-quality images
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1200
pio.kaleido.scope.default_height = 800

# Custom color palettes
seo_colors = {
    'primary': '#4e54c8',
    'secondary': '#8f94fb',
    'highlight': '#ff6384',
    'success': '#36a2eb',
    'warning': '#ffce56',
    'danger': '#ff5252',
    'neutral': '#4bc0c0'
}

seo_cmap = LinearSegmentedColormap.from_list(
    'seo_cmap', [seo_colors['primary'], seo_colors['secondary']]
)

def create_seo_health_visualizations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for SEO health score."""
    figs = {}
    
    # Create gauge chart for overall score
    if 'overall_score' in data:
        overall_score = data['overall_score']
        
        # Determine color based on score
        if overall_score >= 80:
            color = seo_colors['success']
        elif overall_score >= 60:
            color = seo_colors['neutral']
        elif overall_score >= 40:
            color = seo_colors['warning']
        else:
            color = seo_colors['danger']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "SEO Health Score", 'font': {'size': 28, 'color': seo_colors['primary']}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255, 82, 82, 0.2)'},
                    {'range': [40, 60], 'color': 'rgba(255, 206, 86, 0.2)'},
                    {'range': [60, 80], 'color': 'rgba(75, 192, 192, 0.2)'},
                    {'range': [80, 100], 'color': 'rgba(54, 162, 235, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            font={'color': seo_colors['primary'], 'family': "Arial"}
        )
        
        figs['health_score_gauge'] = fig
    
    # Create radar chart for component scores
    if 'component_scores' in data:
        component_scores = data['component_scores']
        
        categories = list(component_scores.keys())
        values = list(component_scores.values())
        
        # Radar chart (polar area chart)
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Component Scores',
            line_color=seo_colors['primary'],
            fillcolor=f'rgba({int(seo_colors["primary"][1:3], 16)}, {int(seo_colors["primary"][3:5], 16)}, {int(seo_colors["primary"][5:7], 16)}, 0.5)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            title="SEO Performance Components",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            height=500
        )
        
        figs['component_radar'] = fig
    
    # Create horizontal bar chart for component comparison
    if 'component_scores' in data:
        component_scores = data['component_scores']
        
        categories = list(component_scores.keys())
        values = list(component_scores.values())
        
        # Format category names
        categories = [cat.replace('_', ' ').title() for cat in categories]
        
        # Colors based on score values
        colors = [seo_colors['danger'] if v < 40 else 
                 seo_colors['warning'] if v < 60 else 
                 seo_colors['neutral'] if v < 80 else 
                 seo_colors['success'] for v in values]
        
        # Sort by score
        sorted_data = sorted(zip(categories, values, colors), key=lambda x: x[1])
        categories, values, colors = zip(*sorted_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=categories,
            x=values,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{v:.1f}" for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="SEO Component Performance",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            xaxis=dict(
                title="Score",
                range=[0, 100]
            ),
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Arial"),
            yaxis=dict(
                title="",
                automargin=True
            )
        )
        
        # Add threshold lines
        fig.add_shape(type="line",
            xref="x", yref="paper",
            x0=40, y0=0,
            x1=40, y1=1,
            line=dict(
                color=seo_colors['danger'],
                width=2,
                dash="dash",
            )
        )
        
        fig.add_shape(type="line",
            xref="x", yref="paper",
            x0=60, y0=0,
            x1=60, y1=1,
            line=dict(
                color=seo_colors['warning'],
                width=2,
                dash="dash",
            )
        )
        
        fig.add_shape(type="line",
            xref="x", yref="paper",
            x0=80, y0=0,
            x1=80, y1=1,
            line=dict(
                color=seo_colors['success'],
                width=2,
                dash="dash",
            )
        )
        
        figs['component_bars'] = fig
    
    # Health categories visualization
    if 'health_categories' in data:
        health_categories = data['health_categories']
        
        categories = ['Excellent', 'Good', 'Needs Improvement', 'Critical']
        counts = [len(health_categories['excellent']), 
                 len(health_categories['good']), 
                 len(health_categories['needs_improvement']), 
                 len(health_categories['critical'])]
        
        if sum(counts) > 0:  # Only create if we have categories
            # Colors for categories
            colors = [seo_colors['success'], seo_colors['neutral'], 
                     seo_colors['warning'], seo_colors['danger']]
            
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=categories,
                values=counts,
                marker=dict(colors=colors),
                textinfo='label+percent',
                insidetextorientation='radial',
                hole=0.4
            ))
            
            fig.update_layout(
                title="Health Status Distribution",
                title_font_size=20,
                title_font_color=seo_colors['primary'],
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Arial")
            )
            
            # Add annotation in the center
            fig.add_annotation(
                text=f"{sum(counts)}<br>Components",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )
            
            figs['health_categories_pie'] = fig
    
    return figs

def create_mobile_visualizations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for mobile optimization analysis."""
    figs = {}
    
    # Create gauge chart for mobile score
    if 'mobile_score' in data:
        mobile_score = data['mobile_score']
        
        # Determine color based on score
        if mobile_score >= 80:
            color = seo_colors['success']
        elif mobile_score >= 60:
            color = seo_colors['neutral']
        elif mobile_score >= 40:
            color = seo_colors['warning']
        else:
            color = seo_colors['danger']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mobile_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Mobile Optimization Score", 'font': {'size': 24, 'color': seo_colors['primary']}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255, 82, 82, 0.2)'},
                    {'range': [40, 60], 'color': 'rgba(255, 206, 86, 0.2)'},
                    {'range': [60, 80], 'color': 'rgba(75, 192, 192, 0.2)'},
                    {'range': [80, 100], 'color': 'rgba(54, 162, 235, 0.2)'}
                ]
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            font={'color': seo_colors['primary'], 'family': "Arial"}
        )
        
        figs['mobile_score_gauge'] = fig
    
    # Create pie chart for mobile issue distribution
    if 'issues_summary' in data and data['issues_summary']:
        issues = data['issues_summary']
        
        labels = list(issues.keys())
        values = list(issues.values())
        
        # Format labels
        labels = [label.replace('_', ' ').title() for label in labels]
        
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            insidetextorientation='radial',
            marker=dict(
                colors=[seo_colors['success'] if label.lower() == 'pass' else seo_colors['danger'] for label in labels]
            )
        ))
        
        fig.update_layout(
            title="Mobile Usability Issues",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Arial")
        )
        
        figs['mobile_issues_pie'] = fig
    
    # Create device performance comparison if available
    if 'device_performance' in data and data['device_performance']:
        devices = data['device_performance']
        
        # Extract metrics for all devices
        device_names = list(devices.keys())
        
        # Prepare data for various metrics
        positions = [devices[d]['position'] for d in device_names]
        ctrs = [devices[d]['ctr'] for d in device_names]
        impressions = [devices[d]['impressions'] for d in device_names]
        clicks = [devices[d]['clicks'] for d in device_names]
        
        # Create subplot with 2x2 metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Average Position by Device", "CTR by Device", 
                           "Impressions by Device", "Clicks by Device"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Position (lower is better)
        fig.add_trace(
            go.Bar(
                x=device_names, 
                y=positions,
                marker_color=[seo_colors['success'] if p <= 10 else seo_colors['warning'] for p in positions],
                text=[f"{p:.1f}" for p in positions],
                textposition='auto',
                name="Position"
            ),
            row=1, col=1
        )
        
        # CTR
        fig.add_trace(
            go.Bar(
                x=device_names, 
                y=ctrs,
                marker_color=seo_colors['primary'],
                text=[f"{c:.2f}%" for c in ctrs],
                textposition='auto',
                name="CTR"
            ),
            row=1, col=2
        )
        
        # Impressions
        fig.add_trace(
            go.Bar(
                x=device_names, 
                y=impressions,
                marker_color=seo_colors['neutral'],
                text=[f"{i:,}" for i in impressions],
                textposition='auto',
                name="Impressions"
            ),
            row=2, col=1
        )
        
        # Clicks
        fig.add_trace(
            go.Bar(
                x=device_names, 
                y=clicks,
                marker_color=seo_colors['secondary'],
                text=[f"{c:,}" for c in clicks],
                textposition='auto',
                name="Clicks"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Device Performance Comparison",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            height=600,
            showlegend=False,
            font=dict(family="Arial")
        )
        
        # Update y-axis for position (reversed)
        fig.update_yaxes(title_text="Position", row=1, col=1, autorange="reversed")
        fig.update_yaxes(title_text="CTR (%)", row=1, col=2)
        fig.update_yaxes(title_text="Impressions", row=2, col=1)
        fig.update_yaxes(title_text="Clicks", row=2, col=2)
        
        figs['device_performance_grid'] = fig
    
    return figs

def create_cannibalization_visualizations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for keyword cannibalization analysis."""
    figs = {}
    
    # Create summary visualization of cannibalization issues
    if 'severity_summary' in data:
        severity = data['severity_summary']
        
        labels = ['High Severity', 'Medium Severity', 'Low Severity']
        values = [severity['high'], severity['medium'], severity['low']]
        
        # Only create if we have issues
        if sum(values) > 0:
            colors = [seo_colors['danger'], seo_colors['warning'], seo_colors['neutral']]
            
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo='label+percent',
                insidetextorientation='radial',
                marker=dict(colors=colors)
            ))
            
            fig.update_layout(
                title="Keyword Cannibalization Issues",
                title_font_size=20,
                title_font_color=seo_colors['primary'],
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Arial")
            )
            
            # Add annotation in the center
            fig.add_annotation(
                text=f"{sum(values)}<br>Issues",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )
            
            figs['cannibalization_pie'] = fig
    
    # Create detailed visualization of top cannibalization issues
    if 'cannibalization_issues' in data and data['cannibalization_issues']:
        issues = data['cannibalization_issues'][:5]  # Top 5 issues
        
        # Horizontal bar chart for position difference
        queries = [issue['query'] if len(issue['query']) < 20 else issue['query'][:17] + "..." for issue in issues]
        best_positions = [issue['metrics']['best_position'] for issue in issues]
        worst_positions = [issue['metrics']['worst_position'] for issue in issues]
        
        fig = go.Figure()
        
        # Add traces for best and worst positions
        fig.add_trace(go.Bar(
            y=queries,
            x=best_positions,
            orientation='h',
            name='Best Position',
            marker=dict(color=seo_colors['success']),
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            y=queries,
            x=worst_positions,
            orientation='h',
            name='Worst Position',
            marker=dict(color=seo_colors['danger']),
            opacity=0.7
        ))
        
        # Update layout
        fig.update_layout(
            title="Position Range for Cannibalized Keywords",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            xaxis=dict(
                title="Position",
                range=[0, max(worst_positions) * 1.1],
                autorange="reversed"  # Reverse axis so position 1 is at the right
            ),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Arial"),
            barmode='overlay',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        figs['cannibalization_positions'] = fig
        
        # Create network visualization of competing pages
        # Prepare data for network graph
        G = nx.Graph()
        
        # Add nodes for queries and pages
        for issue in issues:
            query = issue['query']
            primary_page = issue['primary_page']
            competing_pages = issue['competing_pages']
            severity = issue['severity']
            
            # Truncate long URLs
            def truncate_url(url, max_length=30):
                if len(url) > max_length:
                    parsed = urlparse(url)
                    path_parts = parsed.path.split('/')
                    if len(path_parts) > 2:
                        path = '/'.join(path_parts[:1] + ['...'] + path_parts[-1:])
                    else:
                        path = parsed.path
                    return parsed.netloc + path
                return url
            
            primary_page_display = truncate_url(primary_page)
            
            # Add query node
            G.add_node(query, type='query', severity=severity)
            
            # Add primary page node and edge
            G.add_node(primary_page_display, type='primary_page')
            G.add_edge(query, primary_page_display, weight=3)
            
            # Add competing page nodes and edges
            for page in competing_pages[:2]:  # Limit to first 2 competing pages for clarity
                page_display = truncate_url(page)
                G.add_node(page_display, type='competing_page')
                G.add_edge(query, page_display, weight=1)
        
        # Use more sophisticated layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Extract node positions
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_type = G.nodes[node].get('type', '')
            severity = G.nodes[node].get('severity', '')
            
            if node_type == 'query':
                node_colors.append(seo_colors['primary'])
                node_sizes.append(15)
                node_text.append(f"Query: {node}")
            elif node_type == 'primary_page':
                node_colors.append(seo_colors['success'])
                node_sizes.append(12)
                node_text.append(f"Primary Page: {node}")
            else:
                node_colors.append(seo_colors['danger'])
                node_sizes.append(12)
                node_text.append(f"Competing Page: {node}")
        
        # Extract edge positions
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Color edges based on connection type
            source_type = G.nodes[edge[0]].get('type', '')
            target_type = G.nodes[edge[1]].get('type', '')
            
            if 'primary_page' in (source_type, target_type):
                edge_colors.extend([seo_colors['success'], seo_colors['success'], 'rgba(0,0,0,0)'])
            else:
                edge_colors.extend([seo_colors['danger'], seo_colors['danger'], 'rgba(0,0,0,0)'])
        
        # Create the figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color=edge_colors),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n if len(n) < 15 else n[:12] + '...' for n in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='#FFFFFF')
            )
        ))
        
        # Update layout
        fig.update_layout(
            title="Keyword Cannibalization Network",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            showlegend=False,
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            font=dict(family="Arial")
        )
        
        figs['cannibalization_network'] = fig
    
    return figs

def create_serp_feature_visualizations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for SERP feature analysis."""
    figs = {}
    
    # Create summary visualization of SERP features
    if 'feature_summary' in data and data['feature_summary']:
        features = data['feature_summary']
        
        # Format feature names
        labels = [f.replace('_', ' ').title() for f in features.keys()]
        values = list(features.values())
        
        # Sort by frequency
        sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_data)
        
        fig = go.Figure()
        
        # Horizontal bar chart for feature distribution
        fig.add_trace(go.Bar(
            y=labels,
            x=values,
            orientation='h',
            marker=dict(
                color=[seo_colors['primary'], seo_colors['secondary']] * (len(labels) // 2 + 1),
                colorscale=seo_cmap
            ),
            text=values,
            textposition='auto'
        ))
        
        fig.update_layout(
            title="SERP Features Distribution",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            xaxis=dict(title="Frequency"),
            height=min(500, 100 + 40 * len(labels)),  # Adjust height based on number of features
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Arial")
        )
        
        figs['serp_features_bar'] = fig
    
    # Create opportunity visualization
    if 'feature_opportunities' in data and data['feature_opportunities']:
        opportunities = data['feature_opportunities'][:8]  # Top 8 opportunities
        
        if opportunities:
            # Extract data
            features = [opp['feature'].replace('_', ' ').title() for opp in opportunities]
            positions = [opp['current_position'] for opp in opportunities]
            scores = [opp['opportunity_score'] for opp in opportunities]
            
            # Truncate feature text if needed
            features = [f if len(f) < 20 else f[:17] + "..." for f in features]
            
            # Create subplot with opportunity score and current position
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=("Opportunity Score", "Current Position"),
                               specs=[[{"type": "bar"}, {"type": "bar"}]])
            
            # Opportunity score
            fig.add_trace(
                go.Bar(
                    y=features,
                    x=scores,
                    orientation='h',
                    marker_color=seo_colors['primary'],
                    text=[f"{s:.1f}" for s in scores],
                    textposition='auto',
                    name="Score"
                ),
                row=1, col=1
            )
            
            # Current position
            fig.add_trace(
                go.Bar(
                    y=features,
                    x=positions,
                    orientation='h',
                    marker_color=seo_colors['neutral'],
                    text=[f"{p:.1f}" for p in positions],
                    textposition='auto',
                    name="Position"
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="SERP Feature Opportunities",
                title_font_size=20,
                title_font_color=seo_colors['primary'],
                height=500,
                showlegend=False,
                font=dict(family="Arial")
            )
            
            # Update x-axis for position (reversed)
            fig.update_xaxes(title_text="Score", row=1, col=1)
            fig.update_xaxes(title_text="Position", row=1, col=2, autorange="reversed")
            
            figs['serp_opportunities'] = fig
    
    # Create query-feature network visualization
    if 'query_features' in data and data['query_features']:
        query_features = data['query_features']
        
        # Limit to top queries for visualization clarity
        top_queries = list(query_features.keys())[:8]
        
        if top_queries:
            # Create network graph
            G = nx.Graph()
            
            # Collect all unique features
            all_features = set()
            for query in top_queries:
                features = query_features[query]
                all_features.update(features)
            
            # Add nodes for queries and features
            for query in top_queries:
                # Truncate long queries
                display_query = query if len(query) < 30 else query[:27] + "..."
                G.add_node(display_query, type='query')
                
                # Add feature nodes and connections
                features = query_features[query]
                for feature in features:
                    feature_name = feature.replace('_', ' ').title()
                    G.add_node(feature_name, type='feature')
                    G.add_edge(display_query, feature_name)
            
            # Use better layout for bipartite graphs
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            # Extract node positions
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_type = G.nodes[node].get('type', '')
                
                if node_type == 'query':
                    node_colors.append(seo_colors['primary'])
                    node_sizes.append(15)
                    node_text.append(f"Query: {node}")
                else:
                    node_colors.append(seo_colors['highlight'])
                    node_sizes.append(12)
                    node_text.append(f"Feature: {node}")
            
            # Extract edge positions
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create the figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='rgba(150,150,150,0.5)'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[n if len(n) < 20 else n[:17] + '...' for n in G.nodes()],
                textposition="top center",
                hovertext=node_text,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=1, color='#FFFFFF')
                )
            ))
            
            # Update layout
            fig.update_layout(
                title="Query-Feature Network",
                title_font_size=20,
                title_font_color=seo_colors['primary'],
                showlegend=False,
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                font=dict(family="Arial")
            )
            
            figs['serp_feature_network'] = fig
    
    return figs

def create_backlink_visualizations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for backlink analysis."""
    figs = {}
    
    # Create domain authority gauge
    if 'domain_authority' in data:
        domain_authority = data['domain_authority']
        
        # Determine color based on DA score
        if domain_authority >= 70:
            color = seo_colors['success']
        elif domain_authority >= 40:
            color = seo_colors['neutral']
        elif domain_authority >= 20:
            color = seo_colors['warning']
        else:
            color = seo_colors['danger']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=domain_authority,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Domain Authority", 'font': {'size': 24, 'color': seo_colors['primary']}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(255, 82, 82, 0.2)'},
                    {'range': [20, 40], 'color': 'rgba(255, 206, 86, 0.2)'},
                    {'range': [40, 70], 'color': 'rgba(75, 192, 192, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(54, 162, 235, 0.2)'}
                ]
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            font={'color': seo_colors['primary'], 'family': "Arial"}
        )
        
        figs['domain_authority_gauge'] = fig
    
    # Create backlink summary visualization
    if 'backlink_summary' in data and data['backlink_summary']:
        summary = data['backlink_summary']
        
        # Extract metrics
        metrics = ['Total Backlinks', 'Total Referring Domains', 'Backlinks per Domain']
        values = [
            summary.get('total_backlinks', 0),
            summary.get('total_referring_domains', 0),
            summary.get('backlinks_per_domain', 0)
        ]
        
        fig = go.Figure()
        
        # Add bars with custom styling
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker=dict(
                color=[seo_colors['primary'], seo_colors['secondary'], seo_colors['neutral']]
            ),
            text=[f"{v:,}" for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Backlink Profile Summary",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Arial"),
            yaxis=dict(
                title="Count",
                type='log' if max(values) > 1000 else 'linear'  # Use log scale for large values
            )
        )
        
        figs['backlink_summary_bar'] = fig
    
    # Create anchor text visualization
    if 'anchor_text_analysis' in data and 'categories' in data['anchor_text_analysis']:
        anchor_categories = data['anchor_text_analysis']['categories']
        
        # Format for visualization
        labels = [cat.replace('_', ' ').title() for cat in anchor_categories.keys()]
        values = list(anchor_categories.values())
        
        # Create pie chart
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            insidetextorientation='radial',
            marker=dict(
                colors=px.colors.qualitative.Plotly[:len(labels)]
            )
        ))
        
        fig.update_layout(
            title="Anchor Text Distribution",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Arial")
        )
        
        figs['anchor_text_pie'] = fig
    
    # Create referring domains visualization
    if 'referring_domains' in data and 'top_domains' in data['referring_domains']:
        top_domains = data['referring_domains']['top_domains']
        
        if top_domains:
            # Extract data
            domains = list(top_domains.keys())
            counts = list(top_domains.values())
            
            # Limit to top 10
            if len(domains) > 10:
                domains = domains[:10]
                counts = counts[:10]
            
            # Truncate domain names if needed
            domains = [d if len(d) < 25 else d[:22] + "..." for d in domains]
            
            # Sort by count
            sorted_data = sorted(zip(domains, counts), key=lambda x: x[1])
            domains, counts = zip(*sorted_data)
            
            fig = go.Figure()
            
            # Add horizontal bar chart
            fig.add_trace(go.Bar(
                y=domains,
                x=counts,
                orientation='h',
                marker=dict(color=seo_colors['primary']),
                text=counts,
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Top Referring Domains",
                title_font_size=20,
                title_font_color=seo_colors['primary'],
                xaxis=dict(title="Backlinks"),
                height=min(500, 100 + 30 * len(domains)),  # Adjust height based on number of domains
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Arial")
            )
            
            figs['referring_domains_bar'] = fig
    
    return figs

def create_content_gap_visualizations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for content gap analysis."""
    figs = {}
    
    # Create gap summary visualization
    if 'gap_summary' in data:
        summary = data['gap_summary']
        
        # Extract metrics
        labels = ['Content Gaps', 'Topic Opportunities', 'Competitor Advantage Topics']
        values = [
            summary.get('total_gaps', 0),
            summary.get('total_opportunities', 0),
            summary.get('competitor_advantage_topics', 0)
        ]
        
        fig = go.Figure()
        
        # Add bars with custom styling
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=[seo_colors['danger'], seo_colors['warning'], seo_colors['primary']]
            ),
            text=values,
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Content Gap Summary",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Arial")
        )
        
        figs['content_gap_summary'] = fig
    
    # Create opportunity visualization
    if 'topic_opportunities' in data and data['topic_opportunities']:
        opportunities = data['topic_opportunities'][:8]  # Top 8 opportunities
        
        if opportunities:
            # Extract data
            topics = [opp['topic'] for opp in opportunities]
            scores = [opp['opportunity_score'] for opp in opportunities]
            search_volumes = [opp['search_volume'] for opp in opportunities]
            
            # Truncate topic text if needed
            topics = [t if len(t) < 25 else t[:22] + "..." for t in topics]
            
            # Create subplot with opportunity score and search volume
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=("Opportunity Score", "Search Volume"),
                               specs=[[{"type": "bar"}, {"type": "bar"}]])
            
            # Opportunity score
            fig.add_trace(
                go.Bar(
                    y=topics,
                    x=scores,
                    orientation='h',
                    marker_color=seo_colors['primary'],
                    text=[f"{s:.1f}" for s in scores],
                    textposition='auto',
                    name="Score"
                ),
                row=1, col=1
            )
            
            # Search volume
            fig.add_trace(
                go.Bar(
                    y=topics,
                    x=search_volumes,
                    orientation='h',
                    marker_color=seo_colors['neutral'],
                    text=[f"{v:,}" for v in search_volumes],
                    textposition='auto',
                    name="Volume"
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Content Opportunities",
                title_font_size=20,
                title_font_color=seo_colors['primary'],
                height=500,
                showlegend=False,
                font=dict(family="Arial")
            )
            
            fig.update_xaxes(title_text="Score", row=1, col=1)
            fig.update_xaxes(title_text="Volume", row=1, col=2)
            
            figs['content_opportunities'] = fig
    
    # Create topic network visualization
    if 'topic_opportunities' in data and data['topic_opportunities']:
        opportunities = data['topic_opportunities'][:6]  # Limit to top 6 for network clarity
        
        if opportunities:
            # Create network graph
            G = nx.Graph()
            
            # Add nodes for topics and keywords
            for opp in opportunities:
                topic = opp['topic']
                keywords = opp.get('related_keywords', [])[:3]  # Limit to 3 keywords per topic
                score = opp.get('opportunity_score', 50)
                
                # Add topic node with size proportional to score
                G.add_node(topic, type='topic', score=score)
                
                # Add keyword nodes and connections
                for keyword in keywords:
                    # Truncate long keywords
                    display_keyword = keyword if len(keyword) < 30 else keyword[:27] + "..."
                    G.add_node(display_keyword, type='keyword')
                    G.add_edge(topic, display_keyword)
            
            # Use better layout
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            # Extract node positions
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_type = G.nodes[node].get('type', '')
                
                if node_type == 'topic':
                    score = G.nodes[node].get('score', 50)
                    size_factor = min(30, max(15, score / 10))
                    node_colors.append(seo_colors['primary'])
                    node_sizes.append(size_factor)
                    node_text.append(f"Topic: {node}<br>Score: {score:.1f}")
                else:
                    node_colors.append(seo_colors['secondary'])
                    node_sizes.append(10)
                    node_text.append(f"Keyword: {node}")
            
            # Extract edge positions
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create the figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='rgba(150,150,150,0.5)'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[n if len(n) < 20 else n[:17] + '...' for n in G.nodes()],
                textposition="top center",
                hovertext=node_text,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=1, color='#FFFFFF')
                )
            ))
            
            # Update layout
            fig.update_layout(
                title="Content Opportunity Network",
                title_font_size=20,
                title_font_color=seo_colors['primary'],
                showlegend=False,
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                font=dict(family="Arial")
            )
            
            figs['content_network'] = fig
    
    return figs

def create_serp_preview_visualizations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for SERP preview analysis."""
    figs = {}
    
    # Create optimization score visualization
    if 'previews' in data and data['previews']:
        previews = data['previews']
        
        # Extract scores and URLs
        scores = [preview['optimization_score'] for preview in previews]
        urls = [preview['display_url'] for preview in previews]
        
        # Sort by score and limit to top/bottom 5
        scores_urls = sorted(zip(scores, urls))
        bottom_scores = [s for s, _ in scores_urls[:5]]
        bottom_urls = [u for _, u in scores_urls[:5]]
        
        scores_urls = sorted(zip(scores, urls), reverse=True)
        top_scores = [s for s, _ in scores_urls[:5]]
        top_urls = [u for _, u in scores_urls[:5]]
        
        # Truncate URLs if needed
        bottom_urls = [u if len(u) < 30 else u[:27] + "..." for u in bottom_urls]
        top_urls = [u if len(u) < 30 else u[:27] + "..." for u in top_urls]
        
        # Create subplot for top/bottom scores
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Top Optimized Pages", "Pages Needing Optimization"),
                           specs=[[{"type": "bar"}, {"type": "bar"}]])
        
        # Top scores
        fig.add_trace(
            go.Bar(
                y=top_urls,
                x=top_scores,
                orientation='h',
                marker_color=seo_colors['success'],
                text=[f"{s}/100" for s in top_scores],
                textposition='auto',
                name="Top Scores"
            ),
            row=1, col=1
        )
        
        # Bottom scores
        fig.add_trace(
            go.Bar(
                y=bottom_urls,
                x=bottom_scores,
                orientation='h',
                marker_color=seo_colors['danger'],
                text=[f"{s}/100" for s in bottom_scores],
                textposition='auto',
                name="Bottom Scores"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Meta Tag Optimization Scores",
            title_font_size=20,
            title_font_color=seo_colors['primary'],
            height=400,
            showlegend=False,
            font=dict(family="Arial")
        )
        
        fig.update_xaxes(title_text="Score", row=1, col=1, range=[0, 100])
        fig.update_xaxes(title_text="Score", row=1, col=2, range=[0, 100])
        
        figs['serp_optimization_scores'] = fig
    
    # Create meta issues visualization
    if 'optimization_tips' in data and data['optimization_tips']:
        tips = data['optimization_tips']
        
        if 'title_issues' in tips and 'description_issues' in tips:
            title_issues = tips['title_issues']
            desc_issues = tips['description_issues']
            
            # Prepare data for visualization
            issue_types = ['Missing', 'Too Short', 'Too Long', 'Optimal']
            title_counts = [
                title_issues.get('missing', 0),
                title_issues.get('too_short', 0),
                title_issues.get('too_long', 0),
                title_issues.get('optimal', 0)
            ]
            
            desc_counts = [
                desc_issues.get('missing', 0),
                desc_issues.get('too_short', 0),
                desc_issues.get('too_long', 0),
                desc_issues.get('optimal', 0)
            ]
            
            # Create grouped bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=issue_types,
                y=title_counts,
                name='Title',
                marker_color=seo_colors['primary']
            ))
            
            fig.add_trace(go.Bar(
                x=issue_types,
                y=desc_counts,
                name='Description',
                marker_color=seo_colors['secondary']
            ))
            
            # Update layout
            fig.update_layout(
                title="Meta Tag Issues",
                title_font_size=20,
                title_font_color=seo_colors['primary'],
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Arial"),
                barmode='group',
                xaxis=dict(title="Issue Type"),
                yaxis=dict(title="Count"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            figs['meta_issues_bar'] = fig
    
    return figs
