
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.io as pio
from adjustText import adjust_text
from scipy.stats import linregress
import json

# Configure plotly to generate high-quality images
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1200
pio.kaleido.scope.default_height = 800

def create_query_visualizations(data: Dict) -> Dict:
    """Create visualizations for query analysis."""
    figs = {}
    
    # Query clusters network graph
    if 'query_clusters' in data and len(data['query_clusters']) > 0:
        G = nx.Graph()
        
        # Add nodes and edges based on cluster data
        for cluster in data['query_clusters']:
            for i, q1 in enumerate(cluster['queries']):
                G.add_node(q1, cluster=cluster['id'], clicks=cluster['total_clicks'])
                
                # Connect queries within the same cluster
                for j, q2 in enumerate(cluster['queries']):
                    if i < j:  # Avoid duplicate edges
                        G.add_edge(q1, q2, weight=2)
        
        # Use a more sophisticated layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Extract node attributes for visualization
        node_sizes = []
        node_colors = []
        node_labels = {}
        
        for node in G.nodes():
            # Determine node size based on clicks
            clicks = G.nodes[node].get('clicks', 10)
            node_sizes.append(max(10, min(50, clicks / 10)))
            
            # Color nodes by cluster
            node_colors.append(G.nodes[node].get('cluster', 0))
            
            # Create labels
            node_labels[node] = node if len(node) < 15 else node[:12] + '...'
        
        # Create plotly figure for network graph
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.6, color='#888'),
            hoverinfo='none',
            mode='lines')
            
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node_labels[node] for node in G.nodes()],
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='#FFFFFF')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Query Relationship Network',
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
                        
        figs['query_network'] = fig
    
    # Performance metrics distribution
    if 'performance_metrics' in data:
        metrics = data['performance_metrics']
        if 'performance_segments' in metrics:
            segments = metrics['performance_segments']
            
            # Create a stacked bar chart with more details
            fig = make_subplots(rows=1, cols=2, 
                                specs=[[{"type": "bar"}, {"type": "pie"}]],
                                subplot_titles=("Query Performance Segmentation", "Position Distribution"))
            
            # Bar chart for performance segments
            fig.add_trace(
                go.Bar(
                    x=['High Potential', 'Performing Well', 'Needs Improvement'],
                    y=[
                        segments.get('high_potential', 0),
                        segments.get('performing_well', 0),
                        segments.get('needs_improvement', 0)
                    ],
                    marker_color=['#66BB6A', '#42A5F5', '#EF5350'],
                    text=[
                        f"{segments.get('high_potential', 0)} queries",
                        f"{segments.get('performing_well', 0)} queries",
                        f"{segments.get('needs_improvement', 0)} queries"
                    ],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Pie chart for position distribution
            if 'position_distribution' in metrics:
                pos_dist = metrics['position_distribution']
                
                positions = []
                counts = []
                
                for pos, count in pos_dist.items():
                    if float(pos) <= 20:  # Include only positions 1-20
                        positions.append(f"Pos {pos}")
                        counts.append(count)
                
                fig.add_trace(
                    go.Pie(
                        labels=positions,
                        values=counts,
                        hole=0.4,
                        marker=dict(
                            colors=px.colors.sequential.Viridis[:len(positions)]
                        )
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title_text="Query Performance Analysis",
                height=500
            )
            
            figs['performance_distribution'] = fig
    
    # Keyword wordcloud
    if 'keyword_importance' in data:
        keywords = data['keyword_importance']
        
        if keywords and len(keywords) > 0:
            # Create a dictionary for wordcloud
            word_freq = {item['keyword']: item['frequency'] for item in keywords}
            
            # Generate wordcloud
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                  colormap='viridis', max_words=100).generate_from_frequencies(word_freq)
            
            # Convert matplotlib wordcloud to plotly figure
            fig = px.imshow(wordcloud, binary_string=True)
            fig.update_layout(
                title='Important Keywords Wordcloud',
                xaxis={'showticklabels': False, 'showgrid': False, 'zeroline': False},
                yaxis={'showticklabels': False, 'showgrid': False, 'zeroline': False}
            )
            
            figs['keyword_wordcloud'] = fig
    
    return figs

def create_temporal_visualizations(data: Dict) -> Dict:
    """Create visualizations for temporal analysis."""
    figs = {}
    
    # Time series visualization with forecasting
    if 'time_series' in data:
        metrics = data['time_series']
        
        for metric_name, metric_data in metrics.items():
            if 'dates' in metric_data and 'values' in metric_data:
                dates = metric_data['dates']
                values = metric_data['values']
                
                # Create a more sophisticated time series visualization
                fig = go.Figure()
                
                # Add actual data
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='#1976D2', width=2)
                    )
                )
                
                # Add trend line if we have enough data
                if len(values) > 5:
                    x_numeric = np.arange(len(values))
                    slope, intercept, r_value, _, _ = linregress(x_numeric, values)
                    trend_line = slope * x_numeric + intercept
                    
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=trend_line,
                            mode='lines',
                            name='Trend',
                            line=dict(color='#388E3C', width=2, dash='dash')
                        )
                    )
                    
                    # Add forecast for next 30 days if we have date objects
                    try:
                        forecast_dates = pd.date_range(start=dates[-1], periods=31)[1:]
                        forecast_x = np.arange(len(values), len(values) + 30)
                        forecast_values = slope * forecast_x + intercept
                        
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_dates,
                                y=forecast_values,
                                mode='lines',
                                name='Forecast',
                                line=dict(color='#FF9800', width=2, dash='dot')
                            )
                        )
                        
                        # Add confidence interval
                        upper_bound = forecast_values + np.std(values) * 1.96
                        lower_bound = forecast_values - np.std(values) * 1.96
                        
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                                y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                                fill='toself',
                                fillcolor='rgba(255, 152, 0, 0.2)',
                                line=dict(color='rgba(255, 152, 0, 0)'),
                                hoverinfo='skip',
                                showlegend=False
                            )
                        )
                    except:
                        # If date parsing fails, we'll skip the forecast
                        pass
                
                fig.update_layout(
                    title=f'{metric_name.title()} Trends & Forecast',
                    xaxis_title='Date',
                    yaxis_title=metric_name.title(),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                figs[f'{metric_name}_trend'] = fig
    
    # Time series decomposition
    if 'decomposition' in data:
        for metric, decomp in data['decomposition'].items():
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Trend Component', 'Seasonal Component', 'Residual Component'),
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            # Add trend component
            fig.add_trace(
                go.Scatter(y=decomp['trend'], mode='lines', name='Trend', line=dict(color='#1976D2')),
                row=1, col=1
            )
            
            # Add seasonal component
            fig.add_trace(
                go.Scatter(y=decomp['seasonal'], mode='lines', name='Seasonal', line=dict(color='#388E3C')),
                row=2, col=1
            )
            
            # Add residual component
            fig.add_trace(
                go.Scatter(y=decomp['resid'], mode='lines', name='Residual', line=dict(color='#E53935')),
                row=3, col=1
            )
            
            fig.update_layout(
                height=600,
                title_text=f'{metric.title()} Time Series Decomposition',
                showlegend=False
            )
            
            figs[f'{metric}_decomposition'] = fig
    
    return figs

def create_geographic_visualizations(data: Dict) -> Dict:
    """Create visualizations for geographic analysis."""
    figs = {}
    
    # World map of performance
    if 'market_analysis' in data:
        df = pd.DataFrame.from_dict(data['market_analysis'], orient='index')
        
        # Create a more comprehensive map visualization
        fig = go.Figure()
        
        # Add choropleth map for performance score
        if 'performance_score' in df.columns:
            fig.add_trace(
                go.Choropleth(
                    locations=df.index,
                    z=df['performance_score'],
                    text=[f"{country}: {score:.1f}" for country, score in zip(df.index, df['performance_score'])],
                    colorscale='Viridis',
                    autocolorscale=False,
                    marker_line_color='white',
                    marker_line_width=0.5,
                    colorbar_title='Score'
                )
            )
        
        fig.update_layout(
            title_text='Global Performance Score by Country',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            )
        )
        
        figs['world_performance'] = fig
    
    # Opportunity map
    if 'opportunities' in data:
        opportunities = data['opportunities']
        
        # Create a bubble map for opportunities
        if opportunities and len(opportunities) > 0:
            countries = []
            opportunity_types = []
            scores = []
            sizes = []
            texts = []
            
            for opp in opportunities:
                if 'country' in opp and 'score' in opp:
                    countries.append(opp['country'])
                    opportunity_types.append(opp.get('type', 'general'))
                    scores.append(opp['score'])
                    sizes.append(opp.get('potential', 50))
                    
                    # Create detailed hover text
                    text = f"Country: {opp['country']}<br>"
                    text += f"Opportunity: {opp.get('type', 'general').replace('_', ' ').title()}<br>"
                    text += f"Score: {opp['score']}"
                    
                    if 'current_ctr' in opp and 'benchmark_ctr' in opp:
                        text += f"<br>Current CTR: {opp['current_ctr']:.2f}%"
                        text += f"<br>Benchmark CTR: {opp['benchmark_ctr']:.2f}%"
                        
                    if 'current_position' in opp:
                        text += f"<br>Current Position: {opp['current_position']:.1f}"
                        
                    texts.append(text)
            
            # Create bubble map
            fig = px.scatter_geo(
                locationmode='country names',
                locations=countries,
                color=opportunity_types,
                size=sizes,
                hover_name=countries,
                hover_data={'text': texts},
                projection='natural earth',
                title='Geographic Opportunities',
                color_discrete_map={
                    'ctr_improvement': '#FF9800',
                    'ranking_improvement': '#2196F3',
                    'general': '#4CAF50'
                }
            )
            
            fig.update_layout(
                height=600,
                geo=dict(
                    showland=True,
                    landcolor='rgb(243, 243, 243)',
                    countrycolor='rgb(204, 204, 204)',
                    showcoastlines=True
                )
            )
            
            figs['opportunity_map'] = fig
    
    return figs

def create_device_visualizations(data: Dict) -> Dict:
    """Create visualizations for device analysis."""
    figs = {}
    
    # Device distribution and performance
    if 'metrics' in data:
        metrics = data['metrics']
        
        if metrics and len(metrics) > 0:
            # Extract data for visualization
            devices = list(metrics.keys())
            clicks = [metrics[d]['total_clicks'] for d in devices]
            impressions = [metrics[d]['total_impressions'] for d in devices]
            positions = [metrics[d]['avg_position'] for d in devices]
            ctrs = [metrics[d]['avg_ctr'] for d in devices]
            
            # Create a comprehensive dashboard with subplots
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "pie"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}]
                ],
                subplot_titles=(
                    "Device Distribution (Impressions)", 
                    "Clicks by Device",
                    "Average Position by Device",
                    "Average CTR by Device"
                )
            )
            
            # Add impression distribution pie chart
            fig.add_trace(
                go.Pie(
                    labels=devices,
                    values=impressions,
                    hole=0.4,
                    textinfo='label+percent',
                    marker=dict(colors=px.colors.qualitative.Set3[:len(devices)])
                ),
                row=1, col=1
            )
            
            # Add clicks bar chart
            fig.add_trace(
                go.Bar(
                    x=devices,
                    y=clicks,
                    text=clicks,
                    textposition='auto',
                    marker_color='#42A5F5'
                ),
                row=1, col=2
            )
            
            # Add position bar chart
            fig.add_trace(
                go.Bar(
                    x=devices,
                    y=positions,
                    text=[f"{p:.1f}" for p in positions],
                    textposition='auto',
                    marker_color='#66BB6A'
                ),
                row=2, col=1
            )
            
            # Add CTR bar chart
            fig.add_trace(
                go.Bar(
                    x=devices,
                    y=ctrs,
                    text=[f"{c:.2f}%" for c in ctrs],
                    textposition='auto',
                    marker_color='#FFA726'
                ),
                row=2, col=2
            )
            
            # Update layout with better styling
            fig.update_layout(
                title_text="Device Performance Analysis",
                height=800,
                showlegend=False
            )
            
            # Update y-axis titles
            fig.update_yaxes(title_text="Clicks", row=1, col=2)
            fig.update_yaxes(title_text="Avg Position", row=2, col=1)
            fig.update_yaxes(title_text="CTR (%)", row=2, col=2)
            
            figs['device_dashboard'] = fig
    
    return figs

def create_keyword_research_visualizations(data: Dict) -> Dict:
    """Create visualizations for keyword research."""
    figs = {}
    
    # Trend visualization
    if 'trend_data' in data:
        for keyword, trend_info in data['trend_data'].items():
            if 'dates' in trend_info and 'values' in trend_info:
                dates = trend_info['dates']
                values = trend_info['values']
                
                fig = go.Figure()
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines+markers',
                        name=keyword,
                        line=dict(color='#1976D2', width=2)
                    )
                )
                
                fig.update_layout(
                    title=f'Search Interest Trend: {keyword}',
                    xaxis_title='Date',
                    yaxis_title='Interest',
                    height=400
                )
                
                figs[f'trend_{keyword}'] = fig
    
    # Related queries visualization
    if 'related_queries' in data:
        for keyword, related_info in data['related_queries'].items():
            if 'top' in related_info and len(related_info['top']) > 0:
                # Extract top related queries
                queries = [item.get('query', '') for item in related_info['top'][:10]]
                values = [item.get('value', 0) for item in related_info['top'][:10]]
                
                # Create horizontal bar chart
                fig = go.Figure()
                
                fig.add_trace(
                    go.Bar(
                        y=queries,
                        x=values,
                        orientation='h',
                        marker_color='#5E35B1',
                        text=values,
                        textposition='auto'
                    )
                )
                
                fig.update_layout(
                    title=f'Top Related Queries: {keyword}',
                    yaxis=dict(autorange="reversed"),
                    height=400
                )
                
                figs[f'related_queries_{keyword}'] = fig
    
    # Keyword difficulty visualization
    if 'difficulty' in data:
        difficulties = data['difficulty']
        
        if difficulties and len(difficulties) > 0:
            keywords = list(difficulties.keys())
            scores = list(difficulties.values())
            
            # Create gauge charts for each keyword
            difficulty_figs = []
            
            for keyword, score in difficulties.items():
                # Determine color based on difficulty
                if score < 30:
                    color = "green"
                elif score < 70:
                    color = "orange"
                else:
                    color = "red"
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={"text": f"Difficulty: {keyword}"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 30], "color": "lightgreen"},
                            {"range": [30, 70], "color": "lightyellow"},
                            {"range": [70, 100], "color": "lightcoral"}
                        ]
                    }
                ))
                
                fig.update_layout(height=300)
                difficulty_figs.append(fig)
            
            figs['keyword_difficulty'] = difficulty_figs
    
    return figs

def create_competitor_visualizations(data: Dict) -> Dict:
    """Create visualizations for competitor analysis."""
    figs = {}
    
    # Competitor share visualization
    if 'competitors' in data:
        competitors = data['competitors']
        
        if competitors and len(competitors) > 0:
            domains = list(competitors.keys())
            appearances = [comp['appearances'] for comp in competitors.values()]
            shares = [comp['share'] for comp in competitors.values()]
            
            # Create competitor share visualization
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=domains,
                    y=shares,
                    marker_color='#42A5F5',
                    text=[f"{s:.1f}%" for s in shares],
                    textposition='auto'
                )
            )
            
            fig.update_layout(
                title='Competitor Share of Search Results',
                xaxis_title='Competitor Domain',
                yaxis_title='Share (%)',
                height=400
            )
            
            figs['competitor_share'] = fig
    
    # Ranking comparison visualization
    if 'ranking_comparison' in data:
        comparison = data['ranking_comparison']
        
        if 'queries' in comparison and len(comparison['queries']) > 0:
            queries = list(comparison['queries'].keys())
            positions = [comp['position'] for comp in comparison['queries'].values()]
            
            # Create colorized position chart
            colors = ['#4CAF50' if pos <= 3 else '#FFC107' if pos <= 10 else '#F44336' for pos in positions]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=queries,
                    y=positions,
                    marker_color=colors,
                    text=[f"Pos {p}" if p > 0 else "Not ranked" for p in positions],
                    textposition='auto'
                )
            )
            
            # Add position thresholds
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=3,
                x1=len(queries) - 0.5,
                y1=3,
                line=dict(
                    color="#4CAF50",
                    width=2,
                    dash="dash",
                ),
            )
            
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=10,
                x1=len(queries) - 0.5,
                y1=10,
                line=dict(
                    color="#FFC107",
                    width=2,
                    dash="dash",
                ),
            )
            
            fig.update_layout(
                title='Ranking Position by Query',
                xaxis_title='Query',
                yaxis_title='Position',
                height=400,
                yaxis=dict(
                    autorange='reversed'  # Reverse y-axis so position 1 is at the top
                )
            )
            
            figs['ranking_comparison'] = fig
    
    # Content analysis visualization
    if 'content_analysis' in data:
        content = data['content_analysis']
        
        if content and 'keyword_density' in content:
            # Create keyword density visualization
            keywords = list(content['keyword_density'].keys())[:15]  # Top 15 keywords
            densities = [content['keyword_density'][k] for k in keywords]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    y=keywords,
                    x=densities,
                    orientation='h',
                    marker_color='#5E35B1',
                    text=[f"{d:.2f}%" for d in densities],
                    textposition='auto'
                )
            )
            
            fig.update_layout(
                title='Keyword Density in Competitor Content',
                xaxis_title='Density (%)',
                yaxis=dict(autorange="reversed"),
                height=400
            )
            
            figs['content_keyword_density'] = fig
    
    return figs
