
import pandas as pd
from typing import Dict, List
import numpy as np

def analyze_geographic_data(df: pd.DataFrame) -> Dict:
    """Analyze geographic performance data."""
    # Basic metrics by country
    country_metrics = df.groupby('country').agg({
        'clicks': ['sum', 'mean'],
        'impressions': ['sum', 'mean'],
        'ctr': 'mean',
        'position': 'mean'
    }).round(2)
    
    # Market penetration analysis
    total_clicks = df['clicks'].sum()
    total_impressions = df['impressions'].sum()
    
    market_analysis = {}
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        market_analysis[country] = {
            'market_share': float(country_data['clicks'].sum() / total_clicks * 100),
            'visibility_share': float(country_data['impressions'].sum() / total_impressions * 100),
            'performance_score': calculate_country_performance_score(country_data)
        }
    
    return {
        'country_metrics': country_metrics.to_dict(),
        'market_analysis': market_analysis,
        'opportunities': identify_geographic_opportunities(df)
    }

def calculate_country_performance_score(df: pd.DataFrame) -> float:
    """Calculate a composite performance score for a country."""
    avg_position = df['position'].mean()
    avg_ctr = df['ctr'].mean()
    
    # Normalize metrics (lower position is better)
    position_score = (20 - min(avg_position, 20)) / 20  # Cap at position 20
    ctr_score = avg_ctr / 100
    
    # Composite score (0-100)
    return float((position_score * 0.6 + ctr_score * 0.4) * 100)

def identify_geographic_opportunities(df: pd.DataFrame) -> List[Dict]:
    """Identify geographic areas with growth potential."""
    opportunities = []
    
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        avg_position = country_data['position'].mean()
        avg_ctr = country_data['ctr'].mean()
        
        if avg_position <= 10 and avg_ctr < df['ctr'].mean():
            opportunities.append({
                'country': country,
                'type': 'ctr_improvement',
                'current_ctr': float(avg_ctr),
                'benchmark_ctr': float(df['ctr'].mean()),
                'potential_impact': float((df['ctr'].mean() - avg_ctr) * country_data['impressions'].sum())
            })
        elif 10 < avg_position <= 20:
            opportunities.append({
                'country': country,
                'type': 'ranking_improvement',
                'current_position': float(avg_position),
                'impressions': int(country_data['impressions'].sum()),
                'potential_impact': 'medium'
            })
    
    return opportunities

