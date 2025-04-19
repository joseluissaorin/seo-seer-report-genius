
import pandas as pd
from typing import Dict, List
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

def analyze_temporal_patterns(df: pd.DataFrame) -> Dict:
    """Analyze temporal patterns in search performance."""
    df['date'] = pd.to_datetime(df['date'])
    
    # Time series decomposition
    daily_metrics = df.groupby('date').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).reset_index()
    
    decomposition_results = {}
    for metric in ['clicks', 'impressions', 'position']:
        try:
            decomposition = seasonal_decompose(
                daily_metrics[metric], 
                period=7, 
                extrapolate_trend='freq'
            )
            decomposition_results[metric] = {
                'trend': decomposition.trend.tolist(),
                'seasonal': decomposition.seasonal.tolist(),
                'resid': decomposition.resid.tolist()
            }
        except Exception as e:
            print(f"Error decomposing {metric}: {str(e)}")
    
    # Calculate growth rates
    growth_metrics = calculate_growth_metrics(daily_metrics)
    
    # Detect anomalies
    anomalies = detect_anomalies(daily_metrics)
    
    return {
        'decomposition': decomposition_results,
        'growth_metrics': growth_metrics,
        'anomalies': anomalies,
        'weekday_analysis': analyze_weekday_patterns(df)
    }

def calculate_growth_metrics(df: pd.DataFrame) -> Dict:
    """Calculate growth rates and trends."""
    metrics = {}
    for col in ['clicks', 'impressions', 'position']:
        first_half = df[col].iloc[:len(df)//2].mean()
        second_half = df[col].iloc[len(df)//2:].mean()
        growth_rate = ((second_half - first_half) / first_half) * 100
        
        metrics[col] = {
            'growth_rate': float(growth_rate),
            'trend_direction': 'up' if growth_rate > 0 else 'down',
            'volatility': float(df[col].std() / df[col].mean() * 100)
        }
    
    return metrics

def detect_anomalies(df: pd.DataFrame) -> Dict:
    """Detect anomalies using statistical methods."""
    anomalies = {}
    for col in ['clicks', 'impressions', 'position']:
        z_scores = np.abs(stats.zscore(df[col]))
        anomalies[col] = {
            'dates': df.loc[z_scores > 3, 'date'].dt.strftime('%Y-%m-%d').tolist(),
            'values': df.loc[z_scores > 3, col].tolist()
        }
    
    return anomalies

def analyze_weekday_patterns(df: pd.DataFrame) -> Dict:
    """Analyze patterns by day of week."""
    df['weekday'] = df['date'].dt.day_name()
    weekday_metrics = df.groupby('weekday').agg({
        'clicks': ['mean', 'sum'],
        'impressions': ['mean', 'sum'],
        'position': 'mean'
    }).round(2)
    
    return weekday_metrics.to_dict()

