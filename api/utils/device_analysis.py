
import pandas as pd
from typing import Dict
import numpy as np

def analyze_device_data(df: pd.DataFrame) -> Dict:
    """Analyze device-specific performance metrics."""
    device_metrics = df.groupby('device').agg({
        'clicks': ['sum', 'mean'],
        'impressions': ['sum', 'mean'],
        'ctr': 'mean',
        'position': 'mean'
    }).round(2)
    
    device_trends = {}
    for device in df['device'].unique():
        device_data = df[df['device'] == device]
        device_trends[device] = {
            'total_clicks': int(device_data['clicks'].sum()),
            'total_impressions': int(device_data['impressions'].sum()),
            'avg_position': float(device_data['position'].mean()),
            'avg_ctr': float(device_data['ctr'].mean()),
            'click_share': float(device_data['clicks'].sum() / df['clicks'].sum() * 100),
            'impression_share': float(device_data['impressions'].sum() / df['impressions'].sum() * 100)
        }
    
    return {
        'metrics': device_trends,
        'recommendations': generate_device_recommendations(device_trends)
    }

def generate_device_recommendations(metrics: Dict) -> List[str]:
    """Generate device-specific recommendations based on performance metrics."""
    recommendations = []
    
    for device, data in metrics.items():
        if data['avg_position'] > 10:
            recommendations.append(f"Improve {device} rankings - current average position is {data['avg_position']:.1f}")
        if data['avg_ctr'] < 2:
            recommendations.append(f"Optimize {device} CTR - currently at {data['avg_ctr']:.1f}%")
            
    return recommendations

