
import pandas as pd
from typing import Dict, List
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
    
    # Find the device with the highest CTR to use as a benchmark
    best_device = max(metrics.items(), key=lambda x: x[1]['avg_ctr'])[0]
    best_ctr = metrics[best_device]['avg_ctr']
    
    # Find the device with the best position
    best_position_device = min(metrics.items(), key=lambda x: x[1]['avg_position'])[0]
    best_position = metrics[best_position_device]['avg_position']
    
    for device, data in metrics.items():
        # Position-based recommendations
        if data['avg_position'] > 10:
            recommendations.append(f"Improve {device} rankings - current average position is {data['avg_position']:.1f}")
        elif data['avg_position'] > best_position + 2:
            recommendations.append(f"Consider improving {device} content and optimization - position ({data['avg_position']:.1f}) lags behind {best_position_device} ({best_position:.1f})")
        
        # CTR-based recommendations
        if data['avg_ctr'] < 2:
            recommendations.append(f"Optimize {device} CTR - currently at {data['avg_ctr']:.1f}%")
        elif data['avg_ctr'] < best_ctr * 0.7 and data['impression_share'] > 20:
            recommendations.append(f"Enhance {device} title and meta descriptions to improve CTR ({data['avg_ctr']:.1f}%) compared to {best_device} ({best_ctr:.1f}%)")
        
        # Traffic share recommendations
        if data['impression_share'] > 30 and data['click_share'] < data['impression_share'] * 0.7:
            recommendations.append(f"Prioritize {device} optimization - high impression share ({data['impression_share']:.1f}%) but underperforming in clicks ({data['click_share']:.1f}%)")
            
    # Mobile-specific recommendations
    if 'mobile' in metrics and 'desktop' in metrics:
        mobile_metrics = metrics['mobile']
        desktop_metrics = metrics['desktop']
        
        if mobile_metrics['impression_share'] > desktop_metrics['impression_share'] and mobile_metrics['avg_position'] > desktop_metrics['avg_position']:
            recommendations.append("Improve mobile SEO - mobile searches are more common but rankings are lower than desktop")
        
        if mobile_metrics['avg_ctr'] < desktop_metrics['avg_ctr'] * 0.8:
            recommendations.append(f"Optimize for mobile engagement - mobile CTR ({mobile_metrics['avg_ctr']:.1f}%) is significantly lower than desktop ({desktop_metrics['avg_ctr']:.1f}%)")
    
    # Desktop-specific recommendations
    if 'desktop' in metrics and desktop_metrics['impression_share'] < 30 and desktop_metrics['avg_position'] > 5:
        recommendations.append("Improve desktop visibility - currently underrepresented in search results with room for ranking improvement")
    
    # Tablet-specific recommendations
    if 'tablet' in metrics:
        tablet_metrics = metrics['tablet']
        if tablet_metrics['impression_share'] > 5 and tablet_metrics['avg_position'] > 8:
            recommendations.append(f"Don't ignore tablet optimization - significant traffic source with average position {tablet_metrics['avg_position']:.1f}")
            
    return recommendations
