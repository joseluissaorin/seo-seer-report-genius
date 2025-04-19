
import pandas as pd
from typing import Dict, List, Any
import numpy as np

def analyze_mobile_friendliness(data_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze mobile-friendliness based on GSC data."""
    results = {
        'mobile_score': 0,
        'issues_summary': {},
        'page_scores': [],
        'recommendations': []
    }
    
    # Check if we have specific mobile data
    if 'mobile' in data_frames:
        mobile_df = data_frames['mobile']
        results = _analyze_mobile_data(mobile_df)
    
    # If we have device data, use it to supplement the analysis
    if 'devices' in data_frames:
        devices_df = data_frames['devices']
        device_results = _analyze_device_performance(devices_df)
        
        # Combine results
        if not results['recommendations']:
            results['recommendations'] = device_results['recommendations']
        else:
            results['recommendations'].extend(device_results['recommendations'])
        
        # Include device performance metrics
        results['device_performance'] = device_results['metrics']
        
        # Update mobile score if we don't have one yet
        if results['mobile_score'] == 0:
            results['mobile_score'] = device_results['mobile_score']
    
    return results

def _analyze_mobile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze specific mobile data."""
    results = {
        'mobile_score': 0,
        'issues_summary': {},
        'page_scores': [],
        'recommendations': []
    }
    
    if df.empty:
        results['mobile_score'] = 50  # Default score
        return results
    
    # Check for mobile friendliness score column
    if 'mobile_friendly_score' in df.columns:
        avg_score = df['mobile_friendly_score'].mean()
        results['mobile_score'] = avg_score
    
    # Check for mobile usability issues
    if 'mobile_usability' in df.columns:
        # Count pages with issues
        issue_counts = df['mobile_usability'].value_counts().to_dict()
        results['issues_summary'] = issue_counts
        
        # Calculate percentage of pages with issues
        total_pages = df.shape[0]
        pages_with_issues = df[df['mobile_usability'] == 'issues'].shape[0]
        issue_percentage = (pages_with_issues / total_pages) * 100 if total_pages > 0 else 0
        
        # Set mobile score based on issue percentage if not already set
        if results['mobile_score'] == 0:
            if issue_percentage <= 5:
                results['mobile_score'] = 90
            elif issue_percentage <= 15:
                results['mobile_score'] = 75
            elif issue_percentage <= 30:
                results['mobile_score'] = 60
            elif issue_percentage <= 50:
                results['mobile_score'] = 40
            else:
                results['mobile_score'] = 25
    
    # If there are specific mobile issues, analyze them
    if 'issue_type' in df.columns:
        issue_types = df['issue_type'].value_counts().to_dict()
        results['issue_types'] = issue_types
        
        # Generate recommendations based on most common issues
        for issue, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:3]:
            if issue == 'viewport_not_set':
                results['recommendations'].append({
                    'issue': 'Viewport Not Set',
                    'description': 'Add a viewport meta tag to your HTML head: <meta name="viewport" content="width=device-width, initial-scale=1">',
                    'impact': 'high'
                })
            elif issue == 'text_too_small':
                results['recommendations'].append({
                    'issue': 'Text Too Small',
                    'description': 'Use a font size of at least 16px for body text to ensure readability on mobile devices.',
                    'impact': 'medium'
                })
            elif issue == 'clickable_elements_too_close':
                results['recommendations'].append({
                    'issue': 'Clickable Elements Too Close',
                    'description': 'Ensure buttons and links have at least 48x48px touch target size and adequate spacing between them.',
                    'impact': 'high'
                })
            elif issue == 'content_wider_than_screen':
                results['recommendations'].append({
                    'issue': 'Content Wider Than Screen',
                    'description': 'Use relative width values (%, em, rem) instead of fixed pixels and ensure images scale properly.',
                    'impact': 'high'
                })
    
    # Generate page-specific scores
    for _, row in df.iterrows():
        page_data = {
            'url': row.get('page', row.get('url', '')),
            'score': row.get('mobile_friendly_score', 0)
        }
        
        # If no direct score, estimate based on usability
        if page_data['score'] == 0 and 'mobile_usability' in row:
            page_data['score'] = 90 if row['mobile_usability'] == 'pass' else 30
        
        if page_data['url']:
            results['page_scores'].append(page_data)
    
    # Add general recommendations if none exist yet
    if not results['recommendations']:
        results['recommendations'] = [
            {
                'issue': 'General Mobile Optimization',
                'description': 'Ensure your site uses responsive design with flexible layouts and images that adapt to different screen sizes.',
                'impact': 'high'
            },
            {
                'issue': 'Page Speed',
                'description': 'Optimize images, minimize CSS/JS, and leverage browser caching to improve mobile page load times.',
                'impact': 'high'
            },
            {
                'issue': 'Touch Elements',
                'description': 'Make sure all buttons, links and form elements are large enough (min 48x48px) for easy tapping on mobile.',
                'impact': 'medium'
            }
        ]
    
    return results

def _analyze_device_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance across different devices from GSC data."""
    results = {
        'metrics': {},
        'mobile_score': 50,  # Default score
        'recommendations': []
    }
    
    if df.empty:
        return results
    
    # Group by device and calculate metrics
    device_metrics = df.groupby('device').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean',
        'ctr': 'mean'
    }).reset_index()
    
    # Convert to dict for easy access
    for _, row in device_metrics.iterrows():
        device = row['device']
        results['metrics'][device] = {
            'clicks': int(row['clicks']),
            'impressions': int(row['impressions']),
            'position': float(row['position']),
            'ctr': float(row['ctr'])
        }
    
    # Calculate relative performance between mobile and desktop
    if 'mobile' in results['metrics'] and 'desktop' in results['metrics']:
        mobile = results['metrics']['mobile']
        desktop = results['metrics']['desktop']
        
        # Calculate ratios for key metrics
        position_ratio = desktop['position'] / mobile['position'] if mobile['position'] > 0 else 1
        ctr_ratio = mobile['ctr'] / desktop['ctr'] if desktop['ctr'] > 0 else 1
        
        # Mobile score based on relative performance
        position_score = 90 if position_ratio >= 0.9 else 75 if position_ratio >= 0.8 else 60 if position_ratio >= 0.7 else 45
        ctr_score = 90 if ctr_ratio >= 0.9 else 75 if ctr_ratio >= 0.75 else 60 if ctr_ratio >= 0.6 else 45
        
        # Combined mobile score
        results['mobile_score'] = (position_score + ctr_score) / 2
        
        # Generate recommendations based on performance gap
        if position_ratio < 0.8:
            results['recommendations'].append({
                'issue': 'Mobile Ranking Gap',
                'description': f'Your mobile rankings ({mobile["position"]:.1f}) are significantly worse than desktop ({desktop["position"]:.1f}). Focus on improving mobile page experience.',
                'impact': 'high' if position_ratio < 0.7 else 'medium'
            })
        
        if ctr_ratio < 0.8:
            results['recommendations'].append({
                'issue': 'Mobile CTR Gap',
                'description': f'Your mobile CTR ({mobile["ctr"]:.2f}%) is significantly lower than desktop ({desktop["ctr"]:.2f}%). Improve mobile titles and meta descriptions.',
                'impact': 'high' if ctr_ratio < 0.6 else 'medium'
            })
    
    return results
