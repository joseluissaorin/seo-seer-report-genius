
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict

def detect_keyword_cannibalization(data_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Detect keyword cannibalization issues in the website."""
    results = {
        'cannibalization_issues': [],
        'severity_summary': {
            'high': 0,
            'medium': 0,
            'low': 0
        },
        'recommendations': []
    }
    
    # Check if we have query data with pages
    if 'queries' not in data_frames:
        return results
    
    queries_df = data_frames['queries']
    
    # Check if we have both query and page columns
    if 'query' not in queries_df.columns or 'page' not in queries_df.columns:
        return results
    
    # Group data by query
    query_groups = queries_df.groupby('query')
    
    # Find queries that appear with multiple pages
    for query, group in query_groups:
        unique_pages = group['page'].unique()
        
        # If a query appears on multiple pages, check for cannibalization
        if len(unique_pages) > 1:
            # Get performance metrics for each page
            page_metrics = {}
            for page in unique_pages:
                page_data = group[group['page'] == page]
                
                # Calculate average metrics
                avg_position = page_data['position'].mean()
                total_clicks = page_data['clicks'].sum()
                total_impressions = page_data['impressions'].sum()
                avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
                
                page_metrics[page] = {
                    'position': avg_position,
                    'clicks': total_clicks,
                    'impressions': total_impressions,
                    'ctr': avg_ctr
                }
            
            # Calculate query metrics
            total_query_impressions = sum(metrics['impressions'] for metrics in page_metrics.values())
            best_position = min(metrics['position'] for metrics in page_metrics.values())
            worst_position = max(metrics['position'] for metrics in page_metrics.values())
            position_difference = worst_position - best_position
            
            # Determine severity of cannibalization
            severity = _determine_cannibalization_severity(
                total_query_impressions, 
                position_difference, 
                len(unique_pages)
            )
            
            # Only include if severity is not none
            if severity != 'none':
                # Sort pages by position (best first)
                sorted_pages = sorted(
                    [(page, metrics) for page, metrics in page_metrics.items()],
                    key=lambda x: x[1]['position']
                )
                
                primary_page = sorted_pages[0][0]
                competing_pages = [page for page, _ in sorted_pages[1:]]
                
                # Create cannibalization issue entry
                issue = {
                    'query': query,
                    'severity': severity,
                    'primary_page': primary_page,
                    'competing_pages': competing_pages,
                    'metrics': {
                        'best_position': best_position,
                        'worst_position': worst_position,
                        'position_difference': position_difference,
                        'total_impressions': total_query_impressions
                    }
                }
                
                results['cannibalization_issues'].append(issue)
                results['severity_summary'][severity] += 1
    
    # Sort issues by severity and then by total impressions
    results['cannibalization_issues'] = sorted(
        results['cannibalization_issues'],
        key=lambda x: (
            0 if x['severity'] == 'high' else 1 if x['severity'] == 'medium' else 2,
            -x['metrics']['total_impressions']
        )
    )
    
    # Generate recommendations
    results['recommendations'] = _generate_cannibalization_recommendations(results['cannibalization_issues'])
    
    return results

def _determine_cannibalization_severity(impressions: int, position_diff: float, page_count: int) -> str:
    """Determine the severity of a cannibalization issue."""
    # High severity: high impressions, significant position difference, multiple pages
    if impressions >= 1000 and position_diff <= 5 and page_count >= 3:
        return 'high'
    # Medium severity: moderate impressions, moderate position difference
    elif impressions >= 500 and position_diff <= 10:
        return 'medium'
    # Low severity: lower impressions or larger position difference
    elif impressions >= 100 and position_diff <= 15:
        return 'low'
    # No significant cannibalization
    else:
        return 'none'

def _generate_cannibalization_recommendations(issues: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Generate recommendations for fixing cannibalization issues."""
    recommendations = []
    
    # General recommendation
    if issues:
        recommendations.append({
            'title': 'Consolidate Content for Competing Pages',
            'description': 'Identify the best-performing page for each query and consolidate content from competing pages to strengthen the primary page.',
            'impact': 'high'
        })
    
    # Specific recommendations for high-severity issues
    high_severity_issues = [issue for issue in issues if issue['severity'] == 'high']
    
    if high_severity_issues:
        # Get the top 3 highest impression issues
        for issue in high_severity_issues[:3]:
            primary_page = issue['primary_page']
            competing_pages = issue['competing_pages']
            competing_pages_text = ", ".join(competing_pages[:2])
            
            if len(competing_pages) > 2:
                competing_pages_text += f" and {len(competing_pages) - 2} more"
            
            recommendations.append({
                'title': f'Resolve Cannibalization for "{issue["query"]}"',
                'description': f'Make {primary_page} the primary page for this query. Set up 301 redirects or add canonical tags on {competing_pages_text} pointing to the primary page.',
                'impact': 'high'
            })
    
    # Add implementation recommendations
    recommendations.append({
        'title': 'Implement Proper Internal Linking',
        'description': 'Ensure internal links for a specific keyword consistently point to the designated primary page to reinforce its relevance for that query.',
        'impact': 'medium'
    })
    
    recommendations.append({
        'title': 'Set Up Content Planning System',
        'description': 'Implement a content planning system that maps target keywords to specific URLs to prevent future cannibalization issues.',
        'impact': 'medium'
    })
    
    return recommendations
