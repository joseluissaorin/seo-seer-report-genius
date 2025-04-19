
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Set
from collections import defaultdict
import re

def analyze_serp_features(data_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze SERP features present for keywords."""
    results = {
        'feature_summary': {},
        'feature_opportunities': [],
        'query_features': {},
        'recommendations': []
    }
    
    # Check if we have SERP feature data
    if 'serp_features' in data_frames:
        serp_df = data_frames['serp_features']
        feature_results = _analyze_serp_feature_data(serp_df)
        results.update(feature_results)
    else:
        # If we don't have direct SERP feature data, try to infer from queries
        if 'queries' in data_frames:
            queries_df = data_frames['queries']
            inferred_results = _infer_serp_features_from_queries(queries_df)
            results.update(inferred_results)
    
    return results

def _analyze_serp_feature_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze explicit SERP feature data."""
    results = {
        'feature_summary': {},
        'feature_opportunities': [],
        'query_features': {},
        'recommendations': []
    }
    
    if df.empty:
        return results
    
    # Count occurrences of each feature type
    feature_counts = df['feature'].value_counts().to_dict()
    results['feature_summary'] = feature_counts
    
    # Group by query to see which features appear for each query
    query_features = defaultdict(list)
    for _, row in df.iterrows():
        query = row['query']
        feature = row['feature']
        query_features[query].append(feature)
    
    # Convert to regular dict
    results['query_features'] = {k: list(set(v)) for k, v in query_features.items()}
    
    # Identify opportunities based on position and features
    feature_opportunities = []
    
    for _, row in df.iterrows():
        query = row['query']
        feature = row['feature']
        position = row.get('position', 0)
        
        # Features where we can identify opportunities
        opportunity_features = {
            'featured_snippet': 'Featured Snippet',
            'people_also_ask': 'People Also Ask',
            'knowledge_panel': 'Knowledge Panel',
            'local_pack': 'Local Pack',
            'image_pack': 'Image Pack',
            'video': 'Video',
            'shopping': 'Shopping Results',
            'faq': 'FAQ'
        }
        
        # If this is a feature we track and position is in top 10 but not winning
        if feature in opportunity_features and 1 < position <= 10:
            opportunity = {
                'query': query,
                'feature': opportunity_features[feature],
                'current_position': position,
                'opportunity_score': _calculate_opportunity_score(feature, position)
            }
            feature_opportunities.append(opportunity)
    
    # Sort opportunities by score (highest first)
    results['feature_opportunities'] = sorted(
        feature_opportunities,
        key=lambda x: x['opportunity_score'],
        reverse=True
    )
    
    # Generate recommendations based on feature opportunities
    results['recommendations'] = _generate_serp_recommendations(results)
    
    return results

def _infer_serp_features_from_queries(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer SERP features from query keywords and performance data."""
    results = {
        'feature_summary': {},
        'feature_opportunities': [],
        'query_features': {},
        'recommendations': []
    }
    
    if df.empty:
        return results
    
    # Define patterns for identifying potential SERP features
    feature_patterns = {
        'featured_snippet': [r'\b(how|what|why|when|where|which|who)\b', r'\bdefinition\b', r'\bvs\b'],
        'people_also_ask': [r'\b(how|what|why|when|where|which|who)\b', r'\bcompare\b'],
        'local_pack': [r'\b(near me|nearby|close to|in [a-z\s]+|local)\b', r'\b(store|restaurant|hotel|business)\b'],
        'image_pack': [r'\b(image|picture|photo|wallpaper|logo|icon|design)\b'],
        'video': [r'\b(video|how to|tutorial|guide)\b'],
        'shopping': [r'\b(buy|price|cheap|best|review|top)\b', r'\bfor sale\b'],
        'faq': [r'\b(faq|question|difference between)\b']
    }
    
    # Analyze queries for potential SERP features
    query_features = defaultdict(set)
    feature_counts = defaultdict(int)
    
    for _, row in df.iterrows():
        query = row['query'].lower()
        
        # Check each feature pattern
        for feature, patterns in feature_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    query_features[query].add(feature)
                    feature_counts[feature] += 1
                    break
    
    # Convert to regular dict with lists
    results['query_features'] = {k: list(v) for k, v in query_features.items()}
    results['feature_summary'] = dict(feature_counts)
    
    # Identify potential opportunities
    feature_opportunities = []
    
    for query, features in query_features.items():
        query_data = df[df['query'].str.lower() == query]
        
        if not query_data.empty:
            position = query_data['position'].mean()
            impressions = query_data['impressions'].sum()
            
            for feature in features:
                # If ranking on page 1 but not position 1, this is an opportunity
                if 1 < position <= 10:
                    opportunity = {
                        'query': query,
                        'feature': feature.replace('_', ' ').title(),
                        'current_position': position,
                        'impressions': impressions,
                        'opportunity_score': _calculate_opportunity_score(feature, position, impressions)
                    }
                    feature_opportunities.append(opportunity)
    
    # Sort opportunities by score (highest first)
    results['feature_opportunities'] = sorted(
        feature_opportunities,
        key=lambda x: x['opportunity_score'],
        reverse=True
    )
    
    # Generate recommendations
    results['recommendations'] = _generate_serp_recommendations(results)
    
    return results

def _calculate_opportunity_score(feature: str, position: float, impressions: int = 1000) -> float:
    """Calculate an opportunity score for a SERP feature."""
    # Base score depends on position - higher positions have more potential
    position_score = max(0, 11 - position) / 10 * 100
    
    # Feature importance multiplier
    feature_multipliers = {
        'featured_snippet': 1.5,
        'people_also_ask': 1.2,
        'knowledge_panel': 1.3,
        'local_pack': 1.4,
        'image_pack': 1.1,
        'video': 1.2,
        'shopping': 1.3,
        'faq': 1.2
    }
    
    multiplier = feature_multipliers.get(feature, 1.0)
    
    # Adjust score based on impressions
    impression_factor = min(1.5, max(0.5, (impressions / 1000) * 0.5))
    
    # Calculate final score
    score = position_score * multiplier * impression_factor
    
    return round(score, 1)

def _generate_serp_recommendations(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate recommendations based on SERP feature analysis."""
    recommendations = []
    
    # Get top opportunities
    opportunities = results.get('feature_opportunities', [])
    
    # Generate feature-specific recommendations
    feature_recommendations = {
        'Featured Snippet': {
            'title': 'Optimize for Featured Snippets',
            'description': 'Structure content with clear headings, concise definitions, and well-formatted lists to capture featured snippets.',
            'impact': 'high'
        },
        'People Also Ask': {
            'title': 'Target People Also Ask Boxes',
            'description': 'Research related questions and include them with clear answers in your content to appear in PAA boxes.',
            'impact': 'medium'
        },
        'Knowledge Panel': {
            'title': 'Enhance Entity Information',
            'description': 'Ensure your entity information is consistent across the web and implement proper schema markup.',
            'impact': 'medium'
        },
        'Local Pack': {
            'title': 'Optimize for Local Pack',
            'description': 'Complete your Google Business Profile, get more reviews, and ensure NAP consistency across the web.',
            'impact': 'high'
        },
        'Image Pack': {
            'title': 'Optimize for Image Results',
            'description': 'Use descriptive filenames, alt text, and ensure images are high-quality and properly sized.',
            'impact': 'medium'
        },
        'Video': {
            'title': 'Create Video Content',
            'description': 'Develop video content for key topics and implement proper video schema markup.',
            'impact': 'medium'
        },
        'Shopping Results': {
            'title': 'Optimize for Shopping Results',
            'description': 'Implement product schema and ensure product data feeds are complete and accurate.',
            'impact': 'high'
        },
        'FAQ': {
            'title': 'Implement FAQ Schema',
            'description': 'Add FAQ schema markup to pages with question and answer content to enhance SERP visibility.',
            'impact': 'medium'
        }
    }
    
    # Add recommendations for top 3 feature opportunities
    featured_recommendations = set()
    
    for opportunity in opportunities[:3]:
        feature = opportunity['feature']
        
        if feature in feature_recommendations and feature not in featured_recommendations:
            recommendations.append(feature_recommendations[feature])
            featured_recommendations.add(feature)
    
    # Add general recommendation if we have opportunities
    if opportunities:
        recommendations.append({
            'title': 'Track SERP Features Systematically',
            'description': 'Implement a system to track SERP features for your target keywords and measure your success in capturing these features over time.',
            'impact': 'medium'
        })
    
    return recommendations
