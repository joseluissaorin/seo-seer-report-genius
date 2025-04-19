
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Set
from collections import defaultdict
import re
import random

def analyze_content_gaps(data_frames: Dict[str, pd.DataFrame], competitor_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze content gaps compared to competitors."""
    results = {
        'gap_summary': {},
        'topic_opportunities': [],
        'keyword_gaps': {},
        'content_recommendations': []
    }
    
    # Check if we have query data and competitor analysis
    if 'queries' in data_frames:
        queries_df = data_frames['queries']
        
        # If we have competitor analysis, use it. Otherwise, try to infer competitors.
        if competitor_data and 'competitors' in competitor_data:
            competitors = competitor_data['competitors']
        else:
            # Infer competitors from queries data if 'page' column exists
            competitors = _infer_competitors_from_queries(queries_df)
        
        # Analyze gaps based on available data
        if competitors:
            gap_results = _analyze_query_based_gaps(queries_df, competitors)
            results.update(gap_results)
    
    return results

def _infer_competitors_from_queries(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer competitors from query data."""
    competitors = {}
    
    # If we don't have position or query data, return empty
    if 'query' not in df.columns or 'position' not in df.columns:
        return competitors
    
    # Generate simulated competitors for demo
    # In a real implementation, this would analyze the actual data
    sample_competitors = [
        "competitor1.com", "topcompetitor.net", "industry-leader.org", 
        "rival-site.com", "alternative-solution.net"
    ]
    
    # Assign random metrics to competitors
    for comp in sample_competitors:
        competitors[comp] = {
            'appearances': random.randint(10, 50),
            'share': round(random.uniform(2, 15), 1),
            'avg_position': round(random.uniform(1, 8), 1)
        }
    
    # Normalize shares to sum to 100
    total_share = sum(comp['share'] for comp in competitors.values())
    for comp in competitors.values():
        comp['share'] = round((comp['share'] / total_share) * 100, 1)
    
    return competitors

def _analyze_query_based_gaps(df: pd.DataFrame, competitors: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze content gaps based on query performance."""
    results = {
        'gap_summary': {},
        'topic_opportunities': [],
        'keyword_gaps': {},
        'content_recommendations': []
    }
    
    # If we don't have query data, return empty results
    if df.empty or 'query' not in df.columns:
        return results
    
    # Group queries by performance
    queries_by_performance = {
        'top_performing': [],
        'competing': [],
        'underperforming': [],
        'missing': []
    }
    
    # Analyze query performance
    for _, row in df.iterrows():
        query = row['query']
        position = float(row['position']) if 'position' in row else 20
        
        if position <= 3:
            queries_by_performance['top_performing'].append(query)
        elif position <= 10:
            queries_by_performance['competing'].append(query)
        else:
            queries_by_performance['underperforming'].append(query)
    
    # Simulate missing queries (queries where competitors rank but we don't)
    # In a real implementation, this would compare actual competitor data
    sample_missing_queries = [
        "industry term we don't target", "competitor specific product", 
        "alternative solution keyword", "related service we don't offer",
        "complementary tool search", "competitor brand term",
        "technical term in industry", "emerging trend keyword",
        "related product category", "competitor feature keyword"
    ]
    
    # Add random selection of missing queries
    num_missing = random.randint(5, 10)
    queries_by_performance['missing'] = random.sample(sample_missing_queries, min(num_missing, len(sample_missing_queries)))
    
    # Generate topic clusters from queries
    topic_clusters = _generate_topic_clusters(queries_by_performance)
    
    # Identify gaps and opportunities
    gap_summary = {
        'total_gaps': len(topic_clusters['gaps']),
        'total_opportunities': len(topic_clusters['opportunities']),
        'competitor_advantage_topics': len(topic_clusters['competitor_advantage'])
    }
    
    # Score opportunities
    scored_opportunities = []
    for topic in topic_clusters['opportunities'] + topic_clusters['gaps']:
        opportunity = {
            'topic': topic['label'],
            'related_keywords': topic['keywords'][:5],  # Limit to 5 related keywords
            'competitor_coverage': random.randint(2, len(competitors)),  # Simulated competitor coverage
            'current_avg_position': topic.get('avg_position', 0),
            'search_volume': random.randint(500, 5000),  # Simulated search volume
            'opportunity_score': _calculate_opportunity_score(topic, competitors)
        }
        scored_opportunities.append(opportunity)
    
    # Sort opportunities by score
    scored_opportunities = sorted(scored_opportunities, key=lambda x: x['opportunity_score'], reverse=True)
    
    # Compile results
    results['gap_summary'] = gap_summary
    results['topic_opportunities'] = scored_opportunities
    results['keyword_gaps'] = {
        'missing_keywords': len(queries_by_performance['missing']),
        'underperforming_keywords': len(queries_by_performance['underperforming']),
        'sample_missing': queries_by_performance['missing'][:5]  # Limited sample
    }
    
    # Generate content recommendations
    results['content_recommendations'] = _generate_content_recommendations(scored_opportunities[:5])
    
    return results

def _generate_topic_clusters(queries_by_performance: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
    """Generate topic clusters from queries."""
    # In a real implementation, this would use NLP clustering
    # Here we use a simplified approach for demo purposes
    
    topic_clusters = {
        'opportunities': [],  # Topics where we have content but could improve
        'gaps': [],  # Topics where we have no content but competitors do
        'strengths': [],  # Topics where we outperform competitors
        'competitor_advantage': []  # Topics where competitors significantly outperform us
    }
    
    # Generate opportunity topics from underperforming queries
    underperforming = queries_by_performance['underperforming']
    if underperforming:
        # Simulate topic clusters (in reality would use NLP)
        unique_terms = set()
        for query in underperforming:
            unique_terms.update(query.lower().split())
        
        # Create simulated topic clusters
        num_opportunities = min(5, len(underperforming) // 3 + 1)
        for i in range(num_opportunities):
            # Create a random topic label
            topic_terms = random.sample([t for t in unique_terms if len(t) > 3], min(2, len(unique_terms)))
            topic_label = " ".join(topic_terms).title()
            
            # Sample related queries
            related_queries = random.sample(underperforming, min(5, len(underperforming)))
            avg_position = round(random.uniform(12, 30), 1)
            
            topic = {
                'label': topic_label,
                'keywords': related_queries,
                'avg_position': avg_position
            }
            
            topic_clusters['opportunities'].append(topic)
    
    # Generate gap topics from missing queries
    missing = queries_by_performance['missing']
    if missing:
        num_gaps = min(3, len(missing) // 2 + 1)
        for i in range(num_gaps):
            # Create a random topic label
            if len(missing) >= 2:
                base_query = random.choice(missing)
                topic_label = " ".join([w.capitalize() for w in base_query.split()[:2]])
            else:
                topic_label = "Missing Topic " + str(i+1)
            
            # Sample related queries
            related_queries = random.sample(missing, min(4, len(missing)))
            
            topic = {
                'label': topic_label,
                'keywords': related_queries
            }
            
            topic_clusters['gaps'].append(topic)
    
    # Generate strength topics from top performing
    top_performing = queries_by_performance['top_performing']
    if top_performing:
        num_strengths = min(2, len(top_performing) // 3 + 1)
        for i in range(num_strengths):
            # Create a random topic label
            if len(top_performing) >= 2:
                base_query = random.choice(top_performing)
                topic_label = " ".join([w.capitalize() for w in base_query.split()[:2]])
            else:
                topic_label = "Strength Topic " + str(i+1)
            
            # Sample related queries
            related_queries = random.sample(top_performing, min(4, len(top_performing)))
            avg_position = round(random.uniform(1, 3), 1)
            
            topic = {
                'label': topic_label,
                'keywords': related_queries,
                'avg_position': avg_position
            }
            
            topic_clusters['strengths'].append(topic)
    
    # Generate competitor advantage topics
    # These would normally be derived from actual competitor analysis
    competitor_topics = [
        "Advanced Industry Solutions",
        "Enterprise Integration Options",
        "Technical Documentation Center",
        "Industry Compliance Resources"
    ]
    
    for topic_name in competitor_topics:
        # Generate simulated keywords
        keywords = [f"{topic_name.lower()} for beginners",
                   f"best {topic_name.lower()}",
                   f"{topic_name.lower()} tutorial",
                   f"{topic_name.lower()} guide"]
        
        topic = {
            'label': topic_name,
            'keywords': keywords
        }
        
        topic_clusters['competitor_advantage'].append(topic)
    
    return topic_clusters

def _calculate_opportunity_score(topic: Dict[str, Any], competitors: Dict[str, Any]) -> float:
    """Calculate an opportunity score for a content gap topic."""
    # Base score depends on search volume (simulated)
    search_volume = random.randint(100, 5000)
    volume_score = min(100, (search_volume / 50))
    
    # Competition factor
    competitor_coverage = topic.get('competitor_coverage', 0)
    competition_factor = min(1.5, max(0.7, (competitor_coverage / len(competitors)) * 1.2))
    
    # Current position factor (lower is better, 0 means we don't rank)
    current_position = topic.get('avg_position', 0)
    position_factor = 1.0
    if current_position > 0:
        position_factor = max(0.5, min(1.5, (30 - current_position) / 20))
        
    # Calculate final score
    score = volume_score * competition_factor * position_factor
    
    return round(score, 1)

def _generate_content_recommendations(opportunities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Generate content recommendations based on identified opportunities."""
    recommendations = []
    
    # Add recommendations based on top opportunities
    for i, opportunity in enumerate(opportunities):
        topic = opportunity['topic']
        keywords = opportunity['related_keywords']
        
        # Create a recommendation
        recommendation_type = "New Content Creation" if opportunity.get('current_avg_position', 0) == 0 else "Content Improvement"
        keywords_str = ", ".join(keywords[:3])
        
        recommendation = {
            'title': f"{recommendation_type}: {topic}",
            'description': f"{'Create new content' if recommendation_type.startswith('New') else 'Improve existing content'} targeting the topic '{topic}' and related keywords ({keywords_str}).",
            'impact': 'high' if i < 2 else 'medium',
            'estimated_search_volume': opportunity.get('search_volume', 'Unknown')
        }
        
        recommendations.append(recommendation)
    
    # Add general recommendations
    recommendations.append({
        'title': 'Develop a Content Gap Strategy',
        'description': 'Create a systematic approach to regularly identify and address content gaps compared to competitors.',
        'impact': 'medium'
    })
    
    recommendations.append({
        'title': 'Update Existing Content',
        'description': 'Enhance underperforming content by expanding depth, adding multimedia, and updating with current information.',
        'impact': 'medium'
    })
    
    return recommendations
