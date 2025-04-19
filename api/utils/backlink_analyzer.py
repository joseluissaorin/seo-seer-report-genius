
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import re
import random

def analyze_backlinks(data_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze backlink data for the website."""
    results = {
        'backlink_summary': {},
        'domain_authority': 0,
        'referring_domains': {},
        'anchor_text_analysis': {},
        'recommendations': []
    }
    
    # Check if we have backlink data
    if 'backlinks' in data_frames:
        backlinks_df = data_frames['backlinks']
        backlink_results = _analyze_backlink_data(backlinks_df)
        results.update(backlink_results)
    else:
        # Generate simulated backlink data for demo purposes
        simulated_results = _generate_simulated_backlink_data()
        results.update(simulated_results)
        results['simulated'] = True
    
    return results

def _analyze_backlink_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze real backlink data."""
    results = {
        'backlink_summary': {},
        'domain_authority': 0,
        'referring_domains': {},
        'anchor_text_analysis': {},
        'recommendations': []
    }
    
    if df.empty:
        return results
    
    # Extract basic metrics
    total_backlinks = 0
    if 'backlinks' in df.columns:
        total_backlinks = df['backlinks'].sum()
    
    total_referring_domains = 0
    if 'referring_domains' in df.columns:
        total_referring_domains = df['referring_domains'].sum()
    
    domain_authority = 0
    if 'domain_authority' in df.columns:
        domain_authority = df['domain_authority'].mean()
    
    results['backlink_summary'] = {
        'total_backlinks': int(total_backlinks),
        'total_referring_domains': int(total_referring_domains),
        'backlinks_per_domain': round(total_backlinks / max(1, total_referring_domains), 1)
    }
    
    results['domain_authority'] = round(domain_authority, 1)
    
    # Analyze anchor text if available
    if 'anchor_text' in df.columns:
        anchor_text_counts = df['anchor_text'].value_counts().to_dict()
        
        # Categorize anchor text
        anchor_categories = {
            'branded': 0,
            'exact_match': 0,
            'partial_match': 0,
            'generic': 0,
            'url': 0,
            'other': 0
        }
        
        # TODO: Implement proper anchor text categorization
        # For now, use a simplified approach
        for anchor, count in anchor_text_counts.items():
            if re.search(r'^https?://', anchor):
                anchor_categories['url'] += count
            elif re.search(r'click here|learn more|read more|here', anchor, re.I):
                anchor_categories['generic'] += count
            else:
                anchor_categories['other'] += count
        
        results['anchor_text_analysis'] = {
            'top_anchors': {k: v for k, v in sorted(anchor_text_counts.items(), key=lambda item: item[1], reverse=True)[:10]},
            'categories': anchor_categories
        }
    
    # Analyze referring domains if available
    if 'referring_domain' in df.columns:
        domain_counts = df['referring_domain'].value_counts().to_dict()
        results['referring_domains'] = {
            'top_domains': {k: v for k, v in sorted(domain_counts.items(), key=lambda item: item[1], reverse=True)[:10]}
        }
    
    # Generate recommendations
    results['recommendations'] = _generate_backlink_recommendations(results)
    
    return results

def _generate_simulated_backlink_data() -> Dict[str, Any]:
    """Generate simulated backlink data for demo purposes."""
    # Simulated metrics
    total_backlinks = random.randint(500, 5000)
    total_referring_domains = random.randint(50, total_backlinks // 5)
    domain_authority = random.uniform(20, 50)
    
    # Simulated anchor text distribution
    anchor_categories = {
        'branded': random.randint(20, 40),
        'exact_match': random.randint(10, 30),
        'partial_match': random.randint(15, 35),
        'generic': random.randint(10, 20),
        'url': random.randint(5, 15),
        'other': random.randint(5, 15)
    }
    
    # Normalize to 100%
    total_percent = sum(anchor_categories.values())
    anchor_categories = {k: round((v / total_percent) * 100, 1) for k, v in anchor_categories.items()}
    
    # Simulated top anchors
    sample_anchors = [
        "brand name", "keyword one", "click here", "read more", 
        "product name", "service category", "https://example.com", 
        "learn more", "official site", "best keyword"
    ]
    
    top_anchors = {}
    remaining = total_backlinks
    for anchor in sample_anchors:
        count = random.randint(10, min(remaining, total_backlinks // 6))
        top_anchors[anchor] = count
        remaining -= count
    
    # Simulated top referring domains
    sample_domains = [
        "example-blog.com", "industry-news.com", "partner-site.org", 
        "review-platform.net", "directory.com", "forum-site.com",
        "social-platform.com", "news-site.org", "blogger.com", 
        "business-directory.net"
    ]
    
    top_domains = {}
    remaining = total_referring_domains
    for domain in sample_domains:
        count = random.randint(1, min(remaining, total_referring_domains // 5))
        top_domains[domain] = count
        remaining -= count
    
    # Assemble results
    results = {
        'backlink_summary': {
            'total_backlinks': total_backlinks,
            'total_referring_domains': total_referring_domains,
            'backlinks_per_domain': round(total_backlinks / total_referring_domains, 1)
        },
        'domain_authority': round(domain_authority, 1),
        'anchor_text_analysis': {
            'top_anchors': top_anchors,
            'categories': anchor_categories
        },
        'referring_domains': {
            'top_domains': top_domains
        }
    }
    
    # Generate recommendations
    results['recommendations'] = _generate_backlink_recommendations(results)
    
    return results

def _generate_backlink_recommendations(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate recommendations based on backlink analysis."""
    recommendations = []
    
    # Extract data
    total_backlinks = data.get('backlink_summary', {}).get('total_backlinks', 0)
    total_domains = data.get('backlink_summary', {}).get('total_referring_domains', 0)
    domain_authority = data.get('domain_authority', 0)
    
    # Base recommendations
    if total_backlinks < 1000:
        recommendations.append({
            'title': 'Increase Backlink Quantity',
            'description': 'Focus on building more backlinks through guest posting, content marketing, and outreach campaigns.',
            'impact': 'high'
        })
    
    if total_domains < 100:
        recommendations.append({
            'title': 'Diversify Referring Domains',
            'description': 'Increase the variety of websites linking to your site to improve authority and reduce risk from individual link sources.',
            'impact': 'high'
        })
    
    # Domain authority recommendations
    if domain_authority < 30:
        recommendations.append({
            'title': 'Improve Domain Authority',
            'description': 'Focus on acquiring high-quality backlinks from authoritative websites in your industry to boost overall domain authority.',
            'impact': 'high'
        })
    
    # Anchor text recommendations
    anchor_categories = data.get('anchor_text_analysis', {}).get('categories', {})
    
    generic_percent = anchor_categories.get('generic', 0) + anchor_categories.get('url', 0)
    if generic_percent > 40:
        recommendations.append({
            'title': 'Optimize Anchor Text Distribution',
            'description': 'Work to reduce generic anchor text (like "click here") and URL anchors in favor of more descriptive, keyword-relevant text.',
            'impact': 'medium'
        })
    
    exact_match_percent = anchor_categories.get('exact_match', 0)
    if exact_match_percent > 30:
        recommendations.append({
            'title': 'Diversify Anchor Text Profile',
            'description': 'Reduce over-optimization risk by diversifying your anchor text profile away from too many exact-match keyword anchors.',
            'impact': 'high'
        })
    
    # General recommendations
    recommendations.append({
        'title': 'Create Link-Worthy Content',
        'description': 'Develop high-quality, original research, infographics, or guides that naturally attract backlinks from industry websites.',
        'impact': 'medium'
    })
    
    recommendations.append({
        'title': 'Monitor Competitor Backlinks',
        'description': 'Regularly analyze competitors\' new backlinks to identify opportunities and stay ahead in your link building strategy.',
        'impact': 'medium'
    })
    
    return recommendations
