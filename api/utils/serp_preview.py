
import pandas as pd
from typing import Dict, List, Any
import re
from urllib.parse import urlparse

def generate_serp_previews(data_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Generate SERP previews for pages based on meta data."""
    results = {
        'previews': [],
        'optimization_tips': {},
        'recommendations': []
    }
    
    # Check if we have meta data
    if 'meta_data' in data_frames:
        meta_df = data_frames['meta_data']
        results = _analyze_meta_data(meta_df)
    
    # If we don't have meta data but have query/page data, extract URLs from there
    elif 'queries' in data_frames:
        queries_df = data_frames['queries']
        if 'page' in queries_df.columns:
            # Get unique pages
            unique_pages = queries_df['page'].unique()
            
            # Create placeholder meta data
            pages_data = []
            for page in unique_pages:
                pages_data.append({
                    'url': page,
                    'title': _extract_title_from_url(page),
                    'meta_description': ''
                })
            
            meta_df = pd.DataFrame(pages_data)
            results = _analyze_meta_data(meta_df)
    
    return results

def _analyze_meta_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze meta data for SERP previews."""
    results = {
        'previews': [],
        'optimization_tips': {},
        'recommendations': []
    }
    
    if df.empty:
        return results
    
    # Ensure we have the necessary columns
    required_columns = ['url']
    
    # Check if we have the required columns
    if not all(col in df.columns for col in required_columns):
        return results
    
    # SERP display limits
    title_max_chars = 60
    description_max_chars = 160
    
    # Analysis counters
    title_issues = {
        'missing': 0,
        'too_short': 0,
        'too_long': 0,
        'optimal': 0
    }
    
    description_issues = {
        'missing': 0,
        'too_short': 0,
        'too_long': 0,
        'optimal': 0
    }
    
    # Generate previews for each page
    previews = []
    
    for _, row in df.iterrows():
        url = row['url']
        
        # Extract domain for display
        display_url = _format_display_url(url)
        
        # Get title or generate from URL if missing
        if 'title' in row and pd.notna(row['title']) and row['title'].strip():
            title = row['title'].strip()
            title_status = 'missing' if not title else 'too_short' if len(title) < 30 else 'too_long' if len(title) > title_max_chars else 'optimal'
            title_issues[title_status] += 1
        else:
            title = _extract_title_from_url(url)
            title_issues['missing'] += 1
        
        # Get description or placeholder if missing
        if 'meta_description' in row and pd.notna(row['meta_description']) and row['meta_description'].strip():
            description = row['meta_description'].strip()
            desc_status = 'missing' if not description else 'too_short' if len(description) < 70 else 'too_long' if len(description) > description_max_chars else 'optimal'
            description_issues[desc_status] += 1
        else:
            description = "No meta description provided for this page."
            description_issues['missing'] += 1
        
        # Create preview object
        preview = {
            'url': url,
            'display_url': display_url,
            'title': title,
            'description': description,
            'title_length': len(title),
            'description_length': len(description),
            'title_truncated': len(title) > title_max_chars,
            'description_truncated': len(description) > description_max_chars,
            'displayed_title': title if len(title) <= title_max_chars else title[:title_max_chars - 3] + '...',
            'displayed_description': description if len(description) <= description_max_chars else description[:description_max_chars - 3] + '...',
            'optimization_score': _calculate_meta_optimization_score(title, description)
        }
        
        previews.append(preview)
    
    # Sort previews by optimization score (ascending)
    previews = sorted(previews, key=lambda x: x['optimization_score'])
    
    results['previews'] = previews
    results['optimization_tips'] = {
        'title_issues': title_issues,
        'description_issues': description_issues
    }
    
    # Generate recommendations
    results['recommendations'] = _generate_serp_preview_recommendations(title_issues, description_issues)
    
    return results

def _extract_title_from_url(url: str) -> str:
    """Extract a title from URL if no title is available."""
    try:
        # Parse URL to get path
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Get the last part of the path
        parts = [p for p in path.split('/') if p]
        
        if parts:
            # Take the last part and convert to title
            last_part = parts[-1]
            
            # Remove file extensions if present
            if '.' in last_part:
                last_part = last_part.split('.')[0]
            
            # Replace hyphens and underscores with spaces
            last_part = last_part.replace('-', ' ').replace('_', ' ')
            
            # Capitalize the first letter of each word
            title = ' '.join(word.capitalize() for word in last_part.split())
            
            if title:
                return title
        
        # If path doesn't work, use domain
        domain = parsed_url.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain.capitalize()
    except:
        return "Page Title"

def _format_display_url(url: str) -> str:
    """Format URL for display in SERP preview."""
    try:
        parsed_url = urlparse(url)
        
        # Get scheme and netloc (domain)
        scheme = parsed_url.scheme
        domain = parsed_url.netloc
        
        # Get path with limited length
        path = parsed_url.path
        if len(path) > 30:
            path_parts = path.split('/')
            if len(path_parts) > 3:
                path = '/'.join(path_parts[:2]) + '/...' + path_parts[-1]
        
        # Format display URL
        display_url = f"{domain}{path}"
        
        return display_url
    except:
        return url

def _calculate_meta_optimization_score(title: str, description: str) -> int:
    """Calculate an optimization score for meta title and description."""
    score = 100
    
    # Title scoring
    if not title:
        score -= 40
    else:
        title_length = len(title)
        if title_length < 30:
            score -= 15
        elif title_length > 60:
            score -= 10
        
        # Check for keyword stuffing (keyword appears multiple times in title)
        words = title.lower().split()
        for word in set(words):
            if len(word) > 3 and words.count(word) > 1:
                score -= 5
                break
    
    # Description scoring
    if not description:
        score -= 30
    else:
        desc_length = len(description)
        if desc_length < 70:
            score -= 15
        elif desc_length > 160:
            score -= 10
    
    # Ensure score doesn't go below 0
    return max(0, score)

def _generate_serp_preview_recommendations(title_issues: Dict[str, int], description_issues: Dict[str, int]) -> List[Dict[str, str]]:
    """Generate recommendations based on SERP preview analysis."""
    recommendations = []
    
    # Title recommendations
    if title_issues['missing'] > 0:
        recommendations.append({
            'title': 'Add Missing Meta Titles',
            'description': f"Add meta titles to {title_issues['missing']} pages that are currently missing them.",
            'impact': 'high'
        })
    
    if title_issues['too_long'] > 0:
        recommendations.append({
            'title': 'Optimize Long Titles',
            'description': f"Shorten {title_issues['too_long']} meta titles that exceed the 60-character limit to prevent truncation in search results.",
            'impact': 'medium'
        })
    
    # Description recommendations
    if description_issues['missing'] > 0:
        recommendations.append({
            'title': 'Add Missing Meta Descriptions',
            'description': f"Create compelling meta descriptions for {description_issues['missing']} pages that are currently missing them.",
            'impact': 'high'
        })
    
    if description_issues['too_long'] > 0:
        recommendations.append({
            'title': 'Optimize Long Descriptions',
            'description': f"Shorten {description_issues['too_long']} meta descriptions that exceed the 160-character limit to prevent truncation in search results.",
            'impact': 'medium'
        })
    
    # General recommendations
    recommendations.append({
        'title': 'Include Keywords in Meta Tags',
        'description': 'Add primary keywords to the beginning of your meta titles and descriptions to improve relevance and click-through rates.',
        'impact': 'medium'
    })
    
    recommendations.append({
        'title': 'Add Compelling Call-to-Actions',
        'description': 'Include a clear call-to-action in your meta descriptions to encourage users to click on your search result.',
        'impact': 'medium'
    })
    
    return recommendations
