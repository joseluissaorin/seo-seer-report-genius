
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

class CompetitorAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def identify_competitors(self, queries: List[str], max_competitors: int = 5) -> Dict:
        """Identify competitors based on search queries."""
        competitors = {}
        domains_counter = Counter()
        
        def _extract_domain(url):
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain
            except:
                return None
        
        def _search_query(query):
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            local_domains = []
            
            try:
                response = requests.get(search_url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    search_results = soup.select('div.g a')
                    
                    for result in search_results[:10]:  # Consider top 10 results
                        href = result.get('href', '')
                        if href.startswith('http') and not href.startswith('https://support.google.com'):
                            domain = _extract_domain(href)
                            if domain:
                                local_domains.append(domain)
            except Exception as e:
                logging.error(f"Error searching for query '{query}': {str(e)}")
                
            return local_domains
        
        # Use ThreadPoolExecutor to parallelize requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            all_domains = list(executor.map(_search_query, queries[:10]))  # Limit to first 10 queries
        
        # Flatten the list of domains
        all_domains_flat = [domain for sublist in all_domains for domain in sublist]
        
        # Count domain occurrences
        domains_counter.update(all_domains_flat)
        
        # Get the most common domains
        top_competitors = domains_counter.most_common(max_competitors)
        
        # Format the results
        for domain, count in top_competitors:
            competitors[domain] = {
                'appearances': count,
                'share': round(count / len(all_domains_flat) * 100, 2) if all_domains_flat else 0
            }
        
        return competitors
    
    def analyze_competitor_content(self, competitor_url: str) -> Dict:
        """Analyze competitor content to extract insights."""
        analysis = {
            'word_count': 0,
            'keyword_density': {},
            'headings': [],
            'meta_data': {
                'title': '',
                'description': ''
            },
            'links': {
                'internal': 0,
                'external': 0
            }
        }
        
        try:
            response = requests.get(competitor_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                text_content = soup.get_text(separator=' ', strip=True)
                words = re.findall(r'\b\w+\b', text_content.lower())
                analysis['word_count'] = len(words)
                
                # Calculate keyword density
                word_counter = Counter(words)
                total_words = len(words)
                if total_words > 0:
                    for word, count in word_counter.most_common(20):
                        if len(word) > 3:  # Ignore short words
                            analysis['keyword_density'][word] = round(count / total_words * 100, 2)
                
                # Extract headings
                for i in range(1, 7):
                    headings = soup.find_all(f'h{i}')
                    for heading in headings:
                        analysis['headings'].append({
                            'level': i,
                            'text': heading.get_text(strip=True)
                        })
                
                # Extract meta data
                title_tag = soup.find('title')
                if title_tag:
                    analysis['meta_data']['title'] = title_tag.get_text(strip=True)
                
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc:
                    analysis['meta_data']['description'] = meta_desc.get('content', '')
                
                # Count links
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href.startswith('http'):
                        parsed_url = urlparse(href)
                        comp_domain = urlparse(competitor_url).netloc
                        if parsed_url.netloc == comp_domain or parsed_url.netloc == '':
                            analysis['links']['internal'] += 1
                        else:
                            analysis['links']['external'] += 1
                    else:
                        analysis['links']['internal'] += 1
                        
        except Exception as e:
            logging.error(f"Error analyzing competitor content at {competitor_url}: {str(e)}")
        
        return analysis
    
    def compare_rankings(self, queries: List[str], target_domain: str) -> Dict:
        """Compare rankings for target domain vs competitors for specific queries."""
        comparison = {
            'queries': {},
            'summary': {
                'avg_position': 0,
                'top_3_count': 0,
                'top_10_count': 0,
                'not_ranked': 0
            }
        }
        
        total_queries = 0
        total_position = 0
        
        def _check_ranking(query, domain):
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num=30"
            positions = []
            
            try:
                response = requests.get(search_url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    search_results = soup.select('div.g a')
                    
                    for idx, result in enumerate(search_results, 1):
                        href = result.get('href', '')
                        if domain in href:
                            positions.append(idx)
            except Exception as e:
                logging.error(f"Error checking ranking for query '{query}': {str(e)}")
                
            return min(positions) if positions else 0
        
        # Check rankings for each query
        for query in queries[:10]:  # Limit to first 10 queries
            position = _check_ranking(query, target_domain)
            
            comparison['queries'][query] = {
                'position': position,
                'in_top_3': position > 0 and position <= 3,
                'in_top_10': position > 0 and position <= 10
            }
            
            if position > 0:
                total_queries += 1
                total_position += position
                if position <= 3:
                    comparison['summary']['top_3_count'] += 1
                if position <= 10:
                    comparison['summary']['top_10_count'] += 1
            else:
                comparison['summary']['not_ranked'] += 1
        
        # Calculate average position
        comparison['summary']['avg_position'] = round(total_position / total_queries, 2) if total_queries > 0 else 0
        
        return comparison
