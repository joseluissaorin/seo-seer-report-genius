import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pytrends.request import TrendReq
import time
from collections import defaultdict
import json

from .google_suggester import GoogleSuggester
# Placeholder AI Service
from .ai_service import AIService, ModelType 

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

class CompetitorLandscapeAnalyzer:
    def __init__(self):
        # Initialize pytrends
        # Consider adding proxy/retry options if running into rate limits frequently
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=3, backoff_factor=0.5)
        except Exception as e:
            logging.error(f"Failed to initialize TrendReq: {e}")
            self.pytrends = None

    def analyze_keyword_landscape(self, keywords: List[str], timeframe: str = 'today 3-m', geo: str = '') -> Dict[str, Any]:
        """
        Analyzes the competitive landscape for a list of keywords using Google Trends.
        Provides insights into related queries and topics.
        """
        if not self.pytrends:
            logging.error("Pytrends is not initialized. Cannot perform landscape analysis.")
            return {}

        if not keywords:
            logging.warning("No keywords provided for landscape analysis.")
            return {}

        # Ensure keywords is a list
        if isinstance(keywords, pd.Series):
            keywords = keywords.tolist()
        elif not isinstance(keywords, list):
            logging.warning(f"Keywords input is not a list or Series, converting: {type(keywords)}")
            try:
                keywords = list(keywords)
            except TypeError:
                logging.error("Could not convert keywords to a list.")
                return {}

        # Limit to top N keywords if too many are provided (optional, adjust N as needed)
        keywords = keywords[:10] # Limit to 10 keywords to keep analysis focused
        
        landscape_data = defaultdict(lambda: {
            'related_queries_top': pd.DataFrame(),
            'related_queries_rising': pd.DataFrame(),
            'related_topics_top': pd.DataFrame(),
            'related_topics_rising': pd.DataFrame()
        })
        
        max_retries = 3
        base_delay = 2 # seconds

        # Process keywords in batches of 5 (pytrends limitation)
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i+5]
            batch = [kw for kw in batch if isinstance(kw, str)] # Ensure all items are strings
            
            if not batch:
                logging.warning(f"Skipping empty or invalid batch at index {i}.")
                continue

            logging.info(f"Analyzing landscape for keywords: {batch}")
            
            retries = 0
            while retries < max_retries:
                try:
                    self.pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo, gprop='')
                    
                    # Get related queries
                    related_queries_data = self.pytrends.related_queries()
                    
                    # Get related topics
                    related_topics_data = self.pytrends.related_topics()

                    # Process results for each keyword in the batch
                    for kw in batch:
                        # Check related queries
                        if kw in related_queries_data and isinstance(related_queries_data[kw], dict):
                            if 'top' in related_queries_data[kw]:
                                top_df = related_queries_data[kw]['top']
                                if isinstance(top_df, pd.DataFrame):
                                    landscape_data[kw]['related_queries_top'] = top_df
                                elif top_df is not None:
                                    logging.warning(f"Related queries 'top' for '{kw}' is not a DataFrame (type: {type(top_df)}). Skipping.")
                                    
                            if 'rising' in related_queries_data[kw]:
                                rising_df = related_queries_data[kw]['rising']
                                if isinstance(rising_df, pd.DataFrame):
                                    landscape_data[kw]['related_queries_rising'] = rising_df
                                elif rising_df is not None:
                                     logging.warning(f"Related queries 'rising' for '{kw}' is not a DataFrame (type: {type(rising_df)}). Skipping.")
                        else:
                            logging.debug(f"Keyword '{kw}' not found in related queries data or data is not a dict.")

                        # Check related topics
                        if kw in related_topics_data and isinstance(related_topics_data[kw], dict):
                            if 'top' in related_topics_data[kw]:
                                top_df = related_topics_data[kw]['top']
                                if isinstance(top_df, pd.DataFrame):
                                     landscape_data[kw]['related_topics_top'] = top_df
                                elif top_df is not None:
                                    logging.warning(f"Related topics 'top' for '{kw}' is not a DataFrame (type: {type(top_df)}). Skipping.")
                                    
                            if 'rising' in related_topics_data[kw]:
                                rising_df = related_topics_data[kw]['rising']
                                if isinstance(rising_df, pd.DataFrame):
                                    landscape_data[kw]['related_topics_rising'] = rising_df
                                elif rising_df is not None:
                                    logging.warning(f"Related topics 'rising' for '{kw}' is not a DataFrame (type: {type(rising_df)}). Skipping.")
                        else:
                            logging.debug(f"Keyword '{kw}' not found in related topics data or data is not a dict.")
                    
                    logging.info(f"Successfully processed batch: {batch}")
                    time.sleep(1) # Short delay between successful batches
                    break # Exit retry loop if successful

                except requests.exceptions.Timeout:
                    retries += 1
                    logging.warning(f"Timeout occurred for keywords {batch}. Retrying ({retries}/{max_retries})...")
                    time.sleep(base_delay * (2 ** retries)) # Exponential backoff
                except Exception as e:
                    # Check for specific pytrends/Google Trends errors if possible
                    # e.g., response status code 429 for rate limiting
                    error_message = str(e)
                    if "response retrieval failed" in error_message or "429" in error_message:
                        retries += 1
                        logging.warning(f"Rate limit or retrieval error for keywords {batch}: {error_message}. Retrying ({retries}/{max_retries})...")
                        time.sleep(base_delay * (2 ** retries)) # Exponential backoff
                    else:
                        logging.error(f"Unexpected error analyzing landscape for keywords {batch}: {e}")
                        break # Non-retryable error

            if retries == max_retries:
                 logging.error(f"Failed to analyze landscape for batch {batch} after {max_retries} retries.")

        # Convert defaultdict to dict and format results
        final_results = {}
        for kw, data in landscape_data.items():
            final_results[kw] = {
                'related_queries_top': data['related_queries_top'].to_dict('records') if not data['related_queries_top'].empty else [],
                'related_queries_rising': data['related_queries_rising'].to_dict('records') if not data['related_queries_rising'].empty else [],
                'related_topics_top': data['related_topics_top'].to_dict('records') if not data['related_topics_top'].empty else [],
                'related_topics_rising': data['related_topics_rising'].to_dict('records') if not data['related_topics_rising'].empty else []
            }
        
        # Add general recommendations based on findings
        recommendations = self._generate_landscape_recommendations(final_results)

        return {
            'landscape_data': final_results,
            'recommendations': recommendations
        }

    def _generate_landscape_recommendations(self, landscape_data: Dict) -> List[Dict[str, str]]:
        """Generate recommendations based on the landscape analysis."""
        recommendations = []
        has_rising_queries = False
        has_rising_topics = False
        
        for kw_data in landscape_data.values():
            if kw_data.get('related_queries_rising'):
                has_rising_queries = True
            if kw_data.get('related_topics_rising'):
                has_rising_topics = True

        if has_rising_queries:
            recommendations.append({
                'title': 'Explore Rising Related Queries',
                'description': 'Investigate the "rising" related queries identified for your target keywords. These often represent emerging trends or new user needs that could be targeted with fresh content.',
                'impact': 'medium'
            })

        if has_rising_topics:
            recommendations.append({
                'title': 'Monitor Rising Related Topics',
                'description': 'Pay attention to the "rising" related topics. Expanding content into these areas or incorporating them into existing content can capture growing interest.',
                'impact': 'medium'
            })

        if landscape_data:
             recommendations.append({
                'title': 'Analyze Top Queries/Topics',
                'description': 'Review the "top" related queries and topics. Ensure your current content adequately addresses these established areas of interest related to your core keywords.',
                'impact': 'high'
            })
             recommendations.append({
                'title': 'Refine Keyword Strategy',
                'description': 'Use the related queries and topics to refine your keyword strategy. Identify potential long-tail keywords, question-based queries, and broader thematic areas for content development.',
                'impact': 'high'
            })
        
        if not recommendations:
             recommendations.append({
                'title': 'Expand Keyword Input',
                'description': 'Consider providing a broader or more diverse set of initial keywords to gain richer insights into the competitive landscape and related trends.',
                'impact': 'low'
            })

        return recommendations

# Example usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    analyzer = CompetitorLandscapeAnalyzer()
    
    # Test with a sample list of keywords
    test_keywords = ['seo reporting tool', 'google trends api', 'keyword research']
    landscape = analyzer.analyze_keyword_landscape(test_keywords)
    
    if landscape:
        print("\n--- Competitor Landscape Analysis ---")
        for kw, data in landscape.get('landscape_data', {}).items():
            print(f"\nKeyword: {kw}")
            print("  Top Related Queries:", data.get('related_queries_top', [])[:3]) # Show top 3
            print("  Rising Related Queries:", data.get('related_queries_rising', [])[:3])
            print("  Top Related Topics:", data.get('related_topics_top', [])[:3])
            print("  Rising Related Topics:", data.get('related_topics_rising', [])[:3])
        
        print("\n--- Recommendations ---")
        for rec in landscape.get('recommendations', []):
            print(f"- {rec['title']}: {rec['description']} (Impact: {rec['impact']})")
    else:
        print("Could not retrieve landscape analysis.")

class SimplifiedCompetitorAnalysisService:
    """Performs simplified competitor analysis using suggestions and AI."""

    def __init__(self, lang: str = 'en', country_code: str = 'us'):
        self.suggester = GoogleSuggester(lang=lang, country_code=country_code)
        self.ai_service = AIService() # Placeholder
        
    async def analyze_competitors(self, 
                                  seed_keyword: str, 
                                  api_key: str, 
                                  model: ModelType) -> Dict[str, Any]:
        """
        Identifies potential competitors and analyzes the competitive landscape 
        using Google suggestions and AI interpretation.
        """
        if not seed_keyword:
            return {"error": "Seed keyword cannot be empty."}
            
        logging.info(f"Starting simplified competitor analysis for: '{seed_keyword}'")
        
        # 1. Get comparison suggestions to find potential competitor names/types
        comparison_suggestions = self.suggester.get_comparison_suggestions(seed_keyword)
        logging.info(f"Found {len(comparison_suggestions)} comparison suggestions.")
        
        # 2. Get general suggestions to understand the topic space
        basic_suggestions = self.suggester.get_suggestions(seed_keyword)
        
        # 3. Use AI to interpret suggestions and analyze the landscape
        prompt = f"""
        Analyze the competitive landscape for the seed keyword "{seed_keyword}".
        Consider the following Google suggestions:
        - Comparison suggestions (like 'vs', potentially naming competitors): {json.dumps(comparison_suggestions[:20])}
        - General suggestions: {json.dumps(basic_suggestions[:10])}

        Return a JSON object with the following structure:
        {{
            "seed_keyword": "{seed_keyword}",
            "potential_competitors": [
                {{
                    "name_or_type": "Identified competitor name or type (e.g., Blog, SaaS Tool, E-commerce Site)",
                    "reasoning": "Why this is considered a competitor based on suggestions or keyword analysis."
                }}, ...
            ],
            "competitive_landscape_summary": {{
                "estimated_difficulty": "Low | Medium | High | Very High", // AI's estimation of competitiveness
                "key_competitive_factors": ["Factor 1 (e.g., Content Quality)", "Factor 2 (e.g., Brand Authority)", ...],
                "potential_angles_for_user": ["Angle 1 (e.g., Focus on niche audience)", "Angle 2 (e.g., Offer unique feature)", ...]
            }}
        }}

        Rules:
        1. Identify 3-5 key potential competitors or competitor types.
        2. Provide a brief reasoning for each identified competitor.
        3. Estimate the overall competitive difficulty for this keyword.
        4. List the most important factors for succeeding against competitors in this space.
        5. Suggest unique angles or strategies the user could take.
        6. Ensure the output is valid JSON.
        """
        
        try:
            analysis_result = await self.ai_service.generate_structured_response(
                prompt=prompt,
                api_key=api_key,
                model=model,
                response_format='json_object'
            )
            
            # Basic validation
            if not isinstance(analysis_result, dict) or 'potential_competitors' not in analysis_result or 'competitive_landscape_summary' not in analysis_result:
                logging.error(f"AI competitor analysis result has unexpected structure: {analysis_result}")
                raise ValueError("AI competitor analysis returned data in an unexpected format.")
                
            logging.info("Successfully received and parsed AI competitor analysis.")
            analysis_result['comparison_suggestions_used'] = comparison_suggestions[:20] # Add used suggestions for context
            return analysis_result
            
        except Exception as e:
            logging.error(f"AI competitor analysis failed for '{seed_keyword}': {e}")
            return {
                "seed_keyword": seed_keyword,
                "potential_competitors": [],
                "competitive_landscape_summary": {},
                "error": f"AI analysis failed: {str(e)}"
            }

# Placeholder for AIService - replace with actual implementation
# (Assuming it's defined elsewhere, e.g., in api/utils/ai_service.py)
# from .ai_service import AIService, ModelType 
class AIService:
     async def generate_structured_response(self, prompt: str, api_key: str, model: ModelType, response_format: str) -> Dict[str, Any]:
         logging.warning("Using placeholder AIService. Implement actual AI call.")
         time.sleep(0.5)
         # Mock response based on prompt structure
         mock_analysis = {
             "seed_keyword": "placeholder_seed",
             "potential_competitors": [
                 {"name_or_type": "Major SEO Blog (e.g., Moz Blog)", "reasoning": "High authority content often appears in related searches."}, 
                 {"name_or_type": "SEO Software Tool (e.g., SEMrush)", "reasoning": "Tool pages rank for analysis-related keywords."}], 
             "competitive_landscape_summary": {
                 "estimated_difficulty": "High", 
                 "key_competitive_factors": ["Content Depth", "Domain Authority", "Backlink Profile"], 
                 "potential_angles_for_user": ["Focus on a specific niche within the topic", "Create highly practical, actionable guides"]
                 }
             }
         # Extract seed keyword from prompt for mock response
         match = re.search(r'seed keyword "(.*?)"' , prompt)
         if match:
             mock_analysis['seed_keyword'] = match.group(1)
         return mock_analysis
import re

# Example Usage (for testing)
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    analyzer = SimplifiedCompetitorAnalysisService()
    
    api_key = "YOUR_API_KEY"
    model_to_use: ModelType = "chatgpt"
    seed = "link building strategies"
    
    print(f"\n--- Starting Competitor Analysis for '{seed}' ---")
    results = await analyzer.analyze_competitors(seed, api_key, model_to_use)
    
    print("\n--- Analysis Results ---")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
