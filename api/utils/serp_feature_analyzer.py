import pandas as pd
from pytrends.request import TrendReq
from typing import Dict, List, Any, Optional
import logging
import time
from collections import defaultdict
import requests # Import requests for exception handling
import json

# Import the new suggester and AI service
from .google_suggester import GoogleSuggester
from .ai_service import AIService, ModelType # Placeholder import

class KeywordTrendAnalyzer:
    def __init__(self):
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=3, backoff_factor=0.5)
        except Exception as e:
            logging.error(f"Failed to initialize TrendReq: {e}")
            self.pytrends = None

    def analyze_trends_and_context(self, keywords: List[str], timeframe: str = 'today 12-m', geo: str = 'GLOBAL') -> Dict[str, Any]:
        """
        Analyzes keyword trends and context using Google Trends, focusing on 
        interest over time, regional interest, and related queries/topics.
        This replaces the previous SERP feature analysis which is not possible with pytrends.
        """
        if not self.pytrends:
            logging.error("Pytrends is not initialized. Cannot perform trend analysis.")
            return {}

        if not keywords:
            logging.warning("No keywords provided for trend analysis.")
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

        # Limit keywords if necessary
        keywords = keywords[:5] # Process up to 5 keywords at a time for this analysis
        
        trend_data = defaultdict(lambda: {
            'interest_over_time': pd.DataFrame(),
            'interest_by_region': pd.DataFrame(),
            'related_queries_top': pd.DataFrame(),
            'related_queries_rising': pd.DataFrame(),
            'related_topics_top': pd.DataFrame(),
            'related_topics_rising': pd.DataFrame()
        })
        
        max_retries = 3
        base_delay = 2 # seconds
        retries = 0

        batch = [kw for kw in keywords if isinstance(kw, str)] # Ensure all items are strings
        if not batch:
            logging.error("No valid string keywords found in the input batch.")
            return {}

        logging.info(f"Analyzing trends for keywords: {batch}")
        
        while retries < max_retries:
            try:
                self.pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo if geo else 'GLOBAL', gprop='')
                
                # Get interest over time
                iot_df = self.pytrends.interest_over_time()
                if not iot_df.empty and 'isPartial' in iot_df.columns:
                     iot_df = iot_df.drop(columns=['isPartial'])
                
                # Get interest by region
                # Use a default resolution, can be parameterized later if needed
                ibr_df = self.pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True)
                
                # Get related queries
                related_queries_data = self.pytrends.related_queries()
                
                # Get related topics
                related_topics_data = self.pytrends.related_topics()

                # Assign data to the main dictionary
                trend_data['global']['interest_over_time'] = iot_df
                trend_data['global']['interest_by_region'] = ibr_df

                # Process results for each keyword individually where applicable
                for kw in batch:
                    if kw in related_queries_data:
                        if 'top' in related_queries_data[kw] and related_queries_data[kw]['top'] is not None:
                            trend_data[kw]['related_queries_top'] = related_queries_data[kw]['top']
                        if 'rising' in related_queries_data[kw] and related_queries_data[kw]['rising'] is not None:
                            trend_data[kw]['related_queries_rising'] = related_queries_data[kw]['rising']
                    
                    if kw in related_topics_data:
                        if 'top' in related_topics_data[kw] and related_topics_data[kw]['top'] is not None:
                            trend_data[kw]['related_topics_top'] = related_topics_data[kw]['top']
                        if 'rising' in related_topics_data[kw] and related_topics_data[kw]['rising'] is not None:
                            trend_data[kw]['related_topics_rising'] = related_topics_data[kw]['rising']
                
                logging.info(f"Successfully processed trends for: {batch}")
                break # Exit retry loop if successful

            except requests.exceptions.Timeout:
                retries += 1
                logging.warning(f"Timeout occurred analyzing trends for {batch}. Retrying ({retries}/{max_retries})...")
                time.sleep(base_delay * (2 ** retries)) # Exponential backoff
            except Exception as e:
                error_message = str(e)
                if "response retrieval failed" in error_message or "429" in error_message:
                    retries += 1
                    logging.warning(f"Rate limit or retrieval error analyzing trends for {batch}: {error_message}. Retrying ({retries}/{max_retries})...")
                    time.sleep(base_delay * (2 ** retries)) # Exponential backoff
                else:
                    logging.error(f"Unexpected error analyzing trends for {batch}: {e}")
                    # Store the error maybe?
                    trend_data['error'] = str(e)
                    break # Non-retryable error

        if retries == max_retries:
             logging.error(f"Failed to analyze trends for {batch} after {max_retries} retries.")
             trend_data['error'] = f"Failed after {max_retries} retries."

        # Format results
        final_results = {}
        if 'error' in trend_data:
            final_results['error'] = trend_data['error']
        
        # Process global data
        global_data = trend_data.get('global', {})
        final_results['interest_over_time'] = {
            'dates': global_data.get('interest_over_time', pd.DataFrame()).index.strftime('%Y-%m-%d').tolist(),
            'data': {
                col: global_data['interest_over_time'][col].tolist() 
                for col in global_data.get('interest_over_time', pd.DataFrame()).columns
            }
        } if not global_data.get('interest_over_time', pd.DataFrame()).empty else {}
        
        final_results['interest_by_region'] = global_data.get('interest_by_region', pd.DataFrame()).to_dict('index') if not global_data.get('interest_by_region', pd.DataFrame()).empty else {}
        
        # Process keyword-specific data
        final_results['keywords'] = {}
        for kw in batch:
            kw_data = trend_data.get(kw, {})
            final_results['keywords'][kw] = {
                'related_queries_top': kw_data.get('related_queries_top', pd.DataFrame()).to_dict('records') if not kw_data.get('related_queries_top', pd.DataFrame()).empty else [],
                'related_queries_rising': kw_data.get('related_queries_rising', pd.DataFrame()).to_dict('records') if not kw_data.get('related_queries_rising', pd.DataFrame()).empty else [],
                'related_topics_top': kw_data.get('related_topics_top', pd.DataFrame()).to_dict('records') if not kw_data.get('related_topics_top', pd.DataFrame()).empty else [],
                'related_topics_rising': kw_data.get('related_topics_rising', pd.DataFrame()).to_dict('records') if not kw_data.get('related_topics_rising', pd.DataFrame()).empty else []
            }

        # Add recommendations
        recommendations = self._generate_trend_recommendations(final_results)
        final_results['recommendations'] = recommendations

        return final_results

    def _generate_trend_recommendations(self, trend_data: Dict) -> List[Dict[str, str]]:
        """Generate recommendations based on trend analysis."""
    recommendations = []
    
        # Check for significant trends or regional interest
        iot_data = trend_data.get('interest_over_time', {})
        ibr_data = trend_data.get('interest_by_region', {})
        kw_data = trend_data.get('keywords', {})

        if iot_data and 'data' in iot_data:
            for kw, values in iot_data['data'].items():
                if len(values) > 2:
                    # Simple check for upward trend in the last quarter
                    q_len = len(values) // 4
                    if q_len > 1 and sum(values[-q_len:]) / q_len > sum(values[-2*q_len:-q_len]) / q_len:
                         recommendations.append({
                            'title': f'Capitalize on Rising Interest for "{kw}"',
                            'description': f'Interest for "{kw}" appears to be trending upwards recently. Consider increasing focus or creating timely content related to this keyword.',
            'impact': 'high'
                        })
                         break # Only add one recommendation for rising trend

        if ibr_data:
            top_region = max(ibr_data, key=lambda r: sum(ibr_data[r].values())) if ibr_data else None
            if top_region:
                recommendations.append({
                    'title': f'Target High-Interest Regions ({top_region})',
                    'description': f'Keywords show the highest relative interest in {top_region}. Consider tailoring content or marketing efforts for this region if relevant to your goals.',
            'impact': 'medium'
                })
        
        has_rising_queries = any(d.get('related_queries_rising') for d in kw_data.values())
        has_rising_topics = any(d.get('related_topics_rising') for d in kw_data.values())

        if has_rising_queries:
            recommendations.append({
                'title': 'Explore Rising Related Queries',
                'description': 'Investigate the "rising" related queries. These often signal emerging user needs or search trends that can be targeted.',
                'impact': 'medium'
            })
            
        if has_rising_topics:
        recommendations.append({
                'title': 'Monitor Rising Related Topics',
                'description': 'Keep an eye on "rising" related topics. Expanding content into these adjacent areas could capture growing user interest.',
            'impact': 'medium'
        })
        
        if not recommendations and not trend_data.get('error'):
            recommendations.append({
                'title': 'Review Keyword Trends Periodically',
                'description': 'No strong immediate trends detected, but continue to monitor keyword interest over time and by region to spot future opportunities or shifts.',
                'impact': 'low'
            })
        elif trend_data.get('error'):
            recommendations.append({
                'title': 'Trend Analysis Incomplete',
                'description': f'Could not fully complete trend analysis due to an error: {trend_data["error"]}. Results may be partial.',
                'impact': 'N/A'
        })
    
    return recommendations

# Example usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    analyzer = KeywordTrendAnalyzer()
    
    # Test with a sample list of keywords
    test_keywords = ['generative ai', 'large language model', 'seo automation']
    trends = analyzer.analyze_trends_and_context(test_keywords, geo='US')
    
    if trends:
        print("\n--- Keyword Trend Analysis ---")
        if trends.get('interest_over_time'):
            print("  Interest Over Time (Sample):", list(trends['interest_over_time']['data'].keys()))
        if trends.get('interest_by_region'):
            print("  Interest by Region (Top 5):", dict(list(trends['interest_by_region'].items())[:5]))
        
        for kw, data in trends.get('keywords', {}).items():
            print(f"\n  Keyword: {kw}")
            print("    Top Related Queries:", data.get('related_queries_top', [])[:2])
            print("    Rising Related Queries:", data.get('related_queries_rising', [])[:2])
            print("    Top Related Topics:", data.get('related_topics_top', [])[:2])
            print("    Rising Related Topics:", data.get('related_topics_rising', [])[:2])

        print("\n--- Recommendations ---")
        for rec in trends.get('recommendations', []):
            print(f"- {rec['title']}: {rec['description']} (Impact: {rec['impact']})")
    else:
        print("Could not retrieve trend analysis.")
