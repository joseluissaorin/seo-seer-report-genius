
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from typing import Dict, List, Tuple
import logging
import time

class KeywordResearcher:
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), retries=2, backoff_factor=0.1)
        
    def research_keyword_trends(self, keywords: List[str], timeframe: str = 'today 12-m') -> Dict:
        """Research keyword trends using Google Trends."""
        results = {}
        
        if not keywords:
            return results
            
        # Process keywords in batches of 5 (Google Trends limitation)
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i+5]
            try:
                self.pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo='', gprop='')
                
                # Get interest over time
                interest_over_time_df = self.pytrends.interest_over_time()
                
                # Get related queries
                related_queries = self.pytrends.related_queries()
                
                # Get related topics
                related_topics = self.pytrends.related_topics()
                
                for keyword in batch:
                    if keyword not in results:
                        results[keyword] = {
                            'trend_data': {},
                            'related_queries': {
                                'top': [],
                                'rising': []
                            },
                            'related_topics': {
                                'top': [],
                                'rising': []
                            }
                        }
                    
                    # Process trend data if available
                    if not interest_over_time_df.empty and keyword in interest_over_time_df.columns:
                        results[keyword]['trend_data'] = {
                            'dates': interest_over_time_df.index.strftime('%Y-%m-%d').tolist(),
                            'values': interest_over_time_df[keyword].tolist()
                        }
                    
                    # Process related queries if available
                    if keyword in related_queries and related_queries[keyword]:
                        for query_type in ['top', 'rising']:
                            if query_type in related_queries[keyword] and not related_queries[keyword][query_type].empty:
                                df = related_queries[keyword][query_type]
                                results[keyword]['related_queries'][query_type] = df.head(10).to_dict('records')
                    
                    # Process related topics if available
                    if keyword in related_topics and related_topics[keyword]:
                        for topic_type in ['top', 'rising']:
                            if topic_type in related_topics[keyword] and not related_topics[keyword][topic_type].empty:
                                df = related_topics[keyword][topic_type]
                                results[keyword]['related_topics'][topic_type] = df.head(10).to_dict('records')
                
                # Sleep to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logging.error(f"Error researching trends for keywords {batch}: {str(e)}")
        
        return results
    
    def get_keyword_suggestions(self, seed_keyword: str) -> List[Dict]:
        """Get keyword suggestions based on a seed keyword."""
        suggestions = []
        
        try:
            # Get suggestions
            self.pytrends.build_payload([seed_keyword], cat=0, timeframe='today 12-m', geo='', gprop='')
            related_queries = self.pytrends.related_queries()
            
            if seed_keyword in related_queries and related_queries[seed_keyword]:
                # Extract top related queries
                if 'top' in related_queries[seed_keyword] and not related_queries[seed_keyword]['top'].empty:
                    top_df = related_queries[seed_keyword]['top']
                    for _, row in top_df.iterrows():
                        suggestions.append({
                            'keyword': row['query'],
                            'score': int(row['value']) if 'value' in row else 0,
                            'type': 'top'
                        })
                
                # Extract rising related queries
                if 'rising' in related_queries[seed_keyword] and not related_queries[seed_keyword]['rising'].empty:
                    rising_df = related_queries[seed_keyword]['rising']
                    for _, row in rising_df.iterrows():
                        suggestions.append({
                            'keyword': row['query'],
                            'score': row['value'] if 'value' in row else 'Breakout',
                            'type': 'rising'
                        })
        
        except Exception as e:
            logging.error(f"Error getting keyword suggestions for {seed_keyword}: {str(e)}")
        
        return suggestions
    
    def analyze_keyword_difficulty(self, keywords: List[str]) -> Dict[str, float]:
        """Estimate keyword difficulty based on Google Trends data."""
        difficulty_scores = {}
        
        if not keywords:
            return difficulty_scores
        
        try:
            # Build the payload for these keywords
            self.pytrends.build_payload(keywords[:5], cat=0, timeframe='today 12-m', geo='', gprop='')
            
            # Get interest over time
            interest_over_time_df = self.pytrends.interest_over_time()
            
            # Calculate difficulty score based on popularity and volatility
            for keyword in keywords[:5]:
                if keyword in interest_over_time_df.columns:
                    values = interest_over_time_df[keyword].values
                    
                    # Basic metrics
                    avg_interest = np.mean(values)
                    volatility = np.std(values)
                    
                    # Normalized difficulty score (0-100)
                    # Higher interest and lower volatility = higher difficulty
                    difficulty = min(100, (avg_interest * (1 - volatility/100)) * 10)
                    difficulty_scores[keyword] = round(max(0, difficulty), 2)
                else:
                    difficulty_scores[keyword] = 0
        
        except Exception as e:
            logging.error(f"Error analyzing keyword difficulty: {str(e)}")
            
        return difficulty_scores
