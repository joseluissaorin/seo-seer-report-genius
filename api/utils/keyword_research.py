import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor

# Import the new suggester
from .google_suggester import GoogleSuggester
# Assuming an AI service utility exists similar to the React project
# We might need to create/adapt this later
from .ai_service import AIService, ModelType # Placeholder import

class KeywordResearchService:
    """Performs keyword research using Google Suggest and AI enrichment."""

    def __init__(self, lang: str = 'en', country_code: str = 'us'):
        self.suggester = GoogleSuggester(lang=lang, country_code=country_code)
        self.ai_service = AIService() # Placeholder initialization
        # Optional: Keep pytrends for basic trend data if needed, handle potential errors
            try:
            self.pytrends = TrendReq(hl=f'{lang}-{country_code.upper()}', tz=360, timeout=(10, 25), retries=2, backoff_factor=0.1)
        except Exception as e:
            logging.warning(f"Failed to initialize pytrends: {e}. Trend data might be unavailable.")
            self.pytrends = None
        
    def _get_combined_suggestions(self, keyword: str) -> List[str]:
        """Combines suggestions from different methods."""
        suggestions = set()
        
        # Use ThreadPoolExecutor to fetch suggestions in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.suggester.get_suggestions, keyword),
                executor.submit(self.suggester.get_alphabetical_suggestions, keyword),
                executor.submit(self.suggester.get_question_suggestions, keyword),
                executor.submit(self.suggester.get_comparison_suggestions, keyword)
            ]
            for future in futures:
                try:
                    suggestions.update(future.result())
                except Exception as e:
                    logging.error(f"Error getting suggestions for '{keyword}' in thread: {e}")

        # Add the original keyword itself
        suggestions.add(keyword)
        
        # Basic cleaning (optional: more sophisticated cleaning can be added)
        cleaned_suggestions = [s.strip() for s in suggestions if s.strip()]
        
        logging.info(f"Generated {len(cleaned_suggestions)} unique suggestions for '{keyword}'.")
        return sorted(list(set(cleaned_suggestions)))

    def get_trend_data(self, keywords: List[str], timeframe: str = 'today 12-m') -> Optional[pd.DataFrame]:
        """Attempts to get interest over time data using pytrends."""
        if not self.pytrends or not keywords:
            return None
            
        # Limit to 5 keywords due to pytrends limitation
        keywords_to_fetch = keywords[:5]
        
        try:
            self.pytrends.build_payload(keywords_to_fetch, cat=0, timeframe=timeframe, geo='', gprop='')
            interest_df = self.pytrends.interest_over_time()
            if not interest_df.empty:
                # Remove the 'isPartial' column if it exists
                if 'isPartial' in interest_df.columns:
                    interest_df = interest_df.drop(columns=['isPartial'])
                return interest_df
            else:
                return None
            except Exception as e:
            logging.error(f"Pytrends error fetching trends for {keywords_to_fetch}: {e}")
            # Handle specific errors like 429 (Too Many Requests) or 400 (Bad Request) if needed
            if "429" in str(e) or "400" in str(e):
                logging.warning("Pytrends rate limit or bad request error.")
            return None

    async def perform_ai_keyword_analysis(self,
                                          seed_keyword: str,
                                          suggestions: List[str],
                                          api_key: str,
                                          model: ModelType) -> Dict[str, Any]:
        """Uses AI to cluster, analyze, and estimate metrics for suggestions."""
        
        # Limit suggestions sent to AI to avoid overly large prompts
        MAX_SUGGESTIONS_FOR_AI = 100 
        suggestions_subset = suggestions[:MAX_SUGGESTIONS_FOR_AI]
        
        logging.info(f"Sending {len(suggestions_subset)} suggestions for AI analysis (Model: {model}).")
        
        # Define the prompt for the AI service
        # This prompt asks for clustering and metric estimation
        prompt = f"""
        Analyze the following list of keywords related to the seed keyword "{seed_keyword}". 
        Group them into logical clusters/themes. For each cluster, identify the main keyword(s). 
        For the top 5-10 most important keywords overall (or representative keywords from each cluster), 
        provide estimated metrics.

        Return a JSON object with the following structure:
        {{ 
            "seed_keyword": "{seed_keyword}",
            "clusters": [
                {{
                    "theme": "Name of the theme/cluster",
                    "keywords": ["keyword1", "keyword2", ...],
                    "representative_keyword": "primary keyword for this theme"
                }}, ...
            ],
            "keyword_analysis": [
                {{
                    "keyword": "Specific Keyword",
                    "estimated_volume": number | "N/A",
                    "estimated_difficulty": number | "N/A",  // Scale 1-100
                    "estimated_cpc": number | "N/A",
                    "primary_intent": "Informational | Commercial | Transactional | Navigational | N/A",
                    "cluster_theme": "Theme name it belongs to"
                }}, ...
            ],
            "questions": ["question keyword1", "question keyword2", ...]
        }}

        Keyword Suggestions List:
        {json.dumps(suggestions_subset)}

        Rules:
        1. Cluster logically based on topic/intent.
        2. Provide realistic, but clearly *estimated*, metrics. Use "N/A" if an estimate isn't possible.
        3. Identify question-based keywords accurately.
        4. Ensure the output is valid JSON.
        """
        
        try:
            analysis_result = await self.ai_service.generate_structured_response(
                prompt=prompt,
                api_key=api_key,
                model=model,
                response_format='json_object' # Assuming ai_service supports this
            )
            
            # Basic validation of the returned structure
            if not isinstance(analysis_result, dict) or 'clusters' not in analysis_result or 'keyword_analysis' not in analysis_result:
                logging.error(f"AI analysis result has unexpected structure: {analysis_result}")
                raise ValueError("AI analysis returned data in an unexpected format.")
                
            logging.info("Successfully received and parsed AI keyword analysis.")
            return analysis_result
        
        except Exception as e:
            logging.error(f"AI keyword analysis failed for '{seed_keyword}': {e}")
            # Return a default/error structure
            return {
                "seed_keyword": seed_keyword,
                "clusters": [],
                "keyword_analysis": [],
                "questions": [],
                "error": f"AI analysis failed: {str(e)}"
            }

    async def research_keywords(self, 
                                seed_keyword: str, 
                                api_key: str, 
                                model: ModelType, 
                                fetch_trends: bool = True) -> Dict[str, Any]:
        """
        Performs comprehensive keyword research for a seed keyword.
        Combines Google Suggest, AI analysis, and optional trend data.
        """
        if not seed_keyword:
            return {"error": "Seed keyword cannot be empty."}
            
        logging.info(f"Starting keyword research for: '{seed_keyword}'")
        
        # 1. Get suggestions
        suggestions = self._get_combined_suggestions(seed_keyword)
        if not suggestions:
            logging.warning(f"No suggestions found for '{seed_keyword}'. Proceeding without them.")
        
        # 2. Perform AI Analysis (clustering, metrics estimation)
        ai_analysis = await self.perform_ai_keyword_analysis(
            seed_keyword=seed_keyword,
            suggestions=suggestions,
            api_key=api_key,
            model=model
        )
        
        # 3. Get Trend Data (Optional)
        trend_df = None
        if fetch_trends and self.pytrends:
            # Try fetching trends for the main keywords identified by AI
            top_keywords = [kw_data.get('keyword') for kw_data in ai_analysis.get('keyword_analysis', [])]
            top_keywords = [kw for kw in top_keywords if kw] # Filter out None/empty
            if not top_keywords and suggestions: # Fallback to suggestions if AI didn't identify keywords
                 top_keywords = suggestions 
                 
            if top_keywords:
                logging.info(f"Fetching trends for top keywords: {top_keywords[:5]}")
                trend_df = self.get_trend_data(top_keywords)
            else:
                 logging.info("No top keywords identified to fetch trends for.")

        # 4. Combine results
        final_result = {
            "seed_keyword": seed_keyword,
            "ai_analysis": ai_analysis, # Contains clusters, estimated metrics, questions
            "raw_suggestions_count": len(suggestions),
            "trend_data": None
        }

        if trend_df is not None:
            try:
                # Convert DataFrame to a JSON-serializable format
                trend_data_dict = {
                    'dates': trend_df.index.strftime('%Y-%m-%d').tolist(),
                    'series': {}
                }
                for col in trend_df.columns:
                    trend_data_dict['series'][col] = trend_df[col].fillna('N/A').tolist() # Handle NaN
                final_result['trend_data'] = trend_data_dict
                logging.info("Successfully added trend data.")
            except Exception as e:
                 logging.error(f"Failed to process trend DataFrame: {e}")
                 final_result['trend_data'] = {"error": "Failed to process trend data"}
        else:
            logging.info("Trend data not fetched or unavailable.")

        logging.info(f"Keyword research completed for: '{seed_keyword}'")
        return final_result

# Placeholder for AIService - replace with actual implementation
class AIService:
     async def generate_structured_response(self, prompt: str, api_key: str, model: ModelType, response_format: str) -> Dict[str, Any]:
         logging.warning("Using placeholder AIService. Implement actual AI call.")
         # Simulate an AI call and return a plausible structure
         time.sleep(0.5) 
         # Basic mock response based on prompt structure
         mock_analysis = {
             "seed_keyword": "placeholder_seed",
             "clusters": [
                 {"theme": "General", "keywords": ["kw1", "kw2"], "representative_keyword": "kw1"},
                 {"theme": "Questions", "keywords": ["how kw1?", "what kw2?"], "representative_keyword": "how kw1?"}
             ],
             "keyword_analysis": [
                 {"keyword": "kw1", "estimated_volume": 1000, "estimated_difficulty": 50, "estimated_cpc": 1.2, "primary_intent": "Informational", "cluster_theme": "General"},
                 {"keyword": "how kw1?", "estimated_volume": 100, "estimated_difficulty": 30, "estimated_cpc": 0.5, "primary_intent": "Informational", "cluster_theme": "Questions"}
             ],
             "questions": ["how kw1?", "what kw2?"]
         }
         # Extract seed keyword from prompt for mock response
         match = re.search(r'seed keyword "(.*?)"' , prompt)
         if match:
             mock_analysis['seed_keyword'] = match.group(1)
         return mock_analysis
import re # Add import for regex used in mock

# Example Usage (for testing)
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    researcher = KeywordResearchService()
    
    # Replace with your actual API key and desired model
    # DUMMY VALUES - DO NOT USE IN PRODUCTION
    api_key = "YOUR_API_KEY" 
    model_to_use: ModelType = "chatgpt" # or "gemini", "anthropic"
    
    seed = "content marketing"
    
    print(f"\n--- Starting Research for '{seed}' ---")
    results = await researcher.research_keywords(seed, api_key, model_to_use, fetch_trends=True)
    
    print("\n--- Research Results ---")
    # Print results selectively to avoid overly long output
    print(f"Seed: {results.get('seed_keyword')}")
    print(f"Suggestions Found: {results.get('raw_suggestions_count')}")
    
    ai_analysis = results.get('ai_analysis', {})
    print(f"AI Clusters: {len(ai_analysis.get('clusters', []))}")
    print(f"AI Analyzed Keywords: {len(ai_analysis.get('keyword_analysis', []))}")
    print(f"AI Identified Questions: {len(ai_analysis.get('questions', []))}")
    if 'error' in ai_analysis:
         print(f"AI Analysis Error: {ai_analysis['error']}")
         
    print("\n--- Sample AI Analysis ---")
    if ai_analysis.get('keyword_analysis'):
        print(json.dumps(ai_analysis['keyword_analysis'][:2], indent=2)) # Print first 2 analyzed keywords
    
    print("\n--- Trend Data --- (if available)")
    if results.get('trend_data') and isinstance(results['trend_data'], dict) and 'error' not in results['trend_data']:
        print(f"Dates: {len(results['trend_data']['dates'])} points")
        print(f"Series: {list(results['trend_data']['series'].keys())}")
    elif results.get('trend_data') and isinstance(results['trend_data'], dict) and 'error' in results['trend_data']:
        print(f"Trend Data Error: {results['trend_data']['error']}")
                else:
        print("No trend data.")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
