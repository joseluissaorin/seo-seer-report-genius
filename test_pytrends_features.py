import logging
import pandas as pd
from api.utils.competitor_analysis import CompetitorLandscapeAnalyzer
from api.utils.serp_feature_analyzer import KeywordTrendAnalyzer
from api.utils.keyword_research import KeywordResearcher # Assuming this exists and uses pytrends

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Test Data ---
# Use keywords relevant to potential use cases or from sample GSC data if available
# sample_keywords_general = ["seo analysis report", "google trends python", "keyword cannibalization seo"]
sample_keywords_general = ["seo", "python", "keywords"]
# sample_keywords_tech = ["fastapi tutorial", "docker best practices", "kubernetes monitoring"]
sample_keywords_tech = ["fastapi", "docker", "kubernetes"]
# sample_keywords_business = ["market research tools", "competitive analysis framework", "customer acquisition cost"]
sample_keywords_business = ["research", "analysis", "marketing"]

# --- Initialization ---
logging.info("Initializing Analyzers...")
try:
    landscape_analyzer = CompetitorLandscapeAnalyzer()
    trend_analyzer = KeywordTrendAnalyzer()
    # Assuming KeywordResearcher exists and is functional
    keyword_researcher = KeywordResearcher() 
    logging.info("Analyzers initialized successfully.")
except Exception as e:
    logging.exception(f"Failed to initialize analyzers: {e}")
    exit(1)

# --- Function to Run and Print Analysis ---
def run_analysis(analyzer_func, keywords, description):
    logging.info(f"\n--- Running: {description} for keywords: {keywords} ---")
    try:
        start_time = pd.Timestamp.now()
        results = analyzer_func(keywords)
        end_time = pd.Timestamp.now()
        logging.info(f"Finished {description} in {end_time - start_time}")
        
        if results:
            # Print a summary of results (adjust based on actual return structure)
            if 'landscape_data' in results: # CompetitorLandscapeAnalyzer
                print("Landscape Analysis Summary:")
                for kw, data in results['landscape_data'].items():
                    print(f"  {kw}: Found {len(data.get('related_queries_top',[]))} top queries, {len(data.get('related_topics_top',[]))} top topics.")
            elif 'interest_over_time' in results: # KeywordTrendAnalyzer
                print("Trend Analysis Summary:")
                if results.get('interest_over_time'): print(f"  Interest Over Time: Found data for {len(results['interest_over_time'].get('data',{}))} keywords.")
                if results.get('interest_by_region'): print(f"  Interest by Region: Found data for {len(results['interest_by_region'])} regions.")
            elif any(isinstance(v, dict) and 'trend_data' in v for v in results.values()): # KeywordResearcher (heuristic check)
                print("Keyword Research Summary:")
                print(f"  Analyzed {len(results)} keywords.")
                kw_example = next(iter(results.keys()))
                if results[kw_example].get('trend_data'): print("  Includes trend data.")
                if results[kw_example].get('related_queries',{}).get('top'): print("  Includes related queries.")
            else:
                 print("Results Structure (first level keys):", list(results.keys()))
            
            if 'recommendations' in results:
                print("Recommendations:")
                for rec in results['recommendations'][:2]: # Show first 2 recommendations
                    print(f"  - {rec.get('title', 'N/A')}")
            print("-" * 20)
        else:
            logging.warning(f"{description} returned no results.")
            
    except Exception as e:
        logging.exception(f"Error running {description}: {e}")

# --- Run Tests ---
# Note: Running multiple pytrends requests sequentially might hit rate limits.
# Consider adding delays or running tests selectively.

# Test Competitor Landscape Analyzer
run_analysis(landscape_analyzer.analyze_keyword_landscape, sample_keywords_general, "Competitor Landscape Analysis (General)")
# run_analysis(landscape_analyzer.analyze_keyword_landscape, sample_keywords_tech, "Competitor Landscape Analysis (Tech)")

# Test Keyword Trend Analyzer
run_analysis(trend_analyzer.analyze_trends_and_context, sample_keywords_business, "Keyword Trend Analysis (Business)")
# run_analysis(trend_analyzer.analyze_trends_and_context, sample_keywords_general[:2], "Keyword Trend Analysis (General - Subset)") # Test with fewer keywords

# Test Existing Keyword Researcher (if needed, assuming it works)
# run_analysis(keyword_researcher.research_keyword_trends, sample_keywords_tech[:3], "Keyword Research (Tech - Subset)")

logging.info("\n--- Test Script Finished ---") 