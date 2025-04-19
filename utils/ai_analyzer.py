import logging
from typing import Dict, Any, List
import pandas as pd
import google.generativeai as genai
import json
# Assuming google.generativeai is installed and configured elsewhere
# from google.generativeai import GenerativeModel

logger = logging.getLogger(__name__)

# Constants
MAX_KEYWORDS_FOR_PROMPT = 100 # Limit the number of keywords sent to the AI to manage prompt size

def analyze_keywords_with_ai(client: genai.GenerativeModel, keyword_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes keyword data using the AI model to generate insights.

    Args:
        client: The configured GenerativeModel client.
        keyword_data: DataFrame containing keyword data (expected columns: 
                      'query', 'clicks', 'impressions', 'ctr', 'position').

    Returns:
        A dictionary containing AI-generated insights, or an empty dict if analysis fails.
    """
    logger.info(f"Performing AI analysis on {len(keyword_data)} keywords...")
    ai_insights = {}

    if keyword_data.empty:
        logger.warning("Keyword data is empty. Skipping AI analysis.")
        return ai_insights

    # Prepare data for the prompt - sample if too large
    if len(keyword_data) > MAX_KEYWORDS_FOR_PROMPT:
        logger.warning(f"Keyword data larger than {MAX_KEYWORDS_FOR_PROMPT} rows, sampling.")
        sampled_data = keyword_data.sample(n=MAX_KEYWORDS_FOR_PROMPT, random_state=42)
    else:
        sampled_data = keyword_data

    # Convert relevant data to string format for the prompt
    # Using a simple format, consider CSV or JSON string for more complex data
    prompt_data = sampled_data[['query', 'clicks', 'impressions', 'ctr', 'position']].to_string(index=False)

    prompt = f"""
Analyze the following Google Search Console keyword data:

{prompt_data}

Based on this data, provide the following insights in JSON format:
1.  **themes**: Identify 3-5 main topical themes or clusters represented by these keywords.
2.  **keyword_analysis**: Analyze the top 5-10 keywords (by impressions or clicks). For each, briefly assess its potential or any issues indicated by the metrics (e.g., high impressions/low clicks = opportunity, high position/low CTR = title/meta issue). Mention the specific query and relevant metrics in the analysis.
3.  **content_ideas**: Suggest 3-5 specific content ideas (e.g., blog post titles, page topics) based on the identified themes and keyword analysis.

Please return ONLY the JSON object, without any introductory text or markdown formatting.
Example JSON structure:
{{
  "themes": ["Theme 1", "Theme 2", "Theme 3"],
  "keyword_analysis": [
    {{"query": "example query 1", "analysis": "High impressions, low clicks suggest potential but poor visibility or relevance. Consider improving snippet."}},
    {{"query": "example query 2", "analysis": "Good position but low CTR indicates title/meta description may need optimization."}}
  ],
  "content_ideas": [
    "Blog Post: 10 Tips for Improving Example Query 1 Performance",
    "Guide: Understanding Example Query 2 and User Intent"
  ]
}}
"""

    try:
        logger.info("Sending keyword analysis prompt to Gemini...")
        response = client.generate_content(prompt)

        # Attempt to parse the JSON response
        logger.info("Parsing Gemini response for keyword analysis...")
        # Clean up potential markdown code fences
        cleaned_response_text = response.text.strip().lstrip('```json').rstrip('```').strip()
        ai_insights = json.loads(cleaned_response_text)
        logger.info("Successfully parsed AI keyword insights.")

    except json.JSONDecodeError as json_err:
        logger.error(f"Error parsing JSON response from Gemini: {json_err}")
        logger.error(f"Raw response: {response.text}")
        ai_insights = {"error": "Failed to parse AI response", "raw_response": response.text}
    except Exception as e:
        logger.error(f"Error during AI keyword analysis: {e}")
        logger.error(f"Prompt Data Sample:\n{prompt_data[:500]}...") # Log start of prompt data
        ai_insights = {"error": f"An error occurred during AI analysis: {e}"}

    return ai_insights

def analyze_competitors_with_ai(client: genai.GenerativeModel, competitor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes competitor landscape data (e.g., from pytrends related queries/topics)
    using the AI model.

    Args:
        client: The configured GenerativeModel client.
        competitor_data: Dictionary containing competitor landscape data. 
                         Expected keys might include 'related_queries', 'related_topics' 
                         (structure depends on the upstream competitor analysis function).

    Returns:
        A dictionary containing AI-generated insights, or an empty dict if analysis fails.
    """
    logger.info("Performing AI analysis on competitor data...")
    ai_insights = {}

    if not competitor_data:
        logger.warning("Competitor data is empty. Skipping AI analysis.")
        return ai_insights

    # Prepare data for the prompt
    # Convert dict to a readable string format, e.g., JSON
    # Limit the size if necessary
    try:
        # Let's focus on related queries and topics if they exist
        prompt_data_dict = {}
        if 'related_queries' in competitor_data:
            prompt_data_dict['related_queries'] = competitor_data['related_queries'] # Might need further processing/sampling
        if 'related_topics' in competitor_data:
             prompt_data_dict['related_topics'] = competitor_data['related_topics'] # Might need further processing/sampling
        
        if not prompt_data_dict:
            logger.warning("No relevant keys (related_queries, related_topics) found in competitor_data. Skipping AI analysis.")
            return ai_insights
            
        # Convert the selected data to a JSON string for the prompt
        # TODO: Implement more robust sampling/truncation if data is large
        prompt_data = json.dumps(prompt_data_dict, indent=2)
        # Truncate if too long (simple truncation for now)
        if len(prompt_data) > 5000: # Arbitrary limit, adjust as needed
             logger.warning("Competitor prompt data truncated due to size.")
             prompt_data = prompt_data[:5000] + "... (truncated)"
             
    except Exception as e:
        logger.error(f"Error processing competitor data for prompt: {e}")
        return {"error": "Failed to process competitor data for AI prompt"}

    prompt = f"""
Analyze the following competitor landscape data, likely derived from tools like Google Trends for a set of primary keywords:

{prompt_data}

Based on this data (especially related queries and topics), provide the following insights in JSON format:
1.  **landscape_summary**: Briefly summarize the competitive landscape. What are the dominant related themes or search trends?
2.  **opportunities**: Identify 2-3 potential content or keyword opportunities suggested by rising trends, related queries, or underserved related topics.
3.  **threats**: Identify 2-3 potential threats, such as highly competitive topics or declining trends relevant to the primary keywords.

Please return ONLY the JSON object, without any introductory text or markdown formatting.
Example JSON structure:
{{
  "landscape_summary": "The landscape shows strong interest in [Related Topic X] and rising queries around [Related Query Y], suggesting a focus on [Theme Z].",
  "opportunities": [
    {{"opportunity": "Create content targeting the rising query [Related Query Y]", "reasoning": "Captures growing interest."}},
    {{"opportunity": "Explore the underserved topic [Related Topic A]", "reasoning": "Potential gap in competitor coverage."}}
  ],
  "threats": [
    {{"threat": "High competition around [Related Topic X]", "reasoning": "Requires significant effort to rank."}},
    {{"threat": "Declining interest in queries related to [Old Trend]", "reasoning": "May indicate a need to shift focus."}}
  ]
}}
"""

    try:
        logger.info("Sending competitor analysis prompt to Gemini...")
        response = client.generate_content(prompt)

        # Attempt to parse the JSON response
        logger.info("Parsing Gemini response for competitor analysis...")
        cleaned_response_text = response.text.strip().lstrip('```json').rstrip('```').strip()
        ai_insights = json.loads(cleaned_response_text)
        logger.info("Successfully parsed AI competitor insights.")

    except json.JSONDecodeError as json_err:
        logger.error(f"Error parsing JSON response from Gemini: {json_err}")
        logger.error(f"Raw response: {response.text}")
        ai_insights = {"error": "Failed to parse AI response", "raw_response": response.text}
    except Exception as e:
        logger.error(f"Error during AI competitor analysis: {e}")
        logger.error(f"Prompt Data Sample:\n{prompt_data[:500]}...") # Log start of prompt data
        ai_insights = {"error": f"An error occurred during AI analysis: {e}"}

    return ai_insights

def analyze_content_gaps_with_ai(client: genai.GenerativeModel, content_gap_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes content gap data using the AI model to identify opportunities.

    Args:
        client: The configured GenerativeModel client.
        content_gap_data: DataFrame containing content gap analysis results.
                          Expected columns might include 'query', 'volume', 
                          'difficulty', 'current_position', 'gap_score'. 
                          (Exact structure depends on the upstream gap analyzer).

    Returns:
        A dictionary containing AI-generated insights, or an empty dict if analysis fails.
    """
    logger.info(f"Performing AI analysis on {len(content_gap_data)} potential content gaps...")
    ai_insights = {}

    if content_gap_data.empty:
        logger.warning("Content gap data is empty. Skipping AI analysis.")
        return ai_insights

    # Prepare data for the prompt - sample if too large
    # Sort by a potential 'gap_score' or 'volume' if available, otherwise just sample
    sort_col = next((col for col in ['gap_score', 'volume', 'query'] if col in content_gap_data.columns), None)
    if sort_col:
         sampled_data = content_gap_data.nlargest(MAX_KEYWORDS_FOR_PROMPT, columns=sort_col) if len(content_gap_data) > MAX_KEYWORDS_FOR_PROMPT else content_gap_data
    else:
         sampled_data = content_gap_data.sample(n=min(len(content_gap_data), MAX_KEYWORDS_FOR_PROMPT), random_state=42)

    logger.info(f"Using {len(sampled_data)} gap items for AI prompt (sorted by {sort_col}).")
    
    # Convert relevant data to string format
    # Ensure only relevant columns are included
    cols_to_include = [col for col in ['query', 'volume', 'difficulty', 'current_position', 'gap_score'] if col in sampled_data.columns]
    if not 'query' in cols_to_include:
        logger.error("'query' column missing from content_gap_data. Cannot proceed with AI analysis.")
        return {"error": "'query' column missing from content gap data"}
        
    prompt_data = sampled_data[cols_to_include].to_string(index=False)

    prompt = f"""
Analyze the following content gap data, which lists keywords the site potentially underperforms on or doesn't target:

{prompt_data}

Based on this data, provide the following insights in JSON format:
1.  **gap_summary**: Briefly summarize the main types of content gaps identified (e.g., are they informational keywords, comparison keywords, etc.? Are there clusters?).
2.  **priority_gaps**: Identify the top 3-5 highest priority content gaps based on the provided data (consider volume, difficulty, existing position if low, gap score if available). For each, list the query and briefly explain why it's a priority.
3.  **content_recommendations**: Suggest 2-3 specific content pieces (e.g., blog posts, landing pages, FAQ sections) that could address the highest priority gaps. Mention the target keyword(s) for each piece.

Please return ONLY the JSON object, without any introductory text or markdown formatting.
Example JSON structure:
{{
  "gap_summary": "The main gaps appear around comparison keywords for [Product Category] and informational queries related to [Topic Y].",
  "priority_gaps": [
    {{"query": "compare product a vs b", "reasoning": "High volume, no current ranking. Addresses comparison intent."}},
    {{"query": "how to use feature z", "reasoning": "Informational query with decent volume, currently ranking poorly (position 35)."}}
  ],
  "content_recommendations": [
    {{"title_idea": "Product A vs. B: An In-Depth Comparison", "target_keywords": ["compare product a vs b", "product a review"]}},
    {{"title_idea": "Step-by-Step Guide: Getting Started with Feature Z", "target_keywords": ["how to use feature z", "feature z tutorial"]}}
  ]
}}
"""

    try:
        logger.info("Sending content gap analysis prompt to Gemini...")
        response = client.generate_content(prompt)

        # Attempt to parse the JSON response
        logger.info("Parsing Gemini response for content gap analysis...")
        cleaned_response_text = response.text.strip().lstrip('```json').rstrip('```').strip()
        ai_insights = json.loads(cleaned_response_text)
        logger.info("Successfully parsed AI content gap insights.")

    except json.JSONDecodeError as json_err:
        logger.error(f"Error parsing JSON response from Gemini: {json_err}")
        logger.error(f"Raw response: {response.text}")
        ai_insights = {"error": "Failed to parse AI response", "raw_response": response.text}
    except Exception as e:
        logger.error(f"Error during AI content gap analysis: {e}")
        logger.error(f"Prompt Data Sample:\n{prompt_data[:500]}...")
        ai_insights = {"error": f"An error occurred during AI analysis: {e}"}

    return ai_insights

def analyze_serp_features_with_ai(client: genai.GenerativeModel, serp_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes SERP context and keyword trend data (e.g., from pytrends) using the AI model.
    Note: This focuses on trends, related topics, etc., not specific SERP layout features.

    Args:
        client: The configured GenerativeModel client.
        serp_data: Dictionary containing SERP context/trend data. Expected keys 
                   might include 'interest_over_time', 'interest_by_region', 
                   'related_queries', 'related_topics' (structure depends on the 
                   upstream SERP feature analyzer function).

    Returns:
        A dictionary containing AI-generated insights, or an empty dict if analysis fails.
    """
    logger.info("Performing AI analysis on SERP context/trend data...")
    ai_insights = {}

    if not serp_data:
        logger.warning("SERP data is empty. Skipping AI analysis.")
        return ai_insights

    # Prepare data for the prompt
    # Convert relevant parts of the dict to a readable string format (JSON)
    try:
        # Select relevant keys for analysis (adjust as needed based on actual serp_data structure)
        prompt_data_dict = {}
        for key in ['interest_over_time', 'interest_by_region', 'related_queries', 'related_topics']:
            if key in serp_data:
                 # Potentially sample or summarize DataFrame/list data within these keys if large
                 # For now, include as is, assuming reasonable size or upstream processing
                 prompt_data_dict[key] = serp_data[key]
        
        if not prompt_data_dict:
            logger.warning("No relevant keys found in serp_data for AI analysis. Skipping.")
            return ai_insights

        # Convert the selected data to a JSON string
        # TODO: Implement more robust sampling/truncation for large data
        prompt_data = json.dumps(prompt_data_dict, indent=2, default=str) # Use default=str for non-serializable items like Timestamps
        if len(prompt_data) > 7000: # Increased limit slightly, adjust as needed
            logger.warning("SERP prompt data truncated due to size.")
            prompt_data = prompt_data[:7000] + "... (truncated)"

    except Exception as e:
        logger.error(f"Error processing SERP data for prompt: {e}")
        return {"error": "Failed to process SERP data for AI prompt"}

    prompt = f"""
Analyze the following SERP context and keyword trend data, likely derived from tools like Google Trends for a set of primary keywords:

{prompt_data}

Based on this data (interest over time, regions, related queries/topics), provide the following insights in JSON format:
1.  **trend_summary**: Summarize the key trends observed. Is interest in the core topic growing, stable, or declining? Are there notable regional differences? What are the dominant related topics/queries?
2.  **feature_implications**: Although specific SERP features (like snippets, PAA) aren't detailed here, infer the likely *implications* based on trends and related queries. For example, rising informational queries might suggest a higher chance of PAA/Featured Snippets appearing. High interest in specific products might suggest more shopping/commercial features.
3.  **strategic_recommendations**: Based on the trends and implications, suggest 2-3 strategic actions. Examples: capitalize on a rising trend, address declining interest, target high-interest regions, create content for dominant related topics.

Please return ONLY the JSON object, without any introductory text or markdown formatting.
Example JSON structure:
{{
  "trend_summary": "Interest in the core topic is stable overall but shows a recent spike in the [Region A] region. Related queries focus heavily on [Query Type], and [Related Topic X] is a dominant related theme.",
  "feature_implications": "The rise in informational queries suggests increased prominence of PAA and Featured Snippets. Strong regional interest might correlate with more Local Pack results in those areas.",
  "strategic_recommendations": [
    {{"recommendation": "Develop region-specific content for [Region A] to capture local interest.", "reasoning": "Leverages observed regional trend."}},
    {{"recommendation": "Create comprehensive guides addressing [Query Type] queries.", "reasoning": "Aligns with dominant related query themes and likely SERP feature presence (PAA/Snippets)."}}
  ]
}}
"""

    try:
        logger.info("Sending SERP analysis prompt to Gemini...")
        response = client.generate_content(prompt)

        # Attempt to parse the JSON response
        logger.info("Parsing Gemini response for SERP analysis...")
        cleaned_response_text = response.text.strip().lstrip('```json').rstrip('```').strip()
        ai_insights = json.loads(cleaned_response_text)
        logger.info("Successfully parsed AI SERP insights.")

    except json.JSONDecodeError as json_err:
        logger.error(f"Error parsing JSON response from Gemini: {json_err}")
        logger.error(f"Raw response: {response.text}")
        ai_insights = {"error": "Failed to parse AI response", "raw_response": response.text}
    except Exception as e:
        logger.error(f"Error during AI SERP analysis: {e}")
        logger.error(f"Prompt Data Sample:\n{prompt_data[:500]}...")
        ai_insights = {"error": f"An error occurred during AI analysis: {e}"}

    return ai_insights 