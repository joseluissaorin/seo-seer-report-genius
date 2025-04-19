import requests
import xml.etree.ElementTree as ET
import logging
from typing import List, Optional
import time
import string

class GoogleSuggester:
    """Fetches keyword suggestions from the Google Autocomplete API."""

    BASE_URL = "http://google.com/complete/search?output=toolbar"

    def __init__(self, lang: str = 'en', country_code: str = 'us'):
        """
        Initializes the suggester.

        Args:
            lang: Language code (e.g., 'en').
            country_code: Two-letter country code (e.g., 'us').
        """
        self.lang = lang
        self.country_code = country_code
        logging.info(f"GoogleSuggester initialized for lang='{lang}', country='{country_code}'")

    def _make_request(self, query: str) -> Optional[ET.Element]:
        """Makes a request to the Google Autocomplete API."""
        url = f"{self.BASE_URL}&hl={self.lang}&gl={self.country_code}&q={requests.utils.quote(query)}"
        try:
            result = requests.get(url, timeout=5)
            result.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            # Check if content is valid XML before parsing
            if not result.content or not result.content.strip().startswith(b'<?xml'):
                 logging.warning(f"Non-XML response received for query '{query}': {result.content[:100]}...")
                 return None
                 
            tree = ET.ElementTree(ET.fromstring(result.content))
            return tree.getroot()
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for query '{query}': {e}")
            return None
        except ET.ParseError as e:
             logging.error(f"Failed to parse XML for query '{query}': {e}. Content: {result.content[:200]}...")
             return None
        except Exception as e:
            logging.error(f"An unexpected error occurred for query '{query}': {e}")
            return None

    def get_suggestions(self, query: str) -> List[str]:
        """Gets suggestions for a single query."""
        suggestions = []
        root = self._make_request(query)
        if root is not None:
            for suggestion_element in root.findall('CompleteSuggestion'):
                suggestion = suggestion_element.find('suggestion')
                if suggestion is not None:
                    data = suggestion.attrib.get('data')
                    if data:
                        suggestions.append(data)
        return suggestions

    def get_alphabetical_suggestions(self, base_keyword: str, delay: float = 0.1) -> List[str]:
        """Gets suggestions by prefixing the base keyword with each letter (a-z)."""
        all_suggestions = set()
        logging.info(f"Fetching alphabetical suggestions for '{base_keyword}'")
        for char_code in range(ord('a'), ord('z') + 1):
            char = chr(char_code)
            query = f"{base_keyword} {char}"
            suggestions = self.get_suggestions(query)
            all_suggestions.update(suggestions)
            time.sleep(delay) # Be respectful to the API
        logging.info(f"Found {len(all_suggestions)} alphabetical suggestions.")    
        return sorted(list(all_suggestions))

    def get_question_suggestions(self, base_keyword: str, delay: float = 0.1) -> List[str]:
        """Gets suggestions by prefixing with common question words."""
        question_prefixes = [
            "how", "what", "when", "where", "why", "who", "which", "are", "is", "can", "do", "does"
        ]
        all_suggestions = set()
        logging.info(f"Fetching question suggestions for '{base_keyword}'")
        for prefix in question_prefixes:
            query = f"{prefix} {base_keyword}"
            suggestions = self.get_suggestions(query)
            # Filter suggestions to ensure they still contain the base keyword (more likely to be relevant)
            filtered = [s for s in suggestions if base_keyword.lower() in s.lower()]
            all_suggestions.update(filtered)
            time.sleep(delay)
        logging.info(f"Found {len(all_suggestions)} question suggestions.")
        return sorted(list(all_suggestions))
        
    def get_comparison_suggestions(self, base_keyword: str, delay: float = 0.1) -> List[str]:
        """Gets suggestions using comparison terms like 'vs'."""
        comparison_prefixes = ["vs", "versus", "compare"]
        all_suggestions = set()
        logging.info(f"Fetching comparison suggestions for '{base_keyword}'")
        for prefix in comparison_prefixes:
            # Query both prefix and suffix positions
            queries = [f"{prefix} {base_keyword}", f"{base_keyword} {prefix}"]
            for query in queries:
                suggestions = self.get_suggestions(query)
                filtered = [s for s in suggestions if base_keyword.lower() in s.lower()]
                all_suggestions.update(filtered)
                time.sleep(delay)
        logging.info(f"Found {len(all_suggestions)} comparison suggestions.")
        return sorted(list(all_suggestions))

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    suggester = GoogleSuggester(lang='en', country_code='us')
    
    keyword = "seo analysis"
    
    print(f"\n--- Basic Suggestions for '{keyword}' ---")
    basic_suggestions = suggester.get_suggestions(keyword)
    print(basic_suggestions[:10]) # Print first 10
    
    print(f"\n--- Alphabetical Suggestions for '{keyword}' ---")
    alpha_suggestions = suggester.get_alphabetical_suggestions(keyword)
    print(alpha_suggestions[:20]) # Print first 20
    
    print(f"\n--- Question Suggestions for '{keyword}' ---")
    question_suggestions = suggester.get_question_suggestions(keyword)
    print(question_suggestions[:20]) # Print first 20
    
    print(f"\n--- Comparison Suggestions for '{keyword}' ---")
    comparison_suggestions = suggester.get_comparison_suggestions(keyword)
    print(comparison_suggestions[:20]) # Print first 20 