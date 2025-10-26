import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _parse_json_response(response, stage_name: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response from AI provider with robust error handling"""
    try:
        # If response is already a dictionary, return it directly
        if isinstance(response, dict):
            logger.info(f"✅ {stage_name} received dictionary directly")
            return response
        
        # If response is a string, parse it as JSON
        if isinstance(response, str):
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
        
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                
                # Clean up common JSON issues
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                json_str = json_str.replace('  ', ' ')  # Remove double spaces
                
                # Try to parse JSON
                try:
                    parsed = json.loads(json_str)
                    logger.info(f"✅ {stage_name} parsed successfully")
                    return parsed
                except json.JSONDecodeError as json_err:
                    logger.warning(f"⚠️ First JSON parse failed, attempting to fix: {json_err}")
                    
                    # Check if response was truncated
                    if json_str.count('{') > json_str.count('}'):
                        logger.warning(f"⚠️ {stage_name} response appears truncated - missing closing braces")
                        # Try to complete the JSON by adding missing closing braces
                        missing_braces = json_str.count('{') - json_str.count('}')
                        json_str += '}' * missing_braces
                        logger.info(f"🔧 Added {missing_braces} closing braces to complete JSON")
                    
                    # Check for incomplete JSON objects (missing closing quotes, etc.)
                    if json_str.endswith(','):
                        json_str = json_str.rstrip(',') + '}'
                        logger.info(f"🔧 Removed trailing comma and added closing brace")
                    
                    # Fix common JSON issues
                    import re
                    # Fix unescaped quotes in strings
                    json_str = re.sub(r'([^\\])\"([^"]*)\"([^,}\]]*)\"', r'\1"\2\3"', json_str)
                    
                    try:
                        parsed = json.loads(json_str)
                        logger.info(f"✅ {stage_name} parsed successfully after fixing")
                        return parsed
                    except json.JSONDecodeError as final_err:
                        logger.error(f"❌ Failed to parse {stage_name} JSON: {final_err}")
                        return None
            else:
                logger.error(f"❌ No JSON found in {stage_name} response")
                return None
        else:
            logger.error(f"❌ Invalid response type for {stage_name}: {type(response)}")
            return None
            
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"❌ Failed to parse {stage_name} JSON: {e}")
        logger.error(f"Response content: {response[:1000]}...")
        return None
