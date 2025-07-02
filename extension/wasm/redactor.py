#!/usr/bin/env python3
"""
Python module for text redaction and restoration.
This will be compiled to WebAssembly for use in the browser extension.
"""

import re
import json
from typing import Dict, List, Tuple


class TextRedactor:
    """
    A class for redacting and restoring sensitive information in text.
    """
    
    def __init__(self):
        # Dictionary to store original values for restoration
        self.redaction_map: Dict[str, str] = {}
        
        # Counter for generating unique redaction tokens
        self.redaction_counter = 0
        
        # Default patterns for redaction (can be extended)
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Simple name pattern
        }
    
    def _generate_redaction_token(self, category: str = "REDACTED") -> str:
        """Generate a unique redaction token."""
        self.redaction_counter += 1
        return f"[{category}_{self.redaction_counter}]"
    
    def redact(self, text: str, patterns: Dict[str, str] = None) -> str:
        """
        Redact sensitive information from text.
        
        Args:
            text: The input text to redact
            patterns: Optional custom patterns dictionary
            
        Returns:
            The redacted text with sensitive information replaced by tokens
        """
        if not text:
            return text
            
        patterns_to_use = patterns or self.patterns
        redacted_text = text
        
        for category, pattern in patterns_to_use.items():
            matches = re.finditer(pattern, redacted_text, re.IGNORECASE)
            for match in reversed(list(matches)):  # Reverse to maintain indices
                original_value = match.group()
                token = self._generate_redaction_token(category.upper())
                
                # Store the mapping for restoration
                self.redaction_map[token] = original_value
                
                # Replace the match with the token
                start, end = match.span()
                redacted_text = redacted_text[:start] + token + redacted_text[end:]
        
        return redacted_text
    
    def restore(self, text: str) -> str:
        """
        Restore redacted information in text using stored mappings.
        
        Args:
            text: The redacted text to restore
            
        Returns:
            The restored text with original values
        """
        if not text:
            return text
            
        restored_text = text
        
        # Sort tokens by length (longest first) to avoid partial replacements
        sorted_tokens = sorted(self.redaction_map.keys(), key=len, reverse=True)
        
        for token in sorted_tokens:
            if token in restored_text:
                original_value = self.redaction_map[token]
                restored_text = restored_text.replace(token, original_value)
        
        return restored_text
    
    def clear_mappings(self):
        """Clear all stored redaction mappings."""
        self.redaction_map.clear()
        self.redaction_counter = 0
    
    def get_redaction_stats(self) -> Dict:
        """Get statistics about current redactions."""
        return {
            'total_redactions': len(self.redaction_map),
            'redaction_categories': list(set(
                token.split('_')[0][1:] for token in self.redaction_map.keys()
            ))
        }


# Global instance for the WASM interface
_redactor = TextRedactor()


def redact(text: str) -> str:
    """
    Main redact function exposed to WASM.
    
    Args:
        text: Text to redact
        
    Returns:
        Redacted text
    """
    try:
        return _redactor.redact(text)
    except Exception as e:
        print(f"Error in redact: {e}")
        return text


def restore(text: str) -> str:
    """
    Main restore function exposed to WASM.
    
    Args:
        text: Text to restore
        
    Returns:
        Restored text
    """
    try:
        return _redactor.restore(text)
    except Exception as e:
        print(f"Error in restore: {e}")
        return text


def clear_redaction_mappings() -> None:
    """Clear all redaction mappings."""
    _redactor.clear_mappings()


def get_redaction_stats() -> str:
    """Get redaction statistics as JSON string."""
    return json.dumps(_redactor.get_redaction_stats())


# Test functions for development
def test_redaction():
    """Test function to verify redaction works correctly."""
    test_text = "Contact John Doe at john.doe@email.com or call 555-123-4567"
    
    print("Original:", test_text)
    redacted = redact(test_text)
    print("Redacted:", redacted)
    restored = restore(redacted)
    print("Restored:", restored)
    
    return test_text == restored


if __name__ == "__main__":
    # Run tests when executed directly
    test_redaction() 