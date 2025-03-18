"""
Preprocessing module for Document Relevance Classification System.
Handles text cleaning, normalization, and preparation.
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Optional
import logging

from document_relevance.models.document import Document
from document_relevance.utils.logging import get_logger, log_execution_time


class DocumentPreprocessor:
    """Class to preprocess document text."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document preprocessor.
        
        Args:
            config: Configuration options for preprocessing
        """
        self.config = config or {}
        self.logger = get_logger("preprocessing")
        
        # Configure preprocessing options
        self.min_line_length = self.config.get('min_line_length', 2)
        self.max_consecutive_spaces = self.config.get('max_consecutive_spaces', 2)
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_emails = self.config.get('remove_emails', False)
        self.normalize_whitespace = self.config.get('normalize_whitespace', True)
        self.normalize_unicode = self.config.get('normalize_unicode', True)
        self.replace_newlines = self.config.get('replace_newlines', True)
        self.min_content_length = self.config.get('min_content_length', 10)
        
        # Compile regular expressions
        self.whitespace_pattern = re.compile(r'\s+')
        self.newline_pattern = re.compile(r'[\n\r\t]+')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        
        self.logger.info("DocumentPreprocessor initialized")
        
    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing excessive whitespace, URLs, etc.
        
        Args:
            text: The input text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            self.logger.warning(f"Invalid text input: {type(text)}")
            return ""
            
        # Make a copy to avoid modifying the original
        cleaned_text = text
        
        # Normalize Unicode if enabled
        if self.normalize_unicode:
            cleaned_text = unicodedata.normalize('NFKC', cleaned_text)
        
        # Remove URLs if enabled
        if self.remove_urls:
            cleaned_text = self.url_pattern.sub(' ', cleaned_text)
            
        # Remove emails if enabled
        if self.remove_emails:
            cleaned_text = self.email_pattern.sub(' ', cleaned_text)
            
        # Replace newlines with spaces if enabled
        if self.replace_newlines:
            cleaned_text = self.newline_pattern.sub(' ', cleaned_text)
            
        # Normalize whitespace if enabled
        if self.normalize_whitespace:
            cleaned_text = self.whitespace_pattern.sub(' ', cleaned_text)
            
        # Remove multiple spaces (more than max_consecutive_spaces)
        if self.max_consecutive_spaces > 0:
            cleaned_text = re.sub(r'\s{' + str(self.max_consecutive_spaces+1) + ',}', 
                                 ' ' * self.max_consecutive_spaces, cleaned_text)
        
        # Remove very short lines
        if self.min_line_length > 0:
            lines = cleaned_text.split('\n')
            cleaned_lines = [line for line in lines if len(line.strip()) >= self.min_line_length]
            cleaned_text = '\n'.join(cleaned_lines)
            
        return cleaned_text.strip()
    
    @log_execution_time()
    def preprocess(self, document: Document) -> Document:
        """
        Preprocess the document content.
        
        Args:
            document: The document to preprocess
            
        Returns:
            Preprocessed document
        """
        self.logger.debug(f"Preprocessing document: {document.id}")
        
        # Skip preprocessing if content is too short
        if len(document.content) < self.min_content_length:
            self.logger.warning(f"Document {document.id} content too short ({len(document.content)} chars), skipping preprocessing")
            return document
        
        start_length = len(document.content)
        document.content = self.clean_text(document.content)
        end_length = len(document.content)
        
        # Calculate reduction percentage
        if start_length > 0:
            reduction_pct = 100 * (1 - end_length / start_length)
            document.metadata['preprocessing_reduction_pct'] = reduction_pct
            
            # Log a warning if preprocessing reduced content too much
            if reduction_pct > 50:
                self.logger.warning(
                    f"Preprocessing reduced document {document.id} content by more than 50%: "
                    f"{start_length} -> {end_length} chars"
                )
            else:
                self.logger.debug(
                    f"Preprocessed document {document.id}: {start_length} -> {end_length} chars "
                    f"({reduction_pct:.1f}% reduction)"
                )
                
        return document


class AdvancedDocumentPreprocessor(DocumentPreprocessor):
    """
    Advanced document preprocessor with additional text processing capabilities.
    Use this for more sophisticated text preprocessing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced document preprocessor."""
        super().__init__(config)
        
        # Additional configuration
        self.remove_numbers = self.config.get('remove_numbers', False)
        self.lowercase = self.config.get('lowercase', False)
        self.remove_punctuation = self.config.get('remove_punctuation', False)
        self.min_word_length = self.config.get('min_word_length', 2)
        
        # Initialize language detection if needed
        self.detect_language = self.config.get('detect_language', False)
        self.language_detector = None
        
        if self.detect_language:
            try:
                import langdetect
                self.language_detector = langdetect.detect
                self.logger.info("Language detection enabled")
            except ImportError:
                self.logger.warning("langdetect not installed, language detection disabled")
                self.detect_language = False
                
        self.logger.info("AdvancedDocumentPreprocessor initialized")
        
    def clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning with additional processing options.
        
        Args:
            text: The input text to clean
            
        Returns:
            Cleaned text
        """
        # Use the base cleaning first
        cleaned_text = super().clean_text(text)
        
        # Remove punctuation if enabled
        if self.remove_punctuation:
            cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))
            
        # Remove numbers if enabled
        if self.remove_numbers:
            cleaned_text = re.sub(r'\d+', '', cleaned_text)
            
        # Convert to lowercase if enabled
        if self.lowercase:
            cleaned_text = cleaned_text.lower()
            
        # Filter out short words if configured
        if self.min_word_length > 1:
            words = cleaned_text.split()
            cleaned_text = ' '.join([w for w in words if len(w) >= self.min_word_length])
            
        return cleaned_text
    
    def preprocess(self, document: Document) -> Document:
        """
        Enhanced document preprocessing with additional metadata.
        
        Args:
            document: The document to preprocess
            
        Returns:
            Preprocessed document with enhanced metadata
        """
        # Call the base preprocessing
        document = super().preprocess(document)
        
        # Detect language if enabled
        if self.detect_language and self.language_detector and document.content:
            try:
                detected_lang = self.language_detector(document.content)
                document.metadata['detected_language'] = detected_lang
                self.logger.debug(f"Detected language for {document.id}: {detected_lang}")
            except Exception as e:
                self.logger.warning(f"Language detection failed for {document.id}: {str(e)}")
                
        # Calculate some basic statistics
        document.metadata['word_count'] = len(document.content.split())
        document.metadata['char_count'] = len(document.content)
        
        return document