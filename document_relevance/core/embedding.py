"""
Embedding module for Document Relevance Classification System.
Converts documents into semantic vector representations.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

from document_relevance.models.document import Document
from document_relevance.utils.logging import get_logger, log_execution_time

# Import sentence-transformers conditionally
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class DocumentEmbedder:
    """
    Class to create document embeddings using transformer models.
    Converts document text into semantic vector representations.
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document embedder.
        
        Args:
            model_name: Name of the Sentence Transformer model to use
            config: Configuration options for embedding
        """
        self.config = config or {}
        self.logger = get_logger("embedding")
        self.model_name = model_name
        self.model = None
        
        # Configure embedding options
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 8)
        self.chunk_size = self.config.get('chunk_size', 10000)
        self.use_chunking = self.config.get('use_chunking', True)
        self.embedding_dim = self.config.get('embedding_dim', 768)  # Default dimension
        
        # Initialize the model if sentence-transformers is available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.info(f"Initializing DocumentEmbedder with model: {model_name}")
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"Model initialized successfully. Embedding dimension: {self.embedding_dim}")
            except Exception as e:
                self.logger.error(f"Error initializing model {model_name}: {str(e)}")
                raise
        else:
            self.logger.warning("sentence-transformers not installed, using fallback embeddings")
            
    def _check_model(self):
        """Check if the model is available and raise an error if not."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers package is required for document embedding. "
                "Install it with: pip install sentence-transformers"
            )
        if self.model is None:
            raise ValueError("Model not initialized. Check the logs for errors.")
            
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text:
            self.logger.warning("Attempting to embed empty text, returning zero vector")
            return np.zeros(self.embedding_dim)
            
        try:
            self._check_model()
            return self.model.encode(text, show_progress_bar=False)
        except Exception as e:
            self.logger.error(f"Error creating embedding: {e}")
            # Fallback to zeros if error occurs
            return np.zeros(self.embedding_dim)
            
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding long documents.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of text chunks
        """
        # If text is short enough or chunking is disabled, return as is
        if len(text) <= self.chunk_size or not self.use_chunking:
            return [text]
            
        chunks = []
        words = text.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
            else:
                chunks.append(current_chunk)
                current_chunk = word
                
        if current_chunk:
            chunks.append(current_chunk)
            
        self.logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
        
    def embed_long_document(self, text: str) -> np.ndarray:
        """
        Create embedding for a long document by chunking and averaging.
        
        Args:
            text: Document text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        chunks = self.chunk_text(text)
        
        if len(chunks) == 1:
            return self.create_embedding(chunks[0])
            
        # Embed each chunk
        embeddings = []
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"Embedding chunk {i+1}/{len(chunks)}")
            embeddings.append(self.create_embedding(chunk))
            
        # Average the embeddings
        return np.mean(embeddings, axis=0)
    
    @log_execution_time()
    def embed_document(self, document: Document) -> Document:
        """
        Create embedding for a document.
        
        Args:
            document: The document to embed
            
        Returns:
            Document with added embedding
        """
        self.logger.info(f"Embedding document: {document.id}")
        
        # Skip if content is empty
        if not document.content:
            self.logger.warning(f"Document {document.id} has empty content, skipping embedding")
            document.embedding = np.zeros(self.embedding_dim)
            return document
        
        document.embedding = self.embed_long_document(document.content)
        
        # Add embedding metadata
        document.metadata['embedding_model'] = self.model_name
        document.metadata['embedding_dimension'] = self.embedding_dim
        
        return document
    
    @log_execution_time()
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Create embeddings for multiple documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of documents with added embeddings
        """
        if not documents:
            return documents
            
        self.logger.info(f"Embedding {len(documents)} documents")
        
        # Process documents in batches to avoid memory issues
        result_documents = []
        batch_size = min(self.batch_size, len(documents))
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            for doc in batch:
                try:
                    self.embed_document(doc)
                    result_documents.append(doc)
                except Exception as e:
                    self.logger.error(f"Error embedding document {doc.id}: {str(e)}")
                    # Add document without embedding rather than skipping it entirely
                    doc.embedding = np.zeros(self.embedding_dim)
                    result_documents.append(doc)
                
        return result_documents


class FallbackDocumentEmbedder(DocumentEmbedder):
    """
    Fallback document embedder that doesn't require sentence-transformers.
    Uses simpler techniques like TF-IDF for embedding if needed.
    """
    
    def __init__(self, model_name: str = 'tfidf', config: Optional[Dict[str, Any]] = None):
        """Initialize the fallback document embedder."""
        self.config = config or {}
        self.logger = get_logger("embedding.fallback")
        self.model_name = model_name
        
        # Configure embedding options
        self.max_features = self.config.get('max_features', 5000)
        self.embedding_dim = self.config.get('embedding_dim', 768)
        self.batch_size = self.config.get('batch_size', 32)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = None
        self.initialized = False
        
        self.logger.info(f"Initializing FallbackDocumentEmbedder with method: {model_name}")
        
    def _initialize_vectorizer(self, documents: Optional[List[str]] = None):
        """Initialize the TF-IDF vectorizer."""
        if self.initialized:
            return
            
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english'
            )
            
            # Fit the vectorizer if documents are provided
            if documents:
                self.vectorizer.fit(documents)
                
            self.initialized = True
            self.logger.info("TF-IDF vectorizer initialized")
        except ImportError:
            self.logger.error("scikit-learn not installed, simple hash-based embeddings will be used")
            
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create a simple embedding using TF-IDF or hashing if sklearn not available.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text:
            return np.zeros(self.embedding_dim)
            
        # Use TF-IDF if available
        if self.vectorizer:
            try:
                # Get sparse vector and convert to dense
                sparse_vec = self.vectorizer.transform([text])
                # Convert to dense and resize if needed
                dense_vec = sparse_vec.toarray()[0]
                
                # Pad or truncate to match embedding_dim
                if len(dense_vec) < self.embedding_dim:
                    result = np.zeros(self.embedding_dim)
                    result[:len(dense_vec)] = dense_vec
                    return result
                else:
                    return dense_vec[:self.embedding_dim]
            except Exception as e:
                self.logger.error(f"TF-IDF embedding failed: {str(e)}")
                
        # Fallback to simple hash-based embedding
        self.logger.debug("Using simple hash-based embedding")
        return self._hash_embedding(text)
        
    def _hash_embedding(self, text: str) -> np.ndarray:
        """
        Create a very simple hash-based embedding as last resort.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Initialize result vector
        result = np.zeros(self.embedding_dim)
        
        # Skip if empty text
        if not text:
            return result
            
        # Use words as features
        words = text.split()
        
        for word in words:
            # Use hash of word to determine position and value
            word_hash = hash(word)
            position = abs(word_hash) % self.embedding_dim
            value = (word_hash % 10000) / 10000.0  # Value between 0 and 1
            
            # Add to the vector
            result[position] += value
            
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
            
        return result
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Create embeddings for multiple documents using TF-IDF.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of documents with added embeddings
        """
        if not documents:
            return documents
        
        self.logger.info(f"Embedding {len(documents)} documents using fallback method")
        
        # Initialize vectorizer with all documents
        if not self.initialized:
            self._initialize_vectorizer([doc.content for doc in documents if doc.content])
        
        # Process documents
        for doc in documents:
            try:
                if doc.content:
                    doc.embedding = self.create_embedding(doc.content)
                else:
                    doc.embedding = np.zeros(self.embedding_dim)
                
                # Add embedding metadata
                doc.metadata['embedding_model'] = self.model_name
                doc.metadata['embedding_dimension'] = self.embedding_dim
            except Exception as e:
                self.logger.error(f"Error embedding document {doc.id}: {str(e)}")
                doc.embedding = np.zeros(self.embedding_dim)
        
        return documents