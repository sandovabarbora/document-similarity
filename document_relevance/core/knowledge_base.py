"""
Knowledge Base module for Document Relevance Classification System.
Manages the storage and retrieval of reference documents and their embeddings.
"""

import os
import time
import pickle
import json
import datetime
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

from document_relevance.models.document import Document
from document_relevance.core.embedding import DocumentEmbedder
from document_relevance.utils.logging import get_logger, log_execution_time


class DocumentKnowledgeBase:
    """
    Class to manage the knowledge base of reference documents.
    Stores document embeddings and provides methods for document retrieval.
    """
    
    def __init__(self, embedder: DocumentEmbedder, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge base.
        
        Args:
            embedder: Document embedder for creating embeddings
            config: Configuration options
        """
        self.config = config or {}
        self.logger = get_logger("knowledge_base")
        self.reference_documents = []
        self.reference_embeddings = None
        self.embedder = embedder
        self.folder_structure = {}  # Map of folder paths to document indices
        
        # Store creation time and other metadata
        self.metadata = {
            'created_at': datetime.datetime.now().isoformat(),
            'updated_at': datetime.datetime.now().isoformat(),
            'document_count': 0,
            'model_name': embedder.model_name,
            'embedding_dimension': embedder.embedding_dim
        }
        
        self.logger.info("Document knowledge base initialized")
    
    @log_execution_time()
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Number of documents added
        """
        if not documents:
            self.logger.warning("No documents to add")
            return 0
            
        self.logger.info(f"Adding {len(documents)} documents to knowledge base")
        added_count = 0
        
        for doc in documents:
            try:
                # Create embedding if not present
                if doc.embedding is None:
                    doc = self.embedder.embed_document(doc)
                
                # Mark as reference document
                doc.is_reference = True
                
                # Check for duplicate ID
                if any(existing.id == doc.id for existing in self.reference_documents):
                    self.logger.warning(f"Document with ID {doc.id} already exists in the knowledge base. Skipping.")
                    continue
                
                # Add to reference documents
                self.reference_documents.append(doc)
                added_count += 1
                
                # Update folder structure
                folder_path = doc.get_relative_folder_path()
                if folder_path:
                    if folder_path not in self.folder_structure:
                        self.folder_structure[folder_path] = []
                    self.folder_structure[folder_path].append(len(self.reference_documents) - 1)
            except Exception as e:
                self.logger.error(f"Error adding document {doc.id}: {str(e)}")
            
        # Update the embeddings matrix if documents were added
        if added_count > 0:
            self._update_embeddings()
            
            # Update metadata
            self.metadata['document_count'] = len(self.reference_documents)
            self.metadata['updated_at'] = datetime.datetime.now().isoformat()
            
        self.logger.info(
            f"Added {added_count} documents to knowledge base. "
            f"Total documents: {len(self.reference_documents)}"
        )
        
        return added_count
        
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the knowledge base by ID.
        
        Args:
            doc_id: ID of the document to remove
            
        Returns:
            True if document was removed, False otherwise
        """
        initial_count = len(self.reference_documents)
        
        # Find index of document with the given ID
        doc_index = None
        for i, doc in enumerate(self.reference_documents):
            if doc.id == doc_id:
                doc_index = i
                break
        
        # Remove from reference documents if found
        if doc_index is not None:
            removed_doc = self.reference_documents.pop(doc_index)
            
            # Update folder structure
            folder_path = removed_doc.get_relative_folder_path()
            if folder_path in self.folder_structure:
                # Remove this index
                if doc_index in self.folder_structure[folder_path]:
                    self.folder_structure[folder_path].remove(doc_index)
                
                # Update indices that are greater than the removed index
                for path, indices in self.folder_structure.items():
                    self.folder_structure[path] = [idx if idx < doc_index else idx - 1 for idx in indices]
            
            # Update embeddings matrix
            self._update_embeddings()
            
            # Update metadata
            self.metadata['document_count'] = len(self.reference_documents)
            self.metadata['updated_at'] = datetime.datetime.now().isoformat()
            
            self.logger.info(f"Removed document with ID {doc_id}")
            return True
        else:
            self.logger.warning(f"No document found with ID {doc_id}")
            return False
        
    def _update_embeddings(self):
        """Update the embeddings matrix from all reference documents."""
        if not self.reference_documents:
            self.reference_embeddings = None
            self.logger.debug("No reference documents, embeddings matrix is None")
            return
            
        # Create a matrix of all embeddings
        self.reference_embeddings = np.vstack([doc.embedding for doc in self.reference_documents])
        self.logger.debug(f"Updated embeddings matrix: {self.reference_embeddings.shape}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by its ID.
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        for doc in self.reference_documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def get_documents_by_folder(self, folder_path: str) -> List[Document]:
        """
        Get all documents in a specific folder path.
        
        Args:
            folder_path: Relative folder path
            
        Returns:
            List of documents in the folder
        """
        if folder_path in self.folder_structure:
            return [self.reference_documents[idx] for idx in self.folder_structure[folder_path]]
        return []
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to a query embedding.
        
        Args:
            query_embedding: Embedding vector to compare against
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.reference_embeddings is None or len(self.reference_documents) == 0:
            self.logger.warning("No reference documents available for similarity search")
            return []
            
        # Reshape query to 2D if it's 1D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Compute similarities
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, self.reference_embeddings)[0]
        except ImportError:
            self.logger.warning("scikit-learn not installed, using numpy for similarity calculation")
            # Normalize vectors for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
                
            # Calculate dot product with normalized vectors (equivalent to cosine similarity)
            norms = np.linalg.norm(self.reference_embeddings, axis=1, keepdims=True)
            normalized_embeddings = self.reference_embeddings / np.maximum(norms, 1e-10)
            similarities = np.dot(query_embedding, normalized_embeddings.T)[0]
            
        # Get indices of top_k similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return documents with similarity scores
        results = []
        for idx in top_indices:
            results.append((self.reference_documents[idx], float(similarities[idx])))
            
        return results
        
    @log_execution_time()
    def save(self, filepath: str):
        """
        Save the knowledge base to a file.
        
        Args:
            filepath: Path to save the knowledge base
        """
        try:
            self.logger.info(f"Saving knowledge base to {filepath}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
                
            self.logger.info(f"Saved knowledge base with {len(self.reference_documents)} documents to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving knowledge base: {str(e)}")
            raise
    
    def save_metadata(self, filepath: str):
        """
        Save knowledge base metadata to a JSON file.
        
        Args:
            filepath: Path to save the metadata
        """
        try:
            meta = self.metadata.copy()
            
            # Add folder structure information
            meta['folders'] = {}
            for folder, doc_indices in self.folder_structure.items():
                meta['folders'][folder] = len(doc_indices)
                
            # Add document IDs and titles
            meta['documents'] = [
                {'id': doc.id, 'title': doc.title, 'folder': doc.get_relative_folder_path()}
                for doc in self.reference_documents
            ]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved knowledge base metadata to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
            
    @staticmethod
    def load(filepath: str) -> 'DocumentKnowledgeBase':
        """
        Load the knowledge base from a file.
        
        Args:
            filepath: Path to the saved knowledge base file
            
        Returns:
            Loaded knowledge base
        """
        logger = get_logger("knowledge_base")
        try:
            logger.info(f"Loading knowledge base from {filepath}")
            
            with open(filepath, 'rb') as f:
                kb = pickle.load(f)
                
            logger.info(f"Loaded knowledge base with {len(kb.reference_documents)} documents")
            return kb
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            raise