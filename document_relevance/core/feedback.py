"""
Feedback management module for the Document Relevance Classification System.
Handles collection, storage, and utilization of user feedback.
"""

import os
import pickle
import time
import datetime
import json
from typing import List, Dict, Tuple, Optional, Any, Union
import logging

from document_relevance.models.document import Document, FeedbackItem
from document_relevance.utils.logging import get_logger, log_execution_time


class FeedbackManager:
    """
    Manages user feedback for document classifications to improve system accuracy over time.
    Feedback is stored and used to train machine learning models for better classification.
    """
    
    def __init__(self, storage_path: str = 'data/feedback/feedback.pkl', config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FeedbackManager.
        
        Args:
            storage_path: Path to store feedback data
            config: Configuration options
        """
        self.config = config or {}
        self.storage_path = storage_path
        self.feedback_data = []
        self.logger = get_logger("feedback")
        
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(os.path.abspath(storage_path)), exist_ok=True)
        
        # Load existing feedback if available
        self.load()
        
    @log_execution_time()
    def add_feedback(self, document: Document, is_relevant: bool, notes: Optional[str] = None) -> FeedbackItem:
        """
        Add user feedback for a document classification.
        
        Args:
            document: The document being classified
            is_relevant: Whether the document is relevant (True) or not (False)
            notes: Optional notes about the feedback
            
        Returns:
            FeedbackItem: The created feedback item
        """
        # Create feedback item with timestamp
        feedback_item = FeedbackItem(
            document_id=document.id,
            is_relevant=is_relevant,
            timestamp=time.time(),
            document=document,
            notes=notes
        )
        
        # Check if we already have feedback for this document
        existing_idx = None
        for idx, (item) in enumerate(self.feedback_data):
            if item.document_id == document.id:
                existing_idx = idx
                break
                
        if existing_idx is not None:
            # Update existing feedback
            self.feedback_data[existing_idx] = feedback_item
            self.logger.info(f"Updated feedback for document {document.id}: relevant={is_relevant}")
        else:
            # Add new feedback
            self.feedback_data.append(feedback_item)
            self.logger.info(f"Added feedback for document {document.id}: relevant={is_relevant}")
            
        # Save updated feedback
        self.save()
        
        return feedback_item
        
    def get_feedback_for_document(self, document_id: str) -> Optional[FeedbackItem]:
        """
        Get feedback for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            FeedbackItem if found, None otherwise
        """
        for item in self.feedback_data:
            if item.document_id == document_id:
                return item
        return None
        
    def get_all_feedback(self) -> List[FeedbackItem]:
        """
        Get all feedback items.
        
        Returns:
            List of feedback items
        """
        return self.feedback_data
        
    def get_labeled_documents(self) -> List[Tuple[Document, bool]]:
        """
        Get all documents with their feedback labels, suitable for model training.
        
        Returns:
            List of (document, is_relevant) tuples
        """
        labeled_docs = []
        for item in self.feedback_data:
            if item.document is not None:
                labeled_docs.append((item.document, item.is_relevant))
        return labeled_docs
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected feedback.
        
        Returns:
            Dictionary with feedback statistics
        """
        if not self.feedback_data:
            return {
                'total': 0,
                'relevant': 0,
                'not_relevant': 0,
                'relevant_percentage': 0,
                'earliest_timestamp': None,
                'latest_timestamp': None
            }
            
        relevant_count = sum(1 for item in self.feedback_data if item.is_relevant)
        total_count = len(self.feedback_data)
        not_relevant_count = total_count - relevant_count
        
        timestamps = [item.timestamp for item in self.feedback_data]
        earliest = min(timestamps) if timestamps else None
        latest = max(timestamps) if timestamps else None
        
        if earliest:
            earliest_date = datetime.datetime.fromtimestamp(earliest).strftime('%Y-%m-%d %H:%M:%S')
        else:
            earliest_date = None
            
        if latest:
            latest_date = datetime.datetime.fromtimestamp(latest).strftime('%Y-%m-%d %H:%M:%S')
        else:
            latest_date = None
        
        return {
            'total': total_count,
            'relevant': relevant_count,
            'not_relevant': not_relevant_count,
            'relevant_percentage': (relevant_count / total_count * 100) if total_count > 0 else 0,
            'earliest_timestamp': earliest_date,
            'latest_timestamp': latest_date
        }
    
    def clear_feedback(self) -> None:
        """
        Clear all feedback data.
        """
        self.feedback_data = []
        self.save()
        self.logger.info("Cleared all feedback data")
    
    def export_feedback(self, export_path: str, format: str = 'json') -> None:
        """
        Export feedback data to a file.
        
        Args:
            export_path: Path to export the data to
            format: Format to export as ('json' or 'csv')
        """
        if format.lower() == 'json':
            # Convert to serializable format
            serializable_data = []
            for item in self.feedback_data:
                item_dict = {
                    'document_id': item.document_id,
                    'is_relevant': item.is_relevant,
                    'timestamp': item.timestamp,
                    'timestamp_human': datetime.datetime.fromtimestamp(item.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    'notes': item.notes
                }
                
                # Include document info if available
                if item.document:
                    item_dict['document'] = {
                        'title': item.document.title,
                        'filepath': item.document.filepath,
                        'metadata': item.document.metadata
                    }
                    
                serializable_data.append(item_dict)
                
            # Write to JSON file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2)
                
        elif format.lower() == 'csv':
            import csv
            
            with open(export_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['document_id', 'is_relevant', 'timestamp', 'timestamp_human', 'document_title', 'notes'])
                
                # Write data
                for item in self.feedback_data:
                    timestamp_human = datetime.datetime.fromtimestamp(item.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    doc_title = item.document.title if item.document else ''
                    writer.writerow([
                        item.document_id,
                        item.is_relevant,
                        item.timestamp,
                        timestamp_human,
                        doc_title,
                        item.notes or ''
                    ])
        else:
            raise ValueError(f"Unsupported export format: {format}. Use 'json' or 'csv'.")
            
        self.logger.info(f"Exported {len(self.feedback_data)} feedback items to {export_path} ({format})")
    
    def save(self) -> None:
        """
        Save feedback data to the storage path.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            
            with open(self.storage_path, 'wb') as f:
                pickle.dump(self.feedback_data, f)
                
            self.logger.debug(f"Saved {len(self.feedback_data)} feedback items to {self.storage_path}")
        except Exception as e:
            self.logger.error(f"Error saving feedback data: {e}")
            
    def load(self) -> None:
        """
        Load feedback data from the storage path.
        """
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'rb') as f:
                    self.feedback_data = pickle.load(f)
                    
                self.logger.info(f"Loaded {len(self.feedback_data)} feedback items from {self.storage_path}")
            except Exception as e:
                self.logger.error(f"Error loading feedback data: {e}")
                self.feedback_data = []
        else:
            self.logger.info(f"No feedback data found at {self.storage_path}")
            self.feedback_data = []


class ActiveLearningManager:
    """
    Implements active learning strategies to improve model performance
    by identifying the most valuable documents for human labeling.
    """
    
    def __init__(self, feedback_manager: FeedbackManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ActiveLearningManager.
        
        Args:
            feedback_manager: FeedbackManager instance to store feedback
            config: Configuration options
        """
        self.config = config or {}
        self.feedback_manager = feedback_manager
        self.logger = get_logger("active_learning")
        
        # Strategy options
        self.strategies = {
            'uncertainty': self._uncertainty_sampling,
            'diversity': self._diversity_sampling,
            'random': self._random_sampling
        }
        
        # Default strategy
        self.default_strategy = self.config.get('default_strategy', 'uncertainty')
    
    def select_documents_for_labeling(
        self, 
        documents: List[Document], 
        classifier,  # Will be DocumentRelevanceClassifier instance
        count: int = 5,
        strategy: Optional[str] = None
    ) -> List[Document]:
        """
        Select documents for manual labeling using active learning strategies.
        
        Args:
            documents: List of unlabeled documents
            classifier: Classifier instance to use for selecting documents
            count: Number of documents to select
            strategy: Strategy to use ('uncertainty', 'diversity', 'random')
            
        Returns:
            List of selected documents for labeling
        """
        if not documents:
            return []
            
        # Use specified strategy or default
        strategy_name = strategy or self.default_strategy
        
        if strategy_name not in self.strategies:
            self.logger.warning(f"Unknown strategy: {strategy_name}. Using {self.default_strategy} instead.")
            strategy_name = self.default_strategy
            
        # Get strategy function
        strategy_func = self.strategies[strategy_name]
        
        # Apply strategy
        selected_docs = strategy_func(documents, classifier, count)
        
        self.logger.info(f"Selected {len(selected_docs)} documents for labeling using {strategy_name} strategy")
        
        return selected_docs
    
    def _uncertainty_sampling(self, documents: List[Document], classifier, count: int) -> List[Document]:
        """
        Select documents with highest uncertainty (closest to classification threshold).
        
        Args:
            documents: Documents to select from
            classifier: Classifier to use for uncertainty calculation
            count: Number of documents to select
            
        Returns:
            Selected documents
        """
        # Get prediction scores for all documents
        results = []
        for doc in documents:
            # Get classification result
            result = classifier.classify(doc)
            
            # Calculate uncertainty as distance from 0.5
            uncertainty = abs(0.5 - result.relevance_score)
            
            results.append((doc, uncertainty))
            
        # Sort by uncertainty (ascending, as lower value means more uncertain)
        results.sort(key=lambda x: x[1])
        
        # Select top 'count' documents
        selected = [doc for doc, _ in results[:count]]
        
        return selected
    
    def _diversity_sampling(self, documents: List[Document], classifier, count: int) -> List[Document]:
        """
        Select diverse set of documents based on embeddings.
        
        Args:
            documents: Documents to select from
            classifier: Classifier (used for embedding documents)
            count: Number of documents to select
            
        Returns:
            Selected documents
        """
        import numpy as np
        from sklearn.cluster import KMeans
        
        # Ensure documents have embeddings
        embedded_docs = []
        for doc in documents:
            if doc.embedding is None:
                # Use classifier's embedder to embed the document
                doc = classifier.embedder.embed_document(doc)
            embedded_docs.append(doc)
            
        if not embedded_docs:
            return []
            
        # Extract embeddings
        embeddings = np.vstack([doc.embedding for doc in embedded_docs])
        
        # Adjust count if we have fewer documents than requested
        k = min(count, len(embedded_docs))
        
        # Use K-means clustering to find diverse documents
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Find the document closest to each cluster center
        selected_indices = []
        for i in range(k):
            # Find documents in this cluster
            cluster_indices = [idx for idx, cluster in enumerate(clusters) if cluster == i]
            
            if not cluster_indices:
                continue
                
            # Find document closest to cluster center
            cluster_docs = [embedded_docs[idx] for idx in cluster_indices]
            cluster_embeddings = np.vstack([doc.embedding for doc in cluster_docs])
            
            # Calculate distances to cluster center
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            
            # Get index of closest document to center
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)
            
        # Get selected documents
        selected = [embedded_docs[idx] for idx in selected_indices]
        
        return selected
    
    def _random_sampling(self, documents: List[Document], classifier, count: int) -> List[Document]:
        """
        Select random documents for labeling.
        
        Args:
            documents: Documents to select from
            classifier: Not used in this strategy
            count: Number of documents to select
            
        Returns:
            Selected documents
        """
        import random
        
        # Adjust count if we have fewer documents than requested
        k = min(count, len(documents))
        
        # Randomly select documents
        selected = random.sample(documents, k)
        
        return selected
        
    def process_user_feedback(self, document: Document, is_relevant: bool, notes: Optional[str] = None) -> None:
        """
        Process user feedback and store it.
        
        Args:
            document: Document that was classified
            is_relevant: Whether the document is relevant
            notes: Optional notes about the feedback
        """
        # Add feedback to the feedback manager
        self.feedback_manager.add_feedback(document, is_relevant, notes)