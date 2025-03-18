"""
Classification module for Document Relevance Classification System.
Provides methods to classify documents based on similarity to reference documents.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

from document_relevance.models.document import Document, ClassificationResult
from document_relevance.core.knowledge_base import DocumentKnowledgeBase
from document_relevance.core.embedding import DocumentEmbedder
from document_relevance.utils.logging import get_logger, log_execution_time


class DocumentRelevanceClassifier:
    """
    Class to classify documents based on their relevance to reference documents.
    Provides both similarity-based and ML-based classification.
    """
    
    def __init__(
        self, 
        knowledge_base: DocumentKnowledgeBase, 
        embedder: DocumentEmbedder, 
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the document relevance classifier.
        
        Args:
            knowledge_base: Knowledge base with reference documents
            embedder: Document embedder for creating embeddings
            config: Configuration options
        """
        self.config = config or {}
        self.logger = get_logger("classification")
        self.knowledge_base = knowledge_base
        self.embedder = embedder
        
        # Classification parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.6)
        self.top_k = self.config.get('top_k', 5)  # Number of similar documents to consider
        self.use_folder_boost = self.config.get('use_folder_boost', True)
        self.folder_boost_factor = self.config.get('folder_boost_factor', 0.1)
        
        # ML classifier (initialized when trained)
        self.classifier = None
        self.ml_config = self.config.get('ml_classifier', {})
        
        self.logger.info(
            f"DocumentRelevanceClassifier initialized with threshold: {self.similarity_threshold}"
        )
    
    def compute_similarity(self, document: Document) -> Dict[str, Any]:
        """
        Compute similarity between a document and reference documents.
        
        Args:
            document: Document to classify
            
        Returns:
            Dictionary with similarity results
        """
        # Check if knowledge base is empty
        if not self.knowledge_base.reference_documents:
            self.logger.warning("Knowledge base is empty, cannot compute similarity")
            return {
                'max_similarity': 0.0,
                'most_similar_idx': -1,
                'similarities': [],
                'top_indices': []
            }
            
        # Create embedding if needed
        if document.embedding is None:
            document = self.embedder.embed_document(document)
            
        # Get document folder for boosting if enabled
        doc_folder = document.get_relative_folder_path() if self.use_folder_boost else None
            
        # Compute similarities with all reference documents
        try:
            # Use sklearn's cosine_similarity if available
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(
                document.embedding.reshape(1, -1), 
                self.knowledge_base.reference_embeddings
            )[0]
        except ImportError:
            self.logger.warning("scikit-learn not installed, using numpy for similarity calculation")
            # Normalize vectors for cosine similarity
            document_embedding = document.embedding.reshape(1, -1)
            query_norm = np.linalg.norm(document_embedding)
            if query_norm > 0:
                document_embedding = document_embedding / query_norm
                
            # Calculate dot product with normalized vectors (equivalent to cosine similarity)
            reference_embeddings = self.knowledge_base.reference_embeddings
            norms = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
            normalized_embeddings = reference_embeddings / np.maximum(norms, 1e-10)
            similarities = np.dot(document_embedding, normalized_embeddings.T)[0]
            
        # Apply folder boosting if enabled
        if self.use_folder_boost and doc_folder:
            for i, ref_doc in enumerate(self.knowledge_base.reference_documents):
                ref_folder = ref_doc.get_relative_folder_path()
                # Boost similarity for documents in the same folder
                if ref_folder and ref_folder == doc_folder:
                    similarities[i] += self.folder_boost_factor
                    # Ensure we don't exceed 1.0
                    similarities[i] = min(similarities[i], 1.0)
        
        # Get indices of top k similar documents
        top_indices = similarities.argsort()[-self.top_k:][::-1]
        
        # Find maximum similarity and index
        max_similarity = np.max(similarities)
        most_similar_idx = np.argmax(similarities)
        
        return {
            'max_similarity': float(max_similarity),
            'most_similar_idx': int(most_similar_idx),
            'similarities': similarities,
            'top_indices': top_indices
        }
    
    @log_execution_time()
    def classify_simple(self, document: Document) -> ClassificationResult:
        """
        Classify a document using simple similarity threshold.
        
        Args:
            document: Document to classify
            
        Returns:
            Classification result
        """
        start_time = time.time()
        self.logger.info(f"Classifying document using simple threshold: {document.id}")
        
        if not self.knowledge_base.reference_documents:
            self.logger.warning("Knowledge base is empty, cannot classify")
            return ClassificationResult(
                document=document,
                is_relevant=False,
                relevance_score=0.0,
                classification_time=time.time() - start_time
            )
            
        # Compute similarity
        sim_result = self.compute_similarity(document)
        max_similarity = sim_result['max_similarity']
        most_similar_idx = sim_result['most_similar_idx']
        similarities = sim_result['similarities']
        
        # Determine if document is relevant based on threshold
        is_relevant = max_similarity >= self.similarity_threshold
        
        # Get most similar document details
        most_similar_doc = None
        most_similar_id = None
        most_similar_title = None
        
        if most_similar_idx >= 0:
            most_similar_doc = self.knowledge_base.reference_documents[most_similar_idx]
            most_similar_id = most_similar_doc.id
            most_similar_title = most_similar_doc.title
            
        # Create similarity scores dictionary
        similarity_scores = {}
        for i, doc in enumerate(self.knowledge_base.reference_documents):
            similarity_scores[doc.id] = float(similarities[i])
            
        # Calculate classification time
        classification_time = time.time() - start_time
        
        self.logger.info(
            f"Classified document {document.id} in {classification_time:.2f}s. "
            f"Relevant: {is_relevant}, Score: {max_similarity:.4f}"
        )
            
        return ClassificationResult(
            document=document,
            is_relevant=is_relevant,
            relevance_score=float(max_similarity),
            most_similar_document_id=most_similar_id,
            most_similar_document_title=most_similar_title,
            similarity_scores=similarity_scores,
            confidence=float(max_similarity),
            classification_time=classification_time
        )
    
    @log_execution_time()
    def train_ml_classifier(self, labeled_documents: List[Tuple[Document, bool]]):
        """
        Train a machine learning classifier on labeled documents.
        
        Args:
            labeled_documents: List of (document, is_relevant) tuples
        """
        if not labeled_documents:
            self.logger.warning("No labeled documents provided for training")
            return
            
        if not self.knowledge_base.reference_documents:
            self.logger.error("Knowledge base is empty, cannot train classifier")
            return
            
        self.logger.info(f"Training ML classifier with {len(labeled_documents)} labeled documents")
        
        # Prepare features and labels
        X = []
        y = []
        
        for doc, is_relevant in labeled_documents:
            # Create embedding if needed
            if doc.embedding is None:
                doc = self.embedder.embed_document(doc)
                
            # Compute similarities to all reference documents
            sim_result = self.compute_similarity(doc)
            similarities = sim_result['similarities']
            
            # Use similarities as features
            X.append(similarities)
            y.append(1 if is_relevant else 0)
            
        # Train classifier based on configuration
        try:
            classifier_type = self.ml_config.get('type', 'random_forest')
            
            if classifier_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                self.classifier = RandomForestClassifier(
                    n_estimators=self.ml_config.get('n_estimators', 100),
                    max_depth=self.ml_config.get('max_depth', None),
                    random_state=42,
                    class_weight='balanced'
                )
            elif classifier_type == 'svm':
                from sklearn.svm import SVC
                self.classifier = SVC(
                    probability=True,
                    gamma='scale',
                    class_weight='balanced',
                    random_state=42
                )
            else:
                self.logger.warning(f"Unknown classifier type: {classifier_type}, using RandomForest")
                from sklearn.ensemble import RandomForestClassifier
                self.classifier = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced'
                )
                
            # Train the classifier
            self.classifier.fit(X, y)
            
            # Compute training accuracy
            y_pred = self.classifier.predict(X)
            accuracy = np.mean(y_pred == y)
            self.logger.info(f"ML classifier trained with accuracy: {accuracy:.4f}")
            
        except ImportError:
            self.logger.error("scikit-learn not installed, cannot train ML classifier")
            self.classifier = None
        except Exception as e:
            self.logger.error(f"Error training ML classifier: {str(e)}")
            self.classifier = None
    
    @log_execution_time()
    def classify_ml(self, document: Document) -> ClassificationResult:
        """
        Classify a document using the ML classifier.
        
        Args:
            document: Document to classify
            
        Returns:
            Classification result
        """
        if self.classifier is None:
            self.logger.warning("ML classifier not trained, falling back to simple classification")
            return self.classify_simple(document)
            
        start_time = time.time()
        self.logger.info(f"Classifying document using ML classifier: {document.id}")
        
        if not self.knowledge_base.reference_documents:
            self.logger.warning("Knowledge base is empty, cannot classify")
            return ClassificationResult(
                document=document,
                is_relevant=False,
                relevance_score=0.0,
                classification_time=time.time() - start_time
            )
            
        # Create embedding if needed
        if document.embedding is None:
            document = self.embedder.embed_document(document)
            
        # Compute similarities to all reference documents
        sim_result = self.compute_similarity(document)
        similarities = sim_result['similarities']
        most_similar_idx = sim_result['most_similar_idx']
        
        # Predict using the classifier
        try:
            relevance_proba = self.classifier.predict_proba([similarities])[0][1]
            is_relevant = relevance_proba >= 0.5
        except Exception as e:
            self.logger.error(f"Error making ML prediction: {str(e)}")
            # Fall back to simple threshold
            relevance_proba = sim_result['max_similarity']
            is_relevant = relevance_proba >= self.similarity_threshold
        
        # Get most similar document
        most_similar_doc = self.knowledge_base.reference_documents[most_similar_idx]
        
        # Create similarity scores dictionary
        similarity_scores = {}
        for i, doc in enumerate(self.knowledge_base.reference_documents):
            similarity_scores[doc.id] = float(similarities[i])
            
        # Calculate classification time
        classification_time = time.time() - start_time
        
        self.logger.info(
            f"Classified document {document.id} using ML in {classification_time:.2f}s. "
            f"Relevant: {is_relevant}, Score: {relevance_proba:.4f}"
        )
            
        return ClassificationResult(
            document=document,
            is_relevant=is_relevant,
            relevance_score=float(relevance_proba),
            most_similar_document_id=most_similar_doc.id,
            most_similar_document_title=most_similar_doc.title,
            similarity_scores=similarity_scores,
            confidence=float(relevance_proba),
            classification_time=classification_time
        )
    
    def classify(self, document: Document, use_ml: bool = False) -> ClassificationResult:
        """
        Classify a document using either simple threshold or ML.
        
        Args:
            document: Document to classify
            use_ml: Whether to use ML classification if available
            
        Returns:
            Classification result
        """
        if use_ml and self.classifier is not None:
            return self.classify_ml(document)
        else:
            return self.classify_simple(document)
            
    def explain_classification(self, result: ClassificationResult) -> Dict[str, Any]:
        """
        Provide explanation for a classification result.
        
        Args:
            result: Classification result to explain
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'is_relevant': result.is_relevant,
            'relevance_score': result.relevance_score,
            'threshold': self.similarity_threshold,
            'top_similar_documents': []
        }
        
        # Only explain if we have reference documents
        if not self.knowledge_base.reference_documents:
            explanation['error'] = "No reference documents available"
            return explanation
            
        # Get similarity scores
        similarity_scores = result.similarity_scores or {}
        
        # Get top similar documents
        if similarity_scores:
            # Sort documents by similarity score
            sorted_docs = sorted(
                [(doc_id, score) for doc_id, score in similarity_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Add top documents to explanation
            for doc_id, score in sorted_docs[:self.top_k]:
                doc = self.knowledge_base.get_document_by_id(doc_id)
                if doc:
                    explanation['top_similar_documents'].append({
                        'id': doc_id,
                        'title': doc.title,
                        'similarity_score': score,
                        'folder': doc.get_relative_folder_path()
                    })
        
        # Add folder analysis if enabled
        if self.use_folder_boost:
            folder_scores = {}
            for doc_id, score in similarity_scores.items():
                doc = self.knowledge_base.get_document_by_id(doc_id)
                if doc:
                    folder = doc.get_relative_folder_path()
                    if folder not in folder_scores:
                        folder_scores[folder] = []
                    folder_scores[folder].append(score)
            
            # Calculate average score per folder
            folder_avg_scores = {
                folder: sum(scores) / len(scores)
                for folder, scores in folder_scores.items()
            }
            
            # Sort folders by average score
            sorted_folders = sorted(
                [(folder, score) for folder, score in folder_avg_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            explanation['folder_relevance'] = [
                {'folder': folder, 'average_score': score}
                for folder, score in sorted_folders[:5]
            ]
        
        return explanation