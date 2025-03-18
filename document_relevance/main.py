"""
Main entry point for the Document Relevance Classification System.
Provides a simplified interface to the core functionality.
"""

import os
import time
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

from document_relevance.utils.logging import get_logger
from document_relevance.config import load_config
from document_relevance.models.document import Document, ClassificationResult

# Import all required modules
from document_relevance.utils.document_loader import DocumentLoader, DocumentPreprocessor
from document_relevance.core.embedding import DocumentEmbedder
from document_relevance.core.knowledge_base import DocumentKnowledgeBase
from document_relevance.core.classification import DocumentRelevanceClassifier
from document_relevance.core.feedback import FeedbackManager, ActiveLearningManager


class DocumentRelevanceSystem:
    """
    Main class for the Document Relevance Classification System.
    Coordinates all components and provides a simplified interface.
    """
    
    def __init__(self, config_file: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the system with configuration.
        
        Args:
            config_file: Path to a JSON configuration file
            config: Configuration dictionary (overrides file if provided)
        """
        # Set up configuration
        self.config = load_config(config_file) if config_file else {}
        if config:
            self.config.update(config)  # Override with provided config
        
        # Configure logging
        log_level_name = self.config.get('log_level', 'INFO')
        log_level = getattr(logging, log_level_name)
        self.logger = get_logger("main", config={'log_level': log_level})
        
        self.logger.info("Initializing document relevance system")
        
        # Initialize components
        self._initialize_components()
        
        # Load knowledge base if path is specified
        kb_path = self.config.get('kb_path')
        if kb_path and os.path.exists(kb_path):
            self.load_knowledge_base(kb_path)
        
    def _initialize_components(self) -> None:
        """
        Initialize system components.
        """
        # Create preprocessor
        preprocessor_config = self.config.get('preprocessor', {})
        self.preprocessor = DocumentPreprocessor(config=preprocessor_config)
        
        # Create document loader
        loader_config = self.config.get('loader', {})
        self.document_loader = DocumentLoader(
            preprocessor=self.preprocessor,
            config=loader_config
        )
        
        # Create document embedder
        model_name = self.config.get('model_name', 'all-mpnet-base-v2')
        embedder_config = self.config.get('embedder', {})
        self.embedder = DocumentEmbedder(
            model_name=model_name,
            config=embedder_config
        )
        
        # Create knowledge base
        self.knowledge_base = DocumentKnowledgeBase(
            embedder=self.embedder
        )
        
        # Create feedback manager
        feedback_path = self.config.get('feedback_path', './data/feedback/feedback.pkl')
        self.feedback_manager = FeedbackManager(
            storage_path=feedback_path
        )
        
        # Create active learning manager
        self.active_learning = ActiveLearningManager(
            feedback_manager=self.feedback_manager
        )
        
        # Create classifier (will be initialized when knowledge base is loaded)
        self.classifier = None
    
    def initialize_knowledge_base(
        self, 
        reference_dir: str, 
        kb_path: Optional[str] = None,
        recursive: bool = True
    ) -> int:
        """
        Initialize the knowledge base with reference documents.
        
        Args:
            reference_dir: Directory containing reference documents
            kb_path: Path to save the knowledge base
            recursive: Whether to process subdirectories recursively
            
        Returns:
            Number of documents processed
        """
        self.logger.info(f"Initializing knowledge base with documents from {reference_dir}")
        
        # Load reference documents
        start_time = time.time()
        reference_docs = self.document_loader.load_documents_from_directory(
            directory=reference_dir,
            is_reference=True,
            recursive=recursive
        )
        
        if not reference_docs:
            self.logger.warning(f"No documents found in {reference_dir}")
            return 0
        
        # Create embeddings for reference documents
        self.logger.info(f"Creating embeddings for {len(reference_docs)} documents")
        embedded_docs = self.embedder.embed_documents(reference_docs)
        
        # Add documents to knowledge base
        self.knowledge_base.add_documents(embedded_docs)
        
        # Save knowledge base if path is specified
        if kb_path:
            os.makedirs(os.path.dirname(os.path.abspath(kb_path)), exist_ok=True)
            self.knowledge_base.save(kb_path)
            self.logger.info(f"Knowledge base saved to {kb_path}")
            
        # Initialize classifier
        similarity_threshold = self.config.get('similarity_threshold', 0.65)
        classifier_config = self.config.get('classifier', {'similarity_threshold': similarity_threshold})
        self.classifier = DocumentRelevanceClassifier(
            knowledge_base=self.knowledge_base,
            embedder=self.embedder,
            config=classifier_config
        )
        
        # Train ML classifier if we have feedback
        feedback_data = self.feedback_manager.get_labeled_documents()
        if feedback_data and len(feedback_data) >= 10 and hasattr(self.classifier, 'train_ml_classifier'):
            self.logger.info(f"Training ML classifier with {len(feedback_data)} feedback items")
            self.classifier.train_ml_classifier(feedback_data)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Knowledge base initialized with {len(embedded_docs)} documents in {elapsed:.2f}s")
        
        return len(embedded_docs)
    
    def load_knowledge_base(self, kb_path: str) -> bool:
        """
        Load an existing knowledge base.
        
        Args:
            kb_path: Path to the knowledge base file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(kb_path):
            self.logger.warning(f"Knowledge base not found at {kb_path}")
            return False
        
        try:
            self.logger.info(f"Loading knowledge base from {kb_path}")
            self.knowledge_base = DocumentKnowledgeBase.load(kb_path)
            
            # Initialize classifier
            similarity_threshold = self.config.get('similarity_threshold', 0.65)
            classifier_config = self.config.get('classifier', {'similarity_threshold': similarity_threshold})
            self.classifier = DocumentRelevanceClassifier(
                knowledge_base=self.knowledge_base,
                embedder=self.embedder,
                config=classifier_config
            )
            
            # Train ML classifier if we have feedback
            feedback_data = self.feedback_manager.get_labeled_documents()
            if feedback_data and len(feedback_data) >= 10 and hasattr(self.classifier, 'train_ml_classifier'):
                self.logger.info(f"Training ML classifier with {len(feedback_data)} feedback items")
                self.classifier.train_ml_classifier(feedback_data)
            
            self.logger.info(f"Loaded knowledge base with {len(self.knowledge_base.reference_documents)} documents")
            return True
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
            return False
    
    def add_reference_documents(
        self, 
        directory: Optional[str] = None,
        files: Optional[List[str]] = None,
        recursive: bool = True
    ) -> int:
        """
        Add reference documents to the knowledge base.
        
        Args:
            directory: Directory containing reference documents
            files: List of individual files to add
            recursive: Whether to process subdirectories recursively
            
        Returns:
            Number of documents added
        """
        if not self.knowledge_base:
            self.logger.error("Knowledge base not initialized")
            return 0
        
        documents = []
        
        # Load documents from directory
        if directory:
            self.logger.info(f"Loading documents from directory: {directory}")
            dir_docs = self.document_loader.load_documents_from_directory(
                directory=directory,
                is_reference=True,
                recursive=recursive
            )
            documents.extend(dir_docs)
        
        # Load individual files
        if files:
            self.logger.info(f"Loading {len(files)} individual files")
            for filepath in files:
                try:
                    doc = self.document_loader.load_document(
                        filepath=filepath,
                        is_reference=True
                    )
                    documents.append(doc)
                except Exception as e:
                    self.logger.warning(f"Error loading document {filepath}: {e}")
        
        if not documents:
            self.logger.warning("No documents to add")
            return 0
        
        # Create embeddings
        self.logger.info(f"Creating embeddings for {len(documents)} documents")
        embedded_docs = self.embedder.embed_documents(documents)
        
        # Add to knowledge base
        self.knowledge_base.add_documents(embedded_docs)
        
        # Save knowledge base
        kb_path = self.config.get('kb_path')
        if kb_path:
            self.knowledge_base.save(kb_path)
            self.logger.info(f"Updated knowledge base saved to {kb_path}")
        
        return len(embedded_docs)
    
    def classify_document(self, filepath: str, use_ml: Optional[bool] = None) -> ClassificationResult:
        """
        Classify a single document.
        
        Args:
            filepath: Path to the document
            use_ml: Whether to use ML classification (if available)
            
        Returns:
            Classification result
        """
        # Check if classifier is initialized
        if not self.classifier:
            self.logger.error("Classifier not initialized. Load or initialize knowledge base first.")
            # Return empty result
            return ClassificationResult(
                document=Document(id="error", title="Error", content="", filepath=filepath),
                is_relevant=False,
                relevance_score=0.0,
                classification_time=0.0
            )
        
        # Load document
        try:
            self.logger.info(f"Loading document: {filepath}")
            document = self.document_loader.load_document(filepath=filepath)
        except Exception as e:
            self.logger.error(f"Error loading document {filepath}: {e}")
            # Return error result
            return ClassificationResult(
                document=Document(id="error", title=f"Error: {str(e)}", content="", filepath=filepath),
                is_relevant=False,
                relevance_score=0.0,
                classification_time=0.0
            )
        
        # Classify document
        self.logger.info(f"Classifying document: {document.title}")
        
        # Determine whether to use ML
        if use_ml is None:
            use_ml = self.config.get('use_ml_if_available', True)
        
        start_time = time.time()
        result = self.classifier.classify(document, use_ml=use_ml)
        result.classification_time = time.time() - start_time
        
        self.logger.info(
            f"Classification result: is_relevant={result.is_relevant}, "
            f"score={result.relevance_score:.4f}, time={result.classification_time:.2f}s"
        )
        
        return result
    
    def classify_documents(
        self, 
        directory: str, 
        use_ml: Optional[bool] = None,
        recursive: bool = True
    ) -> List[ClassificationResult]:
        """
        Classify all documents in a directory.
        
        Args:
            directory: Path to the directory containing documents
            use_ml: Whether to use ML classification (if available)
            recursive: Whether to process subdirectories recursively
            
        Returns:
            List of classification results
        """
        # Check if classifier is initialized
        if not self.classifier:
            self.logger.error("Classifier not initialized. Load or initialize knowledge base first.")
            return []
        
        # Load documents
        self.logger.info(f"Loading documents from {directory}")
        try:
            documents = self.document_loader.load_documents_from_directory(
                directory=directory,
                recursive=recursive
            )
        except Exception as e:
            self.logger.error(f"Error loading documents from {directory}: {e}")
            return []
        
        if not documents:
            self.logger.warning(f"No documents found in {directory}")
            return []
        
        # Create embeddings
        self.logger.info(f"Creating embeddings for {len(documents)} documents")
        embedded_docs = self.embedder.embed_documents(documents)
        
        # Classify documents
        self.logger.info(f"Classifying {len(embedded_docs)} documents")
        
        # Determine whether to use ML
        if use_ml is None:
            use_ml = self.config.get('use_ml_if_available', True)
        
        results = []
        start_time = time.time()
        for i, doc in enumerate(embedded_docs):
            self.logger.info(f"Classifying document {i+1}/{len(embedded_docs)}: {doc.title}")
            result = self.classifier.classify(doc, use_ml=use_ml)
            result.classification_time = time.time() - start_time
            results.append(result)
            start_time = time.time()  # Reset for next document
        
        self.logger.info(f"Classified {len(results)} documents")
        
        return results
    
    def add_feedback(self, document: Document, is_relevant: bool, notes: Optional[str] = None) -> None:
        """
        Add user feedback for a document classification.
        
        Args:
            document: Document that was classified
            is_relevant: Whether the document is relevant
            notes: Optional notes about the feedback
        """
        self.logger.info(f"Adding feedback for document {document.id}: is_relevant={is_relevant}")
        
        # Add feedback
        self.feedback_manager.add_feedback(document, is_relevant, notes)
        
        # Retrain classifier if we have enough feedback
        feedback_data = self.feedback_manager.get_labeled_documents()
        if len(feedback_data) >= 10 and self.classifier and hasattr(self.classifier, 'train_ml_classifier'):
            self.logger.info(f"Retraining classifier with {len(feedback_data)} feedback items")
            self.classifier.train_ml_classifier(feedback_data)
    
    def get_active_learning_suggestions(
        self, 
        documents: List[Document], 
        count: int = 5, 
        strategy: str = 'uncertainty'
    ) -> List[Document]:
        """
        Get suggestions for documents to label based on active learning.
        
        Args:
            documents: List of unlabeled documents
            count: Number of documents to suggest
            strategy: Active learning strategy to use
            
        Returns:
            List of suggested documents
        """
        if not self.classifier:
            self.logger.error("Classifier not initialized. Cannot provide suggestions.")
            return []
        
        self.logger.info(f"Getting {count} active learning suggestions using {strategy} strategy")
        
        # Get suggestions
        suggestions = self.active_learning.select_documents_for_labeling(
            documents=documents,
            classifier=self.classifier,
            count=count,
            strategy=strategy
        )
        
        return suggestions


def main():
    """Simple example usage."""
    # Initialize system
    system = DocumentRelevanceSystem()
    
    # Define directories
    reference_dir = './reference_documents'
    new_docs_dir = './new_documents'
    
    # Ensure directories exist
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(new_docs_dir, exist_ok=True)
    
    # Check if knowledge base exists
    kb_path = './data/knowledge_base/kb.pkl'
    kb_exists = os.path.exists(kb_path)
    
    if not kb_exists:
        print("Knowledge base not found. Initializing with reference documents...")
        if not os.listdir(reference_dir):
            print(f"Warning: No documents found in {reference_dir}")
            return
            
        system.initialize_knowledge_base(reference_dir, kb_path)
    else:
        print(f"Loading existing knowledge base from {kb_path}")
        system.load_knowledge_base(kb_path)
    
    # Check for documents to classify
    if os.listdir(new_docs_dir):
        example_doc = os.path.join(new_docs_dir, os.listdir(new_docs_dir)[0])
        print(f"Classifying example document: {example_doc}")
        
        result = system.classify_document(example_doc)
        
        print(f"Document: {result.document.title}")
        print(f"Is relevant: {result.is_relevant}")
        print(f"Relevance score: {result.relevance_score:.4f}")
        if result.most_similar_document_title:
            print(f"Most similar to: {result.most_similar_document_title}")
    else:
        print(f"No documents found in {new_docs_dir} for classification")


if __name__ == "__main__":
    main()