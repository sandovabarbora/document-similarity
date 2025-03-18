"""
Document data models for the Document Relevance Classification System.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any, List
import numpy as np
import json
import os


@dataclass
class Document:
    """Class to represent a document with its metadata and content."""
    id: str
    title: str
    content: str
    filepath: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    is_reference: bool = False  # Whether this is a reference document
    
    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            include_embedding: Whether to include the embedding vector
            
        Returns:
            Dictionary representation of the document
        """
        doc_dict = asdict(self)
        
        # Convert numpy array to list for serialization if needed
        if include_embedding and self.embedding is not None:
            doc_dict['embedding'] = self.embedding.tolist()
        elif not include_embedding:
            doc_dict.pop('embedding', None)
            
        return doc_dict
    
    def get_folder_path(self) -> str:
        """Get the folder path from the document filepath."""
        return os.path.dirname(self.filepath)
    
    def get_relative_folder_path(self) -> str:
        """Get the relative folder path from the document ID."""
        return os.path.dirname(self.id)
    
    def get_subfolders(self) -> List[str]:
        """Get the list of subfolders in the path."""
        path = self.get_relative_folder_path()
        if not path or path == '.':
            return []
        return path.split(os.sep)
    
    def save_to_json(self, filepath: str, include_embedding: bool = False):
        """Save the document to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(include_embedding=include_embedding), f, 
                      indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'Document':
        """Load document from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Convert embedding back to numpy array if present
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
            
        return cls(**data)


@dataclass
class ClassificationResult:
    """Class to represent a document classification result."""
    document: Document
    is_relevant: bool
    relevance_score: float
    most_similar_document_id: Optional[str] = None
    most_similar_document_title: Optional[str] = None
    similarity_scores: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    classification_time: Optional[float] = None
    
    def to_dict(self, include_document: bool = True, include_embedding: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            include_document: Whether to include the full document
            include_embedding: Whether to include the document embedding
            
        Returns:
            Dictionary representation of the classification result
        """
        result_dict = asdict(self)
        if include_document:
            result_dict['document'] = self.document.to_dict(include_embedding=include_embedding)
        else:
            result_dict.pop('document', None)
        return result_dict
    
    def save_to_json(self, filepath: str, include_document: bool = True, include_embedding: bool = False):
        """Save the classification result to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(include_document=include_document, include_embedding=include_embedding), 
                     f, indent=2, ensure_ascii=False)


@dataclass
class FeedbackItem:
    """Class to represent user feedback on a classification."""
    document_id: str
    is_relevant: bool
    timestamp: float
    document: Optional[Document] = None
    notes: Optional[str] = None
    
    def to_dict(self, include_document: bool = True, include_embedding: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        feedback_dict = asdict(self)
        if include_document and self.document is not None:
            feedback_dict['document'] = self.document.to_dict(include_embedding=include_embedding)
        elif not include_document:
            feedback_dict.pop('document', None)
        return feedback_dict