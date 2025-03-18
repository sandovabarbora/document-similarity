"""
Command-line interface for the Document Relevance Classification System.
Supports recursive processing of document directories.
"""

import os
import sys
import argparse
import logging
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from document_relevance.utils.logging import get_logger
from document_relevance.config import load_config, save_config
from document_relevance.main import DocumentRelevanceSystem


def init_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """
    Initialize the system with reference documents.
    
    Args:
        args: Command arguments
        config: System configuration
    """
    logger = get_logger("cli.init")
    
    # Update config with command args
    config.update({
        'reference_dir': args.reference_dir,
        'kb_path': args.kb_path,
        'model_name': args.model
    })
    
    logger.info(f"Initializing system with reference documents from {args.reference_dir}")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Knowledge base will be saved to: {args.kb_path}")
    
    # Create system instance
    system = DocumentRelevanceSystem(config=config)
    
    # Initialize knowledge base with reference documents
    start_time = time.time()
    doc_count = system.initialize_knowledge_base(
        reference_dir=args.reference_dir,
        kb_path=args.kb_path,
        recursive=args.recursive
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Initialization complete. Processed {doc_count} documents in {elapsed:.2f}s")
    print(f"✅ Initialization complete. Processed {doc_count} documents in {elapsed:.2f}s")


def add_documents_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """
    Add reference documents to existing knowledge base.
    
    Args:
        args: Command arguments
        config: System configuration
    """
    logger = get_logger("cli.add")
    
    # Update config
    if args.reference_dir:
        config['reference_dir'] = args.reference_dir
    config['kb_path'] = args.kb_path
    
    # Create system instance
    system = DocumentRelevanceSystem(config=config)
    
    # Load existing knowledge base
    kb_exists = system.load_knowledge_base(args.kb_path)
    if not kb_exists:
        logger.error(f"Knowledge base not found at {args.kb_path}. Run 'init' command first.")
        print(f"❌ Knowledge base not found at {args.kb_path}. Run 'init' command first.")
        return
    
    # Add documents
    start_time = time.time()
    if args.reference_dir:
        logger.info(f"Adding documents from directory: {args.reference_dir}")
        doc_count = system.add_reference_documents(
            directory=args.reference_dir,
            recursive=args.recursive
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Added {doc_count} documents from directory in {elapsed:.2f}s")
        print(f"✅ Added {doc_count} documents from directory in {elapsed:.2f}s")
        
    elif args.files:
        logger.info(f"Adding {len(args.files)} individual files")
        doc_count = system.add_reference_documents(
            files=args.files
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Added {doc_count} documents in {elapsed:.2f}s")
        print(f"✅ Added {doc_count} documents in {elapsed:.2f}s")
    else:
        logger.error("No documents specified. Use --reference-dir or --files")
        print("❌ No documents specified. Use --reference-dir or --files")


def classify_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """
    Classify documents against reference knowledge base.
    
    Args:
        args: Command arguments
        config: System configuration
    """
    logger = get_logger("cli.classify")
    
    # Update config
    config['kb_path'] = args.kb_path
    
    # Create system instance
    system = DocumentRelevanceSystem(config=config)
    
    # Load knowledge base
    kb_exists = system.load_knowledge_base(args.kb_path)
    if not kb_exists:
        logger.error(f"Knowledge base not found at {args.kb_path}. Run 'init' command first.")
        print(f"❌ Knowledge base not found at {args.kb_path}. Run 'init' command first.")
        return
    
    # Classify documents
    start_time = time.time()
    
    if os.path.isdir(args.input):
        logger.info(f"Classifying documents from directory: {args.input}")
        results = system.classify_documents(
            directory=args.input,
            use_ml=args.use_ml,
            recursive=args.recursive
        )
        
        # Print summary
        elapsed = time.time() - start_time
        relevant_count = sum(1 for r in results if r.is_relevant)
        
        logger.info(f"Classification complete. Found {relevant_count} relevant documents out of {len(results)}")
        logger.info(f"Classification took {elapsed:.2f}s")
        
        print(f"\n✅ Classification complete!")
        print(f"Found {relevant_count} relevant documents out of {len(results)}")
        print(f"Classification took {elapsed:.2f}s")
        
        # Print top results
        if results:
            print("\nTop results:")
            # Sort by relevance score (descending)
            sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)
            for i, result in enumerate(sorted_results[:5]):  # Show top 5
                if i >= 5:
                    break
                status = "✓" if result.is_relevant else "✗"
                print(f"{i+1}. {status} {result.document.title} (Score: {result.relevance_score:.4f})")
                
            # Save results if output specified
            if args.output:
                save_results(results, args.output)
                print(f"\nResults saved to: {args.output}")
                
    else:
        # Classify single document
        logger.info(f"Classifying single document: {args.input}")
        result = system.classify_document(
            filepath=args.input,
            use_ml=args.use_ml
        )
        
        # Print result
        elapsed = time.time() - start_time
        status = "✓" if result.is_relevant else "✗"
        
        print(f"\n✅ Classification complete!")
        print(f"Document: {result.document.title}")
        print(f"Relevant: {status} (Score: {result.relevance_score:.4f})")
        if result.most_similar_document_title:
            print(f"Most similar to: {result.most_similar_document_title}")
        print(f"Classification took {elapsed:.2f}s")
        
        # Save result if output specified
        if args.output:
            save_results([result], args.output)
            print(f"\nResult saved to: {args.output}")


def save_results(results: List, output_path: str) -> None:
    """
    Save classification results to a file.
    
    Args:
        results: Classification results
        output_path: Path to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert results to serializable dictionaries
    serializable_results = []
    for result in results:
        result_dict = {
            'document_id': result.document.id,
            'document_title': result.document.title,
            'document_filepath': result.document.filepath,
            'is_relevant': result.is_relevant,
            'relevance_score': float(result.relevance_score),
            'most_similar_document_id': result.most_similar_document_id,
            'most_similar_document_title': result.most_similar_document_title,
            'classification_time': result.classification_time
        }
        serializable_results.append(result_dict)
    
    # Save based on file extension
    _, ext = os.path.splitext(output_path.lower())
    
    if ext == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
    elif ext == '.csv':
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if not serializable_results:
                return
                
            # Get headers from first result
            fieldnames = serializable_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(serializable_results)
    else:
        # Default to json if extension not recognized
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Document Relevance Classification System")
    
    # Common arguments
    parser.add_argument('--config', type=str, help="Path to config file")
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help="Logging level")
    parser.add_argument('--recursive', action='store_true', default=True,
                     help="Process subdirectories recursively (default: True)")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize the system')
    init_parser.add_argument('--reference-dir', type=str, required=True,
                          help="Directory containing reference documents")
    init_parser.add_argument('--kb-path', type=str, default='./data/knowledge_base/kb.pkl',
                          help="Path to save knowledge base")
    init_parser.add_argument('--model', type=str, default='all-mpnet-base-v2',
                          help="Sentence transformer model to use")
    
    # Add documents command
    add_parser = subparsers.add_parser('add', help='Add reference documents')
    add_parser.add_argument('--reference-dir', type=str,
                         help="Directory containing reference documents to add")
    add_parser.add_argument('--files', nargs='+', type=str,
                         help="Individual files to add as reference documents")
    add_parser.add_argument('--kb-path', type=str, default='./data/knowledge_base/kb.pkl',
                         help="Path to knowledge base")
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify documents')
    classify_parser.add_argument('--input', type=str, required=True,
                             help="File or directory to classify")
    classify_parser.add_argument('--output', type=str,
                             help="Path to save classification results")
    classify_parser.add_argument('--kb-path', type=str, default='./data/knowledge_base/kb.pkl',
                             help="Path to knowledge base")
    classify_parser.add_argument('--use-ml', action='store_true',
                             help="Use ML classifier if available")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Override log level if specified
    if args.log_level:
        config['log_level'] = args.log_level
    
    # Initialize logger
    logger = get_logger("cli", config={'log_level': getattr(logging, config['log_level'])})
    logger.info(f"Running command: {args.command}")
    
    # Execute command
    if args.command == 'init':
        init_command(args, config)
        
    elif args.command == 'add':
        add_documents_command(args, config)
        
    elif args.command == 'classify':
        classify_command(args, config)
        

if __name__ == "__main__":
    main()