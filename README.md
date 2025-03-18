# Document Relevance Classification System

A tool to automatically identify relevant documents for your team based on your existing document collection.

![Version](https://img.shields.io/badge/version-1.0.0-blue)

## What is this?

This system helps you automatically determine if a new document (like a contract) is relevant to your team by comparing it with your existing documents. It saves you time by:

- Automatically scanning through new documents
- Determining their relevance to your department
- Highlighting the most similar existing documents
- Learning from your feedback to improve over time

## Quick Start

### Installation

1. Clone this repository:
   ```
   git clone [repository-url]
   cd document-relevance
   ```

2. Set up your environment:
   ```
   make setup
   make install
   ```

3. Initialize the system with your documents:
   ```
   make init
   ```

### Daily Workflow

1. **Add new documents to classify** to the `new_documents` folder

2. **Classify these documents**:
   ```
   make classify
   ```

3. **Review the results** in the terminal or check the JSON file at `data/results/results.json`

4. **Provide feedback** on any incorrect classifications to help the system learn

## Key Features

- **Works with your folder structure**: Preserves your existing SharePoint/folder organization
- **Handles multiple document types**: PDFs, Word documents, Excel files, and plain text
- **Shows similarity scores**: See how confident the system is about each classification
- **Shows similar documents**: Identifies which of your existing documents are most similar
- **Learns from feedback**: Continuously improves as you provide corrections

## How It Works

1. **Setup phase**: The system analyzes your team's existing documents to understand what's relevant to you
2. **Classification phase**: When new documents arrive, they're compared against your document set
3. **Learning phase**: Your feedback teaches the system to make better decisions

## Use Cases

### Contract Relevance

Determine if incoming contracts from a central registry are relevant to your department, saving time spent reviewing irrelevant documents.

### Document Organization

Automatically suggest appropriate folders for new documents based on your existing organization structure.

### Knowledge Discovery

Find similar documents in your repository that might contain relevant information to a current project.

## Configuration

The system uses sensible defaults, but you can customize it by creating a `config.json` file:

```json
{
  "similarity_threshold": 0.65,
  "reference_dir": "./my_reference_documents",
  "use_ml_if_available": true
}
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `make init` | Initialize the system with your documents |
| `make classify` | Classify new documents |
| `make add-docs` | Add more reference documents |
| `make help` | Show available commands |

## Providing Feedback

After classification, you can provide feedback on results:

```python
from document_relevance.main import DocumentRelevanceSystem

system = DocumentRelevanceSystem()
system.load_knowledge_base('./data/knowledge_base/kb.pkl')

# Add feedback if the classification was wrong
document_id = "path/to/document.pdf"  # The ID shown in results
system.add_feedback_by_id(document_id, is_relevant=True)  # Mark as relevant
```

## Troubleshooting

### System shows "No documents found"

Make sure you've added documents to the correct folders:
- Reference documents go in `reference_documents/`
- Documents to classify go in `new_documents/`

### Low relevance scores across all documents

You may need more reference documents. Try adding more examples of relevant documents to the `reference_documents/` folder and run `make init` again.

### Classification seems inaccurate

The system improves with feedback. After providing corrections on 10+ documents, it will start using machine learning to improve accuracy.
