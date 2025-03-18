# Recursive Document Processing

The Document Relevance Classification System supports recursive processing of documents in folders and subfolders.

## How It Works

When you initialize the system or add new reference documents, the system will:

1. Start at the root reference directory
2. Scan all files in the current directory
3. Process any supported documents (PDF, DOCX, TXT, MD)
4. Recursively enter each subdirectory and repeat steps 2-4

This allows you to organize your documents in a hierarchical folder structure that matches your organization's document management system.

## Benefits of Hierarchical Organization

- **Maintain existing structure**: You can keep your documents organized in logical folders and subfolders
- **Preserve context**: The system captures folder information as metadata
- **Simplified migration**: Copy your entire document structure without reorganizing
- **Better categorization**: Use the folder structure for additional context in classification

## Document IDs and Paths

When processing documents recursively:

- Document IDs are created using the relative path from the reference directory
- Parent folder information is stored in document metadata
- The full file path is preserved for reference

## Example Structure

```
reference_documents/
├── Contracts/
│   ├── 2023/
│   │   ├── Q1/
│   │   │   ├── contract1.pdf
│   │   │   └── contract2.pdf
│   │   └── Q2/
│   │       ├── contract3.pdf
│   │       └── contract4.pdf
│   └── Templates/
│       └── standard_contract.docx
├── Policies/
│   ├── HR/
│   │   └── employee_handbook.pdf
│   └── IT/
│       └── security_policy.pdf
└── README.md
```

The system will process all documents in this structure, maintaining the hierarchical information.

## Usage

No special flags are needed - recursive processing is enabled by default:

```bash
# Initialize with documents from all subdirectories
make init

# Add more documents, scanning all subdirectories
make add-docs

# Classify new documents
make classify
```

If you want to disable recursive processing for some reason, you can use:

```bash
python -m document_relevance.cli init --reference-dir ./reference_documents --recursive=false
```
