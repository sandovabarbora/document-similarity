# Machine Learning Theory Behind Document Relevance Classification

This document explains the theoretical foundations of the machine learning approaches used in the Document Relevance Classification System.

## 1. Document Embeddings

### 1.1 Transformer-Based Embeddings

The system uses transformer-based models from the Sentence Transformers library to convert documents into dense vector representations (embeddings). These models are based on the Transformer architecture that has revolutionized natural language processing.

#### Key Concepts:

- **Contextual Embeddings**: Unlike traditional word embeddings (Word2Vec, GloVe), transformer models capture contextual information, meaning the same word can have different embeddings depending on its context.

- **Sentence-BERT Architecture**: Based on BERT (Bidirectional Encoder Representations from Transformers), but optimized to create sentence-level embeddings that can be compared via cosine similarity.

- **Siamese Network Training**: Sentence transformers are typically fine-tuned using a siamese or triplet network structure to ensure that semantically similar sentences have similar embeddings.

- **Transfer Learning**: The pre-trained models encapsulate linguistic knowledge learned from massive text corpora, which can be leveraged for domain-specific tasks with relatively small amounts of data.

### 1.2 Vector Space Model

The embedding approach is founded on the Vector Space Model (VSM) from information retrieval theory:

- Documents are represented as points in a high-dimensional space
- Semantic similarity is measured by proximity in this space
- The "direction" of the vector captures the semantic meaning
- The model assumes the Distributional Hypothesis: words/documents that occur in similar contexts tend to have similar meanings

## 2. Similarity-Based Classification

### 2.1 Cosine Similarity

The primary method for comparing document embeddings is cosine similarity:

$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{A}||\mathbf{B}|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$

Where:
- $\mathbf{A}$ and $\mathbf{B}$ are the document embedding vectors
- $\theta$ is the angle between the vectors

Key properties:
- Bounded between -1 and 1
- Measure of orientation, not magnitude
- Insensitive to document length
- Value of 1 means perfectly similar, 0 means orthogonal (unrelated), -1 means opposite

### 2.2 Threshold-Based Classification

The simple classification approach uses a threshold on similarity:

1. Compute similarities between the new document and all reference documents
2. Find the maximum similarity score
3. If this score exceeds a threshold, classify the document as relevant

This approach is:
- Interpretable and transparent
- Computationally efficient
- Requires minimal training
- Well-suited for cases with limited labeled data

## 3. Machine Learning Classification

### 3.1 Random Forest Classifier

When sufficient feedback data is available, the system uses a Random Forest classifier:

#### Why Random Forest?

- **Handles Non-Linear Relationships**: Can capture complex patterns in similarity scores
- **Robust to Overfitting**: Ensemble method combines multiple decision trees
- **Feature Importance**: Provides insights into which reference documents are most influential
- **Works Well with Modest Data**: Effective even when feedback data is limited
- **Handles Imbalanced Classes**: Important when most documents might be non-relevant

#### Theory of Operation:

1. Each document is represented by its similarity scores to all reference documents
2. These similarity scores form the feature vector for classification
3. Multiple decision trees are trained on random subsets of the training data
4. Each tree "votes" on the classification
5. The majority vote determines the final classification

### 3.2 Feedback Learning Mechanism

The system implements an active learning approach:

1. User provides feedback on classifications
2. This feedback is stored with the document
3. Once sufficient feedback is collected, the ML classifier is trained
4. The classifier learns patterns from the similarity profiles
5. As more feedback is collected, the classifier becomes more accurate

This human-in-the-loop approach leverages both algorithmic power and human expertise.

## 4. Evaluation Metrics

The performance of the system can be evaluated using standard classification metrics:

- **Precision**: The proportion of relevant classifications that are actually relevant
- **Recall**: The proportion of actually relevant documents that are classified as relevant
- **F1 Score**: The harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

## 5. Theoretical Limitations and Considerations

- **Cold Start Problem**: Initial classifications rely on similarity until feedback is collected
- **Semantic Drift**: Company terminology and relevance criteria may change over time
- **Domain Specificity**: Pre-trained embeddings may not capture specialized terminology
- **Document Length**: Very long documents may lose important information when averaged
- **Cross-Modal Understanding**: Visual elements in documents are not captured by text embeddings

## 6. Future Enhancements

- **Fine-Tuning**: Domain-specific fine-tuning of embedding models
- **Hierarchical Embeddings**: Representing documents at multiple levels of granularity
- **Explainable AI**: Providing rationales for why documents are classified as relevant
- **Drift Detection**: Detecting when document patterns change and adapting the model
