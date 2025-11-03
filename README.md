
# SEO-Content-Quality-Duplicate-Detector

A machine learning pipeline that analyzes web content for SEO quality assessment and duplicate detection using NLP techniques and similarity algorithms.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/RabeehAsli/seo-content-quality-duplicate-detector
cd seo-contentquality-duplicate-detector

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place it in data/data.csv
# https://www.kaggle.com/datasets/naveen1729/dataset-for-assignment

# Run the notebook
jupyter notebook notebooks/seo_pipeline.ipynb
```

## Project Structure

```
seo-content-detector/
├── data/
│   ├── data.csv                    # Input dataset (60-70 URLs with HTML)
│   ├── extracted_content.csv       # Parsed content
│   ├── features.csv                # Computed features
│   └── duplicates.csv              # Duplicate pairs
├── notebooks/
│   └── seo_pipeline.ipynb          # Main analysis notebook
├── models/
│   └── quality_model.pkl           # Trained classifier
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

- **HTML Parsing**: Extracts clean text from HTML content using BeautifulSoup
- **Feature Engineering**: Computes readability scores, keyword extraction, and text embeddings
- **Duplicate Detection**: Identifies near-duplicate content using cosine similarity (threshold: 0.80)
- **Quality Classification**: ML model predicting Low/Medium/High quality content
- **Real-time Analysis**: `analyze_url()` function for analyzing new URLs

## Key Decisions

**1. HTML Parsing Approach**
- Used BeautifulSoup with prioritized content selectors (`<article>`, `<main>`)
- Removed navigation, scripts, and style elements for clean text extraction
- Fallback to paragraph extraction if main content not found

**2. Feature Selection**
- **Word count & sentence count**: Basic content metrics
- **Flesch Reading Ease**: Industry-standard readability score (0-100 scale)
- **TF-IDF keywords**: Top 5 most relevant terms per document
- **Sentence embeddings**: `all-MiniLM-L6-v2` model for semantic similarity

**3. Similarity Threshold**
- Set at **0.80** for duplicate detection based on empirical testing
- Balances false positives/negatives for SEO duplicate content
- Lower threshold (0.75) used for "similar content" recommendations

**4. Model Selection**
- **Random Forest Classifier** chosen for interpretability and performance
- Clear synthetic labeling rules prevent overlapping classes:
  - High: >1500 words AND readability 50-70
  - Low: <500 words OR readability <30
  - Medium: everything else
- Baseline comparison: simple word-count rules

**5. Library Choices**
- `sentence-transformers`: State-of-the-art semantic embeddings
- `textstat`: Reliable readability metrics
- `scikit-learn`: Robust ML pipeline with minimal dependencies
- No headless browsers for simplicity and speed

## Results Summary

### Model Performance
- **Accuracy**: 0.78
- **Baseline Accuracy**: 0.64
- **Improvement**: +0.14 (22% relative improvement)

### Quality Distribution
- High Quality: 40% (word count >1500, good readability)
- Medium Quality: 35% (moderate content)
- Low Quality: 25% (thin content or poor readability)

### Duplicate Detection
- **Total pages analyzed**: 60
- **Duplicate pairs found**: 3 (5%)
- **Thin content pages**: 6 (10%)

### Top Feature Importance
1. word_count (0.45)
2. flesch_reading_ease (0.32)
3. sentence_count (0.23)

## Usage

### Analyze New URL
```python
result = analyze_url('https://example.com/article', reference_df=features_df)
print(json.dumps(result, indent=2))
```

**Output:**
```json
{
  "url": "https://example.com/article",
  "word_count": 1450,
  "readability": 65.2,
  "quality_label": "High",
  "is_thin": false,
  "similar_to": [
    {"url": "https://example.com/related", "similarity": 0.76}
  ]
}
```

## Limitations

- **Scraping rate limits**: 1-2 second delays may make large-scale analysis slow
- **Language dependency**: Optimized for English content only
- **Static labeling**: Quality labels are rule-based synthetic labels, not human-annotated
- **Embedding model size**: sentence-transformers requires ~100MB download on first run

## Technical Stack

- Python 3.9+
- BeautifulSoup4 (HTML parsing)
- sentence-transformers (embeddings)
- scikit-learn (ML models)
- NLTK & textstat (NLP features)
- pandas & numpy (data processing)

## Future Improvements

- Add support for multilingual content
- Implement more sophisticated duplicate clustering (DBSCAN, hierarchical)
- Include sentiment analysis and entity recognition
- Add visualization dashboard for content insights

---

**Author**: Rabeeh Asli Villan  
**Date**: 03 November 2025  
