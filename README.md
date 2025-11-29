# Resume Classifier - AI-Powered Resume Classification

This application uses machine learning to automatically classify resumes into 25+ job categories.
---
Project link: [Click Here](https://resume-classifier-project.streamlit.app)
---

## Features

- **High Accuracy**: 99.47% F1-Score with cross-validation
- **25 Job Categories**: Classifies resumes across diverse technical and non-technical roles
- **Multiple Format Support**: PDF, DOCX, and TXT files
- **Interactive UI**: Built with Streamlit for easy use
- **Confidence Scores**: Shows prediction confidence for each category

## Setup Instructions

### 1. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 2. Verify Installation

All dependencies have been installed. You can verify by running:
```bash
pip list
```

### 3. Train the Model (Optional)

The model has already been trained, but if you need to retrain:
```bash
python train_model.py
```

This will create three files:
- `model.pkl` - The trained SVM classifier
- `tfidf.pkl` - The TF-IDF vectorizer
- `encoder.pkl` - The label encoder

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Project Structure

```
resume_classifier/
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── UpdatedResumeDataSet.csv    # Training dataset
├── requirements.txt            # Python dependencies
├── venv/                       # Virtual environment
├── model.pkl                   # Trained model
├── tfidf.pkl                   # TF-IDF vectorizer
└── encoder.pkl                 # Label encoder
```

## Supported Job Categories

1. Advocate
2. Arts
3. Automation Testing
4. Blockchain
5. Business Analyst
6. Civil Engineer
7. Data Science
8. Database
9. DevOps Engineer
10. DotNet Developer
11. ETL Developer
12. Electrical Engineering
13. HR
14. Hadoop
15. Health and fitness
16. Java Developer
17. Mechanical Engineer
18. Network Security Engineer
19. Operations Manager
20. PMO
21. Python Developer
22. SAP Developer
23. Sales
24. Testing
25. Web Designing

## Technical Details

- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Feature Engineering**: TF-IDF with unigrams and bigrams
- **Vocabulary Size**: 5,000 features
- **Cross-Validation**: Stratified 5-fold CV
- **Class Balancing**: Balanced class weights

## Usage

1. Open the application in your browser
2. Upload one or more resume files (PDF, DOCX, or TXT)
3. View the predicted job category with confidence scores
4. Analyze resume statistics and detailed scores

## Notes

- **Python Version**: This project uses Python 3.14
- **PyArrow**: Optional dependency for better performance (not required for core functionality)
- The application works without PyArrow, though some Streamlit features may use fallback implementations

## Troubleshooting

If you encounter issues:

1. Make sure the virtual environment is activated
2. Verify all model files (model.pkl, tfidf.pkl, encoder.pkl) exist
3. If model files are missing, run `python train_model.py`
4. Ensure you have enough disk space and memory for model training

## Performance

- **Training Accuracy**: 100%
- **Cross-Validation F1-Score**: 99.47%
- **Model Size**: ~1.4 MB total
- **Inference Time**: < 1 second per resume
