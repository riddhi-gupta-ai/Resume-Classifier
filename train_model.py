"""
Resume Classifier Training Script

Data Science Best Practices:
1. Use original data (no oversampling) to avoid overfitting
2. Stratified K-Fold Cross-Validation for robust evaluation
3. Hyperparameter tuning with GridSearchCV
4. Better feature engineering with n-grams
5. Class weight balancing instead of oversampling
6. Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
import pickle
import re
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Resume Classifier - Improved Training Pipeline")
print("=" * 70)

# ========================================
# 1. Load Data
# ========================================
print("\n[1/7] Loading dataset...")
df = pd.read_csv('UpdatedResumeDataSet.csv')
print(f"✓ Loaded {len(df)} resumes across {df['Category'].nunique()} categories")
print(f"✓ Original class distribution:\n{df['Category'].value_counts().head(10)}")

# ========================================
# 2. Text Cleaning
# ========================================
print("\n[2/7] Cleaning text...")

def cleanResume(txt):
    """Clean resume text using regex"""
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText.strip()

df['Resume_Cleaned'] = df['Resume'].apply(cleanResume)
print("✓ Text cleaning completed")

# ========================================
# 3. Encode Labels
# ========================================
print("\n[3/7] Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(df['Category'])
print(f"✓ Encoded {len(le.classes_)} categories")

# ========================================
# 4. Feature Engineering - Improved TF-IDF
# ========================================
print("\n[4/7] Vectorizing text with improved TF-IDF...")

# Use n-grams (1,2) for better context capture
# Limit features to reduce overfitting
# Use sublinear TF scaling
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),  # Unigrams + Bigrams
    max_features=5000,   # Limit vocabulary size
    min_df=2,            # Ignore rare words
    max_df=0.8,          # Ignore too common words
    sublinear_tf=True    # Apply sublinear TF scaling
)

X = tfidf.fit_transform(df['Resume_Cleaned'])
print(f"✓ Created feature matrix: {X.shape}")
print(f"✓ Vocabulary size: {len(tfidf.vocabulary_):,}")

# ========================================
# 5. Model Training with Cross-Validation
# ========================================
print("\n[5/7] Training model with Stratified Cross-Validation...")

# Use stratified K-fold to maintain class distribution
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# SVM with class weight balancing (better than oversampling)
svc_base = SVC(
    kernel='linear',
    class_weight='balanced',  # Handle imbalance
    random_state=42
)

# Cross-validation scores
print("  Running 5-fold cross-validation...")
cv_scores = cross_val_score(svc_base, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f"  ✓ CV F1-Scores: {cv_scores}")
print(f"  ✓ Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ========================================
# 6. Hyperparameter Tuning
# ========================================
print("\n[6/7] Hyperparameter tuning with GridSearchCV...")

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
}

svc_tuned = SVC(
    kernel='rbf',  # RBF often works better than linear
    class_weight='balanced',
    random_state=42
)

grid_search = GridSearchCV(
    svc_tuned,
    param_grid,
    cv=3,  # 3-fold for speed
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

print(f"\n  ✓ Best parameters: {grid_search.best_params_}")
print(f"  ✓ Best CV score: {grid_search.best_score_:.4f}")

# ========================================
# 7. Final Model Training
# ========================================
print("\n[7/7] Training final model on full dataset...")

# Use best model from grid search
final_model = grid_search.best_estimator_

# Get predictions for evaluation
y_pred = final_model.predict(X)

# Evaluation metrics
accuracy = accuracy_score(y, y_pred)
print(f"\n✓ Training Accuracy: {accuracy:.4f}")

print("\n" + "=" * 70)
print("Classification Report:")
print("=" * 70)
print(classification_report(
    y, y_pred,
    target_names=le.classes_,
    zero_division=0
))

# ========================================
# 8. Save Models
# ========================================
print("\n[8/8] Saving models...")

pickle.dump(final_model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(le, open('encoder.pkl', 'wb'))

print("✓ Saved model.pkl")
print("✓ Saved tfidf.pkl")
print("✓ Saved encoder.pkl")

# ========================================
# Summary
# ========================================
print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"Model: SVC with RBF kernel")
print(f"Features: TF-IDF (1,2)-grams, max_features=5000")
print(f"Cross-Validation F1: {cv_scores.mean():.4f}")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Training Accuracy: {accuracy:.4f}")
print("=" * 70)

# ========================================
# Data Science Insights
# ========================================
print("\n📊 DATA SCIENCE INSIGHTS:")
print("=" * 70)
print("1. Class Imbalance Handled:")
print("   - Used class_weight='balanced' instead of oversampling")
print("   - This avoids overfitting on duplicated samples")
print()
print("2. Feature Engineering:")
print("   - Added bigrams (1,2)-grams for better context")
print("   - Limited to 5000 features to reduce overfitting")
print("   - Removed rare and too common words")
print()
print("3. Robust Validation:")
print("   - Stratified K-Fold maintains class distribution")
print("   - Cross-validation provides realistic performance estimate")
print()
print("4. Hyperparameter Optimization:")
print("   - Grid search found optimal C and gamma")
print("   - Better generalization expected")
print("=" * 70)
