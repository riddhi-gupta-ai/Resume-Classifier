import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import os
import sys

# Load pre-trained model and TF-IDF vectorizer
try:
    # Try new model first, fall back to old model
    model_file = 'model.pkl' if os.path.exists('model.pkl') else 'clf.pkl'

    if not os.path.exists(model_file):
        st.error(f"Model file not found. Please run: python train_model.py")
        sys.exit(1)
    if not os.path.exists('tfidf.pkl'):
        st.error("TF-IDF vectorizer file 'tfidf.pkl' not found.")
        sys.exit(1)
    if not os.path.exists('encoder.pkl'):
        st.error("Label encoder file 'encoder.pkl' not found.")
        sys.exit(1)

    svc_model = pickle.load(open(model_file, 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    le = pickle.load(open('encoder.pkl', 'rb'))

except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    sys.exit(1)


# Function to clean resume text
def cleanResume(txt):
    if not txt or not isinstance(txt, str):
        return ""
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText.strip()


# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + ' '
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")


# Function to extract text from DOCX
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = ''
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text += paragraph.text + '\n'
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting text from DOCX: {str(e)}")


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        # Try using utf-8 encoding for reading the text file
        content = file.read()
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            # In case utf-8 fails, try 'latin-1' encoding as a fallback
            file.seek(0)
            content = file.read()
            text = content.decode('latin-1')
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting text from TXT: {str(e)}")


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume with confidence scores
def pred(input_resume, return_probabilities=False):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume)

    # Validate input
    if not cleaned_text or len(cleaned_text.strip()) < 10:
        raise ValueError("Resume text is too short or empty after cleaning. Please provide more content.")

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense (only if needed)
    if hasattr(vectorized_text, 'toarray'):
        vectorized_text_dense = vectorized_text.toarray()
    else:
        vectorized_text_dense = vectorized_text

    # Prediction
    predicted_category = svc_model.predict(vectorized_text_dense)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    if return_probabilities:
        # Get decision function scores (distance from hyperplane)
        decision_scores = svc_model.decision_function(vectorized_text_dense)[0]

        # Get all category names
        all_categories = le.classes_

        # Create a dictionary of category scores
        # Handle both SVC (new model) and OneVsRestClassifier (old model)
        category_scores = {}

        if hasattr(svc_model, 'estimators_'):
            # Old model (OneVsRestClassifier)
            for estimator_idx, estimator in enumerate(svc_model.estimators_):
                category_idx = estimator_idx
                if category_idx < len(all_categories):
                    category_scores[all_categories[category_idx]] = decision_scores[category_idx]
        else:
            # New model (direct SVC) - decision_scores is already an array for all classes
            for idx, category in enumerate(all_categories):
                category_scores[category] = decision_scores[idx]

        # Sort by scores
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)

        return predicted_category_name[0], sorted_categories

    return predicted_category_name[0]  # Return the category name


# Streamlit app layout
def main():
    st.set_page_config(
        page_title="AI Resume Classifier",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .category-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            border-left: 5px solid #1f77b4;
            margin: 10px 0;
        }
        .stat-box {
            padding: 15px;
            border-radius: 8px;
            background-color: #e8f4f8;
            text-align: center;
            margin: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">🤖 AI-Powered Resume Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automatically categorize resumes into 25+ job categories using Machine Learning</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This application uses a Support Vector Machine (SVM) model with "
            "TF-IDF vectorization to classify resumes into 25 different job categories."
        )

        st.header("📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F1-Score", "99.47%")
        with col2:
            st.metric("Categories", "25")

        st.header("📝 Supported Formats")
        st.markdown("- 📄 PDF (.pdf)")
        st.markdown("- 📝 Word (.docx)")
        st.markdown("- 📃 Text (.txt)")

        st.header("🎯 Job Categories")
        categories = sorted(le.classes_)
        with st.expander("View All Categories"):
            for i, cat in enumerate(categories, 1):
                st.write(f"{i}. {cat}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Upload Resume")
        uploaded_files = st.file_uploader(
            "Choose resume file(s)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="You can upload multiple resumes at once"
        )

    with col2:
        st.header("⚙️ Options")
        show_text = st.checkbox("Show extracted text", value=False)
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_top_n = st.slider("Show top N predictions", 1, 5, 3)

    # Process uploaded files
    if uploaded_files:
        st.markdown("---")
        st.header("📊 Results")

        for idx, uploaded_file in enumerate(uploaded_files, 1):
            with st.expander(f"📄 {uploaded_file.name}", expanded=True):
                try:
                    # Extract text
                    with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                        resume_text = handle_file_upload(uploaded_file)

                    st.success(f"✅ Successfully extracted {len(resume_text)} characters")

                    # Display extracted text if requested
                    if show_text:
                        st.subheader("📝 Extracted Text")
                        st.text_area(
                            "Resume Content",
                            resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""),
                            height=200,
                            key=f"text_{idx}"
                        )

                    # Make prediction
                    with st.spinner("Analyzing resume..."):
                        if show_confidence:
                            category, scores = pred(resume_text, return_probabilities=True)
                        else:
                            category = pred(resume_text)

                    # Display primary prediction
                    st.markdown("### 🎯 Primary Prediction")
                    st.markdown(
                        f'<div class="category-box">'
                        f'<h2 style="color: #1f77b4; margin: 0;">{category}</h2>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Display confidence scores
                    if show_confidence:
                        st.markdown("### 📊 Confidence Scores")
                        st.write(f"Showing top {show_top_n} predictions:")

                        # Create columns for top predictions
                        cols = st.columns(min(show_top_n, 3))
                        for i, (cat, score) in enumerate(scores[:show_top_n]):
                            with cols[i % 3]:
                                # Normalize score for display (simple scaling)
                                normalized_score = max(0, min(100, (score + 10) * 5))
                                st.metric(
                                    label=cat,
                                    value=f"{normalized_score:.1f}",
                                    delta="Primary" if i == 0 else None
                                )

                        # Show detailed scores in a table (using checkbox instead of expander)
                        if st.checkbox("📈 View All Scores", key=f"scores_{idx}"):
                            import pandas as pd
                            scores_df = pd.DataFrame(scores[:10], columns=["Category", "Score"])
                            scores_df["Normalized Score"] = scores_df["Score"].apply(
                                lambda x: max(0, min(100, (x + 10) * 5))
                            )
                            st.dataframe(
                                scores_df.style.background_gradient(subset=["Score"], cmap="Blues"),
                                use_container_width=True
                            )

                    # Resume statistics
                    st.markdown("### 📈 Resume Statistics")
                    stat_cols = st.columns(4)
                    with stat_cols[0]:
                        st.metric("Characters", f"{len(resume_text):,}")
                    with stat_cols[1]:
                        word_count = len(resume_text.split())
                        st.metric("Words", f"{word_count:,}")
                    with stat_cols[2]:
                        cleaned = cleanResume(resume_text)
                        st.metric("Cleaned Words", f"{len(cleaned.split()):,}")
                    with stat_cols[3]:
                        st.metric("File Type", uploaded_file.name.split('.')[-1].upper())

                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    st.exception(e)

    else:
        # Show welcome message when no file is uploaded
        st.info("👆 Upload one or more resume files to get started!")

        # Show example
        with st.expander("💡 See Example"):
            st.markdown("""
            **Sample Resume Text:**

            ```
            Data Scientist with 5+ years of experience in machine learning,
            deep learning, and statistical analysis. Proficient in Python,
            TensorFlow, scikit-learn, and data visualization tools.

            Skills:
            - Machine Learning & Deep Learning
            - Python, R, SQL
            - TensorFlow, PyTorch, scikit-learn
            - Data Analysis & Visualization
            ```

            **Expected Output:** `Data Science`
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Built with Streamlit • Powered by SVM & TF-IDF • 99.47% F1-Score (Cross-Validated)"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
