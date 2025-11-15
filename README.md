# **Fake Job Detection using Machine Learning & Deep Learning**
A complete end-to-end AI system that identifies fraudulent job postings using Machine Learning (TF-IDF + Logistic Regression), Deep Learning (Bi-LSTM), and a Hybrid approach.
This project includes data preprocessing, modeling, evaluation, deployment, and documentation.

🔗 **Live App**: https://fakejobdetection01.streamlit.app/

📘 **Medium Article:** https://medium.com/p/c04410608b38/edit

💻 **GitHub Repository:** https://github.com/HiraArif666/FakeJobDetection-main

# **Problem Statement**
Online job portals (LinkedIn, Indeed, Google Jobs) face an increasing number of fake job postings created to scam users, steal personal information, or promote malicious links.
Manual moderation is slow and unreliable.

**Goal**: Build an AI system that can automatically analyze job descriptions and classify them as Real or Fake, improving safety for job seekers and recruitment platforms.

 # **Dataset**

**Dataset Name:** Fake Job Posting Prediction

**Source:** GitHub
🔗 https://github.com/TharunKumarReddyN/Fake-Job-Posting-Prediction/tree/master/data

**Rows**: ~18,000 job postings

**Target Column:** fraudulent (0 = real, 1 = fake)

**Key Features:**
1. title
2. company_profile
3. description
4. requirements
5. employment_type
6. fraudulent

Dataset is imbalanced, with only ~8% fake posts → handled using preprocessing + model weighting.

# **Models Used**

## **1. TF-IDF + Logistic Regression (Machine Learning)**
1. Converts text into numerical vectors based on word importance
2. Fast, interpretable, strong baseline
3. Achieved **~97.2%** accuracy

## **2. Bi-LSTM (Deep Learning)**
1. Learns context + sequence of words
2. Processes text in forward + backward directions
3. Captures scam-like writing patterns
4. Achieved ~97.3% accuracy


 # **Deployment**
Final model deployed on Streamlit Cloud.

**📌 Live App Link:**
👉 https://fakejobdetection01.streamlit.app/

# **Features of Web App:**
1. Text input for job descriptions
2. Predicts “Real” or “Fake”
3. Clean UI ready for demonstration/interviews

# **Project Workflow**
                 ┌────────────────────┐
                 │    DATASET         │
                 │ (Fake Job Posts)   │
                 └─────────┬──────────┘
                           │
                     Data Cleaning
                           │
                  Feature Engineering
                           │
            ┌──────────────┴────────────────┐
            │                               │
    TF-IDF Vectorization             Text Tokenization
    (Logistic Regression Model)         (Embedding + Bi-LSTM)
            │                               │
            └───────────────┬───────────────┘
                             │
                        Evaluation
                             │
                        Deployment
                     (Streamlit App)



# **Performance Summary**

| Model                          | Accuracy | Precision | Recall | F1-Score |
|--------------------------------|----------|-----------|--------|----------|
| TF-IDF + Logistic Regression   | 97.26%   | 96.8%     | 96.4%  | 96.6%    |
| Bi-LSTM (Deep Learning)        | 97.37%   | 97.0%     | 96.9%  | 97.0%    |


Both models perform extremely well → validates dataset quality + preprocessing steps.

# **How to Run Locally**
**1. Clone the Repository**

git clone https://github.com/HiraArif666/FakeJobDetection-main
cd FakeJobDetection-main

**2. Create Virtual Environment (Optional but recommended)**

python -m venv venv
venv\Scripts\activate

**3. Install Requirements**

pip install -r requirements.txt

**4. Run the Streamlit App**

cd app
streamlit run app.py

App will open automatically in your browser.

# **📂 Folder Structure**

``` 
FakeJobDetection-main/
│
├── app/
│   └── app.py
│
├── data/
│   └── fake_job_postings.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_model.pkl
│   ├── tokenizer.pkl
│   ├── bilstm_model.h5
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── reports/
│   └── final_report.pdf
│
├── requirements.txt
└── README.md
``` 
# **📚 Medium Article**

Full project explanation with visuals & insights:
👉 https://medium.com/p/c04410608b38/edit



# **🙌 Acknowledgements**

1. Dataset provided by GitHub open-source contributors

2. Streamlit Cloud for free deployment

3. TensorFlow & Scikit-learn teams

4. Buildables DS Fellowship inspiration
