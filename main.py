import streamlit as st
from PyPDF2 import PdfReader
from keybert import KeyBERT
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from matplotlib.patches import Circle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from pymongo import MongoClient
import base64

import nltk
nltk.download('vader_lexicon')


# # MongoDB connection
# client = MongoClient("mongodb+srv://ubaidkhanub5:#Besthacker234@cluster0.5nkhw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
# db = client['beta-user-data']
# collection = db['CA']

# Access secrets from the .streamlit/secrets.toml file
mongo_host = st.secrets["mongo"]["host"]
mongo_username = st.secrets["mongo"]["username"]
mongo_password = st.secrets["mongo"]["password"]
db_name = st.secrets["mongo"]["db_name"]
collection_name = st.secrets["mongo"]["collection_name"]

# MongoDB connection string using the credentials from the secrets file
client = MongoClient(f"mongodb+srv://{mongo_username}:{mongo_password}@{mongo_host}/?retryWrites=true&w=majority")

# Access the database and collection
db = client[db_name]
collection = db[collection_name]


# Function to compute keyword analysis
def keyword_analysis(resume_text, job_description_text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([job_description_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(vectors.sum(axis=0)).flatten()
    top_indices = np.argsort(scores)[::-1][:top_n]
    top_keywords = [feature_names[i] for i in top_indices]

    resume_vectorizer = CountVectorizer(stop_words='english')
    resume_vectors = resume_vectorizer.fit_transform([resume_text])
    resume_feature_names = resume_vectorizer.get_feature_names_out()
    
    resume_keyword_counts = dict(zip(resume_feature_names, resume_vectors.sum(axis=0).A1))
    
    keyword_analysis_df = pd.DataFrame(top_keywords, columns=['Keyword'])
    keyword_analysis_df['Matches'] = keyword_analysis_df['Keyword'].map(resume_keyword_counts).fillna(0)
    
    return keyword_analysis_df

# Function to compute semantic similarity using Sentence-BERT
def sentence_bert_similarity(resume_text, job_description_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = [resume_text, job_description_text]
    embeddings = model.encode(sentences)
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# # Function to compute readability score
# def readability_score(text):
#     scores = {
#         "Flesch-Kincaid": textstat.flesch_kincaid_grade(text),
#         "Gunning Fog": textstat.gunning_fog(text)
#     }
#     return scores

# Function to compute readability score
def readability_score_fk(cv_text, jd_text):
    scores = {
        "Flesch-Kincaid-cv": textstat.flesch_kincaid_grade(cv_text),
        "Flesch-Kincaid-jd": textstat.flesch_kincaid_grade(jd_text)
    }
    return scores

# Function to compute readability score
def readability_score_gf(cv_text, jd_text):
    scores = {
        "Gunning-Fog-cv": textstat.gunning_fog(cv_text),
        "Gunning-Fog-jd": textstat.gunning_fog(jd_text)
    }
    return scores

# Function to perform sentiment analysis
def sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    return sentiment

# Define the function to plot the sentiment meter using Plotly
# def plot_sentiment_meter(sentiment_score):
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=sentiment_score,
#         gauge={'axis': {'range': [-1, 1]},
#                'bar': {'color': "darkblue"},
#                'steps': [
#                    {'range': [-1, -0.5], 'color': "red"},
#                    {'range': [-0.5, 0.5], 'color': "yellow"},
#                    {'range': [0.5, 1], 'color': "green"}]},
#         domain={'x': [0, 1], 'y': [0, 1]},
#         title={'text': "Weighted Sentiment Intensity Meter"}))
    
#     return fig

def plot_sentiment_meter(sentiment_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        gauge={
            'axis': {'range': [-1, 1], 'tickcolor': 'black', 'tickwidth': 2},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0.5], 'color': "yellow"},
                {'range': [0.5, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {
                    'color': "white",  # Color of the needle
                    'width': 6           # Width of the needle
                },
                'thickness': 0.75,
                'value': sentiment_score
            }
        },
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Weighted Sentiment Intensity Meter"}
    ))

    return fig

# Function to compute the overall matching score
def compute_matching_score_with_keybert(resume_text, job_description_text):
    keyword_df = keyword_analysis(resume_text, job_description_text)
    sentence_bert_score = sentence_bert_similarity(resume_text, job_description_text)
    sentiment_scores = sentiment_analysis(resume_text)
        
    return {
        "Keyword Analysis": keyword_df,
        "Sentence-BERT Similarity": sentence_bert_score,
        "Sentiment Scores": sentiment_scores
    }




def draw_circles_with_scores_fk(fk_score_cv, fk_score_jd):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Flesch-Kincaid circle for Resume
    fk_circle = Circle((0.5, 0.5), 0.3, color='lightblue', ec='black')
    ax.add_patch(fk_circle)
    ax.text(0.5, 0.85, "Resume", color='black', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.5, f"{fk_score_cv:.2f}", color='black', fontsize=14, ha='center', va='center')
    
    # Flesch-Kincaid circle for Job-description
    gf_circle = Circle((1.5, 0.5), 0.3, color='lightcoral', ec='black')
    ax.add_patch(gf_circle)
    ax.text(1.5, 0.85, "Job-description", color='black', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(1.5, 0.5, f"{fk_score_jd:.2f}", color='black', fontsize=14, ha='center', va='center')
    
    plt.title("Flesch-Kincaid Scores", fontsize=16)
    plt.show()
    st.pyplot(fig)

def draw_circles_with_scores_gf(gf_score_cv, gf_score_jd):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Gunning-Fog circle for Resume
    fk_circle = Circle((0.5, 0.5), 0.3, color='lightblue', ec='black')
    ax.add_patch(fk_circle)
    ax.text(0.5, 0.85, "Resume", color='black', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.5, f"{gf_score_cv:.2f}", color='black', fontsize=14, ha='center', va='center')
    
    # Gunning-Fog circle for Job-description
    gf_circle = Circle((1.5, 0.5), 0.3, color='lightcoral', ec='black')
    ax.add_patch(gf_circle)
    ax.text(1.5, 0.85, "Job-description", color='black', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(1.5, 0.5, f"{gf_score_jd:.2f}", color='black', fontsize=14, ha='center', va='center')
    
    plt.title("Gunning-Fog Scores", fontsize=16)
    plt.show()
    st.pyplot(fig)

# Function to return interpretation of the Gunning Fog scores
def interpret_scores_gf(cv_score, jd_score):
    def get_gf_interpretation(gf_score, text_type):
        if gf_score <= 8:
            return f"**{text_type} Interpretation:** The Gunning Fog score is {gf_score:.2f}, indicating the text is very easy to digest. Ideal for ensuring quick and effortless comprehension by any reader."
        elif 8 <= gf_score <= 12:
            return f"**{text_type} Interpretation:** The score is {gf_score:.2f}, making the text readable for most with a high school education. It should pose no significant challenge for the average recruiter."
        elif 12 <= gf_score <= 16:
            return f"**{text_type} Interpretation:** The score of {gf_score:.2f} suggests the text requires a college-level reading ability. It's moderately complex and may require closer attention to detail."
        elif 16 <= gf_score <= 20:
            return f"**{text_type} Interpretation:** The score is {gf_score:.2f}, indicating the text is complex and may require a higher level of education to fully grasp. It's appropriate for a more specialized or educated audience."
        else:
            return f"**{text_type} Interpretation:** A score over 20 indicates the text is highly complex, potentially requiring postgraduate education to fully understand."

    # Interpretations for both resume and job description
    resume_interpretation = get_gf_interpretation(cv_score, "Resume")
    jd_interpretation = get_gf_interpretation(jd_score, "Job-description")
    
    # Comparison between resume and job description scores
    if cv_score < jd_score:
        comparison = f"**Comparison:** The resume is easier to read than the job description."
    elif cv_score > jd_score:
        comparison = f"**Comparison:** The resume is more challenging to read than the job description."
    else:
        comparison = f"**Comparison:** The resume and the job description have the same readability level."

    return resume_interpretation, jd_interpretation, comparison

# Function to return interpretation of the scores
def interpret_scores_fk(cv_score, jd_score):
    def get_fk_interpretation(fk_score, text_type):
        if fk_score <= 4:
            return f"**{text_type} Interpretation:** The Flesch-Kincaid score is {fk_score:.2f}, indicating the text is extremely easy to read. Suitable for a broad audience with minimal reading difficulty."
        elif 5 <= fk_score <= 8:
            return f"**{text_type} Interpretation:** The score is {fk_score:.2f}, meaning the text is straightforward, akin to middle school reading level. It should be easily comprehensible for most readers."
        elif 9 <= fk_score <= 12:
            return f"**{text_type} Interpretation:** The score is {fk_score:.2f}, suggesting high school-level readability. The text is moderately complex, requiring an average reading proficiency."
        elif 13 <= fk_score <= 16:
            return f"**{text_type} Interpretation:** The score is {fk_score:.2f}, indicating the text is on par with college-level material. It's more demanding and may require a focused, well-educated audience."
        else:
            return f"**{text_type} Interpretation:** The score of {fk_score:.2f} implies a highly challenging read, suitable for graduate-level comprehension."
    
    # Interpretations for both resume and job description
    resume_interpretation = get_fk_interpretation(cv_score, "Resume")
    jd_interpretation = get_fk_interpretation(jd_score, "Job-description")
    
    # Comparison between resume and job description scores
    if cv_score < jd_score:
        comparison = f"**Comparison:** The resume is easier to read than the job description."
    elif cv_score > jd_score:
        comparison = f"**Comparison:** The resume is more challenging to read than the job description."
    else:
        comparison = f"**Comparison:** The resume and the job description have the same readability level."

    return resume_interpretation, jd_interpretation, comparison




# Initialize session state for inputs
if 'text_input' not in st.session_state:
    st.session_state.text_input = ''
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False

# Title
st.title("Resume Analyzer")

# Text input
text = st.text_input("Enter the Job Description:")
st.session_state.text_input = text

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    st.session_state.pdf_uploaded = True

# Enable the 'Analyse' button only if both text and PDF are provided
if st.session_state.text_input and st.session_state.pdf_uploaded:
    analyse_button = st.button("Analyse", key="enabled", help="Click to analyze", disabled=False)
else:
    analyse_button = st.button("Analyse", key="disabled", help="Please enter text and upload a PDF", disabled=True)




# Button turns red when enabled
if analyse_button:
    st.markdown('<style>.stButton button{background-color: red;}</style>', unsafe_allow_html=True)
    
    # Process the PDF to extract text
    reader = PdfReader(uploaded_file)
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text()
    
    # Run the analysis
    results = compute_matching_score_with_keybert(resume_text, st.session_state.text_input)
    
    # Display CV Fit Score (Sentence-BERT Similarity) using Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=results['Sentence-BERT Similarity'] * 100,
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "blue"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "CV Fit Score"}))
    st.plotly_chart(fig)

    # Add a gap between the two sections
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)    




    # GF Readability
    # Calculate readability scores
    gf_scores = readability_score_gf(resume_text, st.session_state.text_input)
    # Draw circles with scores
    draw_circles_with_scores_gf(gf_scores["Gunning-Fog-cv"], gf_scores["Gunning-Fog-jd"])
    # Interpret the scores
    gf_interpretation_cv, gf_interpretation_jd, gf_comparison  = interpret_scores_gf(gf_scores["Gunning-Fog-cv"], gf_scores["Gunning-Fog-jd"])
    # Display interpretation in dialogue boxes
    st.info(gf_interpretation_cv)
    st.info(gf_interpretation_jd)
    st.info(gf_comparison)    
    # Add a gap between the two sections
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True) 

    # FK Readability
    # Calculate readability scores
    fk_scores = readability_score_fk(resume_text, st.session_state.text_input)
    # Draw circles with scores
    draw_circles_with_scores_fk(fk_scores["Flesch-Kincaid-cv"], fk_scores["Flesch-Kincaid-jd"])
    # Interpret the scores
    fk_interpretation_cv, fk_interpretation_jd, fk_comparison  = interpret_scores_fk(fk_scores["Flesch-Kincaid-cv"], fk_scores["Flesch-Kincaid-jd"])
    # Display interpretation in dialogue boxes
    st.info(fk_interpretation_cv)
    st.info(fk_interpretation_jd)
    st.info(fk_comparison)    
    # Add a gap between the two sections
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True) 




    # Display keyword analysis as a table
    st.markdown("<h2 style='text-align: center;'>Keyword Matching - Resume x JD</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><div style='display: inline-block; width: 80%;'>", unsafe_allow_html=True)
    st.dataframe(results["Keyword Analysis"], width=700)
    st.markdown("</div></div>", unsafe_allow_html=True)
    # Add a gap between the two sections
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True) 




    # Display sentiment scores in a table
    st.markdown("<h2 style='text-align: center;'>Sentiment Analysis of Resume</h2>", unsafe_allow_html=True)
    # Create the DataFrame and rename the Sentiment values
    sentiment_df = pd.DataFrame(list(results["Sentiment Scores"].items())[0:3], columns=["Sentiment", "Score"])
    # Rename the sentiment labels
    sentiment_df["Sentiment"] = sentiment_df["Sentiment"].map({
        "neg": "Negative",
        "neu": "Neutral",
        "pos": "Positive"
        # "compound": "Compound"
    })
    # Convert the scores to percentage format
    sentiment_df["Score"] = (sentiment_df["Score"] * 100).map("{:.2f} %".format)
    # Display the table
    st.table(sentiment_df)
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True) 

    # Plot sentiment meter using Plotly
    st.plotly_chart(plot_sentiment_meter(results["Sentiment Scores"]['compound']))



    pdf_base64 = base64.b64encode(uploaded_file.read()).decode('utf-8')

    keyword_analysis_dict = results["Keyword Analysis"].set_index('Keyword').to_dict(orient='index')
    keyword_analysis_dict = {str(k): v for k, v in keyword_analysis_dict.items()}

    document = {
        "job_description": st.session_state.text_input,
        "resume_text": resume_text,
        "pdf_file": pdf_base64,
        "metrics": {
            "Keyword Analysis": keyword_analysis_dict,
            "Sentence-BERT Similarity": float(results["Sentence-BERT Similarity"]),
            "Sentiment Scores": results["Sentiment Scores"],
            "Flesch-Kincaid Scores": fk_scores,
            "Gunning-Fog Scores": gf_scores
        }
    }
    collection.insert_one(document)

    st.markdown("---")
