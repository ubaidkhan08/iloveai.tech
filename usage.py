from PyPDF2 import PdfReader
from keybert import KeyBERT
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

class ResumeAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sid = SentimentIntensityAnalyzer()

    def keyword_analysis(self, resume_text, job_description_text, top_n=10):
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

    def sentence_bert_similarity(self, resume_text, job_description_text):
        sentences = [resume_text, job_description_text]
        embeddings = self.model.encode(sentences)
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def readability_score_fk(self, cv_text, jd_text):
        scores = {
            "Flesch-Kincaid-cv": textstat.flesch_kincaid_grade(cv_text),
            "Flesch-Kincaid-jd": textstat.flesch_kincaid_grade(jd_text)
        }
        return scores

    def readability_score_gf(self, cv_text, jd_text):
        scores = {
            "Gunning-Fog-cv": textstat.gunning_fog(cv_text),
            "Gunning-Fog-jd": textstat.gunning_fog(jd_text)
        }
        return scores

    def sentiment_analysis(self, text):
        return self.sid.polarity_scores(text)

    def compute_matching_score_with_keybert(self, resume_text, job_description_text):
        keyword_df = self.keyword_analysis(resume_text, job_description_text)
        sentence_bert_score = self.sentence_bert_similarity(resume_text, job_description_text)
        sentiment_scores = self.sentiment_analysis(resume_text)
        
        return {
            "Keyword Analysis": keyword_df,
            "Sentence-BERT Similarity": sentence_bert_score,
            "Sentiment Scores": sentiment_scores
        }

    def interpret_scores_fk(self, cv_score, jd_score):
        def get_fk_interpretation(fk_score, text_type):
            if fk_score <= 4:
                return f"{text_type} Interpretation: The Flesch-Kincaid score is {fk_score:.2f}, indicating the text is extremely easy to read."
            elif 5 <= fk_score <= 8:
                return f"{text_type} Interpretation: The score is {fk_score:.2f}, meaning the text is straightforward."
            elif 9 <= fk_score <= 12:
                return f"{text_type} Interpretation: The score is {fk_score:.2f}, suggesting high school-level readability."
            elif 13 <= fk_score <= 16:
                return f"{text_type} Interpretation: The score is {fk_score:.2f}, indicating the text is on par with college-level material."
            else:
                return f"{text_type} Interpretation: The score of {fk_score:.2f} implies a highly challenging read."
        
        resume_interpretation = get_fk_interpretation(cv_score, "Resume")
        jd_interpretation = get_fk_interpretation(jd_score, "Job-description")
        
        if cv_score < jd_score:
            comparison = "The resume is easier to read than the job description."
        elif cv_score > jd_score:
            comparison = "The resume is more challenging to read than the job description."
        else:
            comparison = "The resume and the job description have the same readability level."

        return resume_interpretation, jd_interpretation, comparison

    def interpret_scores_gf(self, cv_score, jd_score):
        def get_gf_interpretation(gf_score, text_type):
            if gf_score <= 8:
                return f"{text_type} Interpretation: The Gunning Fog score is {gf_score:.2f}, indicating the text is very easy to digest."
            elif 8 <= gf_score <= 12:
                return f"{text_type} Interpretation: The score is {gf_score:.2f}, making the text readable for most."
            elif 12 <= gf_score <= 16:
                return f"{text_type} Interpretation: The score of {gf_score:.2f} suggests college-level readability."
            else:
                return f"{text_type} Interpretation: A score over 16 indicates high complexity."
        
        resume_interpretation = get_gf_interpretation(cv_score, "Resume")
        jd_interpretation = get_gf_interpretation(jd_score, "Job-description")

        if cv_score < jd_score:
            comparison = "The resume is easier to read than the job description."
        elif cv_score > jd_score:
            comparison = "The resume is more challenging to read than the job description."
        else:
            comparison = "The resume and the job description have the same readability level."
        
        return resume_interpretation, jd_interpretation, comparison
