# Initialize the class
analyzer = ResumeAnalyzer()

# Sample texts
resume_text = "Sample resume text..."
job_description_text = "Sample job description text..."

# Perform keyword analysis
keywords_df = analyzer.keyword_analysis(resume_text, job_description_text)

# Compute sentence BERT similarity
similarity_score = analyzer.sentence_bert_similarity(resume_text, job_description_text)

# Compute readability scores (Flesch-Kincaid)
readability_scores_fk = analyzer.readability_score_fk(resume_text, job_description_text)

# Compute readability scores (Gunning Fog)
readability_scores_gf = analyzer.readability_score_gf(resume_text, job_description_text)

# Perform sentiment analysis
sentiment = analyzer.sentiment_analysis(resume_text)

# Compute matching score using KeyBERT
matching_score = analyzer.compute_matching_score_with_keybert(resume_text, job_description_text)

# Interpret Flesch-Kincaid scores
fk_interpretation = analyzer.interpret_scores_fk(readability_scores_fk['Flesch-Kincaid-cv'], readability_scores_fk['Flesch-Kincaid-jd'])

# Interpret Gunning Fog scores
gf_interpretation = analyzer.interpret_scores_gf(readability_scores_gf['Gunning-Fog-cv'], readability_scores_gf['Gunning-Fog-jd'])
