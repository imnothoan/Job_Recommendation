import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from underthesea import word_tokenize
from config import *

def preprocess_text(text):
    if pd.isna(text):
        return ""
    return word_tokenize(str(text).lower().strip(), format="text")

def extract_salary(salary_series):
    def parse_salary(s):
        if pd.isna(s) or s == "Thỏa thuận":
            return np.nan
        s = s.replace(",", "").replace(" triệu", "").replace("Trên ", "").strip()
        parts = s.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2 if len(parts) == 2 else float(parts[0])
        except (ValueError, IndexError):
            return np.nan
    return pd.to_numeric(salary_series.apply(parse_salary), errors="coerce")

def extract_experience(exp_series):
    def parse_exp(s):
        if pd.isna(s) or s in ["Không yêu cầu", "Chưa có kinh nghiệm", "Không yêu cầu kinh nghiệm"]:
            return 0
        if s == "Dưới 1 năm":
            return 0.5
        s = s.replace(" năm", "").replace("Trên ", "").replace("Dưới ", "").strip()
        parts = s.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2 if len(parts) == 2 else float(parts[0])
        except (ValueError, IndexError):
            return 0
    return pd.to_numeric(exp_series.apply(parse_exp), errors="coerce")

def calculate_similarity(user_df, job_df, weights=DEFAULT_WEIGHTS):

    vectorizer_industry = TfidfVectorizer()
    vectorizer_job = TfidfVectorizer()
    vectorizer_skills = TfidfVectorizer()
    vectorizer_degree = TfidfVectorizer()

    user_industry_tfidf = vectorizer_industry.fit_transform(user_df["Industry"])
    job_industry_tfidf = vectorizer_industry.transform(job_df["Industry"])

    user_job_tfidf = vectorizer_job.fit_transform(user_df["Desired Job"])
    job_job_tfidf = vectorizer_job.transform(job_df["Job Title"])

    user_skills_tfidf = vectorizer_skills.fit_transform(user_df["Skills"])
    job_skills_tfidf = vectorizer_skills.transform(job_df["Job Requirements"])

    user_degree_tfidf = vectorizer_degree.fit_transform(user_df["Degree"])
    job_degree_tfidf = vectorizer_degree.transform(job_df["Job Requirements"])

    industry_similarity = cosine_similarity(user_industry_tfidf, job_industry_tfidf)
    job_similarity = cosine_similarity(user_job_tfidf, job_job_tfidf)
    skills_similarity = cosine_similarity(user_skills_tfidf, job_skills_tfidf)
    degree_similarity = cosine_similarity(user_degree_tfidf, job_degree_tfidf)

    salary_similarity = np.where(
        np.isnan(user_df["Desired Salary"].values[:, None]) | np.isnan(job_df["Salary"].values[None, :]),
        1.0,
        1 - np.abs(user_df["Desired Salary"].values[:, None] - job_df["Salary"].values[None, :]) / max(job_df["Salary"].max(), 1)
    )
    experience_similarity = 1 - np.abs(user_df["Work Experience"].values[:, None] - job_df["Years of Experience"].values[None, :]) / max(job_df["Years of Experience"].max(), 1)

    final_similarity = (
            weights["industry"] * industry_similarity +
            weights["job"] * job_similarity +
            weights["experience"] * experience_similarity +
            weights["salary"] * salary_similarity +
            weights["skills"] * skills_similarity +
            weights["degree"] * degree_similarity
    )
    return final_similarity

def main():
    user_df = pd.read_csv(USER_DATA_PATH)
    job_df = pd.read_csv(JOB_DATA_PATH)

    user_df["Industry"] = user_df["Industry"].apply(preprocess_text)
    job_df["Industry"] = job_df["Industry"].apply(preprocess_text)
    user_df["Desired Job"] = user_df["Desired Job"].apply(preprocess_text)
    job_df["Job Title"] = job_df["Job Title"].apply(preprocess_text)
    user_df["Skills"] = user_df["Skills"].apply(preprocess_text)
    job_df["Job Requirements"] = job_df["Job Requirements"].apply(preprocess_text)
    user_df["Degree"] = user_df["Degree"].apply(preprocess_text)

    user_df["Desired Salary"] = extract_salary(user_df["Desired Salary"])
    job_df["Salary"] = extract_salary(job_df["Salary"])
    user_df["Work Experience"] = extract_experience(user_df["Work Experience"])
    job_df["Years of Experience"] = extract_experience(job_df["Years of Experience"])

    user_df["Desired Salary"] = user_df["Desired Salary"].fillna(user_df["Desired Salary"].median())
    job_df["Salary"] = job_df["Salary"].fillna(job_df["Salary"].median())
    user_df["Work Experience"] = user_df["Work Experience"].fillna(user_df["Work Experience"].median())
    job_df["Years of Experience"] = job_df["Years of Experience"].fillna(job_df["Years of Experience"].median())

    similarity_matrix = calculate_similarity(user_df, job_df)

    with open(SIMILARITY_PATH, "wb") as file:
        pickle.dump(similarity_matrix, file)
    user_df.to_pickle(USER_DF_PATH)
    job_df.to_pickle(JOB_DF_PATH)

    print("model trained")

if __name__ == "__main__":
    main()