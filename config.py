import os

BASE_DIR = "data"
USER_DATA_PATH = os.path.join(BASE_DIR, "USER_DATA_FINAL.csv")
JOB_DATA_PATH = os.path.join(BASE_DIR, "JOB_DATA_FINAL.csv")
SIMILARITY_PATH = "similarity_matrix.pkl"
USER_DF_PATH = "user_df.pkl"
JOB_DF_PATH = "job_df.pkl"

DEFAULT_WEIGHTS = {
    "industry": 0.3,
    "job": 0.25,
    "experience": 0.15,
    "salary": 0.1,
    "skills": 0.15,
    "degree": 0.05
}