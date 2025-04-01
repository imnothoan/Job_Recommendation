import streamlit as st
import pandas as pd
import pickle
from recommendation import recommend_jobs
from config import *

try:
    with open(SIMILARITY_PATH, "rb") as file:
        similarity_matrix = pickle.load(file)
    user_df = pd.read_pickle(USER_DF_PATH)
    job_df = pd.read_pickle(JOB_DF_PATH)
except FileNotFoundError:
    st.error("không tìm được dữ liệu đã train")
    st.stop()

st.title("🔍 Job Recommend System")

user_id = st.number_input("Input User ID:", min_value=1, step=1)

if st.button("Suggest"):
    try:
        user_name, desired_job, workplace_desired, recommended_jobs = recommend_jobs(
            user_id, user_df, job_df, similarity_matrix, n=5
        )
        st.write(f"👨‍💼 Customer Information:\n\nName: {user_name}\n\nDesired job: {desired_job}\n\nWorkplace desired: {workplace_desired}")
        st.write("📋 List of job recommended:")

        if not recommended_jobs:
            st.write("No valid job found!")
        else:
            for i, job in enumerate(recommended_jobs, 1):
                st.write(f"**#{i} - Job title: {job['job_title']} at {job['company_name']}**")
                st.write(f"Reason: {job['reason']}")
                # st.write(f"Điểm khớp: {job['matching_score']}")
                st.write("---")
    except ValueError as e:
        st.error(str(e))