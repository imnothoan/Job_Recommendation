import pandas as pd
import numpy as np
from config import *

def calculate_matching_score(user_row, job_row, similarity_matrix, user_index, job_index):
    """Tính điểm khớp chi tiết giữa user và job."""
    score = similarity_matrix[user_index][job_index]  # Điểm tổng từ similarity_matrix

    # Các tiêu chí bổ sung để tinh chỉnh
    reasons = []
    max_criteria = 6  # Số tiêu chí tối đa: industry, job, workplace, salary, skills, degree
    matched_criteria = 0

    # Tiêu chí 1: Industry (đã có trong similarity_matrix)
    if user_row["Industry"] == job_row["Industry"]:
        matched_criteria += 1

    # Tiêu chí 2: Workplace
    if user_row["Workplace Desired"] == job_row["Job Address"]:
        matched_criteria += 1
        reasons.append(f"Vị trí gần ({job_row['Job Address']})")

    # Tiêu chí 3: Job Title (dùng similarity_matrix thay vì khớp chính xác)
    job_similarity = similarity_matrix[user_index][job_index] / DEFAULT_WEIGHTS["job"]  # Chuẩn hóa lại từ final_similarity
    if job_similarity > 0.7:  # Ngưỡng tương đồng tối thiểu
        matched_criteria += 1
        if user_row["Desired Job"].lower() in job_row["Job Title"].lower():
            reasons.append(f"Công việc đúng mong muốn ({job_row['Job Title']})")
        else:
            reasons.append(f"Công việc tương đồng ({job_row['Job Title']})")

    # Tiêu chí 4: Salary
    if not pd.isna(user_row["Desired Salary"]) and not pd.isna(job_row["Salary"]):
        if job_row["Salary"] >= user_row["Desired Salary"] * 0.9:
            matched_criteria += 1
            reasons.append(f"Mức lương phù hợp ({job_row['Salary']} VND)")

    # Tiêu chí 5: Skills
    if user_row["Skills"].lower() in job_row["Job Requirements"].lower():
        matched_criteria += 1
        reasons.append("Kỹ năng phù hợp với yêu cầu công việc")

    # Tiêu chí 6: Degree
    if user_row["Degree"].lower() in job_row["Job Requirements"].lower():
        matched_criteria += 1
        reasons.append("Bằng cấp đáp ứng yêu cầu")

    # Tính tỷ lệ khớp (0-1)
    match_ratio = matched_criteria / max_criteria
    final_score = score * 0.7 + match_ratio * 0.3  # Kết hợp similarity_matrix (70%) và tỷ lệ khớp (30%)

    reason = " | ".join(reasons) if reasons else "Đề xuất dựa trên độ tương đồng tổng thể"
    return final_score, reason

def recommend_jobs(user_id, user_df, job_df, similarity_matrix, n=5):

    if user_id not in user_df["UserID"].values:
        raise ValueError(f"UserID {user_id} không tồn tại trong dữ liệu!")

    user_index = user_df[user_df["UserID"] == user_id].index[0]
    user_row = user_df.loc[user_index]

    # Lọc job cơ bản: ít nhất phải khớp Industry
    valid_jobs = job_df[job_df["Industry"] == user_row["Industry"]]
    if valid_jobs.empty:
        return user_row["User Name"], user_row["Desired Job"], user_row["Workplace Desired"], []

    # Tính điểm khớp cho từng job
    scores_and_reasons = []
    for job_index, job_row in valid_jobs.iterrows():
        score, reason = calculate_matching_score(user_row, job_row, similarity_matrix, user_index, job_row.name)
        scores_and_reasons.append((score, job_row, reason))

    # Sắp xếp theo điểm từ cao xuống thấp
    scores_and_reasons.sort(key=lambda x: x[0], reverse=True)
    top_n = scores_and_reasons[:n]

    # Tạo danh sách gợi ý
    recommendations = []
    for score, job_row, reason in top_n:
        recommendations.append({
            "job_id": job_row["JobID"],
            "job_title": job_row["Job Title"],
            "company_name": job_row["Name Company"],
            "reason": reason,
            "matching_score": round(score, 2)  # Thêm điểm để kiểm tra
        })

    return (
        user_row["User Name"],
        user_row["Desired Job"],
        user_row["Workplace Desired"],
        recommendations
    )