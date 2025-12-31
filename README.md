# Dự đoán sự hài lòng của khách hàng ngân hàng

## 1. Giới thiệu đề tài
Trong những năm gần đây, dữ liệu ngày càng gia tăng nhanh chóng, đặt ra nhu cầu cấp thiết về các phương pháp phân tích và dự đoán hiệu quả.  
Dự án này tập trung giải quyết **bài toán [phân loại / hồi quy / nhận dạng / dự đoán]** bằng cách áp dụng các kỹ thuật **Machine Learning**.

### Mục tiêu đề tài
- Xây dựng pipeline xử lý dữ liệu hoàn chỉnh
- Huấn luyện và đánh giá mô hình Machine Learning
- Triển khai demo dự đoán (inference) trên dữ liệu mới
- So sánh và đánh giá hiệu quả mô hình bằng các metric phù hợp

---

## 2. Dataset

### Nguồn dữ liệu
- Tên dataset: **Churn Modelling**
- Nguồn: **Kaggle**
- Link tải: https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers


### Mô tả dữ liệu
| Tên cột | Kiểu dữ liệu | Mô tả |
|-------|------------|------|
| feature_1 | float | Mô tả |
| feature_2 | int | Mô tả |
| label | int | Nhãn |

---

## 3. Pipeline hệ thống

Quy trình xử lý và huấn luyện mô hình bao gồm các bước sau:

1. **Tiền xử lý dữ liệu**
   - Làm sạch dữ liệu
   - Xử lý missing values
   - Chuẩn hóa / mã hóa dữ liệu
   - Xử lý mất cân bằng dữ liệu

2. **Huấn luyện mô hình**
   - Chia tập train / test
   - Huấn luyện mô hình Machine Learning

3. **Đánh giá mô hình**
   - Tính toán các metric đánh giá
   - So sánh kết quả

4. **Inference / Demo**
   - Dự đoán trên dữ liệu mới

---

## 4. Mô hình sử dụng

Dự án sử dụng các mô hình sau:

- **Logistic Regression (LR)**
- **Decision Tree (DT)**
- **Random Forest (RF)**

### Lý do lựa chọn
- LR: mô hình cơ bản, dễ giải thích
- DT: mô hình phi tuyến, trực quan
- RF: cải thiện độ chính xác, giảm overfitting

---

## 5. Kết quả

### Metric đánh giá
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

### Kết qủa sau khi xử lý mất cân bằng dữ liệu
|  | Accuracy | Precision | Recall | F1-score |
|---|----------|-----------|--------|----------|
| Logistic Regression | 0.72 | 0.39 | 0.70 | 0.50 |
| SVM | 0.79 | 0.49 | 0.71 | 0.58 |
| Random Forest | 0.84 | 0.62 | 0.58 | 0.60 |

---

## 6. Hướng dẫn chạy dự án

### 6.1 Cài đặt môi trường

```bash
git clone https://github.com/username/project-name.git
cd project-name
pip install -r requirements.txt

