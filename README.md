# Dự đoán sự hài lòng của khách hàng ngân hàng

## 1. Giới thiệu đề tài
Bài toán dự đoán sự hài lòng của khách hàng ngân hàng nhằm hỗ trợ các tổ chức tài chính hiểu rõ hơn hành vi và mức độ hài lòng của khách hàng đối với các sản phẩm, dịch vụ ngân hàng. Mục tiêu của bài toán là áp dụng các kỹ thuật học máy để dự đoán sớm mức độ hài lòng của khách hàng dựa trên các thông tin thu thập được. Từ đó, ngân hàng có thể chủ động cải thiện chất lượng dịch vụ, xây dựng các chiến lược chăm sóc khách hàng phù hợp và nâng cao khả năng giữ chân khách hàng.

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
- Bộ dữ liệu gồm 10000 bản ghi và 14 thuộc tính sau:

| Tên cột | Kiểu dữ liệu | Mô tả |
|-------|------------|------|
| RowNumber | int | Số thứ tự |
| CustomerID | int | Mã khách hàng |
| Surname | object | Họ khách hàng |
| Creditscore | int | Điểm tín dụng |
| Geography | object | Quốc gia |
| Gender | object | Giới tính |
| Age | int | Tuổi khách hàng |
| Tenure | int | Số năm gắn bó với ngân hàng |
| Balance | float | Số dư tài khoản trung bình |
| NumOfProducts | int | Số lượng sản phẩm/dịch vụ đang sử dụng |
| HashCrCard | int | Có thẻ tính dụng |
| IsActiveMemer | int | Thành viên tích cực |
| EstimatedSalary | float | Mức lương ước tính |
| Exited | int | Rời bỏ ngân hàng (Nhãn dự đoán) |

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

### Kết quả trước khi xử lý mất cân bằng dữ liệu
| Model | Accuracy | Precision | Recall | F1-score |
|---|----------|-----------|--------|----------|
| Logistic Regression | 0.81 | 0.59 | 0.19 | 0.28 |
| SVM | 0.86 | 0.85 | 0.39 | 0.54 |
| Random Forest | 0.86 | 0.77 | 0.45 | 0.57 |

### Kết quả sau khi xử lý mất cân bằng dữ liệu
| Model | Accuracy | Precision | Recall | F1-score |
|---|----------|-----------|--------|----------|
| Logistic Regression | 0.72 | 0.39 | 0.70 | 0.50 |
| SVM | 0.79 | 0.49 | 0.71 | 0.58 |
| Random Forest | 0.84 | 0.62 | 0.58 | 0.60 |

---

## 6. Hướng dẫn chạy dự án

### 6.1 Cài đặt môi trường

```bash
git clone https://github.com/nvdat16/ML_basic.git
cd ML_basic
pip install -r requirements.txt
```

### 6.2 Training

#### Logistic Regresion
```bash
python app/train.py --model 'lr'
```
#### SVM
```bash
python app/train.py --model 'svm'
```
#### Random Forest
```bash
python app/train.py --model 'rf'
```

### 6.3 Prediction
```bash
python app/predict.py --model 'app/checkpoints/{lr/svm/rf}.pkl' --input_path 'input.csv'
```

### 6.4 Demo
```bash
python demo/app.py
```

---

## Tác giả 
- Họ và tên: Nguyễn Văn Đạt
- Mã sinh viên: 12423061
- Lớp: 12423TN