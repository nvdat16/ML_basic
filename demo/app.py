import streamlit as st
import pandas as pd
import joblib

# Cấu hình Ứng dụng
st.set_page_config(page_title="Demo", layout="centered")

# Tải Mô hình và các thành phần tiền xử lý
model = joblib.load('svm.pkl')
scaler = joblib.load('scaler.pkl')

# Xây dựng Giao diện Người dùng (Input Sidebar)
st.title('Ứng dụng dự đoán độ hài lòng của khách hàng')
st.markdown('### Nhập thông tin khách hàng để dự đoán')

# Sử dụng Sidebar để giữ cho giao diện chính gọn gàng
st.sidebar.header('Thông tin khách hàng')

# Input 0: CreditScore
credit_score = st.sidebar.slider(
    '1. CreditScore (Điểm tín dụng)', 
    min_value=300, max_value=850, value=650
)

# Input 3: Age
age = st.sidebar.slider(
    '2. Age (Tuổi)', 
    min_value=18, max_value=100, value=40
)

# Input 4: Tenure
tenure = st.sidebar.slider(
    '3. Tenure (Số năm gắn bó)', 
    min_value=0, max_value=10, value=5
)

# Input 5: Balance
balance = st.sidebar.number_input(
    '4. Balance (Số dư tài khoản)', 
    min_value=0.0, value=50000.0, step=1000.0
)

# Input 6: NumOfProducts
num_of_products = st.sidebar.slider(
    '5. NumOfProducts (Số lượng sản phẩm)', 
    min_value=1, max_value=4, value=1
)

# Input 9: EstimatedSalary
estimated_salary = st.sidebar.number_input(
    '6. EstimatedSalary (Lương ước tính)', 
    min_value=0.0, value=100000.0, step=1000.0
)

st.sidebar.markdown('---')
st.sidebar.subheader('Dữ liệu Phân loại & Nhị phân')

# Input 1: Geography
GEOGRAPHY_OPTIONS = ['France', 'Spain', 'Germany']
geography = st.sidebar.selectbox(
    '7. Geography (Quốc gia)', 
    GEOGRAPHY_OPTIONS
)

# Input 2: Gender
GENDER_OPTIONS = ['Male', 'Female']
gender = st.sidebar.selectbox(
    '8. Gender (Giới tính)', 
    GENDER_OPTIONS
)

# Input 7: HasCrCard
has_cr_card = st.sidebar.selectbox(
    '9. HasCrCard (Có thẻ tín dụng?)', 
    options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'
)

# Input 8: IsActiveMember
is_active_member = st.sidebar.selectbox(
    '10. IsActiveMember (Thành viên tích cực?)', 
    options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'
)

if st.button('Dự đoán Rủi ro'):
    
    # Tạo DataFrame từ đầu vào
    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }
    
    input_df = pd.DataFrame([data])
    
    st.subheader('Dữ liệu khách hàng')
    st.dataframe(input_df, hide_index=True)

    # Chuẩn hóa (Scaling) dữ liệu số liệu
    numerical_cols = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts','EstimatedSalary']
    df_num_raw = input_df[numerical_cols]
    scaled_data = scaler.transform(df_num_raw)
    df_num_scaled = pd.DataFrame(scaled_data, columns=numerical_cols)

    # Mã hóa (Encoding) dữ liệu phân loại
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['Geography'] = input_df['Geography'].map({'France': 0, 'Spain': 2, 'Germany': 1})
    categorical_cols = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
    df_cat_encoded = input_df[categorical_cols]

    # Ghép nối dữ liệu
    final_df = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
    features = final_df.values 
        
    # Dự đoán xác suất
    predict = model.predict(features)

    st.markdown('---')
    st.subheader(f'Kết quả dự đoán: {predict}')