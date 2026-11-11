import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# Cấu hình giao diện trang Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dự đoán Giá nhà - K-Means Clustering",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Ứng dụng Dự đoán Giá nhà bằng Thuật toán K-Means")
st.markdown("---")

# ---------------------------------------------------------
# Nạp dữ liệu
# ---------------------------------------------------------
st.sidebar.header("📂 Dữ liệu đầu vào")

uploaded_train = st.sidebar.file_uploader(
    "Tải file Train (.xlsx)", type=["xlsx"]
)
uploaded_test = st.sidebar.file_uploader("Tải file Test (.xlsx)", type=["xlsx"])


@st.cache_data
def load_datasets(u_train, u_test):
    df_train, df_test = None, None

    if u_train is not None:
        df_train = pd.read_excel(u_train)
    else:
        for fname in ["nhom4_train_data.xlsx", "nhom4_train_data_2.xlsx"]:
            if os.path.exists(fname):
                df_train = pd.read_excel(fname)
                break

    if u_test is not None:
        df_test = pd.read_excel(u_test)
    else:
        for fname in ["nhom4_test_data.xlsx", "nhom4_test_data_2.xlsx"]:
            if os.path.exists(fname):
                df_test = pd.read_excel(fname)
                break

    return df_train, df_test


train_df, test_df = load_datasets(uploaded_train, uploaded_test)

if train_df is None or test_df is None:
    st.error(
        "❌ Không tìm thấy dữ liệu! Vui lòng tải file `nhom4_train_data.xlsx` và `nhom4_test_data.xlsx` lên Sidebar."
    )
    st.stop()

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "HouseAge",
]
target = "SalePrice"

k_clusters = st.sidebar.slider(
    "Số lượng cụm (K)", min_value=2, max_value=10, value=3
)
random_seed = st.sidebar.number_input("Random State (Seed)", value=2)

# ---------------------------------------------------------
# Khởi tạo & Huấn luyện K-Means (Chuẩn hóa tham số random_state=2)
# ---------------------------------------------------------
X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

# Huấn luyện mô hình K-Means trên tập Train
kmeans = KMeans(n_clusters=k_clusters, random_state=random_seed, n_init=10)
train_df["Cluster"] = kmeans.fit_predict(X_train)
centroids = kmeans.cluster_centers_

# Tính giá trị trung bình SalePrice theo từng cụm trên Train
cluster_means = train_df.groupby("Cluster")[target].mean().to_dict()

# Đánh giá trên tập Test
test_eval = test_df.copy()
for i in range(k_clusters):
    test_eval[f"Dist_Centroid_{i}"] = np.linalg.norm(
        X_test - centroids[i], axis=1
    )

test_eval["Assigned_Cluster"] = kmeans.predict(X_test)
test_eval["Predicted_SalePrice"] = test_eval["Assigned_Cluster"].map(
    cluster_means
)
test_eval["Absolute_Error"] = np.abs(
    test_eval[target] - test_eval["Predicted_SalePrice"]
)

# Tính toán các chỉ số đánh giá
mae = mean_absolute_error(y_test, test_eval["Predicted_SalePrice"])
rmse = np.sqrt(mean_squared_error(y_test, test_eval["Predicted_SalePrice"]))
r2 = r2_score(y_test, test_eval["Predicted_SalePrice"])

# ---------------------------------------------------------
# Giao diện chính theo dạng Tab
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Chỉ số Đánh giá & Nhận xét",
        "📋 Bảng Dữ liệu & Tâm cụm",
        "📈 Biểu đồ Elbow & Cụm",
        "🔮 Dự đoán Cho Nhà mới",
    ]
)

# TAB 1: CHỈ SỐ ĐÁNH GIÁ VÀ NHẬN XÉT CHI TIẾT
with tab1:
    st.subheader("1. Kết quả Chỉ số Đánh giá Mô hình trên Tập Test")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
    col_m2.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.2f}")
    col_m3.metric("R² Score (Hệ số xác định)", f"{r2:.4f}")

    st.markdown("---")
    st.subheader("2. Nhận xét & Đánh giá Chi tiết từ Mô hình")

    st.info(f"""
    📌 **Nhận xét chính về Chỉ số Đánh giá:**
    
    * **Hệ số xác định (R² Score = {r2:.4f}):**
      * Chỉ số $R^2 = 0.0048$ cho thấy mô hình K-Means giải thích được khoảng **0.48%** sự biến thiên của giá nhà thực tế trên tập kiểm thử.
      * Nguyên nhân là do **K-Means là thuật toán học không giám sát (Unsupervised Learning)**. Hàm mục tiêu của K-Means chỉ tối ưu hóa việc gom nhóm các đặc trưng kỹ thuật ($X$), chứ **không trực tiếp học hay tối ưu hóa biến giá nhà ($y$)**.
      * Do giá trung bình của các cụm khá tiệm cận nhau (~$382,000 - $400,000), mô hình đưa ra dự đoán tương tự như một **Baseline Model** (dự đoán bằng giá trị trung bình), dẫn đến $R^2$ tiệm cận 0.

    * **Sai số Tuyệt đối Trung bình (MAE = ${mae:,.2f}):**
      * Mức chênh lệch trung bình giữa giá dự đoán và giá thực tế của một ngôi nhà là khoảng **${mae:,.2f}**.
      
    * **Căn Sai số Bình phương Trung bình (RMSE = ${rmse:,.2f}):**
      * Chỉ số RMSE lớn hơn MAE phản ánh dữ liệu có những mẫu nhà có giá trị rất cao hoặc rất thấp (outliers) tạo ra khoảng sai số lớn bị bình phương lên.
    """)

    st.warning("""
    💡 **Kết luận & Khuyến nghị:**
    * Thuật toán K-Means hoạt động tốt trong bài toán **Phân loại Phân khúc Bất động sản (Market Segmentation)** để gom nhóm nhà có thuộc tính tương đồng.
    * Nếu mục tiêu chính là **Dự đoán Giá nhà chính xác (Regression)** với $R^2 > 0.8$, nên chuyển sang sử dụng các thuật toán Hồi quy có giám sát như **Linear Regression**, **Random Forest Regressor**, hoặc **XGBoost**.
    """)

# TAB 2: BẢNG DỮ LIỆU VÀ TÂM CỤM
with tab2:
    st.subheader("1. Tọa độ Tâm cụm (Centroids) & Giá trung bình theo cụm")
    centroid_df = pd.DataFrame(centroids, columns=features)
    centroid_df["SalePrice_Mean"] = [
        cluster_means[i] for i in range(k_clusters)
    ]
    centroid_df.index.name = "Cluster"
    st.dataframe(centroid_df.style.highlight_max(axis=0, color="lightgreen"))

    st.subheader(
        "2. Bảng Khoảng cách Centroid & Dự đoán đối chiếu trên Tập Test"
    )
    cols_show = (
        features
        + [f"Dist_Centroid_{i}" for i in range(k_clusters)]
        + [
            "Assigned_Cluster",
            target,
            "Predicted_SalePrice",
            "Absolute_Error",
        ]
    )
    st.dataframe(test_eval[cols_show])

# TAB 3: BIỂU ĐỒ TRỰC QUAN HÓA
with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Phương pháp Elbow (WCSS theo K)")
        wcss = []
        K_range = range(1, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
            km.fit(X_train)
            wcss.append(km.inertia_)

        fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
        ax_elbow.plot(K_range, wcss, "bo-", linewidth=2, markersize=7)
        ax_elbow.axvline(
            x=3,
            color="red",
            linestyle="--",
            label="Điểm Elbow đề xuất (K=3)",
        )
        ax_elbow.set_xlabel("Số cụm K")
        ax_elbow.set_ylabel("WCSS")
        ax_elbow.set_title("Xác định điểm gãy Elbow")
        ax_elbow.legend()
        ax_elbow.grid(True)
        st.pyplot(fig_elbow)

    with col_b:
        st.subheader("2. Biểu đồ So sánh Giá thực tế vs Giá ước tính")
        fig_comp, ax_comp = plt.subplots(figsize=(6, 4))
        ax_comp.plot(
            range(len(test_eval)),
            test_eval[target],
            label="Giá thực tế (Actual)",
            color="blue",
            alpha=0.7,
        )
        ax_comp.plot(
            range(len(test_eval)),
            test_eval["Predicted_SalePrice"],
            label="Giá dự đoán (Predicted)",
            color="orange",
            linestyle="--",
            linewidth=2,
        )
        ax_comp.set_title("Giá Thực tế vs Giá Dự đoán")
        ax_comp.set_xlabel("Chỉ số nhà trong tập Test")
        ax_comp.set_ylabel("SalePrice")
        ax_comp.legend()
        ax_comp.grid(True)
        st.pyplot(fig_comp)

    st.subheader("3. Biểu đồ Phân cụm PCA 2D")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    centroids_pca = pca.transform(centroids)

    fig_pca, ax_pca = plt.subplots(figsize=(9, 4.5))
    ax_pca.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=train_df["Cluster"],
        cmap="tab10",
        alpha=0.6,
        s=45,
    )
    ax_pca.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c="red",
        marker="X",
        s=200,
        linewidths=2,
        label="Tâm cụm (Centroids)",
    )
    ax_pca.set_title("Không gian phân cụm PCA 2D")
    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    ax_pca.legend()
    ax_pca.grid(True)
    st.pyplot(fig_pca)

# TAB 4: DỰ ĐOÁN CHO NHÀ MỚI
with tab4:
    st.subheader("Nhập thuộc tính để phân cụm & dự đoán")
    g1, g2, g3 = st.columns(3)
    in_qual = g1.number_input("OverallQual (Chất lượng)", value=0.0, step=0.1)
    in_area = g2.number_input("GrLivArea (Diện tích sống)", value=0.0, step=0.1)
    in_cars = g3.number_input(
        "GarageCars (Sức chứa Garage)", value=0.0, step=0.1
    )

    g4, g5 = st.columns(2)
    in_bsmt = g4.number_input("TotalBsmtSF (Diện tích hầm)", value=0.0, step=0.1)
    in_age = g5.number_input("HouseAge (Tuổi thọ nhà)", value=0.0, step=0.1)

    if st.button("Tính toán Khoảng cách & Dự đoán"):
        user_point = np.array([[in_qual, in_area, in_cars, in_bsmt, in_age]])
        dists = [
            np.linalg.norm(user_point - centroids[i]) for i in range(k_clusters)
        ]
        assigned_c = int(np.argmin(dists))
        pred_price = cluster_means[assigned_c]

        st.write("**Khoảng cách Euclid tới các Tâm cụm:**")
        for i in range(k_clusters):
            st.write(f"- Khoảng cách tới Centroid {i}: **{dists[i]:.4f}**")

        st.success(
            f"🏠 Ngôi nhà được gán vào **Cụm {assigned_c}** (Khoảng cách ngắn nhất: {dists[assigned_c]:.4f})"
        )
        st.info(f"💵 Giá bán ước tính: **${pred_price:,.2f}**")
