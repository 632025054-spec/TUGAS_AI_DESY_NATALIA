import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="ML Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    
)

# ============================================================
# KONFIGURASI DATASET TOY DARI SKLEARN
# ============================================================
from sklearn.datasets import (
    load_diabetes, load_linnerud, make_regression,      # Regresi
    load_iris, load_wine, load_breast_cancer, load_digits  # Klasifikasi
)

# Mapping nama dataset -> fungsi loader
REGRESSION_DATASETS = {
    "Diabetes 🩺": load_diabetes,
    "Linnerud 🏋️": load_linnerud,
    "Synthetic Regression 🎲": lambda: make_regression(n_samples=300, n_features=5, noise=15, random_state=42)
}

CLASSIFICATION_DATASETS = {
    "Iris 🌸": load_iris,
    "Wine 🍷": load_wine,
    "Breast Cancer 🎗️": load_breast_cancer,
    "Digits 🔢": load_digits
}

# ============================================================
# CACHE DATA: load & split dataset agar tidak reload tiap rerun
# ============================================================
@st.cache_data(show_spinner=False)
def load_dataset(task_type, dataset_name, test_size):
    """Load dataset toy sklearn dan split menjadi train/test."""
    from sklearn.model_selection import train_test_split
    
    if task_type == "Regresi":
        loader = REGRESSION_DATASETS[dataset_name]
        data = loader()
        X, y = data.data, data.target
        # Linnerud punya target 3D, ambil kolom pertama
        if y.ndim > 1:
            y = y[:, 0]
    else:
        loader = CLASSIFICATION_DATASETS[dataset_name]
        data = loader()
        X, y = data.data, data.target
    
    # Split data
    stratify = y if task_type == "Klasifikasi" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, data.feature_names if hasattr(data, 'feature_names') else None


# ============================================================
# FUNGSI TRAINING MODEL
# ============================================================
def get_regression_model(model_name):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    return {
        "Linear Regression 📏": LinearRegression(),
        "Ridge 🛡️": Ridge(alpha=1.0),
        "Lasso ✂️": Lasso(alpha=0.1),
        "Decision Tree 🌳": DecisionTreeRegressor(random_state=42),
        "Random Forest 🌲🌲": RandomForestRegressor(n_estimators=100, random_state=42),
    }[model_name]

def get_classification_model(model_name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    return {
        "Logistic Regression 📈": LogisticRegression(max_iter=1000),
        "KNN 👥": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree 🌳": DecisionTreeClassifier(random_state=42),
        "Random Forest 🌲🌲": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM 🎯": SVC(kernel="rbf"),
    }[model_name]


# ============================================================
# HEADER
# ============================================================
st.title("🤖 Machine Learning Playground")
st.caption("Eksperimen cepat model ML dengan dataset toy dari scikit-learn ✨")
st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # Pilih jenis task
    task_type = st.radio(
        "🎯 Jenis Task",
        ["Regresi", "Klasifikasi"],
        help="Pilih apakah kamu ingin melakukan Regresi atau Klasifikasi"
    )
    
    # Pilih dataset sesuai task
    if task_type == "Regresi":
        dataset_name = st.selectbox("📊 Pilih Dataset", list(REGRESSION_DATASETS.keys()))
        model_name = st.selectbox(
            "🧠 Pilih Model",
            ["Linear Regression 📏", "Ridge 🛡️", "Lasso ✂️",
             "Decision Tree 🌳", "Random Forest 🌲🌲"]
        )
    else:
        dataset_name = st.selectbox("📊 Pilih Dataset", list(CLASSIFICATION_DATASETS.keys()))
        model_name = st.selectbox(
            "🧠 Pilih Model",
            ["Logistic Regression 📈", "KNN 👥", "Decision Tree 🌳",
             "Random Forest 🌲🌲", "SVM 🎯"]
        )
    
    # Slider test size
    test_size = st.slider(
        "🔀 Test Size (%)",
        min_value=10, max_value=50, value=20, step=5,
        help="Persentase data yang digunakan untuk testing"
    ) / 100.0
    
    st.markdown("---")
    train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.caption("💡 Tips: Pilih dataset & model, atur test size, lalu klik **Train Model**.")


# ============================================================
# MAIN AREA
# ============================================================
if not train_button and "trained" not in st.session_state:
    st.info("👈 Silakan atur konfigurasi di sidebar dan klik **🚀 Train Model** untuk memulai.")
    st.stop()

# Proses training saat tombol diklik
if train_button:
    with st.spinner("⏳ Melatih model... mohon tunggu sebentar"):
        # Load data
        X_train, X_test, y_train, y_test, feature_names = load_dataset(
            task_type, dataset_name, test_size
        )
        
        # Ambil model
        if task_type == "Regresi":
            model = get_regression_model(model_name)
        else:
            model = get_classification_model(model_name)
        
        # Training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Simpan ke session_state
        st.session_state.trained = True
        st.session_state.task_type = task_type
        st.session_state.model_name = model_name
        st.session_state.dataset_name = dataset_name
        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train

# Pastikan sudah trained
if "trained" not in st.session_state:
    st.stop()

# Ambil dari session_state
task_type = st.session_state.task_type
model_name = st.session_state.model_name
dataset_name = st.session_state.dataset_name
model = st.session_state.model
X_test = st.session_state.X_test
y_test = st.session_state.y_test
y_pred = st.session_state.y_pred

# ============================================================
# INFO MODEL
# ============================================================
col1, col2, col3 = st.columns(3)
col1.metric("🎯 Task", task_type)
col2.metric("📊 Dataset", dataset_name)
col3.metric("🧠 Model", model_name.split(" ", 1)[0] if " " in model_name else model_name)

st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["📈 Visualisasi", "📊 Evaluation Metrics"])

# -------------------- TAB 1: VISUALISASI --------------------
with tab1:
    if task_type == "Regresi":
        st.subheader("🎯 Actual vs Predicted")
        
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=y_pred, ax=ax, color="#3498db",
                            s=70, edgecolor="black", alpha=0.7)
            # Garis referensi ideal (y = x)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val],
                    'r--', lw=2, label="Ideal (y = x)")
            ax.set_xlabel("Actual Values", fontsize=12)
            ax.set_ylabel("Predicted Values", fontsize=12)
            ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=13)
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col_b:
            st.markdown("#### 📌 Info")
            st.write(f"- **Jumlah test samples:** {len(y_test)}")
            st.write(f"- **Range actual:** [{y_test.min():.2f}, {y_test.max():.2f}]")
            st.write(f"- **Range predicted:** [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    else:  # Klasifikasi
        st.subheader("🧩 Confusion Matrix")
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_test, y_pred)
        
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        cbar=True, ax=ax, square=True,
                        annot_kws={"size": 12})
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)
            ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
            st.pyplot(fig)
        
        with col_b:
            st.markdown("#### 📌 Info")
            st.write(f"- **Jumlah test samples:** {len(y_test)}")
            st.write(f"- **Jumlah kelas:** {len(np.unique(y_test))}")
            st.write(f"- **Total benar:** {np.trace(cm)}")
            st.write(f"- **Total salah:** {cm.sum() - np.trace(cm)}")


# -------------------- TAB 2: EVALUATION --------------------
with tab2:
    if task_type == "Regresi":
        st.subheader("📊 Regression Metrics")
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Kartu metrik
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📏 MAE", f"{mae:.4f}")
        c2.metric("📐 MSE", f"{mse:.4f}")
        c3.metric("📉 RMSE", f"{rmse:.4f}")
        c4.metric("🎯 R² Score", f"{r2:.4f}")
        
        st.markdown("---")
        
        # Tabel ringkasan
        metrics_df = pd.DataFrame({
            "Metric": ["MAE", "MSE", "RMSE", "R²"],
            "Value": [mae, mse, rmse, r2],
            "Deskripsi": [
                "Mean Absolute Error — rata-rata error absolut (semakin kecil semakin baik)",
                "Mean Squared Error — rata-rata error kuadrat (semakin kecil semakin baik)",
                "Root Mean Squared Error — akar dari MSE (semakin kecil semakin baik)",
                "Koefisien Determinasi — seberapa baik model menjelaskan variansi (mendekati 1 semakin baik)"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    else:  # Klasifikasi
        st.subheader("📊 Classification Metrics")
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, classification_report
        )
        
        # Untuk multiclass, gunakan average='weighted'
        n_classes = len(np.unique(y_test))
        avg = "binary" if n_classes == 2 else "weighted"
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
        
        # Kartu metrik
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Accuracy", f"{acc:.4f}")
        c2.metric("🎯 Precision", f"{prec:.4f}")
        c3.metric("🔍 Recall", f"{rec:.4f}")
        c4.metric("⚖️ F1-Score", f"{f1:.4f}")
        
        st.caption(f"ℹ️ Averaging: `{avg}` (jumlah kelas: {n_classes})")
        
        st.markdown("---")
        st.subheader("📋 Classification Report")
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df.style.format("{:.4f}"),
            use_container_width=True
        )
        
        # Bar chart perbandingan metrik
        st.markdown("---")
        st.subheader("📊 Perbandingan Metrik")
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        metrics_values = [acc, prec, rec, f1]
        colors = ["#3498db", "#9b59b6", "#e67e22", "#2ecc71"]
        bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor="black")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"Perbandingan Metrik — {model_name}")
        for bar, val in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("💡 Dibuat dengan Streamlit • Dataset dari scikit-learn • Selamat bereksperimen! 🚀")