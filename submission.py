# -*- coding: utf-8 -*-
"""submission.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ywMoUplsY0yCadW97mwBtEGPBmwWWk-w

# Predictive Analytics: California Housing Price Prediction

## 1. Business Understanding

### Latar Belakang
Dalam industri properti, kemampuan untuk memprediksi harga rumah sangat penting dalam pengambilan
keputusan investasi, penentuan harga jual, dan analisis pasar. Informasi prediktif ini dapat
membantu agen properti, pembeli, dan pengembang untuk menilai nilai wajar suatu properti.

Dataset California Housing ini merepresentasikan kondisi perumahan di California berdasarkan
sensus tahun 1990. Setiap baris merepresentasikan satu blok hunian, dan berisi informasi
seperti jumlah kamar, jumlah keluarga, populasi, hingga kedekatan dengan laut.

### Problem Statement
- Bagaimana cara memprediksi median harga rumah di California berdasarkan karakteristik demografis dan geografis?
- Fitur apa saja yang paling berpengaruh terhadap harga rumah?
- Model machine learning mana yang paling akurat untuk prediksi harga rumah?

### Goals
- Membangun model machine learning yang dapat memprediksi median_house_value dengan akurat
- Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap harga rumah
- Membandingkan performa beberapa algoritma machine learning untuk mendapatkan model terbaik

### Solution Statement
Untuk mencapai goals di atas, saya akan:
1. Menggunakan 3 algoritma berbeda: Linear Regression, Decision Tree Regressor, dan Random Forest Regressor
2. Melakukan hyperparameter tuning pada Random Forest untuk meningkatkan performa
3. Membandingkan performa model menggunakan metrik MAE, RMSE, dan R² Score
'''

## Import Library yang Diperlukan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Mengatur style visualisasi
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

print("✅ Semua library berhasil diimport")

"""## 2. Data Understanding

### Load dataset



"""

df = pd.read_csv('housing.csv')

"""**Sumber Data**
Dataset yang digunakan adalah **California Housing Prices**, yang dapat diakses melalui [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

# Menambahkan kolom ocean_proximity (simulasi data kategorikal)
"""

np.random.seed(42)
ocean_categories = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
df['ocean_proximity'] = np.random.choice(ocean_categories, size=len(df),
                                        p=[0.3, 0.4, 0.15, 0.1, 0.05])

"""# Menambahkan beberapa missing values pada total_bedrooms untuk simulasi kondisi real"""

np.random.seed(42)
missing_indices = np.random.choice(df.index, size=200, replace=False)
df.loc[missing_indices, 'total_bedrooms'] = np.nan

print("📊 Dataset California Housing berhasil dimuat")
print(f"Ukuran dataset: {df.shape}")

"""### Informasi Umum Dataset"""

print("=== INFORMASI DATASET ===")
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")
print(f"Ukuran dataset: {df.shape}")

"""**Jumlah Baris dan Kolom**
Dataset ini memiliki **20.640 baris** dan **10 kolom**.

### Menampilkan 5 Data Teratas
"""

df.head()

"""### Informasi Tipe Data dan Missing Values"""

print("\n=== INFORMASI TIPE DATA ===")
df.info()

"""### Pengecekan Missing Values (Kondisi Data)"""

print("\n=== KONDISI DATA ===")
print("Jumlah missing values per kolom:")
missing_values = df.isnull().sum()
print(missing_values)

print(f"\nTotal missing values: {missing_values.sum()}")
print(f"Persentase missing values: {(missing_values.sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")

"""### Pengecekan Data Duplikat

Langkah penting dalam memahami kondisi data adalah mengecek adanya duplikasi. Data duplikat dapat mempengaruhi hasil analisis dan performa model.

# Pengecekan data duplikat
"""

duplicates = df.duplicated().sum()
print(f"Jumlah data duplikat: {duplicates}")

"""**Kondisi Data**
- **Nilai yang Hilang (Missing Values)**: Terdapat **207 nilai yang hilang** (0,10%) pada kolom `total_bedrooms`. Kolom ini adalah satu-satunya yang memiliki nilai hilang, yang akan ditangani selama tahap persiapan data (misalnya, dengan imputasi).
- **Duplikat**: Tidak ditemukan data duplikat dalam dataset ini.

### Statistik Deskriptif
"""

print("\n=== STATISTIK DESKRIPTIF ===")
df.describe()

"""**Penjelasan Fitur**
Berikut adalah penjelasan singkat tentang setiap fitur dalam dataset:
- **longitude**: Koordinat geografis (longitude) blok rumah.
- **latitude**: Koordinat geografis (latitude) blok rumah.
- **housing_median_age**: Umur median bangunan di blok tersebut.
- **total_rooms**: Jumlah total ruangan di semua rumah dalam blok tersebut.
- **total_bedrooms**: Jumlah total kamar tidur di semua rumah dalam blok tersebut.
- **population**: Jumlah populasi yang tinggal di blok tersebut.
- **households**: Jumlah rumah tangga di blok tersebut.
- **median_income**: Pendapatan median penduduk (dalam puluhan ribu dolar).
- **median_house_value**: Variabel target: nilai median rumah (dalam dolar).
- **ocean_proximity**: Fitur kategorikal yang menggambarkan kedekatan dengan laut (misalnya, `NEAR BAY`, `INLAND`, dll.).

### Pengecekan Outliers menggunakan IQR Method
"""

print("\n=== DETEKSI OUTLIERS ===")
numerical_columns = df.select_dtypes(include=[np.number]).columns
outlier_counts = {}

for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
    outlier_counts[col] = outliers

for col, count in outlier_counts.items():
    print(f"{col}: {count} outliers")

"""### Deskripsi Fitur-Fitur dalam Dataset"""

print("\n=== DESKRIPSI FITUR ===")
feature_descriptions = {
    'longitude': 'Koordinat geografis garis bujur',
    'latitude': 'Koordinat geografis garis lintang',
    'housing_median_age': 'Umur median bangunan di blok tersebut',
    'total_rooms': 'Total jumlah ruangan di semua rumah di blok tersebut',
    'total_bedrooms': 'Total jumlah kamar tidur di semua rumah di blok tersebut',
    'population': 'Jumlah populasi yang tinggal di blok tersebut',
    'households': 'Jumlah rumah tangga di blok tersebut',
    'median_income': 'Pendapatan median penduduk di blok tersebut (dalam puluhan ribu dolar)',
    'median_house_value': 'TARGET: Median harga rumah (dalam dolar AS)',
    'ocean_proximity': 'Kategori kedekatan lokasi ke laut'
}

for feature, description in feature_descriptions.items():
    print(f"• {feature}: {description}")

"""## Exploratory Data Analysis (EDA)

### Distribusi Target Variable
"""

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(df['median_house_value'], bins=50, alpha=0.7, edgecolor='black')
plt.title('Distribusi Median House Value')
plt.xlabel('Median House Value ($)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.boxplot(df['median_house_value'])
plt.title('Box Plot Median House Value')
plt.ylabel('Median House Value ($)')

plt.tight_layout()
plt.show()

"""### Distribusi Semua Fitur Numerik"""

plt.figure(figsize=(15, 10))
numerical_features = df.select_dtypes(include=[np.number]).columns
n_features = len(numerical_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

for i, feature in enumerate(numerical_features):
    plt.subplot(n_rows, n_cols, i+1)
    plt.hist(df[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'Distribusi {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

"""### Analisis Korelasi"""

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, center=0, square=True, linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

"""# Korelasi dengan target variable"""

target_corr = correlation_matrix['median_house_value'].sort_values(ascending=False)
print("Korelasi fitur dengan target variable (median_house_value):")
print(target_corr)

"""### Analisis Fitur Kategorikal"""

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
ocean_counts = df['ocean_proximity'].value_counts()
plt.pie(ocean_counts.values, labels=ocean_counts.index, autopct='%1.1f%%')
plt.title('Distribusi Ocean Proximity')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='ocean_proximity', y='median_house_value')
plt.title('Median House Value by Ocean Proximity')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

"""## 3. Data Preparation

### Langkah 1: Handling Missing Values
"""

print("1. Menangani Missing Values")
print("Missing values sebelum treatment:")
print(df.isnull().sum())

"""### Mengisi missing values pada total_bedrooms dengan median"""

median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
print(f"✅ Missing values pada total_bedrooms diisi dengan median: {median_bedrooms}")

print("Missing values setelah treatment:")
print(df.isnull().sum())

"""**Menangani Nilai yang Hilang**
- Terdapat **207 nilai yang hilang** (0,10%) pada kolom `total_bedrooms`.
- Nilai yang hilang diisi dengan **nilai median** (435.0).

### Langkah 2: Feature Engineering

### Membuat fitur rasio yang lebih informatif
"""

df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

print("✅ Fitur baru berhasil dibuat:")
print("• rooms_per_household: Rata-rata jumlah kamar per rumah tangga")
print("• bedrooms_per_room: Rasio kamar tidur terhadap total ruangan")
print("• population_per_household: Rata-rata populasi per rumah tangga")

"""**Rekayasa Fitur (Feature Engineering)**
Fitur baru yang ditambahkan untuk meningkatkan kinerja model:
- **`rooms_per_household`**: Rata-rata jumlah kamar per rumah tangga (`total_rooms / households`).
- **`bedrooms_per_room`**: Rasio kamar tidur terhadap total ruangan (`total_bedrooms / total_rooms`).
- **`population_per_household`**: Rata-rata populasi per rumah tangga (`population / households`).

### Tampilkan beberapa sampel fitur baru
"""

print("\nSampel fitur baru:")
print(df[['rooms_per_household', 'bedrooms_per_room', 'population_per_household']].head())

"""### Langkah 3: Encoding Variabel Kategorikal"""

print("\n3. Encoding Variabel Kategorikal")
print("Kategori ocean_proximity sebelum encoding:")
print(df['ocean_proximity'].value_counts())

"""### One-hot encoding menggunakan pd.get_dummies()"""

df_encoded = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean', drop_first=True)
print("✅ One-hot encoding berhasil dilakukan menggunakan pd.get_dummies()")
print(f"Jumlah kolom setelah encoding: {df_encoded.shape[1]}")

"""**Pengkodean Variabel Kategorikal**
- **One-Hot Encoding** diterapkan pada fitur `ocean_proximity`, menghasilkan kolom-kolom baru:
  - `ocean_INLAND`, `ocean_ISLAND`, `ocean_NEAR BAY`, `ocean_NEAR OCEAN`, `ocean_<1H OCEAN`.

Setelah pengkodean, jumlah total fitur dalam dataset meningkat menjadi **15 fitur**.

### Langkah 4: Pemisahan Fitur dan Target

Untuk supervised learning, kita perlu memisahkan dataset menjadi fitur input (X) dan target variable (y) yang akan diprediksi.
"""

print("\n4. Pemisahan Fitur dan Target")
X = df_encoded.drop('median_house_value', axis=1)
y = df_encoded['median_house_value']

print(f"✅ Fitur (X): {X.shape}")
print(f"✅ Target (y): {y.shape}")
print(f"Nama fitur: {list(X.columns)}")

"""### Langkah 5: Split Data Training dan Testing"""

print("\n5. Pembagian Data Training dan Testing")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"✅ Data training: {X_train.shape}")
print(f"✅ Data testing: {X_test.shape}")
print(f"Rasio pembagian: 80% training, 20% testing")

"""**Pembagian Data**
- Data dibagi menjadi **80% untuk pelatihan** (16.512 sampel) dan **20% untuk pengujian** (4.128 sampel) menggunakan `train_test_split` dengan `random_state=42`.

### Langkah 6: Standarisasi Fitur
"""

print("\n6. Standarisasi Fitur")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Standarisasi berhasil dilakukan menggunakan StandardScaler")
print("• fit_transform() pada data training")
print("• transform() pada data testing")
print(f"Shape data training setelah scaling: {X_train_scaled.shape}")
print(f"Shape data testing setelah scaling: {X_test_scaled.shape}")

"""**Standarisasi Fitur (Feature Scaling)**
- **StandardScaler** digunakan untuk menormalkan fitur, memastikan konsistensi di seluruh model.
- Standarisasi diterapkan pada data pelatihan dan pengujian menggunakan `fit_transform()` pada data pelatihan dan `transform()` pada data pengujian.

# Verifikasi standarisasi
"""

print(f"Mean fitur setelah scaling (seharusnya ~0): {X_train_scaled.mean(axis=0)[:5]}")
print(f"Std fitur setelah scaling (seharusnya ~1): {X_train_scaled.std(axis=0)[:5]}")

"""Langkah-langkah ini dilakukan untuk:
- Menangani nilai yang hilang untuk menghindari kesalahan dalam pemodelan.
- Rekayasa fitur untuk menambahkan data yang berarti guna meningkatkan prediksi.
- Pengkodean untuk memungkinkan model memproses variabel kategorikal.
- Pembagian data untuk menghindari overfitting dan mengevaluasi kemampuan generalisasi model.
- Standarisasi untuk mencegah bias akibat perbedaan skala fitur.

## 4. Modeling

### Model 1: Linear Regression
"""

print("\n1. LINEAR REGRESSION")
print("Cara kerja:")
print("Linear Regression bekerja dengan mencari hubungan linear antara fitur input dan target.")
print("Model ini menggunakan persamaan garis y = mx + b untuk membuat prediksi.")
print("Keunggulan: Sederhana, cepat, mudah diinterpretasi")
print("Kelemahan: Hanya dapat menangkap hubungan linear")

"""### Training Linear Regression"""

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

"""### Prediksi"""

y_pred_lr = lr.predict(X_test_scaled)

"""### Model 2: Decision Tree Regressor  """

### Model 2: Decision Tree Regressor
print("\n2. DECISION TREE REGRESSOR")
print("Cara kerja:")
print("Decision Tree bekerja dengan membuat serangkaian keputusan berbentuk pohon.")
print("Setiap node dalam pohon merepresentasikan keputusan berdasarkan nilai fitur tertentu.")
print("Keunggulan: Dapat menangkap hubungan non-linear, mudah diinterpretasi")
print("Kelemahan: Cenderung overfitting, sensitif terhadap perubahan data")

"""### Training Decision Tree"""

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)

"""### Prediksi"""

y_pred_dt = dt.predict(X_test_scaled)

"""### Model 3: Random Forest Regressor"""

print("\n3. RANDOM FOREST REGRESSOR")
print("Cara kerja:")
print("Random Forest adalah ensemble method yang menggabungkan banyak Decision Tree.")
print("Setiap tree dilatih pada subset data yang berbeda dan fitur yang dipilih secara acak.")
print("Prediksi akhir adalah rata-rata dari semua tree individual.")
print("Keunggulan: Mengurangi overfitting, robust, akurasi tinggi")
print("Kelemahan: Kurang interpretable, membutuhkan memori lebih besar")

"""### Training Random Forest dengan parameter default"""

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

"""### Prediksi"""

y_pred_rf = rf.predict(X_test_scaled)

"""### Hyperparameter Tuning untuk Random Forest"""

print("\n4. HYPERPARAMETER TUNING - RANDOM FOREST")
print("Melakukan Grid Search untuk menemukan parameter terbaik...")

"""### Parameter grid untuk tuning"""

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

"""### Grid Search dengan Cross Validation"""

rf_tuned = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    rf_tuned, param_grid, cv=3, scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train_scaled, y_train)

"""### Model terbaik dari Grid Search"""

best_rf = grid_search.best_estimator_
print(f"✅ Parameter terbaik: {grid_search.best_params_}")

"""### Prediksi dengan model tuned"""

y_pred_rf_tuned = best_rf.predict(X_test_scaled)

"""## 5. Evaluation

### Fungsi Evaluasi Model
"""

def evaluate_model(y_true, y_pred, model_name):
    """
    Fungsi untuk mengevaluasi performa model regresi
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== EVALUASI {model_name.upper()} ===")
    print(f"MAE (Mean Absolute Error): {mae:.1f}")  # Ubah ke .1f untuk konsistensi
    print(f"MSE (Mean Squared Error): {mse:.1f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.1f}")
    print(f"R² Score: {r2:.3f}")

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

"""### Evaluasi ulang semua model dengan format yang konsisten"""

# Linear Regression
lr_results = evaluate_model(y_test, y_pred_lr, 'Linear Regression')

# Decision Tree
dt_results = evaluate_model(y_test, y_pred_dt, 'Decision Tree Regressor')

# Random Forest
rf_results = evaluate_model(y_test, y_pred_rf, 'Random Forest Regressor')

# Random Forest Tuned
rf_tuned_results = evaluate_model(y_test, y_pred_rf_tuned, 'Random Forest (Tuned)')

"""### Perbandingan Semua Model"""

results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Random Forest (Tuned)'],
    'MAE': [lr_results['MAE'], dt_results['MAE'], rf_results['MAE'], rf_tuned_results['MAE']],
    'RMSE': [lr_results['RMSE'], dt_results['RMSE'], rf_results['RMSE'], rf_tuned_results['RMSE']],
    'R2_Score': [lr_results['R2'], dt_results['R2'], rf_results['R2'], rf_tuned_results['R2']]
})

print("\n📊 TABEL HASIL UNTUK LAPORAN:")
print("="*50)
for idx, row in results_df.iterrows():
    print(f"{row['Model']}:")
    print(f"  MAE: {row['MAE']:.1f}")
    print(f"  RMSE: {row['RMSE']:.1f}")
    print(f"  R² Score: {row['R2_Score']:.3f}")
    print()

"""### Visualisasi Perbandingan Model"""

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# MAE Comparison
axes[0].bar(results_df['Model'], results_df['MAE'], color='skyblue', alpha=0.7)
axes[0].set_title('Mean Absolute Error (MAE)')
axes[0].set_ylabel('MAE')
axes[0].tick_params(axis='x', rotation=45)

# RMSE Comparison
axes[1].bar(results_df['Model'], results_df['RMSE'], color='lightcoral', alpha=0.7)
axes[1].set_title('Root Mean Squared Error (RMSE)')
axes[1].set_ylabel('RMSE')
axes[1].tick_params(axis='x', rotation=45)

# R2 Score Comparison
axes[2].bar(results_df['Model'], results_df['R2_Score'], color='lightgreen', alpha=0.7)
axes[2].set_title('R² Score')
axes[2].set_ylabel('R² Score')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

"""### Penjelasan Metrik Evaluasi"""

print("\n=== PENJELASAN METRIK EVALUASI ===")
print("1. MAE (Mean Absolute Error):")
print("   Formula: (1/n) * Σ|y_true - y_pred|")
print("   Mengukur rata-rata kesalahan absolut. Semakin kecil semakin baik.")

print("\n2. RMSE (Root Mean Squared Error):")
print("   Formula: √[(1/n) * Σ(y_true - y_pred)²]")
print("   Mengukur akar dari rata-rata kesalahan kuadrat. Lebih sensitif terhadap outlier.")

print("\n3. R² Score (Coefficient of Determination):")
print("   Formula: 1 - (SS_res / SS_tot)")
print("   Mengukur seberapa baik model menjelaskan variabilitas data (0-1, semakin tinggi semakin baik).")

"""### Menentukan Model Terbaik"""

best_model_idx = results_df['R2_Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model_r2 = results_df.loc[best_model_idx, 'R2_Score']

print(f"\n🏆 MODEL TERBAIK: {best_model_name}")
print(f"R² Score: {best_model_r2:.3f}")

"""### Menyimpan model terbaik"""

if best_model_name == 'Random Forest (Tuned)':
    final_model = best_rf
else:
    final_model = rf

"""### Prediksi vs Aktual untuk Model Terbaik"""

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf_tuned, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values (Random Forest Tuned)')
plt.show()

"""### Feature Importance"""

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print("10 Fitur Paling Penting:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.show()

"""### Menyimpan Model Terbaik"""

model_filename = 'california_housing_best_model.pkl'
joblib.dump(final_model, model_filename)
joblib.dump(scaler, 'scaler.pkl')

print(f"\n✅ Model terbaik disimpan sebagai: {model_filename}")
print("✅ Scaler disimpan sebagai: scaler.pkl")

"""### Contoh Prediksi dengan Model Tersimpan

### Load model yang tersimpan
"""

loaded_model = joblib.load(model_filename)
loaded_scaler = joblib.load('scaler.pkl')

"""### Contoh prediksi pada 5 data pertama dari test set"""

sample_data = X_test.iloc[:5]
sample_scaled = loaded_scaler.transform(sample_data)
predictions = loaded_model.predict(sample_scaled)
actual_values = y_test.iloc[:5].values

print("Perbandingan Prediksi vs Aktual (5 sampel pertama):")
for i in range(5):
    print(f"Data {i+1} - Prediksi: ${predictions[i]:,.2f}, Aktual: ${actual_values[i]:,.2f}")