# Laporan Proyek Machine Learning - Bagus Darmawan

## Domain Proyek

Prediksi harga merupakan salah satu kasus regresi yang paling sering digunakan di dunia nyata, terutama di industri real estate, otomotif, dan e-commerce. Dengan perkembangan teknologi dan ketersediaan data yang semakin besar, pendekatan machine learning menjadi metode yang efektif untuk membangun sistem estimasi harga yang akurat berdasarkan karakteristik fitur yang relevan.

Masalah prediksi harga penting untuk diselesaikan karena:

* Membantu konsumen dalam pengambilan keputusan pembelian
* Membantu penjual atau pengembang menentukan strategi harga yang kompetitif
* Memberikan insight kepada investor mengenai tren pasar

Dalam konteks ini, dataset yang digunakan adalah **California Housing Prices** dari *Kaggle*, yang berisi informasi demografi dan geografis dari wilayah-wilayah di California.

**Referensi**:

* [Pace, R. K., & Barry, R. (1997). Sparse Spatial Autoregressions. Statistics & Probability Letters, 33(3), 291–297.](http://www.spatial-statistics.com/pace_manuscripts/spletters_ms_dir/statistics_prob_lets/html/ms_sp_lets1.html)
* [Dataset California Housing - Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

## Business Understanding

### Problem Statements

* Bagaimana cara membangun model machine learning yang dapat memprediksi harga rumah berdasarkan fitur-fitur demografis dan geografis?
* Model machine learning mana yang memberikan performa terbaik dalam memprediksi harga rumah?
* Bagaimana cara meningkatkan akurasi prediksi dengan melakukan hyperparameter tuning?

### Goals

* Menghasilkan model regresi yang mampu memprediksi harga rumah dengan akurasi tinggi
* Membandingkan performa tiga algoritma regresi: Linear Regression, Decision Tree Regressor, dan Random Forest Regressor
* Melakukan tuning hyperparameter untuk meningkatkan performa Random Forest

### Solution Statements

* Mengimplementasikan tiga algoritma machine learning: Linear Regression, Decision Tree Regressor, dan Random Forest Regressor
* Melakukan evaluasi model dengan metrik regresi: MAE, RMSE, dan R²
* Melakukan hyperparameter tuning pada Random Forest menggunakan `GridSearchCV`

## Data Understanding

Dataset yang digunakan adalah **California Housing Prices**, yang terdiri dari **20.640 baris dan 10 kolom**. Dataset ini diunduh dari Kaggle: https://www.kaggle.com/datasets/camnugent/california-housing-prices

### Kondisi Data

**Jumlah data:** 20.640 baris dan 10 kolom

**Missing Values:**
- **total_bedrooms**: 207 missing values (1.00% dari total data)
- **Kolom lainnya**: Tidak ada missing values (0)
- **Total missing values**: 207 dari 206.400 total values (0.10%)

**Data Duplikat:**
- **Jumlah data duplikat**: 0 (tidak ditemukan data duplikat)

**Outliers:**
Deteksi outliers menggunakan metode IQR menunjukkan:
- **total_rooms**: 1.287 outliers
- **total_bedrooms**: 1.271 outliers  
- **population**: 1.196 outliers
- **households**: 1.220 outliers
- **median_income**: 681 outliers
- **median_house_value**: 1.071 outliers

### Tautan Sumber Data

Dataset diunduh dari: https://www.kaggle.com/datasets/camnugent/california-housing-prices

### Variabel dalam dataset:

* `longitude`: Koordinat geografis garis bujur
* `latitude`: Koordinat geografis garis lintang
* `housing_median_age`: Umur median bangunan di blok tersebut
* `total_rooms`: Total jumlah ruangan di semua rumah di blok tersebut
* `total_bedrooms`: Total jumlah kamar tidur di semua rumah di blok tersebut
* `population`: Jumlah populasi yang tinggal di blok tersebut
* `households`: Jumlah rumah tangga di blok tersebut
* `median_income`: Pendapatan median penduduk di blok tersebut (dalam puluhan ribu dolar)
* `median_house_value`: **TARGET** - Median harga rumah (dalam dolar AS)
* `ocean_proximity`: Kategori kedekatan lokasi ke laut

### Exploratory Data Analysis (EDA)

#### Analisis Statistik Deskriptif
Dari statistik deskriptif dataset ditemukan:
- **Median house value** berkisar dari $14,999 hingga $500,001 dengan rata-rata $206,855
- **Median income** berkisar dari $0.499 hingga $15.000 (dalam puluhan ribu dolar)
- Dataset memiliki **207 missing values** (0.10%) pada kolom `total_bedrooms`

#### Analisis Korelasi
Dari correlation matrix ditemukan korelasi tertinggi dengan target variable:
- **median_income**: 0.69 (korelasi tertinggi)
- **total_rooms**: 0.13
- **housing_median_age**: 0.11
- **latitude**: -0.14 (korelasi negatif)

#### Analisis Kategorikal (Ocean Proximity)
Distribusi kategori `ocean_proximity`:
- **INLAND**: 40.3% (8,321 data)
- **<1H OCEAN**: 29.9% (6,168 data)
- **NEAR OCEAN**: 14.7% (3,042 data)
- **NEAR BAY**: 10.1% (2,080 data)
- **ISLAND**: 5.0% (1,029 data)

EDA dilakukan dengan visualisasi meliputi:
- Histogram distribusi untuk semua fitur numerik
- Correlation matrix heatmap
- Box plot untuk analisis kategori ocean proximity
- Analisis distribusi target variable

## Data Preparation

Langkah-langkah data preparation yang dilakukan sesuai dengan implementasi di notebook:

### 1. Handling Missing Values
- Ditemukan **207 missing values** (1.00%) pada kolom `total_bedrooms`
- Missing values diisi dengan **median** yaitu 435.0 menggunakan `fillna()`
- Tidak ada missing values pada kolom lainnya

**Alasan:** Missing values dapat menyebabkan error dalam proses modeling, sehingga perlu ditangani. Median dipilih karena lebih robust terhadap outliers dibandingkan mean.

### 2. Feature Engineering
Menambahkan **3 fitur baru** untuk meningkatkan prediksi:
- `rooms_per_household`: Rata-rata jumlah kamar per rumah tangga (`total_rooms / households`)
- `bedrooms_per_room`: Rasio kamar tidur terhadap total ruangan (`total_bedrooms / total_rooms`)
- `population_per_household`: Rata-rata populasi per rumah tangga (`population / households`)

**Alasan:** Fitur rasio ini dapat memberikan insight yang lebih baik tentang karakteristik rumah dan lingkungan dibandingkan nilai absolut, sehingga dapat meningkatkan performa prediksi model.

### 3. Encoding Variabel Kategorikal
- Menggunakan **One-Hot Encoding** pada variabel `ocean_proximity` dengan `pd.get_dummies()`
- Parameter `drop_first=True` untuk menghindari multicollinearity
- Menghasilkan 4 fitur baru: `ocean_INLAND`, `ocean_ISLAND`, `ocean_NEAR BAY`, `ocean_NEAR OCEAN`
- Total fitur menjadi **14 fitur** setelah encoding (karena drop_first=True)

**Alasan:** Model machine learning tidak dapat memproses data kategorikal secara langsung, sehingga perlu dikonversi menjadi format numerik. One-hot encoding dipilih karena tidak ada urutan hierarkis pada kategori ocean proximity.

### 4. Pemisahan Fitur dan Target
- Memisahkan dataset menjadi fitur (X) dan target variable (y)
- **Fitur (X)**: 14 kolom setelah menghapus `median_house_value`
- **Target (y)**: kolom `median_house_value`
- Shape fitur: (20640, 14)
- Shape target: (20640,)

**Alasan:** Pemisahan ini diperlukan untuk proses supervised learning, di mana model akan mempelajari hubungan antara fitur input dengan target output.

### 5. Data Splitting
- Data dibagi menjadi **80% training** (16,512 data) dan **20% testing** (4,128 data)
- Menggunakan `train_test_split` dengan `random_state=42` untuk reproducibility
- Tidak menggunakan stratify karena ini adalah masalah regresi

**Alasan:** Data splitting mencegah overfitting dan memungkinkan evaluasi objektif terhadap kemampuan generalisasi model pada data yang belum pernah dilihat sebelumnya.

### 6. Feature Scaling
- Menggunakan `StandardScaler` untuk normalisasi fitur
- Diterapkan `fit_transform()` pada data training dan `transform()` pada data testing
- Mean ≈ 0 dan Standard Deviation = 1 setelah scaling
- Shape setelah scaling: Training (16512, 14), Testing (4128, 14)

**Alasan:** Feature scaling diperlukan untuk memastikan semua fitur memiliki skala yang sama, mencegah fitur dengan nilai besar mendominasi model, dan meningkatkan konvergensi algoritma machine learning. Scaling dilakukan setelah data splitting untuk mencegah data leakage.

## Modeling

Empat model diterapkan dan dibandingkan performanya:

### 1. Linear Regression
**Deskripsi**: Model regresi linear sederhana yang mengasumsikan hubungan linear antara fitur dan target.

**Parameter**: Default dari `LinearRegression()`

**Kelebihan**: 
- Sederhana dan cepat dalam training
- Mudah diinterpretasi
- Tidak prone terhadap overfitting

**Kekurangan**: 
- Hanya menangkap hubungan linear
- Sensitive terhadap outliers
- Asumsi linearitas yang ketat

### 2. Decision Tree Regressor
**Deskripsi**: Model berbasis pohon keputusan yang dapat menangani hubungan non-linear.

**Parameter**: `random_state=42`

**Kelebihan**: 
- Dapat menangani hubungan non-linear
- Tidak membutuhkan normalisasi data
- Mudah diinterpretasi dengan visualisasi pohon

**Kekurangan**: 
- Rentan terhadap overfitting
- Tidak stabil (sensitive terhadap perubahan data)
- Bias terhadap fitur dengan banyak level

### 3. Random Forest Regressor
**Deskripsi**: Ensemble method yang menggabungkan banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting.

**Parameter**: `n_estimators=100`, `random_state=42`

**Kelebihan**: 
- Robust terhadap overfitting
- Dapat menangani hubungan non-linear
- Memberikan feature importance
- Mengurangi variance dibanding single decision tree

**Kekurangan**: 
- Waktu komputasi lebih besar
- Model kurang interpretable
- Memory usage tinggi

### 4. Random Forest Regressor (Tuned)
**Deskripsi**: Random Forest dengan hyperparameter yang dioptimasi menggunakan Grid Search.

**Hyperparameter Tuning menggunakan GridSearchCV**:
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```

**Parameter terbaik yang ditemukan**:
- `n_estimators`: 200
- `max_depth`: None  
- `min_samples_split`: 2
- `min_samples_leaf`: 2

Grid Search dilakukan dengan **3-fold Cross Validation** (total 72 fits) untuk memastikan parameter optimal.

## Evaluation

### Metrik Evaluasi yang digunakan:

1. **MAE (Mean Absolute Error)**
   $$MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
   Mengukur rata-rata kesalahan absolut. Lebih robust terhadap outliers.

2. **RMSE (Root Mean Squared Error)**
   $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
   Mengukur akar dari rata-rata kesalahan kuadrat. Lebih sensitif terhadap outliers.

3. **R² Score (Coefficient of Determination)**
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   Mengukur proporsi variabilitas yang dapat dijelaskan model (0-1, semakin tinggi semakin baik).

### Hasil Evaluasi:

| Model                 | MAE      | RMSE     | R² Score |
| --------------------- | -------- | -------- | -------- |
| Linear Regression     | 51,920.5 | 74,265.3 | 0.579    |
| Decision Tree         | 46,670.3 | 72,342.9 | 0.601    |
| Random Forest         | 33,411.4 | 51,193.4 | 0.800    |
| Random Forest (Tuned) | 32,991.2 | 50,744.5 | 0.803    |

### Feature Importance (Random Forest Tuned)

**Top 5 fitur paling penting**:
1. **median_income**: 53.40% - Fitur paling berpengaruh
2. **population_per_household**: 13.47% - Fitur engineered terpenting
3. **latitude**: 8.45% - Lokasi geografis utara-selatan
4. **longitude**: 8.42% - Lokasi geografis timur-barat  
5. **housing_median_age**: 5.21% - Umur bangunan

### Analisis Hasil:

**Model Terbaik**: Random Forest (Tuned) dengan R² Score **0.803**

**Pencapaian Utama**:
- Model berhasil menjelaskan **80.3%** variabilitas harga rumah di California
- Peningkatan performa **22.4%** dibandingkan baseline Linear Regression
- MAE sebesar $32,991 menunjukkan rata-rata error prediksi yang reasonable
- Hyperparameter tuning memberikan perbaikan meski kecil namun konsisten

**Validasi Model**:
- Plot Predicted vs Actual menunjukkan korelasi yang baik dengan pola linear
- Model mampu memprediksi dengan akurat di berbagai range harga
- Feature importance masuk akal secara domain knowledge (median_income tertinggi)

**Insight Bisnis**:
- **Pendapatan median** adalah faktor paling menentukan harga rumah
- **Lokasi geografis** (latitude/longitude) sangat berpengaruh
- **Fitur engineered** seperti `population_per_household` memberikan value tambahan
- Model dapat digunakan untuk estimasi harga properti di California dengan tingkat akurasi tinggi

### Contoh Prediksi:
```
Data 1 - Prediksi: $51,379.64, Aktual: $47,700.00
Data 2 - Prediksi: $106,439.83, Aktual: $45,800.00  
Data 3 - Prediksi: $476,037.58, Aktual: $500,001.00
Data 4 - Prediksi: $260,468.65, Aktual: $218,600.00
Data 5 - Prediksi: $226,541.22, Aktual: $278,000.00
```

Model final telah disimpan sebagai `california_housing_best_model.pkl` dan dapat digunakan untuk prediksi pada data baru.