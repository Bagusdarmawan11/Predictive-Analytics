# Analisis Prediktif: Prediksi Harga Rumah California

## Gambaran Umum Proyek

Proyek ini bertujuan untuk membangun model *machine learning* yang dapat memprediksi nilai median harga rumah di California secara akurat berdasarkan karakteristik demografi dan geografis. Kemampuan untuk memprediksi harga rumah sangat penting dalam industri *real estate* untuk keputusan investasi, strategi penetapan harga, dan analisis pasar.

Dataset yang digunakan untuk proyek ini adalah **California Housing Prices** dari Kaggle, yang berisi informasi demografi dan geografis dari berbagai wilayah di California berdasarkan sensus tahun 1990.

## Daftar Isi

- [Gambaran Umum Proyek](#gambaran-umum-proyek)
- [Pernyataan Masalah](#pernyataan-masalah)
- [Tujuan Proyek](#tujuan-proyek)
- [Pernyataan Solusi](#pernyataan-solusi)
- [Dataset](#dataset)
- [Eksplorasi Data (EDA)](#eksplorasi-data-eda)
- [Pra-pemrosesan Data](#pra-pemrosesan-data)
- [Pemodelan](#pemodelan)
- [Evaluasi](#evaluasi)
- [Insight Bisnis](#insight-bisnis)
- [Cara Menjalankan Proyek](#cara-menjalankan-proyek)
- [Referensi](#referensi)
- [Penulis](#penulis)

## Pernyataan Masalah

- Bagaimana cara membangun model *machine learning* yang dapat memprediksi harga rumah berdasarkan fitur-fitur demografis dan geografis?
- Fitur-fitur apa saja yang paling berpengaruh terhadap harga rumah?
- Model *machine learning* mana yang memberikan performa terbaik untuk prediksi harga rumah?

## Tujuan Proyek

- Membangun model *machine learning* yang dapat memprediksi `median_house_value` dengan akurat.
- Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap harga rumah.
- Membandingkan performa beberapa algoritma *machine learning* untuk mendapatkan model terbaik.

## Pernyataan Solusi

Untuk mencapai tujuan di atas, langkah-langkah berikut telah dilakukan:
1.  **Pra-pemrosesan Data**: Menangani nilai-nilai yang hilang, mengkodekan fitur kategorikal, dan melakukan penskalaan fitur numerik.
2.  **Eksplorasi Data (EDA)**: Menganalisis distribusi data, korelasi, dan mengidentifikasi *outlier*.
3.  **Pelatihan Model**: Melatih tiga algoritma *machine learning* yang berbeda: Regresi Linear, Regresi Ridge, dan *Random Forest Regressor*.
4.  **Penyetelan *Hyperparameter***: Mengoptimalkan model dengan performa terbaik (*Random Forest*) menggunakan `GridSearchCV`.
5.  **Evaluasi Model**: Mengevaluasi performa model menggunakan metrik seperti R-squared ($R^2$), *Mean Absolute Error* (MAE), dan *Root Mean Squared Error* (RMSE).
6.  **Analisis Pentingnya Fitur (*Feature Importance*)**: Mengidentifikasi fitur-fitur yang paling berpengaruh.
7.  **Penyimpanan Model**: Menyimpan model terbaik dan *scaler* untuk prediksi di masa mendatang.

## Dataset

Dataset yang digunakan adalah dataset **California Housing Prices**, yang tersedia di Kaggle. Dataset ini berisi 20.640 entri dengan 10 fitur, termasuk:

-   `longitude`: Ukuran seberapa jauh ke barat sebuah rumah (-124.35 hingga -114.31)
-   `latitude`: Ukuran seberapa jauh ke utara sebuah rumah (32.54 hingga 41.95)
-   `housing_median_age`: Usia median sebuah rumah dalam satu blok (1.0 hingga 52.0)
-   `total_rooms`: Total jumlah kamar dalam satu blok (2.0 hingga 39320.0)
-   `total_bedrooms`: Total jumlah kamar tidur dalam satu blok (1.0 hingga 6445.0)
-   `population`: Total jumlah orang yang tinggal dalam satu blok (3.0 hingga 35682.0)
-   `households`: Total jumlah rumah tangga, sekelompok orang yang tinggal dalam satu unit rumah (1.0 hingga 6082.0)
-   `median_income`: Pendapatan median untuk rumah tangga dalam satu blok rumah (0.4999 hingga 15.0001)
-   `median_house_value`: Nilai median rumah untuk rumah tangga dalam satu blok (variabel target) (14999.0 hingga 500001.0)
-   `ocean_proximity`: Lokasi rumah relatif terhadap laut (Kategorikal: <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND)

Dataset dapat ditemukan [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

## Eksplorasi Data (EDA)

Temuan kunci dari EDA:
-   **Nilai Hilang**: `total_bedrooms` memiliki 207 nilai yang hilang (1.00%). Ini ditangani menggunakan imputasi median.
-   **Outlier**: Terdeteksi pada `total_rooms`, `total_bedrooms`, `population`, dan `households`. Ini ditangani dengan pembatasan menggunakan metode IQR.
-   **Rekayasa Fitur (*Feature Engineering*)**: Fitur-fitur baru seperti `rooms_per_household`, `bedrooms_per_room`, dan `population_per_household` dibuat untuk menangkap lebih banyak *insight*.
-   **Korelasi**: `median_income` menunjukkan korelasi positif terkuat dengan `median_house_value`. `latitude` dan `longitude` juga menunjukkan korelasi yang signifikan, menunjukkan pentingnya lokasi geografis.

## Pra-pemrosesan Data

Langkah-langkah pra-pemrosesan meliputi:
1.  **Penanganan Nilai Hilang**: Mengisi `total_bedrooms` yang hilang dengan nilai median.
2.  **Penanganan *Outlier***: Membatasi *outlier* pada fitur numerik menggunakan metode IQR.
3.  **Rekayasa Fitur**: Membuat tiga fitur baru: `rooms_per_household`, `bedrooms_per_room`, dan `population_per_household`.
4.  **Pengkodean Kategorikal**: Melakukan *One-hot encoding* pada fitur `ocean_proximity`.
5.  **Penskalaan Fitur**: Melakukan penskalaan fitur numerik menggunakan `MinMaxScaler` untuk memastikan semua fitur berkontribusi secara merata pada model.

## Pemodelan

Tiga model *machine learning* awalnya dilatih dan dievaluasi:

1.  **Regresi Linear**: Model *baseline*.
2.  **Regresi Ridge**: Model linear yang diregulasi.
3.  **Random Forest Regressor**: Metode ensemble yang kuat.

Setelah evaluasi awal, **Random Forest Regressor** menunjukkan performa terbaik dan dipilih untuk penyetelan *hyperparameter* menggunakan `GridSearchCV`.

## Evaluasi

Model terbaik mencapai performa berikut:

**Model Terbaik**: Random Forest (Disetel)

-   **R² Score**: **0.803**
-   **Mean Absolute Error (MAE)**: $32,991
-   **Root Mean Squared Error (RMSE)**: $50,564

**Pencapaian Utama**:
-   Model berhasil menjelaskan **80.3%** variabilitas harga rumah di California.
-   Peningkatan performa **22.4%** dibandingkan *baseline* Regresi Linear.
-   MAE sebesar $32,991 menunjukkan rata-rata kesalahan prediksi yang wajar.
-   Penyetelan *hyperparameter* memberikan peningkatan performa yang kecil namun konsisten.

**Validasi Model**:
-   Plot Predicted vs Actual menunjukkan korelasi yang baik dengan pola linear.
-   Model mampu memprediksi secara akurat di berbagai rentang harga.
-   Pentingnya fitur (*feature importance*) masuk akal berdasarkan pengetahuan domain (`median_income` adalah yang paling penting).

**Contoh Prediksi (dari test set)**:
-   Data 1 - Prediksi: $51,379.64, Aktual: $47,700.00
-   Data 2 - Prediksi: $106,439.83, Aktual: $45,800.00
-   Data 3 - Prediksi: $476,037.58, Aktual: $475,300.00
-   Data 4 - Prediksi: $173,767.11, Aktual: $148,000.00
-   Data 5 - Prediksi: $269,704.93, Aktual: $229,000.00

## Insight Bisnis

-   **Pendapatan Median** adalah faktor paling signifikan yang menentukan harga rumah.
-   **Lokasi Geografis** (*latitude*/*longitude*) memiliki pengaruh yang kuat.
-   **Fitur Rekayasa (*Engineered Features*)** seperti `population_per_household` memberikan informasi berharga tambahan pada model.
-   Model dapat digunakan untuk estimasi harga properti yang akurat di California.

## Cara Menjalankan Proyek

Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut:

1.  **Kloning repositori**:
    ```bash
    git clone <(https://github.com/Bagusdarmawan11/Predictive-Analytics)>
    cd california-housing-price-prediction
    ```

2.  **Instal dependensi**:
    Pastikan Anda telah menginstal Python. Kemudian instal pustaka yang diperlukan:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    ```

3.  **Unduh dataset**:
    Letakkan file `housing.csv` di direktori utama proyek. Anda dapat mengunduhnya dari [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

4.  **Jalankan Jupyter Notebook atau Skrip Python**:
    -   Untuk menjalankan Jupyter Notebook (`submission.ipynb`):
        ```bash
        jupyter notebook submission.ipynb
        ```
    -   Untuk menjalankan skrip Python (`submission.py`):
        ```bash
        python submission.py
        ```
    Skrip akan melakukan pra-pemrosesan data, pelatihan model, evaluasi, dan menyimpan model terbaik (`california_housing_best_model.pkl`) serta *scaler* (`scaler.pkl`).

## Referensi

* [Pace, R. K., & Barry, R. (1997). Sparse Spatial Autoregressions. Statistics & Probability Letters, 33(3), 291–297.](http://www.spatial-statistics.com/pace_manuscripts/spletters_ms_dir/statistics_prob_lets/html/ms_sp_lets1.html)
* [Dataset California Housing - Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

## Penulis

**Bagus Darmawan**

---
