# Laporan Proyek Machine Learning - Zita Natalia Arisenda

## Domain Proyek

Banyak platform digital dan perpustakaan sekolah menghadapi tantangan dalam memberikan rekomendasi buku yang relevan karena preferensi pembaca sangat bervariasi dan dapat berubah seiring waktu. Ketidaksesuaian rekomendasi berpotensi mengurangi keterlibatan pengguna dan menurunkan minat baca. Dengan memanfaatkan data interaksi pengguna dan informasi konten buku, sistem rekomendasi mampu meningkatkan relevansi saran bacaan serta membantu pengguna menemukan buku yang sesuai dengan ketertarikan mereka. Penelitian oleh Ardiansyah et al. (2023) menunjukkan bahwa sistem rekomendasi berbasis konten di perpustakaan sekolah dapat meningkatkan akurasi pencarian bacaan bagi siswa. Sementara itu, studi oleh Irfan et al. (2014) menunjukkan bahwa sistem rekomendasi buku online mampu memberikan saran yang lebih sesuai dengan minat pengguna berdasarkan preferensi kolektif.

Referensi:

Ardiansyah, R., Bianto, M. A., & Saputra, B. D. (2023). Sistem Rekomendasi Buku Perpustakaan Sekolah menggunakan Metode Content-Based Filtering. Coscitech, 4(2).

Irfan, M., Cahyani, A. D., & Hastarita, F. R. (2014). Sistem Rekomendasi: Buku Online dengan Metode Collaborative Filtering. Technoscientia, 7(1).

## Business Understanding

### Problem Statements
- Bagaimana cara memberikan rekomendasi buku yang relevan kepada pengguna berdasarkan preferensi mereka? Pengguna sering kali kesulitan menemukan buku yang sesuai dengan minatnya karena banyaknya pilihan yang tersedia. Diperlukan sistem yang dapat menyaring dan menyarankan buku secara personal.
- Bagaimana memanfaatkan data historis pengguna (seperti rating) untuk membangun sistem rekomendasi yang efektif? Rekomendasi manual atau statis tidak mampu mengikuti perubahan preferensi pengguna. Sistem rekomendasi berbasis data dapat mempelajari pola-pola interaksi dan memberikan saran yang dinamis serta lebih relevan.

### Goals
- Menyusun sistem rekomendasi berbasis content-based dan collaborative filtering yang mampu mempersonalisasi saran buku untuk setiap pengguna. Sistem ini diharapkan dapat meningkatkan pengalaman pengguna dalam menemukan buku yang sesuai minat mereka.
- Mengukur kinerja sistem rekomendasi dengan menggunakan metrik evaluasi seperti precision@k dan RMSE.
Evaluasi ini diperlukan untuk mengetahui seberapa baik model dalam memprediksi rating yang diberikan pengguna terhadap buku.

### Solution statements
- Menerapkan algoritma K-Nearest Neighbors (KNN) untuk content-based filtering. Model ini dipilih karena kesederhanaan dan efektivitasnya dalam mengidentifikasi item serupa berdasarkan fitur-fiturnya.
- Menggunakan Singular Value Decomposition (SVD) untuk collaborative filtering. SVD dapat menemukan faktor-faktor laten yang menjelaskan interaksi pengguna-item, sehingga cocok untuk menghasilkan rekomendasi berdasarkan perilaku dan preferensi pengguna di masa lalu.
- Mengevaluasi kinerja sistem menggunakan metrik precision@k dan Root Mean Squared Error (RMSE). Metrik-metrik ini membantu menilai akurasi dan relevansi rekomendasi yang diberikan oleh sistem.

## Data Understanding

Dataset yang digunakan dalam proyek ini berjudul bookdfcleaned.csv (6 kolom, 267790 baris) dan userdfcleaned.csv (4 kolom, 433671 baris). Dataset ini dapat diakses melalui tautan berikut: https://www.kaggle.com/datasets/adnamard/book-recomendation-good-for-ncf
Dataset ini berisi informasi tentang buku dan rating oleh pengguna yang dapat mendukung pembangunan sistem rekomendasi berbasis content-based maupun collaborative filtering guna memprediksi buku-buku yang sesuai dengan preferensi pembaca. Data sudah terstruktur meskipun memerlukan penanganan outlier di langkah selanjutnya.

### Variabel pada Dataset
Berikut adalah daftar fitur (variabel) yang tersedia dalam dataset:

df_books = bookdfcleaned.csv
- ISBN: Nomor identifikasi unik untuk setiap buku.
- Title: Judul buku.
- Author: Penulis buku.
- PublicationYear: Tahun publikasi buku.
- Publisher: Penerbit buku.
- Cover: URL atau path gambar sampul buku.

df_user = userdfcleaned.csv
- User_id: ID unik untuk setiap pengguna.
- ISBN: Nomor identifikasi buku yang diberi rating oleh pengguna.
- Rating: Nilai rating yang diberikan pengguna untuk buku tersebut(0-10).
- Age: Usia pengguna.

### Eksplorasi Data dan Visualisasi
Beberapa langkah eksplorasi data dilakukan untuk memahami karakteristik data:
- Informasi Dataset.
  - Dataset terdiri dari kolom numerik dan kategorikal.
- Distribusi Variabel Numerik dengan Histogram.
  - df_books: Terlihat outlier pada kolom PublicationYear.
  - df_user: Terlihat outlier pada kolom Age. Sementara kolom Rating masih sesuai dengan rentang 0-10.
- Distribusi Variabel Kategorikal dengan Bar Chart.
  - df_books: Ditampilkan nilai 10 terbesar untuk kolom Author (teratas: Agatha Chirstie) dan Publisher (teratas: Harlequin).
  - df_user: Ditampilkan nilai 10 terbesar untuk kolom User_id (teratas: 11676) dan ISBN (teratas: 0316666343).

## Data Preparation

Pada tahap ini, dilakukan serangkaian proses data preparation untuk memastikan kedua dataset dalam kondisi bersih dan siap digunakan untuk membangun sistem rekomendasi:

1. Pemeriksaan Missing Values
- Langkah: Dilakukan pengecekan missing values pada kedua dataset (books dan user) menggunakan .isnull().sum().
- Hasil: Dataset df_books mengandung missing values pada kolom Title dan Publisher. Sementara dataset df_user tidak memiliki missing values.
- Tindakan: Baris dengan missing values pada df_books dihapus karena presentasenya kecil terhadap jumlah keseluruhan dataset.
- Alasan: Missing values dapat mengganggu proses analisis data.
2. Pemeriksaan Duplikasi Data
- Langkah: Diterapkan fungsi .duplicated() pada kedua dataset.
- Hasil: Tidak ditemukan data duplikat di kedua dataset.
- Alasan: Duplikasi dapat menyebabkan distorsi dalam pembelajaran model dan akurasi rekomendasi.
3. Pemeriksaan dan Deteksi Outlier
- Langkah: Data numerik dianalisis untuk memastikan nilai berada dalam rentang valid.
- Hasil: Beberapa nilai di kolom PublicationYear dan Age terdeteksi sebagai outlier dihapus karena presentasenya kecil terhadap jumlah keseluruhan dataset.
- Alasan: Nilai outlier dalam sistem rekomendasi dapat menurunkan kualitas model.
4. Konversi Tipe Data Kolom PublicationYear
- Langkah: Mengubah tipe data kolom PublicationYear dari numerik (biasanya integer) menjadi string (str) menggunakan fungsi .astype().
- Hasil: Nilai tahun terbit buku sekarang disimpan sebagai string.
- Alasan: Dalam sistem rekomendasi berbasis konten, PublicationYear digunakan sebagai fitur kategorikal, bukan numerik.

## Modeling and Result

Pada tahap ini, dibangun dua model sistem rekomendasi untuk menyarankan buku kepada pengguna. Dua pendekatan yang digunakan adalah Content-Based Filtering menggunakan Nearest Neighbors dan Collaborative Filtering menggunakan algoritma Singular Value Decomposition (SVD).

### Content-Based Filtering (Nearest Neighbors)

Model ini merekomendasikan buku berdasarkan kemiripan fitur buku, bukan dari interaksi pengguna. Kemiripan dihitung berdasarkan fitur, lalu diolah menggunakan teknik representasi teks TF-IDF dan dihitung jaraknya menggunakan cosine similarity. Pendekatan ini tidak bergantung pada data rating, sehingga cocok digunakan bahkan saat data pengguna sangat terbatas.

- Cara Kerja:
  - Menggunakan TfidfVectorizer untuk mengubah fitur buku menjadi vektor numerik.
  - Menghitung kemiripan antar buku menggunakan NearestNeighbors berbasis cosine distance.
  - Ketika pengguna memilih sebuah buku, sistem mencari buku serupa berdasarkan nilai cosine similarity.

- Parameter:
  - text_features = (Title * 3) + (Author * 3) + PublicationYear + Publisher
  - ngram_range = (1, 2)
  - stop_words = 'english'
  - max_features = 10000
  - metric = 'cosine'
  - algorithm = 'brute'
  - top_n = 20

- Kelebihan:
  - Tidak memerlukan data interaksi pengguna.
  - Dapat digunakan untuk pengguna baru.
  - Interpretasi hasil mudah karena berbasis kemiripan konten.

- Kekurangan:
  - Tidak mempertimbangkan selera pengguna secara personal.
  - Rekomendasi terbatas pada kesamaan fitur buku.
 
- Hasil Rekomendasi Top 20:
  
  <img src="https://raw.githubusercontent.com/zitaarisenda/Sistem-Rekomendasi/main/content1.png" width="500"/>
  <img src="https://raw.githubusercontent.com/zitaarisenda/Sistem-Rekomendasi/main/Screenshot%202025-06-04%20195827.png" width="500"/>
  <img src="https://raw.githubusercontent.com/zitaarisenda/Sistem-Rekomendasi/main/Screenshot%202025-06-04%20195929.png" width="500"/>

### Collaborative Filtering (SVD)

SVD merupakan metode matrix factorization yang mempelajari representasi tersembunyi (latent factors) dari pengguna dan buku berdasarkan rating. Model ini dapat memberikan rekomendasi personal yang lebih akurat dengan memanfaatkan hubungan tidak langsung antara pengguna dan item.

- Cara Kerja:
  - Menggunakan algoritma SVD dari pustaka Surprise.
  - Dataset df_user diubah menjadi format Surprise Dataset.
  - Model dilatih menggunakan k-fold cross-validation.
  - Performa dievaluasi menggunakan RMSE.

- Parameter:
  - rating_scale = (0, 10)
  - test_size = 0.2
  - random_state = 42
  - model = SVD()
  - n = 20 
- Parameter Default:
  - n_factors = 100
  - n_epochs = 20
  - lr_all = 0.005
  - reg_all = 0.02

- Kelebihan:
  - Mempertimbangkan preferensi pengguna secara menyeluruh.
  - Mampu memberikan rekomendasi personal dan relevan.

- Kekurangan:
  - Membutuhkan cukup banyak data rating historis.
  - Kurang cocok untuk pengguna baru.

- Hasil Rekomendasi Top 20:

  <img src="https://raw.githubusercontent.com/zitaarisenda/Sistem-Rekomendasi/main/Screenshot%202025-06-04%20200654.png" width="1000"/>
  <img src="https://raw.githubusercontent.com/zitaarisenda/Sistem-Rekomendasi/main/Screenshot%202025-06-04%20200023.png" width="1000"/>
  <img src="https://raw.githubusercontent.com/zitaarisenda/Sistem-Rekomendasi/main/Screenshot%202025-06-04%20200039.png" width="1000"/>

## Evaluation

Tahap evaluasi bertujuan untuk mengukur seberapa efektif sistem rekomendasi dalam memberikan saran buku yang relevan bagi pengguna. Karena sistem rekomendasi bekerja tanpa label target yang eksplisit, maka evaluasi dilakukan dengan pendekatan yang berbeda untuk masing-masing metode.

### Metrik Evaluasi yang Digunakan
- Cosine Similarity (untuk Content-Based Filtering)
  - Digunakan untuk mengukur tingkat kemiripan antara dua dokumen vektor, dalam hal ini representasi TF-IDF dari fitur buku.
    
    <img src="https://raw.githubusercontent.com/zitaarisenda/Sistem-Rekomendasi/main/Screenshot%202025-06-03%20172747.png" width="500"/>
  - Interpretasi: Jika hasil cosine similarity mendekati 1, maka buku-buku tersebut sangat mirip secara konten (berdasarkan teks judul); mendekati 0 berarti tidak mirip.

- Root Mean Squared Error (RMSE) (untuk Collaborative Filtering - SVD)
  - Digunakan untuk mengukur seberapa besar rata-rata kesalahan antara nilai prediksi dan nilai aktual rating pengguna terhadap buku.
    
    <img src="https://raw.githubusercontent.com/zitaarisenda/Sistem-Rekomendasi/main/Screenshot%202025-06-03%20172800.png" width="500"/>
  - Interpretasi: Semakin kecil nilai RMSE, semakin dekat prediksi model ke nilai sebenarnya. RMSE = 0 artinya prediksi sempurna. Dalam konteks rating skala 0–10.

### Hasil Evaluasi Model
1. Content-Based Filtering
- Eksperimen dilakukan terhadap pengguna dengan ID 278633.
- Rata-rata cosine similarity antara buku yang disukai dan buku hasil rekomendasi: 0.005
- Interpretasi: Nilai similarity yang sangat rendah menunjukkan bahwa buku yang direkomendasikan belum terlalu mirip dengan buku yang disukai pengguna. Ini bisa disebabkan oleh keterbatasan fitur yang digunakan.

2. Collaborative Filtering (SVD)
- Evaluasi dilakukan menggunakan dataset uji yang dihasilkan dari Surprise train_test_split.
- RMSE: 1.6
- Skala rating: 0 hingga 10
- Interpretasi: RMSE sebesar 1.6 menunjukkan bahwa terdapat rata-rata kesalahan prediksi sekitar 1.6 poin dari rating sebenarnya. Nilai ini cukup besar, menandakan bahwa model masih memiliki ruang untuk perbaikan.
