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
  - Jumlah missing values df_books: Title: 49, Publisher 57
  - Jumlah missing values df_user: 0
  - Jumlah outlier df_books: PublicationYear: 5524
  - Jumlah outlier df_user: Rating: 10525, Age: 3344
  - Jumlah data duplikat df_books: 0
  - Jumlah data duplikat df_user: 0
- Distribusi Variabel Numerik dengan Histogram.
  - df_books: Terlihat outlier pada kolom PublicationYear.
  - df_user: Terlihat outlier pada kolom Age. Sementara kolom Rating juga memiliki outlier namun masih sesuai dengan rentang 0-10.
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
- Langkah: Mengubah tipe data kolom PublicationYear dari numerik (integer) menjadi string (str) menggunakan fungsi .astype().
- Hasil: Nilai tahun terbit buku sekarang disimpan sebagai string.
- Alasan: Dalam sistem rekomendasi berbasis konten, PublicationYear digunakan sebagai fitur kategorikal, bukan numerik.
5. Pembuatan Fitur text_features dan TF-IDF Vectorization untuk df_books
- Langkah: Beberapa kolom teks yang relevan seperti Title, Author, PublicationYear, dan Publisher digabungkan menjadi satu kolom baru bernama text_features. Untuk menekankan pentingnya informasi dari judul dan penulis buku dalam menentukan kemiripan konten, nilai dari kolom Title dan Author diberi bobot lebih tinggi dengan cara mengulang (mengalikan) masing-masing sebanyak tiga kali sebelum digabungkan. Setelahnya, kolom gabungan ini diubah menjadi representasi numerik menggunakan TfidfVectorizer dengan pengaturan ngram (1,2) untuk mengambil unigram dan bigram untuk menangkap konteks kata yang lebih luas, batas maksimum fitur sebanyak 10.000 untuk membatasi jumlah fitur maksimal untuk menghindari kompleksitas dan overfitting, serta stop_words='english' untuk menghapus kata-kata umum dalam bahasa Inggris yang tidak membawa makna penting.
- Hasil: Setiap buku direpresentasikan dalam bentuk vektor berbasis bobot kata yang mencerminkan pentingnya istilah-istilah dalam konteks seluruh koleksi buku.
- Alasan: Pemberian bobot lebih besar pada Title dan Author bertujuan agar kata-kata dari dua kolom tersebut memiliki pengaruh lebih besar dalam perhitungan kemiripan antar buku, karena keduanya dianggap lebih relevan dalam mendeskripsikan isi dan karakteristik utama sebuah buku dibanding fitur lain seperti penerbit atau tahun terbit.

## Modeling and Result

Pada tahap ini, dibangun dua model sistem rekomendasi untuk menyarankan buku kepada pengguna. Dua pendekatan yang digunakan adalah Content-Based Filtering menggunakan Nearest Neighbors dan Collaborative Filtering menggunakan algoritma Singular Value Decomposition (SVD).

### Content-Based Filtering (Nearest Neighbors)

Pendekatan Content-Based Filtering merekomendasikan buku berdasarkan kemiripan konten antar item, bukan dari riwayat interaksi pengguna. Sistem ini mengidentifikasi buku-buku yang memiliki fitur serupa berdasarkan representasi tekstual, sehingga tetap dapat memberikan rekomendasi meskipun pengguna belum banyak berinteraksi dengan sistem (cold-start friendly).

- Cara Kerja:
  - Modeling dengan Nearest Neighbors

    Vektor TF-IDF yang dihasilkan di proses preprocessing sebelumnya digunakan untuk menghitung kemiripan antar buku menggunakan algoritma NearestNeighbors dari scikit-learn dengan konfigurasi:
    - metric='cosine': Menggunakan cosine similarity untuk mengukur kemiripan antar buku.
    - algorithm='brute': Pendekatan pencarian eksak karena dataset tidak terlalu besar.

  - Fungsi recommend_books()
    - Fungsi menerima input berupa judul buku dari pengguna.
    - Mencari indeks buku berdasarkan judul yang dicari.
    - Mengambil vektor TF-IDF dari buku tersebut.
    - Menggunakan model NearestNeighbors untuk mencari buku lain yang paling mirip.
    - Mengembalikan top-N rekomendasi berdasarkan skor kemiripan tertinggi, dengan mengabaikan buku itu sendiri.

- Parameter:
  - metric = 'cosine' (untuk menghitung jarak kemiripan antar buku)
  - algorithm = 'brute' (pencarian jarak eksak antar semua pasangan item)
  - Model = NearestNeighbors() dari scikit-learn
  - top_n = 20 (banyak buku yang paling mirip ditampilkan dalam hasil rekomendasi)

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

Pendekatan Collaborative Filtering menggunakan algoritma Singular Value Decomposition (SVD) merupakan metode matrix factorization yang bertujuan mempelajari representasi tersembunyi (latent factors) dari pengguna dan item (buku) berdasarkan data interaksi berupa rating. Model ini mampu memberikan rekomendasi yang dipersonalisasi dengan mengidentifikasi pola tidak langsung dalam preferensi pengguna terhadap item yang berbeda.

- Cara Kerja:
  - Data rating dari df_user diubah menjadi format Surprise Dataset menggunakan objek Reader dengan skala nilai 0–10.
  - Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20 menggunakan train_test_split.
  - Model SVD dilatih pada data latih, kemudian digunakan untuk memprediksi rating pengguna terhadap buku-buku yang belum pernah mereka beri rating.
  - Fungsi recommend_books_for_user()
    - Mengambil semua ISBN unik dari df_books.
    - Menentukan ISBN yang belum diberi rating oleh pengguna.
    - Menggunakan model SVD untuk memprediksi rating pengguna terhadap ISBN yang belum dilihat.
    - Mengurutkan hasil prediksi dan memilih n buku dengan rating tertinggi.
    - Menggabungkan ISBN hasil rekomendasi dengan informasi lengkap dari df_books.

- Parameter:
  - Algoritma:	SVD (Singular Value Decomposition) dari pustaka Surprise.
  - n_factors	100 (default): Jumlah dimensi dalam representasi latent factors pengguna dan item. Semakin besar nilainya, semakin kompleks representasi yang dipelajari.
  - n_epochs	20 (default): Jumlah iterasi pelatihan penuh terhadap dataset.
  - lr_all	0.005 (default): Learning rate yang digunakan untuk semua parameter model.
  - reg_all	0.02 (default): Nilai regulasi untuk mencegah overfitting dengan membatasi bobot model.
  - rating_scale	(0, 10): Skala rating dari pengguna. Digunakan saat membangun objek Reader.
  - test_size	0.2: Rasio data uji saat membagi dataset (80% latih, 20% uji).
  - random_state	42: Nilai seed untuk memastikan hasil pembagian data yang konsisten dan reprodusibel.
  - n=20: Jumlah buku yang direkomendasikan untuk setiap pengguna berdasarkan prediksi rating tertinggi.

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
- Precision@K (untuk Content-Based Filtering)
  - Precision@K adalah metrik evaluasi yang mengukur proporsi item relevan di antara K item yang direkomendasikan. Dalam konteks sistem rekomendasi buku, precision@K menunjukkan seberapa banyak dari buku-buku yang direkomendasikan benar-benar termasuk dalam daftar buku yang disukai oleh pengguna (berdasarkan rating historis).

    <img src="https://raw.githubusercontent.com/zitaarisenda/Sistem-Rekomendasi/main/image.png" width="500"/>
  - Interpretasi: Nilai Precision@K yang tinggi menunjukkan bahwa sistem mampu memberikan rekomendasi yang relevan bagi pengguna. Sebaliknya, nilai yang rendah menunjukkan bahwa sebagian besar rekomendasi tidak sesuai dengan preferensi pengguna.
  
- Root Mean Squared Error (RMSE) (untuk Collaborative Filtering - SVD)
  - Digunakan untuk mengukur seberapa besar rata-rata kesalahan antara nilai prediksi dan nilai aktual rating pengguna terhadap buku.
    
    <img src="https://raw.githubusercontent.com/zitaarisenda/Sistem-Rekomendasi/main/Screenshot%202025-06-03%20172800.png" width="500"/>
  - Interpretasi: Semakin kecil nilai RMSE, semakin dekat prediksi model ke nilai sebenarnya. RMSE = 0 artinya prediksi sempurna. Dalam konteks rating skala 0–10.

### Hasil Evaluasi Model
1. Content-Based Filtering (Nearest Neighbors)
- Evaluasi dilakukan pada 100 pengguna pertama untuk mengurangi waktu komputasi, mengingat ukuran dataset yang besar (267.790 buku dan 433.671 interaksi). Proses evaluasi penuh terlalu berat karena setiap pengguna dibandingkan dengan seluruh koleksi buku.
  - Metrik: Precision@10
  - Hasil: 0.0057 (sekitar 0,57%)
- Interpretasi:
  - Nilai precision yang rendah menunjukkan bahwa sebagian besar rekomendasi belum sesuai dengan preferensi pengguna. Hal ini kemungkinan disebabkan oleh:
    - Fitur teks terbatas (judul, penulis, tahun, penerbit)
    - Tidak adanya personalisasi berdasarkan rating
    - Data interaksi yang sangat sparse
  - Meski begitu, Content-Based Filtering tetap bermanfaat untuk menangani kasus cold-start, khususnya saat item belum memiliki rating.

2. Collaborative Filtering (SVD)
- Evaluasi dilakukan menggunakan dataset uji yang dihasilkan dari fungsi train_test_split (pustaka Surprise).
- Metrik yang digunakan: Root Mean Squared Error (RMSE)
- Hasil RMSE: 1.6
- Skala rating: 0 hingga 10
- Interpretasi: Nilai RMSE sebesar 1.6 menunjukkan bahwa prediksi rating yang dihasilkan model memiliki rata-rata deviasi sebesar 1.6 poin dari rating aktual. Mengingat skala rating antara 0 hingga 10, nilai ini tergolong cukup besar. Hal ini menunjukkan bahwa model masih memiliki ruang untuk peningkatan akurasi.

### Keterkaitan Model dengan Business Understanding
Dua pendekatan yang telah diimplementasikan yakni Content-Based Filtering menggunakan K-Nearest Neighbors (KNN) dan Collaborative Filtering berbasis Singular Value Decomposition (SVD) secara langsung menjawab kedua problem statement dan mendukung pencapaian goals yang telah ditetapkan.

1. Menjawab Problem Statement
- Rekomendasi berbasis preferensi pengguna:
  - Content-Based Filtering memanfaatkan fitur buku (judul, penulis, tahun, penerbit) yang direpresentasikan sebagai vektor TF-IDF. Kemudian, digunakan algoritma K-Nearest Neighbors dengan cosine similarity untuk menemukan buku yang paling mirip.
  - Ini membantu pengguna menemukan buku lain yang sejenis dengan yang mereka sukai, bahkan tanpa perlu riwayat interaksi yang banyak.
- Pemanfaatan data historis untuk rekomendasi dinamis:
  - Collaborative Filtering dengan SVD mempelajari pola interaksi historis (rating) antara pengguna dan buku. Model ini dapat memberikan saran berdasarkan perilaku pengguna lain yang mirip, memungkinkan sistem belajar preferensi yang tidak eksplisit.

🎯 Pencapaian Goals
Kedua model berhasil dibangun dan dievaluasi menggunakan metrik yang sesuai:

Precision@10 pada Content-Based Filtering menunjukkan nilai 0.0057 untuk 100 pengguna secara acak. Meski masih rendah, hasil ini memberi insight bahwa model perlu pengayaan fitur konten atau pendekatan hybrid untuk akurasi lebih baik.

RMSE sebesar 1.6 untuk SVD menunjukkan bahwa rata-rata kesalahan prediksi masih sekitar 1.6 poin dari rating aktual (skala 0–10), yang menandakan adanya ruang perbaikan namun sudah berfungsi sebagai baseline model.

Dengan implementasi dua pendekatan yang saling melengkapi, sistem rekomendasi dapat disesuaikan baik untuk pengguna baru (cold-start problem) maupun pengguna lama dengan banyak data interaksi.

📌 Dampak dari Solusi yang Dirancang
KNN untuk Content-Based Filtering:
Cocok digunakan dalam situasi ketika data pengguna masih terbatas, seperti pengguna baru atau buku baru. Proses pencarian buku mirip dapat memberikan pengalaman eksplorasi yang lebih personal dan cepat.

SVD untuk Collaborative Filtering:
Memberikan personalisasi mendalam berdasarkan preferensi historis, cocok untuk pengguna aktif. Model ini memperkaya pengalaman pengguna dan meningkatkan loyalitas mereka terhadap platform.

Evaluasi dengan Precision@k dan RMSE:
Memberikan ukuran objektif terhadap kualitas rekomendasi dan menjadi dasar yang jelas untuk iterasi model berikutnya.


