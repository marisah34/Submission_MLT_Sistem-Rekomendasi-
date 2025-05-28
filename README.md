# Laporan Proyek Machine Learning - Marisah Lofiana

## Project Overview

Proyek ini bertujuan untuk membangun sistem rekomendasi buku berbasis model collaborative filtering menggunakan data Book-Crossing. Sistem rekomendasi seperti ini sangat berguna untuk membantu pengguna menemukan buku yang sesuai dengan preferensi mereka, terutama ketika jumlah pilihan sangat besar. Dengan memanfaatkan interaksi historis pengguna dan item (dalam hal ini buku), model dapat mempelajari pola dan memberikan saran yang relevan.

Seiring dengan semakin melimpahnya koleksi buku digital di berbagai platform, banyak pengguna mengalami kesulitan dalam menemukan bacaan yang cocok dengan minat mereka, menciptakan masalah kelebihan informasi (information overload). Hal ini terutama terasa di lingkungan akademik seperti perpustakaan digital. Sistem rekomendasi hadir sebagai solusi untuk menyajikan saran yang bersifat personal. Namun, tantangan seperti terbatasnya data eksplisit (misalnya rating) dan kebutuhan akan sistem yang adaptif masih perlu diatasi. Penelitian sebelumnya menunjukkan bahwa pendekatan seperti collaborative filtering berbasis matrix factorization mampu mengurangi masalah data sparsity dan cold start dengan cukup baik, meski masih perlu eksplorasi lebih lanjut untuk menghasilkan sistem yang kontekstual dan responsif terhadap kebutuhan pengguna baru.

Masalah sistem rekomendasi menjadi penting karena platform digital saat ini memiliki jumlah konten yang sangat banyak dan pengguna memiliki keterbatasan waktu dan informasi. Sistem yang baik akan meningkatkan pengalaman pengguna serta potensi keterlibatan dan penjualan dalam konteks komersial.

## Business Understanding
### Problem Statement
Berikut beberapa pernyataan Masalah: 
- Bagaimana cara membantu pengguna menemukan buku yang sesuai dengan preferensi mereka, terutama ketika jumlah koleksi buku sangat besar dan membuat proses pencarian menjadi kurang efisien?
- Bagaimana cara membangun sistem rekomendasi yang lebih personal dan relevan, yang mampu mempertimbangkan pola interaksi serta preferensi individual pengguna secara lebih mendalam?
- Bagaimana cara mengatasi cold-start problem, yaitu kondisi di mana pengguna baru atau buku baru belum memiliki riwayat interaksi atau rating, sehingga tetap memungkinkan sistem memberikan rekomendasi yang akurat?

### Goals 

Berikut beberapa tujuan proyeng yang menjawab beberapa pernyataan masalah: 

- Mengembangkan sistem rekomendasi berbasis content-based filtering yang dapat merekomendasikan lima buku serupa berdasarkan informasi penulis bukudan penerbit. Sistem ini akan bekerja dengan mengukur kesamaan antar buku untuk membantu pengguna mengeksplorasi bacaan yang sejenis.
- Membangun sistem rekomendasi berbasis collaborative filtering yang mampu merekomendasikan sepuluh buku baru kepada pengguna berdasarkan histori interaksi pengguna dengan buku-buku sebelumnya. Sistem ini akan memanfaatkan data rating untuk mempelajari pola preferensi pengguna.
- Mengatasi masalah cold-start pada pengguna atau buku baru dengan mengandalkan pendekatan berbasis konten, sehingga tetap memungkinkan sistem memberikan rekomendasi meskipun belum tersedia data eksplisit seperti rating.

### Solution statements

- **Content-Based Filtering**: Metode ini memanfaatkan pendekatan TF-IDF untuk mengubah informasi metadata buku—seperti nama penulis dan penerbit—menjadi representasi vektor numerik. Selanjutnya, sistem menghitung tingkat kemiripan antar buku dengan menggunakan cosine similarity, lalu merekomendasikan lima buku yang paling mirip dengan preferensi pengguna. Pendekatan ini cukup handal untuk menangani masalah cold-start karena tidak memerlukan riwayat interaksi pengguna sebelumnya.

- **Collaborative Filtering dengan RecommenderNet**: Teknik ini menggunakan RecommenderNet, yaitu model deep learning yang mempelajari representasi tersembunyi (laten) dari pengguna dan buku berdasarkan interaksi berupa rating. Model ini kemudian memprediksi skor relevansi untuk buku-buku yang belum dibaca, dan menyarankan sepuluh buku dengan skor tertinggi. Dengan demikian, sistem dapat memberikan rekomendasi yang lebih personal dan relevan berdasarkan pola perilaku pengguna lain yang memiliki kesamaan.

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari platform **Kaggle** dengan nama **Book Recommendation Dataset**. Dataset ini mencakup tiga file utama dalam format CSV, yaitu `Books`, `Ratings`, dan `Users`, yang diperoleh dari interaksi pengguna dengan buku di layanan **Amazon Web Services**. Rincian setiap file sebagai berikut:

- **Books.csv** memuat 271.360 entri dan terdiri atas 8 kolom yang mencakup informasi mendetail tentang buku.

- **Ratings.csv** berisi 1.149.780 data penilaian yang diberikan pengguna terhadap buku, dengan 3 kolom fitur utama.

- **Users.csv** mencakup data 278.858 pengguna, juga dengan 3 kolom yang menjelaskan informasi terkait pengguna.

Sumber Dataset : [Book Rekomendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

### **Deskripsi Variabel**

**Books.csv**

* `ISBN` : Nomor identifikasi unik untuk setiap buku.
* `Book-Title` : Judul buku.
* `Book-Author` : Nama penulis buku.
* `Year-Of-Publication` : Tahun terbit buku.
* `Publisher` : Nama penerbit buku.
* `Image-URL-S` : URL gambar buku ukuran kecil.
* `Image-URL-M` : URL gambar buku ukuran sedang.
* `Image-URL-L` : URL gambar buku ukuran besar.

**Ratings.csv**

* `User-ID` : ID unik pengguna.
* `ISBN` : Nomor ISBN buku yang diberi rating.
* `Book-Rating` : Skor rating yang diberikan pengguna (rentang 0–10).

**Users.csv**

* `User-ID` : ID unik pengguna.
* `Location` : Lokasi tempat tinggal pengguna.
* `Age` : Usia pengguna.

### **Exploratory Data Analysis (Univariate)**

**1. Dataset: Books**

Dataset `Books` menyimpan informasi terkait koleksi buku yang tersedia dalam sistem Book-Crossing. Langkah awal dalam eksplorasi data ini bertujuan untuk memperoleh gambaran umum mengenai karakteristik buku yang ada.

Beberapa statistik deskriptif awal yang ditemukan dari dataset antara lain:

- Total jumlah entri buku berdasarkan ISBN: 271.360

- Jumlah judul buku yang bersifat unik: 242.135

- Total penulis yang berbeda: 102.023

- Jumlah penerbit yang tercatat secara unik: 16.808

- Banyaknya variasi tahun terbit (`Year-Of-Publication`): 202 nilai berbeda

**Perbedaan antara ISBN dan Judul Buku**

Terdapat selisih jumlah antara ISBN dan judul buku, yang menunjukkan bahwa satu judul bisa memiliki beberapa ISBN. Untuk mengevaluasi fenomena ini, digunakan kode pemrograman guna menghitung frekuensi kemunculan tiap judul buku dalam dataset.

```
books['Book-Title'].value_counts()
```

Hasil dari kode tersebut menunjukkan bahwa terdapat beberapa buku dengan judul yang sama.
![image](https://github.com/user-attachments/assets/ca84691e-6ab9-42b2-88f6-80d1c33748d4)

Contohnya, pada judul "Selected Poems", pencarian berdasarkan nama judul yang sama menghasilkan sejumlah entri buku yang memiliki kesamaan pada judul, namun berbeda dalam hal ISBN, nama penulis, penerbit, maupun tahun terbit. Hal ini menunjukkan bahwa satu judul dapat merujuk pada beberapa versi atau edisi buku yang berbeda.

| ISBN       | Book-Title     | Book-Author             | Year-Of-Publication | Publisher                             |
| ---------- | -------------- | ----------------------- | ------------------- | ------------------------------------- |
| 081120958X | Selected Poems | William Carlos Williams | 1985                | New Directions Publishing Corporation |
| 0811201465 | Selected Poems | K. Patchen              | 1957                | New Directions Publishing Corporation |
| 0679750800 | Selected Poems | Rita Dove               | 1993                | Vintage Books USA                     |

**Pembersihan Kolom Tahun Terbit**
![image](https://github.com/user-attachments/assets/5f3f9824-c608-447a-9712-863b41cca14f)

Selama proses eksplorasi terhadap kolom `Year-Of-Publication`, ditemukan adanya sejumlah nilai yang tidak sesuai format tahun, seperti entri yang berisi nama penerbit (contoh: `DK Publishing Inc`, `Gallimard`) ataupun angka yang tidak wajar seperti `0`, `1376`, `2020`, hingga `2050`.

Setelah dilakukan analisis lebih mendalam, diketahui bahwa penyebab utama dari anomali ini adalah pergeseran data antar kolom pada saat proses input. Untuk menjaga kualitas dan konsistensi data, tiga baris dengan kesalahan input tersebut akhirnya dihapus dari dataset.

Kemudian untuk nilai-nilai tahun yang tidak realistis (di luar rentang wajar publikasi, seperti `0` atau di atas `2006`) diubah menjadi nilai kosong (`NaN`) dan diisi dengan rata-rata tahun yang valid menggunakan codingan di bawah ini.

```
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
books.loc[(books['Year-Of-Publication'] > 2006) | (books['Year-Of-Publication'] == 0),'Year-Of-Publication'] = np.nan

# mengganti NaN dengan nilai rata-rata yearOfPublication
books['Year-Of-Publication'].fillna(round(books['Year-Of-Publication'].mean()), inplace=True)
```

Setelah tahapan tersebut, kolom tahun terbit telah berisi nilai yang realistis. Kemudian, tipe data pada kolom `Year-Of-Publication` diubah dari tipe data `object` menjadi `integer` agar bisa diproses secara numerik. 

![image](https://github.com/user-attachments/assets/c88d7b6c-e8c3-4975-9531-5c69ee757888)

**Missing Value**

Berikut adalah tabel jumlah *missing values* (nilai kosong) dalam dataset `books`:

| Kolom               | Missing Values |
| ------------------- | -------------- |
| Book-Author         | 2              |
| Publisher           | 2              |
| ISBN                | 0              |
| Book-Title          | 0              |
| Year-Of-Publication | 0              |

Hanya ada dua kolom yang memiliki missing values, dan jumlahnya sangat kecil sehingga tidak signifikan terhadap keseluruhan dataset.

**2. Dataset: Ratings**

Dataset `Ratings` mencatat evaluasi yang diberikan pengguna terhadap buku-buku yang ada. Dataset ini memegang peranan penting karena menjadi dasar dalam menganalisis preferensi pengguna serta membangun sistem rekomendasi.

Statistik dasar yang diperoleh antara lain:

- Jumlah pengguna yang memberikan penilaian: 105.283 orang

- Total entri rating yang tercatat: 340.556 data

- Jumlah nilai unik dalam skala rating: 11 angka berbeda

**Rentang Skala Rating**

Penilaian diberikan dalam rentang **0 hingga 10**, yang menghasilkan 11 kategori nilai. Berdasarkan pengamatan awal:

- Rating 0 kemungkinan besar menandakan tidak adanya penilaian eksplisit dari pengguna—bisa diartikan sebagai implicit rating atau kemungkinan kesalahan dalam pengisian data.

- Rating antara 1 sampai 10 merepresentasikan penilaian eksplisit, di mana semakin tinggi nilainya, semakin besar tingkat kesukaan pengguna terhadap buku tersebut.

Distribusi Nilai Rating : 

| Skor Rating   | Keterangan                |
| ------------- | ------------------------  |
| 0             | Tidak diberi / default    |
| 1-3           | Penilaian sangat rendah   |
| 4-6           | Penilaian sedang          |
| 7-10          | Penilaian tinggi          |

**Statistik Deskriptif Rating**

Hasil dari `ratings.describe()` memberikan ringkasan sebagai berikut:
![image](https://github.com/user-attachments/assets/f2ea402a-82be-425f-8074-3ee24465dffc)

Nilai tengah (median) berada di 0, mengindikasikan bahwa sebagian besar rating tidak eksplisit (banyak 0). Namun, nilai kuartil atas (75%) adalah 8, yang memperlihatkan bahwa ketika rating diberikan, nilainya cenderung tinggi.

**Missing Values**

Berikut adalah tabel jumlah *missing values* (nilai kosong) dalam dataset `ratings`:

|Kolom	         |Missing Values    |
|--------------  |----------------  |
|User-ID	       |0                 |
|ISBN	           |0                 |
|Book-Rating	   |0                 |

Tidak ada missing value dalam dataset ini.

**3. Dataset: Users**

Dataset `users` mencatat informasi dasar tentang pengguna, seperti ID, lokasi, dan umur.

Informasi awal yang diperoleh:

Jumlah user unik: 278.858
Jumlah lokasi unik (`Location`): 57.339
Jumlah nilai unik pada kolom `Age`: 166

**Umur Pengguna**
![image](https://github.com/user-attachments/assets/7c0716e3-eda7-4294-a13f-93997284e280)

Kolom `Age` memiliki nilai-nilai yang tidak masuk akal seperti `0`, `231`, dan `244`. Maka dilakukan pembersihan dengan cara:

- Umur di bawah 5 tahun atau di atas 90 tahun dianggap tidak valid dan diubah menjadi `NaN`
- Nilai `NaN` kemudian diisi dengan rata-rata umur pengguna yang valid
- Tipe data diubah ke `integer` agar lebih konsisten

Berikut code yang digunakan
```
# Umur di bawah 5 dan di atas 90 tidak masuk akal, maka menggantinya dengan NaN
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan

# Replacing NaNs with mean
users.Age = users.Age.fillna(users.Age.mean())

# Setting the data type as int
users.Age = users.Age.astype(np.int32)
```

Hasil deskriptif umur pengguna setelah dibersihkan:

|Statistik    	|Nilai        |
|-------------  |------------ |
|Rata-rata	    |34.4         |
|Median	        |34           |
|Minimum	      |5            |
|Maksimum	      |90           |

Sebagian besar pengguna berada di rentang usia produktif (sekitar 29–35 tahun), menunjukkan demografi utama pengguna platform ini.

**Missing Values**

Berikut adalah tabel jumlah *missing values* (nilai kosong) dalam dataset `users`:

|Kolom	         |Missing Values    |
|--------------  |----------------  |
|User-ID	       |0                 |
|Location	       |0                 |
|Age        	   |0                 |

Tidak ada missing value dalam dataset ini.

### **Visualisasi Data**

**1. Books**
**Barplot 10 Author Teratas dengan Buku Terbitan Terbanyak**
![image](https://github.com/user-attachments/assets/b6aa64bc-0648-46f0-b9ff-59e673d10ffb)

Insight : Grafik menunjukkan sepuluh penulis dengan jumlah buku terbanyak yang diterbitkan. Agatha Christie berada di posisi teratas dengan lebih dari 600 buku, diikuti oleh William Shakespeare dan Stephen King. Penulis lain seperti Ann M. Martin dan Carolyn Keene juga menunjukkan produktivitas tinggi. Secara keseluruhan, grafik ini menyoroti dominasi penulis-penulis fiksi populer yang sangat produktif dalam dunia penerbitan.

**Barplot 10 Publisher Teratas dengan Buku Terbitan Terbanyak**
![image](https://github.com/user-attachments/assets/b431481e-012b-48ef-a854-00a4d31fd7b4)

Insight : Grafik tersebut menampilkan sepuluh penerbit dengan jumlah buku terbanyak yang diterbitkan. Harlequin menempati posisi teratas secara signifikan dengan lebih dari 7.000 buku, jauh melampaui penerbit lainnya. Di posisi berikutnya terdapat Silhouette, Pocket, dan Ballantine Books dengan jumlah publikasi yang berkisar antara 3.500 hingga 4.200 buku. Penerbit lainnya seperti Scholastic, Penguin Books, dan Warner Books juga masuk dalam daftar ini dengan kontribusi yang cukup besar. Secara keseluruhan, grafik ini menunjukkan bahwa Harlequin mendominasi industri penerbitan, kemungkinan besar karena fokusnya pada genre populer seperti roman.

**2. Ratings**
**Barplot Pembagian Pemeringkatan Buku**
![image](https://github.com/user-attachments/assets/7cdbbf7f-83ee-4c54-9175-bbb31b16e1cc)

Insight: Visualisasi barplot menunjukkan bahwa sebagian besar rating buku berada pada nilai 0, yang menandakan banyak buku belum pernah diberi penilaian. Di luar nilai 0, jumlah rating cenderung meningkat seiring naiknya nilai rating, dengan titik tertinggi pada rating 8, lalu sedikit menurun di rating 9 dan 10. Pola distribusi ini menunjukkan adanya kecenderungan ekstrem, di mana banyak buku tidak dinilai sama sekali, sementara buku yang dinilai cenderung mendapatkan rating yang cukup tinggi.

**3. Users**
**Barplot Distribusi Umur Pengguna**
![image](https://github.com/user-attachments/assets/70d8b4f3-0814-4de6-beb5-65c55dab14a2)

Insight: Boxplot ini menggambarkan sebaran usia pengguna, di mana konsentrasi utama usia berada antara 25 hingga 45 tahun, sebagaimana ditunjukkan oleh bagian tengah plot (kotak). Garis horizontal di dalam kotak merepresentasikan nilai median. Whisker pada kedua sisi menunjukkan jangkauan usia di luar kuartil bawah dan atas. Titik yang berada di atas whisker atas menandakan adanya outlier, yaitu pengguna dengan usia yang jauh melebihi rentang umum dibandingkan mayoritas pengguna lainnya.

## Data Preparation

Pada tahap ini, dilakukan serangkaian proses pembersihan dan transformasi data untuk memastikan bahwa data yang akan digunakan sudah bersih, sesuai konteks, dan layak untuk dianalisis lebih lanjut atau digunakan dalam pengembangan model rekomendasi. Langkah-langkah yang diterapkan antara lain:

### Melihat Ukuran Awal Dataset Buku
```
print('Banyak buku: ', len(books['ISBN'].unique()))
```

Kode `print('Banyak buku: ', len(books['ISBN'].unique()))` digunakan untuk menampilkan jumlah total buku unik yang terdapat dalam dataset `books` berdasarkan kolom 'ISBN'. Informasi ini memberikan gambaran mengenai skala dataset buku secara keseluruhan.

### Reduksi Ukuran Dataset
```
# Mengambil 15.000 baris data secara acak dari DataFrame books dengan random_state=5
reduced_books = books.sample(n=15000, random_state=5)
reduced_books

reduced_ratings = ratings[ratings['ISBN'].isin(unique_isbn_list)]
reduced_ratings
```

Untuk mempercepat proses analisis dan pengembangan model rekomendasi, dilakukan pengurangan skala pada dataset books dan ratings. Salah satunya dengan mengambil sampel acak sebanyak 15.000 baris dari dataset `books` menggunakan `books.sample(n=15000, random_state=5)`. Penggunaan parameter `random_state=5` bertujuan agar proses sampling menghasilkan data yang konsisten dan dapat direproduksi.

### Pengambilan Sampel Data Buku
```
unique_isbn_list = reduced_books['ISBN'].unique().tolist()
len(unique_isbn_list)
```

Dataframe hasil pengambilan sampel ini disimpan dalam variabel `reduced_books`. Jumlah ISBN unik dalam sampel ini kemudian diperiksa menggunakan kode `unique_isbn_list = reduced_books['ISBN'].unique().tolist()` dan `len(unique_isbn_list)`.

### Pemfilteran Data Rating
```
reduced_ratings = ratings[ratings['ISBN'].isin(unique_isbn_list)]
reduced_ratings
```

Dataframe ratings disaring agar hanya memuat data rating untuk buku-buku yang ISBN-nya termasuk dalam daftar ISBN unik dari *reduced_books*. Penyaringan ini dilakukan menggunakan perintah `reduced_ratings = ratings[ratings['ISBN'].isin(unique_isbn_list)]`. Dengan cara ini, reduced_ratings hanya mencakup rating yang relevan dengan sampel buku yang telah dipilih sebelumnya.

### Penghapusan Kolom URL Gambar
```
reduced_books = reduced_books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
```

Tiga kolom yang berisi informasi URL gambar buku, yaitu 'Image-URL-S', 'Image-URL-M', dan 'Image-URL-L', dihapus dari *dataframe* `reduced_books` menggunakan kode `reduced_books = reduced_books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])`. Kolom-kolom ini dianggap tidak relevan untuk analisis rekomendasi berbasis teks atau rating.

### Penggabungan Data Rating dan Buku
```
# Menggabungkan dataframe rating dengan book berdasarkan nilai ISBN
booksrate = pd.merge(reduced_ratings, reduced_books, on='ISBN', how='left')
booksrate
```

Dataframe `reduced_ratings` dan `reduced_books` digabungkan berdasarkan kolom 'ISBN' menggunakan fungsi `pd.merge(reduced_ratings, reduced_books, on='ISBN', how='left')`. Penggunaan `how='left'` memastikan bahwa semua rating dari `reduced_ratings` tetap ada, dan informasi buku yang sesuai dari `reduced_books` ditambahkan. Hasil penggabungan disimpan dalam dataframe `booksrate`.

### Penghitungan Jumlah Rating per Buku
```
# Menghitung jumlah rating kemudian menggabungkannya berdasarkan ISBN
booksrate.groupby('ISBN').sum()
```

Fungsi `booksrate.groupby('ISBN').sum()` digunakan untuk menghitung akumulasi rating berdasarkan ISBN dalam dataframe booksrate. Meskipun fungsi yang digunakan adalah `sum()`, informasi yang lebih relevan dalam konteks ini adalah seberapa sering setiap ISBN muncul, yang mencerminkan seberapa sering sebuah buku telah diberi rating. Hasil ini memberikan gambaran mengenai tingkat popularitas atau seberapa banyak sebuah buku dinilai dalam subset data tersebut.

### Penghapusan Missing Value
Setelah dilakukan proses EDA, ditemukan bahwa masih terdapat beberapa nilai kosong (*missing values*) dalam gabungan dataset `booksrate`. Berikut ini adalah jumlah missing value dari masing-masing kolom:
|Kolom	               |Missing Values   |
|-----------------     |---------------- |
|User-ID               |0                |
|ISBN                  |0                |
|Book-Rating	         |0                |
|Book-Title	           |0                |
|Book-Author	         |0                |
|Year-Of-Publication	 |0                |
|Publisher	           |1                |

Terlihat bahwa hanya ada satu *missing value* pada kolom `Publisher`, dan sisanya sudah bersih. Untuk membersihkan data dari *missing value*, digunakan fungsi `dropna()`:
```
booksrate_clean = booksrate.dropna()
```

Fungsi ini digunakan untuk secara otomatis menghapus baris yang mengandung nilai kosong pada salah satu kolom. Karena hanya satu baris yang tereliminasi, pengaruhnya terhadap keseluruhan dataset sangat minimal. Menghapus nilai yang hilang membantu mencegah potensi error selama proses analisis atau pemodelan, sekaligus menjaga kualitas dan konsistensi data dengan memastikan tidak ada entri yang tidak lengkap.

### Penghapusan Duplikat Data
Setelah membersihkan missing value, langkah selanjutnya adalah memastikan tidak ada data duplikat dalam data buku. Duplikat yang dimaksud di sini adalah data dengan ISBN yang sama muncul lebih dari sekali.
```
preparation = booksrate_clean.drop_duplicates('ISBN')
```

Kolom ISBN berfungsi sebagai identifikasi unik untuk setiap buku. Menghapus data duplikat berdasarkan kolom ini memastikan bahwa setiap buku hanya tercatat satu kali dalam dataset. Langkah ini penting untuk menghindari redundansi yang dapat menyebabkan bias, terutama dalam sistem rekomendasi berbasis konten, di mana akurasi sangat bergantung pada representasi unik dari setiap item.

### Mengonversi Kolom Menjadi List
Setelah data dibersihkan dari duplikat, kolom-kolom penting seperti `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication`, dan `Publisher` kemudian dikonversi ke dalam bentuk list.
```
book_isbn = preparation['ISBN'].tolist()
book_title = preparation['Book-Title'].tolist()
book_author = preparation['Book-Author'].tolist()
book_year = preparation['Year-Of-Publication'].tolist()
book_publisher = preparation['Publisher'].tolist()
```
Seluruh list yang dihasilkan memiliki panjang yang konsisten, yaitu 14.930 elemen, yang menunjukkan bahwa tidak ada data yang terlewat selama proses konversi. Mengubah data ke dalam format list memberikan fleksibilitas lebih dalam manipulasi, terutama pada tahap pengembangan sistem rekomendasi berbasis konten. Format ini juga memudahkan dalam pembuatan struktur seperti dictionary, penerapan TF-IDF vectorizer, maupun dalam penggabungan fitur metadata.

### Data Preparation untuk Content-Based Filtering
**1. Pembuatan Data books_new**

Langkah awal adalah menyusun dataframe baru bernama `books_new`.
```
# Membuat dictionary untuk data ‘book_isbn’, ‘book_title’, ‘book_author’, dan 'book_publisher'
books_new = pd.DataFrame({
    'isbn': book_isbn,
    'title': book_title,
    'author': book_author,
    'year': book_year,
    'publisher': book_publisher
})
books_new
```

Dalam tahap ini, data buku yang semula berbentuk series (kemungkinan hasil dari pembacaan awal dataset) dikonversi ke dalam format list. Data tersebut kemudian disusun ulang ke dalam dataframe baru bernama `books_new` dengan kolom-kolom seperti `'isbn'`, `'book_title'`, `'book_author'`, `'year_of_publication'`, dan `'publisher'`. Pembentukan dataframe ini bertujuan untuk mempermudah proses manipulasi dan transformasi data ke dalam struktur yang sesuai guna digunakan dalam model *content-based filtering*.

**2. Penggabungan Fitur Teks**

Untuk merepresentasikan konten setiap buku, informasi dari kolom 'book_author' dan 'publisher' digabungkan menjadi satu kolom teks baru bernama 'combined'.
```
# Gabungkan author dan publisher menjadi satu kolom string
data['combined'] = data['author'].fillna('') + ' ' + data['publisher'].fillna('')
```

Langkah ini dilakukan dengan mengisi nilai yang hilang (jika ada) pada kedua kolom tersebut dengan string kosong ('') dan kemudian menggabungkannya dengan spasi di antara keduanya. Kolom 'combined' ini akan menjadi dasar untuk perhitungan kemiripan konten antar buku.

**3.Ekstraksi Fitur dengan TF-IDF**

Teknik Term Frequency-Inverse Document Frequency (TF-IDF) digunakan untuk mengekstrak fitur-fitur penting dari teks dalam kolom 'combined'.
**- Inisialisasi `TfidfVectorizer()`**
  ```
  # Inisialisasi TF-IDF
tf = TfidfVectorizer()
```
Sebuah objek `TfidfVectorizer` dibuat. Vectorizer ini akan mengubah teks menjadi matriks representasi numerik, di mana setiap kata dalam korpus akan menjadi sebuah fitur. Bobot TF-IDF diberikan kepada setiap kata dalam setiap dokumen (dalam hal ini, setiap entri di kolom 'combined'), yang mencerminkan seberapa penting kata tersebut dalam dokumen relatif terhadap seluruh korpus.

**- Fit dan Transform ke Kolom 'combined'**
```
# Fit dan transform ke kolom gabungan
tfidf_matrix = tf.fit_transform(data['combined'])
```
Metode `fit_transform()` dari objek `TfidfVectorizer` dipanggil dengan kolom 'combined' sebagai input. Proses `fit` akan mempelajari kosakata dari seluruh teks dalam kolom 'combined', dan proses `transform` akan mengubah setiap entri teks menjadi vektor TF-IDF berdasarkan kosakata yang telah dipelajari. Hasilnya adalah matriks `tfidf_matrix`.

**- Melihat Fitur yang Dihasilkan**
```
# Melihat fitur yang dihasilkan
features = tf.get_feature_names_out()

# print(tfidf_matrix.shape)
tf.get_feature_names_out()
```
`tf.get_feature_names_out()` digunakan untuk mendapatkan daftar semua fitur (kata unik) yang telah diekstrak oleh `TfidfVectorizer`.

**- Melihat Ukuran Matriks TF-IDF**
```
# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tf.fit_transform(data['combined'])

# Melihat ukuran matrix tfidf
tfidf_matrix.shape
```
`tfidf_matrix.shape` memberikan dimensi dari matriks TF-IDF. Jumlah baris sesuai dengan jumlah buku dalam dataset, dan jumlah kolom sesuai dengan jumlah fitur unik yang ditemukan dalam kolom 'combined'.

**- Mengubah ke Matriks Padat**
```
# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()
```
`tfidf_matrix.todense()` mengubah matriks sparse TF-IDF menjadi matriks padat. Meskipun representasi sparse lebih efisien untuk penyimpanan dan perhitungan dengan data teks berdimensi tinggi, representasi padat mungkin lebih mudah dipahami dalam beberapa kasus.

**-Membuat Dataframe Matriks TF-IDF**
```
# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan author, publisher
# Baris diisi dengan nama buku

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=data.title
).sample(22, axis=1).sample(10, axis=0)
```

Sebuah dataframe dibuat dari matriks padat TF-IDF. Kolom-kolom dataframe ini diberi nama sesuai dengan fitur-fitur yang diekstrak (kata-kata dari 'author' dan 'publisher'), dan indeksnya adalah judul buku dari kolom 'title' (kemungkinan kolom 'book_title' telah diubah namanya menjadi 'title' pada tahap sebelumnya). Metode `.sample()` digunakan untuk menampilkan sebagian kecil dari dataframe ini, memudahkan visualisasi bobot TF-IDF untuk beberapa buku dan fitur secara acak.

**Tujuan Keseluruhan**

Tahapan ini bertujuan untuk mengonversi informasi dalam bentuk teks dari penulis dan penerbit buku ke dalam format numerik menggunakan metode TF-IDF. Selanjutnya, dilakukan penghitungan tingkat kemiripan antar buku berdasarkan representasi numerik tersebut dengan menggunakan metode cosine similarity. Hasil dari proses ini berupa matriks kemiripan `(cosine_sim_df)`, yang kemudian dimanfaatkan sebagai dasar dalam sistem rekomendasi berbasis konten. Buku-buku yang memiliki nilai cosine similarity tinggi dianggap memiliki kemiripan isi dan layak untuk saling direkomendasikan.

### Data Preparation untuk Collaborative Filtering

**1. Penggunaan Dataframe `reduced_ratings`:**

```
# Membaca dataset
df = reduced_ratings
df
```
Langkah awal dalam persiapan data untuk collaborative filtering adalah menggunakan dataframe yang disebut `reduced_ratings`. Ini mengindikasikan bahwa mungkin telah dilakukan proses pengurangan atau pemilihan sebagian data rating dari dataset awal. Tujuannya bisa untuk mengurangi kompleksitas komputasi atau fokus pada interaksi pengguna dan item yang lebih relevan.

**2. Encoding User ID dan ISBN**

Proses encoding adalah mengubah data kategorikal (dalam hal ini, ID pengguna dan ISBN buku yang kemungkinan besar berupa string atau angka unik) menjadi representasi numerik berupa indeks integer. Hal ini penting karena model machine learning, termasuk model collaborative filtering, umumnya bekerja lebih baik dengan input numerik.

**- Membuat List ID Unik**: Kode `df['User-ID'].unique().tolist()` dan `df['ISBN'].unique().tolist()` menghasilkan list yang berisi nilai-nilai unik dari kolom 'User-ID' dan 'ISBN'. Ini memastikan bahwa setiap pengguna dan setiap buku hanya direpresentasikan satu kali dalam list.

**- Membuat Dictionary Pemetaan (Encoding)**: Dua dictionary dibuat untuk setiap ID pengguna dan ISBN buku:

  - `user_to_user_encoded` dan `book_to_book_encoded`
    
      ```
      # Melakukan encoding userID
      user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
      print('encoded userID : ', user_to_user_encoded)
      # Melakukan proses encoding ISBN
      book_to_book_encoded = {x: i for i, x in enumerate(book_isbn)}
      ```
      Dictionary ini memetakan setiap nilai unik (ID pengguna atau ISBN) ke sebuah indeks integer yang berurutan. Misalnya,        pengguna dengan ID '22' mungkin dipetakan ke indeks 0, pengguna dengan ID '53' ke indeks 1, dan seterusnya.
      
   - `user_encoded_to_user` dan `book_encoded_to_book`
    
      ```
      # Melakukan proses encoding angka ke ke userID
      user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
      print('encoded angka ke userID: ', user_encoded_to_user)

      # Melakukan proses encoding angka ke ISBN
      book_encoded_to_book = {i: x for i, x in enumerate(book_isbn)}
      ```
      
      Ini adalah dictionary kebalikan dari yang sebelumnya, memetakan indeks integer kembali ke nilai asli (ID pengguna atau       ISBN). Ini berguna untuk interpretasi hasil model.
      
**- Menerapkan Pemetaan ke Dataframe**

  ```
  # Mapping User-ID ke dataframe user
  df['user'] = df['User-ID'].map(user_to_user_encoded)
  
  # Mapping ISBN ke dataframe book
  df['book'] = df['ISBN'].map(book_to_book_encoded)
  ```
Metode `.map()` digunakan untuk menerapkan dictionary encoding ke kolom 'User-ID' dan 'ISBN' dalam dataframe `df`. Ini akan membuat dua kolom baru, 'user' dan 'book', yang berisi indeks integer yang sesuai untuk setiap interaksi (rating).

**3. Mendapatkan Jumlah Pengguna dan Buku:**

```
# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)

# Mendapatkan jumlah buku
num_book = len(book_to_book_encoded)
print(num_book)
```

Setelah proses encoding selesai, jumlah pengguna unik `(num_users)` dan jumlah buku unik `(num_book)` ditentukan dengan menghitung panjang dari dictionary hasil encoding. Data ini sangat penting karena digunakan untuk menentukan ukuran atau dimensi dari embedding layer yang akan digunakan dalam model neural network untuk metode collaborative filtering.

**4. Konversi Tipe Data Rating:**

```
# Mengubah rating menjadi nilai float
df['Book-Rating'] = df['Book-Rating'].values.astype(np.float32)
```

Kolom 'Book-Rating' diubah menjadi tipe data float32. Ini umum dilakukan untuk nilai rating karena model machine learning seringkali bekerja dengan bilangan floating-point.

**5. Identifikasi Rentang Rating:**

```
# Nilai minimum rating
min_rating = min(df['Book-Rating'])

# Nilai maksimal rating
max_rating = max(df['Book-Rating'])
```

Nilai minimum (min_rating) dan maksimum (max_rating) dari kolom 'Book-Rating' diidentifikasi. Informasi ini akan digunakan untuk normalisasi nilai rating ke dalam rentang yang lebih kecil (biasanya 0 hingga 1).

**6. Pengacakan Dataset:**

```
# Mengacak dataset
df = df.sample(frac=1, random_state=42)
df
```

Dataframe `df` diacak menggunakan `df.sample(frac=1`, `random_state=42)`. `frac=1` berarti semua baris akan dikembalikan, tetapi dalam urutan acak. `random_state=42` digunakan untuk memastikan bahwa pengacakan akan menghasilkan urutan yang sama setiap kali kode dijalankan, yang penting untuk reproducibility.

**7. Pemisahan Fitur dan Label:**

```
# Membuat variabel x untuk mencocokkan data user dan book menjadi satu value
x = df[['user', 'book']].values

# Membuat variabel y untuk membuat rating dari hasil
y = df['Book-Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
```

  - Fitur (x): Kolom 'user' dan 'book', yang telah melalui proses encoding menjadi indeks, dipilih sebagai fitur atau input (x). Kombinasi indeks pengguna dan buku ini akan digunakan sebagai masukan dalam model collaborative filtering. Untuk mengonversi data dari bentuk dataframe ke array NumPy, digunakan fungsi `.values`.

  - Label (y): Kolom 'Book-Rating' digunakan sebagai target atau label (y). Nilai rating ini dinormalisasi ke dalam skala 0 hingga 1 menggunakan rumus: (x - min_rating) / (max_rating - min_rating). Proses normalisasi ini bertujuan untuk meningkatkan kestabilan dan efisiensi saat model melakukan pembelajaran.

**8. Pembagian Data Latih dan Validasi:**
```
# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```

Dataset dibagi menjadi set pelatihan (80%) dan validasi (20%). ``train_indices`` dihitung untuk menentukan titik pemisahan. Kemudian, fitur (`x`) dan label (`y`) dibagi menjadi `x_train`, `x_val`, `y_train`, dan `y_val`. Set pelatihan akan digunakan untuk melatih model, sedangkan set validasi akan digunakan untuk mengevaluasi kinerja model selama pelatihan dan membantu dalam *tuning hyperparameter*.

**Tujuan Keseluruhan**

Secara umum, tahap ini bertujuan untuk mempersiapkan data rating antara pengguna dan buku ke dalam bentuk numerik yang dapat digunakan dalam pelatihan model collaborative filtering. Identitas pengguna dan buku diubah menjadi indeks melalui proses encoding, sementara nilai rating dinormalisasi agar berada dalam skala yang konsisten. Selain itu, pembagian data dilakukan untuk memastikan proses pelatihan dan evaluasi model dapat berjalan secara optimal.

## Modeling 

Pada tahap ini, dua pendekatan algoritma yang berbeda diimplementasikan untuk membangun sistem rekomendasi buku: Content-Based Filtering (CBF) dan Collaborative Filtering (CF). Setiap pendekatan memiliki cara kerja, kelebihan, dan kekurangan yang berbeda dalam menghasilkan rekomendasi.

### Content-Based Filtering

Pendekatan Content-Based Filtering merekomendasikan buku kepada pengguna berdasarkan kemiripan atribut konten antar buku. Dalam implementasi ini, fitur 'author' dan 'publisher' dari buku digunakan sebagai dasar untuk mengukur kemiripan.

**Cara Kerja:**

**1. Representasi Fitur:** Teks dari kolom 'combined', yang merupakan gabungan dari informasi 'author' dan 'publisher', dikonversi menjadi representasi numerik menggunakan metode Term Frequency-Inverse Document Frequency (TF-IDF). Teknik ini memberikan bobot pada setiap kata berdasarkan seberapa sering kata tersebut muncul dalam satu dokumen dibandingkan dengan seluruh koleksi dokumen lainnya.

**2. Pengukuran Kemiripan:** Setelah didapatkan vektor TF-IDF untuk masing-masing buku, kemiripan antar buku dihitung menggunakan fungsi `cosine_similarity()` dari pustaka `sklearn.metrics.pairwise`. Cosine similarity mengukur derajat kesamaan antara dua vektor dalam ruang berdimensi banyak, dengan nilai antara -1 (sangat tidak mirip) hingga 1 (sangat mirip). Hasil pengukuran ini disimpan dalam bentuk matriks NumPy bernama `cosine_sim`.

**3. Konversi Matriks Kemiripan ke DataFrame:** Matriks `cosine_sim` kemudian diubah menjadi sebuah DataFrame bernama `cosine_sim_df`, dengan judul buku (dari kolom 'title') sebagai indeks dan juga nama kolom. Setiap elemen (i, j) dalam DataFrame ini menunjukkan skor cosine similarity antara buku ke-i dan buku ke-j. Karena setiap buku dibandingkan dengan seluruh buku lainnya, ukuran DataFrame ini berbentuk persegi dengan dimensi sebanyak jumlah total buku.

**4. Proses Rekomendasi:** Untuk menghasilkan rekomendasi, sistem akan mencari buku-buku dengan nilai cosine similarity tertinggi terhadap buku yang dipilih, berdasarkan informasi dalam `cosine_sim_df`. Buku-buku yang memiliki skor tertinggi dianggap paling relevan dan layak direkomendasikan.

#### Implementasi Fungsi `book_recommendations`:

```
def book_recommendations(title, similarity_data=cosine_sim_df, items=data[['title', 'author', 'publisher']], k=5):
  index = similarity_data.loc[:,title].to_numpy().argpartition(
        range(-1, -k, -1))

  # Mengambil data dengan similarity terbesar dari index yang ada
  closest = similarity_data.columns[index[-1:-(k+2):-1]]

  # Drop title agar nama buku yang dicari tidak muncul dalam daftar rekomendasi
  closest = closest.drop(title, errors='ignore')

  return pd.DataFrame(closest).merge(items).head(k)
```

Fungsi `book_recommendations` menerima judul buku (`title`), dataframe kemiripan (`similarity_data`), dataframe informasi buku (`items`), dan jumlah rekomendasi yang diinginkan (`k`) sebagai input. Fungsi ini bekerja dengan:

1. Mencari posisi atau indeks dari buku yang sesuai dengan judul yang diberikan dalam DataFrame cosine similarity.

2. Menggunakan metode `argpartition` untuk menentukan indeks dari k buku dengan nilai cosine similarity tertinggi, yang berarti buku-buku tersebut memiliki tingkat kemiripan paling besar.

3. Mengambil judul-judul buku yang paling mirip dari DataFrame kemiripan berdasarkan indeks yang telah ditemukan.

4. Mengeluarkan judul buku yang menjadi input awal dari daftar hasil rekomendasi, agar buku tersebut tidak direkomendasikan kepada dirinya sendiri.

5. Menggabungkan daftar buku yang direkomendasikan dengan DataFrame yang memuat detail buku, seperti nama penulis dan penerbit, agar informasi yang ditampilkan lebih lengkap.

6. Menghasilkan dan mengembalikan DataFrame yang berisi k buku teratas berdasarkan skor kemiripan tertinggi sebagai hasil akhir rekomendasi.

**Contoh Hasil Rekomendasi CBF:**

Untuk buku "Harry Potter and the Prisoner of Azkaban (Book 3)", sistem merekomendasikan 5 buku teratas berdasarkan kemiripan penulis dan penerbit:

![image](https://github.com/user-attachments/assets/d1d25229-9cae-4a73-9b22-5b2ae8af9850)


**Kelebihan Content-Based Filtering (CBF):**

- Tidak butuh data pengguna: Hanya menggunakan data dari item itu sendiri.

- Jelas alasannya: Rekomendasi didasarkan pada kemiripan fitur, jadi mudah dijelaskan.

- Bisa rekomendasi item baru: Cocok untuk buku baru yang belum punya rating.

**Kekurangan CBF:**

- Terbatas fitur: Hasil rekomendasi sangat bergantung pada kelengkapan data item.

- Terlalu mirip: Cenderung hanya menyarankan item yang sangat serupa, kurang variasi.

- Tidak pakai data pengguna lain: Tidak memanfaatkan selera pengguna lain yang mirip.

### Collaborative Filtering

Pendekatan Collaborative Filtering merekomendasikan buku kepada pengguna berdasarkan pola interaksi (rating) pengguna lain yang memiliki preferensi serupa. Dalam implementasi ini, digunakan model neural network dengan arsitektur `RecommenderNet`.

**Cara Kerja:**

**1. Matriks Interaksi:** Data rating diubah menjadi matriks yang menunjukkan hubungan antara pengguna dan buku, dengan nilai berupa rating yang diberikan.

**2. Pembelajaran Embedding:** Model `RecommenderNet` mempelajari vektor representasi tersembunyi (embedding) untuk setiap pengguna dan buku, berdasarkan pola rating.

**3. Prediksi Rating:** Model memperkirakan rating yang mungkin diberikan pengguna terhadap buku yang belum pernah dibaca.

**4. Rekomendasi:** Buku dengan prediksi rating tertinggi dan belum pernah diakses oleh pengguna akan direkomendasikan.

#### Implementasi Model `RecommenderNet`:
```
import tensorflow as tf

class RecommenderNet(tf.keras.Model):

  # Insialisasi fungsi
  def __init__(self, num_users, num_book, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_book = num_book
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.book_embedding = layers.Embedding( # layer embeddings book
        num_book,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.book_bias = layers.Embedding(num_book, 1) # layer embedding book bias

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    book_vector = self.book_embedding(inputs[:, 1]) # memanggil layer embedding 3
    book_bias = self.book_bias(inputs[:, 1]) # memanggil layer embedding 4

    dot_user_book = tf.tensordot(user_vector, book_vector, 2)

    x = dot_user_book + user_bias + book_bias

    return tf.nn.sigmoid(x) # activation sigmoid
```
Model `RecommenderNet` adalah kelas Keras Model yang terdiri dari:

**- User Embedding Layer**: Memetakan setiap indeks pengguna ke vektor embedding berdimensi `embedding_size`.
**- User Bias Layer**: Mempelajari bias spesifik untuk setiap pengguna.
**- Book Embedding Layer**: Memetakan setiap indeks buku ke vektor embedding berdimensi `embedding_size`.
**- Book Bias Layer**: Mempelajari bias spesifik untuk setiap buku.

Fungsi `call` dalam model melakukan dot product antara embedding pengguna dan buku, lalu menambahkan bias masing-masing. Hasilnya diproses dengan fungsi sigmoid untuk menghasilkan prediksi rating dalam rentang 0–1 (karena rating sudah dinormalisasi).

Model dikompilasi dengan loss function `BinaryCrossentropy` (meskipun prediksi rating bersifat regresi, fungsi ini umum dipakai untuk data implicit), menggunakan optimizer `Adam`, dan dievaluasi dengan metrik `RMSE`. Proses pelatihan dilakukan dengan data `x_train`, `y_train`, lalu diuji menggunakan `x_val`, `y_val`.

**Contoh Hasil Rekomendasi CF:**

Untuk seorang pengguna dengan ID 11676, sistem merekomendasikan 10 buku teratas yang belum pernah dibaca oleh pengguna tersebut, berdasarkan prediksi rating model:

```
Showing recommendations for users: 75355
===========================
Book with high ratings from user
--------------------------------
The Professor and the Madman: A Tale of Murder, Insanity, and the Making of The Oxford English Dictionary : Simon Winchester - Perennial
--------------------------------
Top 10 books recommendation
--------------------------------
One Fish Two Fish Red Fish Blue Fish (I Can Read It All by Myself Beginner Books) : DR SEUSS - Random House Books for Young Readers
The Baby Book: Everything You Need to Know About Your Baby from Birth to Age Two : Martha Sears - Little, Brown
Warchild : Karin Lowachee - Aspect
Life Is So Good : George Dawson - Penguin Books
The Teenage Liberation Handbook: How to Quit School and Get a Real Life and Education : Grace Llewellyn - Lowry House Pub
Shadowplay : Ron Cyr - LoonBooks
Dark Gold : Christine Feehan - Love Spell
Le Combat ordinaire, tome 1 : Larcenet - Dargaud
Keeping Watch : LAURIE R. KING - Bantam
Sluggy Freelance: Worship the Comic (Book 2) : Peter Abrams - Plan Nine Pub
```

**Kelebihan CF:**

**- Rekomendasi personal**: Disesuaikan dengan selera pengguna lain yang mirip.

**- Temukan item tak terduga**: Bisa menyarankan item yang tidak serupa secara konten tapi disukai pengguna serupa.

**- Tak perlu info detail item**: Tetap berfungsi meski fitur item terbatas.

**Kekurangan CF:**

**- Cold start**: Sulit merekomendasikan untuk pengguna atau item baru tanpa data interaksi.

**- Data jarang**: Jika sedikit interaksi, akurasi bisa menurun.

**- Kurang efisien di skala besar**: Butuh komputasi tinggi jika data pengguna dan item sangat banyak.

Dalam implementasi ini, CBF merekomendasikan buku yang mirip berdasarkan penulis dan penerbit, sedangkan CF merekomendasikan buku berdasarkan pola rating pengguna dengan selera serupa. Menggabungkan keduanya dalam sistem rekomendasi hybrid biasanya menghasilkan rekomendasi yang lebih baik, karena dapat memanfaatkan kelebihan dan mengurangi kekurangan masing-masing metode.

## Evaluation 

Bagian ini akan membahas metrik evaluasi yang digunakan untuk mengukur kinerja kedua pendekatan sistem rekomendasi, Content-Based Filtering (CBF) dan Collaborative Filtering (CF), serta menganalisis hasil proyek berdasarkan metrik tersebut.

### Evaluasi Content-Based Filtering

Untuk mengevaluasi kinerja sistem rekomendasi berbasis konten, digunakan metrik Precision, Recall, dan F1-Score. Metrik ini umum digunakan dalam tugas klasifikasi dan relevan untuk mengevaluasi apakah buku-buku yang direkomendasikan memang mirip dengan buku yang menjadi dasar rekomendasi.

* **Precision:**Precision mengukur seberapa banyak dari buku yang direkomendasikan benar-benar relevan dengan preferensi pengguna. Dengan kata lain, dari seluruh rekomendasi yang diberikan, berapa proporsi yang sesuai dengan ground truth kemiripan. Nilai Precision yang tinggi menunjukkan sistem mampu memberikan rekomendasi yang akurat dan tepat sasaran. Secara matematis, Precision didefinisikan sebagai:

  $$\text{Precision} = \frac{\text{Jumlah Buku Relevan yang Direkomendasikan}}{\text{Total Jumlah Buku yang Direkomendasikan}}$$

  Precision yang tinggi menunjukkan bahwa ketika sistem merekomendasikan sebuah buku, kemungkinan besar buku tersebut memang relevan.

* **Recall:** Recall mengukur seberapa banyak buku relevan yang berhasil direkomendasikan oleh sistem dari seluruh buku yang memang relevan dengan preferensi pengguna. Dengan kata lain, dari semua buku yang sesuai ground truth, berapa banyak yang berhasil ditemukan oleh sistem. Nilai Recall yang tinggi menunjukkan sistem efektif dalam menangkap sebagian besar buku yang relevan. Secara matematis, Recall didefinisikan sebagai:

  $$\text{Recall} = \frac{\text{Jumlah Buku Relevan yang Direkomendasikan}}{\text{Total Jumlah Buku yang Sebenarnya Relevan}}$$

  Recall yang tinggi menunjukkan bahwa sistem mampu mengidentifikasi sebagian besar buku yang relevan.

* **F1-Score:** adalah rata-rata harmonik dari Precision dan Recall, yang memberikan gambaran seimbang tentang performa model. Metrik ini berguna untuk menilai kemampuan sistem dalam memberikan rekomendasi yang akurat sekaligus lengkap. F1-Score membantu menghindari sistem yang hanya fokus pada salah satu sisi, seperti merekomendasikan sedikit buku tapi semua relevan, atau merekomendasikan banyak buku tapi sedikit yang relevan. Secara matematis, F1-Score didefinisikan sebagai:

  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Dalam implementasi, kemiripan *cosine* antara buku digunakan sebagai dasar untuk menentukan relevansi. Sebuah *threshold* kemiripan ditetapkan (dalam kasus ini 0.6). Jika kemiripan antara dua buku melebihi *threshold*, buku tersebut dianggap relevan. *Ground truth* dibuat berdasarkan *threshold* ini, dan prediksi biner dibuat berdasarkan apakah skor kemiripan melebihi *threshold*.

**Hasil Evaluasi CBF:**

![image](https://github.com/user-attachments/assets/33c29fed-3d9d-4bea-b674-04b7a29b0460)

Hasil evaluasi menunjukkan bahwa dengan threshold 0,6, model Content-Based Filtering mencapai skor Precision, Recall, dan F1-Score sempurna (1,0). Artinya, semua buku yang dianggap mirip oleh model memang relevan sesuai ground truth, dan model berhasil menemukan semua pasangan buku relevan pada sampel uji. Namun, hasil ini hanya berlaku pada data sampel dan sangat bergantung pada threshold yang dipakai, sehingga mungkin tidak sama untuk seluruh dataset atau threshold lain.

### Evaluasi Collaborative Filtering

Untuk menilai kinerja model Collaborative Filtering, digunakan metrik Root Mean Squared Error (RMSE). RMSE mengukur seberapa jauh prediksi rating model berbeda dari rating asli yang diberikan pengguna. Nilai RMSE yang lebih kecil menunjukkan prediksi model yang lebih akurat.

**- Root Mean Squared Error (RMSE):** Dihitung sebagai akar kuadrat dari rata-rata kuadrat perbedaan antara nilai prediksi ($\hat{y}_i$) dan nilai sebenarnya ($y_i$). Secara matematis, RMSE didefinisikan sebagai:

![0_hH_3TLAdSJnD9nuK](https://github.com/user-attachments/assets/d465489c-f386-432a-b42c-428de85d8f09)

Di mana n adalah jumlah total prediksi, RMSE bekerja dengan menghitung rata-rata besarnya kesalahan prediksi rating. Untuk setiap interaksi pengguna dan buku di data validasi, model memprediksi rating, lalu dihitung selisih antara prediksi dan rating asli. Selisih ini dikuadratkan agar nilai negatif hilang dan kesalahan besar lebih diperhatikan. Setelah itu, diambil rata-rata dari semua selisih kuadrat, lalu akar kuadrat dari rata-rata tersebut diambil. Hasil RMSE menunjukkan seberapa dekat prediksi model dengan rating asli, di mana nilai yang lebih rendah berarti prediksi lebih akurat.

**Hasil Evaluasi CF:**

Berdasarkan grafik metrik model yang ditampilkan, terlihat bahwa nilai RMSE pada data pelatihan terus menurun seiring dengan bertambahnya epoch. Nilai RMSE pada data validasi juga menurun pada awalnya, namun kemudian cenderung stabil atau bahkan sedikit meningkat setelah epoch tertentu.

Nilai RMSE terakhir pada epoch ke-20 adalah:

**- RMSE Training**: Sekitar 0.2728
**- RMSE Validasi**: Sekitar 0.3605

Berikut visualisasi model metrik.

![image](https://github.com/user-attachments/assets/a48dcebc-bd9b-45e1-aec6-eda3677c9eba)

Perbedaan antara RMSE pada data pelatihan dan validasi menunjukkan sedikit overfitting, di mana model terlalu cocok dengan data pelatihan sehingga performanya menurun sedikit pada data baru. Namun, RMSE validasi sebesar 0,3605 menandakan model masih cukup baik dalam memprediksi rating pengguna. Karena rating sudah dinormalisasi antara 0 dan 1, nilai RMSE ini berarti rata-rata kesalahan prediksi sekitar 0,36 pada skala tersebut.

### Evaluasi Tujuan Proyek Berdasarkan Problem Statements

Berdasarkan hasil evaluasi:

**- Bagaimana cara membantu pengguna menemukan buku yang sesuai dengan preferensi mereka?**
CBF merekomendasikan buku berdasarkan kemiripan konten (penulis, penerbit), sedangkan CF menggunakan pola rating pengguna lain untuk memperluas pilihan berdasarkan preferensi bersama.

**- Bagaimana cara membangun sistem rekomendasi yang lebih personal dan relevan?**
CF mempelajari embedding dari data interaksi untuk menghasilkan rekomendasi yang disesuaikan, sementara CBF tetap memberikan rekomendasi relevan berdasarkan konten yang diminati pengguna.

**- Bagaimana cara mengatasi cold-start problem?**
CBF lebih unggul karena hanya butuh atribut konten, sedangkan CF kesulitan tanpa data interaksi pengguna dan buku baru.

**Kesimpulan:**
Secara keseluruhan, kedua pendekatan berhasil diimplementasikan dan dievaluasi. Content-Based Filtering efektif dalam merekomendasikan buku yang kontennya mirip, sementara Collaborative Filtering mampu memberikan rekomendasi yang dipersonalisasi berdasarkan pola interaksi pengguna. Pemilihan pendekatan yang paling sesuai atau kombinasi keduanya dapat bergantung pada kebutuhan spesifik dan karakteristik dataset yang lebih luas.

