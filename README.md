## Hướng dẫn chạy phần mềm

### Bước 1: Tạo môi trường ảo (Virtual Environment)

```bash
python -m venv venv
```

### Bước 2: Kích hoạt môi trường ảo

* **Trên Windows:**

```bash
venv\Scripts\activate
```

* **Trên macOS / Linux:**

```bash
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện cần thiết

```bash
pip install -r requirements.txt
```

### Bước 4: Chạy backend

```bash
python src/main.py
```

### Bước 5: Chạy frontend

```bash
cd frontend
npm install
npm run dev
```
