# Lab MLOps Thực Hành: Từ Thực Nghiệm Cục Bộ Đến Triển Khai Liên Tục

Course: AIInAction - VinUni
Buổi: Day 21 - CI/CD cho AI Systems
Khoá: K3

---

## Mục Tiêu Học Tập

Sau khi hoàn thành lab này, bạn có khả năng:

1. Thiết lập quá trình theo dõi thí nghiệm máy học bằng MLflow trên máy tính cá nhân.
2. Quản lý và phiên bản hóa dữ liệu bằng DVC với cloud object storage (GCP / AWS / Azure) làm remote.
3. Xây dựng pipeline CI/CD hoàn chỉnh trên GitHub Actions với ba giai đoạn: kiểm thử, huấn luyện, triển khai.
4. Triển khai mô hình lên máy chủ ảo trên cloud (GCE / EC2 / Azure VM) dưới dạng REST API bằng FastAPI.
5. Mô phỏng quy trình huấn luyện liên tục: bổ sung dữ liệu mới và kích hoạt pipeline hoàn toàn tự động.

---

## Tổng Quan Kiến Trúc

Toàn bộ lab được triển khai theo ba bước liên tiếp, mỗi bước xây dựng trên kết quả của bước trước:

```
[Máy tính cá nhân]
      |
      |  git push
      v
[GitHub repository]
      |
      |  GitHub Actions kích hoạt tự động
      v
[Runner: Unit Test -> Train -> Eval (>= 0.70) -> Deploy]
      |                                    |
      |  dvc pull                          |  dvc push (model)
      v                                    v
[Cloud Object Storage]               [Cloud VM]
  data/                                mlops-serve (FastAPI)
  models/latest/                         POST /predict
```

Bước 1 chỉ chạy trên máy tính cá nhân. Bước 2 và Bước 3 sử dụng toàn bộ kiến trúc trên.

---

## Yêu Cầu Trước Khi Bắt Đầu

Phần mềm cần cài đặt trên máy tính cá nhân:

- Python 3.10 trở lên
- Git và tài khoản GitHub (tạo một repo public mới, chưa có nội dung)
- Tài khoản cloud (chọn một trong ba: GCP, AWS, hoặc Azure — gói miễn phí/trial đủ dùng cho lab này)
- CLI của cloud provider đã chọn (xem hướng dẫn cài đặt chi tiết tại tasks/buoc-2.md)

Kiểm tra cài đặt:

```bash
python --version     # Python 3.10.x trở lên
git --version
# Kiểm tra CLI của cloud provider đã chọn (một trong ba):
gcloud --version     # GCP
aws --version        # AWS
az --version         # Azure
```

---

## Tập Dữ Liệu

Tập dữ liệu **Wine Quality** (UCI Machine Learning Repository) chứa thông tin hóa học của 6497 mẫu rượu vang đỏ và trắng Bồ Đào Nha. Nhiệm vụ là phân loại chất lượng rượu vang dựa trên các đặc trưng hóa học.

Nguồn: https://archive.ics.uci.edu/dataset/186/wine+quality

Đặc trưng đầu vào (12 cột):

| Tên cột            | Mô tả                             |
| -------------------- | ----------------------------------- |
| fixed_acidity        | Độ axit cố định                |
| volatile_acidity     | Độ axit bay hơi                  |
| citric_acid          | Axit citric                         |
| residual_sugar       | Lượng đường còn lại          |
| chlorides            | Nồng độ clorua                   |
| free_sulfur_dioxide  | SO2 tự do                          |
| total_sulfur_dioxide | Tổng SO2                           |
| density              | Mật độ                           |
| pH                   | Độ pH                             |
| sulphates            | Sunphat                             |
| alcohol              | Nồng độ cồn                     |
| wine_type            | Loại rượu (0 = đỏ, 1 = trắng) |

Nhãn dự đoán (cột `target`):

| Giá trị | Ý nghĩa                 | Điểm chất lượng gốc |
| --------- | ------------------------- | ------------------------- |
| 0         | Chất lượng thấp       | 3 - 5                     |
| 1         | Chất lượng trung bình | 6                         |
| 2         | Chất lượng cao         | 7 - 9                     |

Phân chia dữ liệu:

| File                  | Số mẫu | Mục đích                                                                   |
| --------------------- | -------- | ----------------------------------------------------------------------------- |
| data/train_phase1.csv | 2998     | Huấn luyện ở Bước 1 và 2                                                |
| data/eval.csv         | 500      | Đánh giá mô hình (held-out set, không bao giờ dùng để huấn luyện) |
| data/train_phase2.csv | 2998     | Dữ liệu mới bổ sung ở Bước 3                                           |

Chạy script sau một lần duy nhất để tải và chia dữ liệu:

```bash
python generate_data.py
```

Kết quả mong đợi:

```
train_phase1.csv : 2998 mẫu
eval.csv         :  500 mẫu
train_phase2.csv : 2998 mẫu
```

---

## Cấu Trúc Thư Mục

Cấu trúc này là kết quả cuối cùng sau khi hoàn thành cả ba bước:

```
mlops-lab/
├── .github/
│   └── workflows/
│       └── mlops.yml          <- Pipeline CI/CD (Bước 2)
├── .dvc/
│   └── config                 <- Cấu hình DVC remote (Bước 2)
├── data/
│   ├── train_phase1.csv.dvc   <- Con trỏ DVC (Bước 2)
│   ├── eval.csv.dvc
│   └── train_phase2.csv.dvc
├── src/
│   ├── __init__.py
│   ├── train.py               <- Script huấn luyện (Bước 1)
│   └── serve.py               <- API suy luận (Bước 2)
├── tests/
│   ├── __init__.py
│   └── test_train.py          <- Unit test (Bước 2)
├── generate_data.py           <- Script tạo dữ liệu (đã cung cấp)
├── add_new_data.py            <- Script thêm dữ liệu mới (đã cung cấp)
├── params.yaml                <- Siêu tham số mô hình
├── requirements.txt           <- Thư viện Python
└── .gitignore
```

---

## Cài Đặt Môi Trường

### Bước chuẩn bị (thực hiện một lần)

```bash
# 1. Clone hoặc khởi tạo repo của bạn
git clone <URL_REPO_CUA_BAN>
cd mlops-lab

# 2. Tạo và kích hoạt môi trường ảo
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# 3. Cài đặt thư viện
pip install -r requirements.txt

# 4. Tải dữ liệu
python generate_data.py
```

### `.gitignore`

```
mlflow.db
mlartifacts/
models/
outputs/
data/train_phase1.csv
data/eval.csv
data/train_phase2.csv
sa-key.json
.env
.venv/
__pycache__/
```

### `requirements.txt`

```
mlflow==2.13.0
scikit-learn==1.4.2
pandas==2.2.2
# DVC extra theo provider: [gs]=GCP, [s3]=AWS, [azure]=Azure
dvc[gs]==3.50.1
pathspec==0.11.2
pytest==8.2.0
fastapi==0.111.0
uvicorn==0.29.0
joblib==1.4.2
# Cloud SDK theo provider: google-cloud-storage (GCP), boto3 (AWS), azure-storage-blob (Azure)
google-cloud-storage==2.16.0
pyyaml==6.0.1
```

---

## Hướng Dẫn Lab

| Bước | Nội dung                                             | File hướng dẫn |
| ------ | ----------------------------------------------------- | ----------------- |
| 1      | Thực nghiệm cục bộ và theo dõi bằng MLflow     | tasks/buoc-1.md   |
| 2      | Pipeline CI/CD tự động với GitHub Actions và DVC | tasks/buoc-2.md   |
| 3      | Huấn luyện liên tục khi có dữ liệu mới        | tasks/buoc-3.md   |

Bắt đầu từ [Bước 1](tasks/buoc-1.md).

---

## Rubric Chấm Điểm

### Tiêu chí chính (80 điểm)

| Hạng mục                 | Tiêu chí đánh giá                                                                       | Điểm tối đa |
| -------------------------- | -------------------------------------------------------------------------------------------- | --------------- |
| Bước 1 - MLflow tracking | MLflow UI hiển thị ít nhất 3 lần chạy với các siêu tham số khác nhau              | 12              |
| Bước 1 - Độ đo        | Mỗi lần chạy ghi nhận đủ cả`accuracy` và `f1_score`                              | 8               |
| Bước 1 - Phân tích     | Xác định và giải thích bộ siêu tham số tốt nhất                                   | 4               |
| Bước 2 - DVC             | Remote đã cấu hình,`dvc push` thành công, dữ liệu hiển thị trên cloud storage   | 12              |
| Bước 2 - CI/CD           | Cả bốn GitHub Actions jobs (Test, Train, Eval, Deploy) đều qua (màu xanh)               | 16              |
| Bước 2 - Eval gate       | Deploy job tự động bị chặn khi accuracy dưới ngưỡng 0.70                            | 4               |
| Bước 2 - Serving         | VM trả về kết quả đúng tại endpoint POST /predict                                     | 12              |
| Bước 3 - Tự động hóa | Một commit dữ liệu mới kích hoạt toàn bộ pipeline không cần tác động thủ công | 12              |
| Tổng                      |                                                                                              | 80              |

### Thang điểm chi tiết

| Khoảng điểm | Nhận xét                                                                                                 |
| -------------- | ---------------------------------------------------------------------------------------------------------- |
| 90 - 100       | Xuất sắc. Toàn bộ pipeline hoạt động chính xác, đầy đủ bằng chứng và có điểm bonus.     |
| 72 - 89        | Tốt. Hoàn thành toàn bộ tiêu chí chính, có thể còn thiếu một phần bằng chứng.              |
| 56 - 71        | Đạt yêu cầu tối thiểu. Hoàn thành được các bước chính nhưng còn lỗi hoặc thiếu bước. |
| Dưới 56      | Chưa đạt. Nhiều phần chưa được thực hiện hoặc không hoạt động.                             |

### Hướng dẫn nộp bài

Nộp các hạng mục sau:

1. URL repo GitHub công khai chứa toàn bộ code và cấu hình.
2. Chuỗi chụp màn hình theo thứ tự:
   - MLflow UI hiển thị ít nhất 3 thí nghiệm.
   - GitHub Actions tab hiển thị cả bốn jobs màu xanh (Bước 2 và Bước 3).
   - Kết quả của lệnh `curl http://VM_IP:8000/health` và `curl http://VM_IP:8000/predict`.
   - Cloud Storage Console hiển thị các file dữ liệu và model đã được push lên.
3. File báo cáo ngắn (không quá 1 trang A4) ghi lại:
   - Bộ siêu tham số đã chọn và lý do (kết quả Bước 1).
   - Bất kỳ khó khăn nào gặp phải và cách giải quyết.

---

## Thách Thức Nâng Cao (Bonus)

Các thách thức dưới đây không bắt buộc. Hoàn thành đủ cả 5 thách thức sẽ được cộng tối đa 20 điểm, nâng tổng điểm lên 100.

### Bonus 1: Tracking MLflow Từ Xa Với DagsHub (4 điểm)

Thay vì lưu MLflow vào file cục bộ (`sqlite:///mlflow.db`), kết nối đến server MLflow miễn phí trên DagsHub:

- Tạo tài khoản tại https://dagshub.com và kết nối repo GitHub của bạn.
- Thêm các biến môi trường MLflow vào GitHub Secrets.
- Cập nhật `mlops.yml` để sử dụng tracking server của DagsHub thay vì file cục bộ.

Kết quả: Mỗi lần chạy trong GitHub Actions sẽ được ghi lên DagsHub, có thể xem từ bất cứ đâu.

### Bonus 2: Thí Nghiệm Với Nhiều Thuật Toán (4 điểm)

Mở rộng `src/train.py` để hỗ trợ nhiều thuật toán ngoài RandomForest:

- Thêm tham số `model_type` vào `params.yaml` (ví dụ: `random_forest`, `gradient_boosting`, `logistic_regression`).
- Viết logic chọn thuật toán tương ứng với giá trị của tham số đó.
- Chạy thí nghiệm với ít nhất 2 thuật toán khác nhau và so sánh trên MLflow UI.

### Bonus 3: Báo Cáo Hiệu Suất Tự Động (4 điểm)

Thêm một bước trong `mlops.yml` để tự động tạo báo cáo hiệu suất sau mỗi lần huấn luyện:

- Tính toán confusion matrix và in ra ở dạng văn bản (không cần ảnh).
- Tính thêm `precision` và `recall` cho từng lớp (0, 1, 2) và ghi vào `outputs/report.txt`.
- Dùng `actions/upload-artifact` để lưu file này cùng với `metrics.json`.

### Bonus 4: Hoàn Trả Về Phiên Bản Trước (4 điểm)

Xây dựng cơ chế an toàn: nếu model mới có accuracy thấp hơn model hiện tại đang chạy, pipeline tự động hủy deploy:

- Trước khi deploy, tải `outputs/metrics.json` của lần chạy trước từ cloud storage (nếu có).
- So sánh accuracy mới với accuracy cũ.
- Chỉ deploy khi accuracy mới cao hơn hoặc bằng accuracy cũ.
- Ghi lại kết quả so sánh vào log của pipeline.

### Bonus 5: Cảnh Báo Lệch Lạc Dữ Liệu (4 điểm)

Thêm bước kiểm tra phân phối dữ liệu trước khi huấn luyện:

- Tính phân phối nhãn (tỷ lệ mẫu của từng lớp 0, 1, 2) trong tập huấn luyện.
- Nếu bất kỳ lớp nào chiếm ít hơn 10% tổng mẫu, in cảnh báo rõ ràng vào log.
- Ghi tỷ lệ phân phối nhãn vào `outputs/metrics.json` bên cạnh accuracy và f1_score.

---

## Xử Lý Sự Cố Thường Gặp

Xem phần xử lý sự cố chi tiết trong từng file hướng dẫn:

- Lỗi DVC authentication: tasks/buoc-2.md
- Lỗi GitHub Actions dvc pull: tasks/buoc-2.md
- Pipeline Bước 3 không được kích hoạt: tasks/buoc-3.md
- Service trên VM không khởi động: tasks/buoc-2.md

---

Bắt đầu: [Bước 1 - Thực nghiệm cục bộ](tasks/buoc-1.md)
