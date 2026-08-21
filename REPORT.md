# Báo cáo - Lab MLOps (Wine Quality)

## Bước 1 - Siêu tham số đã chọn

`params.yaml`: `n_estimators=300, max_depth=null (không giới hạn), min_samples_split=2`.

Chạy 17 run trên MLflow với các tổ hợp `n_estimators` (100-1000), `max_depth`
(3-40, None), `min_samples_split` (2-10). Bộ trên cho accuracy cao nhất
(**0.682**) trên tập `eval.csv`. Tăng `max_depth`/`n_estimators` quá cao gây
overfit nhẹ (accuracy giảm), giảm `min_samples_split` xuống 2 cho kết quả
ổn định nhất qua nhiều lần chạy.

## Bước 2 - Vấn đề: accuracy không vượt ngưỡng 0.70

Sau bộ tham số tốt nhất ở Bước 1, đã thử thêm hai hướng để cố vượt 0.70:

1. **Feature engineering** (6 đặc trưng mới: tỷ lệ SO2 tự do/tổng, tổng độ
   axit, tương tác alcohol×sulphates, alcohol/density, đường/alcohol) →
   accuracy **giảm** còn 0.674. RandomForest tự học các tương tác phi tuyến
   qua splits, thêm cột chỉ pha loãng tín hiệu ở mỗi lần chọn feature ngẫu
   nhiên.
2. **RandomizedSearchCV** quét 60 tổ hợp trên không gian tham số rộng hơn
   (`bootstrap`, `criterion`, `max_features`, `min_samples_leaf`, cross-val
   3-fold) → tốt nhất chỉ **0.662**, vẫn kém hơn baseline.

**Kết luận:** 0.682 là trần thực tế của `RandomForestClassifier` trên đúng
tập `train_phase1.csv` (2998 mẫu) / `eval.csv` (500 mẫu) này, không phải do
chọn tham số dở. Giữ nguyên kết quả thật thay vì chỉnh sửa để "qua ngưỡng"
giả tạo.

**Hệ quả:** `EVAL_THRESHOLD = 0.70` trong `src/train.py` chặn đúng chức
năng — job `Eval` fail có kiểm soát (`FAILED: accuracy 0.6820 < 0.70`), job
`Deploy` bị skip. Đã verify toàn bộ pipeline tới đúng điểm này qua GitHub
Actions (run [32446657400](https://github.com/PhongSEVN/K3-Track2-Day21-2A202601241-NguyenVanPhong/actions/runs/32446657400)):
`Unit Test` ✓, `Train` ✓ (log MLflow, pull/push DVC, upload GCS đều thành
công), `Eval` chặn đúng lúc, `Deploy` chưa chạy vì phụ thuộc `Eval`.

## Khó khăn khác gặp phải

- **`.dvc/config` chứa `credentialpath` tương đối** (`../sa-key.json`) chỉ
  đúng trên máy cá nhân, hỏng trên GitHub Actions runner (401 Invalid
  Credentials khi `dvc pull`). Sửa: bỏ `credentialpath`, dùng
  `GOOGLE_APPLICATION_CREDENTIALS` (env var CI đã set) cho cả local lẫn CI.
- **Google Cloud SDK Shell mặc định là cmd.exe**, không phải PowerShell —
  cú pháp `$VAR = "..."` không chạy. Chuyển sang PowerShell thường (gcloud
  đã có sẵn trong PATH hệ thống).
- **Project ID vs Project Name**: nhầm tên hiển thị project
  (`track2-day16-...`) với Project ID thật (`mineral-aegis-505503-i2`) khi
  `gcloud config set project`.
