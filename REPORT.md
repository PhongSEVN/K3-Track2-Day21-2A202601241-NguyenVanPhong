# Báo cáo - Lab MLOps (Wine Quality)

## Bước 1 - Siêu tham số đã chọn

`params.yaml`: `model_type: random_forest`, `n_estimators=300, max_depth=null
(không giới hạn), min_samples_split=2`.

Chạy 17 run trên MLflow với các tổ hợp `n_estimators` (100-1000), `max_depth`
(3-40, None), `min_samples_split` (2-10). Bộ trên cho accuracy cao nhất
(**0.682**) trên tập `eval.csv` (2998 mẫu train, Bước 1-2). Tăng
`max_depth`/`n_estimators` quá cao gây overfit nhẹ (accuracy giảm), giảm
`min_samples_split` xuống 2 cho kết quả ổn định nhất qua nhiều lần chạy.

Đã thử thêm 2 hướng để cố vượt 0.70 ngay ở Bước 1-2 (chưa có thêm dữ liệu):

1. **Feature engineering** (6 đặc trưng mới: tỷ lệ SO2 tự do/tổng, tổng độ
   axit, tương tác alcohol×sulphates, alcohol/density, đường/alcohol) →
   accuracy **giảm** còn 0.674 (RandomForest tự học tương tác phi tuyến qua
   splits, thêm cột chỉ pha loãng tín hiệu).
2. **RandomizedSearchCV** quét 60 tổ hợp (`bootstrap`, `criterion`,
   `max_features`, `min_samples_leaf`, cross-val 3-fold) → tốt nhất chỉ
   **0.662**.

Kết luận: với đúng 2998 mẫu train, 0.682 là trần thực tế của
`RandomForestClassifier`, không phải do chọn tham số dở.

## Bước 2 - Eval gate chặn đúng chức năng

`EVAL_THRESHOLD = 0.70` trong `src/train.py` chặn `Deploy` khi accuracy chưa
đạt: job `Eval` fail có kiểm soát (`FAILED: accuracy 0.6820 < 0.70`), job
`Deploy` bị skip (verify qua run
[32446657400](https://github.com/PhongSEVN/K3-Track2-Day21-2A202601241-NguyenVanPhong/actions/runs/32446657400)):
`Unit Test` ✓, `Train` ✓ (log MLflow, DVC pull/push, upload GCS), `Eval`
chặn đúng lúc.

## Bước 3 - Thêm dữ liệu mới, vượt ngưỡng, deploy thành công

`python add_new_data.py` gộp `train_phase2.csv` (2998 mẫu) vào
`train_phase1.csv` → **5996 mẫu train**. Huấn luyện lại với cùng bộ tham số
Bước 1: accuracy tăng từ **0.682 lên 0.746**. Dữ liệu nhiều hơn giúp model
tổng quát hoá tốt hơn, đúng như kỳ vọng của continuous training.

| Chỉ số   | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) |
| -------- | ------------------ | ------------------ |
| accuracy | 0.6820              | 0.7460              |
| f1_score | 0.6811              | ~0.745               |

Với accuracy 0.746 ≥ 0.70, `Eval` pass, `Deploy` chạy thành công lần đầu.
Toàn bộ 4 job xanh, verify tại run
[32448309157](https://github.com/PhongSEVN/K3-Track2-Day21-2A202601241-NguyenVanPhong/actions/runs/32448309157).
Xác nhận VM đang serve model mới:

```
curl http://VM_IP:8000/health
{"status":"ok"}

curl -X POST http://VM_IP:8000/predict -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
{"prediction":0,"label":"thap"}
```

## Bonus đã hoàn thành (code trong `src/train.py` + `.github/workflows/mlops.yml`)

- **Bonus 2 (đa thuật toán)**: `model_type` trong `params.yaml` chọn
  `random_forest` / `gradient_boosting` / `logistic_regression` qua
  `MODEL_REGISTRY`. Test `test_train_with_gradient_boosting` xác nhận cả 2
  thuật toán chạy được.
- **Bonus 3 (báo cáo tự động)**: `write_report()` tính confusion matrix +
  precision/recall từng lớp, ghi `outputs/report.txt`, upload cùng
  `metrics.json` qua `actions/upload-artifact`.
- **Bonus 4 (rollback)**: `fetch_previous_accuracy()` đọc
  `models/latest/metrics.json` trên GCS, `should_deploy()` so sánh với
  accuracy mới. Job `Train` chỉ ghi đè `models/latest/` khi `deploy_ok=true`;
  job `Eval` chặn thêm nếu `deploy_ok=false`.
- **Bonus 5 (cảnh báo lệch dữ liệu)**: `check_drift()` cảnh báo lớp nào
  chiếm dưới 10% tổng mẫu train, ghi `label_distribution` +
  `drift_warnings` vào `outputs/metrics.json`.
- **Bonus 1 (DagsHub)**: kết nối repo với DagsHub
  (`https://dagshub.com/PhongSEVN/K3-Track2-Day21-2A202601241-NguyenVanPhong`),
  workflow tự bật bước "Configure remote MLflow tracking" khi 3 secret
  `MLFLOW_TRACKING_URI/USERNAME/PASSWORD` được cấu hình. Verify thật qua run
  [32450239074](https://github.com/PhongSEVN/K3-Track2-Day21-2A202601241-NguyenVanPhong/actions/runs/32450239074) —
  cả 4 job xanh, run xuất hiện trên DagsHub Experiments UI.

**Tất cả 5 bonus đã hoàn thành và verify thật trên CI/CD** (không chỉ code
tĩnh) — 4/4 job xanh hoàn toàn ở run trên.

## Khó khăn gặp phải và cách giải quyết

- **`.dvc/config` chứa `credentialpath` tương đối** (`../sa-key.json`) chỉ
  đúng trên máy cá nhân, hỏng trên GitHub Actions runner (401 Invalid
  Credentials khi `dvc pull`). Sửa: bỏ `credentialpath`, dùng
  `GOOGLE_APPLICATION_CREDENTIALS` (env var CI đã set) cho cả local lẫn CI.
- **Repo là fork** của template giảng viên
  (`VinUni-AI20k/K3-Track2-Day21-CI-CD-for-AI-Systems`) — GitHub mặc định
  **tắt Actions-on-push cho repo fork**, dù `workflow_dispatch` (chạy tay)
  vẫn hoạt động bình thường và API `actions/permissions` báo `enabled:true`
  (gây nhầm lẫn). Đây là banner UI-only trong tab Actions
  ("I understand my workflows, go ahead and enable them"), không có API/CLI
  nào bật thay được. Sau khi bấm 1 lần, push tự trigger bình thường — đúng
  yêu cầu "không cần thao tác thủ công" cho các lần sau.
- **VM thiếu `src/serve.py` và `sa-key.json`** dù bước cấu hình VM báo
  thành công — 2 lệnh `scp` với đường dẫn `~/...` không chạy đúng trên
  Windows (`pscp` không hiểu `~` ở remote path), lỗi bị nuốt mất. Sửa: dùng
  đường dẫn tuyệt đối (`/home/<user>/...`) khi `gcloud compute scp`.
- **Google Cloud SDK Shell mặc định là cmd.exe**, không phải PowerShell —
  cú pháp `$VAR = "..."` không chạy. Chuyển sang PowerShell thường (gcloud
  đã có sẵn trong PATH hệ thống).
- **DagsHub MLflow server trả 404** khi tạo run trên repo hoàn toàn trống
  (chưa có experiment nào). `mlflow.start_run()` không tự tạo experiment
  "Default" như file-store cục bộ. Sửa: gọi `mlflow.set_experiment(...)`
  tường minh trong `train.py` trước `start_run()`.
- **Health check deploy fail dù server chạy tốt**: `sleep 5` quá ngắn — model
  RandomForest 300 cây tải từ GCS + unpickle mất ~8s. Sửa: retry loop 6 lần
  x5s thay vì sleep cố định 1 lần.
- **Project ID vs Project Name**: nhầm tên hiển thị project
  (`track2-day16-...`) với Project ID thật (`mineral-aegis-505503-i2`) khi
  `gcloud config set project`.
