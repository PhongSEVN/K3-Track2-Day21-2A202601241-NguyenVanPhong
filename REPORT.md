# Báo cáo - Lab MLOps (Wine Quality)

## Bộ siêu tham số đã chọn và lý do

`params.yaml`: `model_type: random_forest`, `n_estimators=300, max_depth=null
(không giới hạn), min_samples_split=2`.

Ở Bước 1, em chạy 21 run trên MLflow, đổi qua lại `n_estimators` (100-1000),
`max_depth` (3-40 hoặc None), `min_samples_split` (2-10) để tìm bộ tham số
tốt nhất. Bộ trên cho accuracy cao nhất trên `eval.csv`, đạt 0.682 với 2998
mẫu train. Tăng `max_depth` hoặc `n_estimators` lên quá cao thì bắt đầu
overfit nhẹ, accuracy giảm; `min_samples_split=2` cho kết quả ổn định nhất
qua các lần chạy lại. (MLflow UI hiện đang hiển thị 33 run vì các lần train
sau này - làm Bước 3, thử `gradient_boosting` cho Bonus 2 - cũng được ghi
chung vào cùng chỗ.)

Vì 0.682 chưa đạt ngưỡng 0.70, em thử thêm hai hướng trước khi chuyển sang
Bước 2:

1. Feature engineering - thêm 6 đặc trưng mới (tỷ lệ SO2 tự do/tổng, tổng
   độ axit, alcohol×sulphates, alcohol/density, đường/alcohol). Kết quả tệ
   hơn, accuracy tụt xuống 0.674. Đoán là RandomForest đã tự học được các
   tương tác phi tuyến qua splits rồi, thêm cột chỉ làm loãng tín hiệu mỗi
   lần model chọn feature ngẫu nhiên.
2. RandomizedSearchCV quét 60 tổ hợp tham số rộng hơn hẳn (bootstrap,
   criterion, max_features, min_samples_leaf, cross-validation 3 fold).
   Tốt nhất chỉ được 0.662.

Vậy nên em kết luận 0.682 là trần thực tế của RandomForest với đúng 2998
mẫu train này, không phải do chọn tham số dở, và giữ nguyên kết quả thay vì
cố ép cho qua ngưỡng bằng cách nào đó không thật.

## Eval gate hoạt động đúng ở Bước 2

`EVAL_THRESHOLD = 0.70` trong `src/train.py` chặn job `Deploy` khi accuracy
chưa đạt: `Eval` fail có kiểm soát với thông báo `FAILED: accuracy 0.6820 <
0.70`, và `Deploy` bị skip theo sau. Run
[32446657400](https://github.com/PhongSEVN/K3-Track2-Day21-2A202601241-NguyenVanPhong/actions/runs/32446657400)
verify đúng luồng này: `Unit Test` và `Train` (log MLflow, DVC pull/push,
upload GCS) đều pass, `Eval` chặn đúng lúc cần chặn.

## Bước 3: thêm dữ liệu, vượt ngưỡng, deploy thành công

Chạy `python add_new_data.py` gộp `train_phase2.csv` (2998 mẫu) vào
`train_phase1.csv`, tổng cộng 5996 mẫu train. Train lại với y hệt bộ tham
số ở Bước 1, accuracy nhảy từ 0.682 lên 0.746 - dữ liệu nhiều hơn giúp
model tổng quát hoá tốt hơn hẳn, đúng tinh thần continuous training.

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) |
|---|---|---|
| accuracy | 0.6820 | 0.7460 |
| f1_score | 0.6811 | 0.7449 |

0.746 vượt 0.70 nên `Eval` pass, `Deploy` chạy thành công lần đầu tiên -
cả 4 job xanh, verify tại run
[32448309157](https://github.com/PhongSEVN/K3-Track2-Day21-2A202601241-NguyenVanPhong/actions/runs/32448309157).
Commit push này ("data: bổ sung 2998 mẫu dữ liệu mới") thực ra đúng dữ
liệu và đúng nội dung, chỉ là lúc em push thì Actions-on-push cho repo
fork chưa được bật thủ công (banner riêng trên tab Actions, chỉ bật qua
web UI được), nên run xanh đầu tiên lại bị một push khác kích hoạt trễ
hơn, tên run không hiện đúng chữ "data: ...". Sau khi bật xong, em chạy
lại `add_new_data.py` một lần nữa để có bằng chứng commit dữ liệu tự kích
hoạt pipeline đúng nghĩa đen - run
[32476981771](https://github.com/PhongSEVN/K3-Track2-Day21-2A202601241-NguyenVanPhong/actions/runs/32476981771),
tên hiện đúng "data: bo sung du lieu (lan 2, minh hoa lai co che trigger
tu dong)", `Triggered via push`, cả 4 job xanh, accuracy vẫn 0.746 (dữ
liệu thêm vào lần 2 trùng với lần 1 nên không có thông tin mới, đúng như
kỳ vọng - model không đổi kết quả vì học lại đúng pattern cũ).

Test luôn VM đang serve đúng model mới:

```
curl http://VM_IP:8000/health
{"status":"ok"}

curl -X POST http://VM_IP:8000/predict -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
{"prediction":0,"label":"thap"}
```

## Bonus đã làm

Cả 5 bonus đều nằm trong `src/train.py` và `.github/workflows/mlops.yml`,
và em verify thật trên CI/CD chứ không chỉ code chạy được cục bộ - run
[32450239074](https://github.com/PhongSEVN/K3-Track2-Day21-2A202601241-NguyenVanPhong/actions/runs/32450239074)
là bằng chứng cả 4 job xanh với đủ 5 bonus đang bật.

**Bonus 2 - đa thuật toán.** `model_type` trong `params.yaml` chọn giữa
`random_forest`, `gradient_boosting`, `logistic_regression` qua
`MODEL_REGISTRY`. Có test riêng (`test_train_with_gradient_boosting`) xác
nhận đổi thuật toán vẫn chạy được.

**Bonus 3 - báo cáo tự động.** `write_report()` tính confusion matrix cộng
precision/recall cho từng lớp, ghi ra `outputs/report.txt`, rồi upload
cùng `metrics.json` qua `actions/upload-artifact`.

**Bonus 4 - rollback khi model tệ đi.** `fetch_previous_accuracy()` đọc
`models/latest/metrics.json` đang có trên GCS, `should_deploy()` so với
accuracy mới train ra. Job `Train` chỉ ghi đè `models/latest/` khi
`deploy_ok=true`, và job `Eval` chặn thêm một lần nữa nếu `deploy_ok=false`
- model mới kém hơn thì không có cách nào lọt qua được cả hai lớp kiểm tra.

**Bonus 5 - cảnh báo lệch dữ liệu.** `check_drift()` rà phân phối nhãn
trong tập train, cảnh báo nếu lớp nào tụt dưới 10% tổng mẫu, ghi cả
`label_distribution` và `drift_warnings` vào `metrics.json`.

**Bonus 1 - tracking từ xa qua DagsHub.** Kết nối repo với
`https://dagshub.com/PhongSEVN/K3-Track2-Day21-2A202601241-NguyenVanPhong`,
workflow tự bật bước "Configure remote MLflow tracking" khi 3 secret
`MLFLOW_TRACKING_URI/USERNAME/PASSWORD` đã được cấu hình. Run log lên được
DagsHub Experiments UI thật, xem từ máy nào cũng được, không cần vào VPN
hay SSH vào máy em.

## Khó khăn gặp phải và cách giải quyết

Phần này dài hơn dự tính vì gặp mấy lỗi khá hóc búa, đáng ghi lại:

- `.dvc/config` có dòng `credentialpath = ../sa-key.json`, đường dẫn tương
  đối này chỉ đúng trên máy em lúc tạo, còn trên GitHub Actions runner thì
  trỏ ra ngoài repo, không tồn tại - `dvc pull` báo 401 Invalid
  Credentials. Sửa bằng cách bỏ hẳn `credentialpath`, dùng
  `GOOGLE_APPLICATION_CREDENTIALS` (biến môi trường CI đã set sẵn) cho cả
  local lẫn CI luôn cho đồng nhất.

- Repo của em là fork từ template của thầy cô
  (`VinUni-AI20k/K3-Track2-Day21-CI-CD-for-AI-Systems`). GitHub mặc định
  tắt trigger `push` cho Actions trên repo fork, dù chạy tay
  (`workflow_dispatch`) vẫn bình thường và API kiểm tra permission vẫn báo
  `enabled: true` - dễ gây hiểu lầm là đã bật rồi. Hoá ra đây là một banner
  chỉ hiện trên giao diện web, tab Actions, phải tự bấm "I understand my
  workflows, go ahead and enable them" một lần, không có API hay CLI nào
  làm thay được. Bấm xong thì `push` tự trigger bình thường từ đó về sau.

- VM thiếu cả `src/serve.py` lẫn `sa-key.json` dù bước cấu hình VM báo
  chạy xong không lỗi gì. Hoá ra hai lệnh `scp` dùng đường dẫn `~/...` ở
  đích không chạy đúng trên Windows vì `pscp` (công cụ scp mà gcloud dùng
  trên Windows) không hiểu ký hiệu `~`, và lỗi đó lại không hiện rõ ràng
  lúc chạy. Đổi sang đường dẫn tuyệt đối `/home/<user>/...` là hết.

- Google Cloud SDK Shell trên máy em mặc định mở ra là cmd.exe chứ không
  phải PowerShell, nên cú pháp `$VAR = "..."` báo lỗi ngay từ đầu. Chuyển
  qua dùng PowerShell thường là được, vì `gcloud` đã nằm sẵn trong PATH hệ
  thống rồi.

- Kết nối DagsHub xong, `train.py` báo lỗi 404 khi tạo run. Do repo trên
  DagsHub hoàn toàn trống, chưa có experiment nào, mà `mlflow.start_run()`
  không tự tạo experiment "Default" như khi dùng file-store cục bộ. Thêm
  một dòng `mlflow.set_experiment(...)` tường minh trước `start_run()` là
  xong.

- Deploy chạy xong, health check vẫn fail dù server thật ra đã chạy tốt
  (SSH vào kiểm tra thấy service `active (running)`, gọi curl tay thì OK).
  Chỉ là `sleep 5` quá ngắn - model RandomForest 300 cây tải về từ GCS rồi
  unpickle mất gần 8 giây. Đổi sang retry loop 6 lần, mỗi lần cách nhau 5
  giây, thay vì chỉ đợi 5 giây một lần.

- Lúc `gcloud config set project`, em nhầm tên hiển thị của project
  (`track2-day16-...`) với Project ID thật (`mineral-aegis-505503-i2`) -
  hai cái này khác nhau, `gcloud` cần đúng Project ID.
