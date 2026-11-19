# Báo cáo Day 08 — 2A202603227 - Doãn Tùng Lâm

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | person 0.3 (mặc định), bicycle 0.2, motorbike 0.1, car 0.65, bus 0.3 (mặc định), truck 0.3 (mặc định) | r02 (conf 0.25) và r03 (conf 0.5): 0.5 mất nhiều người và xe máy ở xa (img_007 person 15 -> 6), nên không dùng một ngưỡng cao chung. `threshold_sweep` (4 ảnh gold, mục tiêu recall 90%): bicycle 0.2 đạt recall 5/5, precision 5/6; motorbike 0.1 đạt recall 11/12 nhưng precision chỉ 11/25; person và car không đạt 90% ở ngưỡng nào ≤ 0.5 (recall tối đa 18/24 và 7/8): person giữ mặc định 0.3; car recall 7/8 không đổi đến ngưỡng 0.85 nên tăng lên 0.65 để giảm box thừa từ 7 xuống 3 mà không mất recall (bỏ ngưỡng cao hơn vì gold chỉ có 8 ô tô); bus (1 vật) và truck (0 vật) dưới 5 vật nên không chọn ngưỡng riêng. |
| three_wheeler | YOLOE-26s, prompt mặc định của taxonomy (`--classes three_wheeler`) | 0.15 | r04: bắt đúng tuk-tuk và xích lô rõ (img_002, img_032); r05 (prompt "three-wheeled vehicle") thêm báo sai trên xe máy nên không dùng; r06 (prompt "tuk-tuk") chỉ bắt 1/3 xích lô ở img_032. Sweep: 0.15 đạt recall 7/7, precision 7/9; ở 0.3 recall giảm còn 5/7. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "bicycle=0.2,motorbike=0.1,three_wheeler=0.15,car=0.65" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` (ra 705 box cho 24 ảnh batch + 4 ảnh gold; ảnh gold không nạp vào CVAT).
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 base (`sam2.1_b.pt`). Mask bám đường biên tốt khi box đúng (r07, img_004: người và tuk-tuk xanh); chạy được bằng GPU 8 GB, không cần quyền truy cập đặc biệt như SAM 3 (gated). Chậm hơn YOLO26 (3.56 s/ảnh so với 0.74 s/ảnh ở r07 và r02). Chưa so với SAM 1 / mobile_sam.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? YOLO26 không có lớp `three_wheeler` nên tuk-tuk bị gọi `car` (r07: img_004, img_013), xích lô bị gọi `bicycle` (r02: img_032) hoặc `motorbike` (img_007); bán tải và SUV bị gọi `truck` (img_002, img_019). Đổi ngưỡng chỉ bớt hoặc thêm box, không đổi được lớp. Cách sửa: lấy `three_wheeler` từ YOLOE, còn lớp xe khác lấy từ YOLO26, và khi review phải kiểm tay box `truck` và các box `car` nằm đè lên box `three_wheeler`.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): Điểm cao mà sai: img_002 bán tải bị gọi `truck` 0.73 (r02) và `three_wheeler` 0.56 (r05); img_013 tuk-tuk bị gọi `car` 0.41 (r07). Điểm thấp mà đúng: xích lô xa ở img_032 chỉ 0.27 (r04) nên bị ngưỡng 0.3 cắt; xe máy nhỏ ở xa img_007 và img_015 điểm 0.27–0.4.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: ngưỡng được chọn và chấm trên cùng dữ liệu, và chỉ có 4 ảnh (24 người, 12 xe máy, 8 ô tô, 7 xe ba bánh, 5 xe đạp, 1 xe buýt, 0 xe tải) nên bicycle, three_wheeler dựa trên 5–7 vật, bus 1 vật, truck 0 vật. Ngưỡng car 0.65 chỉ dựa trên 8 ô tô gold nên có thể bỏ sót ô tô bị che khuất ở ảnh batch. Ngưỡng motorbike 0.1 đổi precision (11/25) lấy recall (11/12): lúc review sẽ phải xoá nhiều box thừa, và nếu chọn 0.2 thì precision 10/15, recall 10/12.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không):
  - Dùng claude chat để giải thích các định nghĩa
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)?
    - Không. Tôi tự gán tay 4 ảnh gold trong CVAT, chưa xem pre-label và chưa hỏi AI về các ảnh này
- **Ca tick `can_review` ở gold (2 box, đều vì nghi ngờ quy tắc, không phải vì vật khó vẽ):**
  - `img_012`, `person`: người ngồi trong tuk-tuk ở góc trên trái, bị mui, khung xe và kính che, chỉ thấy thấp thoáng qua cạnh hở. Không chắc phần nhìn thấy có đạt 20% (§4.1.2) hay không; mắt thường khó ước lượng 20%, hai người đánh giá có thể lệch nhau (câu mở số 4, §6). Nếu quyết là không đủ 20% thì xoá box.
  - `img_020`, `person`: tài xế tuk-tuk bên phải, nhìn qua kính chắn gió, bị khung mui và thanh chắn che. Cùng vấn đề ngưỡng 20% (câu mở số 4), thêm câu mở số 5: kính chắn gió có thuộc xe không, nên người nhìn xuyên kính có tính là bị che không.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt:
  - Windows 11, RTX 4060 Laptop 8 GB, CVAT v2.81.0 (Docker). CVAT 2.81 không nhận `"values": []` cho thuộc tính nhãn trong `labels_generated.json` ("attribute values must be a non-empty array"); đã dán bản sửa `values: [""]` cho `conf` và `["false"]` cho `can_review`. `check_submission.py` dùng regex `\d{5,}` cho MSHV nên không nhận MSHV có chữ cái (`2A202603227`).
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
