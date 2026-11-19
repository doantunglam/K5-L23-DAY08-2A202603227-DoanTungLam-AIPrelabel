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
| accepted | 479 | 79.8% | 600 box máy (24 ảnh batch+gold; đã loại 27 đối tượng quanh bẫy) |
| edited | 19 | 3.2% | 600 box máy |
| deleted | 98 | 16.3% | 600 box máy |
| added | 29 | 5.5% | 527 đối tượng sau duyệt |

(Nguồn: `qc/review_stats.md`, chạy sau khi export `batch_reviewed.json`. Trong 98 deleted + 29 added có 9 ca đổi lớp, nên máy sót thật khoảng 20 vật.)

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? `three_wheeler`: chỉ 6/40 box được giữ nguyên (accepted), 11 sửa, 23 xoá (23/40 = 58% bị xoá). Máy (YOLOE) gán `three_wheeler` cho mái che, ô dù, quang gánh và xe máy chở hàng (img_003, img_005, img_009, img_014, img_018), và hay chia một xe thành hai box hoặc gộp hai xe thành một (img_010, img_013, img_023, img_035). Tiếp theo là `car`: 16 ô tô bị máy sót, hầu hết ở ảnh đêm img_024 (hàng xe phía xa). `motorbike` có 54/234 = 23% box bị xoá, phần lớn là box trùng hoặc cao dưới 20 px ở các ảnh dày (img_005, img_018, img_026, img_034), hệ quả của ngưỡng 0.1.
- So với gold: precision / recall của **máy** 62.3% / 84.2% (TP 48, FP 29, FN 9; mẫu số: 77 box máy cho precision, 57 đối tượng gold cho recall; khoảng tin cậy 95%: 51–72% và 73–91%; số đo lạc quan vì ngưỡng chọn trên chính 4 ảnh gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 10/12 = 83.3% (khoảng tin cậy 95%: 55–95%; ngưỡng đạt 80%). Bắt đủ missing 3/3, extra 3/3, wrong_class 3/3; loại lọt là `loose` (box lỏng), bắt 1/3: lọt ở img_024 và img_029. Vì sao: bước 4 (box lỏng) là bước yếu nhất của lượt duyệt đầu, vì hai ảnh lọt đều là ảnh đêm, rất dày xe, box lệch ít pixel khó thấy bằng mắt.
- Accepted cao (79.8%) mà bắt bẫy vẫn đạt 83.3% nói gì về cách duyệt? Giữ nguyên phần lớn box máy không làm mất khả năng bắt lỗi rõ (thừa, sót, sai lớp: 9/9), nhưng lỗi nhỏ như box lỏng thì dễ lọt khi accepted cao: cần bước kiểm cạnh box riêng thay vì chỉ nhìn đối tượng có được bao hay không.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 16 phút / 4 ảnh = 4.0 phút/ảnh vs duyệt 146 phút / 24 ảnh = 6.1 phút/ảnh.
- Thiên lệch (bạn gán gold trước nên đã quen cảnh): có ít nhất ba thiên lệch làm so sánh này không công bằng.
  - (1) Tôi gán gold trước nên đã quen kiểu ảnh phố và các quy tắc guideline (xe ba bánh, người ngồi sau xe), nên lượt duyệt đáng lẽ nhanh hơn, vậy mà vẫn chậm hơn gán tay mỗi ảnh.
  - (2) Hai tập ảnh không cùng độ khó: gold có 60 đối tượng / 4 ảnh = 15 đối tượng/ảnh, còn batch có 541 / 24 = 22.5 đối tượng/ảnh (máy gợi ý 614 box, hơn 25 box/ảnh), nhiều ảnh đêm và ảnh dày xe máy (img_005, img_018, img_024, img_026, img_029), nên phần chậm hơn một phần do ảnh dày hơn, không chỉ do duyệt.
  - (3) Gold chỉ 4 ảnh nên 4.0 phút/ảnh là ước lượng thô
- Kết luận: có cho `person`, `car`, `bus`, `bicycle` (giữ nguyên 260/272 = 96% box `person`; sửa ít) và dùng có điều kiện cho `motorbike`; không nên tin `three_wheeler` của YOLOE mà không duyệt kỹ (chỉ 6/40 = 15% box được giữ nguyên), dựa trên `qc/review_stats.md` (accepted 79.8%, bắt bẫy 10/12).
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Với mỗi ảnh, đếm vật bằng mắt trước khi bật nhãn máy (Bước 0) và ghi số đếm vào ghi chú; kiểm tra: số đếm so với tổng sau duyệt không lệch quá vài vật.
  2. Duyệt riêng một lượt chỉ để kiểm cạnh box (bước 4) cho ảnh đêm hoặc ảnh dày; kiểm tra: tỉ lệ bắt bẫy loại `loose` tăng từ 1/3 lên ít nhất 2/3 ở lượt chạy sau.
  3. Với lớp `three_wheeler` luôn duyệt từng box, không dùng accepted hàng loạt; kiểm tra: tỉ lệ accepted của lớp này (hiện 15%) được báo riêng khỏi tỉ lệ tổng.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 (tuk-tuk Thái, có tài xế và hành khách) | SAM 2.1 base (`sam2.1_b.pt`) | 6 (3 dương, 3 âm) | Mask phủ cả xe, mui và khung; tài xế và hành khách bị phủ **một phần** (loang lổ, có lỗ), không loại sạch dù đã click âm lên tài xế và hành khách; hàng hoá không có | Mask lấn vào người trong xe (tài xế phủ phần lớn, khách phủ rời rạc); 1 polygon, 43.7% ảnh | `masks/tuktuk_img013.jpg` |
| img_010 (xích lô đầu hàng, nhìn từ sau, có người đạp) | SAM 2.1 base (`sam2.1_b.pt`) | 5 (3 dương, 2 âm) | Mask bám mui đỏ, khung, bánh và yên; **không** gồm người đạp (click âm lên người đạp có tác dụng); không có hàng | Mask bị chia 4 mảnh do người đạp che giữa xe; nan khung mảnh bị cắt rời rạc; 11.3% ảnh | `masks/cyclo_img010.jpg` |
| img_023 (tuk-tuk đỏ ở giữa, bị tuk-tuk xanh phía trước che một phần) | SAM 2.1 base (`sam2.1_b.pt`) | 5 (3 dương, 2 âm) | Mask gồm mui, thân và bánh của xe giữa, không gồm người đàn ông áo vàng đứng cạnh (click âm đúng chỗ); không có hàng | Mép chỗ giáp xe xanh phía trước cắt khá sạch; chỗ khuất sau xe trước không đoán thêm (đúng nguyên tắc chỉ vẽ phần nhìn thấy); 7.1% ảnh | `masks/tuktuk_mid_img023.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? Giúp: (3) chỉ mask phần nhìn thấy và (4) hai vật chạm nhau không chồng lên nhau: ở img_023 phần giáp xe giữa và xe trước thuộc xe phía trước, mask xe giữa không lấn sang; quy tắc lỗ thủng (2) giúp quyết định bỏ phần khung hở xích lô (img_010) ra khỏi mask. Thiếu: mục 4.2 không nói người ngồi trong xe có tính thuộc mask xe không, và kính chắn gió (câu mở số 5, mục 6) nên ở img_013 không biết nên loại hay giữ phần người nhìn qua cạnh hở; phần người nằm sau thanh chắn làm mask loang lổ nên chưa theo được quy tắc 4.1.4 (một box `person` riêng, xe một mask riêng). Đề xuất đưa vào v1.1: mask xe loại bỏ người ngồi bên trong.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: chưa thử; cả 3 mask ở trên đều tạo offline bằng `scripts/sam2_points.py` với `sam2.1_b.pt` (click dương và âm trên ảnh gốc), nên chưa so được với Interactor trong CVAT.

## 6. Calibration (Part 6)

- Số quyết định: 7 (`calibration_decisions.csv`: CAL-01 đến CAL-07; 2 `can_review`, 3 `cau_hoi_mo`, 2 `loi_may`); quan trọng nhất: CAL-04 (xe ba bánh luôn là `three_wheeler`: YOLO26 gọi tuk-tuk và xích lô là `car`, `bicycle`, `motorbike`, và `three_wheeler` là lớp máy tệ nhất, chỉ 6/40 box được giữ nguyên) và CAL-06 (cụm xe máy đỗ dày: vẽ từng xe hay `ignore_region`, vì đây là kiểu lỗi `thua` lặp lại ở bước QC).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: chốt câu mở #2 (xe máy đỗ dày: ngưỡng 20 px, che quá nửa thì `ignore_region`), #3 (hàng nhô ≤ 20% chiều dài xe thuộc box xe), #4 (thay "20% diện tích" bằng tiêu chí "thấy rõ đầu hoặc thân trên liên tục"), #5 (kính thuộc xe, người sau kính không tính là che; mask xe không phủ người bên trong); thêm quy tắc xe ba bánh luôn là `three_wheeler`, hàng nhiều không đổi lớp, mái che / ô dù / quang gánh không phải vật thuộc lớp nào. Chưa chốt: #1 (xe đạp điện), #6, #7, vì chưa có ví dụ rõ trong 28 ảnh.
- Nguyên nhân gốc: pha trộn. Guideline mơ hồ: CAL-01, 02, 03, 06, 07 (câu mở #2, #3, #4, #5). Máy: CAL-04, 05 (YOLO26 không có lớp `three_wheeler`; YOLOE nhầm mái che, quang gánh và xe máy chở hàng). Chưa thấy ca nào do chưa nắm quy tắc.

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ngẫu nhiên, seed 42: img_003, img_006, img_014, img_031, img_035), 109 đối tượng; người kiểm: Doãn Tùng Lâm.
- Lỗi theo **đối tượng**: 5/109 = 4.6% · theo **ảnh**: 2/5 = 40.0% (ngưỡng Release ≤ 2%), nguồn `qc/qc_report_auto.md`.
- Top lỗi: `thua` (box thừa) 5/5 lỗi, không có lỗi bỏ sót, sai lớp, sai vị trí, sai thuộc tính hay sai mask trong mẫu. Ví dụ: img_006 (4 box `motorbike` điểm thấp 0.10–0.33 chồng lên cụm xe đỗ bên trái: #64, #70, #71, #72) và img_035 (#520 `motorbike` 0.14 trùng #530 0.47).
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): 2 ảnh lỗi trong 5 ảnh mẫu, cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là 77%. Ngưỡng 2% không thể được chứng minh bằng 5 ảnh. Mẫu cũng không có ảnh dày xe nào (img_005, img_018, img_024, img_026, img_029) nên lỗi thật trên toàn lô có thể cao hơn 4.6%.
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): kiểm được số lỗi theo phiếu, tỉ lệ accepted / edited / deleted / added, bắt bẫy. Không kiểm được box có ôm sát vật thật không (ví dụ 2 bẫy `loose` ở img_024 và img_029 lọt, bắt bẫy `loose` chỉ 1/3), vật mà cả người duyệt lẫn máy cùng bỏ sót ngoài gold, và việc duyệt có theo đúng checklist hay không.

**Quyết định: Rework**, ngày 2026-11-19. Lý do (đi theo Hold → Rework → Release): Không Hold vì guideline đủ rõ để gán nhãn (3 câu mở #1, #6, #7 chưa chốt nhưng không gây lỗi nào trong mẫu) và dữ liệu hợp lệ (`validate_labels.py` sạch). Rework vì ảnh có lỗi 2/5 = 40% vượt xa ngưỡng 2%, lỗi `thua` chiếm 5/5 lỗi nên là lỗi hệ thống (box máy điểm thấp chồng lên cụm xe đỗ, hệ quả của ngưỡng `motorbike=0.1`), và bắt bẫy `loose` chỉ 1/3. Không thể Release.
Nếu Rework / Hold, việc cần làm tiếp: xoá 5 box thừa (img_006 #64, #70, #71, #72; img_035 #520) hoặc gộp cụm xe đỗ của img_006 thành `ignore_region` theo CAL-06; rà các ảnh dày xe còn lại (img_005, img_018, img_024, img_026, img_029) tìm box trùng cùng kiểu; cân nhắc tăng ngưỡng `motorbike` lên 0.25 để giảm box thừa; export `submission/labels/qc_fixed.json`, chạy lại `build_release.py` và `qc_sample.py --force`, ghi kết quả vòng sau vào báo cáo.

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
- Dùng cache `data/cache/` thay vì tự chạy? Không: đã tự chạy `prelabel.py` cho YOLO26 và YOLOE trên toàn bộ 36 ảnh (run r08 và r09 trong `work/runs.csv`).
- Lỗi / bug script đã gặp (nếu có): `scripts/check_submission.py` dòng 48 dùng regex `(\d{5,})` để bắt MSHV trong tiêu đề README, chỉ nhận MSHV toàn chữ số, nên MSHV của tôi (`2A202603227`, có chữ `A`) bị báo lỗi "README.md phải có dòng tiêu đề" dù tiêu đề đúng mẫu `SUBMISSION.md`, và phép kiểm `traps_key.json` khớp MSHV bị bỏ qua. Đã sửa regex thành `([0-9A-Za-z]{5,})` ở máy tôi để chạy kiểm tra; chưa có xác nhận từ giảng viên và script chấm có thể vẫn dùng bản gốc. Tôi không đổi MSHV trong tiêu đề.
- Một điều bạn làm khác lần sau: Không chọn ngưỡng `motorbike=0.1` chỉ vì đạt recall 90% trên 4 ảnh gold (precision chỉ 11/25). Ngưỡng đó sinh nhiều box thừa chồng lên các cụm xe đỗ (img_006, img_035), và cả 5/5 lỗi ở bước QC là loại `thua`. Lần sau tôi sẽ thử ngưỡng 0.25 (số box phải sửa trên gold chỉ 7 so với 16 ở 0.1) rồi so số box phải xoá khi review, và chọn mẫu QC có cả các ảnh dày xe (img_005, img_018, img_024, img_026, img_029) thay vì chỉ 5 ảnh ngẫu nhiên.
