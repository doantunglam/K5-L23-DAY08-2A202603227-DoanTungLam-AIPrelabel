# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603227 - Doãn Tùng Lâm

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 2 | Khối xe máy đỗ dày đặc gần camera: từng xe hay `ignore_region`? | Vẽ từng xe nếu cao ≥ 20 px và nhìn thấy khoảng một nửa trở lên (còn rõ bánh hoặc đầu xe). Cụm từ 5 xe trở lên che nhau quá nửa, không tách được từng xe thì vẽ **một** `ignore_region` bao cụm và xoá các box trùng bên trong. | img_026, img_005 | CAL-06 |
| 3 | Hàng nhô ra khỏi xe: box xe có gồm hàng không? Mask thì sao? | Hàng nhô ra không quá 20% chiều dài xe thì thuộc box xe; nhô quá 20% thì box xe không ôm phần nhô thêm. Mask chỉ lấy phần hàng gắn cố định trong khung xe, bỏ phần nhô ra ngoài. | img_031, img_009 | CAL-07 |
| 4 | Người trong xe ba bánh có mui che: ngưỡng "còn thấy ≥ 20%" có hợp lý không? | Giữ ý tưởng nhưng **không đo bằng phần trăm diện tích** (hai người ước lượng 20% bằng mắt sẽ lệch nhau). Thay bằng tiêu chí quan sát được: gán `person` nếu thấy rõ **đầu hoặc một phần thân trên liên tục** (vai, ngực) và box cao ≥ 20 px; chỉ thấy vài mảng quần áo mờ thì không gán. | img_012, img_020 | CAL-01, CAL-02 |
| 5 | Kính chắn gió / cửa sổ xe thuộc mask xe không? Người nhìn xuyên kính có bị che không? | Kính thuộc xe (mask xe gồm cả kính). Người nhìn xuyên kính **không** tính là bị che; thanh chắn mảnh không giảm phần nhìn thấy. Người ngồi trong xe vẫn là một box `person` riêng, mask xe không phủ lên người. | img_013, img_020 | CAL-03 |

**Chưa chốt:** câu #1 (xe đạp điện: `bicycle` hay `motorbike`), #6 (box máy chồng nhau trong ảnh đêm) và #7 (mức tin cậy tối thiểu để nhận box máy không cần sửa). Lý do: không gặp xe đạp điện rõ ràng trong 4 ảnh gold và 24 ảnh batch nên chưa đủ ví dụ để quyết; #6 chỉ gặp ở img_024, img_026, img_029 nhưng chưa đo; #7 là quy tắc chất lượng cần thêm dữ liệu (tương quan giữa `conf` và box phải sửa), mẫu hiện tại quá nhỏ.

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_013, img_004 | Tuk-tuk Thái có mui | YOLO26 gọi `car` (0.41, 0.37); đổi ngưỡng không sửa được vì không có lớp `three_wheeler` | Mọi xe ba bánh luôn là `three_wheeler`; xoá box `car` trùng với box ba bánh, giữ một box ôm cả mui và khung | Mục 2, dòng `three_wheeler`: thêm cột "máy hay nhầm" (CAL-04) |
| img_032, img_007 | Xích lô đỗ ven đường | YOLO26 gọi `bicycle` (img_032) hoặc `motorbike` (img_007) | Xích lô là `three_wheeler`; xoá box `bicycle`, `motorbike` nằm trên bánh xích lô | Mục 2, dòng `three_wheeler` (CAL-04) |
| img_009 | Xe máy chở chuối chồng cao | YOLOE gọi `three_wheeler` | Vẫn là `motorbike`; box chỉ ôm xe, hàng nhô ra hơn 20% không tính | Mục 4.1.5: nhấn mạnh "hàng nhiều không đổi lớp" (CAL-05) |
| img_003, img_014 | Mái che bạt, quang gánh hàng rong | YOLOE gán `three_wheeler` | Không gán: mái che, ô dù, quang gánh không thuộc lớp nào | Mục 2, "vật không thuộc lớp nào": thêm mái che, ô dù, quang gánh (CAL-05) |
| img_012 | Người ngồi trong tuk-tuk, chỉ thấy thấp thoáng qua cạnh hở | (gold gán tay, tick `can_review`) | Giữ box `person`; áp dụng tiêu chí "đầu hoặc thân trên liên tục" | Mục 4.1.2 (CAL-01) |
| img_020 | Tài xế tuk-tuk nhìn qua kính chắn gió | (gold gán tay, tick `can_review`) | Giữ box `person`; kính thuộc xe, không tính là che | Mục 4.1.2 và 4.2.3 (CAL-02) |
| img_013 | Mask SAM lấn vào tài xế và hành khách | SAM 2.1 từ box `car` của YOLO26 | Mask xe loại bỏ người ngồi bên trong; người là box `person` riêng | Mục 4.2.4 (CAL-03) |
| img_026 | Bãi xe máy đêm rất dày ở xa | YOLOE / YOLO26 cho nhiều box xe máy nhỏ chồng nhau | Cụm xa, nhỏ, che nhau quá nửa: một `ignore_region`, xoá box trùng bên trong | Mục 4.1.6 (CAL-06) |
| img_031 | Xe đạp chở mâm hàng nhô ra khoảng 18% | — | Hàng nhô ≤ 20% chiều dài xe thuộc box xe | Mục 4.1.5 (CAL-07) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu mở #2, #3, #4, #5; thêm quy tắc xe ba bánh luôn là `three_wheeler`, hàng nhiều không đổi lớp, mái che / ô dù / quang gánh không phải vật thuộc lớp nào, mask xe loại bỏ người trong xe | CAL-01 đến CAL-07 |
