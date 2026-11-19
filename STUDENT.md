# Thông tin học viên

- Họ và tên: Doãn Tùng Lâm
- MSHV: 2A202603227
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: doantunglam
- Repo bài nộp (URL): https://github.com/doantunglam/K5-L23-DAY08-2A202603227-DoanTungLam-AIPrelabel
- Commit chốt bài (hash): <<>>
- Máy: Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU 8 GB (CUDA 12.4, torch 2.6.0+cu124)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 479 / 19 / 98 (mẫu số: 600 box máy; 79.8% / 3.2% / 16.3%); added: 29 (mẫu số: 527 đối tượng sau duyệt; 5.5%)
- So với gold: precision / recall của **máy** 62.3% / 84.2% (TP 48, FP 29, FN 9; mẫu số: 77 box máy cho precision, 57 đối tượng gold cho recall; số đo lạc quan vì ngưỡng chọn trên chính 4 ảnh gold)
- Bắt bẫy: 10/12 = 83.3%
- Thời gian: gán tay 4.0 phút/ảnh (16 phút / 4 ảnh); duyệt có pre-label 6.1 phút/ảnh (146 phút / 24 ảnh)
- Lỗi QC: 40.0% theo ảnh (2/5), 4.6% theo đối tượng (5/109); cận trên 95% của tỉ lệ ảnh lỗi thật là 77% (mẫu nhỏ)
- **Quyết định:** Rework — lý do một dòng: ảnh có lỗi 2/5 = 40% vượt ngưỡng 2%, và lỗi `thua` chiếm 5/5 lỗi (box máy điểm thấp chồng lên cụm xe đỗ), nên là lỗi hệ thống
