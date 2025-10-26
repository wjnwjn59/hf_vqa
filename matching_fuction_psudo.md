# Thuật toán matching cho Infographic

## Ký hiệu và đầu vào
- Tập layout: $\mathcal{L} = \{\ell_1, \dots, \ell_M\}$. Mỗi layout $\ell$ có danh sách bbox $\mathcal{B}(\ell) = \{b_1, \dots, b_K\}$ với nhãn $\mathrm{cat}(b) \in \{\text{background}, \text{element}, \text{text}\}$.
- Mỗi bbox $b$: toạ độ $\mathrm{tl}(b)=(x_{\min},y_{\min})$, $\mathrm{br}(b)=(x_{\max},y_{\max})$, diện tích $A(b)=(x_{\max}-x_{\min})(y_{\max}-y_{\min})$.
- Canvas cố định: $\mathcal{F}=[0,0]\times[896,2240]$.
- Dữ liệu Narrator: cấu trúc `generated_infographic` chứa:
  - Caption đầy đủ $C$ (full_image_caption)
  - Danh sách figures $F$ với mỗi figure có trường `ideas`
  - Tập mô tả figure $I=(i_1,\dots,i_{|I|})$ được random chọn từ `ideas` của mỗi figure
  - Tập văn bản $T=(t_1,\dots,t_{|T|})$ trích từ chuỗi trong dấu ngoặc kép "…" trong $C$
- Bảng màu `color_idx`: ánh xạ tên↔id màu; bảng font `font_idx`: ánh xạ token font English↔id.
- Chỉ số bắt đầu $s$ (start_wiki_idx).

## Lựa chọn layout và thuộc tính kiểu chữ/màu
- Chọn layout ngẫu nhiên từ tập layout đang có mà chưa được sử dụng. Khi hết layout, wrap around và tái sử dụng.
- Phân loại bbox theo loại: $\mathcal{B}_{bg}$, $\mathcal{B}_{el}$, $\mathcal{B}_{txt}$.
- Lọc element và text bbox: $\mathcal{B}^*_{el}=\{b\in\mathcal{B}_{el}\mid \mathrm{br}(b)\neq(896,2240)\}$, $\mathcal{B}^*_{txt}=\{b\in\mathcal{B}_{txt}\mid \mathrm{br}(b)\neq(896,2240)\}$.
- Trích font/màu từ `font_color_info` các bbox text:
  - Nếu như font đang có đã là font English (pattern `en-font-X`), ta giữ nguyên. Nếu không, thay thế bằng một font English ngẫu nhiên từ `font_idx`.
  - Giữ nguyên mã màu từ các bbox text, bỏ mã màu "white". Nếu không có mã màu nào hợp lệ, chọn ngẫu nhiên 1-4 màu từ bảng màu (không bao gồm "white").

## Xây dựng các lớp (layers)
1) **Base**: Ta lấy full image figure để xây dựng layout cho toàn ảnh với caption đã làm sạch $C'$ (loại đoạn "Figure: … .", chuẩn hoá khoảng trắng):
   - $L_{base}=(\text{category}=\text{base},\ \mathrm{tl}=(0,0),\ \mathrm{br}=(896,2240),\ \text{caption}=C')$.
2) **Background**: Sử dụng background đã có sẵn từ layout nếu có. Trường hợp không có background có sẵn, ta dùng một background mặc định:
   - Nếu có: $L_{bg}=(\text{category}=\text{element},\ \mathrm{tl}=\mathrm{tl}(b_{bg}),\ \mathrm{br}=\mathrm{br}(b_{bg}),\ \text{caption}=\text{caption}(b_{bg}))$
   - Nếu không có: $L_{bg}=(\text{category}=\text{element},\ \mathrm{tl}=(0,0),\ \mathrm{br}=(896,2240),\ \text{caption}=\text{default\_bg\_caption})$
3) **Elements** (figure trang trí): Xử lý theo thứ tự ưu tiên để tránh overlap với text:
   - Lọc regular elements: $\mathcal{B}^*_{el}=\{b\in\mathcal{B}_{el}\mid \mathrm{br}(b)\neq(896,2240)\}$
   - Giới hạn số figure: $|I'| = \min(|I|, |\mathcal{B}^*_{el}|)$, truncate $I$ nếu cần
   - Sắp xếp $\mathcal{B}^*_{el}$ theo diện tích giảm dần, chọn non-overlapping bboxes
   - Gán caption theo thứ tự: bbox lớn nhất ← $i_1$, bbox lớn thứ hai ← $i_2$, ...
   - Sắp xếp lại các cặp (bbox, caption) theo reading order (top→bottom, left→right)
4) **Text**: Chọn text bbox không overlap với element đã chọn, có cơ chế fallback:
   - Lọc regular text: $\mathcal{B}^*_{txt}=\{b\in\mathcal{B}_{txt}\mid \mathrm{br}(b)\neq(896,2240)\}$
   - Tìm text bbox không overlap với elements: $\mathcal{B}_{txt}^{safe} = \{b \in \mathcal{B}^*_{txt} \mid \neg \text{overlap}(b, \mathcal{B}_{el}^{selected})\}$
   - Nếu $|\mathcal{B}_{txt}^{safe}| < |T|$: iteratively remove elements có nhiều overlap nhất cho đến khi đủ text bbox
   - Adjust $|T|$ nếu vẫn không đủ text bbox
   - Gán text và format: `Text "content" in <color-X>, <en-font-Y>`

## Đầu ra và chỉ số
- Thứ tự lớp: $\text{layers\_all}=[L_{base}, L_{bg}, \{L^{el}_k\}_{k=1}^{|\mathcal{S}|}, \{L^{txt}_j\}_{j=1}^{|\mathcal{S}_t|}]$.
- Chỉ số wiki duy nhất: $w = s + \mathrm{idx} + 1$ (với `idx` là chỉ số 0-based trong dãy đầu vào).
- Lưu kèm `original_bbox_index` là chỉ số layout đã chọn.

## Pseudo-code 
- Trích $I$ từ figures.ideas (random select), $T$ từ quoted strings trong $C$; chọn layout $\ell$ và tách $\mathcal{B}$ thành background/element/text.
- Trích font token $f$ và color IDs từ bbox text layout.
- Tạo `base` layer với cleaned caption $C'$, `background` layer từ layout hoặc fallback.
- Chọn và gán element bboxes: sort by area → select non-overlapping → assign captions → sort by reading order.
- Chọn text bboxes với overlap avoidance: find safe positions → remove conflicting elements if needed → assign texts với font/color từ individual bbox.
- Kết hợp các lớp theo thứ tự: base → background → elements → text; sinh bản ghi kết quả với `index`, `layers_all`, `full_image_caption`, `original_bbox_index`.
