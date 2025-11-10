# `data/build_vqa_conversations.py` 
- Convert our json to training data format for Llama Factory.
**Input**
`--input`: Path tới link .jsonl .
`--output`: Đặt tên và path của output file.
`--dataset-root`: Thư mục chứa toàn bộ file .jsonl, image folder, ... (để nối path)
`--has_reasoning`: Thêm reasoning và reasoning format vào phần response trong output dataset.
`--has_vqa_prompt`: Thêm vqa prompt vào phần question trong output dataset.

**Output**
- File JSONL, mỗi dòng có dạng:
  - `id`: id.
  - `image`: tên ảnh suy ra từ `image_id` với đuôi `.png`.
  - `conversations`: mảng 2 lượt thoại:
    - `{"from": "human", "value": "<image>...question..."}` (có thể chèn VQA prompt nếu bật `--has_vqa_prompt true`).
    - `{"from": "gpt", "value": "<think>...reasoning...</think><answer>...answer...</answer>"}` nếu `--has_reasoning true`, ngược lại chỉ là chuỗi câu trả lời.
- Lưu ý: nếu `--has_reasoning true`, tên file output được tự động thêm hậu tố `_reasoning`; nếu `--has_vqa_prompt true`, thêm hậu tố `_vqaprompt`.
- Xem các file như 

# `data/convert2lmf.py
- Chuyển .jsonl của các dataset sang định dạng llamafactory (lmf) để evaluate.
**Input**
`--dataset`: dataset mong muốn (đã cài trước các path).
`--vqa_prompt`: output mặc định chứa sẵn vqa prompt nếu đặt là True.
**Output**
- Xem các file như `textvqa_val_lmf.jsonl`, `infographicvqa_val_lmf.jsonl`, ...