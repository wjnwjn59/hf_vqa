# NARRATOR Dataset Notes

## Dataset Statistics

- **Total validation entries**: 11,873 QA pairs
- **Valid answers**: 5,928 (49.93%)
- **Empty answers (unanswerable)**: 5,945 (50.07%)

## Unanswerable Questions

The NARRATOR dataset follows SQuAD v2 format, which includes unanswerable questions where the `answers.text` list is empty. These questions are designed to test whether models can recognize when an answer cannot be determined from the context/image.

### Handling in Evaluation

For questions with empty answers:
- Ground truth is set to empty string (`""`)
- Models should ideally respond with "unanswerable" or similar
- ANLS metric will handle these appropriately

### Example Unanswerable Questions

From the validation set:
- "Who gave their name to Normandy in the 1000's and 1100's" (ID: 5ad39d53604f3c001a3fe8d1)
- "What is France a region of?" (ID: 5ad39d53604f3c001a3fe8d2)
- "Who did King Charles III swear fealty to?" (ID: 5ad39d53604f3c001a3fe8d3)

## Bug Fix

**Issue**: IndexError when accessing `qa["answers"]["text"][0]` for empty answer lists

**Solution**: Added check in `process_batch_data()`:
```python
if qa["answers"]["text"]:
    gts.append(qa["answers"]["text"][0])
else:
    gts.append("")  # Empty string for unanswerable questions
```

This ensures the inference pipeline handles both answerable and unanswerable questions correctly.
