# Bounding Box Selection Methodology

## Processing Pipeline

The bounding box refinement follows a sequential 3-step pipeline:

```
extracted_bboxes_base.json 
    |
    v
[Step 1] clean_small_text_bboxes.py (Automated)
    |
    v
[Step 2] toggle_bbox_category.py (Manual)
    |
    v
extracted_bboxes_process.json
```

### Pipeline Steps

1. **Automated Text Filtering** (`clean_small_text_bboxes.py`)
   - Removes text bboxes with area < 15,000 pixels²
   - Preserves all element and background bboxes
   - First pass to eliminate obviously too-small text regions

2. **Manual Refinement** (`toggle_bbox_category.py`)
   - Layout deletion: Remove entire layouts with insufficient quality
   - Category toggling: Convert element bboxes to text (or vice versa) based on position
   - Bbox deletion: Remove problematic text or element bboxes
   - Conflict resolution: Handle overlapping bboxes

3. **Final Output** (`extracted_bboxes_process.json`)
   - High-quality layouts suitable for infographic generation
   - Clean bbox distributions without excessive overlap
   - Balanced text and element placements

## Selection Rules

The following criteria were applied during manual refinement:

### Rule 1: Layout Deletion

**Criterion**: Layouts with fewer than 6 text bboxes after automated filtering

**Rationale**: Infographics require sufficient text content to be informative. Layouts with too few text regions produce poor-quality results.

**Example**: Layout 71 had only 4 text bboxes and 20 element bboxes - deleted for insufficient text content.

### Rule 2: Text Bbox Deletion

**Criteria**:
- Area < 15,000 pixels² (automated by script)
- Overlapping with other text bboxes (manual deletion)
- Poor positioning or visual quality issues (manual deletion)

**Rationale**: Small text bboxes are difficult to render legibly in infographics. Overlapping text creates visual clutter and reading difficulties.

**Example**: Layout 4 had 3 text bboxes removed with areas around 4,898 pixels² each - all fell below the area threshold.

### Rule 3: Element to Text Toggle

**Criteria**:
- Element bbox is in a good position (centered in frame or visually prominent)
- Element bbox does NOT overlap significantly with other figure elements
- Position is suitable for text placement (adequate space, good visibility)

**Rationale**: Some element bboxes are better suited for text content based on their position and size. Converting them to text increases text density and improves infographic balance.

**Example**: Layout 10 had 2 element bboxes converted to text bboxes. These were positioned in clear areas without conflicting with other visual elements.

### Rule 4: Conflict Resolution

**Criteria**:
- When text bbox overlaps with element bbox:
  - If text is large: Delete the overlapping element bbox
  - If text is small: Delete the text bbox
- When multiple text bboxes overlap:
  - Delete the smaller bbox

**Rationale**: Overlapping content creates visual confusion in infographics. Prioritizing larger bboxes ensures the most important content is preserved.

**Example**: Layout 15 had 1 element bbox deleted (area: 30,084 pixels²) due to conflict with text bboxes, preserving the more important text content.

## Statistics

### Overall Summary

| Metric | Base File | Process File | Change |
|--------|-----------|--------------|--------|
| Total Layouts | 100 | 85 | -15 |
| Text Bboxes | 1,241 | 880 | -361 |
| Element Bboxes | 1,710 | 1,284 | -426 |

### Changes Applied

| Change Type | Count | Percentage |
|-------------|-------|------------|
| Deleted Layouts | 15 | 15% of base |
| Layouts with Bbox Changes | 59 | 69.4% of surviving |
| Text Bboxes Deleted | 215 | 17.3% of base text |
| Element Bboxes Deleted | 98 | 5.7% of base elements |
| Element → Text Toggles | 42 | 2.5% of base elements |
| Text → Element Toggles | 17 | 1.4% of base text |

### Deleted Layouts Analysis

**Integer-indexed layouts deleted**: 11 layouts  
Indices: 0, 5, 20, 32, 44, 53, 56, 71, 92, 99, 108

**String-indexed layouts deleted**: 4 layouts  
Indices: cn_0, es_4, fr_6, jp_11

**Deletion reasons**:
- 1 layout explicitly had < 6 text bboxes (layout 71: 4 text bboxes)
- 14 layouts had other quality issues (excessive small text, poor layout quality, insufficient usable text after filtering)

### Bbox Changes Breakdown

Of the 85 surviving layouts:
- **59 layouts** (69.4%) had at least one bbox modification
- **26 layouts** (30.6%) remained unchanged

**Text bbox deletions** (215 total):
- Small area deletions: Majority (~80%) were automated (area < 15,000 px²)
- Manual deletions: ~20% removed due to overlap or quality issues

**Element bbox deletions** (98 total):
- Primarily due to conflicts with text bboxes
- Some removed for poor positioning or visual quality

**Category toggles** (59 total):
- Element → Text: 42 conversions (71.2% of toggles)
- Text → Element: 17 conversions (28.8% of toggles)
- Toggle criteria: Position quality, overlap avoidance, layout balance

## Key Findings

1. **Text Content Priority**: The refinement process prioritizes maintaining high-quality text bboxes, as evidenced by the higher deletion rate of element bboxes in conflict situations.

2. **Area Threshold Effectiveness**: The 15,000 pixel² threshold successfully filtered out most unreadable text, with ~173 text bboxes (13.9% of base) removed automatically.

3. **Layout Quality Focus**: 15% of layouts were deemed unsuitable for infographic generation and removed entirely, ensuring only high-quality layouts remain.

4. **Balanced Toggle Approach**: Element-to-text conversions (42) significantly outnumber text-to-element conversions (17), reflecting the strategy of maximizing text density where position allows.

5. **Manual Refinement Impact**: 69.4% of surviving layouts required manual intervention, demonstrating that automated filtering alone is insufficient for optimal results.

## Validation Checklist

For any future bbox refinement, ensure:

- [ ] Layouts have at least 6 usable text bboxes
- [ ] Text bboxes have area ≥ 15,000 pixels²
- [ ] No significant overlap between text bboxes
- [ ] Element bboxes in good positions are converted to text where appropriate
- [ ] Conflicts between text and elements are resolved (prioritize larger/more important content)
- [ ] Overall layout has good visual balance between text and elements

## Tools Used

- **clean_small_text_bboxes.py**: Automated text filtering (area threshold)
- **toggle_bbox_category.py**: Manual bbox toggling and deletion
- **compare_bbox_files.py**: Analysis and validation tool

## Files Generated

- **Input**: `/home/thinhnp/hf_vqa/bboxes/extracted_bboxes_base.json`
- **Output**: `/home/thinhnp/hf_vqa/bboxes/extracted_bboxes_process.json`
- **Report**: `/home/thinhnp/hf_vqa/bbox_comparison_report.json`

## Conclusion

The bbox selection methodology successfully refined 100 base layouts into 85 high-quality layouts suitable for infographic generation. Through a combination of automated filtering (area thresholds) and manual refinement (position-based decisions, conflict resolution), the process reduced bbox count by 29.1% (from 2,951 to 2,164 bboxes) while maintaining layout quality and improving visual balance.

The resulting dataset emphasizes readable text content, well-positioned elements, and conflict-free layouts - all critical factors for generating high-quality infographic images in the NARRATOR VQA pipeline.

