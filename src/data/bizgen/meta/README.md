## Structure of the meta files:
The meta information for each infographic/slide is organized in a dict format with keys:
- **index**: An identifier
- **full_image_caption**: The global prompt that includes all the contents in an infographic/slide
- **layers_all**: The bounding box and caption for all the layers, organized in z-order, from bottom to top. The first layer is the global prompt. Text layers are always on the top. Each layer has keys:
    - ***category***: "base" for global; "text" for "text"; "element" for non-text.
    - ***top_left***: Bounding box, x1,y1
    - ***bottom_right***: Bounding box, x2,y2
    - ***text***: for text layers, the visual text content
    - ***caption***: the regional prompt, for text layers the captions are visual text content with special tokens representing font type and color.
    - ***cfg***: the cfg value for this single layer. You can add this key when refining images with lcfg.
