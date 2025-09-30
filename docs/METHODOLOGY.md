# Methodology

## Background Modelling Techniques

This document explains the theoretical background and implementation details of the three experiments.

## Experiment 1: Consecutive Frame Differencing

### Theory

Frame differencing is the simplest background subtraction technique. It detects moving objects by comparing consecutive frames in a video sequence.

**Algorithm:**
1. Convert current frame to grayscale: I(t)
2. Convert previous frame to grayscale: I(t-1)
3. Calculate absolute difference: D(t) = |I(t) - I(t-1)|
4. Apply threshold: B(t) = D(t) > T
5. Apply morphological operations to reduce noise
6. Find contours and convert to bounding boxes

**Mathematical Formulation:**
Grayscale conversion:
Gray(x,y) = 0.299R(x,y) + 0.587G(x,y) + 0.114*B(x,y)
Absolute difference:
D(x,y,t) = |I(x,y,t) - I(x,y,t-1)|
Binary mask:
B(x,y,t) = {
255  if D(x,y,t) > T
0    otherwise
}

### Advantages
- Simple and fast
- No training required
- Low computational cost
- Works well for fast-moving objects

### Limitations
- Cannot detect stationary objects
- Sensitive to lighting changes
- Produces fragmented detections
- High false positive rate
- Camera shake causes false detections

### Implementation Details

**Parameters:**
- `THRESHOLD`: Intensity difference threshold (25-35 recommended)
- `MIN_CONTOUR_AREA`: Minimum object size (1500-2500 pixels)
- `EROSION_ITERATIONS`: Noise removal (2-3)
- `DILATION_ITERATIONS`: Object reconstruction (3-4)

**From-Scratch Components:**
- Grayscale conversion using weighted RGB formula
- Absolute difference using NumPy broadcasting
- Binary thresholding with array masking
- IoU calculation for evaluation

## Evaluation Metrics

### Intersection over Union (IoU)

Measures bounding box overlap between prediction and ground truth.
IoU = Area of Intersection / Area of Union

**Implementation:**
```python
intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
union = area1 + area2 - intersection
iou = intersection / union
