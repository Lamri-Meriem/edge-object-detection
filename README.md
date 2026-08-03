# Edge-Optimized Object Detection & Live Counting Dashboard

A lightweight person/vehicle detection tool built and optimized for edge deployment
inspired by Qareeb's edge-computing approach (on-device AI, no cloud dependency).

## What it does
Processes a video frame-by-frame using a pretrained YOLOv8n model, counts detected 
objects per frame, and logs the results to a CSV file a small building block for 
surveillance/monitoring use cases (similar to Q-Vision).

## Why edge optimization matters
Most CV demos stop at "does it detect things accurately." For real-world edge deployment 
(cameras on-site, no powerful GPU, no cloud connection), speed and model size matter just 
as much as accuracy. This project benchmarks three versions of the same model to show 
that tradeoff concretely.

## Results

| Model version         | Size    | Total time (520 frames) | FPS   |
|------------------------|---------|---------------------------|-------|
| PyTorch (yolov8n.pt)   | 6.24 MB | 312.1s                    | ~1.67 |
| ONNX (float32 export)  | 12.26 MB| 318.7s                    | ~1.63 |
| Quantized ONNX (int8)  | 3.34 MB | 284.3s                    | ~1.83 |

**Key finding:** exporting to ONNX alone did *not* improve speed or size — it actually 
increased both slightly, since plain export just changes file format, not precision. 
The real gain came from **quantization** (converting weights from float32 to int8), 
which cut model size by ~46% and gave a modest speed improvement. On CPU, quantization 
gains are smaller than what you'd typically see on ARM/edge hardware this is an honest 
limitation of testing on a laptop rather than actual edge hardware.

## Live counting dashboard
`dashboard.py` runs detection frame-by-frame using the quantized model and logs a 
per-frame object count to `detection_log.csv` real output data rather than just a 
console demo.

**Known limitation:** the model occasionally misclassifies objects in the scene as 
"boat" in frames with multiple detections a reminder that generic pretrained models 
need domain-specific fine-tuning for production use, which would be a natural next step.

## Setup
```bash
pip install ultralytics onnx onnxruntime onnxslim
```
## Usage
```python
python dashboard.py
```

## Tech stack
Python, Ultralytics YOLOv8, ONNX, ONNX Runtime (quantization)

## Next steps
- Fine-tune on domain-specific data (surveillance or agricultural footage)
- Add zone-based intrusion/access-control logic
- Test on actual edge hardware (Raspberry Pi / Jetson) for realistic FPS numbers
