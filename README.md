Deploy RealESRGAN to RK3588S with single python script and rknn model.

#Usage
`python srknn.py image_path save_path`
#Convert 
Real-ESRGAN pytorch model to onnx: 
1. Clone and install Real-ESRGAN repo
2. Run command `python3 pytorch2onnx.py --input <path-to-your-realesrgan-model> --params`
Onnx to rknn:
`from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx("realesrgan.onnx")
rknn.build(do_quantization=False)
rknn.export_rknn("realesrgan.rknn")
`
