
## test video
```
python tools/test_video.py --cfg configs/QNRF_final.py --pretrained pretrained/QNRF_mae_77.8_mse_138.0.pth --video [video_path]
```

## test image
```
python tools/test_img.py --cfg configs/QNRF_final.py --pretrained pretrained/QNRF_mae_77.8_mse_138.0.pth --img [img_path]
```

## torchscript
```
python tools/torchscript.py --cfg configs/QNRF_final.py --checkpoint pretrained/QNRF_mae_77.8_mse_138.0.pth
```