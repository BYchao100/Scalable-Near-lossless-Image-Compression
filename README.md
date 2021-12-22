# Scalable Near-lossless Image Compression

Source code of our CVPR 2021 paper "Learning Scalable ℓ<sub>∞</sub>-constrained Near-lossless Image Compression via Joint Lossy Image and Residual Compression".

## Usage
To run the code requires Python 3.6 and TensorFlow 1.15.
```
pip install tensorflow-compression==1.3
pip install range-coder
```
The `main.py` provides `compress` and `decompress` functions, and an example to encode/decode `./test_patch/kodim05_p128.png`.
Users can test their own images.

The model `ckp_003` can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1bssjYQoK5_NTpLbwapRKLQ), with access code `snic`.

Please note: the current implementation is not optimized for speed. Residual compression is slow. We are working on the fast version :computer:.

## Citation

```
@InProceedings{Bai_2021_SNIC,
  title={Learning Scalable $\ell_\infty$-constrained Near-lossless Image Compression via Joint Lossy Image and Residual Compression},
  author={Bai, Yuanchao and Liu, Xianming and Zuo, Wangmeng and Wang, Yaowei and Ji, Xiangyang},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```
