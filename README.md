# Deep Self-Dissimilarities as Powerful Visual Fingerprints
### Official pytorch implementation of the paper: "Deep Self-Dissimilarities as Powerful Visual Fingerprints"

<!-- <p align="center"> -->
  <!-- <img width="992" height="372" src="/figures/sampled.png"> -->
<!-- </p> -->

Please refer our [paper](https://proceedings.neurips.cc/paper/2021/hash/20479c788fb27378c2c99eadcf207e7f-Abstract.html) for more details.



## Citation
If you use this code for your research, please cite the paper:

```
@article{kligvasser2021deep,
  title={Deep Self-Dissimilarities as Powerful Visual Fingerprints},
  author={Kligvasser, Idan and Shaham, Tamar and Bahat, Yuval and Michaeli, Tomer},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## Code

### Clone repository
Clone this repository into any place you want.

```
git clone --recursive https://github.com/kligvasser/DSD
cd ./DSD
```

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code tested in PyTorch 1.8.1.

### Image quality assessment
The DSD property can be used as a powerful visual fingerprint. Specifically, as full-reference and no-reference image quality measures.
To run these measures:

```
cd ./image-quality-assessment
python3 main.py --input-dir <path-to-input-dir> --target-dir <path-to-target-dir> --metric-list dsd noref-dsd
```

### DSD regressor
You may train your own regressor for predicting the DSD values, in no-reference scenario:

```
cd ./regression
python3 main.py --root <path-to-sr-dataset> --model resnet_se --crop-size 80 --epochs 2000 --step-size 800
```

As well, you may find in ./regression/experiments/ some of the experiments which were conducted in the paper.

### Super-resolution
In addition, incorporating DSD as a loss function in super-resolution leads to results that are at least as photo-realistic as those obtained by GAN based methods, while not requiring adversarial training. Pretrained models are avaible at: [LINK](https://www.dropbox.com/s/8yvkb2vgw3105n1/pre-trained.zip?dl=0).



#### Data preperation
For the super-resolution task, the dataset should contains a low and high resolution pairs, in folder structure of:

```txt
train
├── img
├── img_x2
├── img_x4
val
├── img
├── img_x2
├── img_x4
```

You may prepare your own data by using the matlab script:

```
./super-resolution/scripts/matlab/bicubic_subsample.m
```

#### Train SRGAN x4 PSNR model
```
python3 main.py --root <path-to-dataset> --gen-model g_xsrgan --gen-model-config "{'scale':4, 'num_blocks':10}" --reconstruction-weight 1 --crop-size 40
```

#### Train xSRGAN x4 model
```
python3 main.py --root <path-to-dataset> --gen-model g_xsrgan --gen-model-config "{'scale':4, 'num_blocks':10}" --reconstruction-weight 1 --perceptual-weight 1 --recurrent-style-weight 100 --gen-to-load <path-to-psnr-model-pt>
```

#### Eval xSRGAN x4 model
```
python3 main.py --root <path-to-dataset> --gen-model g_xsrgan --gen-model-config "{'scale':4, 'num_blocks':10}" --evaluation --gen-to-load <path-to-pretrained-pt>