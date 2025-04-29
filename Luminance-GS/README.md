<br/>

***"The straight line belongs to men, the curved line belongs to God"
\
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp;-- Antoni Gaudi***

<br/>

### ☀️: Enviroment setup (follow [GS-Splat](https://github.com/nerfstudio-project/gsplat)):

Clone our project and then install ***Pytorch*** first and then:

```
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

### ☀️: Dataset setup:
Our Luminance-GS is evluated on 3 datasets (LOM-lowlight, LOM-overexposure and MipNeRF360-varying).

For **LOM** dataset (lowlight and overexposure), please refer to [Aleth-NeRF](https://github.com/cuiziteng/Aleth-NeRF), download link [here](https://drive.google.com/file/d/1orgKEGApjwCm6G8xaupwHKxMbT2s9IAG/view).

For **MipNeRF360-varying** dataset, please download from [g-drive link (8.47GB)](https://drive.google.com/file/d/1x0EHT5z9ZrA6JV7-y8A8ijQNFCRTjVMW/view?usp=sharing).

***Note***: MipNeRF360-varying is a synthesized dataset based on [MipNeRF360 dataset](https://jonbarron.info/mipnerf360/), featuring 360° views and inconsistent lighting across images, making NVS more challenging.

Then datasets should be set up as (under this folder):

```
-- Luminance-GS
    -- data
        -- LOM_full (For NVS under low-light and overexposure)
            -- bike
            -- buu
            -- chair
            -- shrub
            -- sofa
        -- NeRF_360 (For NVS under vary-exposure), we only provide downscale ratio 8 for efficiency
            -- bicycle
                -- images
                -- images_8
                -- images_8_variance
                -- sparse
                -- ...
            -- bonsai
            -- counter
            -- ... (total 7 scenes)
```

### ☀️: Model Training:

1. 
```
cd examples
```

2.
For LOM dataset low-light ("buu" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --result_dir (place you save weights & results)
```

For LOM dataset over-exposure ("buu" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name over_exp --result_dir (place you save weights & results)
```

For MipNeRF360-varying dataset varying exposure ("bicycle" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/NeRF_360/bicycle --exp_name variance --data_factor 8 --result_dir (place you save weights & results)
```

### ☀️: Model Evaluation:
For the model evaluation, we provide the pretrained weights and rendering results in G-drive and BaiduYun(百度云网盘) as follow:

***Note***: Results of MipNeRF360-varying different scenes have been divided into 3 folders because 百度云 requires a paid upload :)

| LOM (low-light) | LOM (overexposure) | MipNeRF360-varying 1 | MipNeRF360-varying 2 | MipNeRF360-varying 3 | 
|  ---- |  ---- | ---- | ---- | ----  | 
|  [G-drive](https://drive.google.com/file/d/1Za6WbdZyMfJYPTziDvJj-hmLpQ5sh7TD/view?usp=sharing) | [G-drive](https://drive.google.com/file/d/1bF-tKc_UYRYfRcMvsoe4BzzKmnFVvbPM/view?usp=sharing)  | [G-drive](https://drive.google.com/file/d/1ON4rWEeU578axI5aMbXDFvWig17HO7gh/view?usp=sharing) | [G-drive](https://drive.google.com/file/d/1fkpVjBlsbT4PX73rhYixwig8XUIReg3t/view?usp=sharing)  |  [G-drive](https://drive.google.com/file/d/1MJK-FX3qDDwyj3fXskTFp_Bca-RNRTok/view?usp=sharing) |
|  [百度云(密码 1111)](https://pan.baidu.com/s/1BxaKkQ_7vr1A_AbLFhoYAg)   | [百度云(密码 1111)](https://pan.baidu.com/s/1X8ysXnO4MFGJP_bpPjtmYQ)  | [百度云(密码 1111)](https://pan.baidu.com/s/1wHdbB4GJ9zfixf2NUnijyA) |  [百度云(密码 1111)](https://pan.baidu.com/s/1jflRw246RPwNAgqhpDx2_w) |  [百度云(密码 1111)](https://pan.baidu.com/s/1WbQ1tcJP1xg3F-fdRK-Saw) |


Compare to the training code, directly add the ckpt file provided in above links to make evaluation:

LOM dataset low-light "buu" scene for example:
```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --result_dir (place you save results) --ckpt (place of weights)
```


### ☀️: Notice and Others:

***1***. Directly refer to [thie file](https://github.com/cuiziteng/Luminance-GS/blob/main/Luminance-GS/examples/simple_trainer_ours.py) to check the details of Luminance-GS model structure.

***2***. Please note that if you want render a nice video results, you should change rendering views [line 1023 & line 1024 for view and interpolate selection](https://github.com/cuiziteng/Luminance-GS/blob/e963cb1bcd285e5416383a9d034d5e89fb9c0d3a/Luminance-GS/examples/simple_trainer_ours.py#L1023) and [line 1061 for speed](https://github.com/cuiziteng/Luminance-GS/blob/e963cb1bcd285e5416383a9d034d5e89fb9c0d3a/Luminance-GS/examples/simple_trainer_ours.py#L1061).

***3***. We deeply thanks to [GS-Splat](https://github.com/nerfstudio-project/gsplat) for their nice codebase :).

### ☀️: Some Great Co-current Works (same direction):

**sRGB-based:** 

PG 2024 [Gaussian in the Dark: Real-Time View Synthesis From Inconsistent Dark Images Using Gaussian Splatting](https://arxiv.org/abs/2408.09130)

IROS 2024 [DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration](https://tyz1030.github.io/proj/darkgs.html)

CVPR 2025 [lita-gs: illumination-agnostic novel view synthesis via reference-free 3d gaussian splatting and physical priors](https://arxiv.org/html/2504.00219v1)

Arxiv 2025 [LL-Gaussian: Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting for Novel View Synthesis](https://sunhao242.github.io/LL-Gaussian_web.github.io/)

**RAW-based:**

NIPS 2024 [Lighting Every Darkness with 3DGS: Fast Training and Real-Time Rendering for HDR View Synthesis](https://srameo.github.io/projects/le3d)

NIPS 2024 [From Chaos to Clarity: 3DGS in the Dark](https://arxiv.org/html/2406.08300v1)


