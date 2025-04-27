### ☀️: Enviroment setup (based on [GS-Splat](https://github.com/nerfstudio-project/gsplat)):
```
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

### ☀️: Dataset setup:
Our Luminance-GS is evluated on 3 datasets (LOM-lowlight, LOM-overexposure and MipNeRF360-varying).

For **LOM** dataset (lowlight and overexposure), please refer to [Aleth-NeRF](https://github.com/cuiziteng/Aleth-NeRF).

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
        -- NeRF_360 (For NVS under vary-exposure)
            -- bicycle
            -- bonsai
            -- counter
            -- ... (total 7 scenes)
```

### ☀️: Model Training:



### ☀️: Model Evaluation:
For the model evaluation, we provide the pretrained weights and rendering results in G-drive and BaiduYun(百度云网盘) as follow:

| LOM (low-light) | LOM (overexposure) | MipNeRF360-varying 1 | MipNeRF360-varying 2 | MipNeRF360-varying 3 | 
|  ---- |  ---- | ---- | ---- | ----  | 
|  [G-drive](https://drive.google.com/file/d/1Za6WbdZyMfJYPTziDvJj-hmLpQ5sh7TD/view?usp=sharing) | [G-drive](https://drive.google.com/file/d/1bF-tKc_UYRYfRcMvsoe4BzzKmnFVvbPM/view?usp=sharing)  | [G-drive](https://drive.google.com/file/d/1ON4rWEeU578axI5aMbXDFvWig17HO7gh/view?usp=sharing) | [G-drive](https://drive.google.com/file/d/1fkpVjBlsbT4PX73rhYixwig8XUIReg3t/view?usp=sharing)  |  [G-drive](https://drive.google.com/file/d/1MJK-FX3qDDwyj3fXskTFp_Bca-RNRTok/view?usp=sharing) |
|  [百度云(密码 1111)](https://pan.baidu.com/s/1BxaKkQ_7vr1A_AbLFhoYAg)   | [百度云(密码 1111)](https://pan.baidu.com/s/1X8ysXnO4MFGJP_bpPjtmYQ)  | [百度云(密码 1111)](https://pan.baidu.com/s/1wHdbB4GJ9zfixf2NUnijyA) |  [百度云(密码 1111)](https://pan.baidu.com/s/1jflRw246RPwNAgqhpDx2_w) |  [百度云(密码 1111)](https://pan.baidu.com/s/1WbQ1tcJP1xg3F-fdRK-Saw) |



