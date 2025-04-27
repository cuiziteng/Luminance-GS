### 📖: Enviroment setup (based on [GS-Splat](https://github.com/nerfstudio-project/gsplat)):
```
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

### 📖: Dataset setup:
Our Luminance-GS is evluated on 3 datasets (LOM-lowlight, LOM-overexposure and MipNeRF360-varying).

For **LOM** dataset (lowlight and overexposure), please refer to [Aleth-NeRF](https://github.com/cuiziteng/Aleth-NeRF).

For **MipNeRF360-varying** dataset, please download from [g-drive link (8.47GB)](https://drive.google.com/file/d/1x0EHT5z9ZrA6JV7-y8A8ijQNFCRTjVMW/view?usp=sharing).

***Note***: MipNeRF360-varying is a synthesized dataset based on [MipNeRF360 dataset](https://jonbarron.info/mipnerf360/), featuring 360° views and inconsistent lighting across images, making NVS more challenging.

Then datasets should be set up as (under this folder):

```
-- Luminance-GS
    -- LOM_full (For NVS under low-light and overexposure)
        -- bike
        -- buu
        -- chair
        -- shrub
        -- sofa 
```
