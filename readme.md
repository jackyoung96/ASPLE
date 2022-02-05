# Blind-spot detection through sound

This is the repository of the paper "Blind-spot detection through sound". The experimental data, code, and the result images are contained.

## Dataset

The dataset is from the reference study. It is the github repository of the paper "Hearing What You Cannot See: Acoustic Vehicle Detection Around Corners". The details are in the below link.  
https://github.com/tudelft-iv/occluded_vehicle_acoustic_detection

Dataset is in [**cls_features**](./cls_features) folder.  

- **out_multi.csv** file is the result of SRP-PHAT algorithm.  
- **ueye_stereo_vid.mp4** file is the obtained images by camera on the ego vehicle.  
- **extracted_features.csv** file is the result of SRP-PHAT algorithm of entire dataset that the one-second before appearing at line-of-sight.

## How to run the code

SRP-PHAT algorithm running first
```bash
sh featureExtract.sh
```

For classification test
```bash
sh classification_test.sh
```

For tracking test
```bash
sh tracking_test.sh
```

Get particle filter convergence variance image
```bash
python plotVarianceImg.py
```

- classification experiment results are in **log.txt** file  
- tracking experiment results are  


## Result images

---
---

### SA2 Environment
---
---
<center>
<h4>Left<h4>
</center>


|Data|Tracking result|Variance|
|:-:|:---------:|:---------:|
|1_01_0036|![image](./cls_features/SA2/left/1_01_0036/tracking.png)|![image](./cls_features/SA2/left/1_01_0036/variance.png)|
|1_01_0038|![image](./cls_features/SA2/left/1_01_0038/tracking.png)|![image](./cls_features/SA2/left/1_01_0038/variance.png)|
|1_01_0041|![image](./cls_features/SA2/left/1_01_0041/tracking.png)|![image](./cls_features/SA2/left/1_01_0041/variance.png)|
|1_01_0044|![image](./cls_features/SA2/left/1_01_0044/tracking.png)|![image](./cls_features/SA2/left/1_01_0044/variance.png)|

<center>
<h4>Right<h4>
</center>

|Data|Tracking result|Variance|
|:-:|:---------:|:---------:|
|3_01_0040|![image](./cls_features/SA2/right/3_01_0040/tracking.png)|![image](./cls_features/SA2/right/3_01_0040/variance.png)|
|3_01_0047|![image](./cls_features/SA2/right/3_01_0047/tracking.png)|![image](./cls_features/SA2/right/3_01_0047/variance.png)|
|3_01_0050|!![image](./cls_features/SA2/right/3_01_0050/tracking.png)|![image](./cls_features/SA2/right/3_01_0050/variance.png)|
|3_01_0051|![image](./cls_features/SA2/right/3_01_0051/tracking.png)|![image](./cls_features/SA2/right/3_01_0051/variance.png)|

---
---

### SB1 Environment
---
---
<center>
<h4>Left<h4>
</center>


|Data|Tracking result|Variance|
|:-:|:---------:|:---------:|
|1_02_0086|![image](./cls_features/SB1/left/1_02_0086/tracking.png)|![image](./cls_features/SB1/left/1_02_0086/variance.png)|
|1_02_0093|![image](./cls_features/SB1/left/1_02_0093/tracking.png)|![image](./cls_features/SB1/left/1_02_0093/variance.png)|
|1_02_0094|![image](./cls_features/SB1/left/1_02_0094/tracking.png)|![image](./cls_features/SB1/left/1_02_0094/variance.png)|

<center>
<h4>Right<h4>
</center>

|Data|Tracking result|Variance|
|:-:|:---------:|:---------:|
|3_02_0093|![image](./cls_features/SB1/right/3_02_0093/tracking.png)|![image](./cls_features/SB1/right/3_02_0093/variance.png)|
|3_02_0094|![image](./cls_features/SB1/right/3_02_0094/tracking.png)|![image](./cls_features/SB1/right/3_02_0094/variance.png)|
|3_02_0097|!![image](./cls_features/SB1/right/3_02_0097/tracking.png)|![image](./cls_features/SB1/right/3_02_0097/variance.png)|
|3_02_0099|![image](./cls_features/SB1/right/3_02_0099/tracking.png)|![image](./cls_features/SB1/right/3_02_0099/variance.png)|
|3_02_0108|![image](./cls_features/SB1/right/3_02_0108/tracking.png)|![image](./cls_features/SB1/right/3_02_0108/variance.png)|

---
---

### SB2 Environment

---
---
<center>
<h4>Left<h4>
</center>


|Data|Tracking result|Variance|
|:-:|:---------:|:---------:|
|1_03_0062|![image](./cls_features/SB2/left/1_03_0062/tracking.png)|![image](./cls_features/SB2/left/1_03_0062/variance.png)|
|1_03_0066|![image](./cls_features/SB2/left/1_03_0066/tracking.png)|![image](./cls_features/SB2/left/1_03_0066/variance.png)|
|1_03_0067|![image](./cls_features/SB2/left/1_03_0067/tracking.png)|![image](./cls_features/SB2/left/1_03_0067/variance.png)|
|1_03_0070|![image](./cls_features/SB2/left/1_03_0070/tracking.png)|![image](./cls_features/SB2/left/1_03_0070/variance.png)|
|1_03_0075|![image](./cls_features/SB2/left/1_03_0075/tracking.png)|![image](./cls_features/SB2/left/1_03_0075/variance.png)|
|1_03_0085|![image](./cls_features/SB2/left/1_03_0085/tracking.png)|![image](./cls_features/SB2/left/1_03_0085/variance.png)|

<center>
<h4>Right<h4>
</center>

|Data|Tracking result|Variance|
|:-:|:---------:|:---------:|
|3_03_0059|![image](./cls_features/SB2/right/3_03_0059/tracking.png)|![image](./cls_features/SB2/right/3_03_0059/variance.png)|
|3_03_0064|![image](./cls_features/SB2/right/3_03_0064/tracking.png)|![image](./cls_features/SB2/right/3_03_0064/variance.png)|
|3_03_0065|!![image](./cls_features/SB2/right/3_03_0065/tracking.png)|![image](./cls_features/SB2/right/3_03_0065/variance.png)|
|3_03_0082|![image](./cls_features/SB2/right/3_03_0082/tracking.png)|![image](./cls_features/SB2/right/3_03_0082/variance.png)|

---
---



## Authors

copyright  
Autonomous Robot Intelligence Lab, SNU  

Jae-Kyung Cho  
Seong-Woo Kim