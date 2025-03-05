## TimeSeek: Multi-Task and Multi-Domain Pretraining for Generalized Time Series Analytics


## Requirements
To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
You can access the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/1x0SP6-vbB4JE7g0DQD5m_m9Y2UmwMdD5?usp=drive_link), then place the downloaded contents under ./data

## Pretraining Parameters
You can download the pre-trained parameters of TimeSeek from [Google Drive](https://drive.google.com/drive/folders/1KuxQDRgIcJHR2ODR2k0h5I9I62o1vXpI?usp=drive_link), then place the downloaded contents under ./checkpoints


## Quick Demos
1. Download datasets and place them under ./data
2. Download pre-trained parameters and place them under ./checkpoints
3. Evaluation. As a general time series analysis model, TimeSeek can perform forecasting, anomaly detection, and classification tasks on various target scenarios with zero-shot and few-shot settings. You can directly test on target datasets as the following scripts:
- Forecasting
  ```
   bash script/TimeSeek_forecasting/zero_shot.sh
  ```
- Anomaly Detection
  ```
   bash script/TimeSeek_anomaly_detection/zero_shot.sh
  ```
- Classification
  ```
   bash script/TimeSeek_classification/few_shot.sh 
  ```
