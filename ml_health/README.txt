To launch the KDD results reproducibility scripts

1. Download data sets
samsung: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
video: http://archive.ics.uci.edu/ml/datasets/Online+Video+Characteristics+and+Transcoding+Time+Dataset
telco: http://mldata.org/repository/data/viewslug/realm-im2015-vod-traces

2. Prepare data sets: In the `ml_health/univariate/tests` folder, run 
a. samsung_data_creation.py
b. telco_ds_double_algo_exp.py
c. video_dataset_creation.py

these scripts accept two arguments "--path" and "--output" (path being the location of the data and output being
the desired location of the prepared datasets)

3. Run the kdd results scripts in `ml_health/univariate/tests`

python3 KDD2019.py --samsung-path "/tmp/samsung/" --video-path "/tmp/video" --telco-path "/tmp/telco"
