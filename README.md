# multitenant-classification
This repository contains the data processing flow for our multi-tenant power side-channel classification process. This includes scripts neccesary for parsing raw sensor output, preprocessing data, and finally training and testing a classifier network. The details of the classification pipeline can be found in our corresponding paper. 

The datasets used in our paper are also contianed linked to this repository through git lfs. The following steps will assume you have pulled the data sets to your local environment, and that you are familar with these steps as we have outlined them in our paper. 

The raw data returned from our Tunable Dual-Polarity Time-to-Digital Converter is first pre-processed into segments which the fft is computed on. The first step in this process is to decompress the files held by git lfs, which can be done in the `power_traces` directory with:
```
for f in *.tar.gz; do tar -xvf "$f"; done
```
All 100 traces captured for each 13 applications for each of the 5 test boards for each of the 6 sensor tuning arrangments discussed in our paper are then availible in raw form. To begin the pre-processing into segements and then ffts, run the following:
```
python3 fft.py
```
Once the ffts are computed we ready to train and test our classifier. The default configuration is a 13-way classification accuracy on 5-board,leave-one-out 
cross-validation. This can be re-configured however one pleases in the configuration files. You may regenerate the training and classification used in our paper with the following:

### Rising Transition Min Theta and Min Phi
```
python3 multitenant_fpga_net.py -c configs/pos-minphi-mintheta-config.yaml
```

### Falling Transition Max Theta and Min Phi
```
python3 multitenant_fpga_net.py -c configs/neg-minphi-maxtheta-config.yaml
```

### Falling Transition Max Theta and Max Phi
```
python3 multitenant_fpga_net.py -c configs/neg-maxphi-maxtheta-config.yaml
```

### Falling Transition Max Theta and Max Phi with Background Subtraction
```
python3 multitenant_fpga_net.py -c configs/neg-backsub-maxphi-maxtheta-config.yaml
```

### Rising Transition Max Theta and Max Phi with Background Subtraction
```
python3 multitenant_fpga_net.py -c configs/pos-backsub-maxphi-maxtheta-config.yaml
```
