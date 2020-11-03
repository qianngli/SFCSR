# SFCSR

**This is an implementation of  Hyperspectral Image Super-Resolution Using Spectrum and Feature Context.**

Dataset
------
**Three public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), [Harvard](http://vision.seas.harvard.edu/hyperspec/explore.html "Harvard"), [Foster](https://personalpages.manchester.ac.uk/staff/d.h.foster/Local\_Illumination\_HSIs/Local\_Illumination\_HSIs\_2015.html "Foster"), are employed to verify the effectiveness of the  proposed MCNet. Since there are too few images in these datasets for deep learning algorithm, we augment the training data. With respect to the specific details, please see the implementation details section.**

**Moreover, we also provide the code about data pre-processing in floder [data pre-processing](https://github.com/qianngli/MCNet/tree/master/data_pre-processing "data pre-processing"). The floder contains three parts, including training set augment, test set pre-processing, and band mean for all training set.**

Requirement
---------
**python 2.7, Pytorch 0.3.1, cuda 9.0**

Train and Test
--------
**The ADAM optimizer with beta_1 = 0.9, beta _2 = 0.999 is employed to train our network.  The learning rate is initialized as 10^-4 for all layers, which decreases by a half at every 35 epochs.**

**You can train or test directly from the command line as such:**

###### # python train.py --cuda --datasetName CAVE  --upscale_factor 4
###### # python test.py --cuda --model_name checkpoint/model_4_epoch_200.pth

Result
--------
**To qualitatively measure the proposed MCNet, three evaluation methods are employed to verify the effectiveness of the algorithm, including  peak signal-to-noise ratio (PSNR), structural similarity (SSIM), and spectral angle mapping (SAM).**


| Scale  |  CAVE |  Harvard |  Pavia Centre |
| :------------: | :------------: | :------------: | :------------: | 
|  x2 |  45.300 / 0.9739 / 2.217 | 46.342 / 0.9830 / 1.880  | 58.859 / 0.9988 / 4.052 | 
|  x3 |  41.198 / 0.9524 / 2.794  |  42.778 / 0.9632 / 2.203 | 54.575 / 0.9967 / 5.046  |   
|  x4 | 39.192 / 0.9321 / 3.221 |  40.077 / 0.9373 / 2.407 | 52.215 / 0.9939 / 5.618  | 

--------
If you has any questions, please send e-mail to liqmges@gmail.com.

