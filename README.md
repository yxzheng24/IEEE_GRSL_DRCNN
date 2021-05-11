# IEEE_GRSL_DRCNN

## [Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening](https://ieeexplore.ieee.org/document/8874962)

## [Hyperspectral Pansharpening Based on Guided Filter and Deep Residual Learning](https://ieeexplore.ieee.org/document/8899015)

![Framework](https://github.com/yxzheng24/IEEE_GRSL_DRCNN/blob/main/Framework_GRSL20.png "Framework of the proposed method for hyperspectral pansharpening.")

## Usage
Here we take the experiments conducted on the [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/) data set as an example for illustration.

*  Training:
1.  Download the [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/) data set and put the data into the "./data_process/cave_ref" folder, which are served as the reference images.
2.  Run "get_hspan_cave.m" to obtain LR-HSI and HR-PAN images.
3.  Run "get_Hini_Hres_cave.m" to genertate the initialized HSI and the residual HSI.
4.  Randomly select 22 HSI pairs from "cave_Hini" and "cave_Hres" folders to form the training set.
5.  Run "get_traindata_h5.m" to produce the HDF5 file for training.
6.  Run *train_DRCNN.py*.

## Citation
(1) Y. Zheng, J. Li, Y. Li, K. Cao and K. Wang, "Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening," IEEE Geoscience and Remote Sensing Letters, vol. 17, no. 8, pp. 1435-1439, Aug. 2020, doi: 10.1109/LGRS.2019.2945424.

(2) Y. Zheng, J. Li and Y. Li, "Hyperspectral Pansharpening Based on Guided Filter and Deep Residual Learning," 2019 IEEE International Geoscience and Remote Sensing Symposium, Jul. 2019, pp. 616-619, doi: 10.1109/IGARSS.2019.8899015.

    @ARTICLE{Zheng2020GRSL,
    author={Y. {Zheng} and J. {Li} and Y. {Li} and K. {Cao} and K. {Wang}},
    journal={IEEE Geosci. Remote Sens. Lett.}, 
    title={Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening}, 
    year={2020},
    month={Aug.},
    volume={17},
    number={8},
    pages={1435-1439},
    doi={10.1109/LGRS.2019.2945424}}
    
    @INPROCEEDINGS{Zheng2019IGARSS,
    author={Zheng, Yuxuan and Li, Jiaojiao and Li, Yunsong},
    booktitle={2019 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2019)}, 
    title={Hyperspectral Pansharpening Based on Guided Filter and Deep Residual Learning}, 
    year={2019},
    month={Jul.},
    pages={616-619},
    doi={10.1109/IGARSS.2019.8899015}}

## Contact Information
Yuxuan Zheng is with the State Key Laboratory of Integrated Services Networks, School of Telecommunications Engineering, Xidian University, Xi’an 710071, China (e-mail: yxzheng24@163.com).
