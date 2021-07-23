# IGARSS2020_DRSAN

## [Deep Residual Spatial Attention Network for Hyperspectral Pansharpening](https://ieeexplore.ieee.org/document/9323620)

**Tensorflow implementation of our proposed DRSAN method for hyperspectral pansharpening.**

![Overview](https://github.com/yxzheng24/IGARSS2020_DRSAN/blob/main/Flowchart_IGARSS20.png "Overview of the proposed method for hyperspectral pansharpening.")

## Usage
Here we take the experiments conducted on the [Pavia Center](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_scene) data set as an example for illustration.

*   Training:
1.   Download the [Pavia Center](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_scene) scene, partition the top-left 960 × 640 × 102 part into 24 cubic-patches of size 160 × 160 × 102 with no overlap, and put the subimages into the __./data_process/pavia_subimgs/__ folder. Note that you can also download the Pavia Center data set (subimages in .mat format) from the Baidu Cloud links: https://pan.baidu.com/s/1QBYLFZpS5VHnx1A3Dkz4Dg (Access Code: prz6).
2.   

## Citation
Y. Zheng, J. Li, Y. Li, Y. Shi and J. Qu, "Deep Residual Spatial Attention Network for Hyperspectral Pansharpening," IGARSS 2020 - 2020 IEEE International Geoscience and Remote Sensing Symposium, Waikoloa, HI, USA, 2020, pp. 2671-2674, doi: 10.1109/IGARSS39084.2020.9323620.

    @INPROCEEDINGS{Zheng2020IGARSS,
    author={Y. {Zheng} and J. {Li} and Y. {Li} and Y. {Shi} and J. {Qu}},
    booktitle={Proc. IEEE Int. Geosci. Remote Sens. Symp. (IGARSS)}, 
    title={Deep Residual Spatial Attention Network for Hyperspectral Pansharpening}, 
    year={2020},
    pages={2671-2674},
    doi={10.1109/IGARSS39084.2020.9323620}}


## Contact Information
Yuxuan Zheng is with the State Key Laboratory of Integrated Services Networks, School of Telecommunications Engineering, Xidian University, Xi’an 710071, China (e-mail: yxzheng24@163.com).
