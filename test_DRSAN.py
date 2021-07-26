###########################################################################
# Created by: Yuxuan Zheng
# Email: yxzheng24@163.com
# Testing code for DRSAN proposed in the paper titled "Deep Residual Spatial Attention Network for Hyperspectral Pansharpening"

# Citation
# Y. Zheng, J. Li, Y. Li, Y. Shi and J. Qu, "Deep Residual Spatial Attention Network for Hyperspectral Pansharpening," 
# IGARSS 2020 - 2020 IEEE International Geoscience and Remote Sensing Symposium, Waikoloa, HI, USA, 2020, pp. 2671-2674, doi: 10.1109/IGARSS39084.2020.9323620.
###########################################################################

from __future__ import absolute_import, division
import numpy as np
from keras.models import Model
import h5py
import scipy.io as sio
import load_mat73

from train_DRSAN import eval_drsan

if __name__ == "__main__":

    inputs, outputs = eval_drsan()
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights('./models/model_drsan_pa.h5', by_name=True)
    
    for i in range(7):
        
        ind = i+1
        
        print ('processing for %d'%ind)

        # load pre-upsampled Hu for the subsequent summation
        data_Hu = load_mat73.loadmat('./data_process/Pavia_Hu/test_7Hu/Hu_%d.mat'%ind)
        
        data_Hu = np.float64((data_Hu['I_HS']))

        # load the input (Isp) for testing
        data = h5py.File('./data_process/Pavia_SpanSlhs/test_7Isp/%d.mat'%ind)
        
        data = np.transpose(data['S_panlhs'])
        
        data = np.expand_dims(data,0)
    
        data_res = model.predict(data, batch_size=1, verbose=1)
        
        data_res = np.reshape(data_res, (160, 160, 102))
        
        data_res = np.array(data_res, dtype=np.float64)
        
        # Obtaining the fused HSI Hf by element-wise summation
        data_fus = data_Hu + data_res
        
        sio.savemat('./get_pa7Hf/getHf_%d.mat'%ind, {'Hf': data_fus})
    