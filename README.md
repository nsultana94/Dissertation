# COMP0184 Code 

Code for COMP0184 Final Year Project for semantic segmentation network for CaDIS dataset. 

The networks are listed in `networks.py` which includes the U-net structure, convolutional LSTM cell and overall architecture model.

The model is trained using the file `train.py` which includes both training and evaluation function.

The model is tested using the `test.py` file on the testset.

The sequential image data is loaded for training and testing using the `dataloader.py` file.

All training, data loading and testing functions can be found in the U-net folder for training the baseline U-net model.

The notebook `optical_flow_generation.ipynb` describes the process of producing pseudolabels for a sequence of frames from the ground truth mask using the RAFT model.