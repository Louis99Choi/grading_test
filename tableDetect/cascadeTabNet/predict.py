
# -*- coding: utf-8 -*-
#import sys, os
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

# Load model
config_file = './Config/cascade_mask_rcnn_hrnetv2p_w32_20e_v2.py'
checkpoint_file = './weights/epoch_13_Table_BankBoth_table_detection.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

### ./weights/epoch_24_General_Model_table_detection.pth
### ./weights/epoch_1_ICDAR13_table_detection.pth
### ./weights/epoch_14_ICDAR19(Track_A_Modern)table_detection.pth
### ./weights/epoch_17_Table_Bank_Word_table_detection.pth
### ./weights/epoch_14_Table_Bank_Latex_table_detection.pth
### ./weights/epoch_13_Table_BankBoth_table_detection.pth
### ./weights/epoch_36_ICDAR19(TrackB2Modern)table_structure_recognition.pth

# Test a single image
path_pdfImgDir = "../../pdfFolder/class/jpg/a_test/a_test_0.jpg"
path_result = "../../pdfFolder/class/jpg/a_test/a_test/result_table/res_a_test_0.jpg"

# Run Inference
result = inference_detector(model, path_pdfImgDir)

# Visualization results
#show_result_pyplot(path_pdfImgDir, result, model.CLASSES)

# or save the visualization results to image files
model.show_result(path_pdfImgDir, result, out_file=path_result)