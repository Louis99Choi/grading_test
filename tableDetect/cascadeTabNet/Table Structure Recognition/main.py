import border
from mmdet.apis import inference_detector, init_detector
import cv2
from Functions.blessFunc import borderless
import lxml.etree as etree
import glob


# Load model
config_file = '../Config/cascade_mask_rcnn_hrnetv2p_w32_20e_v2.py'
checkpoint_file = '../weights/epoch_13_Table_BankBoth_table_detection.pth'
#model = init_detector(config_file, checkpoint_file, device='cuda:0')

### ./weights/epoch_24_General_Model_table_detection.pth
### ./weights/epoch_1_ICDAR13_table_detection.pth
### ./weights/epoch_14_ICDAR19(Track_A_Modern)table_detection.pth
### ./weights/epoch_17_Table_Bank_Word_table_detection.pth
### ./weights/epoch_14_Table_Bank_Latex_table_detection.pth
### ./weights/epoch_13_Table_BankBoth_table_detection.pth
### ./weights/epoch_36_ICDAR19(TrackB2Modern)table_structure_recognition.pth

# Test a single image
path_pdfImgDir = "../../../pdfFolder/class/jpg/a_test/a_test_0.jpg"
path_result = "../../../pdfFolder/class/jpg/a_test/result_table/"


############ To Do ############
image_path = path_pdfImgDir
xmlPath = path_result

config_fname = config_file
#checkpoint_path = "path to checkpoint directory"
#epoch = 'epoch_file.name'
##############################


model = init_detector(config_fname, checkpoint_file)

# List of images in the image_path
imgs = glob.glob(image_path)
for i in imgs:
    result = inference_detector(model, i)
    res_border = []
    res_bless = []
    res_cell = []
    root = etree.Element("document")
    ## for border
    for r in result[0][0]:
        if r[4]>.85:
            res_border.append(r[:4].astype(int))
    ## for cells
    for r in result[0][1]:
        if r[4]>.85:
            r[4] = r[4]*100
            res_cell.append(r.astype(int))
    ## for borderless
    for r in result[0][2]:
        if r[4]>.85:
            res_bless.append(r[:4].astype(int))

    ## if border tables detected 
    if len(res_border) != 0:
        ## call border script for each table in image
        for res in res_border:
            try:
                root.append(border(res,cv2.imread(i)))  
            except:
                pass
    if len(res_bless) != 0:
        if len(res_cell) != 0:
            for no,res in enumerate(res_bless):
                root.append(borderless(res,cv2.imread(i),res_cell))

    myfile = open(xmlPath+i.split('/')[-1][:-3]+'xml', "w")
    myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    myfile.write(etree.tostring(root, pretty_print=True,encoding="unicode"))
    myfile.close()