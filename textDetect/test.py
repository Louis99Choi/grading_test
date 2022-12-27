"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
#import sys
import shutil
import os
import time
import argparse

import torch
#import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

#from PIL import Image

import cv2
#from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
#import json
#import zipfile

from craft import CRAFT
from collections import OrderedDict

##### set PATH ####
path_pdfImgDir = "../pdfFolder/class/jpg/test/"
path_result = path_pdfImgDir + "result_detect/"


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='./weights/craft_ic15_20k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.0005, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.23, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.63, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default=path_pdfImgDir, type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='./weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


result_folder = path_result

# remove eixist dir ##################################################
if os.path.exists(result_folder):
    shutil.rmtree(result_folder)
# mkdir result_folder
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

print("image_list : ", image_list) ##########


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    #print("score_text : (shape : ", score_text.shape," ) \n") ###########
    #print("\nscore_link : (shape : ", score_link.shape," ) \n") ###########

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()


    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]


    ################### top, left > bottom, right priority sorting ######################
    #flat_boxes = np.array(boxes).reshape((-1, 8)) # shape : (n, 4, 2) > (n, 8)

    #left_sorted_boxes = flat_boxes[flat_boxes[:, 0].argsort()]
    #top_left_sorted_boxes = left_sorted_boxes[left_sorted_boxes[:, 1].argsort()]
    #sorted_boxes = top_left_sorted_boxes.reshape((-1, 4, 2))
    ###################################################################################

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        #print("image : (image.shape : ( ", image.shape, ", type : ", type(image), " )\n") ############

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        #print("\nbboxes.shape : ", bboxes.shape) ###############

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        # save marking img(.jpg)
        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder, texts=np.array(bboxes).astype(np.int32))

        # save each answer img(.png) ############################
        cutAnswerDir = result_folder + "{}".format(filename) + "_answer/"
        if not os.path.isdir(cutAnswerDir):
            os.mkdir(cutAnswerDir)

        # index for saving answer piece to .png
        answerNum = 0
        pre_bottom = 0
        lostNum = 0

        for i, box in enumerate(bboxes):

            left = int(min(box[0][0], box[3][0]))
            right = int(max(box[1][0], box[2][0]))
            top = int(min(box[0][1], box[1][1]))
            bottom = int(max(box[2][1], box[3][1]))

            height = bottom-top
            width = right-left

            if( ((left >= 1250) and (left <= 4400)) and ((top >= 1700) and (top <= 6400)) ) :

                # check probability of lost answer
                if( (answerNum !=0) and ((top - pre_bottom) > 230) ):
                    lostNum = lostNum + 1
                    print("\n!!!!! You must check answer { %d.png }, it would be lost !!!!!\n"
                          "\tLost Num : %d\n" %(answerNum,lostNum))
                    answerNum = answerNum + 1

                # Check probability of big box that contain previous answer
                if(height > 300) :
                    print("!!!!! Check this answer { %d.png }, it would contain previous answer !!!!!" %(answerNum+1))
                    answerNum = answerNum + 1

                pre_bottom = bottom


                print("... saving answer {", answerNum, "\b.png}...")
                imgCopy = image[:,:,::-1]

                # save answer of ROI to .png from PIL Image
                cv2.imwrite(cutAnswerDir + "{}".format(answerNum) + ".png",
                            imgCopy[top-20:bottom+20, left-20:right+20, : ].copy())

                # answerNum++
                answerNum = answerNum + 1


    print("elapsed time : {}s".format(time.time() - t))
