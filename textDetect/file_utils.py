# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:

            poly_infoSector = np.array([[600, 400],[4850, 400],[4850, 1800],[600, 1800]])
            poly_questionSector = np.array([[600, 1700],[1290,1700],[1290, 6300],[600,6300]])
            poly_answerSector = np.array([[1250, 1700],[4400, 1700],[4400, 6300],[1250, 6300]])

            cv2.polylines(img, [poly_infoSector], True, color=(255, 0, 0), thickness=4)
            cv2.polylines(img, [poly_questionSector], True, color=(255, 255, 0), thickness=4)
            cv2.polylines(img, [poly_answerSector], True, color=(255, 0, 0), thickness=4)

            for i, box in enumerate(boxes):
                #print("box : ", box) ##########################################

                poly = np.array(box).astype(np.int32).reshape((-1))
                #print("poly : ", poly) ##########################################

                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                poly = poly.reshape(-1, 2)
                #print("poly.reshape(-1, 2) : \n", poly) ##########################################

                #print("poly.reshape((-1, 1, 2)) : \n", poly.reshape((-1, 1, 2)))

                # draw ROI
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1)

                # cut ROI to .png


                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(i), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=0.5)
                    cv2.putText(img, "{}".format(i), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=0.5)



        # Save result image
        cv2.imwrite(res_img_file, img)

