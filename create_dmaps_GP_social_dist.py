#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:15:24 2018

@author: zq
"""
import json

import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
import scipy.io as sio
# from skimage.transform import downscale_local_mean
# import skimage.io
import os
import sys

import math
import time
import random
from random import shuffle
import pickle
import h5py
import glob

import camera_proj_Zhang as proj

# write in the json file
def writeInJason(v_projection, gt_file):
    #gt_file = 'jason_csv/via_view'+str(view_id)+'.json'
    with open(gt_file, 'r') as data_file:
        v_pmap_json = json.load(data_file)
    
    n = v_projection.shape[0]
    for i in range(n):
        print(i)
        pmapi = v_projection[i,:]
        fnum = int(pmapi[0])
        pid = int(pmapi[1])
        #whole_id = int(pmapi[2])
        x  = pmapi[2]
        y  = pmapi[3]
        height = pmapi[4]
    #    pnum = (x1-1)*h + y1 - 1
    #    pmapi2 = projection_table[pnum, :]
    #    x = pmapi2[2]
    #    y = pmapi2[3]
        fname_start = 'frame_0' + format(fnum, '03d')
        for key in v_pmap_json.keys():
            key2 = key.encode('ascii', 'ignore')
            if key2.startswith(fname_start):
                fname_full = key2
                break
        new_point ={
                    "shape_attributes":
                         {"name":"point",
                          "cx": x,
                          "cy": y,
                          "height": height}
                    #"region_attributes":{"whole_ID":0}
        }
        #print new_point
        v_pmap_json[fname_full]['regions'][str(pid)] = new_point
    return v_pmap_json



def Image_to_World(view, imgcoords):
    N = imgcoords.shape[0]
    wld_coords = []
    for i in range(N):
        imgcoords_i = imgcoords[i, :]

        Xi=imgcoords_i[0]
        Yi=imgcoords_i[1]
        Zw=imgcoords_i[2]

        XYw = proj.Image2World(view, Xi, Yi, Zw)
        wld_coords.append(XYw)
    wld_coords = np.asarray(wld_coords)
    return wld_coords

def World_to_Image(view, wldcoords):
    N = wldcoords.shape[0]
    imgcoords = []
    for i in range(N):
        wldcoords_i = wldcoords[i, :]

        Xw=wldcoords_i[0]
        Yw=wldcoords_i[1]
        Zw=wldcoords_i[2]

        XYi = proj.World2Image(view, Xw, Yw, Zw)
        imgcoords.append(XYi)
    imgcoords = np.asarray(imgcoords)
    return imgcoords

def generate_density_map(pmap, w, h):
    density_map = []
    img_id_all = np.unique(pmap[:, 0])

    for i in range(len(img_id_all)):
        img_id = img_id_all[i]
        print(img_id)

        img_dmap_i = np.zeros((h, w))
        pmap_i = pmap[pmap[:, 0]==img_id]
        for j in range(pmap_i.shape[0]):
            x = pmap_i[j, 2]
            y = pmap_i[j, 3]
            img_pmap_i = np.zeros((h, w))
            if x>=w or x<0 or y>=h or y<0:
                continue
            # if x<=50 and y<=50:
            #     continue
            else:
                img_pmap_i[y, x] = 1
                sigma = [10, 10]
                img_dmap_i += scipy.ndimage.filters.gaussian_filter(img_pmap_i, sigma, mode = 'reflect')
        density_map.append(img_dmap_i)
    density_map = np.asarray(density_map).astype('f')
    return density_map

def find_height_GP3(v_pmap, type):
    h = 1750
    # bbox = [-31, 29, -45, 25]  # assuming half-man plane/ 1.75/2*1000
    bbox = [352*0.8, 522*0.8]
    image_size = [380, 676]
    resolution_scaler = 76.25

    # camera file
    view1_groundPlane = 'view1'
    groundPlane_view2 = 'view2'
    groundPlane_view3 = 'view3'

    img_id = v_pmap[:, 0:1]
    pt_id = v_pmap[:, 1:2]
    cx1 = v_pmap[:, 2:3]*4
    cy1 = v_pmap[:, 3:4]*4

    # get GP coords x y
    num = v_pmap.shape[0]
    hi_all = np.ones([num, 1]) * h
    imgcoords = np.concatenate([cx1, cy1, hi_all], 1)

    if type == 1:
        view1_wc = Image_to_World(view1_groundPlane, imgcoords)
    if type == 2:
        view1_wc = Image_to_World(groundPlane_view2, imgcoords)
    if type == 3:
        view1_wc = Image_to_World(groundPlane_view3, imgcoords)

    view1_wc = np.asarray(view1_wc[:, :2])

    view1_wc_offset = view1_wc / resolution_scaler
    view1_wc_offset_x = view1_wc_offset[:, 0:1] + bbox[0] # * resolution_scaler
    # view1_wc_offset_x = view1_wc_offset_x/w1*w0
    view1_wc_offset_y = view1_wc_offset[:, 1:2] + bbox[1] # * resolution_scaler
    view1_wc_pmap = np.concatenate([img_id, pt_id, view1_wc_offset_x, view1_wc_offset_y, hi_all]
                                   , axis=1)
    view1_wc_pmap = np.asarray(view1_wc_pmap)

    return view1_wc_pmap


def find_height_GP2(v_pmap, type):
    ph = range(1560, 1960, 10)
    #bbox = [-31, 29, -45, 25]  # assuming half-man plane/ 1.75/2*1000
    bbox = [352*0.8, 522*0.8]
    image_size = [380, 676]
    resolution_scaler = 76.25

    # camera file
    view1_groundPlane = 'view1'
    groundPlane_view2 = 'view2'
    groundPlane_view3 = 'view3'

    img_id = v_pmap[:, 0:1]
    pt_id = v_pmap[:, 1:2]
    cxA = v_pmap[:, 2:3]*4
    cyA = v_pmap[:, 3:4]*4
    cxB = v_pmap[:, 4:5]*4
    cyB = v_pmap[:, 5:6]*4
    dist_error = []

    if type==1:
        viewA = groundPlane_view2
        viewB = groundPlane_view3
    if type==2:
        viewA = view1_groundPlane
        viewB = groundPlane_view3
    if type==3:
        viewA = view1_groundPlane
        viewB = groundPlane_view2

    view1_wc_pmap = []
    num = v_pmap.shape[0]

    for hi in ph:
        hi_all = np.ones([num, 1]) * hi
        imgcoords1 = np.concatenate([cxA, cyA, hi_all], 1)
        g1 = Image_to_World(viewA, imgcoords1)
        g1_w = g1[:, (0,1)]

        imgcoords2 = np.concatenate([cxB, cyB, hi_all], 1)
        g2 = Image_to_World(viewB, imgcoords2)
        g2_w = g2[:, (0,1)]

        g_avg = (g1_w + g2_w)/2
        error_i = (g1_w-g_avg)*(g1_w-g_avg)+(g2_w-g_avg)*(g2_w-g_avg)
        error_i = np.sum(error_i, axis=1)
        dist_error.append(error_i)

    dist_error = np.asarray(dist_error)
    assert len(dist_error) == len(ph)
    index = np.argmin(dist_error, axis=0)

    ph_all = np.expand_dims(np.asarray(ph), axis=1)
    # ph_all = np.repeat(ph_all, num, axis=1)
    h = ph_all[index, : ]

    # get GP coords x y
    imgcoords = np.concatenate([cxA, cyA, h], axis=1)
    #imgcoords = np.expand_dims(imgcoords, axis=0)
    view1_wc = Image_to_World(viewA, imgcoords)
    view1_wc = np.asarray(view1_wc[:, :2])

    view1_wc_offset = view1_wc /resolution_scaler
    view1_wc_offset_x = view1_wc_offset[:, 0:1] + bbox[0] # * resolution_scaler
    # view1_wc_offset_x = view1_wc_offset_x/w1*w0
    view1_wc_offset_y = view1_wc_offset[:, 1:2] + bbox[1]  # * resolution_scaler
    view1_wc_pmap = np.concatenate([img_id, pt_id, view1_wc_offset_x, view1_wc_offset_y, h]
                                   , axis=1)
    # view1_wc_pmap.append(view1_wc_pmap_i)
    view1_wc_pmap = np.asarray(view1_wc_pmap)

    return view1_wc_pmap

def find_height_GP(v_pmap):
    ph = range(1560, 1960, 10)
    bbox = [352*0.8, 522*0.8]
    image_size = [380, 676]
    resolution_scaler = 76.25

    # camera file
    view1_groundPlane = 'view1'
    groundPlane_view2 = 'view2'
    groundPlane_view3 = 'view3'

    GP_pmap = []
    view_height = []
    view1_wc_pmap = []

    img_id = v_pmap[:, 0:1]
    pt_id = v_pmap[:, 1:2]
    cx1 = v_pmap[:, 2:3]*4
    cy1 = v_pmap[:, 3:4]*4
    cx2 = v_pmap[:, 4:5]*4
    cy2 = v_pmap[:, 5:6]*4
    cx3 = v_pmap[:, 6:7]*4
    cy3 = v_pmap[:, 7:]*4
    dist_error = []

    num = v_pmap.shape[0]

    for hi in ph:
        hi_all = np.ones([num, 1])*hi
        imgcoords1 = np.concatenate([cx1, cy1, hi_all], 1)
        g1 = Image_to_World(view1_groundPlane, imgcoords1)
        g1_w = g1[:, (0,1)]

        imgcoords2 = np.concatenate([cx2, cy2, hi_all], 1)
        g2 = Image_to_World(groundPlane_view2, imgcoords2)
        g2_w = g2[:, (0,1)]

        imgcoords3 = np.concatenate([cx3, cy3, hi_all], 1)
        g3 = Image_to_World(groundPlane_view3, imgcoords3)
        g3_w = g3[:, (0,1)]

        g_avg = (g1_w + g2_w + g3_w)/3
        error_i = (g1_w-g_avg)*(g1_w-g_avg)+(g2_w-g_avg)*(g2_w-g_avg)+ \
                  (g3_w-g_avg)*(g3_w-g_avg)
        error_i = np.sum(error_i, axis=1)
        dist_error.append(error_i)

    dist_error = np.asarray(dist_error)
    assert len(dist_error) == len(ph)
    index = np.argmin(dist_error, axis=0)
    ph_all = np.expand_dims(np.asarray(ph), axis=1)
    h = ph_all[index, : ]
    #view_height.append(h)

    # get GP coords x y
    imgcoords = np.concatenate([cx1, cy1, h], axis=1)
    view1_wc = Image_to_World(view1_groundPlane, imgcoords)
    view1_wc = np.asarray(view1_wc[:, :2])

    view1_wc_offset = view1_wc / resolution_scaler
    view1_wc_offset_x = view1_wc_offset[:, 0:1] + bbox[0]  # * resolution_scaler
    # view1_wc_offset_x = view1_wc_offset_x/w1*w0
    view1_wc_offset_y = view1_wc_offset[:, 1:2] + bbox[1]  # * resolution_scaler
    view1_wc_pmap = np.concatenate([img_id, pt_id, view1_wc_offset_x, view1_wc_offset_y, h]
                                   , axis=1)

    view1_wc_pmap = np.asarray(view1_wc_pmap)
    # view_height.append(h)
    # view_height = np.expand_dims(np.asarray(view_height), 1)

    return view1_wc_pmap

def check_all_points(v1_pmap_i, v2_pmap_i, v3_pmap_i):

    w0 = 676
    h0 = 380

    w1 = 640
    h1 = 768

    point_num_1 = np.unique(v1_pmap_i[:, 1])
    point_num_2 = np.unique(v2_pmap_i[:, 1])
    point_num_3 = np.unique(v3_pmap_i[:, 1])
    # assert (max(point_num_1) <= max(point_num_3))
    # assert (max(point_num_2) <= max(point_num_3))

    num_GP_i = np.unique(np.asarray(list(point_num_1)+list(point_num_2)+list(point_num_3)))
    num_GP_i = num_GP_i.shape[0]

    height_i = np.zeros([num_GP_i, 1])
    GP_pmap_i = np.zeros([num_GP_i, 2])

    ptID_pmap_height111 = np.zeros([1, 8])
    ptID_pmap_height011 = np.zeros([1, 6])
    ptID_pmap_height101 = np.zeros([1, 6])
    ptID_pmap_height110 = np.zeros([1, 6])
    ptID_pmap_height100 = np.zeros([1, 4])
    ptID_pmap_height010 = np.zeros([1, 4])
    ptID_pmap_height001 = np.zeros([1, 4])
    ptID_pmap_height000 = np.zeros([1, 4])

    for i in range(num_GP_i):
        # print i
        try:
            v1_i_xy = v1_pmap_i[i==v1_pmap_i[:, 1], :][0]
        except IndexError:
            check1 = 0
        else:
            check1 = check_point(v1_i_xy[2:])

        try:
            v2_i_xy = v2_pmap_i[i==v2_pmap_i[:, 1], :][0]
        except IndexError:
            check2 = 0
        else:
            check2 = check_point(v2_i_xy[2:])

        try:
            v3_i_xy = v3_pmap_i[i==v3_pmap_i[:, 1], :][0]
        except IndexError:
            check3 = 0
        else:
            check3 = check_point(v3_i_xy[2:])

        check123 = [check1, check2, check3]
        if check123 in [[1, 1, 1]]:
            coords = list(v1_i_xy) + list(v2_i_xy[2:])+list(v3_i_xy[2:])
            coords = np.asarray(coords)
            coords = np.expand_dims(coords, axis=0)
            ptID_pmap_height111 = np.row_stack([ptID_pmap_height111, coords])

        if check123 in [[0, 1, 1]]:
            coords = list(v2_i_xy) + list(v3_i_xy[2:])
            coords = np.asarray(coords)
            coords = np.expand_dims(coords, axis=0)
            ptID_pmap_height011 = np.row_stack([ptID_pmap_height011, coords])

        if check123 in [[1, 0, 1]]:
            coords = list(v1_i_xy) + list(v3_i_xy[2:])
            coords = np.asarray(coords)
            coords = np.expand_dims(coords, axis=0)
            ptID_pmap_height101 = np.row_stack([ptID_pmap_height101, coords])

        if check123 in [[1, 1, 0]]:
            coords = list(v1_i_xy) + list(v2_i_xy[2:])
            coords = np.asarray(coords)
            coords = np.expand_dims(coords, axis=0)
            ptID_pmap_height110 = np.row_stack([ptID_pmap_height110, coords])

        if check123 in [[1, 0, 0]]:
            coords = list(v1_i_xy)
            coords = np.asarray(coords)
            coords = np.expand_dims(coords, axis=0)
            ptID_pmap_height100 = np.row_stack([ptID_pmap_height100, coords])

        if check123 in [[0, 1, 0]]:
            coords = list(v2_i_xy)
            coords = np.asarray(coords)
            coords = np.expand_dims(coords, axis=0)
            ptID_pmap_height010 = np.row_stack([ptID_pmap_height010, coords])

        if check123 in [[0, 0, 1]]:
            coords = list(v3_i_xy)
            coords = np.asarray(coords)
            coords = np.expand_dims(coords, axis=0)
            ptID_pmap_height001 = np.row_stack([ptID_pmap_height001, coords])

        if check123 in [[0, 0, 0]]:
        #     height_i[i] = -1
        #     GP_pmap_i[i] = [-1, -1]
            num_GP_i = num_GP_i - 1

    ptID_pmap_height111 = ptID_pmap_height111[1:, :]
    ptID_pmap_height011 = ptID_pmap_height011[1:, :]
    ptID_pmap_height101 = ptID_pmap_height101[1:, :]
    ptID_pmap_height110 = ptID_pmap_height110[1:, :]
    ptID_pmap_height100 = ptID_pmap_height100[1:, :]
    ptID_pmap_height010 = ptID_pmap_height010[1:, :]
    ptID_pmap_height001 = ptID_pmap_height001[1:, :]
    ptID_pmap_height000 = ptID_pmap_height000[1:, :]

    return [num_GP_i, ptID_pmap_height111,ptID_pmap_height011,ptID_pmap_height101,
                       ptID_pmap_height110,ptID_pmap_height100,ptID_pmap_height010,
                       ptID_pmap_height001]

def check_point(v1_i_xy):
    # w0 = 676
    # h0 = 380
    # if v1_i_xy[0] > 0 and v1_i_xy[0] < w0 and v1_i_xy[1] > 0 and v1_i_xy[1] < h0:
    #     return 1
    # else:
    #     return 0
    return 1


# dist cal function:
def dist_cal(v1_pmap_i1, v1_pmap_i):
    if v1_pmap_i1==[]:
        return []

    n1 = v1_pmap_i1.shape[0]

    if n1<2:
        dist = np.ones((1, 1))*10000 # only one person, then a large nearest distance is assigned.
    else:

        dist_col = v1_pmap_i1[:, 2:4]
        dist_col_x = dist_col[:, 0:1]
        dist_col_y = dist_col[:, 1:]
        dist_col_x_re = np.tile(dist_col_x, (1, n1))
        dist_col_y_re = np.tile(dist_col_y, (1, n1))

        dist_row = np.transpose(dist_col)
        dist_row_x = dist_row[0:1, :]
        dist_row_y = dist_row[1:, :]
        dist_row_x_re = np.tile(dist_row_x, (n1, 1))
        dist_row_y_re = np.tile(dist_row_y, (n1, 1))

        dist_mat = np.sqrt(np.power(dist_col_x_re - dist_row_x_re, 2) + np.power(dist_col_y_re - dist_row_y_re, 2))
        dist_mat = np.sort(dist_mat, axis=1)
        dist = dist_mat[:, 1:2]*0.1
        # the second smallest is the nearest distance.
        # 1 pixel stands for 0.1m on the ground.

    v1_point_id = v1_pmap_i1[:, 1]
    v1_point_id2 = v1_pmap_i[:, 1]
    point_id = []

    v1_pmap_i_re = v1_pmap_i
    n2 = v1_pmap_i.shape[0]
    if n1!=n2:
        v1_pmap_i_re = [v1_pmap_i[i, :] for i in range(n2) if v1_pmap_i[i, 1] in list(v1_pmap_i1[:, 1])]
        v1_pmap_i_re = np.asarray(v1_pmap_i_re)
    v1_pmap_i1 = np.column_stack((v1_pmap_i_re, dist))

    return v1_pmap_i1









########################################################################################################################################################
if __name__ =='__main__':
    ## read the labels of view 1
    gt_file =  'labels/'
    gt_file1  = gt_file + 'via_region_data_view1.json'
    gt_file2 =  gt_file + 'via_region_data_view2.json'
    gt_file3 =  gt_file + 'via_region_data_view3.json'

    save_folder = 'GP_dmaps'
    w0 = 676
    h0 = 380

    # read masks
    mask1 = np.load('ROI_maps/ROIs/camera_view/mask1_ic.npz')
    mask1 = mask1.f.arr_0
    mask2 = np.load('ROI_maps/ROIs/camera_view/mask2_ic.npz')
    mask2 = mask2.f.arr_0
    mask3 = np.load('ROI_maps/ROIs/camera_view/mask3_ic.npz')
    mask3 = mask3.f.arr_0

    # view 1
    with open(gt_file1) as data_file:
        v1_pmap_json = json.load(data_file)
    v1_pmap = []
    for key in v1_pmap_json.keys():
        img_id = int(key[6:10])
        regions = v1_pmap_json[key]['regions']
        for point_id in regions:
            point_id_num = int(float(point_id))
            # whole_id = regions[point_id]['region_attributes']['whole_ID']
            try:
                cx = int(regions[point_id]['shape_attributes']['cx'])
                cy = int(regions[point_id]['shape_attributes']['cy'])
            except TypeError:
                continue
            if cx<0 or cx>=w0*4 or cy<0 or cy>=h0*4 or (mask1[cy, cx]==0):
                continue
            else:
                v1_pmapi = [img_id, point_id_num, cx/4, cy/4]
                v1_pmap.append(v1_pmapi)
    v1_pmap = np.asarray(v1_pmap)
    v1_pmap_1 = v1_pmap[np.argsort(v1_pmap[:, 0]), :]

    # view 2
    with open(gt_file2) as data_file2:
        v2_pmap_json = json.load(data_file2)
    v2_pmap = []
    for key in v2_pmap_json.keys():
        img_id = int(key[6:10])
        regions = v2_pmap_json[key]['regions']
        for point_id in regions:
            point_id_num = int(float(point_id))
            # whole_id = regions[point_id]['region_attributes']['whole_ID']
            try:
                cx = int(regions[point_id]['shape_attributes']['cx'])
                cy = int(regions[point_id]['shape_attributes']['cy'])
            except TypeError:
                continue
            if cx < 0 or cx >=w0 * 4 or cy < 0 or cy >=h0 * 4 or (mask2[cy, cx] == 0):
                continue
            else:
                v2_pmapi = [img_id, point_id_num, cx / 4, cy / 4]
                v2_pmap.append(v2_pmapi)
    v2_pmap = np.asarray(v2_pmap)
    v2_pmap_1 = v2_pmap[np.argsort(v2_pmap[:, 0]), :]

    #
    # # view 3
    with open(gt_file3) as data_file3:
        v3_pmap_json = json.load(data_file3)
    v3_pmap = []
    for key in v3_pmap_json.keys():
        img_id = int(key[6:10])
        regions = v3_pmap_json[key]['regions']
        for point_id in regions:
            point_id_num = int(float(point_id))
            # whole_id = regions[point_id]['region_attributes']['whole_ID']
            try:
                cx = int(regions[point_id]['shape_attributes']['cx'])
                cy = int(regions[point_id]['shape_attributes']['cy'])
            except TypeError:
                continue
            if cx < 0 or cx >= w0 * 4 or cy < 0 or cy >= h0 * 4 or (mask3[cy, cx] == 0):
                continue
            else:
                v3_pmapi = [img_id, point_id_num, cx / 4, cy / 4]
                v3_pmap.append(v3_pmapi)
    v3_pmap = np.asarray(v3_pmap)
    v3_pmap_1 = v3_pmap[np.argsort(v3_pmap[:, 0]), :]
    
    # image_num
    img_num = v1_pmap[:, 0]
    img_num = np.unique(img_num)
    img_num = np.sort(img_num)
    
    num_GP = []
    ptID_pmap_height111 = np.zeros([1, 8])
    ptID_pmap_height011 = np.zeros([1, 6])
    ptID_pmap_height101 = np.zeros([1, 6])
    ptID_pmap_height110 = np.zeros([1, 6])
    ptID_pmap_height100 = np.zeros([1, 4])
    ptID_pmap_height010 = np.zeros([1, 4])
    ptID_pmap_height001 = np.zeros([1, 4])
    
    for i in img_num:
        print('img', i)

        v1_pmap_i = v1_pmap[v1_pmap[:, 0]==i, :]
        v1_pmap_i = v1_pmap_i[np.argsort(v1_pmap_i[:, 1]), :]
        
        v2_pmap_i = v2_pmap[v2_pmap[:, 0]==i, :]
        v2_pmap_i = v2_pmap_i[np.argsort(v2_pmap_i[:, 1]), :]
        
        v3_pmap_i = v3_pmap[v3_pmap[:, 0]==i, :]
        v3_pmap_i = v3_pmap_i[np.argsort(v3_pmap_i[:, 1]), :]

        # get height and GP pmap
        num_GP_i, ptID_pmap_height111_i, ptID_pmap_height011_i, ptID_pmap_height101_i,\
        ptID_pmap_height110_i, ptID_pmap_height100_i, ptID_pmap_height010_i,\
        ptID_pmap_height001_i = check_all_points(v1_pmap_i, v2_pmap_i, v3_pmap_i)

        print('num_GP', num_GP_i)
        num_GP.append(num_GP_i)

        # pt_num = ptID_pmap_height_i.shape[0]
        # v_pmap_GP_i = np.zeros([pt_num, 5])
        # v_pmap_GP_i[:pt_num, 0] = np.ones([pt_num])*i
        # v_pmap_GP_i[:pt_num, 1:5] = ptID_pmap_height_i
        # v_pmap_GP = np.row_stack([v_pmap_GP,v_pmap_GP_i])

        # check points for obtaining height and GP pmaps
        ptID_pmap_height111 = np.row_stack([ptID_pmap_height111, ptID_pmap_height111_i])

        ptID_pmap_height011 = np.row_stack([ptID_pmap_height011, ptID_pmap_height011_i])
        ptID_pmap_height101 = np.row_stack([ptID_pmap_height101, ptID_pmap_height101_i])
        ptID_pmap_height110 = np.row_stack([ptID_pmap_height110, ptID_pmap_height110_i])

        ptID_pmap_height100 = np.row_stack([ptID_pmap_height100, ptID_pmap_height100_i])
        ptID_pmap_height010 = np.row_stack([ptID_pmap_height010, ptID_pmap_height010_i])
        ptID_pmap_height001 = np.row_stack([ptID_pmap_height001, ptID_pmap_height001_i])


    num_GP = np.asarray(num_GP)
    # v_pmap_GP = v_pmap_GP[1:, :]
    # v_pmap_GP = v_pmap_GP[np.argsort(v_pmap_GP[:, 0]), :] # order


    # get pmaps
    v_pmap_GP = np.zeros([1, 5])
    if ptID_pmap_height111.shape[0] > 1 :
        ptID_pmap_height111 = ptID_pmap_height111[1:, :]
        ptID_GP_pmap_height111 = find_height_GP(ptID_pmap_height111)
        v_pmap_GP = np.concatenate([v_pmap_GP, ptID_GP_pmap_height111], axis=0)

    if ptID_pmap_height011.shape[0] > 1:
        ptID_pmap_height011 = ptID_pmap_height011[1:, :]
        ptID_GP_pmap_height011 = find_height_GP2(ptID_pmap_height011, 1)
        v_pmap_GP = np.concatenate([v_pmap_GP, ptID_GP_pmap_height011], axis=0)

    if ptID_pmap_height101.shape[0] > 1:
        ptID_pmap_height101 = ptID_pmap_height101[1:, :]
        ptID_GP_pmap_height101 = find_height_GP2(ptID_pmap_height101, 2)
        v_pmap_GP = np.concatenate([v_pmap_GP, ptID_GP_pmap_height101], axis=0)

    if ptID_pmap_height110.shape[0] > 1:
        ptID_pmap_height110 = ptID_pmap_height110[1:, :]
        ptID_GP_pmap_height110 = find_height_GP2(ptID_pmap_height110, 3)
        v_pmap_GP = np.concatenate([v_pmap_GP, ptID_GP_pmap_height110], axis=0)

    if ptID_pmap_height100.shape[0] > 1:
        ptID_pmap_height100 = ptID_pmap_height100[1:, :]
        ptID_GP_pmap_height100 = find_height_GP3(ptID_pmap_height100, 1)
        v_pmap_GP = np.concatenate([v_pmap_GP, ptID_GP_pmap_height100], axis=0)

    if ptID_pmap_height010.shape[0] > 1:
        ptID_pmap_height010 = ptID_pmap_height010[1:, :]
        ptID_GP_pmap_height010 = find_height_GP3(ptID_pmap_height010, 2)
        v_pmap_GP = np.concatenate([v_pmap_GP, ptID_GP_pmap_height010], axis=0)

    if ptID_pmap_height001.shape[0] > 1:
        ptID_pmap_height001 = ptID_pmap_height001[1:, :]
        ptID_GP_pmap_height001 = find_height_GP3(ptID_pmap_height001, 3)
        v_pmap_GP = np.concatenate([v_pmap_GP, ptID_GP_pmap_height001], axis=0)

    # v_pmap_GP = np.concatenate([ptID_GP_pmap_height111,ptID_GP_pmap_height011,
    #                             ptID_GP_pmap_height101,ptID_GP_pmap_height110,
    #                             ptID_GP_pmap_height100,ptID_GP_pmap_height010,
    #                             ptID_GP_pmap_height001], axis=0)
    v_pmap_GP = v_pmap_GP[1:, :]
    v_pmap_GP = v_pmap_GP[np.argsort(v_pmap_GP[:, 0]), :]


    # calculate the nearest distance:
    v1_pmap_dist = np.zeros([1, 5])
    v2_pmap_dist = np.zeros([1, 5])
    v3_pmap_dist = np.zeros([1, 5])
    for i in img_num:

        print(i)

        v1_pmap_i = v1_pmap[v1_pmap[:, 0] == i, :]
        v1_pmap_i = v1_pmap_i[np.argsort(v1_pmap_i[:, 1]), :]

        v2_pmap_i = v2_pmap[v2_pmap[:, 0] == i, :]
        v2_pmap_i = v2_pmap_i[np.argsort(v2_pmap_i[:, 1]), :]

        v3_pmap_i = v3_pmap[v3_pmap[:, 0] == i, :]
        v3_pmap_i = v3_pmap_i[np.argsort(v3_pmap_i[:, 1]), :]

        v_pmap_GP_i = v_pmap_GP[v_pmap_GP[:, 0] == i, :]
        v_pmap_GP_i = v_pmap_GP_i[np.argsort(v_pmap_GP_i[:, 1]), :]

        # view 1:
        v1_pmap_i1 = np.zeros([1, 5])
        v2_pmap_i2 = np.zeros([1, 5])
        v3_pmap_i3 = np.zeros([1, 5])

        for j in list(v_pmap_GP_i[:, 1].astype(int)):

            if j in list(v1_pmap_i[:, 1].astype(int)):
                v1_pmap_i1_j = v_pmap_GP_i[v_pmap_GP_i[:, 1]==j, :]
                v1_pmap_i1 = np.row_stack([v1_pmap_i1, v1_pmap_i1_j])

            if j in list(v2_pmap_i[:, 1].astype(int)):
                v2_pmap_i2_j = v_pmap_GP_i[v_pmap_GP_i[:, 1]==j, :]
                v2_pmap_i2 = np.row_stack([v2_pmap_i2, v2_pmap_i2_j])

            if j in list(v3_pmap_i[:, 1].astype(int)):
                v3_pmap_i3_j = v_pmap_GP_i[v_pmap_GP_i[:, 1]==j, :]
                v3_pmap_i3 = np.row_stack([v3_pmap_i3, v3_pmap_i3_j])

        v1_pmap_i1 = v1_pmap_i1[1:, :]
        v2_pmap_i2 = v2_pmap_i2[1:, :]
        v3_pmap_i3 = v3_pmap_i3[1:, :]

        v1_pmap_i1_dist = dist_cal(v1_pmap_i1, v1_pmap_i)
        v2_pmap_i2_dist = dist_cal(v2_pmap_i2, v2_pmap_i)
        v3_pmap_i3_dist = dist_cal(v3_pmap_i3, v3_pmap_i)

        v1_pmap_dist = np.row_stack((v1_pmap_dist, v1_pmap_i1_dist))
        v2_pmap_dist = np.row_stack((v2_pmap_dist, v2_pmap_i2_dist))
        v3_pmap_dist = np.row_stack((v3_pmap_dist, v3_pmap_i3_dist))

    v1_pmap_dist = v1_pmap_dist[1:, :]
    v2_pmap_dist = v2_pmap_dist[1:, :]
    v3_pmap_dist = v3_pmap_dist[1:, :]

    # save GP GT number
    f0 = h5py.File(save_folder + '/Street_v1_pmap_dist.h5', 'w')
    num_set = f0.create_dataset('v1_pmap_dist', data = v1_pmap_dist)
    f0.close()

    f0 = h5py.File(save_folder + '/Street_v2_pmap_dist.h5', 'w')
    num_set = f0.create_dataset('v2_pmap_dist', data = v2_pmap_dist)
    f0.close()

    f0 = h5py.File(save_folder + '/Street_v3_pmap_dist.h5', 'w')
    num_set = f0.create_dataset('v3_pmap_dist', data = v3_pmap_dist)
    f0.close()

    f0 = h5py.File(save_folder + '/Street_groundplane_pmap.h5', 'w')
    num_set = f0.create_dataset('v_pmap_GP', data = v_pmap_GP)
    f0.close()

    a = 1






































    # save GP GT number
    f0 = h5py.File(save_folder + '/Street_groundplane_pmap.h5', 'w')
    num_set = f0.create_dataset('v_pmap_GP', data = v_pmap_GP)
    f0.close()

    # create the GP dmaps
    w1 = 640
    h1 = 768
    v_GP_density_map = generate_density_map(v_pmap_GP.astype(int), w1, h1)

    # plt.figure()
    # plt.imshow(v_GP_density_map[0, :, :])

    v_GP_count = np.sum(v_GP_density_map, 2)
    v_GP_count = np.sum(v_GP_count, 1)
    # assert (np.sum(np.abs(v_GP_count - num_GP) <= 1))

    # save GP GT number
    f0 = h5py.File(save_folder + '/Street_groundplane_num.h5', 'w')
    num_set = f0.create_dataset('dmap_counting_num', data = v_GP_count)
    num_set2 = f0.create_dataset('counting_num', data = num_GP)
    f0.close()


    v_GP_density_map = np.expand_dims(v_GP_density_map, axis = 3)

    # save GP dmap
    f1 = h5py.File(save_folder +'/Street_groundplane_dmaps_10.h5', 'w')
    density_map_set = f1.create_dataset('density_maps', data =  v_GP_density_map) #v1_density_map)
    count_set = f1.create_dataset('count', data = v_GP_count)
    f1.close()

    # f2 = h5py.File('GP/PETS_groundplane_test_10.h5', 'w')
    # density_map_set = f2.create_dataset('density_maps', data =  v_GP_density_map[350:]) #v1_density_map)
    # #images_train_set = f_train.create_dataset('images', data = images[:351])
    # f2.close()

    # save GP json
    GP_pmap_json = writeInJason(v_pmap_GP, gt_file3)
    with open(save_folder +'/GP_pmap_height.json', 'w') as fp3:
        json.dump(GP_pmap_json, fp3)

    a = 0