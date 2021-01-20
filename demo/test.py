#!/usr/bin/python
# -*- coding: UTF-8 -*-

from photometric_stereo_det.flaw_bbox import *
from photometric_stereo_det.PhotometricStereoDet import *
from RivetDetect.yolov4rivetsdetection.RivetsDetectionAPI import RivetDetect
import retinanet.DetectionAPI as Detector
import cv2
import numpy as np
import json
import pickle

import time

# with open(dataFor_detect_from_body.pkl, 'rb') as f:
#     img = pickle.load(f)

GRAD=8000
RS=8
STOPID=5
DIR=r'C:\Users\vrlab\Documents\YklabData\YklabData\PlaneHead\SideToSun'
#DIR=r'C:\Users\vrlab\Documents\YklabData\YklabData\PlaneHead\SideToSun\2_20'
TAG=''
FMT='png'
CALI='-Ad_reduce'

bias=3
DIR2=r'C:\Users\vrlab\Documents\YklabData\YklabData\PlaneHead\SideToSun\3_10'
K_size=13
Dist=155
N=10

CropX=1750
CropY=1500
#Test On Board
# K_size=3
# Dist=70
# N=10
# len(id_changed)>=6
def FindCurrentSubDir(Path):

    SubDir=[]
    for root,dirs,file in os.walk(Path):
        if len(dirs)!=0:
            for dir in dirs:
                SubDir.append(os.path.join(root,dir))

    return SubDir
def ReadBin(fn):
    H=4384
    W=6576
    # W=4096
    # H=3000
    with open(fn, 'rb+') as f:
        image_data = f.read()
        #print('image_data',image_data)
    img_raw = np.frombuffer(image_data, dtype=np.int16)
    print(img_raw.shape)
    img_raw = img_raw.reshape((H,W))
    img=img_raw.copy()
    print('Over Expose pixel num:',len(img[img>4095]),img[img>4094])
    #img=img*(256/(2**12))

    return img   
def Generate(Path,LoadPPNG=0):
    data=[]
    print('Acquire Path:',Path)
    if LoadPPNG==0:
        for i in range(7):
            path=f'{Path}/{i}.bin'
            print(path)
            img=ReadBin(path)
            if i==0:
                base_gray=img
            else:
                data.append(img)

        print('Base_gray shape',base_gray.shape)
    else:
        for i in range(7):
            img = cv2.imread(f'{Path}\{i}.png', cv2.IMREAD_UNCHANGED)
            if i ==0:
                base_gray=img
            else:
                data.append(img)
    return base_gray,data


def CropRivetPosition(rivets,shape,dstX=CropX,dstY=CropY):

    h,w,_=shape

    x_left=w/2-dstX/2
    x_right=w/2+dstX/2
    y_top=h/2-dstY/2
    y_bottom=h/2+dstY/2

    crop_rivets=[]
    for rivet in rivets:

        if x_left<=rivet[0]<x_right and y_top<=rivet[1]<=y_bottom \
            and x_left<=rivet[2]<x_right and y_top<=rivet[3]<y_bottom :
            rivet[0]-=x_left
            rivet[1]-=y_top
            rivet[2]-=x_left
            rivet[3]-=y_top
            crop_rivets.append(rivet)
        if (rivet[0]<x_left or rivet[1]<y_top) \
            and x_left<=rivet[2]<x_right and y_top<=rivet[3]<y_bottom :

            if rivet[0]<=x_left:
                rivet[0]=x_left
            if rivet[1]<=y_top:
                rivet[1]=y_top

            rivet[0]-=x_left
            rivet[1]-=y_top
            rivet[2]-=x_left
            rivet[3]-=y_top
            crop_rivets.append(rivet)

        if x_left <= rivet[0] < x_right and y_top <= rivet[1] <= y_bottom \
                and  (rivet[2] >= x_right or  rivet[3] >= y_bottom):

            if  rivet[2] > x_right:
                rivet[2] = x_right-1
            if  rivet[3] >y_bottom:
                rivet[3] = y_bottom-1

            rivet[0]-=x_left
            rivet[1]-=y_top
            rivet[2]-=x_left
            rivet[3]-=y_top

            crop_rivets.append(rivet)

    print('len of origin rivets:',len(rivets))
    print('len of cropping ',len(crop_rivets))
    return crop_rivets
if __name__ == '__main__':
    very_beginning = time.time()


    # 加载相机参数信息
    cam_params_json = ''
    with open('./cameras0430.json', 'r') as f:
        cam_params_json = f.read()
    cam_params = json.loads(cam_params_json)


    # 加载光源标定信息
    text = ''
    print('RS',RS)
    path=f'./lights_11-18_board-Ad.json'
    print('Using cali path:',path)
    with open(path, 'r') as f:
        text = f.read()

    light_data = json.loads(text)

    bright_areas = light_data['bright_areas']
    ball_radius = light_data['ball_radius']
    ball_origin = light_data['ball_origin']
    light_positions = light_data['lights']

    # 计算各点光源的方向向量
    
   
    PM_L_list, black_list = generate_PM_LData_list(bright_areas, ball_origin, ball_radius, light_positions)

    # 生成初始化的相机列表
    # PM_Cam_list = generate_PM_Cam_list(num_of_cam=1)


    # 遍历相机参数，建立相机列表
    PM_Cam_list = []
    PM_id = None
    PN_id = None
    for camera in cam_params['cameras']:
        if camera['name'].lower() == 'photometric':
            PM_id = camera['id']
        elif camera['name'].lower() == 'photoneo':
            PN_id = camera['id']
        PM_Cam_list.append(PM_Cam(cam_id=camera['id'], cam_k=np.array(camera['intrinsic']), cam_rt=np.array(camera['pose']), cam_frame_uid=-1))

    # 生成一个光度相机检测实例
    pm = PhotometricStereoDet(lights=PM_L_list, cams_init=PM_Cam_list, cam_id=PM_id, grad_thresh=GRAD, Photoneo_id=PN_id,
        bboxes_cnt_thresh=10, bbox_size_thresh=8, rs_scale=RS,erosion_kernel=5,K_size=K_size,Dist=Dist,N=N,Dir=DIR)

    time_diff = 0
    time_normal = 0
    time_detection = 0

    STOPID=5
    Path=FindCurrentSubDir(DIR)

    #print('Path',Path)

    detector = Detector.RivetDetector()
    for path in [DIR2]:
        EachTime=time.time()
#         PID=i
#         # 加载Photometric相机采集的图像
#         # 基础图像：无光源
#         base = cv2.imread(f'/root/PCPR/comacphotometric-depth_test/data/{DIR}/{TAG}/0.{FMT}')
#         print(f'/root/PCPR/comacphotometric-depth_test/data/{DIR}/{TAG}/0.{FMT}')

#         # 转换为单通道float32图像
#         print('base.shape',base.shape)
#         base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
#         print('base_gray.shape',base_gray.shape)
#         base_gray = base_gray.astype(np.float32) / 255


#         # 各光源分别点亮时采集的图像
#         data = []
#         for j in range(1, 7):
#             img = cv2.imread(f'/root/PCPR/comacphotometric-depth_test/data/{DIR}/{TAG}/{j}.{FMT}')
#             print(f'/root/PCPR/comacphotometric-depth_test/data/{DIR}/{TAG}/{j}.{FMT}')
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             gray = gray.astype(np.float32) / 255
#             gray=img
#             data.append(gray)
        print('path:',path)

        base_gray,data=Generate(path,LoadPPNG=0)
        # 生成差别图
        srt_t = time.time()
        final_data ,pmForDetect= generate_differential_image_list(base_gray, data, black_list,dstX=CropX,dstY=CropY,crop=1,RawBin=1)
        time_diff += time.time()-srt_t
        
        

        # 加载Photoneo深度图
        PN_depth = None#cv2.imread(f'/root/PCPR/comacphotometric-depth_test/data/10/thumbnail-sp{STOPID}/{STOPID}_{PID}_dm.tiff', cv2.IMREAD_UNCHANGED)
        
        
        # 使用Photometric图像数据检测铆钉/孔
        print('Check Three Channels:',pmForDetect.shape)
        Time=time.time()

        # rivets = RivetDetect(pmForDetect)[0]
        rivets = detector.RivetDetection(pmForDetect)
        rivets=CropRivetPosition(rivets,pmForDetect.shape,dstX=CropX,dstY=CropY)
        RivetTime=time.time()-Time
        print('Rivet time:',RivetTime,' s')
        # rivets=[(0,0)]
        # 开始检测
        # 重建法线图与有效Mask
        srt_t = time.time()
        normal, normalDisplay,mask = pm.construct_normal(final_data, PN_depth, depth_down_scale=RS) #With Photoneo depth data
        # normal, mask = pm.construct_normal(final_data, None) # No Photoneo, mask==None

        time_normal += time.time()-srt_t

        # 缺陷检测
        srt_t = time.time()
        
        
        final_bboxes, status, normal_grad ,Crediet= pm.detect_flaw(np.array(rivets), radius=bias,mask=mask,savepath=path) # Exclude rivet areas and invalid depth areas.
        # final_bboxes, status, normal_grad = pm.detect_flaw(np.array([])) # No rivets.
        time_detection += time.time()-srt_t

        # 上述两条命令等价于以下一条。
        # normal, final_bboxes, status, normal_grad = pm.reconstruction(final_data, np.array([]))

        # print(f'Group {i}: ', len(final_bboxes), ' bbox found.')
        

        # Make clones of normals.
        res_wo_bbox = normalDisplay.copy()
        res = normalDisplay.copy()


        # Draw bounding boxes of detected flaws on normal map.
        for idx,bbox in enumerate( final_bboxes):
            cv2.rectangle(res, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
            cv2.putText(res, str(Crediet[idx]), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        radius=7*8
        for rivet in rivets:
            x1,y1,x2,y2=rivet

            cv2.rectangle(res, (int(x1-bias*RS), int(y1-bias*RS)), (int(x2+bias*RS),int(y2+bias*RS)), (128, 128, 0), 4)


        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("img", 800, 600)
        # cv2.imshow("img", normal_grad)
        # cv2.waitKey(0)

        # cv2.imshow("img", res)
        # cv2.imwrite(f'./photoneo/{STOPID}_{PID}_normal_w_depth.png', res_wo_bbox)
        # cv2.imwrite(f'./results_bdt3/{STOPID}/{STOPID}_{PID}_result_wo_bbox.png', res_wo_bbox)
        cv2.imwrite(f'{path}\\normal-{TAG}_GRAD{GRAD}-RS{RS}_{K_size}-{Dist}-{N}.png', normalDisplay)
        cv2.imwrite(f'{path}\\label-{TAG}_GRAD{GRAD}-RS{RS}_{K_size}-{Dist}-{N}.png', res)
        # print(f'PID={PID} saved successfully.')
        print('Done!')
        print('Rivet Time Cost:',RivetTime,'s')
        print('Each Total Time:',time.time()-EachTime,'s')
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    lenth=len(Path)
    print('Overall time consumed: ', time.time() - very_beginning, 's')
    print('Average diff time: ', f'{time_diff}s / 27 = {time_diff / lenth}')
    print('Average normal time: ', f'{time_normal}s / 27 = {time_normal / lenth}')
    print('Average detection time: ', f'{time_detection}s / 27 = {time_detection / lenth}')
    print('Average running time per frame: ', f'{(time_diff + time_normal + time_detection)}s / 27 = {(time_diff + time_normal + time_detection) / lenth}')
    # print(final_bboxes)
