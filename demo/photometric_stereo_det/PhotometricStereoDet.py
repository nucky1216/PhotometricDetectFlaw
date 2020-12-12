#!/usr/bin/python
# -*- coding: UTF-8 -*-
from photometric_stereo_det.flaw_bbox import *

import cv2
import numpy as np
import torch
import time
import photometric_stereo_det.utilities as ut
#import pcpr
import torch.nn.functional as F
ValidPixels=0.65
CUDA=0

veryBeginTime=time.time()
BT=0
Cur=0
def TestBegin():
    global BT
    BT=time.time()
def TestEnd():
    global Cur
    Cur+=1
    cost=time.time()-BT
    print('\033[5;30;42m',end='')
    print('点位：',Cur,' Liu Time Cost:',cost,' s',end='')
    print('流程总时间：',time.time()-veryBeginTime,end='')
    print('\033[0m')
    return 
    
# def TestGpu_init(cuda):
#     global CUDA
#     CUDA=cuda
#     pynvml.nvmlInit()
# def TestGpu(str):
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     print('Here:----[',str,']-----')
#     print('used/total:',meminfo.used,'/',meminfo.total)
# def TestGpuEnd():
#     pynvml.nvmlShutdown()
    
    
    
def light_dirs(chrom_img, highlights):
    '''
    input:
    chrom_img : h*w 
    highlights : 2*n ndarray
    output: light dirs n*3 numpy array
    '''
    assert highlights.any() != None, 'highlights invalid.'
    assert highlights.shape[0] == 2, 'highlights invalid.'
    assert chrom_img.any() != None, 'chrom_img invalid.'

    cnts,_ =cv2.findContours(chrom_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    assert len(cnts) != 0, "no contours found."
    rect = cv2.boundingRect(cnts[0])

    radius = rect[2]/2.0
    assert radius != 0., "chrome radius should be bigger than 0."
    x = (highlights[0, :] - rect[0] - radius)/radius
    y = (highlights[1, :] - rect[1] - radius)/radius
    z = np.sqrt( 1 - np.power(x, 2) - np.power(y, 2))

    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    z = np.expand_dims(z, axis=1)

    ldirs = np.hstack([x, y, z])
    return ldirs


def compute_light_direction(light_uv, ball_origin, ball_radius):
    '''
    根据光斑的像素坐标、标定球球心与标定球半径，估算光线的入射方向。
    均以像素坐标系为准，单位pixel

    :param light_uv: 光斑[u, v]坐标
    :param ball_origin: 标定球球心坐标
    :param ball_radius: 标定球半径
    :return: 光线入射方向向量
    '''
    offset_x = light_uv[0] - ball_origin[0]
    offset_y = light_uv[1] - ball_origin[1]
    tmp = ball_radius * ball_radius - offset_y * offset_y
    offset_z = (tmp - offset_x * offset_x) ** (1/2)
    normal = np.array([offset_x, offset_y, -offset_z])
    normal = normal / np.linalg.norm(normal)
    normal=normal.astype(np.float32)
    print('CAST normal:',normal)

    ray_out = np.array([0, 0, -1])
    projection = np.dot(ray_out, normal) * normal
    dir = projection - ray_out
    ray_in = -(projection + dir)

    ray_inFinal=ray_in / np.linalg.norm(ray_in)
    return ray_inFinal

def compute_light_direction2(light_uv, ball_origin, ball_radius):
    '''
    根据光斑的像素坐标、标定球球心与标定球半径，估算光线的入射方向。
    均以像素坐标系为准，单位pixel

    :param light_uv: 光斑[u, v]坐标
    :param ball_origin: 标定球球心坐标
    :param ball_radius: 标定球半径
    :return: 光线入射方向向量
    '''
    offset_x = light_uv[0] - ball_origin[0]
    offset_y = light_uv[1] - ball_origin[1]
    tmp = ball_radius * ball_radius - offset_y * offset_y
    offset_z = (tmp - offset_x * offset_x) ** (1/2)
    normal = np.array([offset_x, offset_y, offset_z])
    normal = normal / np.linalg.norm(normal)
    ray_out = np.array([0, 0, -1])
    projection = 2*np.dot(ray_out, normal).dot(normal)
    dir = projection - ray_out
    ray_in = dir
    return ray_in / np.linalg.norm(ray_in)

def generate_PM_LData_list(bright_area_uv_list, ball_origin, ball_radius, positions=None):
    '''
    生成一组PM_LData数据。若某张图的光斑区域为None，则会将其加入排除列表。

    :param bright_area_uv_list: 光斑uv坐标列表
    :param ball_origin: 标定球球心坐标
    :param ball_radius: 标定球球心半径
    :return: List(PM_LData) 光线数据; List(index) 被排除的光源index。
    '''
    
    
    assert len(ball_origin)==2,'Worng Input in \"generate_PM_LData_list()\": ball_origin is required  two-dimensional coordinate, like (x,y) instead of (x,y,z) '
    assert ball_radius>0,'Worng Input in \"generate_PM_LData_list()\": the ball radius must be bigger than 0'
    if len(bright_area_uv_list)!=6:
        print('\033[5;30;42m',end='')
        print('Warning:aquired ',len(bright_area_uv_list),' areas ,recommended aquired 6 areas')
        print('\033[0m')
    
    PM_L_list = [] 
    black_list = []
    
    for idx, light_uv in enumerate(bright_area_uv_list):
        if light_uv:
            ray_in = compute_light_direction(light_uv, ball_origin, ball_radius)
            PM_L_list.append(PM_LData(lid=idx, direction=np.array([ray_in,]), position=np.array(positions[idx] if positions is not None else [0,0,0])))
        else:
            black_list.append(idx)

    return PM_L_list, black_list


def generate_PM_Cam_list(num_of_cam=1):
    '''
    生成一组PM_Cam数据。
    目前生成的相机参数均为单位阵。

    :param num_of_cam: 相机数量
    :return: List(PM_Cam)
    '''
    PM_Cam_list = []
    for i in range(num_of_cam):
        PM_Cam_list.append(PM_Cam(cam_id=i, cam_k=np.eye(3), cam_rt=np.eye(4), cam_frame_uid=-1))
    return PM_Cam_list


def generate_differential_image_list(base_image, pm_image_list, black_list=None,crop=0,dstX=200,dstY=200):
    '''
    生成一组差别图数据。将使用在不同的点光源下拍摄的图像与基础图像（无光源）作差，生成差别图。

    :param base_image: 所有点光源均未点亮时photometric相机采集的图像。float32形式数据，值域[0, 1]
    :param pm_image_list: 点光源分别点亮时photometric相机采集的图像。此列表顺序须与先前生成的PM_LData列表光源顺序相同。float32形式数据，值域[0, 1]
    :param black_list: 被排除的点光源图像index。若某点光源损坏，可将此光源下拍摄的图像排除在外。
    :crop =0 图像不裁剪，=1 裁剪
    ：LenWid 裁剪尺寸 [y_begin,y_end,x_begin,x_end]
    :return: List((id, image_data, uuid))
    ds : 20cm 对应的像素距离
    '''
    assert len(base_image[base_image>0])>0,'the base_image without lights is invalid！'
    
    assert base_image.shape==pm_image_list[0].shape ,'the shapes of base_img and pm_img  are not Consistent'
    assert len(base_image.shape)==2 and len(pm_image_list[0].shape)==2,'required the gray-image input,please convert image to cv2.COLOR_BGR2GRAY'
    #assert isinstance(base_image[0][0],np.float32)== True and isinstance(pm_image_list[0][0][0],np.float32)== True,'Need to image numpu.astype=\'np.float32\''
    #assert base_image[0][0]<=1 and  pm_image_list[0][0][0]<=1,'the pix_element required dvide by 255,such as operate: \'img= img/255\''
    ds=1618
    if len(pm_image_list)!=6:
        print('Warning:aquired ',len(pm_image_list),' pm_images,recommend to require 6 images with lights ')
    if black_list is None:
        black_list = []
    ret = []
    h,w=base_image.shape
    print('base_gray shape',base_image.shape,base_image.dtype)
    if crop==1:
        base_image=base_image[dstY:h-dstY,dstX:w-dstX]
    for idx, image in enumerate(pm_image_list):
        if idx in black_list:
            continue
        if crop==1:
            image=image[dstY:h-dstY,dstX:w-dstX]
        ret.append((idx, image - base_image, idx * 10))

    return ret


def detect_rivets(PM_IData, thres_dist=20, thres_votes=2):
    '''
    调用rivet_detection，检测铆钉与铆钉孔

    :param PM_IData: photometric 采集数据 list of tuples (LID, img， uuid)  [光源的编号， 拍摄的图片(numpy 格式), 照片的uuid]
    :return:
    '''
    #print('PM_IData:',PM_IData[0])
    
    assert len(PM_IData[0])==3,'lack of some elems,the requierd format like (LID(int),img(array),uuid(int))'
    assert thres_votes>0,'the thres_vote should bigger than 0'
    if len(PM_IData)!=6:
        print('\033[5;30;42m',end='')
        print('Warning: aquired',len(PM_IData),'groups data,recommended 6 groups of data',end='')
        print('\033[0m')
    import rivet_detection

    all_rivets = []
    all_holes = []

    print('len of PM_IData',len(PM_IData))
    detector = rivet_detection.RivetDetector()
    detector.set_parameters(
        rivet_area_min=300,
        rivet_area_max=12000,
        hole_area_min=200,
        hole_area_max=12000,
        to_binary_body=180,
        binary_threshold = 80,
        rivet_circle_rate_tsh=0.6,
    )
    # clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(16, 16))

    positions = None
    responses = None
    next_id = 0

    for IData in PM_IData:
        img_id = IData[0]
        img = IData[1]
        img_gray = (img*255).astype(np.uint8)
        img_gray = cv2.equalizeHist(img_gray)

        rivet_points, hole_points = detector.detect_from_body(img_gray)
        all_points = rivet_points.copy()
        all_points.extend(hole_points)

        if positions is None and responses is None:
            if (len(all_points) > 0):
                positions = np.array(all_points).astype(np.float32)
                responses = np.array(range(0, len(positions))).astype(np.float32)
                responses = np.expand_dims(responses, -1)
                next_id = len(positions)
        elif len(all_points) > 0:
            knn = cv2.ml.KNearest_create()
            knn.train(positions.astype(np.float32), cv2.ml.ROW_SAMPLE, responses.astype(np.float32))
            all = np.array(all_points)
            ret, results, neighbours, dist = knn.findNearest(all.astype(np.float32),1)
            positions = np.concatenate([positions, all])
            for i in range(len(results)):
                if dist[i, 0] < thres_dist:
                    responses = np.concatenate([responses, [[results[i, 0], ], ]])
                else:
                    responses = np.concatenate([responses, [[next_id, ], ]])
                    next_id += 1
    #TestGpu('detect Rivet')
    point_groups = [[] for i in range(next_id)]
    if positions is not None:
        for i in range(len(positions)):
            point_groups[int(responses[i, 0])].append(list(positions[i].astype(np.int)))

    filtered_points = []
    for i in range(len(point_groups)):
        if len(point_groups[i]) >= thres_votes:
            filtered_points.append(point_groups[i][0])

    return filtered_points



class PM_LData:
    def __init__(self,**kwargs ):
        '''
        lid: 
        direction: 1*3 ndarray
        '''
        self.LID  = kwargs['lid'] # 光源编号
        self.direction  = kwargs['direction'] # 光源的方向
        self.position = kwargs['position'] if 'position' in kwargs else None


class PM_Cam:
    def __init__(self, **kwargs):
        '''
        这个类描述的是，相机的某一帧对应的内外参
        对于三角相机，这个可以记录每个相机每帧的rt
        对于photometric相机，这个类，单纯用于保存相机内参和相对与三角相机的rt
        cam_id: 相机的id
        cam_k: 相机内参 numpy array: 3*3
        cam_rt: 相机外参 numpy array： 4*4； 都是相对于世界坐标系 cam_rt * p_cam = p_world
        cam_frame_uid: 帧号 ； 当frame uid为-1的时候，表示标定得到的各个相机之间的相对位置， frame 0 也是使用这个数据当作处初始值
        '''
        self.cam_id = kwargs['cam_id']
        self.cam_k = kwargs['cam_k']
        self.cam_rt = kwargs['cam_rt']
        self.cam_frame_uid = kwargs['cam_frame_uid']

    def __str__(self):
        return f'{self.cam_id}, {self.cam_k}, {self.cam_rt}, {self.cam_frame_uid}'


class PhotometricStereoDet:
    '''
    初始化输入：
        lights : 光源位置与编号 list of PM_LData
        cams_init : lsit of PM_cam instance , 所有相机的初始位置，用于得到相机两两之间的rt
        cam_id: 当前用于photometric的相机的id, 和 PM_Cam中相机的id需要对应，计算相机之间的rt就是靠这个id在self.cams中找到对应相机的初始参数
        commited_data : 用于记录commit的结果
        grad_thresh： default=20;
        bboxes_cnt_thresh： default=8;
        bbox_size_thresh： default=100;
        rs_scale： default=5;

    '''
    def __init__(self, **kwargs):
        TestBegin()
        # 本个检测实例对应的相机参数
        # 光源信息
        self.lights = kwargs['lights']
        
        
        # 所有相机的初始位置，用于得到相机两两之间的rt
        # lsit of PM_cam instance
        self.cams_init = kwargs['cams_init']
        
              
              
        # 当前用于photometric的相机的id             
        self.cam_id = kwargs['cam_id']   # 和 PM_Cam中相机的id需要对应，计算相机之间的rt就是靠这个id在self.cams中找到对应相机的初始参数
        #print('self.cam_id',len(self.cam_id))

        # Photoneo相机的id
        self.photoneo_id = kwargs['photoneo_id'] if 'photoneo_id' in kwargs else None   # 和 PM_Cam中相机的id需要对应，计算相机之间的rt就是靠这个id在self.cams中找到对应相机的初始参数

        # 用于记录检测到的结果数据
        self.commited_data = []

        # 用于划痕bbox检测时候的超参

        self.grad_thresh = kwargs['grad_thresh']
        self.bboxes_cnt_thresh = kwargs['bboxes_cnt_thresh']
        self.bbox_size_thresh = kwargs['bbox_size_thresh']
        self.rs_scale = kwargs['rs_scale']

        self.K_size=kwargs['K_size']
        self.Dist = kwargs['Dist']
        self.N = kwargs['N']
        self.Dir=kwargs['Dir']
        
        assert len(self.cams_init)==2,'the num ofk of a cams_init is wrong,requie two cam_init pos'
        assert len(self.lights)==6,'the num of lights is wrong,required 6 lights'
        print('\033[5;30;42m',end='')
        if self.grad_thresh<6 or self.grad_thresh>10:
            print('Warning:the recommended grad_thresh between 7 and 10,this param will Affect detection accuracy，the litter has better accuracy',end='')
        if self.bboxes_cnt_thresh<5 or self.bboxes_cnt_thresh>13:
            print('Warning:the recommended bboxes_cnt_thresh between 5 and 17,this param determine the distance of adjacent bbox，two bbox have less distance will combine to one',end='')
        if self.bbox_size_thresh<8 or self.bbox_size_thresh>18:
            print('Warning:the recommended bboxes_cnt_thresh between 8 and 18,this param determine the preserved bbox\'s area,the less param will detect the the little flaw ',end='')
        if self.rs_scale<4 or self.rs_scale>10:
            print('Warning:the recommended rs_scale between 4 and 10,this param determine Affect detection accuracy and the run time,the bigger param will cost the the less time and lower accuracy ',end='')
        
        print('\033[0m')
        # self.bboxes_cnt_thresh = 8
        # self.bbox_size_thresh = 100
        # self.rs_scale = 5

        # 用于保存当前工作是的变量
        self.normal = None
        self.cur_det_bboxes = None
        self.cur_data = None

        # 缓存深度图方向向量
        self.PN_depth_rays = None
        self.PM_rays = None

        # 腐蚀深度图mask的程度（排除边界）
        self.erosion_kernel = kwargs['erosion_kernel'] if 'erosion_kernel' in kwargs else 5


    def compute_depth_rays(self, height, width):
        '''
        根据Photoneo的内参，计算深度射线方向与位置

        :param height: Photoneo的深度图高度
        :param width: Photoneo的深度图宽度
        :return: None
        '''
        photoneo = filter(lambda x: x.cam_id==self.photoneo_id, self.cams_init)
        photoneo = list(photoneo)[0]
        Ks_tensor = torch.Tensor([photoneo.cam_k, ]).cuda()
        Ts_tensor = torch.Tensor([photoneo.cam_rt, ]).cuda()
        rays, _ = ut.ray_sampling(Ks_tensor, Ts_tensor, [height, width])
        self.PN_depth_rays = rays
        # print('rays', rays)
        # print(Ks_tensor, Ts_tensor)


    def compute_PM_rays(self, height, width):
        '''
        根据Photometric的内参，计算Photometric射线方向与位置

        :param height: Photometric图高度
        :param width: Photometric图宽度
        :return: None
        '''
        photometric = filter(lambda x: x.cam_id==self.cam_id, self.cams_init)
        photometric = list(photometric)[0]
        Ks_tensor = torch.Tensor([photometric.cam_k, ]).cuda()
        Ts_tensor = torch.Tensor([photometric.cam_rt, ]).cuda()
        rays, _ = ut.ray_sampling(Ks_tensor, Ts_tensor, [height, width])
        self.PM_rays = rays
        # print('rays', rays.size())
        # print(Ks_tensor, Ts_tensor)



    def depth_to_PM(self, depth_map, i_height, i_width, down_scale=2):
        '''
        内部函数。
        将Photoneo深度图投影至Photometric相机视角下。

        :param depth_map: Photoneo的深度图。 np.array, float32, [H, W]
        :param i_height: Photometric图像的高度。
        :param i_width: Photometric图像的宽度。
        :param down_scale: 下采样倍数。
        :return: 1. Photometric坐标系下的点云。 2. Photometric下的有效点云mask
        '''
        assert self.PN_depth_rays is not None, 'Depth rays have not been computed!'
        assert self.PM_rays is not None, 'Photometric rays have not been computed!'
        #assert len(depth_map[depth_map>0])>0, 'The Depth_map is incorrect input ,Please Check! ...from Liukh'
        height, width = np.shape(depth_map)
        ray_map_size = self.PN_depth_rays.size()
        assert height*width==ray_map_size[0], 'Sizes of depth map does not match depth rays\'.'

        pn_depth = torch.tensor(depth_map).cuda()
        pn_depth = pn_depth.reshape((height * width, 1))
        world_positions = torch.mul(pn_depth[:], self.PN_depth_rays[:, 0:3])

        non_zero_indexes = world_positions.nonzero()
        non_zero_indexes = torch.unique(non_zero_indexes[:, 0])
        world_positions = world_positions + self.PN_depth_rays[:, 3:6]
        world_positions = world_positions[non_zero_indexes]

        photometric = filter(lambda x: x.cam_id==self.cam_id, self.cams_init)
        photometric = list(photometric)[0]

        K_scaled = np.array(photometric.cam_k) / down_scale
        K_scaled[2, 2] = 1.0
        K_tensor = torch.from_numpy(K_scaled.astype(np.float32)).cuda()
        T_tensor = torch.from_numpy(np.array([
            photometric.cam_rt[0, 2], photometric.cam_rt[1, 2], photometric.cam_rt[2, 2],
            photometric.cam_rt[0, 0], photometric.cam_rt[1, 0], photometric.cam_rt[2, 0],
            photometric.cam_rt[0, 1], photometric.cam_rt[1, 1], photometric.cam_rt[2, 1],
            photometric.cam_rt[0, 3], photometric.cam_rt[1, 3], photometric.cam_rt[2, 3],
        ]).astype(np.float32)).cuda()


        out_depth = torch.zeros(i_height // down_scale , i_width // down_scale ).cuda()
        out_index = torch.zeros(i_height // down_scale , i_width // down_scale , dtype=torch.int32).cuda()
        out_depth, out_index = pcpr.forward(world_positions, K_tensor, T_tensor, out_depth, out_index, 400, 800, 4)

        out_depth_size = out_depth.size()
        out_depth_expand = out_depth.expand([1,1,out_depth_size[0], out_depth_size[1]])

        upsample = torch.nn.Upsample(scale_factor=(down_scale,down_scale))

        final_PM_depth = upsample(out_depth_expand)
        final_PM_depth = final_PM_depth.squeeze()
        # final_PM_depth = out_depth

        f_height, f_width = final_PM_depth.size()
        final_PM_depth_flat = final_PM_depth.reshape((f_height*f_width, 1))
        PM_positions = torch.mul(final_PM_depth_flat[:], self.PM_rays[:, 0:3])
        PM_positions = PM_positions + self.PM_rays[:, 3:6]
        PM_positions = PM_positions.reshape((f_height, f_width, 3))

        norm = PM_positions.norm(dim=2)
        mask = norm!=0.

        del norm, final_PM_depth_flat, final_PM_depth, out_depth_expand, out_depth, out_index, world_positions, non_zero_indexes, pn_depth
        torch.cuda.empty_cache()

        return PM_positions, mask


    def construct_normal(self, PM_IData, PN_DepthData=None, depth_down_scale=1):
        '''
        重建法线。

        :param PM_IData: photometric 采集数据 list of tuples (LID, img， uuid)  [光源的编号， 拍摄的图片(numpy 格式), 照片的uuid]
        :param depth_photoneo: Optional Photoneo相机采集的深度图像
        :return: 1. 重建后的法线图 2. 法线图mask (true 有效 false 无效） 无Photoneo图像时mask为None
        '''
#         print('PN len',len(PN_DepthData[PN_DepthData>0]))
#         print('PN shape',PN_DepthData.shape)
        print('\033[5;30;42m',end='')
        if  PN_DepthData is  None:
            print('Warning:lack of Depth Photo! The program will run without DepthPhoto,which will lead to a bad acurracy! ',end='')
        elif len(PN_DepthData[PN_DepthData>0])<ValidPixels*PN_DepthData.shape[0]*PN_DepthData.shape[1]:
            print('Warning:Too much invalid pixel in Depth_Pthoto Input! Program will run without Depth_Photo! ',end='')
            print('valid /total pixels in DepthPhoto',len(PN_DepthData[PN_DepthData>0]),'/',PN_DepthData.shape[0],'*',PN_DepthData.shape[1],end='')
            PN_DepthData=None
        print('\033[0m')
        #assert isinstance(PM_IData[0][1][0][0],np.float32)==True,'require the img numpy.astype=\'np.float32\' '
        #assert PM_IData[0][1][0][0]<1,'the pix_element should be little than 1,before here ,be sure they have done operater like this:img = img / 255'
        PM_positions = None
        mask = None
        
        if PN_DepthData is not None:
            d_height, d_width = np.shape(PN_DepthData)
            i_height, i_width = np.shape(PM_IData[0][1])
            if self.PN_depth_rays is None:
                self.compute_depth_rays(d_height, d_width)

            if self.PM_rays is None:
                self.compute_PM_rays(i_height, i_width)
            PM_positions, mask = self.depth_to_PM(PN_DepthData, i_height, i_width, down_scale=4)
            # print(PM_depthIdx.cpu().numpy().shape)

        ldirs, imgs = self.prepare_imgs_ldirs(PM_IData, PM_positions, mask)

        imgs = self.filter_img(imgs)
        
        #print('ldirs:',len(ldirs))
        self.normal = self.normal_reconstruct(ldirs, imgs, depth_down_scale=depth_down_scale)
        return self.normal, mask


    def detect_flaw(self, rivits_points, mask=None,bbox_max_size=560,bbox_max_lenth=340):
        '''
        检测缺陷。须先计算法线后才可检测。

        :param rivits_points: 需要排除的铆钉位置列表。将排除这些位置的bbox。
        :return: 缺陷的bbox，检测状态，生成的gradient图。
        '''
        assert self.normal is not None, 'Normals have not been constructed yet. Call construct_normal() first.'
        assert type(rivits_points) ==np.ndarray,'the rivets_point’ type should be \'np.arry\',not the \'list\' type'
        #img_grad = self.normal_gradient(self.normal)
        
        print('normal shape:',self.normal.shape)


        img_grad = self.normal_gradient(self.normal)

        print('img_grad.shape outer:',img_grad.shape)
        down_mask = None
        if self.rs_scale > 1:

            img_expand = np.expand_dims(img_grad.astype(np.float32),axis=0)
            grad_tensor = torch.tensor(img_expand).cuda()
            down_sample = torch.nn.MaxPool2d(self.rs_scale)
            down_grad_tensor = down_sample(grad_tensor)
            down_grad = down_grad_tensor.cpu().numpy().astype(np.uint8)
            down_grad = np.squeeze(down_grad)

            if mask is not None:
                down_mask = (~mask).float().expand((1,-1,-1))
                down_mask = down_sample(down_mask)
                down_mask = (~(down_mask.squeeze().bool())).cpu().numpy()
                down_mask = cv2.erode(down_mask.astype(np.uint8)*255, cv2.getStructuringElement(cv2.MORPH_RECT, (self.erosion_kernel, self.erosion_kernel)))
                down_mask = down_mask.astype(np.bool)

        else:
            down_grad = img_grad

        self.cur_det_bboxes ,bboxCredibility= flaw_bboxes(down_grad, rivits_points, self.rs_scale, self.grad_thresh, self.bboxes_cnt_thresh, self.bbox_size_thresh, down_mask,bbox_max_size,bbox_max_lenth)

        
        torch.cuda.empty_cache()
        TestEnd()
        return np.array(self.cur_det_bboxes), True, img_grad,bboxCredibility


    '''
    将拍摄的数据进行重建和分析

    输入：  
        PM_IData : photometric 采集数据 list of tuples (LID, img， uuid)  [光源的编号， 拍摄的图片(numpy 格式), 照片的uuid]
        rivetpoints: list of 2D rivet points 用来去除铆钉的bbox

    输出：
        1. 重建出的法线图片
        2. 检测出的缺陷list ,  list of bbox (x,y,w,h)
        3. 检测状态，boolean， 自我评估结果是否有效
    '''
    def reconstruction(self, PM_IData, rivits_points, depth_photoneo=None):
        '''
        先生成法线图，再进行检测。
        是construct_normal()与detect_flaw()的顺序执行。

        :param PM_IData: photometric 采集数据 list of tuples (LID, img， uuid)  [光源的编号， 拍摄的图片(numpy 格式), 照片的uuid]
        :param rivits_points: list of 2D rivet points 用来去除铆钉的bbox
        :param depth_photoneo: Optional Photoneo相机采集的深度图像
        :return: 1. 重建出的法线图片 2. 检测出的缺陷list ,  list of bbox (x,y,w,h) 3. 检测状态，boolean， 自我评估结果是否有效 4. 法线Gradient
        '''
        normal = self.construct_normal(PM_IData)
        bboxes, status, grad = self.detect_flaw(rivits_points)
        return normal, bboxes, status, grad


    '''
    将最近的一次调用reconstruction 的结果保存起来

    输入：
        meta_data : 从RivetModelMatching 获取的数据
        cam: PM_Cam的实例，用于得到当前里photometric相机最近的三角相机的位置，因为有了这个位置之后，加上之前知道的这两个相机之间的相对关系，
             就可以知道photometric相机此时的位置，这样后续才可以将bbox反投影到标准数模上面
    '''
    def commit(self,  cam,  meta_data = None):

        cam_cur = None
        cam_nn = None
        for c in self.cams_init:
            if(c.cam_id == cam.cam_id):
                cam_nn = c

            if(c.cam_id == self.cam_id):
                cam_cur = c

        if cam_cur == None or cam_nn == None:
            return False

        # p_cur = cam_rt_nn_2_cur*p_nn
        cam_rt_nn_2_cur = np.matmul(np.linalg.inv(cam_cur.cam_rt), cam_nn.cam_rt)

        # p_world = cur_photometric_rt*p_cam_photometric
        cur_photometric_rt = np.linalg.inv(np.matmul(cam_rt_nn_2_cur, cam.cam_rt))

        # 将 （rt, normal, detected bboxes）添加到历史记录
        self.commited_data.append((cam_cur.cam_k, cur_photometric_rt, self.normal, self.cur_det_bboxes))

        return True






    '''
        分析所有commit 过的数据
    '''
    def analyze(self):
        return



    #############################################################################
    # private functions
    def prepare_imgs_ldirs(self, PM_IData, points=None, mask=None):
        '''
        intput :
        photometric 采集数据 list of tuples (LID, img，uuid)  [光源的编号， 拍摄的图片(numpy 格式), 照片的uuid]
        output :
        imgs: n*h*w
        ldirs: tensor n*3
        '''
        imgs = []
        ldirs = []
        for d in PM_IData:
            imgs.append(np.expand_dims(d[1], axis=0))
            light = None
            for l in self.lights:
                if (l.LID == d[0]):
                    light = l
#                     print('light type:',type(light))
#                     print('l.direction dtype:',type(l.direction))
#                     print('l.direction dtype:',l.direction.dtype)
                    break

            if points is not None:
                height, width, dims = points.size()
                dirs = points - torch.tensor(light.position).cuda()
                mask_expanded = mask.unsqueeze(2).expand(-1, -1, 3)
                if mask is not None:
                    dirs[~mask_expanded] = 0.
                dirs = dirs.reshape(height*width, 3)
                dirs = dirs.unsqueeze(1)
                ldirs.append(dirs)
            else:
                #print('light.direction dtype:',light.direction.dtype)
                ldirs.append(torch.tensor(light.direction).cuda().float())
                #print('ldirs:',len(ldirs))

        imgs = np.concatenate(imgs, axis=0)
        if points is not None:
            ldirs = torch.cat(ldirs, 1)
            # ldirs = ldirs / ldirs.norm(dim=-1).unsqueeze(-1)
        else:
            ldirs = torch.cat(ldirs, 0)
        # print(ldirs.shape)
        # exit()
        print('2  ldirs:',len(ldirs))

        # 保存原始数据， commit时会把原始数据也添加进去
        self.cur_data = PM_IData

        return ldirs, imgs


    def normal_reconstruct(self, lightdir, imgs, depth_down_scale=16):
        '''
        input:
        lightdir: tensor (n*3) or ((h*w)*n*3) if depth enabled
        imgs: nparray  (n*h*w) , np.float32, (0.0-1.0)
        n is the count of captured imgs

        output:
        normal map (h*w*3) np.uint8 (0, 255)
        '''
        img_shape = imgs.shape
        lightdir_shape = lightdir.shape
        normal_flat = None
        
        #print('lightdir shape:',lightdir.shape)
        
        if len(lightdir_shape) == 2:
            lightsdir_inv = np.linalg.pinv(lightdir.cpu().numpy())
            lightsdir_inv = torch.tensor(lightsdir_inv).cuda()
            
            #print('lightdir_shape:',lightsdir_inv.shape)
            imgs_flat = torch.tensor(imgs.reshape((imgs.shape[0], -1))).cuda()
            #print('imgs_flat:',imgs_flat.shape)
            normal_flat = torch.matmul(lightsdir_inv, imgs_flat)

        elif len(lightdir_shape) == 3:
            lightdir_mat = lightdir.reshape((img_shape[1], img_shape[2], img_shape[0], 3))
            height, width, num_lights, num_channels = lightdir_mat.size()

            h_remaining = height % depth_down_scale
            w_remaining = width % depth_down_scale

            expanded_height = height
            expanded_width = width

            if h_remaining != 0:
                first_row = lightdir_mat[0:1, :, :, :].expand(depth_down_scale - h_remaining, width, num_lights, num_channels)
                lightdir_mat = torch.cat([first_row, lightdir_mat], 0)
                expanded_height += depth_down_scale - h_remaining

            if w_remaining != 0:
                left_column = lightdir_mat[:, 0:1, :, :].expand(height, depth_down_scale - w_remaining, num_lights, num_channels)
                lightdir_mat = torch.cat([left_column, lightdir_mat], 1)
                expanded_width += depth_down_scale - w_remaining

            lightdir_mat = lightdir_mat.permute([2,3,0,1])
            lightdir_mat = torch.nn.functional.interpolate(lightdir_mat, scale_factor=(1/depth_down_scale))
            lightdir_mat = lightdir_mat.permute([2,3,0,1])
            lightsdir = lightdir_mat.cpu().numpy()

            del lightdir_mat
            torch.cuda.empty_cache()

            _h, _w, _n, _c = lightsdir.shape
            lightsdir = np.reshape(lightsdir, (_h*_w, _n, _c))
            lightsdir_inv = np.linalg.pinv(lightsdir)
            # print(torch.cuda.memory_summary())

            lightsdir_inv_t = torch.tensor(lightsdir_inv).cuda().reshape(_h, _w, _c, _n)

            lightsdir_inv_t = lightsdir_inv_t.permute([2,3,0,1])
            # print(torch.cuda.memory_summary())
            lightsdir_inv_t = torch.nn.functional.interpolate(lightsdir_inv_t, scale_factor=depth_down_scale, mode='nearest')
            lightsdir_inv_t = lightsdir_inv_t.permute([2,3,0,1])
            lightsdir_inv = lightsdir_inv_t
            lightsdir_inv = lightsdir_inv[expanded_height-height:, expanded_width-width:, :, :]
            lightsdir_inv = torch.reshape(lightsdir_inv, (img_shape[1]*img_shape[2], 3, img_shape[0] ))
            imgs_flat = imgs.reshape((imgs.shape[0], -1))
            imgs_flat = torch.tensor(np.expand_dims(imgs_flat.T, 2)).cuda()

            # normal_flat = np.matmul(lightsdir_inv, imgs_flat)
            normal_flat = torch.matmul(lightsdir_inv, imgs_flat)
            normal_flat = normal_flat.squeeze()
            normal_flat = normal_flat.T

            del lightsdir_inv, imgs_flat
            torch.cuda.empty_cache()

        # normal_flat_norm = np.linalg.norm(normal_flat, axis=0)
        normal_flat_norm = torch.norm(normal_flat, dim=0)
        normal_flat[:, normal_flat_norm>0] = normal_flat[:, normal_flat_norm>0]/normal_flat_norm[normal_flat_norm>0]
        normal = normal_flat.reshape((3, img_shape[1], img_shape[2])).cpu().numpy()
        normal = np.moveaxis(normal, 0, -1)
        normal = (normal + 1)/2*255
        normal = normal.astype(np.uint8)
        normal = normal[:, :, ::-1]

        torch.cuda.empty_cache()
        return normal

    def ReWeightedGrad(self,img,Differ=15):
        w,h,ch=img.shape
        grad=np.zeros([w,h])
        for i in range(w):
            for j in range(h):
                b,g,r=img[i][j]
                if b>Differ and g<Differ and r<Differ:
                    grad[w][h]=b
                elif g>Differ and b<Differ and r<Differ:
                    grad[w][h]=g
                elif r>Differ and b<Differ and g<Differ:
                    grad[w][h]=r
                else:
                    grad[w][h]=(b+g+r)/3
        return grad
    def normal_gradient( self, img):
       
        tb=TEST_begin()
        sobel_kernelxx = np.array([
            [
                [[-1, 0, 1],[-2, 0, 2],[-1,0,1]],
                [[ 0, 0, 0],[ 0, 0, 0],[0,0,0]],
                [[ 0, 0, 0],[ 0, 0, 0],[0,0,0]]],
            [
                [[ 0, 0, 0],[ 0, 0, 0],[0,0,0]],
                [[-1, 0, 1],[-2, 0, 2],[-1,0,1]],
                [[ 0, 0, 0],[ 0, 0, 0],[0,0,0]]],
            [
                [[ 0, 0, 0],[ 0, 0, 0],[0,0,0]],
                [[ 0, 0, 0],[ 0, 0, 0],[0,0,0]],
                [[-1, 0, 1],[-2, 0, 2],[-1,0,1]]]
            ],dtype='float32')
        sobel_kernelyy = np.array([
            [
                [[-1,-2,-1],[ 0, 0, 0],[1, 2, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[-1,-2,-1],[ 0, 0, 0],[1, 2, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[-1,-2,-1],[ 0, 0, 0],[1, 2, 1]]]
        ], dtype='float32')

        #print(img.shape)
        img=np.transpose(img,(2,0,1))
        img=torch.FloatTensor(img.copy()).cuda()
        img = img.unsqueeze(0)
        

        sobel_kernelxx = torch.tensor(sobel_kernelxx).cuda()
        sobel_kernelyy = torch.tensor(sobel_kernelyy).cuda()
        
        
        gx = F.conv2d(img, sobel_kernelxx, bias=None, stride=1, padding=1)
        gy = F.conv2d(img, sobel_kernelyy, bias=None, stride=1, padding=1)
        
        del sobel_kernelxx,sobel_kernelyy
        gx=torch.abs(gx)
        gy=torch.abs(gy)

        gx = gx.squeeze(0)
        gy = gy.squeeze(0)
        gx=gx.permute((1,2,0))
        gy=gy.permute((1,2,0))
        
        
        ##let egde elemts are 0
        gx[0, :] = 0
        gx[:, 0] = 0
        gx[gx.shape[0] - 1, :] = 0
        gx[:, gx.shape[1] - 1] = 0

        gy[0, :] = 0
        gy[:, 0] = 0
        gy[gy.shape[0] - 1, :] = 0
        gy[:, gy.shape[1] - 1] = 0
        print('gx type',type(gx))

       # grad = gx * 0.5 + gy * 0.5
        grad=torch.maximum(gx,gy)
        print('Running Here')
#         gx=gx.cpu().numpy()
#         gy=gy.cpu().numpy()
#         gradXM = np.mean(gx, axis=2)
#         gradYM = np.mean(gy, axis=2)
       
#         cv2.imwrite(f'./gradXM.png',gradXM)
#         cv2.imwrite(f'./gradYM.png',gradYM)
        del gx,gy
        torch.cuda.empty_cache()
        grad=torch.max(grad,axis=2)[0]
        grad=grad.cpu().numpy()

#         grad = np.mean(grad, axis=2)

#         grad = grad.astype(np.uint8)
        TEST_end('nomal grad Sobel2 in pytorch',tb)
        cv2.imwrite(f'../OutPut/{self.Dir}-grad{self.grad_thresh}-{self.K_size}-{self.Dist}-{self.N}.png',grad)
        #grad=cv2.GaussianBlur(grad,(11,11),0)
        #grad=cv2.bilateralFilter(grad,3,350,1)
        #print('bilateralFilter Blur Done!')
        #cv2.imwrite(f'../OutPut/gradBlurOrigin.png', grad)
        return grad
    

    def filter_img(self, imgs, filter_cnt=2):
        '''
        bilateral filter images

        input:
        imgs: ndarray (n*h*w), np.float32, (0.0, 1.0)
        filter_cnt: filter_times
        size: filter kernel size

        output:
        imgs: ndarray (n*h*w), np.float32, (0.0, 1.0) filtererd
        '''

        imgs = imgs.astype(np.float32)
        for cnt in range(filter_cnt):
            for i in range(imgs.shape[0]):
                imgs[i, :, :] = cv2.bilateralFilter(imgs[i, :, :], self.K_size, self.Dist, self.N)

        return imgs
