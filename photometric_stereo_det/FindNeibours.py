import math
import cv2
#import photometric_stereo_det

def search_neiborhood2D(POINTS,NeiborDist_Thresh=60,MinDist=20,Ds=10):
    '''
    找到相邻较近的铆钉
    params: POINTS:    已检测的铆钉坐标集合
           NeiborDist_Thresh: 设定的距离阈值（部件表面可能会有些弯曲，导致平面距离小于真实距离，应将该阈值调整为比设定值略小一些）
           MinDist:    最小的距离，用以排除铆钉检测过程中生成的过近的铆钉坐标

    return: res: 近邻集合
    '''
    assert len(POINTS)>0,'铆钉集合数量必须大于0'
    print('Search Neibourhood----------------------------')
    NeiborDist_Thresh=(NeiborDist_Thresh-Ds)*2.5
    print('NeiborDist_Thresh:',NeiborDist_Thresh)
    Dist_MANHATTAN=NeiborDist_Thresh*1.414
    IndxHasNeibors=[]
    TheNeibours=[]
    Flag=[False for x in range(len(POINTS))]
    cur=-1
    for idx in range(len(POINTS)):
        x = POINTS[idx][0]
        y = POINTS[idx][1]
        
        TheNeibours.append([])
        cur=cur+1
        for idx2 in range(len(POINTS)):
            if idx!=idx2 and Flag[idx2]==False:
                x2=POINTS[idx2][0]
                y2=POINTS[idx2][1]

                distx=abs(x-x2)
                disty=abs(y-y2)
                dist_manhattan=distx+disty

                if dist_manhattan<Dist_MANHATTAN:
                    temp=distx**2 + disty**2
                    dist=math.sqrt(temp)
                    
                    if(MinDist<dist<NeiborDist_Thresh):                        
                        Flag[idx2]=True
                        if len(TheNeibours[cur])==0:
                            TheNeibours[cur].append(POINTS[idx])
                        TheNeibours[cur].append(POINTS[idx2])
    res=[]
    for i in range(len(TheNeibours)):
        if len(TheNeibours[i])>0:
            res.append(TheNeibours[i])
            
        
    return res


def DrawLineInNeibors(img,Neibours):
    '''
    在图像上将相邻的铆钉连线
    params img:        源图像
           Neibours:   近邻集合

    return: NeibourImg: 标注好的图像
    '''
    for neibour in Neibours:
        for idx in range(1,len(neibour)):
            cv2.line(img, (neibour[0][0],neibour[0][1]), (neibour[idx][0],neibour[idx][1]), 25, 3)           
        
    return img

# #---------------------------------------------------------------------------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------------------------------------------------------------------------
# #PhotometricDet already has below dets


# def generate_differential_image_list(base_image, pm_image_list, black_list=None):
#     '''
#     生成一组差别图数据。将使用在不同的点光源下拍摄的图像与基础图像（无光源）作差，生成差别图。

#     :param base_image: 所有点光源均未点亮时photometric相机采集的图像。float32形式数据，值域[0, 1]
#     :param pm_image_list: 点光源分别点亮时photometric相机采集的图像。此列表顺序须与先前生成的PM_LData列表光源顺序相同。float32形式数据，值域[0, 1]
#     :param black_list: 被排除的点光源图像index。若某点光源损坏，可将此光源下拍摄的图像排除在外。
#     :return: List((id, image_data, uuid))
#     '''
#     if black_list is None:
#         black_list = []
#     ret = []
#     for idx, image in enumerate(pm_image_list):
#         if idx in black_list:
#             continue
#         ret.append((idx, image - base_image, idx * 10))

#     return ret

# def detect_rivets(PM_IData, thres_dist=20, thres_votes=2):
#     '''
#     调用rivet_detection，检测铆钉与铆钉孔

#     :param PM_IData: photometric 采集数据 list of tuples (LID, img， uuid)  [光源的编号， 拍摄的图片(numpy 格式), 照片的uuid]
#     :return:
#     '''
#     #print('PM_IData:',PM_IData[0])
    
#     assert len(PM_IData[0])==3,'lack of some elems,the requierd format like (LID(int),img(array),uuid(int))'
#     assert thres_votes>0,'the thres_vote should bigger than 0'
#     if len(PM_IData)!=6:
#         print('Warning: aquired',len(PM_IData),'groups data,recommended 6 groups of data')
#     import rivet_detection

#     all_rivets = []
#     all_holes = []

#     detector = rivet_detection.RivetDetector()
#     detector.set_parameters(
#         rivet_area_min=200,
#         rivet_area_max=12000,
#         hole_area_min=200,
#         hole_area_max=12000,
#         to_binary_body=180,
#         binary_threshold = 80,
#         rivet_circle_rate_tsh=0.6,
#     )
#     # clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(16, 16))

#     positions = None
#     responses = None
#     next_id = 0

#     for IData in PM_IData:
#         img_id = IData[0]
#         img = IData[1]
#         img_gray = (img*255).astype(np.uint8)
#         img_gray = cv2.equalizeHist(img_gray)

#         rivet_points, hole_points = detector.detect_from_body(img_gray)
#         all_points = rivet_points.copy()
#         all_points.extend(hole_points)

#         if positions is None and responses is None:
#             if (len(all_points) > 0):
#                 positions = np.array(all_points).astype(np.float32)
#                 responses = np.array(range(0, len(positions))).astype(np.float32)
#                 responses = np.expand_dims(responses, -1)
#                 next_id = len(positions)
#         elif len(all_points) > 0:
#             knn = cv2.ml.KNearest_create()
#             knn.train(positions.astype(np.float32), cv2.ml.ROW_SAMPLE, responses.astype(np.float32))
#             all = np.array(all_points)
#             ret, results, neighbours, dist = knn.findNearest(all.astype(np.float32),1)
#             positions = np.concatenate([positions, all])
#             for i in range(len(results)):
#                 if dist[i, 0] < thres_dist:
#                     responses = np.concatenate([responses, [[results[i, 0], ], ]])
#                 else:
#                     responses = np.concatenate([responses, [[next_id, ], ]])
#                     next_id += 1

#     point_groups = [[] for i in range(next_id)]
#     for i in range(len(positions)):
#         point_groups[int(responses[i, 0])].append(list(positions[i].astype(np.int)))

#     filtered_points = []
#     for i in range(len(point_groups)):
#         if len(point_groups[i]) >= thres_votes:
#             filtered_points.append(point_groups[i][0])

#     return filtered_points

def demo():
    
    base = cv2.imread(f'/root/Rivets/comacphotometric-depth_test/data/5_104_2_1_1/1_{PID}_0_00_00000_thumbnail.png')
    print(f'/root/Rivets/comacphotometric-depth_test/data/5_104_2_1_1/1_{PID}_0_00_00000_thumbnail.png')
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    base_gray = base_gray.astype(np.float32) / 255
    
    data = []
    for j in range(1, 7):
        img = cv2.imread(f'/root/Rivets/comacphotometric-depth_test/data/5_104_2_1_1/1_{PID}_{j}_00_0000{j+1}_thumbnail.png')
        print(f'/root/Rivets/comacphotometric-depth_test/data/5_104_2_1_1/1_{PID}_{j}_00_0000{j+1}_thumbnail.png')
        if j==2:
            res=img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255
        data.append(gray)
    
    final_data = generate_differential_image_list(base_gray, data)
    
    rivets = detect_rivets(final_data, thres_votes=1)
    
    Neibours=search_neiborhood2D(rivets)
    res=DrawLineInNeibors(res,Neibours)
    #cv2.imwrite(f'./Test_result.png', res)
