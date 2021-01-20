import os 
import cv2
import numpy as np
import copy
import time
debug = False
CallCount=0
DIST=1
AccuracyGradMax=16
AccuracyGradMin=7
def TEST_begin(): 
    return time.time()



def TEST_end(nameStr,btime):
    print()
    print("---------the def ["+nameStr +"]  -----------")
    print("运行时间%.6f"%(time.time()-btime))

    print()
def TEST_end2(nameStr,btime):
    print()
    global CallCount
    CallCount+=1
    print("---------the def [",nameStr, " ",CallCount,"]  -----------")
    print("运行时间%.6f"%(time.time()-btime))

    print()
def Credibility(grad):
    return grad
    if grad > AccuracyGradMax:
        Credibility=100
        return Credibility
    x=grad
    #linear 
    k=1/(AccuracyGradMax-AccuracyGradMin)
    y=k*(x-AccuracyGradMax)+1
    #print('grad:',round(grad,2),' y:',round(y,2))
    Credibility=round(y,2)
    
    return int(Credibility*100)
    
    
def CountGrad(normal,idx,nodes):
    gradTotal=0
    MaxGrad=-1
    for i,node in enumerate(nodes):
        x,y=idx[node]
        gradTotal+=normal[y][x]
        if normal[y][x]>MaxGrad:
            MaxGrad=normal[y][x]
        
    grad=gradTotal/len(nodes)
    
    #print('len of nodes :',len(nodes))
    return MaxGrad

def bboxCountGrad(idx_changed,grads):
    Grad=0
    MaxGrad=-1
    for i in idx_changed:
        Grad+=grads[i]
        if MaxGrad<grads[i]:
            MaxGrad=grads[i]
        
    #return Grad/len(idx_changed)
    return MaxGrad
    
    
def construct_graph(grad_img,vis_mask):
    '''
    construct linked list graph according to normal gradient mask

    input:
    vis_mask: n*m

    output:
    graph: linked list version
    '''
#    tb=TEST_begin();
    idx = []
    reverse_map = []
    cnt = 0
    for x in range(vis_mask.shape[1]):
        reverse_row = []
        for y in range(vis_mask.shape[0]):
            if(vis_mask[y, x] == 1):
                idx.append((x, y))
                reverse_row.append(cnt)
                cnt += 1
            else:
                reverse_row.append(-1)
        reverse_map.append(reverse_row)

    edge_num = 0
    graph = [[] for i in range(len(idx))]
    for x in range(vis_mask.shape[1]-1):
        for y in range(vis_mask.shape[0]-1):
            if(vis_mask[y, x] == 1):

                if(vis_mask[y+1, x] == 1):
                    # idx1 = idx.index((x, y))
                    # idx2 = idx.index((x, y+1))
                    idx1 = reverse_map[x][y]
                    idx2 = reverse_map[x][y+1]
                    graph[idx1].append(idx2)
                    graph[idx2].append(idx1)

                    edge_num += 1

                if(vis_mask[y, x+1] == 1):
                    # idx1 = idx.index((x, y))
                    # idx2 = idx.index((x+1, y))
                    idx1 = reverse_map[x][y]
                    idx2 = reverse_map[x+1][y]

                    graph[idx1].append(idx2)
                    graph[idx2].append(idx1)

                    edge_num += 1

    if debug:
        print('cur img has ', len(idx), ' pixels, constructed graph has ', \
            edge_num, ' edges.')
#     grad=[]
#     cur=0
#     for i in range(len(idx)):
#         grad_count=0
#         lenNode=len(graph[i])
#         if lenNode!=0:
#             for j in range(lenNode):
#                 x,y=idx[graph[i][j]]
#                 grad_count+=grad_img[y][x]
#             grad.append(grad_count/lenNode)
#             cur+=1
#         else: 
#             grad.append(-1)
    

#     #print('graph:',graph)
#     #print('grad:',grad)
#     print('area cur:',cur)
    return idx, graph



def bfs(graph, cur_visted, cur_node):
    '''
    depth first search

    input: 
    graph: list of list, linked list version graph
    cur_node: currently visiting node
    cur_visted: visited flag

    output:
    cur_visted: modified.
    '''
    # if cur_visted[cur_node] == 1:
    # 	return
    # else:
    # 	cur_visted[cur_node] = 1
    # 	for node in graph[cur_node]:
    # 		bfs(graph, cur_visted, node)

    idx_changed = []
    q = []
    q.append(cur_node)
    cur_visted[cur_node] = 1
    idx_changed.append(cur_node)
    while(len(q) > 0):
        top = q[0]
        q.pop(0)
        for node in graph[top]:
            if cur_visted[node] == 0:
                cur_visted[node] = 1
                q.append(node)
                idx_changed.append(node)
    return idx_changed


def bfs_linked_matrix(graph, cur_visited, cur_node):
    idx_changed = []
    q = []
    q.append(cur_node)
    cur_visited[cur_node] = 1
    idx_changed.append(cur_node)
    TotalGrad=0
    while(len(q)>0):
        # print(q)
        top = q[0]
        q.pop(0)
        for idx, node in enumerate(graph[top]):
            # print(top, idx, node)
            if node:
                if cur_visited[idx] == 0:
                    cur_visited[idx] = 1
                    q.append(idx)
                    idx_changed.append(idx)
                    
                    
    return idx_changed


def bbox(list_xy, w, h):
    '''
    generate bounding box from discreet selected points.

    input:
    list_xy: list of points [(x, y), ...]
    w: image width as boundary
    h: imgae height as boundary
    output:
    bbox: [x, y, b_w, b_h]
    '''
#    tb=TEST_begin()
    min_x = w + 1
    max_x = -1
    min_y = h + 1
    max_y = -1

    for x, y in list_xy:
        if(x <= min_x):
            min_x = x

        if(x >= max_x):
            max_x = x

        if(y <= min_y):
            min_y = y

        if(y >= max_y):
            max_y = y 
    return (min_x, min_y, max_x - min_x, max_y - min_y, min_x / 2 + max_x / 2, min_y / 2 + max_y / 2)


def bbox_dist_manhatton(bbx1, bbx2):
    '''
    find manhatton distance between two bounding bboxes
    intput:
    bbx1: bbox 1
    bbx2: bbox 2

    output:
    distance
    '''

    # max_x1 = max(bbx1[0], bbx2[0])
    max_x1 = bbx1[0] if bbx1[0] >= bbx2[0] else bbx2[0]

    r_1 = bbx1[0] + bbx1[2]
    r_2 = bbx2[0] + bbx2[2]
    # min_x2 = min(bbx1[0] + bbx1[2], bbx2[0] + bbx2[2])
    min_x2 = r_1 if r_1 <= r_2 else r_2
    # dis_x = max(max_x1 - min_x2, 0)
    dis_x = max_x1 - min_x2
    if dis_x < 0:
        dis_x = 0

    # max_y1 = max(bbx1[1], bbx2[1])
    max_y1 = bbx1[1] if bbx1[1] >= bbx2[1] else bbx2[1]

    b_1 = bbx1[1] + bbx1[3]
    b_2 = bbx2[1] + bbx2[3]

    # min_y2 = min(bbx1[1] + bbx1[3], bbx2[1] + bbx2[3])
    min_y2 = b_1 if b_1 <= b_2 else b_2
    # dis_y = max(max_y1 - min_y2, 0)
    dis_y = max_y1 - min_y2
    if dis_y < 0:
        dis_y = 0

    return dis_x + dis_y


def obtain_bboxes(normal,idx, graph, w, h,Min_edge=0):
    '''
    find all bbox in graph that built from original normal gradient mask 

    input:
    idx: number of nodes in graph 
    graph: list of list 
    w: image width as boundary
    h: image height as boundary

    output:
    bboxes: list of bounding box
    '''
#    tb=TEST_begin()    
    cur_visted = [0 for i in range(len(idx))]
    bboxes = []
    all_visted = 0
    cur_node_idx = 0
    # while( sum(cur_visted) < len(idx)):
    grads=[]
    while( cur_node_idx < len(idx)):
        if (cur_visted[cur_node_idx] == 0):
            # cur_node = cur_visted.index(0)
            cur_node = cur_node_idx
            # old_visited = copy.copy(cur_visted)
            idx_changed = bfs(graph, cur_visted, cur_node)
            
            grad=CountGrad(normal,idx,idx_changed)
            if len(idx_changed)>3:#Exclude the noise  pixels
                cur_slt_pixels = [idx[i] for i in idx_changed]
                print(f'--------------------------------------------idxchanged len:{len(idx_changed)}')
                grads.append(grad)

                new_bbox=bbox(cur_slt_pixels, w, h)
                #if new_bbox[2]>Min_edge and new_bbox[3]>Min_edge:
                bboxes.append(new_bbox)

                all_visted += len(cur_slt_pixels)
        else:
            pass

        cur_node_idx += 1

    if debug:
        print('valid: all_visted node ', all_visted)
        print('obtained ', len(bboxes), ' bboxes')
    #TEST_end('obtain_bbox',tb)
    
  
    return bboxes,grads


def construct_bbox_graph(bboxes, thresh=5):
    '''
    view each bbox as a graph node, according to thresh to build bboxes graph
    input:
    bboxes: list of bounding boxes generated from normal gradient mask
    thresh: distance thresh between every two bboxes

    output:
    bbox graph: list of list
    '''
    
    n_bboxes = np.array(bboxes)
    print('n_bboxes shape:',n_bboxes.shape)
    
        
    dist = np.abs(n_bboxes[:, None, 4:6] - n_bboxes[:, 4:6]) - (n_bboxes[:, None, 2:4] / 2 + n_bboxes[:, 2:4] / 2)
    
    dist[dist<0] = 0
    dist = dist.sum(-1)

    return dist< thresh



def find_neighboring_bboxes(bbox_idx, bbox_graph,grads):
    '''
    according to bboxes graph to find which bboxes should be connected
    input:
    bbox_idx: number of nodes in bbox_graph
    bbox_graph: list of list

    output:
    bboxes_connection: list of lsit -> neighboring bboxes groups
    '''
#    tb=TEST_begin()
    cur_visted = [0 for i in range(len(bbox_idx))]
    bboxes_connection = []
    all_visted = 0
    
    bboxGrads=[]
    while( sum(cur_visted) < len(bbox_idx)):

        cur_node = cur_visted.index(0)

        idx_changed = bfs_linked_matrix(bbox_graph, cur_visted, cur_node)
        
        bboxGrad=bboxCountGrad(idx_changed,grads)
        bboxGrads.append(bboxGrad)
        
        bboxes_connection.append(idx_changed)
        all_visted += len(idx_changed)
    
    if debug:
        print('valid: all_visted ', all_visted)
        print('valid: len(bbox_idx):', len(bbox_idx))
        print(len(bboxes_connection), ' bboxes_connection')

    return bboxes_connection,bboxGrads



def expand_bbox(bbx1, bbx2):
    '''
    combine two bbox into one
    input:
    bbx1: bbx1
    bbx2: bbx2

    output:
    bbx: merged bounding box 
    '''

    min_x1 = min(bbx1[0], bbx2[0])
    max_x2 = max(bbx1[0] + bbx1[2], bbx2[0] + bbx2[2])

    min_y1 = min(bbx1[1], bbx2[1])
    max_y2 = max(bbx1[1] + bbx1[3], bbx2[1] + bbx2[3])

    return (min_x1, min_y1, max_x2-min_x1, max_y2-min_y1)



def conbine_bboxes(bboxes_connection, bboxes,Min_edge=0):
    '''
    merge all bounding boxes in a bboxes_connection,

    input:
    bboxes_connection: list of list representing which bboxes should be combined
    bboxes: list of bbox (x, y, b_x, b_y) 
    output:
    final_bboxes: merged bounding box , list of bbox (x, y, b_x, b_y) 
    '''

    final_bboxes = []
    
    for i in range(len(bboxes_connection)):
        cur_bbox = bboxes[bboxes_connection[i][0]]
        for j in bboxes_connection[i][1:]:
            
            #if (bboxes[j][2]>Min_edge or bboxes[j][3]>Min_edge ) and ( cur_bbox[2]>Min_edge or cur_bbox[3]>Min_edge):
                cur_bbox = expand_bbox(cur_bbox, bboxes[j])

        final_bboxes.append(cur_bbox)

    return final_bboxes



def flaw_bboxes(img,rivets_points, rs_scale, grad_thresh, bboxes_cnt_thresh, bbox_size_thresh,radius, mask=None,
                bbox_min_edge=2,bbox_max_size=560,bbox_max_lenth=340,bbox_max_height=100,CloseDelete=20):
    '''
    find flaw bboxes from normal gradient 

    input: 
    img: ormal gradient , h*w 
    rivets_points: n*2, rivits_poitns used to remove bounding box of rivits_poitns
    rs_scale: int, resize scale value, resize to small size to reduce computation to accelerate
    grad_thresh: threshold to normal gradient to get a gradient mask
    bboxes_cnt_thresh: bbox distance thresh for connecting bboxes
    bbox_size_thresh: used to remove over-small bounding box

    output:
    filter_bboxes: detected flaw bboxes 
    '''

    
#     print('\033[5;30;42m',end='')
#     print('*************************',end='')
#     print('Using flawBox2- THE INITIAL',end='')
#     print('*************************',end='')
#     print('\033[0m')
#     print()
    if debug:
        print(img.shape)
    print('grad >0:',img[img>0])

    img[img < grad_thresh] = 0
    vis_mask = np.zeros(img.shape)
    vis_mask[img>0] = 1

    
    print('vis_mask>0:',vis_mask[vis_mask>0])
    if mask is not None:
        vis_mask[~mask] = 0

    # Exclude detected rivet areas from mask.
    # for idx in range(rivets_points.shape[0]):
    #     radius = 50 / rs_scale
    #     x_left = rivets_points[idx, 0] / rs_scale - radius
    #     x_right = x_left + 2 * radius
    #     y_top = rivets_points[idx, 1] / rs_scale - radius
    #     y_bottom = y_top + 2 * radius
    #     # print(int(x_left),int(x_right), int(y_top),int(y_bottom))
    #     vis_mask[ int(y_top):int(y_bottom), int(x_left):int(x_right) ] = 0
        #print(vis_mask)
        #cv2.imshow('mask', vis_mask)
    # cv2.waitKey(0)

    idx, graph = construct_graph(img,vis_mask)
    bboxes,grads= obtain_bboxes(img,idx, graph, vis_mask.shape[1], vis_mask.shape[0])

    bbox_idx = [0 for i in range(len(bboxes))]
    bbox_graph = construct_bbox_graph(bboxes, bboxes_cnt_thresh)
    
    bbox_connection,bboxGrads= find_neighboring_bboxes(bbox_idx, bbox_graph, grads) 

    final_bboxes = conbine_bboxes(bbox_connection, bboxes)
    # bboxGrads=grads
    # final_bboxes=bboxes

    ### remove over-small bounding bboxes and bboxes of rivits
    filter_bboxes = []
    bboxCredibility= []
    
    print('Box size',bbox_size_thresh,' rs_scale:',rs_scale)
# ####### Show All Bbox ########################
#     for i,bbox in enumerate(final_bboxes):
#        # print('bbox:',bbox)
#        #if (bbox_max_size>bbox[2] + bbox[3] > bbox_size_thresh/rs_scale) and bbox[2]<bbox_max_lenth and bbox[3]<bbox_max_height:
#         #if (bbox[2] + bbox[3] > bbox_size_thresh/rs_scale)  :
#             print('grad:',bboxGrads[i],' bbox[2]:',bbox[2],'  bbox[3]:',bbox[3],'bbox[2]*bbox[3]= :',bbox[2] * bbox[3] ,' bbox_max_lenth:',bbox_max_lenth)
#             flag = True
#
#             if(flag):
#                 filter_bboxes.append((bbox[0]*rs_scale, bbox[1]*rs_scale, \
#                     bbox[2]*rs_scale, bbox[3]*rs_scale))
#                 bboxCredibility.append(Credibility(bboxGrads[i]))
                
    FlagDelete=[0 for i in range(len(final_bboxes))]
    print('FlagDelete len :',len(FlagDelete))
    for i,bbox in enumerate(final_bboxes):

          if  (bbox[2] * bbox[3]<15) and bboxGrads[i]<120:
              FlagDelete[i] = 1
          if  (bbox[2] * bbox[3]>50) and bboxGrads[i]>200:
              FlagDelete[i] = 1
          if bbox[2] + bbox[3]>bbox_max_size or bbox[2] + bbox[3]< bbox_size_thresh/rs_scale or bbox[2]>bbox_max_lenth or bbox[3]>bbox_max_height  or (bbox[3]<= bbox_min_edge and bboxGrads[i]<60) or  (bbox[2]<=bbox_min_edge and bboxGrads[i]<60) :
                FlagDelete[i]=1
                #print( 'bboxGrad:',bboxGrads[i],'bbox[2]:',bbox[2],'  bbox[3]:',bbox[3],'bbox[2]*bbox[3]= :',bbox[2] * bbox[3] )
                if bbox[2] + bbox[3]>bbox_max_size or bbox[2]>bbox_max_lenth or bbox[3]>bbox_max_height:
                    for j,bbox2 in enumerate(final_bboxes):
                        if  bbox[0]<bbox2[0]<bbox[0]+bbox[2]+CloseDelete and bbox[1]<bbox2[1]<bbox[1]+bbox[3]+CloseDelete:
                            FlagDelete[j]=1



     #exclude the rivets from bbox
    for i,bbox in enumerate(final_bboxes):
        if FlagDelete[i]==0:
            for rivet in rivets_points:
                dist=np.sqrt((bbox[0]-rivet[0]/rs_scale)**2+(bbox[1]-rivet[1]/rs_scale)**2)
                if dist<=radius:
                    FlagDelete[i]=1


    for i,bbox in enumerate(final_bboxes):
        if FlagDelete[i]==0:
            print( 'bboxGrad:',bboxGrads[i],'bbox[2]:',bbox[2],'  bbox[3]:',bbox[3],'bbox[2]*bbox[3]= :',bbox[2] * bbox[3] )
            filter_bboxes.append((bbox[0]*rs_scale, bbox[1]*rs_scale, \
                     bbox[2]*rs_scale, bbox[3]*rs_scale))
            bboxCredibility.append(Credibility(bboxGrads[i]))

    #TEST_end('flaw_bboxes',tb)
    print('bboxCredibility:',bboxCredibility)
    return filter_bboxes,bboxCredibility


def show_bbox(img, bbox, color=(0,255,0), size=1):
    '''
    draw bounding box on image img

    input:
    img: h*w*3, img to draw bounding boxes on
    bbox: boudning box to draw, list of bbox(x, y, b_w, b_h)
    color: bounding box color
    size: painter size

    output:
    img: h*w*3, with bbox drawn
    '''
    for bbox in bbox:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, size)

    return img
