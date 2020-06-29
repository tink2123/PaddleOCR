# -*- coding: UTF-8 -*-
from functools import reduce
import numpy as np
import cv2
import math
import random
import os 
 
# http://planning.cs.uiuc.edu/node102.html
def get_rotate_matrix(x, y, z):
    """
    按照 zyx 的顺序旋转，输入角度单位为 degrees, 均为顺时针旋转
    :param x: X-axis
    :param y: Y-axis
    :param z: Z-axis
    :return:
    """
    x = math.radians(x)
    y = math.radians(y)
    z = math.radians(z)
 
    c, s = math.cos(y), math.sin(y)
    M_y = np.matrix([[c, 0., s, 0.],
                     [0., 1., 0., 0.],
                     [-s, 0., c, 0.],
                     [0., 0., 0., 1.]])
 
    c, s = math.cos(x), math.sin(x)
    M_x = np.matrix([[1., 0., 0., 0.],
                     [0., c, -s, 0.],
                     [0., s, c, 0.],
                     [0., 0., 0., 1.]])
 
    c, s = math.cos(z), math.sin(z)
    M_z = np.matrix([[c, -s, 0., 0.],
                     [s, c, 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
 
    return M_x * M_y * M_z
 
 
def cliped_rand_norm(mu=0, sigma3=1):
    """
    :param mu: 均值
    :param sigma3: 3 倍标准差， 99% 的数据落在 (mu-3*sigma, mu+3*sigma)
    :return:
    """
    # 标准差
    sigma = sigma3 / 3
    dst = sigma * np.random.randn() + mu
    dst = np.clip(dst, 0 - sigma3, sigma3)
    return dst
 
 
def warpPerspective(src, M33, sl, gpu):
    if gpu:
        from libs.gpu.GpuWrapper import cudaWarpPerspectiveWrapper
        dst = cudaWarpPerspectiveWrapper(src.astype(np.uint8), M33, (sl, sl), cv2.INTER_CUBIC)
    else:
        print M33,sl,sl
        dst = cv2.warpPerspective(src, M33, (sl, sl), flags=cv2.INTER_CUBIC)
    return dst
 
 
# https://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
# https://nbviewer.jupyter.org/github/manisoftwartist/perspectiveproj/blob/master/perspective.ipynb
# http://planning.cs.uiuc.edu/node102.html
class PerspectiveTransform(object):
    def __init__(self, x, y, z, scale, fovy):
        self.x = x
        self.y = y
        self.z = z
        self.scale = scale
        self.fovy = fovy
 
    def transform_image(self, src, gpu=False):
        if len(src.shape) > 2:
            H, W, C = src.shape
        else:
            H, W = src.shape
 
        M33, sl, _, ptsOut = self.get_warp_matrix(W, H, self.x, self.y, self.z, self.scale, self.fovy)
        sl = int(sl)
 
        dst = warpPerspective(src, M33, sl, gpu)
 
        return dst, M33, ptsOut
 
    def transform_pnts(self, pnts, M33):
        """
        :param pnts: 2D pnts, left-top, right-top, right-bottom, left-bottom
        :param M33: output from transform_image()
        :return: 2D pnts apply perspective transform
        """
        pnts = np.asarray(pnts, dtype=np.float32)
        pnts = np.array([pnts])
        dst_pnts = cv2.perspectiveTransform(pnts, M33)[0]
 
        return dst_pnts
 
    def get_warped_pnts(self, ptsIn, ptsOut, W, H, sidelength):
        ptsIn2D = ptsIn[0, :]
        ptsOut2D = ptsOut[0, :]
        ptsOut2Dlist = []
        ptsIn2Dlist = []
 
        for i in range(0, 4):
            ptsOut2Dlist.append([ptsOut2D[i, 0], ptsOut2D[i, 1]])
            ptsIn2Dlist.append([ptsIn2D[i, 0], ptsIn2D[i, 1]])
 
        pin = np.array(ptsIn2Dlist) + [W / 2., H / 2.]
        pout = (np.array(ptsOut2Dlist) + [1., 1.]) * (0.5 * sidelength)
        pin = pin.astype(np.float32)
        pout = pout.astype(np.float32)
 
        return pin, pout
 
    def get_warp_matrix(self, W, H, x, y, z, scale, fV):
        fVhalf = np.deg2rad(fV / 2.)
        d = np.sqrt(W * W + H * H)
        sideLength = scale * d / np.cos(fVhalf)
        h = d / (2.0 * np.sin(fVhalf))
        n = h - (d / 2.0)
        f = h + (d / 2.0)
 
        # Translation along Z-axis by -h
        T = np.eye(4, 4)
        T[2, 3] = -h
 
        # Rotation matrices around x,y,z
        R = get_rotate_matrix(x, y, z)
 
        # Projection Matrix
        P = np.eye(4, 4)
        P[0, 0] = 1.0 / np.tan(fVhalf)
        P[1, 1] = P[0, 0]
        P[2, 2] = -(f + n) / (f - n)
        P[2, 3] = -(2.0 * f * n) / (f - n)
        P[3, 2] = -1.0
 
        # pythonic matrix multiplication
        M44 = reduce(lambda x, y: np.matmul(x, y), [P, T, R])
 
        # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way.
        # In C++, this can be achieved by Mat ptsIn(1,4,CV_64FC3);
        ptsIn = np.array([[
            [-W / 2., H / 2., 0.],
            [W / 2., H / 2., 0.],
            [W / 2., -H / 2., 0.],
            [-W / 2., -H / 2., 0.]
        ]])
        ptsOut = cv2.perspectiveTransform(ptsIn, M44)
 
        ptsInPt2f, ptsOutPt2f = self.get_warped_pnts(ptsIn, ptsOut, W, H, sideLength)
 
        # check float32 otherwise OpenCV throws an error
        assert (ptsInPt2f.dtype == np.float32)
        assert (ptsOutPt2f.dtype == np.float32)
        M33 = cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f).astype(np.float32)
 
        return M33, sideLength, ptsInPt2f, ptsOutPt2f
 
def apply_perspective_transform(img, text_box_pnts, max_x, max_y, max_z, gpu=False):
    """
    Apply perspective transform on image
    :param img: origin numpy image
    :param text_box_pnts: four corner points of text
    :param x: max rotate angle around X-axis
    :param y: max rotate angle around Y-axis
    :param z: max rotate angle around Z-axis
    :return:
        dst_img:
        dst_img_pnts: points of whole word image after apply perspective transform
        dst_text_pnts: points of text after apply perspective transform
    """
 
    x = cliped_rand_norm(0, max_x)
    y = cliped_rand_norm(0, max_y)
    z = cliped_rand_norm(0, max_z)
 
    # print("x: %f, y: %f, z: %f" % (x, y, z))
 
    transformer = PerspectiveTransform(x, y, z, scale=1.0, fovy=50)
 
    dst_img, M33, dst_img_pnts = transformer.transform_image(img, gpu)
    #dst_text_pnts = transformer.transform_pnts(text_box_pnts, M33)
 
    return dst_img#, dst_img_pnts, dst_text_pnts


 
def rad(x):
    return x * np.pi / 180
 
def get_warpR(config):
    anglex,angley,anglez,fov,w,h,r = config.anglex,config.angley,config.anglez,config.fov,config.w,config.h,config.r
    # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)
 
    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)
 
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)
 
    r = rx.dot(ry).dot(rz)
 
    # 四对点的生成
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
 
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter
 
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
 
    list_dst = [dst1, dst2, dst3, dst4]
 
    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)
 
    dst = np.zeros((4, 2), np.float32)
 
    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
 
    warpR = cv2.getPerspectiveTransform(org, dst)
    #print(warpR)
    dst1, dst2, dst3, dst4 = dst
    r1 = int(min(dst1[1],dst2[1]))
    r2 = int(max(dst3[1],dst4[1]))
    c1 = int(min(dst1[0],dst3[0]))
    c2 = int(max(dst2[0],dst4[0]))
    #print( (0,0 ), (h,w ) )
    
    try:
        ratio = min(1.0*h/(r2-r1),1.0*w/(c2-c1))# todo , incase zero ERROR
    #warpR[2,2] = ratio
    #print( (r1,c1), (r2,c2),ratio)
        dx = -c1
        dy = -r1
        T1 = np.float32([[1.,0 , dx],
                     [0, 1., dy ],
                     [0 ,0, 1.0/ratio]])
        ret = T1.dot(warpR)
    
    except:
        ratio = 1.0
        T1 = np.float32([[1.,0 , 0],
                     [0, 1., 0 ],
                     [0 ,0, 1.]])
        ret = T1
    return ret,(-r1,-c1),ratio
 
 
    #control()
def get_warpAffine(config):
    #w,h,shearx,sheary,shrink = config.w,config.h,config.shearx,config.sheary,config.shrink
    anglez = config.anglez
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0 ],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0]], np.float32)
    '''
    org = np.array([[0, 0],
                    [w, h],
                    [0, h]], np.float32)
    
    dst_x = np.array([[0, 0],
                    [w, h],
                    [0.3*w , h]], np.float32)
    #Tx = cv2.getAffineTransform(org,dst_x)
    
    org = np.array([[0, 0],
                    [w, h],
                    [w, 0]], np.float32)
    dst_y = np.array([[0, 0],
                    [w, h],
                    [w , 0.3*h]], np.float32)
    #Ty = cv2.getAffineTransform(org,dst_y)
    '''
    return rz#Tx.dot(Ty)
class config:
    def __init__(self,):
        self.anglex = random.random()*30
        self.angley = random.random()*15
        self.anglez = random.random()*10  # 是旋转
        self.fov = 42
        self.r = 0
        self.shearx = random.random()*0.3
        self.sheary = random.random()*0.05
        self.borderMode = cv2.BORDER_REPLICATE #if random.random()>0.5 else cv2.BORDER_REFLECT
        self.affine = True
        self.perspective = True
        self.reverse = True
        self.noise = True
        self.blur = True
        self.color = True
        self.dou = True 
        #self.d_x = 0
        #self.d_y = 0
        self.shrink = 1# - random.random()*0.3
    
    def make_(self,w,h):
        self.w = w
        self.h = h
        

    def make(self,w,h):
        self.anglex = random.random()*30*flag()
        self.angley = random.random()*15*flag()
        self.anglez = -1*random.random()*10*flag()  # 是旋转
        self.fov = 42
        self.r = 0
        self.shearx = 0#random.random()*0.3*flag()
        self.sheary = 0#random.random()*0.05*flag()
        #self.shrink = 1 - random.random()*0.3
        self.borderMode = cv2.BORDER_REPLICATE #if random.random()>0.2 else cv2.BORDER_TRANSPARENT
        self.w = w
        self.h = h
        ra = random.random() 
        self.perspective = True #True if ra > 0.75  else False 
        self.affine = False#True #if ra <= 0.5 else False
        self.reverse = True #if random.random() >0.5 else False
        self.noise = True #if random.random() >0.5 else False
        self.dou = False 
        self.blur = True #if random.random() >0.5 else False
        self.color = True#True #if random.random() >0.5 else False
def flag():
    return 1 if random.random() >0.5000001 else -1     

def cvtColor(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random()*flag()
    hsv[:,:,2] = hsv[:,:,2]*(1+delta)
    #hsv[:,:,1] = hsv[:,:,1]*(1+delta)
    new_img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return new_img

def blur(img):
    h , w ,_ = img.shape
    if h>10 and w>10:
        return cv2.GaussianBlur(img,(5,5),1)
    else:
        return img
def doudong(img):
    w ,h ,_ = img.shape
    if h>10 and w>10:
        thres = min(w ,h )
        s = int(random.random()*thres*0.01)
        src_img = img.copy()
        #img[:w-s,:h-s,:] = (img[:w-s,:h-s,:] + img[s:,s:,:])/2
        for i in range(s):
            img[i:,i:,:] = src_img[:w-i,:h-i,:] 
        return img
    else:
        return img
    
def add_gasuss_noise(image, mean=0, var=0.1):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    #image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + 0.5*noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out
    
#myConfig = config() 
def warp(img,config):
    h, w, _ = img.shape
    config.make(w,h)
    new_img = img
    r1,c1 = 0,0
    ratio = 1.0
    if config.perspective: 
        warpR,(r1,c1),ratio = get_warpR(config)
        #config.d_x = c1
        #config.d_y = r1
        #config.shrink = ratio
        new_img = cv2.warpPerspective(new_img, warpR, (w, h),borderMode=config.borderMode)
    #print(img.dtype,warpT.dtype,type(w),type(h))
    #img.astype('float32')
    if config.affine:
        warpT = get_warpAffine(config )
        new_img = cv2.warpAffine(new_img, warpT,(w, h),borderMode=config.borderMode)
    if config.blur:
        new_img = blur(new_img)
    if config.color:
        new_img = cvtColor(new_img)  
    if config.dou:
        new_img = doudong(new_img)
    if config.noise:
        new_img = add_gasuss_noise(new_img)
    if config.reverse:
        new_img = 255 - new_img
    

    
    
    return new_img 
        
if __name__ =="__main__":
    src_dir ='./'
    trg_dir = 'aug_vgg_synth90K/'
    if not os.path.exists(trg_dir):
        os.mkdir(trg_dir)
    file = input() 
    filelist = open(file).readlines() 
    file_trg = 'aug_'+file
    myConfig = config()
    fp = open(file_trg,'w')
    for i in range(2):
        for line  in filelist:
            #file1 = 'word_404.png'
            item = line.split()
            file1 = item[2]
            img = cv2.imread(src_dir+file1)
            #h , w , _ = img.shape
            #print(img.shape) 
            #config.w , config.h = w, h
            #new_img = apply_perspective_transform(img,None,1,1,1)
            #warpR,(r1,r2),(c1,c2) = get_warpR(config)
            #warpT = get_warpAffine(config)
            #new_img = cv2.warpPerspective(img, warpR, (w, h),borderMode=cv2.BORDER_REPLICATE)
            #print(img.dtype,warpT.dtype,type(w),type(h))
            #img.astype('float32')
            #new_img = cv2.warpAffine(img, warpT,(w, h),borderMode=cv2.BORDER_REPLICATE)
            #print(r1,r2)
            #print(c1,c2)
            #new_img = new_img[r1:r2,c1:c2,:3]
            new_img = warp(img,myConfig)
            file2 = trg_dir + file1[:-4]+'_aug%d_'%i+'.jpg'
            if not os.path.exists(os.path.dirname(file2)):
                os.makedirs( os.path.dirname(file2))
            #print(new_img.shape)
            cv2.imwrite(file2,new_img)
            item[2] = file2
            prints = ' '.join(item)+'\n'
            fp.write(prints)
    fp.close()

    #from  PIL import Image as IM
    #from PIL import ImageFilter 
    #img = IM.open(dir+file1)
    #max_rotate = 30
    #img = img.filter(ImageFilter.BLUR)
    #img = img.filter(ImageFilter.BLUR)
    #rotate = int(random.random()*20)
    #if False:#random.random()>0.5:
    #    img = img.rotate(rotate)
    #else:
    #    img = img.rotate(-1*rotate)
    #img.save(dir+file2)

