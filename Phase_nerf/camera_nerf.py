import cv2
import numpy as np


def get_gaussian_kernel(x,y,u1,u2,p1,p2,q):
    a=(x-u1)*(x-u1)/p1/p1
    b=2*q*(x-u1)*(x-u2)/p1/p2
    c=(y-u2)*(y-u2)/p2/p2


    return np.exp( -1/(2*(1-q*q))*( a-b+c  )  )

img=np.zeros((1000,1000)).astype('float')
u1=10
u2=10
p1=200
p2=200
q=0.2

for i in range(0,1000):
    for j in range(0,1000):
        x=i-500
        y=j-500
        img[i,j]=get_gaussian_kernel(x,y,u1,u2,p1,p2,q)

mean=0
sigma=20
gauss_noise =np.random.normal(mean,sigma,(1000,1000))

img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
img=img+gauss_noise
cv2.imwrite('gau3_gray.png',img)
img=cv2.applyColorMap(img.astype('uint8'),cv2.COLORMAP_JET)
cv2.imshow('test3',img.astype('uint8'))
cv2.imwrite('gau3.png',img)
cv2.waitKey(10)

img=np.zeros((1000,1000)).astype('float')
u1=10
u2=10
p1=200
p2=200
q=0.2

for i in range(0,1000):
    for j in range(0,1000):
        x=i-500
        y=j-500
        img[i,j]=get_gaussian_kernel(x,y,u1,u2,p1,p2,q)

img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
cv2.imwrite('gau1_gray.png',img)
img=cv2.applyColorMap(img.astype('uint8'),cv2.COLORMAP_JET)
cv2.imshow('test1',img.astype('uint8'))
cv2.imwrite('gau1.png',img)
cv2.waitKey(10)

img=np.zeros((1000,1000)).astype('float')
u1=10
u2=10
p1=200
p2=200
q=0

for i in range(0,1000):
    for j in range(0,1000):
        x=i-500
        y=j-500
        img[i,j]=get_gaussian_kernel(x,y,u1,u2,p1,p2,q)

img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
cv2.imwrite('gau2_gray.png',img)
img=cv2.applyColorMap(img.astype('uint8'),cv2.COLORMAP_JET)
cv2.imshow('test2',img.astype('uint8'))
cv2.imwrite('gau2.png',img)
cv2.waitKey(0)



