#!/usr/bin/env python
# coding: utf-8

# In[582]:

from yolo import yolo_init,run_yolo
import cv2
import numpy as np
import math
from numpy import ones,vstack
from numpy.linalg import lstsq
import sys


# In[583]:

# K
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def to_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def warp_perspective(img,mat):
    return cv2.warpPerspective(img ,mat ,(1280,720))
    
def PerspectiveTransform(src,dst):
    mat = cv2.getPerspectiveTransform(src,dst)
    mat_inv = cv2.getPerspectiveTransform(dst,src)
    return mat,mat_inv

def isolate_color_mask(img, low_thresh, high_thresh):
    return cv2.inRange(img, low_thresh, high_thresh)


def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def concat_frames(frames):
    
    scale_percent = 0.25
    height = int(frames[0].shape[0] * 0.5)
    width = int(frames[0].shape[1] * scale_percent)
    dim = (width, height)
    img = cv2.resize(frames[0], dim, interpolation = cv2.INTER_AREA)
    img2= cv2.resize(cv2.cvtColor(frames[4], cv2.COLOR_GRAY2BGR), dim, interpolation = cv2.INTER_AREA)
        
    for i in range(1,len(frames)):
        try:
            frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
        except:
            pass
        
        if(i==4):
            continue
        
        try:
            resized = cv2.resize(frames[i], dim, interpolation = cv2.INTER_AREA)
            if i< (len(frames)/2):
                img = cv2.hconcat([img,resized])
            else:
                img2 = cv2.hconcat([img2,resized])
        except:
            pass
    
    t = cv2.vconcat([img,img2])

    return t

# H
def get_right(image):
    cropped_img=np.zeros_like(image)
    cropped_img[:,image.shape[1]//2:image.shape[1]] = image[:,image.shape[1]//2:image.shape[1]]
    return cropped_img

def get_third_right(image):
    cropped_img=np.zeros_like(image)
    cropped_img[:,image.shape[1]*2//3:image.shape[1]] = image[:,image.shape[1]*2//3:image.shape[1]]
    return cropped_img

def get_left(image):
    cropped_img=np.zeros_like(image)
    cropped_img[:,0:image.shape[1]//2] = image[:,0:image.shape[1]//2]
    return cropped_img

def get_third_left(image):
    cropped_img=np.zeros_like(image)
    cropped_img[:,0:image.shape[1]//3] = image[:,0:image.shape[1]//3]
    return cropped_img

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist
        
    


net,layers_names=yolo_init()

# CHOOSING THE POINTS TO USE IN THE PRESPECTIVE TRANSFORM
input_top_left = [500,480]
input_top_right = [780,480]
input_bottom_right = [1200,700]
input_bottom_left = [170,700]
srcPts = np.float32([input_bottom_left,input_top_left,input_top_right,input_bottom_right])

# STRETCH THE PREVIOUS POINTS ON THE WHOLE FRAME 
dstPts = np.float32([[0,720],[0,0],[1280,0],[1280,720]])

# GET THE ARGUMENTS 
path = sys.argv[1]
mode = sys.argv[2]
out_path = sys.argv[3]


# In[585]:


# INTIALIZING THE MEMORY (HISTORY OF LANES )

# INITIALIZING THE POINTS FOR THE TWO LINES FORMING THE LANES
old_j=0,0
old_i=0,0
old_jr=0,0
old_ir=0,0

#INITALIZNG THE CONTOURS
old_centr=0,0
old_contour=[]
old_contour_w=[]

# INTIALIZNG AN ARRAY TO HOLD THE VALUES OF THE LINE POINTS TO GET THE MEAN LATER ON 
arr=list()
for i in range(4):
    arr.append([])
    
# INITALIZE AN ARRAY TO HOLD THE POINTS OF THE EDGES OF THE CONOTUR
old_pts=[]


# In[586]:

print(path)
# CAPTUTRE THE VIDEO
cap = cv2.VideoCapture(path)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(out_path + "output" +'.mp4', fourcc, 25, (1280,720))

# FRAME COUNTER
counter = 0

while cap.isOpened():
  ret,img = cap.read()
  if not ret:
      break

  # GET THE CENTRE OF THE IMAGE (THIS IS CONSIDERED THE CENTRE OF MY CAR)
  me=(img.shape[1]//2,img.shape[0]//2)

  # CREATE THE MATRICES TO DO THE TRANSFORM GIVEN THE SELECTED POINTS
  mat,mat_inv = PerspectiveTransform(srcPts,dstPts)

  # P IS A POINT AT THE CENTRE OF THE IMAGE REPRESENTING MY CAR BUT AT A HEIGHT THAT CAN BE SEEN
  # IN THE PRESPECTIVE VIEW

  p = me[0],600 # MY ORIGINAL POINT

  # WE NEED TO GET THE LOCATION OF THE POINT AFTER TRANSFORMATION
  px = (mat[0][0]*p[0] + mat[0][1]*p[1] + mat[0][2]) / ((mat[2][0]*p[0] + mat[2][1]*p[1] + mat[2][2]))
  py = (mat[1][0]*p[0] + mat[1][1]*p[1] + mat[1][2]) / ((mat[2][0]*p[0] + mat[2][1]*p[1] + mat[2][2]))
  me = (int(px), int(py)) # AFTER TRANSFORMATION

  # GET THE BIRD_VIEW AND A COPY OF IT FOR VISUALIZATION
  bird_view = warp_perspective(img,mat)
  bird_cpy = bird_view.copy()

  # ENLIGHTEN THE IMAGE QUITE A BIT FOR BETTER DETECTION !
  lightened = adjust_gamma(bird_view,2)

  # CREATE A YELLOW MASK AND A WHITE MASK THEN GET A MASK CONTAINING BOTH
  white_mask = isolate_color_mask(to_hls(bird_view), np.array([0, 200, 0], dtype=np.uint8), np.array([200, 255, 255], dtype=np.uint8))
  yellow_mask = isolate_color_mask(to_hls(lightened), np.array([10, 0, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
  mask = cv2.bitwise_or(white_mask, yellow_mask)

  # APPLY CANNY ON THE MASK IMAGE TO GET THE dilated_edges
  edges = cv2.Canny(mask,70,140)

  # DILATE THE IMAGE TO CONNECT THE LINES AND MAKE IT EASIER TO GET THE CONTOUR
  dilated_edges = cv2.dilate(edges,np.ones((7,7)))


  ######
  ## DEALING WITH THE LEFT LANE
  #####


  # GET THE CONTOURS FROM THE  LEFT SIDE OF THE IMAGE  (YELLOW LANE )
  contours,_= cv2.findContours(get_third_left(dilated_edges), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  # GET THE CONTOUR WITH THE MAXIMUM AREA
  # IF THERE'S NO CONTOURS READ DUE TO NOISE OR THE BIGGEST CONTOUR
  # IS QUITE SMALL THEN USE THE OLD CONTOUR
  try:
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c)>10000 :
      old_contour=c
    else :
      raise Exception()
  except:
    c=old_contour


  # GET THE EXTREME POINTS OF THE BIGGEST CONTOUR (THAT REPRESENTS A POLYGON SURROUNDING THE LANE)
  left = tuple(c[c[:, :, 0].argmin()][0])
  right = tuple(c[c[:, :, 0].argmax()][0])
  top = tuple(c[c[:, :, 1].argmin()][0])
  bottom = left[0] + right[0]-top[0] , left[1]
  points = np.array([bottom,right,top,left])

  # STORE THESE POINTS FOR FURTHER CALCULATIONS
  for i,p in enumerate(points) :
    arr[i].append(p)

  # CALCULATE THE MEAN OF THE POINTS THAT VARIES QUITE A BIT
  medx = np.mean([ar[0] for ar in arr[0]])
  medy= np.mean([ar[1] for ar in arr[0]])
  medk = np.mean([ar[1] for ar in arr[1]])


  # AT THE START SAVE THESE POINTS AS HISTORY
  if counter==0 :
    old_pts = points

  # A

  # AFTER SKIPPING SOME FRAMES TO ENSURE WE REACHED STABILITY
  # USE THE CALCULATED MEAN TO ENSURE THAT THE MEASURED POINTS
  # AREN'T FAR OFF THE MEANS

   # IF THE POINT IS VALID THEN UPDATE THE HISTORY
  if counter > 10 :

    if right[1] > 1.2*medk :
        points[1]= old_pts[1]
        right= old_pts[1]
    else:
        old_pts[1]=right

    if left[0] > 1.2*medx or left[1] > medy :
      points[3]= old_pts[3]
      left= old_pts[3]
    else:
      old_pts[3]=left


    # GET TWO POINTS TO REPRESENT THE LEFT LANE
  j =  (right[0]+top[0])//2,right[1]
  i =  (bottom[0]+left[0])//2, bottom[1]


    # IF THESE POINTS ARE FURTHER FROM THE POINTS OF THE PREVIOUS FRAME THEN DISCARD THEM !
  if counter>20 :
    if (calculateDistance(i[0],i[1],old_i[0],old_i[1]) > 200 and old_i!=(0,0)) or (calculateDistance(j[0],j[1],old_j[0],old_j[1]) > 200 and old_j!=(0,0)):
      j = old_j
      i = old_i
    else:
      old_i=i
      old_j=j

  # SOLVE THE EQUATION TO GET THE Y AND C
  points = [i,j]
  x_coords, y_coords = zip(*points)
  A = vstack([x_coords,ones(len(x_coords))]).T
  m, c_l = lstsq(A, y_coords)[0]

  # GET TWO LINES LYING ON THE STRAIGHT LINE AND DRAW IT !
  j_l = 1280, int(1280*m + c_l)
  i_l = 0, int(0*m + c_l)

  cv2.line(bird_cpy,i_l,j_l,(0,0,255),40)
  
  
  # THE LOWEST POINT OF THE LEFT LANE 
  p=i

  ######
  ## DEALING WITH THE RIGHT LANE
  #####

  contours,_= cv2.findContours(get_third_right(dilated_edges), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  # SINCE THAT THE RIGHT LANE IS CUT , SO WE NEED TO DRAW ALL THE CONTOURS , NOT JUST THE
  # BIGGEST ONE. HERE WE USE BOUNDING RECTANGLE SINCE IT IS SMALL IN AREA
  # SO IT CAN BE APPROXIMATED TO BE A RECTANGLE

  for c in contours :
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    cv2.rectangle(bird_cpy,(x,y),(x+w,y+h),(0,0,255),-1)

  # IF THERE'S NO CONTOURS READ DUE TO NOISE THEN USE THE OLD CONTOUR
  try:
    c = max(contours, key=cv2.contourArea)
    old_contour_w=c
  except:
    c=old_contour_w



  # GET THE POINT LYING ON THE BOUNDING RECTANGLE AND USE IT DRAW A STRAIGHT LINE THAT
  # CAN BE APPROXIMATED TO BE THE LANE FOR DISTANCE CALCULATION

  centr = x,y+h//2
  if counter>30 :
    if calculateDistance(centr[0],centr[1],old_centr[0],old_centr[1])>42 and old_centr!=(0,0) :
      centr=old_centr
    else:
      old_centr=centr

    # W
    
   # GET THE dilated_edges OF THE POLYGON THAT REPRESENTS THE AREA THAT
   # THE CAR CAN MOVE IN AND COLOR IT BLUE 
  RL_U = (centr[0],0)
  RL_D = (centr[0],720)
  pts=np.array([RL_U,RL_D,i_l,j_l])
  
  ## FOR BETTER VISUALIZATION , CREATE IMAGE TO HOLD THE POLYGON 
  gh = np.zeros_like(img)
  cv2.fillConvexPoly(gh, pts,color=(255, 0, 0))
  gh=warp_perspective(gh,mat_inv)
  

  # GET THE CENTRE OF THE LANE BY GETTING THE HALF POINT BETWEEN
  # THE LOWEST POINT ON THE LEFT LANE AND THE RIGHT LANE 
  centreOfLane = ((p[0]+centr[0])//2,p[1])

  # DIST IS THE DISTANCE BETWEEN THE CENTRE OF THE CAR AND THE CENTRE OF THE LANE IN PIXELS
  dist = np.abs(centreOfLane[0]-me[0])
  # DIST2 IS THE DISTANCE BETWEEN THE TWO LANES IN PIXELS 
  dist2=np.abs(p[0]-centr[0])
  # SCALE IS THE DISTANCE IN PIXELS / DISTANCE IN METERS
  scale = dist2/3

  # PUT THE TEXT ON THE IMAGE 
  text = "Distance : "  + str(round(dist*(1/scale),3)) + "m."
  cv2.putText(img=bird_cpy, text=text, org=(400, 400), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 255, 255),thickness=5)


# DRAW A POINT REPRESENTING THE CENTRE OF MY CAR
  cv2.circle(bird_cpy,me,5,(255,255,0),10)


# REGENERATE THE IMAGE TO THE NORMAL PRESPECTIVE 
  re_gen = warp_perspective(bird_cpy,mat_inv)

# COPY THE ORIGINAL IMAGE FOR VISUALIZATION
  imgg = img.copy()
  
  
  run_yolo(net,layers_names,imgg)
  
# ADD THE REGENERATED IMAGE WITH THE ORIGINAL IMAGE 
  k=cv2.threshold(grayscale(re_gen), 0, 255, cv2.THRESH_BINARY)
  masked = cv2.bitwise_and(img,img,mask=k[1])  
  m=cv2.bitwise_xor(masked,imgg)
  cv2.addWeighted(m, 1, re_gen, 1,0, imgg)
  cv2.addWeighted(gh, 1, imgg, 1,0, imgg)
  # A 

  counter+=1
  
  if mode == str(1):
    # PASS THE FULL PIPELINE TO CREATE A VIDEO CONTAINING THE STEPS 
    pipeline=[img,bird_view,yellow_mask,white_mask,edges,dilated_edges,bird_cpy,imgg]
    full_pipeline=concat_frames(pipeline)
    out.write(full_pipeline)
    cv2.imshow("Full pipeline",full_pipeline)
  else:
    cv2.imshow("Output_video",imgg)
    out.write(imgg)


    

  if cv2.waitKey(25) & 0xFF == ord('q'):
    break

cap.release()
out.release
cv2.destroyAllWindows()

# L 

