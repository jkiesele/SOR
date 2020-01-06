

import skimage
from skimage import color
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np

import random
#generate one shape at a time and then add the numpy arrays only where there are zeros

npixel=128


## debug


def makeRectangle(size, pos, image):
    print('pos', pos)
    posa = [pos[0]-size[0]/2., pos[1]-size[1]/2.]
    print('pos',posa)
    print('size',size[0],size[1])
    print(image[int(pos[1])][int(pos[0])])
    print(image[0][0])
    return patches.Rectangle(posa,size[0],size[1],linewidth=1,edgecolor='y',facecolor='none')

##


def getCenterCoords(t1):
    #yes, that is the format - no, there is a bug in here!!!
    coords1b = float(t1[0][1][0][1]+t1[0][1][0][0])/2.
    coords1a = float(t1[0][1][1][1]+t1[0][1][1][0])/2.
    return coords1a, coords1b

def getWidthAndHeight(t1):
    return  float(t1[0][1][1][1]-t1[0][1][1][0]), float(t1[0][1][0][1]-t1[0][1][0][0])

def labeltoonehot(desc):
    if desc[0][0] == 'rectangle':
        return np.array([1,0,0],dtype='float32')
    if desc[0][0] == 'triangle':
        return np.array([0,1,0],dtype='float32')
    if desc[0][0] == 'circle':
        return np.array([0,0,1],dtype='float32')
    
    raise Exception("labeltoonehot: "+desc[0][0])
   
def createPixelTruth(desc,image, ptruth):
    onehot = labeltoonehot(desc)
    coordsa,coordsb = getCenterCoords(desc)
    coords = np.array([coordsa,coordsb], dtype='float32')
    truth =  np.concatenate([coords,onehot])
    truth = np.expand_dims(np.expand_dims(truth, axis=0),axis=0)
    truth = np.tile(truth, [image.shape[0],image.shape[1],1])
    w,h = getWidthAndHeight(desc)
    wh = np.array([w,h], dtype='float32')
    wh = np.expand_dims(np.expand_dims(wh, axis=0),axis=0)
    wh = np.tile(wh, [image.shape[0],image.shape[1],1])
    truth =  np.concatenate([truth,wh],axis=-1)
    if ptruth is None:
        ptruth = np.zeros_like(truth)+255
    truthinfo = np.where(np.tile(np.expand_dims(image[:,:,0]<255, axis=2), [1,1,truth.shape[2]]), truth, ptruth)
    return truthinfo
    

def createMask(ptruth):
    justone = ptruth[:,:,0:1]
    return np.where(justone > 254, np.zeros_like(justone), np.zeros_like(justone)+1.)
    
def addNObjects(ptruth,nobj):
    a = np.array([[[nobj]]],dtype='float32')
    a = np.tile(a, [ptruth.shape[0],ptruth.shape[1],1])
    return np.concatenate([ptruth,a],axis=-1)
    
def checktuple_overlap(t1,t2):
    x1,y1 = getCenterCoords(t1)
    x2,y2 = getCenterCoords(t2)
    diff = (x1-x2)**2 + (y1-y2)**2
    if diff < 30:
        return True
    return False
    
    
def checkobj_overlap(dscs,dsc):
    for d in dscs:
        if checktuple_overlap(dsc,d):
            return True
    return False
    
#fig,ax = plt.subplots(1)
def generate_shape(npixel, seed=None):
    if seed is not None:
        return skimage.draw.random_shapes((npixel, npixel),  max_shapes=1, 
                                      min_size=npixel/5, max_size=npixel/3,
                                      intensity_range=((100, 254),))
    else:
        return skimage.draw.random_shapes((npixel, npixel),  max_shapes=1, 
                                      min_size=npixel/5, max_size=npixel/3,
                                      intensity_range=((100, 254),), random_seed=seed)
    

#adds a shape BEHIND the existing ones
def addshape(image , desclist, npixel, seed=None):
    if image is None:
        image, d = generate_shape(npixel)
        return image, image, d, True
    
    new_image, desc = generate_shape(npixel,seed)
    
    #if checkobj_overlap(desclist,desc):
    #    return image, image, desc, False
    
    shape = new_image.shape
    select = image < 255 #only where the image was not empty before
    
    newobjectadded = np.where(select,image,new_image)
    
    if np.all(newobjectadded == image): 
        image, image, desc, False
    
    return newobjectadded, new_image, desc, True


def create_images(nimages = 1000, npixel=64, seed=None):
    '''
    returns features, truth
    
    returns features as
    B x P x P x F
    with F = colours
    
    returns truth as 
    B x P x P x T
    with T = [mask, true_posx, true_posy, ID_0, ID_1, ID_2, true_width, true_height, n_objects]
    '''
    
    
    pixrange = np.arange(npixel, dtype='float32')
    pix_x = np.tile(np.expand_dims(pixrange,axis=0), [npixel,1])
    pix_x = np.expand_dims(pix_x,axis=2)
    pix_y = np.tile(np.expand_dims(pixrange,axis=1), [1,npixel])
    pix_y = np.expand_dims(pix_y,axis=2)
    pix_coords = np.concatenate([pix_y,pix_x],axis=-1)
    
    
    alltruth = []
    allimages = []
    for e in range(nimages):
        dsc=[]
        image=None
        nobjects = random.randint(1,7)
        addswitch=True
        indivimgs=[]
        indivdesc=[]
        totobjects=0
        ptruth=None
        for i in range(nobjects):
            new_image, obj, des, addswitch = addshape(image,indivdesc, npixel=npixel, seed=seed)
            if addswitch:
                totobjects+=1
                ptruth = createPixelTruth(des, obj, ptruth)
                image = new_image
                
        #print(image)
        if e < 100 and False:
            plt.imshow(image)
            #plt.show()
            plt.savefig("image"+str(e)+".png")
            #exit()
            #
        mask = createMask(ptruth)
        ptruth = np.concatenate([mask,ptruth],axis=-1)
        ptruth = addNObjects(ptruth,totobjects)
        
        #ax.imshow(image)
        image = np.concatenate([image,pix_coords],axis=-1)
        image = np.expand_dims(image,axis=0)
        ptruth = np.expand_dims(ptruth,axis=0)
        
        if seed is not None:
            seed+=1
        
        #for x in range(ptruth.shape[1]):
        #    for y in range(ptruth.shape[2]):
        #        print(ptruth[0][x][y])
        
        allimages.append(image)
        alltruth.append(ptruth)
    
    return np.concatenate(allimages, axis=0), np.concatenate(alltruth,axis=0)
