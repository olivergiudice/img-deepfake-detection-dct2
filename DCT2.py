from PIL import Image
import numpy as np
import scipy.stats as st
import statistics 
import math
import scipy
from scipy import fftpack
import pickle
import os

#   CONSTANTS
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model/clf_boost_noise_300_03_10_80.p')

'''
0: authentic
1-2-3-4-5: deepfake
1: AttGAN --> simple complexity - low resolution GAN
2: GDWCT --> medium complexity - low resolution GAN
3: STARGAN --> high complexity - high resolution GAN
4: StyleGAN --> high complexity - high resolution GAN
5: StyleGAN2 --> high complexity - high resolution GAN with reenactment processing
'''
LABEL = [
    'Authentic',
    'AttGAN',
    'GDWCT', 
    'STARGAN',
    'StyleGAN',
    'StyleGAN2'
]

# -----------
#   MODEL
# -----------
def load_model(path):
    print("Model loading...")
    with open(path, 'rb') as fp: 
        clf = pickle.load(fp)
    print("Model loaded.")
    return clf



def dct2(a):
    return fftpack.dct(fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

# Returns sequence of indices in JPEG-like zigzag scan
def GetId(q):
    mask = [0,1,5,6,14,15,27,28, 2,4,7,13,16,26,29,42, 3,8,12,17,25,30,41,43, 9,11,18,24,31,40,44,53, 10,19,23,32,39,45,52,54, 20,22,33,38,46,51,55,60, 21,34,37,47,50,56,59,61, 35,36,48,49,57,58,62,63]
    return mask.index(q)
  

# JPEG-like image blocking
def image_to_blocks(image):
    blocks=[]
    dctblocks = []
    for i in range(0,len(image),8):
        for j in range(0,len(image[0]),8):
            blocks.append((np.array(image[i:(i+8),j:(j+8)])))
            dctblocks.append(dct2(np.array(image[i:(i+8),j:(j+8)])).reshape(-1))
    return blocks, np.array(dctblocks)



def execute(img):
    """Execution function."""
    try:
        clf = load_model(MODEL_PATH)
        # the image is readed with Image.open
        img = img.convert('L')   
        width, height = img.size
        while True:
            if width % 8 != 0:
                width -= 1
            if height % 8 != 0:
                height -= 1
            if width % 8 == 0 and height % 8 == 0:
                break

        im = np.asarray(img)
        im = im[:height, :width]
        
        blocks, dctblocks = image_to_blocks(im)
        tmpLambda = []
        for i in range(64):
            tmpLambda.append(statistics.stdev(dctblocks[:, GetId(i)])/math.sqrt(2))
        
        # adjust tmpLambda for the prediction
        tmpLambda = np.asarray(tmpLambda[1:]).reshape(1, -1)
        try:
            result = clf.predict_proba(tmpLambda)
        except Exception as e:
            print(f'Try to reload the model - {str(e)}')
            clf = load_model(MODEL_PATH)
            result = clf.predict_proba(tmpLambda)
        
        results = {}
        for idx, res in enumerate(result[0]):
            results[LABEL[idx]] = res

        return results

    except Exception as e:
        print(f'Exception - {str(e)}')
        raise