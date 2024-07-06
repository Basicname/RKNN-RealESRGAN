import threading
import time
import numpy as np
from PIL import Image
from rknnlite.api import RKNNLite

pool = []
total_count = 0
sr_image = None

lock = threading.Lock()

def preprocess_image(img):
    img_array = np.array(img).astype(np.float32) 
    img_array /= 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def postprocess_image(rknn_output):
    img_array = rknn_output[0][0].transpose((1, 2, 0))
    img_array = img_array.clip(0, 1) 
    img_array *= 255.0
    img = Image.fromarray(np.uint8(img_array))
    return img
    
def init(num_thread : int, w, h, model_path):
    global sr_image
    rknn_model = model_path
    sr_image = Image.new('RGB', (w, h))
    for i in range(0, num_thread):
        thread = threading.Thread(target = super_resolve_image, args=(model_path,))
        thread.start()
        time.sleep(0.2)
    
def put(lr_img, x, y):
    pool.append({'img' : lr_img, 'x' : x, 'y' : y})
    
def get():
    global sr_image
    return sr_image

def get_num():
    return total_count

def super_resolve_image(model_path):
    global pool
    global total_count
    global sr_image
    rknn = RKNNLite()
    rknn.load_rknn(model_path)
    rknn.init_runtime()
    while True:
        if len(pool) >= 1:
            lock.acquire()
            img = pool[0]['img']
            x = pool[0]['x']
            y = pool[0]['y']
            pool.pop(0)
            lock.release()
            img_array = preprocess_image(img)
            rknn_output = rknn.inference(inputs=[img_array])
            sr_block = postprocess_image(rknn_output)
            lock.acquire()
            sr_image.paste(sr_block,(x, y))
            total_count += 1
            lock.release()
        else:
            break;
