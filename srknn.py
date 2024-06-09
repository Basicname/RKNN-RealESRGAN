from PIL import Image
import numpy as np
from rknnlite.api import RKNNLite
import time
import sys

rknn = RKNNLite() 

block_size = (96, 96)
scale = 4
model_path='realesrgan_96.rknn'

def divide_image_into_blocks_pillow(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        width, height = img.size
        num_blocks_y = (height + block_size[1] - 1) // block_size[1]
        num_blocks_x = (width + block_size[0] - 1) // block_size[0] 
        blocks = [[[] for _ in range(num_blocks_y)] for _ in range(num_blocks_x)]
        for y in range(0, num_blocks_y * block_size[1], block_size[1]):
            for x in range(0, num_blocks_x * block_size[0], block_size[0]):
                block = img.crop((x, y, x + block_size[0], y + block_size[1]))
                block_array = np.array(block)
                blocks[int(x / block_size[0])][int(y / block_size[1])]=block_array
        
        return blocks, num_blocks_x, num_blocks_y

def preprocess_image(img):
    img_array = np.array(img).astype(np.float32) 
    img_array /= 255.0 
    img_array = img_array.transpose((2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def postprocess_image(rknn_output):
    img_array = rknn_output[0][0].transpose((1, 2, 0))
    img_array = img_array.clip(0, 1) 
    img_array *= 255.0
    img = Image.fromarray(np.uint8(img_array))
    return img

def super_resolve_image(img):
    img_array = preprocess_image(img)
    img_array = np.transpose(img_array,(0,2,3,1))
    rknn_output = rknn.inference(inputs=[img_array])
    super_resolved_img = postprocess_image(rknn_output)
    return super_resolved_img

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python srknn.py image_path save_path')
        sys.exit(0)
    image_path = sys.argv[1]
    save_path = sys.argv[2]
    print(sys.argv[0])
    start_time = time.time()
    print('RKNN initializing...')
    rknn.load_rknn(model_path)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    print('Loading image...')
    image_blocks, block_x, block_y = divide_image_into_blocks_pillow(image_path)
    total_blocks = block_x * block_y
    now_block = 1
    print('Upscaling...')
    sr_image = Image.new('RGB', (block_x * block_size[0] * scale, block_y * block_size[1] * scale))
    for x in range(0, block_x):
        for y in range(0, block_y):
            print(f'Upscaling tile {now_block} / {total_blocks}')
            sr_block = super_resolve_image(image_blocks[x][y])
            sr_image.paste(sr_block,(x * block_size[0] * scale, y * block_size[1] * scale))
            now_block += 1
    sr_image.save(save_path)
    end_time = time.time()
    print(f'Save to {save_path}')
    print(f'Elapsed time:{end_time - start_time}s')
    
