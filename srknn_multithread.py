from PIL import Image
import numpy as np
import time
import sys
from tqdm import tqdm
import pool

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
        
        return width, height, blocks, num_blocks_x, num_blocks_y

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python srknn.py image_path save_path num_thread')
        sys.exit(0)
    image_path = sys.argv[1]
    save_path = sys.argv[2]
    num_thread = int(sys.argv[3])
    start_time = time.time()
    print('Loading image...')
    width, height, image_blocks, block_x, block_y = divide_image_into_blocks_pillow(image_path)
    total_blocks = block_x * block_y
    now_block = 0
    for x in range(0, block_x):
        for y in range(0, block_y):
            sr_block = pool.put(image_blocks[x][y], x * block_size[0] * scale,  y * block_size[1] * scale)
    print('RKNN initializing...')
    pool.init(num_thread, width * scale, height * scale, model_path)
    pbar = tqdm(total = total_blocks)
    while now_block < total_blocks:
        now_block = pool.get_num()
        pbar.n = now_block
        pbar.refresh()
        time.sleep(0.5)
    sr_image = pool.get()
    sr_array = np.array(sr_image)
    sr_image = Image.fromarray(sr_array[0:height * scale, 0:width * scale])
    sr_image.save(save_path)
    end_time = time.time()
    print(f'Save to {save_path}')
    print(f'Elapsed time:{end_time - start_time}s')
    
