"""
生成WSI某个level tile的拼接图并保存
Method：
     用`openslide.deepzoom.DeepZoomGenerator`生成的一系列tile图片来拼接全图并保存
     TODO 删除 `dz_list`, `dz_path`  不需要实际的deepzoom文件， 参照`gym_histo.py` 类函数的处理方式
"""
import os
from PIL import Image
import numpy
import matplotlib.pyplot as plt
import openslide
from openslide.deepzoom import DeepZoomGenerator

# 1.Init Args
dz_level = 13
plt_size = 10
img_path = '/home/zhibo/data/tcga/2.svs'
root = '/home/zhibo/data/tcga/tiles/2/slide_files/'
dz_path = root + str(dz_level) +'/' 
result_path = './'
print("image path: %s\
        \nimage save path:%s \
        \ntile path : %s\
        \ndeepzoom level: %s\
        \nplot size: %s \
        " %(img_path,result_path, root, dz_level, plt_size))

# 2.Init slide, deepzoom ,dz_list<str list> ,tile<PIL list> , plot row & col
slide = openslide.OpenSlide(img_path)
dz = DeepZoomGenerator(osr=slide, tile_size=254, overlap=1, limit_bounds=False)

def sort_tile(dz_path):
    """
    Create dz_list with correct order
    Input: 
        path to tile directory
    return list with correct order, idx1,idx2
        ([0_0.jpeg, ...0_16.jpeg, 1_0.jpeg, ....22_16.jpeg],
            22, 16)
    

    """
    dz_list = sorted(os.listdir(dz_path))
    dz_list.sort(key = len) #多位数sort 解决 0 10 11 1 2的排序错误
    _file, _extension = os.path.splitext(dz_list[-1])
    idx1, idx2 = _file.split('_')
    idx1, idx2 = int(idx1), int(idx2)
    #print(idx1, idx2)

    _dz_list = []
    for i in range(idx1+1):
        for j in range(idx2 + 1):
            _dz_list.append(str(i) + "_" + str(j) + _extension)
    return _dz_list, idx1+1 , idx2+1 

dz_list, plt_col, plt_row = sort_tile(dz_path)    
print("level %s size is (raw,col): (%s, %s)" % (dz_level,plt_row, plt_col))
# print(dz_list[-1])


# # 3.Get row col for subplot
# # 5_4.jpeg  -->  row = 5, col = 6
# tmp = os.path.splitext(dz_list[-1])[0]
# plt_col ,plt_row  = tmp.split('_')
# plt_row, plt_col = int(plt_row)+1, int(plt_col)+1
# print("level %s size is (raw,col): (%s, %s)" % (dz_level,plt_row, plt_col))

tile = []
for i in dz_list:
    tile.append(Image.open(dz_path + i))

# # Tile Rotation
# #用Image.rotate时候会子图title会失效 需要注释掉
# for i in dz_list:
#     img_raw = Image.open(dz_path + i)
#     img_rota = img_raw.transpose(Image.ROTATE_270)
#     tile.append(img_rota)



# 4.Plot

#fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
fig, axs = plt.subplots(nrows=plt_row, ncols=plt_col, figsize=(plt_size, plt_size))
fig.suptitle('deepzoom level '+ str(dz_level), fontsize=20)
fig.tight_layout()#调整整体空白
axs = axs.T.flatten() # 让plt按照列来作图
#axs = axs.flatten()
i = 0
for t, ax in zip(tile, axs):
#for t, ax in zip(tile,axs.T):  
    subplot_title = os.path.splitext(os.path.basename(tile[i].filename))[0]
    ax.set_title(subplot_title)
    ax.set_xticklabels([])# 取消x坐标
    ax.set_yticklabels([])# 取消y坐标
    ax.imshow(t)
    i = i+1

plt.subplots_adjust(wspace=0, hspace=0)# 调整子图间距 取消间隔
#plt.show()
# fig.savefig(result_path + 'deepzoom level '+ str(dz_level)+ ".png")
fig.savefig("%sdeepzoom_level_%s.png" %(result_path,str(dz_level)))
print("image save at %sdeepzoom_level_%s.png" %(result_path,str(dz_level)))
slide.close()
del dz, slide, tile