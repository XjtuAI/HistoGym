#!/usr/bin/env python3
import sys
sys.path.append('/opt/ASAP/bin') # set pythonpath for ASAP
import multiresolutionimageinterface as mir
import matplotlib.pyplot as plt
import os.path as osp
import os
import openslide
import matplotlib.pyplot as plt
from pathlib import Path
import glob
#please make sure the same number of files in the folder of tumor file and folder of annotation files
#please change the slide_path, anno_path, mask_path accordingly, and leave everything else untouched. 

# slide_path = '/home/wli/Downloads/CAMELYON16/training/tumor'
# anno_path = '/home/wli/Downloads/CAMELYON16/training/Lesion_annotations'
# mask_path = '/home/wli/Downloads/CAMELYON16/masking2'
slide_path = '/home/zhibo/data/CAMELYON16/training/tumor'
anno_path = '/home/zhibo/data/CAMELYON16/training/annotations'
mask_path = '/home/zhibo/data/CAMELYON16/training/mask'
tumor_paths = glob.glob(osp.join(slide_path, '*.tif'))
tumor_paths.sort()
anno_tumor_paths = glob.glob(osp.join(anno_path, '*.xml'))
anno_tumor_paths.sort()
if not osp.exists(mask_path):
    os.makedirs(mask_path)

#image_pair = zip(tumor_paths, anno_tumor_paths)  
#image_pair = list(image_mask_pair)
print(tumor_paths)
reader = mir.MultiResolutionImageReader()
i=0
while i < len(tumor_paths):
    mr_image = reader.open(tumor_paths[i])
    annotation_list=mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(anno_tumor_paths[i])
    xml_repository.load()
    annotation_mask=mir.AnnotationToMask()
    camelyon17_type_mask = False
    label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 255, '_1': 255, '_2': 0}
    conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
    output_path= osp.join(mask_path, osp.basename(tumor_paths[i]).replace('.tif', '_mask.tif'))
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)
    i=i+1
