
#If you need more information about how ElementTree package handle XML file, please follow the link: 
#https://docs.python.org/3/library/xml.etree.elementtree.html

import matplotlib.pyplot as plt
import cv2
import numpy as np
import xml.etree.ElementTree as et
import pandas as pd
import math
import openslide

wsi_level = 6
img_path = '/home/zhibo/data/CAMELYON16/training/tumor/tumor_001.tif'
xml_path = '/home/zhibo/data/CAMELYON16/training/annotations/tumor_001.xml'

slide = openslide.OpenSlide(img_path)
Ximageorg, Yimageorg = slide.level_dimensions[0]
dims = slide.level_dimensions[wsi_level]

#!/usr/bin/env python3



def convert_xml_df (file):
    parseXML = et.parse(file)
    root = parseXML.getroot()
    dfcols = ['Name', 'Order', 'X', 'Y']
    df_xml = pd.DataFrame(columns=dfcols)
    for child in root.iter('Annotation'):
        for coordinate in child.iter('Coordinate'):
            Name = child.attrib.get('Name')
            Order = coordinate.attrib.get('Order')
            X_coord = float(coordinate.attrib.get('X'))
           # X_coord = X_coord - 30000
            X_coord = ((X_coord)*dims[0])/Ximageorg
            Y_coord = float(coordinate.attrib.get('Y'))
           # Y_coord = Y_coord - 155000
            Y_coord = ((Y_coord)*dims[1])/Yimageorg
            df_xml = df_xml.append(pd.Series([Name, Order, X_coord, Y_coord], index = dfcols), ignore_index=True)
            df_xml = pd.DataFrame(df_xml)
    return (df_xml)  

annotations = convert_xml_df(xml_path)

#x_values = list(annotations['X'].get_values())
#y_values = list(annotations['Y'].get_values())
#xy = list(zip(x_values,y_values))
def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

final_list = Remove(annotations['Name'])

coxy = [[] for x in range(len(final_list))]

i = 0
for n in final_list:
    newx = annotations[annotations['Name']== n]['X']
    newy = annotations[annotations['Name']== n]['Y']
    print(n)
    print(newx, newy)
    newxy = list(zip(newx, newy))
    coxy[i] = np.array(newxy, dtype=np.int32)
    i=i+1
    

#image = cv2.imread('/home/wli/Downloads/tumor_036.xml', -1)
#tile =mr_image.getUCharPatch(0, 0, dims[0], dims[1], wsi_level)
tile = slide.read_region((0,0), wsi_level, (dims[0], dims[1]))
tile = np.asarray(tile)
#canvas = np.zeros((Ximage, Yimage, 3), np.uint8) # fix the division
#coords = np.array([xy], dtype=np.int32)

#cv2.drawContours(canvas, [coords],-1, (0,255,0), -1)
#cv2.polylines(tile, [coords], isClosed=True, color=(0,0,255), thickness=5)

cv2.drawContours(tile, coxy, -1, (0, 255, 0), wsi_level)


cv2.imshow("tile", tile)
cv2.waitKey();cv2.destroyAllWindows()

#cv2.fillConvexPoly(mask, coords,1)
#mask = mask.astype(np.bool)

#output = np.zeros_like(image)
#output[mask] = image[mask]

#cv2.imshow('image',output)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
