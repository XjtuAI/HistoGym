"""
Visualize Segmentation 
"""

#import openslide
#from openslide.deepzoom import DeepZoomGenerator
#import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as et
from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry import box as Box




def parse_xml(xml_path):
    '''
    Function to parse coordinates in XML format into a list of tuples

    Arguments:
       - xml_filename: XML filename
       - xml_directory: directory where the XML is located
    
    Returns:
       - coordinates: a list of tuples containing coordinates, divided
                      into segments ([[segment_1], [segment_2], etc])
                      where each segment consists of (x,y) tuple
    '''
    with open(xml_path, 'rt') as f:
        tree = et.parse(f)
        root = tree.getroot()

    # Get the number of coordinates lists inside the xml file
    count = 0
    for Annotation in root.findall('.//Annotation'):
        count += 1

    # Make a list of tuples containing the coordinates
    temp = []
    for Coordinate in root.findall('.//Coordinate'):
        order = float(Coordinate.get('Order'))
        x = float(Coordinate.get('X'))
        y = float(Coordinate.get('Y'))
        temp.append((order, x,y))

    # Separate list of tuples into lists depending on how many segments are annotated
    coordinates = [[] for i in range(count)]
    i = -1
    for j in range(len(temp)):
        if temp[j][0] == 0:
            i += 1
        x = temp[j][1]
        y = temp[j][2]
        coordinates[i].append((x,y))
        #coordinates[i].append([x,y])
    
    return coordinates

    
def get_dz_coor(dz, coordinates, dz_level):
    """
    Get DeepZoom Level <dz_level>'s Coordinate:

    Arguments:
         - dz: DeepZoom Object
    - coordinates: list of segment from wsi [[segment_0], [segment_1], etc],
                   where each segment consist of [x,y] list

    Return: 
        - dz_coor: [[segment_1],[segment_2],etc],
                    where each segment consist of [x,y] list
    """
    X_COOR_DZ, Y_COOR_DZ = dz.level_dimensions[dz_level]
    X_COOR_SLIDE, Y_COOR_SLIDE = dz.level_dimensions[-1]

    X_RATIO = X_COOR_DZ/X_COOR_SLIDE
    Y_RATIO = Y_COOR_DZ/Y_COOR_SLIDE
    
    coor = [[] for x in range(len(coordinates))]
    for i , _ in enumerate(coor):
        coor[i] = [list(elem) for elem in coordinates[i]]

    for i in range(len(coor)):
        for j in range(len(coor[i])):
            #print(coor[i][j])
            coor[i][j][0] = coor[i][j][0] * X_RATIO
            coor[i][j][1] = coor[i][j][1] * Y_RATIO
    return coor       

def get_dz_coor_all(dz, coordinates):
    """
    Get  All DeepZoom Level Coordinate:

    Arguments:
        - dz: Deepzoom Object
        - coordinates: list of segment from wsi [[segment_0], [segment_1], etc],
                   where each segment consist of (x,y) tuple

    Return: 
        - coor_all: [[segment_level_0], [segment_level_1],etc], 
                 where each segment_level_0 consist of [[segment_0],[segment_1],etc]
    """

    coor_all = [[] for x in range(len(dz.level_dimensions))]
    for level in range(len(dz.level_dimensions)):
        coor_all[level] = get_dz_coor(dz, coordinates, level)

    return coor_all



def get_slide_coor(slide, coordinates, slide_level):
    """
    Get Slide Level <slide_level>'s Coordinate:
    Return: [[segment_1],[segment_2],etc],
        where each segment consist of [x,y] list
    """
    X_COOR_DZ, Y_COOR_DZ = slide.level_dimensions[slide_level]
    X_COOR_SLIDE, Y_COOR_SLIDE = slide.level_dimensions[0]

    X_RATIO = X_COOR_DZ/X_COOR_SLIDE
    Y_RATIO = Y_COOR_DZ/Y_COOR_SLIDE

    coor = [[] for x in range(len(coordinates))]
    for i , _ in enumerate(coor):
        coor[i] = [list(elem) for elem in coordinates[i]]
    # coor = coordinates.copy()
    
    for i in range(len(coor)):
        for j in range(len(coor[i])):
            #print(coor[i][j])
            coor[i][j][0] = coor[i][j][0] * X_RATIO
            coor[i][j][1] = coor[i][j][1] * Y_RATIO
    
    return coor

def print_coor(coor): 
    print("length of sementation cooridates is %s" %len(coor))
    for i in range(5) : print(coor[0][i])


def slide_contour_image(tile, coor):
    """
    take tile = entire wsi at level slide_level, and corresponding coordinate, 
    Return: PIL image with segmentation boundary
    """
    tile = np.asarray(tile)
    coxy = [[] for x in range(len(coor))]
    for i , _ in enumerate(coxy):
        coxy[i] = np.asarray(coor[i], dtype=np.int32)
    tile_contour = cv2.drawContours(tile, coxy, -1, (0, 255, 0), 1) #np.array
    tile_contour_pil=Image.fromarray(tile_contour)
    return tile_contour_pil

def xy_to_coor_dz(x_tile, y_tile, tile_size=256):
    """
    deepzoom tile's posion(x,y) ---> 对应dz_level的整个拼接的wsi的坐标
    """
    x_coor = x_tile * tile_size
    y_coor = y_tile * tile_size
    minx, miny, maxx, maxy = x_coor, y_coor, x_coor+tile_size, y_coor+ tile_size
    return minx,miny,maxx,maxy

def coor_to_xy_dz(x_coor,y_coor, tile_size=256):
    """
    对应dz_level的整个拼接的wsi的坐标 --->  deepzoom tile's posion(x,y)
    """
    x_tile = x_coor / tile_size
    y_tile = y_coor / tile_size
    return x_tile, y_tile

def get_coor_segbox(coor):
    """
    Input: segmentation coordinates [[coor_1], [coor_2], etc]
    Output: segmentation box's corrdinates
    """
    segment = [[] for x in range(len(coor))]
    coor_box = [[] for x in range(len(coor))]
    for i in range(len(coor)):     
        segment[i] = Polygon(coor[i])
        segment[i] = segment[i].buffer(0)
        assert segment[i].is_valid

        minx, miny, maxx, maxy = segment[i].bounds
        box = Box(minx, miny, maxx, maxy)
        coor_box[i] = list(box.exterior.coords)
    
    return coor_box

def get_minxy_box(coor_box):
    """
    Input :Coordinates of box [[coor_1], [coor_2], etc]
    Output:
    """
    xy = [[] for x in range(len(coor_box))]
    for i,coor in enumerate(coor_box):
        x = []
        y = []
        for j,coxy in enumerate(coor):
            x.append(coxy[0])
            y.append(coxy[1])
        xy[i] = [min(x),min(y)]    
    return xy

def get_segment(coor_dz):
    ''' 
    Get Segment for one deepzoom level
    '''
    segment_dz = [[] for x in range(len(coor_dz))]
    for i, coor in enumerate(coor_dz):
        segment_dz[i] = Polygon(coor).buffer(0)
        #segment_dz[i] = segment_dz[i].buffer(0)
    return segment_dz

def get_segment_all(coor_dz_all):
    ''' 
    Get Segment for all deepzoom level
    '''
    segment_all = [[] for x in range(len(coor_dz_all))]
    for level in range(len(coor_dz_all)):
        segment_all[level] = get_segment(coor_dz_all[level])

    return segment_all

def get_tile_box(x_tile, y_tile, tile_size):
    minx,miny,_,_ = xy_to_coor_dz(x_tile,y_tile)
    tile_box = Box(minx, miny, minx+tile_size, miny+tile_size)
    return tile_box
# def check_overlap(coor_dz,x_tile,y_tile,tile_size):
#     if_overlap = False
#     overlap_seg_index = None

#     # Get Segment
#     segment_dz = get_segment(coor_dz)
#     # Get Tile Box
#     tile_box = get_tile_box(x_tile,y_tile, tile_size)
#     # Check Overlap
#     for i, seg in enumerate(segment_dz):
#         if seg.intersects(tile_box):
#             if_overlap = True
#             overlap_seg_index = i
#     # Get Overlap Ratio
#     if if_overlap:
#         overlap = segment_dz[overlap_seg_index].intersection(tile_box)
#         overlap_ratio = overlap.area / tile_box.area
#     else:
#         overlap_ratio =0
#     return if_overlap, overlap_seg_index, overlap_ratio

def check_overlap(coor_dz_all,dz_level, x_tile,y_tile,tile_size):
    if_overlap = False
    overlap_seg_index = None

    # Get Segment
    segment_dz = get_segment_all(coor_dz_all)[dz_level]
    # Get Tile Box
    tile_box = get_tile_box(x_tile,y_tile, tile_size)
    # Check Overlap
    for i, seg in enumerate(segment_dz):
        if seg.intersects(tile_box):
            if_overlap = True
            overlap_seg_index = i
    # Get Overlap Ratio
    if if_overlap:
        overlap = segment_dz[overlap_seg_index].intersection(tile_box)
        overlap_ratio = overlap.area / tile_box.area
    else:
        overlap_ratio =0
    return if_overlap, overlap_seg_index, overlap_ratio


def imshow(dz,dz_level, x_tile, y_tile, plt_size=10): #agent_pos
    """

    """
    tile = dz.get_tile(dz_level,(x_tile,y_tile))

    plt.figure(figsize=(plt_size, plt_size))
    plt.title("tile %s, (%s, %s)"% (dz_level, round(x_tile,2), round(y_tile,2)),fontsize=20)
    plt.axis('off')
    fig = plt.imshow(tile)
    #plt.pause(5)
    plt.waitforbuttonpress()
    plt.draw()

def imshow_slide(slide, slide_level=7,plt_size = 10): #agent_pos
    """

    """
    x_dim_slide , y_dim_slide = slide.level_dimensions[slide_level]
    tile = slide.read_region((0,0),slide_level,(x_dim_slide, y_dim_slide))
    plt.figure(figsize=(plt_size, plt_size))
    plt.title("Slide at Level %s"% (slide_level),fontsize=20)
    plt.axis('off')
    fig = plt.imshow(tile)
    plt.waitforbuttonpress()
    plt.draw()


if __name__ == '__main__':
    import openslide
    from openslide.deepzoom import DeepZoomGenerator
    import os
    from shapely.geometry import Polygon
    from shapely.geometry import box as Box

    dataroot = "/home/zhibo/data/minicamelyon16/"
    img_path = os.path.join(dataroot, "tumor_001.tif")
    mask_path = os.path.join(dataroot, "tumor_001_mask.tif")
    xml_path  = os.path.join(dataroot, "tumor_001.xml")

    slide_level = 7
    dz_level = 15
    tile_size = 256

    slide = openslide.OpenSlide(img_path)
    dz = DeepZoomGenerator(osr=slide, tile_size=tile_size, overlap=0, limit_bounds=False)
    x_dim_slide , y_dim_slide = slide.level_dimensions[slide_level]




    ### 1. Get Segmentation Coordiantions
    coor_xml = parse_xml(xml_path)
    coor_dz = get_dz_coor(dz, coor_xml, dz_level)
    coor_slide = get_slide_coor(slide, coor_xml, slide_level)
    # print_coor(coor_xml)
    # print_coor(coor_dz)
    # print_coor(coor_slide)


    ### 2. Get 1 Tile Location from Segmentation
    # convert segment cooridination to box coordination
    coor_segbox_dz = get_coor_segbox(coor_dz)
    # get upper left box coordination
    xy_segbox_dz = get_minxy_box(coor_segbox_dz)
    # get one of the box upper left coordination
    minx, miny = xy_segbox_dz[1][0], xy_segbox_dz[1][1]
    print(minx,miny)
    # get box's deepzoom level tile location
    x_tile, y_tile = coor_to_xy_dz(minx ,miny)
    print(x_tile, y_tile)


    ### 3. Save Segmentation (origin and box) Image 
    # image with origin segmentaion 
    tile = slide.read_region((0,0),slide_level,(x_dim_slide, y_dim_slide))
    image_seg = slide_contour_image(tile, coor_slide)
    image_seg.save('../results/image_seg.png')
    print('image with segmentation saved at results/image_seg.png')

    # image with box segmentation
    coor_seg_box = get_coor_segbox(coor_slide)
    tile = slide.read_region((0,0),slide_level,(x_dim_slide, y_dim_slide))
    image_segbox = slide_contour_image(tile, coor_seg_box)
    image_segbox.save('../results/image_segbox.png')
    print('image with segmentation box saved at results/image_segbox.png')


    ### 4. Give Tile Position (x_tile, y_tile), check overlap
    minx,miny,_,_ = xy_to_coor_dz(x_tile,y_tile)
    print('Tile: (%s,(%s,%s))'%(dz_level, x_tile,y_tile))

    segment_dz = Polygon(coor_dz[1])
    segment_dz = segment_dz.buffer(0)

    tile_box = Box(minx, miny, minx+tile_size, miny+tile_size) #暂时这样，tile不是正方形
    print('Tile overlap seg: ', segment_dz.intersects(tile_box))
    overlap = segment_dz.intersection(tile_box)
    print('Overlap Ratio: ',overlap.area / tile_box.area)


    imshow(dz, dz_level,x_tile,y_tile, 9)
    imshow_slide(slide)


