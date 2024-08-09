""" 
Original Version, for CLAM use Nozoom version `gymhisto-n.py`
HistoGym: Openslide Deepzoom Gym Environment
Author: Zhi-Bo Liu
Email: zhibo-liu@outlook.com
Author's Homepage: http://zhibo-liu.com

Version 0.0.1 
- deepzoom level fixed  
- action space = 4 up down left right (no zoomin zoomout)
                discrete not continious
- pixel observation
- pesudo label [1,2]   not [[1,2], [5,9]]

Version 0.0.2
- add step size for action
- add tile level label
- add zoomin zoomout action

Version 0.0.3
- add pixel level label
- add continous action

"""

import sys
import os
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import gym
from gym import spaces
import openslide
from openslide.deepzoom import DeepZoomGenerator
from utils import coordination as Coor

class HistoEnv(gym.Env):
    """
    Description:
        gym-histo: Openslide Deepzoom Gym Environment
        Goal is to mimic how doctors see Whole Slide Images.
        Environment is a Whole Slide Image(WSI). Agent starts at the center 
        of an WSI at certain level. Actions include going up, down, right 
        and left, zoom in & zoom out.
    Tile Position & Image Shape 
        Deepzoom tile position: (Depth, Col, Row) or (z,x,y) 
        Imgae Shape: 
                PIL: shape(H,W,C) 
                Pytorch: (batch_size, channels, height, width) 
                SB3: Box(0, 255, [3, height, width]), dtype=np.uint8.  for RGB pixels
                    SB3 recommend channel first, which is pytorch convention
    Arguments:
        OBS_W, OBS_H        =tile_size. Observation pixel space  (3, OBS_W,OBS_H),
                            eg. (3, 256,256)
        STATE_W,STATE_H     State Space: (STATE_W, STATE_H)  
                            eg (20,16) colum & row (x,y) of deepzoom tile
        slide               OpenSlide Object of a .svs image
        dz                  deepzoom object of the slide
    Observation:
        Type: Box(3, STATE_H, STATE_W) raw pixel convert to np.uint8 (0,255) abtained from deepzoom tile
                spaces.Box(low=0, high=255, shape=(3, self.OBS_H, self.OBS_W), dtype=np.uint8) 
        Num     Observation               Min                     Max
        0       Tile PixelImage            0                      255
        1       Label                      0                       5   
    Actions:
        Type: Discrete(6)
        Num   Action
        0     Agent goes one step up and get the tile image 
        1     Agent goes one step down and get the tile image 
        2     Agent goes one step left and get the tile image 
        3     Agent goes one step right and get the tile image
        4     Agent zoom in get the tile image  
        5     Agent zoom out and get the tile image   
        Note: Right now step size is one. 
    Reward:
        Reward is -0.01 for every step taken, including the termination step 
        Reach the goal +1000 
        Pesudo Label:
            eg: agent_pos: [13,3,8]
                label_pose: [[13,3,7],[13,3,8]]
    Starting State:
        z = biggest thumbnail's level +1 , x = y = 0
    Episode Termination:
        Episode length is greater than 200 .    
        Solved Requirements:
            Average return is greater than or equal to x
        - Create _get_all_tile_size() in order to avoid load right_tile and left_tile when taking action
        - Merge _get_bound(), _get_all_bound(), _get_state_hw()
        - Reward Engineering
    """


    # Define Actions: 
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ZOOM_IN = 4
    ZOOM_OUT = 5
    STAY = 6


    def __init__(self, img_path, xml_path, tile_size , result_path):
        super(HistoEnv, self).__init__()
        
        # 1.Init Args 
        self.img_path = img_path
        self.xml_path = xml_path  # self.mask, self.dz_mask, 
        self.tile_size = tile_size
        self.result_path = result_path
        
        self.plt_size = 10
        self.slide = openslide.OpenSlide(img_path)
        self.dz = DeepZoomGenerator(osr=self.slide, tile_size=self.tile_size, overlap=1, limit_bounds=False)
        self.dz_level = self._get_init_position()[0]
        self.OBS_W, self.OBS_H = self.tile_size, self.tile_size
        self.STATE_W, self.STATE_H = self._get_state_wh() 
        
        # Annotaions
        self.coor_xml = Coor.parse_xml(self.xml_path)
        self.coor_dz_all = Coor.get_dz_coor_all(self.dz, self.coor_xml)
        self.segment_dz_all = Coor.get_segment_all(self.coor_dz_all)
        self.if_overlap, self.overlap_seg_index, self.overlap_ratio = False, None, 0
        
        # 2.Define action and observation space (must be gym.spaces objects)
        self.n_actions = 7
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
                low=0, high=255, shape=(3, self.OBS_H, self.OBS_W), dtype=np.uint8) #sb3 pixel obs must be uint8
        self.agent_pos = self._get_init_position() #(z,x,y)
        self.STATE_D = self._get_init_position()[0] # minial level, for set bound
        print(self.STATE_D)
        self.state = None
        self.count = 0 # count step within episode
        self.max_step = 2000
        self.bound = self._get_all_bound()
        
        try:
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)
        except OSError as e:
            print(e)
            sys.exit(1)
        
        # Print Args 
        print("\n###########################################################################\
                \nimage path: %s\
                \nxml path: %s\
                \nimage save path : %s \
                \nObservation size : (%s, %s) \
                " %(self.img_path,self.xml_path, self.result_path,self.OBS_W,self.OBS_H))
        print("Initial DeepZoom Level : %s \
                \nInitial Stat Size : (%s, %s)  \
                \nMin Level : %s  \
                \nInit Position : %s  \
                \n##########################################################################\n \
                "% (self.dz_level,  self.STATE_W, self.STATE_H, self.STATE_D,self.agent_pos))


    def reset(self):
        """
        Important: The observation returned by `reset()` method must be a numpy array
        the observation is PIL (STATE_H, STATE_W, 3)
        :return: 
        """
        print("HistoEnv Reset!... Time Step is %s" % self.count) #debug
        # Initialize the agent at the center of the tile image grid
        self.agent_pos = self._get_init_position() #(z,x,y)
        self.state = self._get_state()
        self.bound = self._get_all_bound()
        #return self.agent_pos
        self.count = 0
        return self.state

    def step(self,action):#action, agent_pos, n=1, tile_size = tile_size, init_z = init_z, plot = True
        n = 1
        agent_pos = self.agent_pos
        tile_size = self.tile_size
        self.count +=1
        if action == self.UP:
            #print("GO UP...")
            #print(agent_pos)
            
            # ALmost Reach the Boundary
            if agent_pos[2] < 1 : #and agent_pos[2] >= 0:
                agent_pos[2] = 0
            #if type(agent_pos[2])!= int:
                #agent_pos[2] = math.floor(agent_pos[2])
                #print(agent_pos)  
            else:
                agent_pos[2] -= 1
                agent_pos[2] = round(agent_pos[2],2)
                
                #print(agent_pos)  
        if action == self.DOWN:        
            #print("GO DOWN...")            
            # Almost Reach bound and tile is not rectangle:
            if agent_pos[2] + 1 >= self._get_bound()[2]:
                agent_pos[2] = self._get_bound()[2]
                #down_tile = self._get_state()
                down_tile = self.dz.get_tile(agent_pos[0],(agent_pos[1],agent_pos[2]))
                if down_tile.size[1]<tile_size:
                    agent_pos[2] = round(agent_pos[2] -((tile_size-down_tile.size[1])/tile_size),2)
            elif agent_pos[2] + 2 > self._get_bound()[2]:
                agent_pos[2] += 1
                down_tile = self.dz.get_tile(agent_pos[0],(agent_pos[1],agent_pos[2]))
                if down_tile.size[1]<tile_size:
                    agent_pos[2] = round(agent_pos[2] -((tile_size-down_tile.size[1])/tile_size),2)
            else:
                agent_pos[2] += 1
                agent_pos[2] = round(agent_pos[2],2)
        if action == self.LEFT:
            #print("GO LEFT...")
            
            # ALmost Reach the Boundary
            if agent_pos[1] < 1: # and agent_pos[1] >=0:
                agent_pos[1] = 0
            else:
                agent_pos[1] -= 1
                agent_pos[1] = round(agent_pos[1],2)
 
    
        if action == self.RIGHT:
            #print("GO RIGHT...")
            # Almost Reach bound and tile is not rectangle:
            if agent_pos[1] + 1 >= self._get_bound()[1]:
                agent_pos[1] = self._get_bound()[1]
                # right_tile = self._get_state()
                right_tile = self.dz.get_tile(agent_pos[0],(agent_pos[1],agent_pos[2])) #TODO _get_all_tile_size() avoid load tile
                if right_tile.size[0]<tile_size:
                    agent_pos[1] = round(agent_pos[1] -((tile_size-right_tile.size[0])/tile_size),2)
            elif agent_pos[1] + 2 > self._get_bound()[1]:
                agent_pos[1] += 1
                right_tile = self.dz.get_tile(agent_pos[0],(agent_pos[1],agent_pos[2]))
                if right_tile.size[0]<tile_size:
                    agent_pos[1] = round(agent_pos[1] -((tile_size-right_tile.size[0])/tile_size),2)
            else:
                agent_pos[1] += 1
                agent_pos[1] = round(agent_pos[1],2)
            

        if action == self.ZOOM_IN:
            """
            NOTE: ouput is fractional
            Example:
                input : [11,2,3]
                output: [12, 4.5, 6.5]
            """
            #print("ZOOM IN...")
            if agent_pos[0] < self._get_bound()[0]: 
                pos_in = self._get_pos_in()
                if pos_in[1]> self.bound[pos_in[0]][0]:
                    pos_in[1] = self.bound[pos_in[0]][0]
                if pos_in[2] > self.bound[pos_in[0]][1]:
                    pos_in[2] = self.bound[pos_in[0]][1]
                agent_pos = pos_in
                

            else:
                agent_pos = agent_pos
        
        if action == self.ZOOM_OUT:
            """
            NOTE: ouput is fractional
            Example:
                input : [11,2,3]
                output: [10,x.5]
            """
            #print("ZOOM OUT...")
            init_z = self._get_init_position()[0]
            if agent_pos[0] > init_z:
                pos_out = self._get_pos_out()
                if pos_out[1]<0:
                    pos_out[1] = 0
                if pos_out[2]<0:
                    pos_out[2] = 0
                if pos_out[1]> self.bound[pos_out[0]][0]:
                    pos_out[1] = self.bound[pos_out[0]][0]
                if pos_out[2] > self.bound[pos_out[0]][1]:
                    pos_out[2] = self.bound[pos_out[0]][1]
                agent_pos = pos_out
                
            else:
                agent_pos = agent_pos
        if action == self.STAY:
            #print("STAY...")
            agent_pos = agent_pos
        #self.imshow()
        agent_pos[2] = round(agent_pos[2],2)
        agent_pos[1] = round(agent_pos[1],2)
        self.agent_pos = agent_pos
        self.state = self._get_state()

        done = False

    

        # Calculate Reward
        if agent_pos[0] == self._get_bound()[0]:
            self.if_overlap, self.overlap_seg_index, self.overlap_ratio = \
                    Coor.check_overlap(self.coor_dz_all,agent_pos[0],agent_pos[1],agent_pos[2],self.tile_size)
        else:
            self.if_overlap, self.overlap_seg_index, self.overlap_ratio = False, None, 0
            
            

        reward = (self.overlap_ratio * 1000) -0.01
        if self.count >= self.max_step:
            done = True
        
        # reward = self.overlap_ratio -0.01
        # if self.count >= self.max_step:
        #     done = True
        # if score > 0.9 and  self.agent_pos[0] > (self.dz.level_count-1-3): 
        #     done = True



        # # -0.01 reward every setp except when reaching the goal 
        # reward = 1000 if done==True else -0.01

        # Optionally we can pass additional info, we are not using that for now
        info = {
                    "agent position: ":"level %s :(%s,%s)"%(self.agent_pos[0],self.agent_pos[1], self.agent_pos[2])
            }

        #return np.array([self.agent_pos]), reward, done, info
        # print(info, reward) # debug
        print(info, reward , self.overlap_ratio)  # debug
        return self.state, reward, done, info

    # # def render(self, mode='console'):
    # #   if mode != 'console':
    # #     raise NotImplementedError()
    def render(self, mode="save"):
        """
        By Convention:
        - human: render to the current display or terminal and return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel 
        image, suitable for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a terminal-style text representation. 
        The text can include newlines and ANSI escape sequences (e.g. for colors).
        """
        assert mode in ["save", "human", "rgb_array"]
        

        fig = plt.imshow(self.state.transpose(1,2,0))
        if mode == "save":# Save the tile image
            plt.savefig("%s/obs_level_%s_pos_%s_%s_size_%s.png" %(
                self.result_path,
                self.agent_pos[0],
                self.agent_pos[1],
                self.agent_pos[2],
                self.tile_size)
                )
            print("image save at %s/obs_level_%s_pos_%s_%s_size_%s.png" %(
                self.result_path,
                self.agent_pos[0],
                self.agent_pos[1],
                self.agent_pos[2],
                self.tile_size)
                )  
        elif mode == "human":
            plt.imshow(self.state.transpose(1,2,0))
            plt.suptitle("step %s, (%s,%s,%s)"%(
                self.count,
                self.agent_pos[0],
                self.agent_pos[1],
                self.agent_pos[2]        
            ), fontsize=20)
            plt.pause(.5)
            plt.draw()
        elif mode == 'rgb_array':
            return self.state.transpose(1,2,0) #(c,h,w)- > (h,w,c)

        
        #plt.show() # doesen't work

    def close(self):
        pass


    def _get_state_wh(self):
        """
        Get state width height
        Input: 
            self.dz_level 
        Output:
            ( STATE_W, STATE_H )  
        Example:  
            Input: 13
            Output: (23, 17)
        
        # STAT_W, STATE_H = dz.level_tiles[dz_level]
        """
        STAT_W, STATE_H = self.dz.level_tiles[self.dz_level]
        return STAT_W, STATE_H


    def _get_state(self): #agent_pos = agent_pos,dz=dz
        """
        2 types of position
            no zoomin: [110,2,3]
            zoomin:    [[13, 17, 13], [13, 17, 14], [13, 18, 13], [13, 18, 14]]
            zoom in tiles are always comb of 4 tiles and then resize
        """
        tile_size = self.tile_size
        if len(self.agent_pos) == 3:
            tile = self.dz.get_tile(self.agent_pos[0],(self.agent_pos[1],self.agent_pos[2]))
        elif len(self.agent_pos) == 4:
            plt_idx = [  #indexed by colum
                (0,0),
                (0,1),
                (1,0),
                (1,1)
            ]

            w = h = 2 * tile_size
            tile_in = Image.new('RGB',(w,h))
            pos_in = self.agent_pos
    #         # If only pass in first tile location of the zoom in
    #         for i,pos in zip( plt_idx,pos_in):
    #             pos_tmp.append([pos[0],pos[1]+i[0],pos[2] + i[1]])
    #             tile_in.paste(dz.get_tile(pos[0],(pos[1]+i[0],pos[2]+i[1])),\
    #                           (i[0]*tile_size,i[1]*tile_size))
            for i,pos in zip( plt_idx,pos_in):
                tile_in.paste(self.dz.get_tile(pos[0],(pos[1],pos[2])),\
                            (i[0]*tile_size,i[1]*tile_size))
            tile = tile_in.resize((tile_size,tile_size))
        #print(tile.size)
        tile = tile.resize((self.OBS_W,self.OBS_H)) #resize to obs shape
        tile = np.array(tile).transpose(2,0,1) # (h,w,c) --> (c,h,w)  for sb3 channel fisrt
        return tile
    

    def imshow(self): #agent_pos
        """
        2 types of position
            no zoomin: [110,2,3]
            zoomin:    [[13, 17, 13], [13, 17, 14], [13, 18, 13], [13, 18, 14]]
            TODO: zoom in title 
        """
        plt_size = self.plt_size
        #tile = dz.get_tile(agent_pos[0],(agent_pos[1],agent_pos[2]))
        tile = self._get_state().transpose(1,2,0) # (c,h,w)- > (h,w,c)
        print(tile.size)
        print(type(tile))
        plt.figure(figsize=(plt_size, plt_size))
        if len(self.agent_pos) == 3:
            plt.title("tile (%s,%s,%s)"% (self.agent_pos[0],self.agent_pos[1],self.agent_pos[2]),fontsize=20)
        elif len(self.agent_pos) == 4:
            plt.title("tile TODO zoom in (level: %s) :" \
                % (self.agent_pos[0][0]), fontsize=20)
        plt.axis('off')
        fig = plt.imshow(tile)
    
        plt.draw()

    def _get_bound(self):#agent_pos = agent_pos,dz=dz
        """
        Dynamically get the max bound of state
            Input: self.agent_pos (z,x,y)
            Output: (max_z, max_x, max_y)
            Note: x, y start from 0, max_x -1 is correct bound
                return is int
        """
        min_z = self.STATE_D
        z,x,y = self.agent_pos
        max_z = len(self.dz.level_tiles) -1
        if z > max_z: z = max_z
        #if z < 0: z = 0
        if z < min_z: z = min_z
        max_x,max_y = self.dz.level_tiles[z]
        # print("Bound:",max_z,max_x-1,max_y-1)
        return max_z, max_x-1, max_y-1
    def _get_all_bound(self): #dz =dz
        """
        Retuen bound for all level for dz.get_tile / _get_state
        Example:
        input : dz
        output: {9: (0.42, 0.04),
                10: (1.84, 1.07),
                11: (4.68, 3.14),
                12: (10.36, 7.29),
                13: (21.72, 15.57),
                14: (44.43, 32.14),
                15: (89.87, 65.27),
                16: (180.73, 131.54),
                17: (362.46, 264.08)}
        Note: bound is float not int
        """
        
        min_z = self.STATE_D
        max_z = len(self.dz.level_tiles) -1
        tile_size = self.tile_size
        bound = {}
        #print(max_z)
        #max_x,max_y = dz.level_tiles[z]
        

        for z in range(min_z, max_z+1):
            max_x,max_y = self.dz.level_tiles[z] 
            max_x -=1
            max_y -=1
            #imshow([z,max_x,max_y])
            rd_tile = self.dz.get_tile(z,(max_x,max_y)) # rghit down tile
            if rd_tile.size[0]<tile_size:
                max_x = round(max_x -((tile_size-rd_tile.size[0])/tile_size),2)
            if rd_tile.size[1]< tile_size:
                max_y = round(max_y -((tile_size-rd_tile.size[1])/tile_size),2)
            #imshow([z,max_x,max_y])
            bound[z] = (max_x, max_y)
            # print("bond is (%s, %s, %s)"% (z, max_x,max_y))
        return bound 

    def _get_pos_in(self, n = 1):#agent_pos, n = 1
        """
        Get location of next n level's high resolution tile, output should look identical to input
        input: agent position (y,x,z)
                n: how many level to zoomin
        output: (y',x',z') 4 cooresponding level k+1 tiles location
        Example:
            input : [11, 1.75, 1.25] , n = 1
            output: [12,4,3]
            input : [12,4,3]
            output: [13, 8.5, 6.5]
        NOTE: n =1, TODO n = 2,3,4
        """
        # assert type(n)==int and n > 0

        z = self.agent_pos[0]+ n
        x = self.agent_pos[1] * (2 ** n)
        y = self.agent_pos[2] * (2 ** n)
        #print(z,x,y)
            
        pos_in = []
        for i in range(2 ** n):
            for j in range(2 ** n):
                pos_in.append([z,x+i,y+j])
        pos = [z, pos_in[0][1]+0.5, pos_in[0][2]+0.5]
        return pos

    def _get_pos_out(self, n = 1): #agent_pos, n = 1
        """
        Get location of next n level's high resolution tile, output should look identical to input  
        Input: agent position (y,x,z)
                n: how many level to zoomin
        Output:  4 cooresponding level k+1 tiles location
        Example:
            input [12,4,3] , n = 1
            output: [11, 1.75, 1.25]
        NOTE: n =1, TODO n = 2,3,4
        """

        # assert type(n)==int and n > 0


        z = self.agent_pos[0]- n
        x = (self.agent_pos[1]-0.5) / (2 ** n)
        y = (self.agent_pos[2]-0.5) / (2 ** n)
        #print(z,x,y)

        pos = [z,x,y]
        # print(pos)
        return pos
    def _get_init_position(self):#dz=dz
        """
        Get initial position. Level init_z should contain at least 2 tiles 
                            Level init_z-1 should contain 1 tile
        Input : dz
        Output: [ini_z,0,0]   init_z: initial level
        """
        for i,(dim, size) in enumerate(zip(self.dz.level_tiles,self.dz.level_dimensions)):
            if (dim[0]>1 or dim[1]>1) and \
                (size[0]>self.tile_size and size[1]>self.tile_size):
                #print("first level tile size is {}".format(size))
                #print("first level at deepzoom level %s" % i)
                return [i,0,0]

  


if __name__ == "__main__":


    import numpy as np



    """### Testing the environment"""

    # 1.Init Args
    #img_path = '/home/zhibo/data/tcga/2.svs'
    img_path ='/home/zhibo/data/minicamelyon16/tumor_001.tif'
    xml_path = '/home/zhibo/data/minicamelyon16/tumor_001.xml'
    # tile_size = 256
    tile_size = 128 # best for humman
    # tile_size = 64 # seems too small
    # tile_size =32

    result_path = './results'


    env = HistoEnv(img_path,xml_path, tile_size,result_path)

    # # Validate env
    # from stable_baselines3.common.env_checker import check_env
    # # If the environment don't follow the interface, an error will be thrown
    # check_env(env, warn=True)

    obs = env.reset()


    print("Observation Space : %s "% env.observation_space)
    print("Action Space : %s "% env.action_space)
    print("Sample an Action : %s "% env.action_space.sample())

    Action = np.array(
    [4,4,1,4,3,3])
    n_steps = len(Action)
    # Action = np.repeat(4,20)

    i = 0
    #env.reset()
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        #print('action=', Action[i])
        obs, reward, done, info = env.step(Action[i])
        # print('action=', Action[i], 'info=', info, 'obs=', obs, 'reward=', reward, 'done=', done)
        print('action=', Action[i], 'info=', info, 'reward=', reward, 'done=', done)

        #env.render(mode = "save")
        env.render(mode = "human")
        #env.render(mode = "rgb_array")
        if done:
            print("Episode Terminiated", "reward=", reward)
            break
        i +=1





# 3. Record Video
# TODO
