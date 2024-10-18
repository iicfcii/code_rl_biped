from idealab_tools.kinematics.kinematics import Quaternion
from typing import Dict, Union
import mujoco
from gymnasium import spaces
import gymnasium as gym
import gymnasium
import numpy as np
import numpy
from gymnasium import utils
# from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import math
import os


class EarlyTerm(RuntimeError):
    pass


def mycontroller(model, data):

    A = .25
    d = 0 
    ts = .25
    f = .5
    l1 = 0
    l2 = .5

    t = data.time
    linear_act = t*d/ts
    if linear_act < d:
        linear_act = d
    pos1 = A*math.sin(2*math.pi*(f*t-l1))
    pos2 = A*math.sin(2*math.pi*(f*t-l2))
    data.ctrl = [pos1,pos2]
    return

class BipedEnv(gymnasium.Env):
    metadata = {'render_modes': ['human','none']}
    motor_joint_names = ['j1','j2']
    timestep = 1e-3
    num_actuators = 2
    tip_angle_limit = 90
    body_max_height = .3
    constraint_violation_angle_limit = 20
    width=800
    height = 600

    reward_labels = ['forward_progress'
                     ,'episode_time'
                     ,'current_velocity'
                     ,'z_violation'
                     ,'tip_angle'
                     ,'control_cost'
                     ,'constraint_violation_angle'
                     ,'efficiency'
                     ,'energy_produced_per_gait_cycle'
                     ,'energy_consumed_per_gait_cycle'
                     ,'cycle_efficiency'
                     ,'action_actual'
                     ]

    reward_weights = {}
    reward_weights['forward_progress'] = 3e0
    reward_weights['current_velocity'] = 1e1

  
    def __init__(self, render_mode=None, controller_freq = 100):
        self.action_space = spaces.Box(low=-math.pi/2, high=math.pi/2, shape=(self.num_actuators,),dtype=np.float64)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(18,),dtype=np.float64)

        my_model_path ='my_envs/biped.xml'

        with open(my_model_path,) as f:
            xml_template = f.read()

        xml_string = xml_template.format(l_ankle=.03,k=.1,b=.01,ts=self.timestep,width=self.width,height=self.height)
        
        self.controller_freq = controller_freq
        self.frame_skip = int(1/self.timestep/self.controller_freq)

        self.render_mode = render_mode

        self.camera_name, self.camera_id = "target", 1
    
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        self.model = model
        self.data = data
        self.last_time = self.data.time
        self.early_term = False

        self.mujoco_renderer = None

        if render_mode == 'human':

            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            self.mujoco_renderer = MujocoRenderer(self.model, self.data)

        
        self.init_qpos = numpy.array([ 0,0,1.05e-01,  1.00000000e+00,
        2.44928688e-19, -4.18995264e-14,  6.04504389e-12,  1.31314879e-13,
        4.96048416e-13, -5.28383532e-13, -1.30259920e-13, -3.73671632e-13,
        5.71505308e-13])
        self.init_qvel = numpy.zeros(self.data.qvel.shape)

        mujoco.set_mju_user_warning(self._mju_user_warning)    

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = numpy.array(self.init_qpos)
        self.data.qvel = numpy.array(self.init_qvel)
        self.last_x_pos = self.data.body('trunk').xpos.copy()[0]

        if self.render_mode == "human":
            self.render()
    
        observation = self._get_obs()
        info = {}
        return observation, info
    

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
        }    
    
    def render(self,*args,**kwargs):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "human":
            width, height = self.width, self.height
            camera_name, camera_id = self.camera_name, self.camera_id
            self.mujoco_renderer.render(self.render_mode)
        if self.render_mode =='none':
            pass

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()    


    def _get_obs(self):

        obs = []
        obs.append(self.data.sensor('body_pos').data.copy())        
        obs.append(self.data.sensor('body_v').data.copy())      
        obs.append(self.data.sensor('body_w').data.copy())        
        obs.append(self.data.sensor('body_x').data.copy())        
        obs.append(self.data.sensor('body_y').data.copy())        
        obs.append(self.data.sensor('body_z').data.copy())
        obs = numpy.array(obs).flatten()        

        if any(numpy.isinf(obs)):
            raise EarlyTerm

        if any(numpy.isnan(obs)):
            raise EarlyTerm

        return obs


    def step(self, action):
        # print(action.shape)
        if np.array(action).shape != (self.model.nu,):
            raise ValueError(f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(action).shape}")

        last_pos = self.data.body('trunk').xpos

        for ii in range(self.frame_skip):
        
            last_time_inner = self.data.time
            
            self.data.ctrl[:] = action

            mujoco.mj_step(self.model, self.data)

        pos = self.data.body('trunk').xpos
        vel = pos - last_pos

        observation = self._get_obs()
        reward = pos[0]*self.reward_weights['forward_progress']+vel[0]*self.reward_weights['current_velocity']
        
        truncated = False
        info = {}

        self.render()

        return observation, reward, self.early_term, truncated, info

    def _mju_user_warning(self, e):
        self.early_term = True