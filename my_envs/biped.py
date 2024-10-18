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

class BipedEnv(gymnasium.Env):
    metadata = {'render_modes': ['human','none']}
    timestep = 1e-3
    num_actuators = 2
    width=800
    height = 600
    max_steps = 500

    # Scaling for observations which can be determined by
    # sampling some random actions and measuring the std
    # and scaling the std to around 0.3 so that the values
    # are near the -1 to 1 range. 
    body_w_scale = 0.1
    body_z_scale = 1
    q_scale = 1
    dq_scale = 0.1

    # Actions should also be scaled to about -1 to 1. 
    action_scale = 1
    q_max = 1.0

    def __init__(self, render_mode=None, controller_freq = 100):
        # Stable-Baselines3 clips action values based on the lows and highs. 
        # https://github.com/DLR-RM/stable-baselines3/blob/56c153f048f1035f239b77d1569b240ace83c130/stable_baselines3/common/policies.py#L379
        # It may cause the policy to not care about the range and output square waves, which is not desirable. 
        # A large range is chosen here to remove the effect of clipping. 
        # Joint limits can be achieved by adding penalty terms to the reward. 
        self.action_space = spaces.Box(low=-10, high=10, shape=(self.num_actuators,),dtype=np.float64)

        # Not sure whether the observations are clipped based on lows and highs. 
        # Set to infinite anyway since the scaling should handle them. 
        # Worth clipping them to prevent huge values due to simulation errors. 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,),dtype=np.float64)

        my_model_path ='my_envs/biped.xml'
        with open(my_model_path,) as f:
            xml_template = f.read()
        xml_string = xml_template.format(l_ankle=.03,k=.1,b=.01,ts=self.timestep,width=self.width,height=self.height)
        
        self.controller_freq = controller_freq
        self.frame_skip = int(1/self.timestep/self.controller_freq)

        self.render_mode = render_mode
    
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        self.model = model
        self.data = data
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
        self.truncated = False

        self.q_history = np.zeros((100, 2))
        self.q_history_ptr = 0
        self.last_action = np.zeros(self.num_actuators)

        if self.render_mode == "human":
            self.render()
    
        observation = self._get_obs()
        info = {}
        return observation, info  
    
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
            self.mujoco_renderer.render(self.render_mode)
        if self.render_mode == 'none':
            pass

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()    

    def _get_obs(self):
        body_w = self.data.sensor('body_w').data.copy()
        body_z = self.data.sensor('body_z').data.copy()
        q = self.data.actuator_length.copy()
        dq = self.data.actuator_velocity.copy()
        obs = np.concatenate([
            # Body angular velocity
            body_w * self.body_w_scale,
            # Body z-axis for the orientation information
            # There is no yaw information, but it should be enough. 
            body_z * self.body_z_scale,
            # Joint angles and speeds
            q * self.q_scale,
            dq * self.dq_scale,
            # Last actions are very useful for keeping the action smooth by penalizing the action change rate. 
            # The policy needs this information to not change too much at every step. 
            self.last_action
        ])        
        
        return obs


    def step(self, action):
        if np.array(action).shape != (self.model.nu,):
            raise ValueError(f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(action).shape}")
            
        self.data.ctrl[:] = action*self.action_scale
        mujoco.mj_step(self.model, self.data,nstep=self.frame_skip)

        vx = self.data.sensor('body_v').data[0]
        da = (action - self.last_action) / (1/self.controller_freq)
        body_z = self.data.sensor('body_z').data.copy()
        q = self.data.actuator_length.copy()
        q_off_limits = (
            (q - np.array([self.q_max,self.q_max])).clip(min=0) -  # upper bound
            (q - np.array([-self.q_max,-self.q_max])).clip(max=0)  # lower bound
        )
        self.q_history[self.q_history_ptr] = q
        self.q_history_ptr += 1
        if self.q_history_ptr >= self.q_history.shape[0]:
            self.q_history_ptr = 0
        q_mean = np.mean(self.q_history, axis=0)

        rewards = np.array([
            # Encourage walking forward
            1 * vx,
            # Since the body's local velocity is used, the body needs to stay upright so that the robot actually walks forward. 
            # Velocity in another frame may be used if the body shouldn't stay upright. 
            -1e-1 * np.sum(body_z[:2]**2),
            # Penalize action change rate for smooth actions
            -1e-6 * np.sum(da**2),
            # Penalize joints going over limits. 
            -10 * np.sum(q_off_limits**2),
            # Penalize average joint angles so that joints oscillates around 0 (vertical). 
            -0.5 * np.sum(q_mean**2),
        ])

        self.last_action = action
        observation = self._get_obs()
        reward = np.sum(rewards)

        # Once the robot falls, it can't really get back up so the data afterwards are not useful. 
        if np.any(np.abs(body_z[:2])>0.7):
            self.truncated=True
        info = {}

        self.render()

        return observation, reward, False, self.truncated, info

    def _mju_user_warning(self, e):
        self.truncated = True