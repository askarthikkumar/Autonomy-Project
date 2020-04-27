# to remove tensorflow warnings
import warnings
warnings.filterwarnings("ignore")
import os,logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# RL packages
import gym
import time
from gym import spaces
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from stable_baselines.sac.policies import MlpPolicy,CnnPolicy
from stable_baselines import SAC
# RLBench Packages
from empty_container import *
from camera import Camera
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
# other packages
from quaternion import quaternion, as_rotation_vector
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
OBJECT = "Shape"
# N_ACTIONS = 8

# make custom environment class to only read values. It should not directly change values in the other thread.
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, machine, camera, state="state"):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.machine = machine
        self.camera = camera
        self.action_space = spaces.Box(
            low=np.array([-np.pi,-np.pi/4,-np.pi/4]), high=np.array([np.pi,np.pi/4,np.pi/4]))
        if state=="state":
            self.observation_space = spaces.Box(low = -10, high=10, shape = (1,15))
        elif state=="vision":
            self.observation_space = spaces.Box(low = 0, high=1, shape = (128,128,3))
        self.min_force = 1
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        self._gym_cam = VisionSensor.create([640, 360])
        self._gym_cam.set_pose(cam_placeholder.get_pose())
        self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
        # self._gym_cam.set_render_mode(RenderMode.OPENGL3)
        self.state_rep = state

    def step(self, action):
        print(action)
        quat = self.machine.get_full_grasp_pose(action)
        objs = self.machine.get_objects(True)
        pose = objs[OBJECT].get_pose()
        print("ACTION TAKEN", action)
        try:
            path = self.machine.move_to(pose=pose, pad=0, quat=quat)
            self.machine.execute(path)
            cond = self.machine.grasp(objs[OBJECT])
            path = self.machine.move_to(pose=pose, quat=quat, pad=0.20)
            self.machine.execute(path)
            reward, success = self.get_reward(cond)
            obs = self.make_obs()
            done = True
            info = {"success":success}
            print("REWARD",reward)
            return obs,reward,done,info
        except:
            print("COULD NOT FIND PATH")
            reward = -1
            obs=self.make_obs()
            info = {}
            info["success"] = False
            print("REWARD",reward)
            return obs,reward,True,info

    def get_reward(self, cond):
        if cond is False:
            return -1, False
        else:
            forces = self.machine.env._robot.gripper.get_joint_forces()
            for force in forces:
                if force < self.min_force:
                    return -1, False
            return 1, True
        
    def make_obs(self):
        '''
        State vector of length 17
        contains two poses, and length,width and height of object bb
        '''
        state = None
        if self.state_rep=="state":
            objs = self.machine.get_objects(True)
            gripper_pose = list(self.machine.env._scene.get_observation().gripper_pose)
            gripper_quat = gripper_pose[3:]
            gripper_quat[0], gripper_quat[-1] =gripper_quat[-1], gripper_quat[0]
            quat = quaternion(*gripper_quat)
            g_vec = list(as_rotation_vector(quat))
            obj_pose = list(objs[OBJECT].get_pose())
            obj_quat = obj_pose[3:]
            obj_quat[0], obj_quat[-1] = obj_quat[-1], obj_quat[0]
            quat = quaternion(*obj_quat)
            r_vec = list(as_rotation_vector(quat))
            obj_bb = objs[OBJECT].get_bounding_box()
            x_diff = obj_bb[0]-obj_bb[3]
            y_diff = obj_bb[1]-obj_bb[4]
            z_diff = obj_bb[2]-obj_bb[5]
            state = np.array(gripper_pose[:3]+g_vec+ \
                                obj_pose[:3]+r_vec+[x_diff,y_diff,z_diff]).reshape(1,-1)
        elif self.state_rep=="vision":
            obs = self.machine.env._scene.get_observation()
            state = obs.wrist_rgb
            # plt.imsave("image.png",state)
        return state

    def reset(self):
        _,obs = self.machine.task.reset()
        # left_shoulder = obs.left_shoulder_rgb
        # right_shoulder = obs.right_shoulder_rgb
        objs = self.machine.get_objects(True)
        pose = objs[OBJECT].get_pose()
        while True:
            try:
                self.machine.go_to(pose, pad=0.2)
                print("here")
                break
            except:
                self.machine.task.reset()
                objs = self.machine.get_objects(True)
                pose = objs[OBJECT].get_pose()
                print("Path not found. Retrying after task reset")
        obs = self.make_obs()
        return obs

    def render(self, mode='human'):
        pass

    def close (self):
        pass

def test():
    machine = StateMachine()
    machine.initialize(headless=True)
    camera = Camera(machine)
    env = CustomEnv(machine, camera, state="vision")
    path = os.path.join(dir_path,"../Models/Grasp_Model")
    model = DQN.load(path)
    obs = env.reset()
    count = 0
    success = 0
    while count < 100:
        done = False
        print("Count ", count, "Success ", success)
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            # action = env.action_space.sample()
            print(action)
            obs, reward, done, info = env.step(action)
            print("Reward",reward)
        count += 1
        if info["success"]:
            success += 1
    print("Success Rate ", success / count, success, count)

def train():
    machine = StateMachine()
    machine.initialize(headless=True)
    camera = Camera(machine)
    env = CustomEnv(machine, camera, state="vision")
    model = SAC(CnnPolicy, env, verbose=1, learning_starts=32, batch_size=32, \
                target_update_interval=32, tensorboard_log=dir_path+'/Logs/')
    model.learn(total_timesteps=2000, log_interval=1000000)
    model.save("Grasp_Model_Full_Pose")

if __name__ == "__main__":
    # to train
    train()
    # to test
    # test()