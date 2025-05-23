import gym
from gym import spaces
import numpy as np

import pybullet as p
import pybullet_data
from utils import TopologicallyRegularizedLoss, entropy, seed_filling
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import scipy
from scipy import ndimage
import cv2
import numpy as np



class InukshukEnv_v3(gym.Env):
    def __init__(self, render: bool = False):

        self._render = render
        # 定义动作空间
        self.action_space = spaces.MultiDiscrete(np.array([2, 4, 6]))
        '''self.action_space = spaces.Box(
            low=np.array([-1, -1., 0.001, -1, -1, -1]),
            high=np.array([1, 1., -0.001, 1, 1, 1]),
            dtype=np.float64
        )'''

        # self.action_space = spaces.utils.flatten_space(self.unflatten_action_space)
        # 定义状态空间
        self.unflatten_observation_space = spaces.Box(
            low=-1.,
            high=1.,
            shape=(20, 7),
            dtype=np.float64
        )
        self.flatten_observation_space = spaces.flatten_space(self.unflatten_observation_space)
        self.observation_space = spaces.Dict({'pos': self.flatten_observation_space,
                                              'image': spaces.Box(0, 255, (256, 256, 3), np.uint8)})

        self.loss_calculator = TopologicallyRegularizedLoss()
        self.image_real = Image.open('Logger/bw2.jpg')
        # 假设阈值为128，大于等于128的元素转换为255，小于128的元素转换为0
        self.binary = np.where(np.array(self.image_real)[:, :, 0] >= 128, 255, 0)

        #a = Image.open('Logger/bw1.jpg')
        #b = Image.open('Logger/bw2.jpg')
        #print(self.loss_calculator.reconstruct_loss(a, b))
        self.current_reward_similarity = 0
        self.current_reward_x = 0
        self.current_reward_holes = 0
        self.alpha = 0

        # 连接引擎
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)

        # 石块存储器
        self.stone_container = []
        self.state_y = []
        self.state_x = []
        self.h = [0, 1, 1, 0, 0, 0]
        # 计数器
        self.step_num = 0
        # 平地
        self.plane = None
        # 碰撞标记
        self.collision = False

        p.resetSimulation(physicsClientId=self._physics_client_id)
        # 添加资源路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 设置环境重力加速度
        p.setGravity(0, 0, -2.8)
        self.index_size = {0: '1_2_2', 1: '2_2_2', 2: '3_2_2',
                           3: '4_2_2', 4: '5_2_2', 5: '6_2_2',
                           6: '7_2_2', 7: '8_2_2'}
        for i, size_name in self.index_size.items():
            # 视觉属性
            visual_ind = p.createVisualShape(
                fileName='D:/ProgramData/Anaconda3/Lib/site-packages/pybullet_data/model/' + size_name + '.obj',
                shapeType=p.GEOM_MESH,
                physicsClientId=self._physics_client_id,
            )
            # 碰撞属性
            collision_ind = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName='D:/ProgramData/Anaconda3/Lib/site-packages/pybullet_data/model/' + size_name + '.obj',
                physicsClientId=self._physics_client_id
            )
            self.index_size[i] = (visual_ind, collision_ind)

        self.plane = p.loadURDF("D:/ProgramData/Anaconda3/Lib/site-packages/pybullet_data/plane.urdf", physicsClientId=self._physics_client_id)

        # camera
        self.width = 256  # 图像宽度
        self.height = 256  # 图像高度

        fov = 50  # 相机视角
        aspect = self.width / self.height  # 宽高比
        near = 0.01  # 最近拍摄距离
        far = 100  # 最远拍摄距离

        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[4, 0.5, 1],  # 相机位置
            cameraTargetPosition=[0, 0.5, 1.5],  # 目标位置，与相机位置之间的向量构成相机朝向
            cameraUpVector=[0, 0, 1],  # 相机顶端朝向
            physicsClientId=0
        )  # 计算视角矩阵
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)  # 计算投影矩阵


    def __apply_action(self, action):

        stone_id = self.index_size[action[1]]
        height = p.getBasePositionAndOrientation(self.stone_container[-1])[:][0][2] \
            if len(self.stone_container) else 0
        if action[0]:
            position = np.array([0, 0, height + 1.])
            orientation = p.getQuaternionFromEuler(
                np.array([0, 0 * np.pi / 2, np.pi / 2])
            )
            self.stone_container.append(self.creat_stone(stone_id, position, orientation))
        else:
            position = np.array([0, action[2]*0.2, height + 1.])
            orientation = p.getQuaternionFromEuler(
                np.array([0, 1 * np.pi / 2, np.pi / 2])
            )
            self.stone_container.append(self.creat_stone(stone_id, position, orientation))
            position = np.array([0, action[2]*-0.2, height + 1.])
            self.stone_container.append(self.creat_stone(stone_id, position, orientation))

        return len(self.stone_container) < 9

    def __get_observation(self):
        state = np.zeros((20, 7), dtype=np.float64)
        if not len(self.stone_container):
            return {'pos': spaces.flatten(self.unflatten_observation_space, state),
                    'image': self.render('rgb_array')}
        else:
            for i, stone in enumerate(self.stone_container):
                if i < 20:
                    state[i][0:3] = np.array(p.getBasePositionAndOrientation(stone, physicsClientId=self._physics_client_id)[:][0])
                    state[i][3:7] = np.array(p.getBasePositionAndOrientation(stone, physicsClientId=self._physics_client_id)[:][1])
            return {'pos': spaces.flatten(self.unflatten_observation_space, state),
                    'image': self.render('rgb_array')}

    def reset(self):
        self.step_num = 0
        self.current_reward_similarity = 0
        self.current_reward_x = 0
        self.current_reward_holes = 0
        for item in self.stone_container:
            p.removeBody(item, physicsClientId=self._physics_client_id)
        self.stone_container.clear()
        self.collision = False
        self.state_x.clear()
        self.state_y.clear()
        #self.state_z.clear()
        return self.__get_observation()

    def step(self, action):
        #state[i][0:3] = np.array(p.getBaseVelocity(stone, physicsClientId=self._physics_client_id)[:][0])
        continue_place = self.__apply_action(action)

        for t in range(6000):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        state = self.__get_observation()

        self.step_num += 1

        reward_contact = 0
        if continue_place:
            stone_placed = self.stone_container[-1]
            pmin, pmax = p.getAABB(stone_placed, physicsClientId=self._physics_client_id)
            contact_list = p.getOverlappingObjects(pmin, pmax, physicsClientId=self._physics_client_id)

            if len(contact_list) == 1:  # 悬空，接触仅自身0
                reward_contact = -1
            else:
                for item in contact_list:
                    if item[0] == 0:    # 接触地板
                        if len(self.stone_container) > 1:
                            reward_contact = -1
                            self.collision = True   # 接触地板，若非首块石头扣分
                        elif len(self.stone_container) == 1 and len(contact_list) == 2:
                            reward_contact = 1  # 接触地板，若首块石头加分
                        else:
                            print('error-1')
                        break
                    elif item[0] in self.stone_container and item[0] is not stone_placed:   # 接触其他石块，递减加分
                        reward_contact += 0.1 * (7-len(self.stone_container))
                    elif item[0] is stone_placed:
                        pass
                    else:
                        print('error-2')

        pmin, pmax = p.getAABB(self.plane, physicsClientId=self._physics_client_id)
        contact_list = p.getOverlappingObjects(pmin, pmax, physicsClientId=self._physics_client_id)
        done = None
        reward_similarity = 0
        reward_holes = 0
        reward_x = 0
        reward_y = 0
        reward_z = 0
        tem_reward_x = entropy(self.state_x)
        reward_x = tem_reward_x - self.current_reward_x
        self.current_reward_x = tem_reward_x
        tem_reward_similarity = self.loss_calculator.reconstruct_loss(state['image'], self.image_real)
        reward_similarity = tem_reward_similarity - self.current_reward_similarity
        self.current_reward_similarity = tem_reward_similarity

        gray = state['image'][:, :, 0]
        binary_img, points = seed_filling(gray)
        reward_holes = points

        if self.collision is True or len(self.stone_container) >= 20 or len(contact_list) > 2 or not continue_place:
            reward_similarity = 0
            reward_holes = 0
            reward_x = 0
            reward_y = 0
            reward_z = 0
            reward_contact = -1
            done = True
            if not continue_place:
                self.__apply_action([1, np.random.randint(1)+1, 0])
                for t in range(6000):
                    p.stepSimulation(physicsClientId=self._physics_client_id)
                if points > 1:
                    # print(points)
                    plt.imsave(f'Logger/image/{np.random.randint(1000):3d}.jpg', self.render('rgb_array'))
        else:
            reward_contact = reward_contact * 10.# + reward_holes * 200. #+ reward_similarity * 20. + reward_x * 200.
            reward_holes = points
            if points > 1:
                # print(points)
                plt.imsave(f'Logger/image/{np.random.randint(1000):3d}.jpg', self.render('rgb_array'))

        #reward_topo = self.loss_calculator.topo_loss(state['image'], self.image_real)
        # reward
        reward_placed = reward_contact
        self.alpha = 1# if self.step_num > 50000 else 0
        reward = reward_placed# + reward_holes*20. + self.alpha*reward_similarity*5.  # + reward_topo

        info = {'similarity': self.current_reward_similarity, 'reward_placed': reward_placed,
                'reward_y': self.current_reward_x, 'reward_holes': reward_holes}
        return state, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        if mode == 'rgb_array':
            w, h, rgb, depth, seg = p.getCameraImage(self.width, self.height, self.viewMatrix, self.projection_matrix,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)
            #plt.imsave('a.jpg', Image.fromarray(seg))
            #a = Image.open('a.jpg')
            #return np.array(a)
            arr = np.where(seg <= 0, 255, 0).astype(np.uint8)
            plt.imsave('Logger/a.jpg', np.stack([arr, arr, arr], axis=2))

            return np.stack([arr, arr, arr], axis=2)
        else:
            pass

    def close(self):
        if self._physics_client_id >= 0:
            # 断开连接
            p.disconnect()
        self._physics_client_id = -1

    def creat_stone(self, stone_id, position, orientation):
        sphereUid = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=stone_id[0],
            baseVisualShapeIndex=stone_id[1],
            basePosition=position,
            baseOrientation=orientation,
            useMaximalCoordinates=True,
            physicsClientId=self._physics_client_id
        )
        return sphereUid

if __name__ == "__main__":

    from stable_baselines3.common.env_checker import check_env

    env = InukshukEnv_v3(True)
    check_env(env)
