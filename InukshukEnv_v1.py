import gym
from gym import spaces
import numpy as np

import pybullet as p
import pybullet_data


class InukshukEnv_v1(gym.Env):
    def __init__(self, render: bool = False):

        self._render = render
        # 定义动作空间
        self.action_space = spaces.MultiDiscrete(np.array([19, 10, 2, 2, 2]))
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
                                              'image': spaces.Box(0, 255, (224, 224, 3), np.uint8)})

        # 连接引擎
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)

        # 石块存储器
        self.stone_container = []
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
        p.setGravity(0, 0, -0.8)
        self.index_size = { 0: '10_1_1', 1: '8_1_1', 2: '5_1_1', 3: '3_1_1', 4: '2_1_1', 5: '1_1_1', 6: '10_2_1',
                            7: '8_2_1',
                            8: '5_2_1', 9: '3_2_1',
                            10: '2_2_1', 11: '1_2_1', 12: '10_2_2', 13: '8_2_2', 14: '5_2_2', 15: '3_2_2', 16: '2_2_2',
                            17: '1_2_2'}
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
        self.width = 224  # 图像宽度
        self.height = 224  # 图像高度

        fov = 50  # 相机视角
        aspect = self.width / self.height  # 宽高比
        near = 0.01  # 最近拍摄距离
        far = 100  # 最远拍摄距离

        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[3, 0.5, 1],  # 相机位置
            cameraTargetPosition=[0, 0.5, 0.2],  # 目标位置，与相机位置之间的向量构成相机朝向
            cameraUpVector=[0, 0, 1],  # 相机顶端朝向
            physicsClientId=0
        )  # 计算视角矩阵
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)  # 计算投影矩阵


    def __apply_action(self, action):

        index = action[0]
        if index:
            stone_id = self.index_size[index-1]
            height = p.getBasePositionAndOrientation(self.stone_container[-1])[:][0][2] \
                if len(self.stone_container) else 0
            position = np.array([0, action[1] * 0.1, height + 0.2])
            orientation = p.getQuaternionFromEuler(
                np.array([action[2] * np.pi / 2, action[3] * np.pi / 2, action[4] * np.pi / 2])
            )
            self.stone_container.append(self.creat_stone(stone_id, position, orientation))
        else:
            pass
        return index

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

        for item in self.stone_container:
            p.removeBody(item, physicsClientId=self._physics_client_id)
        self.stone_container.clear()
        self.collision = False
        return self.__get_observation()

    def step(self, action):
        #state[i][0:3] = np.array(p.getBaseVelocity(stone, physicsClientId=self._physics_client_id)[:][0])
        placed = self.__apply_action(action)

        for t in range(1000):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        state = self.__get_observation()

        self.step_num += 1

        reward_placed = 0
        reward_contact = 0
        if placed:
            reward_placed = 1
            stone_placed = self.stone_container[-1]
            pmin, pmax = p.getAABB(stone_placed, physicsClientId=self._physics_client_id)
            contact_list = p.getOverlappingObjects(pmin, pmax, physicsClientId=self._physics_client_id)
            if contact_list is None:
                reward_contact = -10
            else:
                for item in contact_list:
                    if item[0] == 0:
                        if len(self.stone_container) > 1:
                            reward_contact = -10
                            self.collision = True
                            break
                        else:
                            reward_contact = 10
                    elif item[0] in self.stone_container and item[0] is not stone_placed:
                        reward_contact += 10

        # reward
        reward = reward_placed + reward_contact

        done = None
        if self.collision is True or len(self.stone_container) >= 20:
            done = True
        else:
            done = False
        info = {}
        return state, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        if mode == 'rgb_array':
            w, h, rgb, depth, seg = p.getCameraImage(self.width, self.height, self.viewMatrix, self.projection_matrix,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)
            return np.array(rgb)[:, :, :3]
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

    env = InukshukEnv_v1(True)
    check_env(env)
