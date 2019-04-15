import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 ######################################
                 # Michael Changed
                 velocity_weight={'x': 1, 'y': 0},
                 goal_distance=5,

                 control_weight=1,
                 contact_weight=1,
                 task_weight=1,
                 healthy_weight=1,

                 control_scale=1,
                 contact_scale=1,
                 task_scale=1,
                 ######################################
                 ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        ######################################
        # Michael Changed
        self.velocity_weight = velocity_weight
        self.goal_distance = goal_distance
        # import os
        # model_path = 'ant.xml'
        # here modify the xml_file
        # fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        # print(fullpath)
        # xml_str = ''.join(open(fullpath, 'r').readlines())  # this is how you get the xml string

        weight_total = float(control_weight + contact_weight + task_weight + healthy_weight)

        self.control_scale = control_scale
        self.control_weight = control_weight/weight_total
        self.contact_scale = contact_scale
        self.contact_weight = contact_weight/weight_total
        self.task_scale = task_scale
        self.task_weight = task_weight/weight_total
        self.healthy_weight = healthy_weight/weight_total

        ######################################

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    ######################################
    # Michael Changed
    # @property
    def get_forward_reward(self, x_velocity, y_velocity):
        weighted_velocity = self.velocity_weight['x']*x_velocity + self.velocity_weight['y']*y_velocity
        return weighted_velocity
    ######################################
    
    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        ######################################
        x_position, y_position = xy_position_after
        ######################################

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        ######################################
        # forward_reward = x_velocity
        forward_reward = self.get_forward_reward(x_velocity=x_velocity, y_velocity=y_velocity)
        self.max_vel = 10
        task_reward = self.max_vel - forward_reward
        ######################################
        healthy_reward = self.healthy_reward

        ######################################
        # rewards = forward_reward + healthy_reward
        # costs = ctrl_cost + contact_cost
        # reward = rewards - costs

        log_task_reward_component = self.task_scale * max(task_reward, 0)  # TODO square this
        log_control_cost_component = self.control_scale * ctrl_cost
        log_contact_cost_component = self.contact_scale * contact_cost
        
        task_reward_component = np.exp(-log_task_reward_component)
        healthy_reward_component = healthy_reward  # already between 0 and 1
        control_cost_component = np.exp(-log_control_cost_component)
        contact_cost_component = np.exp(-log_contact_cost_component)

        reward = self.task_weight*task_reward_component + \
                 self.healthy_weight*healthy_reward_component + \
                 self.control_weight*control_cost_component + \
                 self.contact_weight*contact_cost_component

        ######################################



        done = self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,

            'log_task_reward_component': log_task_reward_component,
            'log_control_cost_component': log_control_cost_component,
            'log_contact_cost_component': log_contact_cost_component,

            'task_reward_component': task_reward_component,
            'healthy_reward_component': healthy_reward_component,
            'control_cost_component': control_cost_component,
            'contact_cost_component': contact_cost_component,
        }
        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
