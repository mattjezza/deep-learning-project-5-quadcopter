import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

"""Take off task """
class Takeoff(Task):
    def get_reward(self):
        # The model seems to work better with a small penalty at every action
        action_penalty = 1
        # Penalise heights below 10m, and reward any height above this (higher gives more reward)
        vertical_reward_or_penalty = 10 - self.sim.pose[2]
        # Reward vertical velocity
        if (self.sim.v[2] > 0) and (self.sim.v[2] < 10):
            vertical_velocity_reward = 5
        else:
            vertical_velocity_reward = 0
        reward = vertical_velocity_reward - action_penalty - vertical_reward_or_penalty 
        return reward

"""Flight Time task """
class FlightTime(Task):
    def get_reward(self):
        # Reward for remaining close to the target boundaries (defined in terms of the environment boundaries)
        # Penalty for being too far away.
        if self.sim.pose[0] >= self.target_pos[0]:
            x_position_reward = (self.sim.upper_bounds[0]/2) + self.target_pos[0] - self.sim.pose[0]
            # Normalise
            x_position_reward = 0.2 * x_position_reward / abs((self.sim.upper_bounds[0]/2))
        elif self.sim.pose[0] < self.target_pos[0]:
            x_position_reward = self.sim.pose[0] - ((self.sim.lower_bounds[0]/2) + self.target_pos[0])
            # Normalise
            x_position_reward = 0.2 * x_position_reward / abs((self.sim.lower_bounds[0]/2))
        
        if self.sim.pose[1] >= self.target_pos[1]:
            y_position_reward = (self.sim.upper_bounds[1]/2) + self.target_pos[1] - self.sim.pose[1]
            # Normalise
            y_position_reward = 0.2 * y_position_reward / abs((self.sim.upper_bounds[1]/2))
        elif self.sim.pose[1] < self.target_pos[1]:
            y_position_reward = self.sim.pose[1] - ((self.sim.lower_bounds[1]/2) + self.target_pos[1])
            # Normalise
            y_position_reward = 0.2 * y_position_reward / abs((self.sim.lower_bounds[1]/2))
        
        # z dimension boundaries are always positive (z=0 is the ground) so handled slightly differently
        if self.sim.pose[2] >= self.target_pos[2]:
            z_position_reward = (self.sim.upper_bounds[2]/4) + self.target_pos[2] - self.sim.pose[2]
            # Normalise
            z_position_reward = 0.6 * z_position_reward / (self.sim.upper_bounds[2]/4)
        elif self.sim.pose[2] < self.target_pos[2]:
            z_position_reward = self.sim.pose[2] - (self.target_pos[2]/2)
            # Normalise
            z_position_reward = 0.6 * z_position_reward / (self.target_pos[2]/2)

        if self.sim.time >= (self.sim.runtime - 1):
            target_time_reward = 5
        else:
            target_time_reward = 0
            
        if ((self.sim.pose[2] <= 0) or 
            (self.sim.pose[1] >= self.sim.upper_bounds[1]) or 
            (self.sim.pose[1] <= self.sim.lower_bounds[1]) or 
            (self.sim.pose[0] >= self.sim.upper_bounds[0]) or 
            (self.sim.pose[0] <= self.sim.lower_bounds[0])):
            crash_penalty = (-0.2 * self.sim.runtime) / self.sim.time
        else:
            crash_penalty = 0
            
        reward = (
                 x_position_reward +
                 y_position_reward +
                 z_position_reward +
                 target_time_reward +
                 crash_penalty)

        return reward