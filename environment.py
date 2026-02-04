import gymnasium as gym
import numpy as np
import pygame

class TSPEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, n_nodes, render_mode=None, screen_size=600, seed=None):
        super().__init__()
        self.n_nodes = n_nodes

        self.screen_size = screen_size
        self.seed(seed)

        # Define action and observation space
        # Note: different from MLP state
        self.action_space = gym.spaces.Discrete(n_nodes)
        self.observation_space = gym.spaces.Dict({
            "nodes": gym.spaces.Box(low=0, high=1, shape=(2, n_nodes), dtype=np.float32),
            "visited": gym.spaces.Sequence(gym.spaces.Discrete(n_nodes))
        })
        
        self.nodes: np.ndarray #= np.random.rand(2, n_nodes)
        self.visited: list[int]


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.nodes = np.random.rand(2, self.n_nodes)
        self.visited = []
        return {"nodes": self.nodes, "visited": self.visited}

    def step(self, action: int):
        self.visited.append(action)
        if len(self.visited) == self.n_nodes:
            done = True
        else:
            done = False
        
        if done:
            reward = 0
            for i in range(self.n_nodes - 1):
                reward -= np.linalg.norm(self.nodes[:, self.visited[i]] - self.nodes[:, self.visited[i + 1]])
            reward -= np.linalg.norm(self.nodes[:, self.visited[-1]] - self.nodes[:, self.visited[0]])
        else:
            reward = 0
        info = {}
        next_state = {"nodes": self.nodes, "visited": self.visited}
        return next_state, reward, done, info


if __name__ == "__main__":
    n_nodes = 10
    env = TSPEnv(n_nodes)
    env.reset()
    done = False
    import time
    for i in range(n_nodes):
        action = i
        next_state, reward, done, info = env.step(action)
        time.sleep(0.5)
    print(reward)
    env.close()