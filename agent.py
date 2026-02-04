import torch
import torch.nn as nn
import numpy as np

class MLPAgent(nn.Module):

    def __init__(self, n_nodes: int, hidden_dim: int = 64):

        super().__init__()
        self.model = torch.nn.Sequential(
            # Input layer: n_nodes * 2 + 4
            # Hidden layers: linear + Tanh
            # Output layer: n_nodes
            torch.nn.Linear(n_nodes * 2 + 4, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, n_nodes),
        )
        self.softmax = torch.nn.Softmax(dim=-1) 

    def forward(self, state: np.ndarray) -> torch.Tensor:
        O_matrix = torch.from_numpy(state["nodes"]).float()
        l_t = state["visited"]
        if len(l_t) > 0:
            first_node = O_matrix[:, l_t[0]]
            current_node = O_matrix[:, l_t[-1]]
        else:
            first_node = -torch.ones(2)
            current_node = -torch.ones(2)

        o_t = torch.cat([O_matrix.T.flatten(), first_node, current_node]).unsqueeze(0)            
        y_t = self.model(o_t)

        mu_t = torch.zeros_like(y_t)
        mu_t[:, l_t] = -1e6
        pi_t = self.softmax(y_t + mu_t)
        return pi_t
    

if __name__ == "__main__":
    from environment import TSPEnv
    import time
    n_nodes = 6
    agent = MLPAgent(n_nodes, hidden_dim=256)
    env = TSPEnv(n_nodes)

    checkpoint = "policy.pt"
    if checkpoint is not None:
        agent.load_state_dict(torch.load(checkpoint))

    done = False
    state = env.reset()
    while not done:
        probs: torch.Tensor = agent(state)
        probs_str = ", ".join([f"{x:.2f}" for x in probs.view(-1).tolist()]) # just for printing purposes
        print(f"The probabilities for this action are: [{probs_str}]")
        action = torch.multinomial(probs, 1).item()
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(3)
    print(f"The order of visit for nodes is {env.visited}")
    print(f"The final reward is {reward}, meaning that the total length of the tour is {-reward}")