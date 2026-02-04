import pygame
from agent import MLPAgent
from environment import TSPEnv
import torch

from baseline_policies import greedy_policy, random_policy
from tqdm import tqdm

class REINFORCE:

    def __init__(self, n_nodes):

        self.env = TSPEnv(n_nodes)
        self.agent = MLPAgent(n_nodes, hidden_dim=256)
        self.buffer = []
        self.scores = []

        # Other hyperparameters
        self.training_epochs = 7500
        self.episodes_per_epoch = 2048
        self.gamma = 1.0
        learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)


    def collect_one_episode(self):

        state = self.env.reset()  # `reset` generates the starting state (a new instance for the TSP)
        done = False
        log_probs = []
        rewards = []
        while not done:
            action_probs = self.agent(state)
            action = torch.multinomial(action_probs, 1).item()
            state, reward, done, _ = self.env.step(action)  # `step` updates the state according to the action selected
            rewards.append(reward)
            log_probs.append(torch.log(action_probs[:, action]))
        self.buffer.append((rewards, log_probs, state))

    def update(self):

        loss = 0
        # Calculate the loss over all episodes collected in the buffer
        for rewards, log_probs, state in self.buffer:  # iterate over all stored trajectories
            # compute reward for NN algorithm
            greedy_policy_length, _ = greedy_policy(state["nodes"])

            returns = []
            G = 0
            for r in rewards[::-1]:
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)

            log_probs = torch.stack(log_probs)
            # compare policy results with NN algorithm
            score = G + greedy_policy_length
            loss -= torch.sum(log_probs) * score
            self.scores.append(score.item())
            
        self.optimizer.zero_grad()
        (loss / len(self.buffer)).backward()  # performs backpropagation, i.e. computes the gradients
        self.optimizer.step()
        self.buffer = []  # reset the buffer
        

    def train(self):

        with tqdm(total=self.training_epochs, position=0, desc="Epoch") as pbar:
            for epoch_num in range(self.training_epochs):
                for _ in tqdm(range(self.episodes_per_epoch), desc="Episode", position=1, leave=False):
                    self.collect_one_episode()
                self.update()
                pbar.update(1)
                # save the scores in a file for plotting them
                with open("scores.txt", "w") as file:
                    file.write("\n".join(str(x) for x in self.scores))
                last_score = sum(self.scores[-self.episodes_per_epoch:])
                mean_score = last_score / self.episodes_per_epoch
                tqdm.write(f"Epoch {epoch_num + 1}, score: {mean_score:.2f}")
                if epoch_num % 100 == 0 or epoch_num == self.training_epochs - 1:
                    self.test()
                    pygame.image.save(self.env.screen, f"epoch_{epoch_num}.jpeg")
                    self.env.close()

                    # save a checkpoint for the agent DNN
                    torch.save(self.agent.state_dict(), "policy.pt")

    def test(self):
        state = self.env.reset()
        done = False
        
        while not done:
            action_probs = self.agent(state)
            action = torch.multinomial(action_probs, 1).item()
            state, _, done, _ = self.env.step(action)
            self.env.render()
        


if __name__ == "__main__":
    reinforce = REINFORCE(n_nodes=6)
    reinforce.train()

    from matplotlib import pyplot as plt
    plt.plot(reinforce.scores)
    plt.savefig("reinforce.png")
    plt.show()