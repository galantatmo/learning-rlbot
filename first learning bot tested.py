import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rlgym.api import RLGym, RewardFunction
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league.reward_functions import CombinedReward, TouchReward
from RocketSim import GameMode

# =========================
# Pfade & Checkpoints
# =========================
MODEL_PATH = "policy_net.pt"
SAVE_INTERVAL = 10
os.makedirs("checkpoints", exist_ok=True)

# Option 1: Alten Checkpoint löschen, wenn Netzwerkgröße geändert
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

# =========================
# 🎯 Belohnungsfunktion
# =========================
class SimpleSelfPlayReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.last_ball_pos = np.array(initial_state.ball.position)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_pos = np.array(state.ball.position)

        for agent in agents:
            car = state.cars[agent]
            reward = 0.0

            # Geschwindigkeit
            speed = np.linalg.norm(car.physics.linear_velocity)
            reward += speed * 0.0001

            # Boost
            if car.boost_amount < 100:
                reward += 0.01

            # Sprung
            if not car.on_ground:
                reward += 0.005

            # Ballkontakt in Richtung Tor (Team 0 = Spieler)
            ball_movement = ball_pos[0] - self.last_ball_pos[0]
            if car.team_num == 0:
                reward += 0.5 * ball_movement  # größere Belohnung

            # Ballkontakt generell
            if hasattr(car, "had_ball_contact") and car.had_ball_contact:
                reward += 1.0  # große Belohnung für Ballkontakt

            # Tor
            if is_terminated[agent]:
                goal_x = ball_pos[0]
                if car.team_num == 0:
                    reward += 100.0 if goal_x > 0 else -100.0
                else:
                    reward -= 100.0 if goal_x < 0 else 100.0

            rewards[agent] = reward

        self.last_ball_pos = ball_pos.copy()
        return rewards

# =========================
# 🧠 Policy Network
# =========================
class PolicyNet(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# 🤖 Agent
# =========================
class Agent:
    def __init__(self, obs_size, action_size, trainable=True):
        self.model = PolicyNet(obs_size, action_size)
        self.trainable = trainable
        if trainable:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.log_probs = []
        self.rewards = []
        if trainable and os.path.exists(MODEL_PATH):
            print("Lade gespeichertes Modell...")
            self.model.load_state_dict(torch.load(MODEL_PATH))

    def act(self, obs, exploration=0.5):
        obs = torch.tensor(np.array(obs).flatten(), dtype=torch.float32)
        probs = self.model(obs)
        # Exploration erhöhen
        probs = probs * (1 - exploration) + exploration / len(probs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        if self.trainable:
            self.log_probs.append(dist.log_prob(action))
        return np.array([action.item()])

    def learn(self):
        if not self.trainable:
            return
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        loss = []
        for log_prob, R in zip(self.log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs = []
        self.rewards = []

    def save(self):
        torch.save(self.model.state_dict(), MODEL_PATH)
        print("💾 Modell gespeichert!")

# =========================
# 🌍 Environment
# =========================
env = RLGym(
    state_mutator=MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    ),
    obs_builder=DefaultObs(),
    action_parser=RepeatAction(LookupTableAction(), repeats=8),
    reward_fn=CombinedReward(
        (SimpleSelfPlayReward(), 1.0),
        (TouchReward(), 2.0)
    ),
    termination_cond=GoalCondition(),
    truncation_cond=AnyCondition(
        TimeoutCondition(timeout_seconds=60.),
        NoTouchTimeoutCondition(timeout_seconds=15.0)
    ),
    transition_engine=RocketSimEngine(game_mode=GameMode.SOCCAR)
)

# =========================
# 🚀 Training Loop
# =========================
obs = env.reset()
agent_ids = list(env.agents)
obs_size = np.array(obs[agent_ids[0]]).flatten().shape[0]
action_size = env.action_spaces[agent_ids[0]][1]

player = Agent(obs_size, action_size, trainable=True)

EPISODES = 10000
avg = 0.0
wins = 0

for episode in range(1, EPISODES + 1):
    obs = env.reset()
    total_reward = 0
    winner = None

    while True:
        actions = {}
        actions[agent_ids[0]] = player.act(obs[agent_ids[0]], exploration=0.5)
        actions[agent_ids[1]] = np.random.randint(0, action_size, size=1)  # einfacher zufälliger Gegner

        next_obs, rewards, terminated, truncated = env.step(actions)
        total_reward += rewards[agent_ids[0]]
        player.rewards.append(rewards[agent_ids[0]])
        obs = next_obs

        if terminated[agent_ids[0]] or terminated[agent_ids[1]] or truncated[agent_ids[0]]:
            break

    # Gewinner bestimmen
    if terminated[agent_ids[0]]:
        wins += 1
        winner = "Player"
    elif terminated[agent_ids[1]]:
        winner = "Opponent"
    else:
        winner = "No goal"

    # Moving Average Reward
    a = 0.05
    avg = a * total_reward + (1 - a) * avg

    print(f"Ep {episode} | Reward: {total_reward:.2f} | Avg: {avg:.2f} | Winner: {winner} | Winrate: {(wins/episode*100):.1f}%")

    player.learn()

    if episode % SAVE_INTERVAL == 0:
        player.save()
