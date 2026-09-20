"""
强化学习算法模块
基于 Q-Learning（表格型）实现网格世界的交互式强化学习演示。

提供四种经典环境：
  - grid_world    空房间网格，目标最短路径
  - cliff_walking 悬崖漫步（Sutton & Barto 经典示例）
  - maze          带墙体的小迷宫
  - windy_grid    有风网格世界，按所在列施加向上风力

纯 NumPy 实现，不依赖额外第三方库。
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# 动作编号：上、右、下、左
ACTIONS = {
    'up': 0,
    'right': 1,
    'down': 2,
    'left': 3,
}
ACTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 与 ACTIONS 顺序一致
ACTION_NAMES = ['上', '右', '下', '左']


class GridWorld:
    """通用网格世界环境。

    格子类型：
      normal  普通可通行格
      wall    墙体（不可进入）
      start   起点
      goal    终点（回合结束）
      cliff   悬崖（掉落后回到起点，给予高额负奖励）
    """

    def __init__(self, rows, cols, cell_types, start, goal,
                 step_reward=-1.0, goal_reward=10.0, cliff_reward=-100.0,
                 wind=None):
        self.rows = rows
        self.cols = cols
        # cell_types[r][c] ∈ {'normal','wall','start','goal','cliff'}
        self.cell_types = cell_types
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.cliff_reward = cliff_reward
        self.wind = wind or [0] * cols
        self.max_steps = 200

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, r, c):
        return self.cell_types[r][c] == 'wall'

    def reset(self):
        return self.start

    def step(self, state, action):
        """执行动作，返回 (next_state, reward, done)。"""
        r, c = state
        dr, dc = ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc

        # 越界或撞墙：留在原地
        if not self.in_bounds(nr, nc) or self.is_wall(nr, nc):
            return (r, c), self.step_reward, False

        # 经典有风网格：按当前所在列的风力，在动作后向上移动。
        nr = max(0, nr - self.wind[c])

        cell = self.cell_types[nr][nc]

        if cell == 'goal':
            return (nr, nc), self.goal_reward, True

        if cell == 'cliff':
            # 掉下悬崖：回到起点，回合不结束（让智能体继续学习）
            return self.start, self.cliff_reward, False

        return (nr, nc), self.step_reward, False


def _make_grid(rows, cols, fill='normal'):
    return [[fill for _ in range(cols)] for _ in range(rows)]


def build_environment(env_name):
    """根据名称构建环境。"""
    if env_name == 'grid_world':
        rows, cols = 8, 8
        cells = _make_grid(rows, cols)
        start, goal = (0, 0), (rows - 1, cols - 1)
        cells[start[0]][start[1]] = 'start'
        cells[goal[0]][goal[1]] = 'goal'
        return GridWorld(rows, cols, cells, start, goal,
                         step_reward=-1.0, goal_reward=10.0, cliff_reward=-100.0)

    if env_name == 'cliff_walking':
        # Sutton & Barto Cliff Walking：4 行 12 列
        rows, cols = 4, 12
        cells = _make_grid(rows, cols)
        start, goal = (3, 0), (3, 11)
        # 第 3 行中间为悬崖
        for c in range(1, 11):
            cells[3][c] = 'cliff'
        cells[start[0]][start[1]] = 'start'
        cells[goal[0]][goal[1]] = 'goal'
        return GridWorld(rows, cols, cells, start, goal,
                         step_reward=-1.0, goal_reward=0.0, cliff_reward=-100.0)

    if env_name == 'maze':
        # 7x7 小迷宫，墙体构成障碍
        rows, cols = 7, 7
        cells = _make_grid(rows, cols)
        start, goal = (0, 0), (6, 6)
        walls = [
            (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3),
            (5, 5), (4, 5), (3, 5), (2, 5),
        ]
        for (r, c) in walls:
            cells[r][c] = 'wall'
        cells[start[0]][start[1]] = 'start'
        cells[goal[0]][goal[1]] = 'goal'
        return GridWorld(rows, cols, cells, start, goal,
                         step_reward=-1.0, goal_reward=10.0, cliff_reward=-100.0)

    if env_name == 'windy_grid':
        rows, cols = 7, 10
        cells = _make_grid(rows, cols)
        start, goal = (3, 0), (3, 7)
        cells[start[0]][start[1]] = 'start'
        cells[goal[0]][goal[1]] = 'goal'
        return GridWorld(rows, cols, cells, start, goal,
                         step_reward=-1.0, goal_reward=0.0,
                         wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

    raise ValueError(f"未知环境: {env_name}")


class ReinforcementLearningAlgorithm:
    """Q-Learning 智能体（表格型）。"""

    def __init__(self):
        self.env = None

    # ---------- 环境序列化 ----------
    def serialize_grid(self, env):
        cells = []
        for r in range(env.rows):
            for c in range(env.cols):
                cells.append({
                    'row': r, 'col': c, 'type': env.cell_types[r][c]
                })
        return {
            'rows': env.rows,
            'cols': env.cols,
            'start': list(env.start),
            'goal': list(env.goal),
            'wind': env.wind,
            'cells': cells,
        }

    # ---------- 训练 ----------
    def train(self, env_name, episodes=500, alpha=0.1, gamma=0.95,
              epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
              random_state=42, checkpoint_every=50):
        env = build_environment(env_name)
        self.env = env
        epsilon_start = epsilon

        rng = np.random.RandomState(random_state)
        n_states = env.rows * env.cols
        n_actions = 4
        Q = np.zeros((n_states, n_actions), dtype=np.float64)
        action_counts = np.zeros((n_states, n_actions), dtype=np.int32)

        def state_to_idx(s):
            return s[0] * env.cols + s[1]

        rewards_history = []
        success_flags = []
        steps_history = []
        checkpoints = []

        for ep in range(1, int(episodes) + 1):
            state = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            reached_goal = False

            while not done and ep_steps < env.max_steps:
                s_idx = state_to_idx(state)
                # epsilon-greedy 动作选择
                if rng.rand() < epsilon:
                    action = rng.randint(n_actions)
                else:
                    action = int(np.argmax(Q[s_idx]))

                next_state, reward, done = env.step(state, action)
                ns_idx = state_to_idx(next_state)

                # Q-Learning 更新
                best_next = 0.0 if done else float(np.max(Q[ns_idx]))
                Q[s_idx, action] += alpha * (reward + gamma * best_next - Q[s_idx, action])
                action_counts[s_idx, action] += 1

                state = next_state
                ep_reward += reward
                ep_steps += 1
                if done:
                    reached_goal = True

            rewards_history.append(float(ep_reward))
            success_flags.append(bool(reached_goal))
            steps_history.append(int(ep_steps))

            # epsilon 衰减
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # 记录检查点：跑一轮示范轨迹，展示"试错"过程
            if ep == 1 or ep % checkpoint_every == 0 or ep == int(episodes):
                demo = self._demo_episode(env, Q, epsilon, rng, max_steps=80)
                recent_count = min(50, len(rewards_history))
                checkpoints.append({
                    'episode': ep,
                    'heatmap': self._heatmap(env, Q, action_counts),
                    'policy': self._greedy_policy(env, Q, action_counts),
                    'greedy_path': self._rollout_path(env, Q),
                    'avg_reward': float(np.mean(rewards_history[-recent_count:])),
                    'success_rate': float(np.mean(success_flags[-recent_count:])),
                    'epsilon': float(epsilon),
                    'demo_path': demo['path'],
                    'demo_reward': demo['reward'],
                    'demo_steps': demo['steps'],
                    'demo_fell': demo['fell'],
                })

        # 训练完成后的策略与最优路径
        policy = self._greedy_policy(env, Q, action_counts)
        path = self._rollout_path(env, Q)

        # 统计最近一段的成功率
        recent = min(100, len(success_flags))
        recent_success = float(np.mean(success_flags[-recent:])) if recent else 0.0
        recent_avg_steps = float(np.mean(steps_history[-recent:])) if recent else 0.0

        result = {
            'grid': self.serialize_grid(env),
            'q_table': Q.tolist(),
            'policy': policy,                 # [rows][cols] -> 动作编号或 -1
            'heatmap': self._heatmap(env, Q, action_counts),  # 已尝试动作的最高 Q，未探索为 None
            'path': path,                    # 贪心路径 [[r,c],...]
            'rewards': rewards_history,
            'checkpoints': checkpoints,
            'stats': {
                'episodes': int(episodes),
                'final_epsilon': float(epsilon),
                'recent_success_rate': recent_success,
                'recent_avg_steps': recent_avg_steps,
                'final_reward': rewards_history[-1] if rewards_history else 0.0,
                'path_reached_goal': bool(path and path[-1] == list(env.goal)),
            },
            'params': {
                'alpha': alpha, 'gamma': gamma,
                'epsilon_start': epsilon_start, 'epsilon_decay': epsilon_decay,
            },
        }
        return result

    def _demo_episode(self, env, Q, epsilon, rng, max_steps=80):
        """用当前 Q 表和当前 epsilon 真实跑一轮，记录轨迹（含掉悬崖）。"""
        def s2i(s): return s[0] * env.cols + s[1]
        state = env.reset()
        path = [[state[0], state[1]]]
        reward_total = 0.0
        fell = 0
        steps = 0
        done = False
        while not done and steps < max_steps:
            s_idx = s2i(state)
            if rng.rand() < epsilon:
                action = int(rng.randint(4))
            else:
                action = int(np.argmax(Q[s_idx]))
            # 尝试移动到的格子
            ar, ac = [(-1,0),(0,1),(1,0),(0,-1)][action]
            nr, nc = state[0]+ar, state[1]+ac
            next_state, reward, done = env.step(state, action)
            reward_total += reward
            steps += 1
            if reward == env.cliff_reward and env.in_bounds(nr,nc):
                # 掉崖：先记录悬崖格，再记录被弹回的起点，让前端看到"掉下去又回来"
                path.append([nr, nc])
                fell += 1
            path.append([next_state[0], next_state[1]])
            state = next_state
        return {
            'path': path, 'reward': float(reward_total),
            'steps': int(steps), 'fell': int(fell)
        }

    def _heatmap(self, env, Q, action_counts):
        """每格已尝试动作的最高 Q 值；未探索和墙体为空。"""
        grid = []
        for r in range(env.rows):
            row = []
            for c in range(env.cols):
                s_idx = r * env.cols + c
                visited_actions = action_counts[s_idx] > 0
                if env.cell_types[r][c] == 'wall' or not np.any(visited_actions):
                    row.append(None)
                else:
                    row.append(float(np.max(Q[s_idx, visited_actions])))
            grid.append(row)
        return grid

    def _greedy_policy(self, env, Q, action_counts):
        """每格当前贪心动作；尚未尝试过的动作不画箭头。"""
        policy = []
        for r in range(env.rows):
            row = []
            for c in range(env.cols):
                if env.cell_types[r][c] in ('wall', 'goal', 'cliff'):
                    row.append(-1)
                else:
                    s_idx = r * env.cols + c
                    action = int(np.argmax(Q[s_idx]))
                    row.append(action if action_counts[s_idx, action] > 0 else -1)
            policy.append(row)
        return policy

    def _rollout_path(self, env, Q):
        """按贪心策略从起点走到终点，返回路径格子列表。"""
        path = []
        state = env.reset()
        visited = set()
        for _ in range(env.max_steps):
            path.append([state[0], state[1]])
            if state == env.goal:
                break
            if state in visited:
                break  # 出现循环，停止
            visited.add(state)
            s_idx = state[0] * env.cols + state[1]
            action = int(np.argmax(Q[s_idx]))
            next_state, _, done = env.step(state, action)
            if done:
                path.append([next_state[0], next_state[1]])
                break
            state = next_state
        return path

    def available_environments(self):
        return {
            'grid_world': {
                'name': '空房间网格',
                'description': '8×8 开放网格，从左上角走到右下角，学习最短路径。',
            },
            'cliff_walking': {
                'name': '悬崖漫步',
                'description': '经典 Cliff Walking：沿悬崖边行走，平衡探索与安全。',
            },
            'maze': {
                'name': '小迷宫',
                'description': '带墙体的 7×7 迷宫，观察智能体如何绕开障碍。',
            },
            'windy_grid': {
                'name': '有风网格世界',
                'description': '经典 Windy Gridworld：部分列持续向上吹风，智能体需要修正路线。',
                'recommended': {'episodes': 500, 'alpha': 0.3},
            },
        }
