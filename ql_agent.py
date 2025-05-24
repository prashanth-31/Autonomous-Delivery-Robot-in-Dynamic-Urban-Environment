import random
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional, Set

class QLearningAgent:
    def __init__(self, grid_size: int, obstacles: List[Tuple[int, int]], goal: Tuple[int, int], 
                 alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1, 
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        """
        Enhanced Q-Learning Agent with improved parameters and features.
        
        Args:
            grid_size: Size of the grid environment
            obstacles: List of obstacle positions
            goal: Goal position
            alpha: Learning rate (reduced for stability)
            gamma: Discount factor (increased for better long-term planning)
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            min_epsilon: Minimum epsilon value
        """
        self.grid_size = grid_size
        self.goal = goal
        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        self.obstacles = set(obstacles)
        
        # Enhanced hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.initial_epsilon = epsilon
        
        # Performance tracking
        self.training_stats = {
            'episodes': 0,
            'successful_episodes': 0,
            'total_steps': 0,
            'average_steps': 0,
            'best_path_length': float('inf')
        }
        
        # Initialize Q-table with better structure
        self.q_table = {}
        self._initialize_q_table()
        
    def _initialize_q_table(self):
        """Initialize Q-table for all valid states."""
        self.q_table.clear()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.obstacles:
                    # Initialize with small random values to break ties
                    self.q_table[(x, y)] = {
                        action: random.uniform(-0.01, 0.01) for action in self.actions
                    }

    def update_obstacles(self, new_obstacles: List[Tuple[int, int]]):
        """Update obstacles and reinitialize Q-table accordingly."""
        old_obstacles = self.obstacles.copy()
        self.obstacles = set(new_obstacles)
        
        # Only reinitialize if obstacles have changed significantly
        if len(old_obstacles.symmetric_difference(self.obstacles)) > 0:
            # Remove obstacle positions from Q-table
            for pos in list(self.q_table.keys()):
                if pos in self.obstacles:
                    del self.q_table[pos]
            
            # Add new free positions to Q-table
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if (x, y) not in self.q_table and (x, y) not in self.obstacles:
                        self.q_table[(x, y)] = {
                            action: random.uniform(-0.01, 0.01) for action in self.actions
                        }

    def get_valid_moves(self, state: Tuple[int, int]) -> List[str]:
        """Get list of valid moves from current state."""
        x, y = state
        valid_moves = []
        
        for action in self.actions:
            dx, dy = self.action_map[action]
            new_x, new_y = x + dx, y + dy
            
            # Check bounds and obstacles
            if (0 <= new_x < self.grid_size and 
                0 <= new_y < self.grid_size and 
                (new_x, new_y) not in self.obstacles):
                valid_moves.append(action)
                
        return valid_moves

    def move(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Execute action and return new state."""
        if action not in self.action_map:
            return state
            
        x, y = state
        dx, dy = self.action_map[action]
        new_x, new_y = x + dx, y + dy
        
        # Boundary and obstacle checking
        if (0 <= new_x < self.grid_size and 
            0 <= new_y < self.grid_size and 
            (new_x, new_y) not in self.obstacles):
            return (new_x, new_y)
        else:
            return state  # Stay in place if invalid move

    def get_reward(self, state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
        """Enhanced reward function with multiple criteria."""
        # Goal reached - high positive reward
        if next_state == self.goal:
            return 100.0
        
        # Hit obstacle or boundary - negative reward
        if next_state in self.obstacles or next_state == state:
            return -10.0
        
        # Distance-based reward (encourage moving closer to goal)
        current_distance = abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])
        next_distance = abs(next_state[0] - self.goal[0]) + abs(next_state[1] - self.goal[1])
        distance_reward = (current_distance - next_distance) * 2.0
        
        # Small penalty for each step to encourage efficiency
        step_penalty = -1.0
        
        return distance_reward + step_penalty

    def choose_action(self, state: Tuple[int, int]) -> Optional[str]:
        """Choose action using epsilon-greedy policy with valid moves only."""
        valid_moves = self.get_valid_moves(state)
        if not valid_moves:
            return None
            
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)
        
        # Greedy action selection among valid moves only
        if state not in self.q_table:
            return random.choice(valid_moves)
            
        q_values = self.q_table[state]
        valid_q_values = {action: q_values[action] for action in valid_moves}
        
        if not valid_q_values:
            return random.choice(valid_moves)
            
        max_q = max(valid_q_values.values())
        best_actions = [action for action, q in valid_q_values.items() if q == max_q]
        
        return random.choice(best_actions)

    def train(self, episodes: int = 1000, start_pos: Tuple[int, int] = (0, 0), 
              max_steps_per_episode: int = 500, verbose: bool = False) -> Dict:
        """
        Train the agent with enhanced tracking and early stopping.
        
        Args:
            episodes: Number of training episodes
            start_pos: Starting position for each episode
            max_steps_per_episode: Maximum steps per episode
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training statistics
        """
        successful_episodes = 0
        total_steps = 0
        episode_lengths = []
        
        for episode in range(episodes):
            state = start_pos
            steps = 0
            episode_reward = 0
            
            while state != self.goal and steps < max_steps_per_episode:
                action = self.choose_action(state)
                if action is None:
                    break  # No valid moves
                    
                next_state = self.move(state, action)
                reward = self.get_reward(state, action, next_state)
                episode_reward += reward
                
                # Q-learning update
                if state in self.q_table:
                    old_q = self.q_table[state][action]
                    
                    # Calculate next state max Q-value
                    next_max_q = 0.0
                    if next_state in self.q_table and next_state != self.goal:
                        valid_next_moves = self.get_valid_moves(next_state)
                        if valid_next_moves:
                            next_q_values = [self.q_table[next_state][a] for a in valid_next_moves]
                            next_max_q = max(next_q_values)
                    
                    # Update Q-value
                    new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
                    self.q_table[state][action] = new_q
                
                state = next_state
                steps += 1
                
                # Early termination if goal reached
                if state == self.goal:
                    successful_episodes += 1
                    episode_lengths.append(steps)
                    break
            
            total_steps += steps
            
            # Decay epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                success_rate = successful_episodes / (episode + 1) * 100
                avg_steps = total_steps / (episode + 1)
                print(f"Episode {episode + 1}: Success Rate: {success_rate:.1f}%, "
                      f"Avg Steps: {avg_steps:.1f}, Epsilon: {self.epsilon:.3f}")
        
        # Update training statistics
        self.training_stats.update({
            'episodes': episodes,
            'successful_episodes': successful_episodes,
            'total_steps': total_steps,
            'average_steps': total_steps / episodes if episodes > 0 else 0,
            'success_rate': successful_episodes / episodes * 100 if episodes > 0 else 0,
            'best_path_length': min(episode_lengths) if episode_lengths else float('inf')
        })
        
        return self.training_stats.copy()

    def get_path_from(self, start: Tuple[int, int], max_steps: int = 200) -> List[Tuple[int, int]]:
        """
        Get optimal path from start to goal using learned Q-values.
        
        Args:
            start: Starting position
            max_steps: Maximum steps to prevent infinite loops
            
        Returns:
            List of positions representing the path
        """
        path = []
        state = start
        visited = set()
        steps = 0
        
        while state != self.goal and steps < max_steps:
            path.append(state)
            
            # Prevent infinite loops
            if state in visited:
                break
            visited.add(state)
            
            # Choose best action based on Q-values
            if state not in self.q_table:
                break
                
            valid_moves = self.get_valid_moves(state)
            if not valid_moves:
                break
            
            # Select action with highest Q-value among valid moves
            q_values = self.q_table[state]
            valid_q_values = {action: q_values[action] for action in valid_moves}
            best_action = max(valid_q_values, key=valid_q_values.get)
            
            # Move to next state
            next_state = self.move(state, best_action)
            if next_state == state:  # Didn't actually move
                break
                
            state = next_state
            steps += 1
        
        # Add goal if reached
        if state == self.goal:
            path.append(self.goal)
            
        return path

    def get_policy_at(self, state: Tuple[int, int]) -> Dict[str, float]:
        """Get the policy (action probabilities) at a given state."""
        if state not in self.q_table:
            return {}
        
        valid_moves = self.get_valid_moves(state)
        if not valid_moves:
            return {}
        
        q_values = {action: self.q_table[state][action] for action in valid_moves}
        return q_values

    def save_model(self, filepath: str):
        """Save the trained model to a file."""
        model_data = {
            'q_table': {str(k): v for k, v in self.q_table.items()},
            'grid_size': self.grid_size,
            'goal': self.goal,
            'obstacles': list(self.obstacles),
            'training_stats': self.training_stats,
            'hyperparameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'min_epsilon': self.min_epsilon
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load a trained model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Convert string keys back to tuples
        self.q_table = {eval(k): v for k, v in model_data['q_table'].items()}
        self.grid_size = model_data['grid_size']
        self.goal = tuple(model_data['goal'])
        self.obstacles = set(tuple(obs) for obs in model_data['obstacles'])
        self.training_stats = model_data['training_stats']
        
        # Load hyperparameters
        hyper = model_data['hyperparameters']
        self.alpha = hyper['alpha']
        self.gamma = hyper['gamma']
        self.epsilon = hyper['epsilon']
        self.epsilon_decay = hyper['epsilon_decay']
        self.min_epsilon = hyper['min_epsilon']

    def reset_epsilon(self):
        """Reset epsilon to initial value for retraining."""
        self.epsilon = self.initial_epsilon

    def get_statistics(self) -> Dict:
        """Get training and performance statistics."""
        stats = self.training_stats.copy()
        stats['q_table_size'] = len(self.q_table)
        stats['total_states'] = self.grid_size * self.grid_size - len(self.obstacles)
        stats['coverage'] = len(self.q_table) / stats['total_states'] * 100 if stats['total_states'] > 0 else 0
        return stats