import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import threading
import time
from collections import deque
from typing import List, Tuple, Dict, Optional
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend for embedding in Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Import the real QLearningAgent
from ql_agent import QLearningAgent

# Enhanced Constants
GRID_SIZE = 20
CELL_SIZE = 25
START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
NUM_OBSTACLES = 50
OBSTACLE_MOVE_PROB = 0.3  # Reduced for more realistic movement

class EnhancedDeliveryApp:
    def __init__(self, master):
        self.master = master
        master.title("Enhanced Autonomous Delivery Robot Simulator")
        master.configure(bg='#f0f0f0')
        
        # Main layout
        self.setup_ui()
        
        # Initialize simulation variables
        self.obstacles = set()
        self.dynamic_obstacles = set()  # Obstacles that can move
        self.static_obstacles = set()   # Fixed obstacles
        self.agent = None
        self.robot_pos = START
        self.path = []
        self.visited = []
        self.step = 0
        self.animating = False
        self.animation_speed = 300  # milliseconds
        self.total_deliveries = 0
        self.successful_deliveries = 0
        
        # Performance tracking
        self.performance_data = {
            'episode': [],
            'path_length': [],
            'success': [],
            'rewards': []  # Track episode rewards
        }
        
        # Initialize environment
        self.generate_obstacles()
        self.reset_simulation()
        self.draw_all()
        self.update_info()

    def setup_ui(self):
        """Setup the enhanced user interface."""
        # Main frame
        main_frame = tk.Frame(self.master, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Canvas
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        canvas_frame = tk.Frame(left_frame, relief=tk.SUNKEN, bd=2)
        canvas_frame.pack(pady=5)
        
        self.canvas = tk.Canvas(
            canvas_frame, 
            width=GRID_SIZE * CELL_SIZE, 
            height=GRID_SIZE * CELL_SIZE,
            bg='white'
        )
        self.canvas.pack()
        
        # Add plotting area for episode rewards
        plot_frame = tk.LabelFrame(left_frame, text="Learning Progress", bg='#f0f0f0')
        plot_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Create matplotlib figure and canvas
        self.fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")
        self.ax.grid(True)
        
        self.plot_canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Speed control
        speed_frame = tk.Frame(left_frame, bg='#f0f0f0')
        speed_frame.pack(pady=5)
        
        tk.Label(speed_frame, text="Animation Speed:", bg='#f0f0f0').pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=5)
        speed_scale = tk.Scale(
            speed_frame, from_=1, to=10, orient=tk.HORIZONTAL,
            variable=self.speed_var, command=self.update_speed,
            bg='#f0f0f0'
        )
        speed_scale.pack(side=tk.LEFT)
        
        # Right panel - Controls and Info
        right_frame = tk.Frame(main_frame, bg='#f0f0f0', width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            right_frame, 
            text="Enhanced Delivery Robot", 
            font=("Arial", 16, "bold"),
            bg='#f0f0f0'
        )
        title_label.pack(pady=(0, 10))
        
        # Control buttons frame
        control_frame = tk.LabelFrame(right_frame, text="Controls", bg='#f0f0f0')
        control_frame.pack(fill=tk.X, pady=5)
        
        # Button grid
        button_frame = tk.Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(pady=5)
        
        self.start_button = tk.Button(
            button_frame, text="Start Mission", 
            command=self.start_mission,
            bg='#4CAF50', fg='white', width=12
        )
        self.start_button.grid(row=0, column=0, padx=2, pady=2)
        
        self.pause_button = tk.Button(
            button_frame, text="Pause", 
            command=self.toggle_pause,
            bg='#FF9800', fg='white', width=12
        )
        self.pause_button.grid(row=0, column=1, padx=2, pady=2)
        
        self.reset_button = tk.Button(
            button_frame, text="Reset", 
            command=self.reset_simulation,
            bg='#2196F3', fg='white', width=12
        )
        self.reset_button.grid(row=1, column=0, padx=2, pady=2)
        
        self.train_button = tk.Button(
            button_frame, text="Train Agent", 
            command=self.train_agent,
            bg='#9C27B0', fg='white', width=12
        )
        self.train_button.grid(row=1, column=1, padx=2, pady=2)
        
        # Environment controls
        env_frame = tk.LabelFrame(right_frame, text="Environment", bg='#f0f0f0')
        env_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(
            env_frame, text="Generate New Map", 
            command=self.generate_new_map,
            width=20
        ).pack(pady=2)
        
        tk.Button(
            env_frame, text="Add Random Obstacles", 
            command=self.add_random_obstacles,
            width=20
        ).pack(pady=2)
        
        # Training parameters frame
        train_params_frame = tk.LabelFrame(right_frame, text="Training Parameters", bg='#f0f0f0')
        train_params_frame.pack(fill=tk.X, pady=5)
        
        # Episodes slider
        episodes_frame = tk.Frame(train_params_frame, bg='#f0f0f0')
        episodes_frame.pack(fill=tk.X, pady=2)
        tk.Label(episodes_frame, text="Episodes:", bg='#f0f0f0', width=10, anchor='w').pack(side=tk.LEFT)
        self.episodes_var = tk.IntVar(value=100)
        episodes_scale = tk.Scale(
            episodes_frame, from_=10, to=1000, orient=tk.HORIZONTAL,
            variable=self.episodes_var, bg='#f0f0f0'
        )
        episodes_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Statistics frame
        stats_frame = tk.LabelFrame(right_frame, text="Statistics", bg='#f0f0f0')
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_text = tk.Text(
            stats_frame, height=8, width=35, 
            font=("Courier", 9), bg='#f8f8f8'
        )
        self.stats_text.pack(pady=5)
        
        # Info display
        info_frame = tk.LabelFrame(right_frame, text="Status", bg='#f0f0f0')
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.info_text = tk.Text(
            info_frame, height=10, width=35,
            font=("Arial", 10), bg='#f8f8f8',
            wrap=tk.WORD
        )
        scrollbar = tk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            right_frame, variable=self.progress_var,
            maximum=100, length=300
        )
        self.progress_bar.pack(pady=5)

    def generate_obstacles(self):
        """Generate a mix of static and dynamic obstacles."""
        self.obstacles.clear()
        self.static_obstacles.clear()
        self.dynamic_obstacles.clear()
        
        # Generate static obstacles (70% of total)
        static_count = int(NUM_OBSTACLES * 0.7)
        while len(self.static_obstacles) < static_count:
            pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if pos != START and pos != GOAL and pos not in self.static_obstacles:
                self.static_obstacles.add(pos)
        
        # Generate dynamic obstacles (30% of total)
        dynamic_count = NUM_OBSTACLES - static_count
        while len(self.dynamic_obstacles) < dynamic_count:
            pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if (pos != START and pos != GOAL and 
                pos not in self.static_obstacles and 
                pos not in self.dynamic_obstacles):
                self.dynamic_obstacles.add(pos)
        
        self.obstacles = self.static_obstacles | self.dynamic_obstacles

    def move_dynamic_obstacles(self):
        """Move dynamic obstacles with intelligent behavior."""
        new_dynamic_obstacles = set()
        
        for obs in self.dynamic_obstacles:
            if random.random() < OBSTACLE_MOVE_PROB:
                # Get valid neighbors
                neighbors = self.get_free_neighbors_for_obstacle(obs)
                if neighbors:
                    # Prefer moving away from robot if close
                    if abs(obs[0] - self.robot_pos[0]) + abs(obs[1] - self.robot_pos[1]) <= 3:
                        # Move away from robot
                        best_neighbor = max(neighbors, key=lambda n: 
                            abs(n[0] - self.robot_pos[0]) + abs(n[1] - self.robot_pos[1]))
                        new_dynamic_obstacles.add(best_neighbor)
                    else:
                        # Random movement
                        new_dynamic_obstacles.add(random.choice(neighbors))
                else:
                    new_dynamic_obstacles.add(obs)
            else:
                new_dynamic_obstacles.add(obs)
        
        self.dynamic_obstacles = new_dynamic_obstacles
        self.obstacles = self.static_obstacles | self.dynamic_obstacles

    def get_free_neighbors_for_obstacle(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get free neighboring positions for obstacle movement."""
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and
                (nx, ny) != self.robot_pos and 
                (nx, ny) not in self.static_obstacles and 
                (nx, ny) not in self.dynamic_obstacles and
                (nx, ny) != START and (nx, ny) != GOAL):
                neighbors.append((nx, ny))
        return neighbors

    def update_speed(self, value):
        """Update animation speed."""
        speed = int(value)
        self.animation_speed = max(50, 600 - speed * 55)  # Inverse relationship

    def draw_grid(self):
        """Draw the grid lines."""
        self.canvas.delete("grid")
        for i in range(GRID_SIZE + 1):
            # Vertical lines
            self.canvas.create_line(
                i * CELL_SIZE, 0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE,
                fill="lightgray", tags="grid"
            )
            # Horizontal lines
            self.canvas.create_line(
                0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE, i * CELL_SIZE,
                fill="lightgray", tags="grid"
            )

    def draw_obstacles(self):
        """Draw static and dynamic obstacles with different colors."""
        self.canvas.delete("obstacle")
        
        # Draw static obstacles (black)
        for x, y in self.static_obstacles:
            x1, y1 = x * CELL_SIZE + 1, y * CELL_SIZE + 1
            x2, y2 = x1 + CELL_SIZE - 2, y1 + CELL_SIZE - 2
            self.canvas.create_rectangle(
                x1, y1, x2, y2, fill="black", outline="gray", tags="obstacle"
            )
        
        # Draw dynamic obstacles (dark red)
        for x, y in self.dynamic_obstacles:
            x1, y1 = x * CELL_SIZE + 1, y * CELL_SIZE + 1
            x2, y2 = x1 + CELL_SIZE - 2, y1 + CELL_SIZE - 2
            self.canvas.create_rectangle(
                x1, y1, x2, y2, fill="darkred", outline="red", tags="obstacle"
            )

    def draw_goal(self):
        """Draw the goal position."""
        self.canvas.delete("goal")
        x, y = GOAL
        x1, y1 = x * CELL_SIZE + 3, y * CELL_SIZE + 3
        x2, y2 = x1 + CELL_SIZE - 6, y1 + CELL_SIZE - 6
        self.canvas.create_oval(x1, y1, x2, y2, fill="gold", outline="orange", width=2, tags="goal")
        
        # Add goal symbol
        cx, cy = x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2
        self.canvas.create_text(cx, cy, text="★", font=("Arial", 12, "bold"), fill="white", tags="goal")

    def draw_start(self):
        """Draw the start position."""
        self.canvas.delete("start")
        x, y = START
        x1, y1 = x * CELL_SIZE + 3, y * CELL_SIZE + 3
        x2, y2 = x1 + CELL_SIZE - 6, y1 + CELL_SIZE - 6
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightblue", outline="blue", width=2, tags="start")

    def draw_robot(self):
        """Draw the robot with direction indicator."""
        self.canvas.delete("robot")
        x, y = self.robot_pos
        cx, cy = x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 3
        
        # Robot body
        self.canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            fill="red", outline="darkred", width=2, tags="robot"
        )
        
        # Robot center dot
        self.canvas.create_oval(
            cx - 2, cy - 2, cx + 2, cy + 2,
            fill="white", tags="robot"
        )

    def draw_path_traced(self):
        """Draw the path traced by the robot."""
        self.canvas.delete("traced_path")
        
        # Draw visited positions
        for i, pos in enumerate(self.visited[:-1]):  # Exclude current position
            x, y = pos
            alpha = max(0.3, 1.0 - (len(self.visited) - i) * 0.1)  # Fade effect
            color_intensity = int(255 * alpha)
            color = f"#{color_intensity:02x}{color_intensity:02x}ff"
            
            x1 = x * CELL_SIZE + CELL_SIZE // 3
            y1 = y * CELL_SIZE + CELL_SIZE // 3
            x2 = x1 + CELL_SIZE // 3
            y2 = y1 + CELL_SIZE // 3
            
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags="traced_path")

        """
        # Draw planned path
        if len(self.path) > self.step + 1:
            for i in range(self.step + 1, len(self.path)):
                x, y = self.path[i]
                x1 = x * CELL_SIZE + CELL_SIZE // 4
                y1 = y * CELL_SIZE + CELL_SIZE // 4
                x2 = x1 + CELL_SIZE // 2
                y2 = y1 + CELL_SIZE // 2
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill="lightgreen", outline="green", tags="traced_path"
                )
        """
    def draw_all(self):
        """Draw all elements of the simulation."""
        self.draw_grid()
        self.draw_start()
        self.draw_obstacles()
        self.draw_goal()
        self.draw_path_traced()
        self.draw_robot()

    def update_info(self, additional_text=""):
        """Update the information display with enhanced details."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        status = "🤖 DELIVERY ROBOT STATUS\n"
        status += "====================\n\n"
        
        # Current position
        status += f"📍 Position: {self.robot_pos}\n"
        
        # Path information
        if self.path:
            status += f"🛣️ Path Length: {len(self.path)}\n"
            if len(self.path) > 1 and self.step < len(self.path):
                next_pos = self.path[min(self.step + 1, len(self.path) - 1)]
                status += f"⏭️ Next Move: {next_pos}\n"
        else:
            status += "❌ No path available\n"
        
        # Distance to goal
        distance_to_goal = abs(self.robot_pos[0] - GOAL[0]) + abs(self.robot_pos[1] - GOAL[1])
        status += f"🎯 Distance to Goal: {distance_to_goal}\n\n"
        
        # Mission stats
        status += f"📊 Total Deliveries: {self.total_deliveries}\n"
        status += f"✅ Successful: {self.successful_deliveries}\n"
        
        if self.total_deliveries > 0:
            success_rate = (self.successful_deliveries / self.total_deliveries) * 100
            status += f"📈 Success Rate: {success_rate:.1f}%\n"
        
        # Q-learning specific info
        if self.agent:
            status += "\n🧠 Q-LEARNING INFO\n"
            status += "====================\n"
            
            # Current state Q-values
            if self.robot_pos in self.agent.q_table:
                q_values = self.agent.q_table[self.robot_pos]
                status += "Q-values at current position:\n"
                for action, value in q_values.items():
                    status += f"  {action}: {value:.2f}\n"
            
            # Current policy
            if hasattr(self.agent, 'get_policy_at'):
                policy = self.agent.get_policy_at(self.robot_pos)
                if policy:
                    best_action = max(policy, key=policy.get)
                    status += f"Best action: {best_action}\n"
        
        status += f"\n{additional_text}"
        
        self.info_text.insert(tk.END, status)
        self.info_text.config(state=tk.DISABLED)
        
        # Update statistics
        self.update_statistics()

    def update_statistics(self):
        """Update the statistics display."""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        stats = "=== ENVIRONMENT STATS ===\n"
        stats += f"Grid Size: {GRID_SIZE}x{GRID_SIZE}\n"
        stats += f"Total Obstacles: {len(self.obstacles)}\n"
        stats += f"Static Obstacles: {len(self.static_obstacles)}\n"
        stats += f"Dynamic Obstacles: {len(self.dynamic_obstacles)}\n"
        stats += f"Free Spaces: {GRID_SIZE*GRID_SIZE - len(self.obstacles) - 2}\n"
        
        if self.agent:
            stats += f"\n=== AGENT STATS ===\n"
            
            # Get agent statistics
            agent_stats = self.agent.get_statistics()
            
            # Display Q-learning parameters
            stats += f"Alpha (learning rate): {self.agent.alpha:.3f}\n"
            stats += f"Gamma (discount): {self.agent.gamma:.3f}\n"
            stats += f"Epsilon: {self.agent.epsilon:.3f}\n"
            
            # Display training statistics
            if 'episodes' in agent_stats and agent_stats['episodes'] > 0:
                stats += f"Training Episodes: {agent_stats['episodes']}\n"
                stats += f"Success Rate: {agent_stats['success_rate']:.1f}%\n"
                stats += f"Avg Steps: {agent_stats['average_steps']:.1f}\n"
            
            # Display Q-table info
            stats += f"Q-table Size: {len(self.agent.q_table)}\n"
            
            # Display reward info if available
            if len(self.performance_data['rewards']) > 0:
                avg_reward = sum(self.performance_data['rewards']) / len(self.performance_data['rewards'])
                stats += f"Avg Reward: {avg_reward:.1f}\n"
                if len(self.performance_data['rewards']) >= 10:
                    last_10_avg = sum(self.performance_data['rewards'][-10:]) / 10
                    stats += f"Last 10 Avg Reward: {last_10_avg:.1f}\n"
            
            # Display pathfinding info
            stats += f"\n=== PATHFINDING ===\n"
            stats += "Method: Hybrid Q-learning + A*\n"
            stats += "• Pure Q-learning for learned paths\n"
            stats += "• Q-value influenced A* as fallback\n"
        
        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)

    def generate_new_map(self):
        """Generate a completely new map."""
        if self.animating:
            messagebox.showwarning("Warning", "Cannot generate new map during animation!")
            return
        
        self.generate_obstacles()
        self.reset_simulation()
        self.draw_all()
        self.update_info("🗺️ New map generated!")

    def add_random_obstacles(self):
        """Add some random obstacles to increase difficulty."""
        if self.animating:
            messagebox.showwarning("Warning", "Cannot add obstacles during animation!")
            return
        
        added = 0
        attempts = 0
        while added < 5 and attempts < 20:
            pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if (pos != START and pos != GOAL and pos != self.robot_pos and 
                pos not in self.obstacles):
                if random.choice([True, False]):
                    self.dynamic_obstacles.add(pos)
                else:
                    self.static_obstacles.add(pos)
                self.obstacles.add(pos)
                added += 1
            attempts += 1
        
        if self.agent:
            self.agent.update_obstacles(list(self.obstacles))
        
        self.draw_all()
        self.update_info(f"➕ Added {added} new obstacles!")

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        self.animating = False
        self.robot_pos = START
        self.visited = [self.robot_pos]
        self.step = 0
        
        # Initialize real Q-learning agent
        self.agent = QLearningAgent(
            grid_size=GRID_SIZE, 
            obstacles=list(self.obstacles), 
            goal=GOAL,
            alpha=0.1,
            gamma=0.95,
            epsilon=0.1,
            epsilon_decay=0.995,
            min_epsilon=0.01
        )
        self.path = self.agent.get_path_from(self.robot_pos)
        
        # Reset performance data
        self.performance_data = {
            'episode': [],
            'path_length': [],
            'success': [],
            'rewards': []
        }
        
        # Reset plot
        self.ax.clear()
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")
        self.ax.grid(True)
        self.plot_canvas.draw()
        
        # Reset UI
        self.progress_var.set(0)
        self.start_button.config(state=tk.NORMAL, text="Start Mission")
        self.pause_button.config(state=tk.DISABLED)
        
        self.draw_all()
        self.update_info("🔄 Simulation reset. Ready to start!")

    def train_agent(self):
        """Train the Q-learning agent with real updates."""
        if self.animating:
            messagebox.showwarning("Warning", "Cannot train during animation!")
            return
        
        def training_thread():
            self.train_button.config(state=tk.DISABLED)
            episodes = self.episodes_var.get()
            
            # Clear previous rewards data
            self.performance_data['episode'] = []
            self.performance_data['rewards'] = []
            
            # Train in batches to update progress bar
            # Use larger batch size for higher episode counts
            batch_size = 20 if episodes > 500 else 10
            num_batches = episodes // batch_size
            
            # Reset epsilon for fresh training
            self.agent.reset_epsilon()
            
            for batch in range(num_batches):
                # Update progress bar
                progress = (batch / num_batches) * 100
                self.progress_var.set(progress)
                self.master.update_idletasks()
                
                # Update info text periodically
                if batch % 5 == 0 or batch == num_batches - 1:
                    current_episode = batch * batch_size
                    self.update_info(f"🧠 Training in progress... {current_episode}/{episodes} episodes completed ({progress:.1f}%)")
                
                # Train for a batch of episodes
                for ep in range(batch_size):
                    current_episode = batch * batch_size + ep
                    state = START
                    episode_reward = 0
                    steps = 0
                    max_steps = GRID_SIZE * GRID_SIZE * 2  # Reasonable max steps
                    
                    while state != GOAL and steps < max_steps:
                        action = self.agent.choose_action(state)
                        if action is None:
                            break
                        
                        next_state = self.agent.move(state, action)
                        reward = self.agent.get_reward(state, action, next_state)
                        episode_reward += reward
                        
                        # Q-learning update - THIS IS THE CRITICAL PART
                        if state in self.agent.q_table:
                            old_q = self.agent.q_table[state][action]
                            
                            # Calculate next state max Q-value
                            next_max_q = 0.0
                            if next_state in self.agent.q_table and next_state != GOAL:
                                valid_next_moves = self.agent.get_valid_moves(next_state)
                                if valid_next_moves:
                                    next_q_values = [self.agent.q_table[next_state][a] for a in valid_next_moves]
                                    next_max_q = max(next_q_values)
                            
                            # Update Q-value
                            new_q = old_q + self.agent.alpha * (reward + self.agent.gamma * next_max_q - old_q)
                            self.agent.q_table[state][action] = new_q
                        
                        state = next_state
                        steps += 1
                        
                        if state == GOAL:
                            break
                    
                    # Record episode data
                    self.performance_data['episode'].append(current_episode + 1)
                    self.performance_data['rewards'].append(episode_reward)
                    
                    # Decay epsilon
                    if self.agent.epsilon > self.agent.min_epsilon:
                        self.agent.epsilon *= self.agent.epsilon_decay
                
                # Update plot every batch
                self.update_reward_plot()
            
            # Final progress update
            self.progress_var.set(100)
            
            # Get path after training
            self.path = self.agent.get_path_from(self.robot_pos)
            
            # Update statistics
            self.update_statistics()
            
            self.train_button.config(state=tk.NORMAL)
            self.update_info(f"🧠 Agent training completed! {episodes} episodes.")
        
        threading.Thread(target=training_thread, daemon=True).start()
        
    def update_reward_plot(self):
        """Update the episode rewards plot."""
        self.ax.clear()
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")
        
        if len(self.performance_data['episode']) > 0:
            self.ax.plot(
                self.performance_data['episode'],
                self.performance_data['rewards'],
                'b-'
            )
            
            # Add moving average if we have enough data
            if len(self.performance_data['rewards']) >= 10:
                window_size = min(10, len(self.performance_data['rewards']))
                rewards = np.array(self.performance_data['rewards'])
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                self.ax.plot(
                    self.performance_data['episode'][window_size-1:],
                    moving_avg,
                    'r-', 
                    linewidth=2,
                    label='Moving Avg'
                )
                self.ax.legend()
        
        self.ax.grid(True)
        self.plot_canvas.draw()

    def start_mission(self):
        """Start or resume the delivery mission."""
        if self.animating:
            return
        
        if self.robot_pos == GOAL:
            self.reset_simulation()
            return
        
        # Get new path
        self.path = self.agent.get_path_from(self.robot_pos)
        
        # If no path found, try using Q-value influenced A* pathfinding as fallback
        if not self.path or len(self.path) <= 1:
            self.path = self.a_star_path(self.robot_pos)
            if not self.path or len(self.path) <= 1:
                self.update_info("❌ No path found! Try training the agent or generating a new map.")
                return
            else:
                self.update_info("⚠️ Agent couldn't find a path. Using Q-value influenced A* pathfinding instead.")
        
        self.step = 0
        self.animating = True
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        
        self.update_info("🚀 Mission started! Robot is moving...")
        self.animate()

    def toggle_pause(self):
        """Toggle pause/resume animation."""
        if self.animating:
            self.animating = False
            self.pause_button.config(text="Resume")
            self.start_button.config(state=tk.NORMAL, text="Resume Mission")
        else:
            self.animating = True
            self.pause_button.config(text="Pause")
            self.start_button.config(state=tk.DISABLED)
            self.animate()

    def animate(self):
        """Main animation loop."""
        if not self.animating:
            return

        # Check if goal reached
        if self.robot_pos == GOAL:
            self.successful_deliveries += 1
            self.total_deliveries += 1
            self.animating = False
            self.start_button.config(state=tk.NORMAL, text="New Mission")
            self.pause_button.config(state=tk.DISABLED)
            self.update_info("🎉 Mission accomplished! Robot reached the goal!")
            return

        # Move dynamic obstacles
        self.move_dynamic_obstacles()
        if self.agent:
            self.agent.update_obstacles(list(self.obstacles))

        # Check if path is still valid
        path_blocked = False
        if self.step + 1 >= len(self.path):
            path_blocked = True
        elif self.path[self.step + 1] in self.obstacles:
            path_blocked = True
            
        if path_blocked:
            # First try using the Q-learning agent's path
            self.path = self.agent.get_path_from(self.robot_pos)
            
            # If that fails, use Q-value influenced A* as fallback
            if not self.path or len(self.path) <= 1:
                self.path = self.a_star_path(self.robot_pos)
                if not self.path or len(self.path) <= 1:
                    self.total_deliveries += 1
                    self.animating = False
                    self.start_button.config(state=tk.NORMAL, text="Retry Mission")
                    self.pause_button.config(state=tk.DISABLED)
                    self.update_info("🚫 Path blocked! Mission failed. Try again or retrain agent.")
                    return
                else:
                    self.update_info("⚠️ Path blocked! Using Q-value influenced A* to find new path.")
            else:
                self.update_info("⚠️ Path blocked! Recalculating route.")
            
            self.step = 0

        # Move robot
        if self.step + 1 < len(self.path):
            self.step += 1
            self.robot_pos = self.path[self.step]
            self.visited.append(self.robot_pos)

        # Update progress
        if len(self.path) > 1:
            progress = (self.step / (len(self.path) - 1)) * 100
            self.progress_var.set(progress)

        # Redraw and continue animation
        self.draw_all()
        self.update_info(f"🚶 Robot moving... Step {self.step}/{len(self.path)-1}")
        
        self.master.after(self.animation_speed, self.animate)

    def a_star_path(self, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm influenced by learned Q-values."""
        import heapq
        
        def heuristic(pos):
            return abs(pos[0] - GOAL[0]) + abs(pos[1] - GOAL[1])
        
        def get_q_value_for_state(state, next_state):
            """Get the Q-value for moving from state to next_state."""
            if state not in self.agent.q_table:
                return 0.0
                
            # Determine which action leads from state to next_state
            dx = next_state[0] - state[0]
            dy = next_state[1] - state[1]
            
            action = None
            if dx == 1:
                action = 'right'
            elif dx == -1:
                action = 'left'
            elif dy == 1:
                action = 'down'
            elif dy == -1:
                action = 'up'
                
            if not action:
                return 0.0
                
            # Return the Q-value if it exists
            return self.agent.q_table[state].get(action, 0.0)
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start)}
        open_set_hash = {start}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
            
            if current == GOAL:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Get valid neighbors
            neighbors = []
            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # left, right, up, down
                nx, ny = x + dx, y + dy
                if (0 <= nx < GRID_SIZE and 
                    0 <= ny < GRID_SIZE and 
                    (nx, ny) not in self.obstacles):
                    neighbors.append((nx, ny))
            
            for neighbor in neighbors:
                # Get Q-value for this transition (if available)
                q_value = get_q_value_for_state(current, neighbor)
                
                # Base step cost is 1.0
                step_cost = 1.0
                
                # Adjust cost based on Q-value: higher Q-values reduce the cost
                # Normalize Q-value to have a reasonable impact
                q_factor = 0.0
                if q_value > 0:
                    # Scale positive Q-values to reduce cost (max reduction of 0.5)
                    q_factor = min(0.5, q_value / 100.0)
                
                # Calculate g_score with Q-value influence
                tentative_g_score = g_score[current] + (step_cost - q_factor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return []  # No path found


def main():
    """Main function to run the enhanced delivery robot simulator."""
    root = tk.Tk()
    
    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Set window size to 90% of screen size
    window_width = int(screen_width * 0.9)
    window_height = int(screen_height * 0.85)
    
    # Allow window resizing
    root.resizable(True, True)
    
    # Set window size and position
    root.geometry(f"{window_width}x{window_height}+{(screen_width-window_width)//2}+{(screen_height-window_height)//2}")
    
    app = EnhancedDeliveryApp(root)
    
    # Add some nice styling
    style = ttk.Style()
    style.theme_use('clam')
    
    root.mainloop()


if __name__ == "__main__":
    main()