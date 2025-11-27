import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Dict, List, Optional, Tuple
from .rendering import Renderer

class MuseumGuideEnv(gym.Env):
    """
    Custom Gymnasium Environment for Museum Guide Agent
    
    An RL agent learns to guide visitors through Ingabo Museum,
    maximizing engagement while respecting preferences and constraints.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    # Exhibit categories
    CULTURAL = 0
    HISTORICAL = 1
    ARTISTIC = 2
    
    def __init__(self, render_mode=None, grid_size=10):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.max_steps = 300 # Define max steps as an instance variable
        
        # Define action space: 12 discrete actions
        self.action_space = spaces.Discrete(12)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            # Normalized to [0, 1] in _get_obs
            'agent_pos': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32), 
            'visitor_engagement': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            # Cast to float32 in _get_obs
            'language_pref': spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32), 
            # Normalized to [0, 1] in _get_obs
            'time_spent': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32), 
            # Cast to float32 in _get_obs
            'artifacts_viewed': spaces.Box(low=0.0, high=1.0, shape=(20,), dtype=np.float32), 
            'interest_vector': spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            'crowding': spaces.Box(low=0.0, high=1.0, shape=(grid_size, grid_size), dtype=np.float32)
        })
        
        # Initialize exhibit positions (artifact locations)
        self._init_exhibits()
        
        # Pygame initialization for rendering
        self.window = None
        self.clock = None
        self.cell_size = 60
        self.lang_switched = False
        self.renderer = Renderer(grid_size, self.cell_size, self.metadata['render_fps'])
    def _init_exhibits(self):
        """Initialize museum exhibit locations and properties"""
        # Only 9 actual exhibits, but artifacts_viewed array size is 20
        self.exhibits = {
            # Cultural exhibits
            0: {'pos': (2, 2), 'type': self.CULTURAL, 'name': 'Traditional Imigongo Art'},
            1: {'pos': (2, 7), 'type': self.CULTURAL, 'name': 'Royal Drum Collection'},
            2: {'pos': (4, 4), 'type': self.CULTURAL, 'name': 'Basket Weaving Heritage'},
            
            # Historical exhibits
            3: {'pos': (7, 2), 'type': self.HISTORICAL, 'name': 'Liberation History'},
            4: {'pos': (7, 7), 'type': self.HISTORICAL, 'name': 'Kingdom of Rwanda'},
            5: {'pos': (5, 6), 'type': self.HISTORICAL, 'name': 'Colonial Period'},
            
            # Artistic exhibits
            6: {'pos': (8, 4), 'type': self.ARTISTIC, 'name': 'Contemporary Art'},
            7: {'pos': (3, 8), 'type': self.ARTISTIC, 'name': 'Sculpture Gallery'},
            8: {'pos': (6, 3), 'type': self.ARTISTIC, 'name': 'Photography Exhibition'}
        }
        
        # Store artifact positions as a list for easy access in helper function
        self.artifact_positions = [v['pos'] for v in self.exhibits.values()]
        
        # Rest areas
        self.rest_areas = [(0, 5), (9, 5)]
        
        # Entry and exit
        self.entry_pos = (0, 0)
        self.exit_pos = (9, 9)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
    
        # Reset agent position to entry
        self.agent_pos = np.array(self.entry_pos, dtype=np.int32)
        self.lang_switched = False

        # Initialize visitor profile randomly
        self.visitor_engagement = 0.8
        self.language_pref = self.np_random.integers(0, 3)
        self.time_spent = 0
        self.artifacts_viewed = np.zeros(20, dtype=np.float32) # Changed to float32
    
        # Random interest profile
        self.interest_vector = self.np_random.random(3).astype(np.float32)
        self.interest_vector /= self.interest_vector.sum()
    
        # Initialize crowding (LESS crowded overall)
        self.crowding = self.np_random.random((self.grid_size, self.grid_size)).astype(np.float32) * 0.3 
    
        self.current_language = 0
        self.steps = 0
    
        observation = self._get_obs()
        info = self._get_info()
    
        return observation, info
    
    def step(self, action):
        self.steps += 1
        self.time_spent += 1
    
        reward = 0
        terminated = False
    
        # Process action
        if action in [0, 1, 2, 3]:  # Movement actions
           reward += self._move_agent(action)
        elif action in [4, 5, 6]:  # Recommend artifact by category
           reward += self._recommend_artifact(action - 4)
        elif action == 7:  # Provide detailed info
           reward += self._provide_info()
        elif action == 8:  # Switch to Kinyarwanda
           reward += self._switch_language(1)
        elif action == 9:  # Switch to English
           reward += self._switch_language(0)
        elif action == 10:  # Suggest rest
           reward += self._suggest_rest()
        elif action == 11:  # End tour
            terminated = True
            if self.steps < 30:  # Penalize ending tour too early
               reward += -20
            else:
               reward += self._end_tour_reward()
    
        # Update engagement (decays over time)
        self._update_engagement()
    
        # Check terminal conditions
        if self.visitor_engagement < 0.2:
           terminated = True
           reward -= 10 
    
        if self.time_spent >= 200:  
           terminated = True
           reward += 15 if self.visitor_engagement > 0.6 else -5
    
        if self.steps >= self.max_steps:
           terminated = True
    
        observation = self._get_obs()
        info = self._get_info()
    
        return observation, reward, terminated, False, info

    def _min_distance_to_unviewed_artifact(self):
        """Calculates the Manhattan distance to the closest unviewed artifact."""
        
        # Only check the first 9 artifact indices
        unviewed_indices = np.where(self.artifacts_viewed[:9] == 0)[0]
        
        if len(unviewed_indices) == 0:
            return 0  # No goals left
        
        min_dist = float('inf')
        current_pos = self.agent_pos
        
        for i in unviewed_indices:
            artifact_pos = self.artifact_positions[i]
            # Manhattan distance (L1 norm)
            dist = np.sum(np.abs(current_pos - artifact_pos))
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    def _move_agent(self, direction):
        """Move agent in grid: 0=North, 1=South, 2=East, 3=West"""
        new_pos = self.agent_pos.copy()
        
        # 1. Get distance BEFORE move (for reward shaping)
        old_dist = self._min_distance_to_unviewed_artifact()
    
        if direction == 0 and self.agent_pos[1] > 0:  # North
           new_pos[1] -= 1
        elif direction == 1 and self.agent_pos[1] < self.grid_size - 1:  # South
           new_pos[1] += 1
        elif direction == 2 and self.agent_pos[0] < self.grid_size - 1:  # East
           new_pos[0] += 1
        elif direction == 3 and self.agent_pos[0] > 0:  # West
           new_pos[0] -= 1
        else:
           # Invalid move (hit wall)
           return -1
    
        # Check crowding penalty 
        crowd_level = self.crowding[new_pos[0], new_pos[1]]
        reward = -3 if crowd_level > 0.7 else -0.1  # Small step penalty
    
        self.agent_pos = new_pos
        
        # 2. Get distance AFTER move
        new_dist = self._min_distance_to_unviewed_artifact()
        
        # 3. Reward Shaping: Encourage progress toward the nearest goal
        progress = old_dist - new_dist
        # A positive progress means the agent moved closer.
        # This small, dense reward guides the agent's movement.
        reward += progress * 0.5 
        
        # Bonus for reaching an exhibit
        at_exhibit = self._check_exhibit_proximity()
        if at_exhibit is not None and self.artifacts_viewed[at_exhibit] == 0:
           # Make the main reward for finding an exhibit substantial
           reward += 5 
           self.artifacts_viewed[at_exhibit] = 1.0 # Set to 1.0 (float)

        return reward
    
    def _recommend_artifact(self, category):
        """Recommend artifact based on category"""
        # Check if at exhibit location
        at_exhibit = self._check_exhibit_proximity()
        
        if at_exhibit is not None:
            exhibit = self.exhibits[at_exhibit]
            
            # Reward based on interest match
            if exhibit['type'] == category:
                interest_score = self.interest_vector[category]
                
                # Only give reward if already viewed by agent (to prevent double-counting)
                # The _move_agent function now sets the artifact to viewed.
                if self.artifacts_viewed[at_exhibit] == 1.0:
                    reward = 10 + (interest_score * 10)
                    self.visitor_engagement = min(1.0, self.visitor_engagement + 0.1)
                    return reward
                else:
                    # Agent is at the location but hasn't "viewed" it yet (bug fix: should not happen if _move_agent is working)
                    return 0
            else:
                # Penalty for poor recommendation
                self.visitor_engagement = max(0.0, self.visitor_engagement - 0.05)
                return -5
        
        return -2  # Small penalty for recommending when not at exhibit
    
    def _provide_info(self):
        """Provide detailed information"""
        at_exhibit = self._check_exhibit_proximity()
        
        if at_exhibit is not None and self.artifacts_viewed[at_exhibit] == 1.0: # Check as float
            # Bonus for using correct language
            lang_bonus = 5 if self.current_language == self.language_pref or self.language_pref == 2 else 0
            self.visitor_engagement = min(1.0, self.visitor_engagement + 0.15)
            return 15 + lang_bonus
        
        return 0
    
    def _switch_language(self, lang):
        """Switch language: 0=English, 1=Kinyarwanda"""
        self.current_language = lang
        match = (lang == self.language_pref or self.language_pref == 2)
        if match:
           if not self.lang_switched:  # NEW: Bonus only once
            self.lang_switched = True
            return 5
        else:
            return 0  # No spam
        return -3
    
    def _suggest_rest(self):
        """Suggest rest area"""
        if tuple(self.agent_pos) in self.rest_areas:
            if self.visitor_engagement < 0.5:  # Good timing
                self.visitor_engagement = min(1.0, self.visitor_engagement + 0.2)
                return 8
        return -2
    
    def _end_tour_reward(self):
        """Calculate reward for ending tour"""
        artifacts_count = int(self.artifacts_viewed.sum())
        
        if artifacts_count >= 5 and self.visitor_engagement > 0.6:
            return 20
        elif artifacts_count >= 3:
            return 10
        else:
            return -10
    
    def _update_engagement(self):
        """Update visitor engagement (natural decay)"""
        # MUCH SLOWER decay: 0.003
        self.visitor_engagement = max(0.0, self.visitor_engagement - 0.003)
    
    def _check_exhibit_proximity(self):
        """Check if agent is at an exhibit"""
        for exhibit_id, exhibit in self.exhibits.items():
            if tuple(self.agent_pos) == exhibit['pos']:
                return exhibit_id
        return None
    
    def _get_obs(self):
        """Returns the current observation dictionary with NORMALIZED values."""
        
        GRID_MAX = self.grid_size - 1  # 9
        TIME_MAX = self.max_steps     # 300 
        
        return {
            # Normalize agent position (0-9) to [0, 1]
            'agent_pos': self.agent_pos.astype(np.float32) / GRID_MAX,
            # Already normalized
            'visitor_engagement': np.array([self.visitor_engagement], dtype=np.float32), 
            # Cast to float32
            'language_pref': np.array([self.language_pref]).astype(np.float32), 
            # Normalize time spent (0-300) to [0, 1]
            'time_spent': np.array([self.time_spent], dtype=np.float32) / TIME_MAX,
            # Already 0 or 1, but ensure float32
            'artifacts_viewed': self.artifacts_viewed,
            'interest_vector': self.interest_vector,
            'crowding': self.crowding
        }
    
    def _get_info(self):
        return {
            'artifacts_viewed_count': int(self.artifacts_viewed.sum()),
            'engagement_level': float(self.visitor_engagement),
            'time_remaining': self.max_steps - self.time_spent
        }
    
    def render(self):
        if self.render_mode == 'human':
            canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size + 100))
            self.renderer.render_human(canvas, self.crowding, self.exhibits, self.artifacts_viewed, self.rest_areas, self.entry_pos, self.exit_pos, self.agent_pos, self.visitor_engagement, self.time_spent, int(self.artifacts_viewed.sum()), self.current_language)
            self.renderer.window.blit(canvas, (0, 0))
            # ... existing pump/update/clock ...
        elif self.render_mode == 'rgb_array':
            return self.renderer.render_rgb_array(self.crowding, self.exhibits, self.artifacts_viewed, self.rest_areas, self.entry_pos, self.exit_pos, self.agent_pos, self.visitor_engagement, self.time_spent, int(self.artifacts_viewed.sum()), self.current_language)
        
    def _render_pygame(self):
        """Render using pygame"""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size, 
                                                    self.grid_size * self.cell_size + 100))
            pygame.display.set_caption("Museum Guide Agent")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.grid_size * self.cell_size, 
                                self.grid_size * self.cell_size + 100))
        canvas.fill((255, 255, 255))
        
        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size)
                
                # Color based on crowding
                crowd = self.crowding[x, y]
                color_intensity = int(255 * (1 - crowd))
                pygame.draw.rect(canvas, (color_intensity, color_intensity, 255), rect)
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)
        
        # Draw exhibits
        for exhibit_id, exhibit in self.exhibits.items():
            x, y = exhibit['pos']
            center = (x * self.cell_size + self.cell_size // 2,
                     y * self.cell_size + self.cell_size // 2)
            
            # Color by type
            if exhibit['type'] == self.CULTURAL:
                color = (255, 200, 0)  # Gold
            elif exhibit['type'] == self.HISTORICAL:
                color = (150, 75, 0)   # Brown
            else:
                color = (255, 0, 255)  # Magenta
            
            pygame.draw.circle(canvas, color, center, self.cell_size // 3)
            
            # Mark if viewed (check as boolean/float 1.0)
            if self.artifacts_viewed[exhibit_id] > 0.5: 
                pygame.draw.circle(canvas, (0, 255, 0), center, self.cell_size // 6)
        
        # Draw rest areas
        for rest_x, rest_y in self.rest_areas:
            rect = pygame.Rect(rest_x * self.cell_size + 10, rest_y * self.cell_size + 10,
                              self.cell_size - 20, self.cell_size - 20)
            pygame.draw.rect(canvas, (0, 200, 0), rect)
        
        # Draw entry/exit
        pygame.draw.rect(canvas, (0, 255, 0), 
                        (0, 0, self.cell_size, self.cell_size), 3)
        pygame.draw.rect(canvas, (255, 0, 0), 
                        ((self.grid_size-1) * self.cell_size, 
                         (self.grid_size-1) * self.cell_size, 
                         self.cell_size, self.cell_size), 3)
        
        # Draw agent
        agent_center = (self.agent_pos[0] * self.cell_size + self.cell_size // 2,
                       self.agent_pos[1] * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(canvas, (255, 0, 0), agent_center, self.cell_size // 4)
        
        # Draw info panel
        info_y = self.grid_size * self.cell_size
        pygame.draw.rect(canvas, (240, 240, 240), (0, info_y, 
                        self.grid_size * self.cell_size, 100))
        
        font = pygame.font.Font(None, 24)
        texts = [
            f"Engagement: {self.visitor_engagement:.2f}",
            f"Time: {self.time_spent}/{self.max_steps} steps",
            f"Artifacts: {self.artifacts_viewed.sum()}/9",
            f"Language: {'EN' if self.current_language == 0 else 'RW'}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (0, 0, 0))
            canvas.blit(text_surface, (10 + i * 150, info_y + 40))
        
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata['render_fps'])
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.renderer.close()