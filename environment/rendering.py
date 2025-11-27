# environment/rendering.py
"""
Visualization GUI components for Museum Guide Env
Separate rendering logic for modularity (Pygame-based 2D grid).
Import in custom_env.py: from .rendering import render_pygame, render_rgb_array
"""

import pygame
import numpy as np
from typing import Optional

class Renderer:
    """
    Pygame renderer for the museum environment.
    Handles drawing grid, agent, exhibits, rests, entry/exit, and info panel.
    """
    
    def __init__(self, grid_size: int = 10, cell_size: int = 60, render_fps: int = 4):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_fps = render_fps
        self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': render_fps}
        self.window = None
        self.clock = None
        self.font = None
    
    def init_pygame(self):
        """Initialize Pygame if not done."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size, 
                                                   self.grid_size * self.cell_size + 100))
            pygame.display.set_caption("Museum Guide Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
    
    def render_grid(self, canvas: pygame.Surface, crowding: np.ndarray):
        """Draw grid cells colored by crowding."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                   self.cell_size, self.cell_size)
                crowd = crowding[x, y]
                color_intensity = int(255 * (1 - crowd))
                pygame.draw.rect(canvas, (color_intensity, color_intensity, 255), rect)
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)
    
    def render_exhibits(self, canvas: pygame.Surface, exhibits: dict, artifacts_viewed: np.ndarray):
        """Draw exhibit circles, colored by type, green inner if viewed."""
        for exhibit_id, exhibit in exhibits.items():
            x, y = exhibit['pos']
            center = (x * self.cell_size + self.cell_size // 2,
                      y * self.cell_size + self.cell_size // 2)
            if exhibit['type'] == 0:  # Cultural
                color = (255, 200, 0)  # Gold
            elif exhibit['type'] == 1:  # Historical
                color = (150, 75, 0)  # Brown
            else:  # Artistic
                color = (255, 0, 255)  # Magenta
            pygame.draw.circle(canvas, color, center, self.cell_size // 3)
            if artifacts_viewed[exhibit_id]:
                pygame.draw.circle(canvas, (0, 255, 0), center, self.cell_size // 6)
    
    def render_rest_areas(self, canvas: pygame.Surface, rest_areas: list):
        """Draw green rest rectangles."""
        for rest_x, rest_y in rest_areas:
            rect = pygame.Rect(rest_x * self.cell_size + 10, rest_y * self.cell_size + 10,
                               self.cell_size - 20, self.cell_size - 20)
            pygame.draw.rect(canvas, (0, 200, 0), rect)
    
    def render_entry_exit(self, canvas: pygame.Surface, entry_pos: tuple, exit_pos: tuple, grid_size: int, cell_size: int):
        """Draw green entry and red exit borders."""
        pygame.draw.rect(canvas, (0, 255, 0), 
                         (entry_pos[0] * cell_size, entry_pos[1] * cell_size, cell_size, cell_size), 3)
        pygame.draw.rect(canvas, (255, 0, 0), 
                         ((grid_size-1) * cell_size, (grid_size-1) * cell_size, cell_size, cell_size), 3)
    
    def render_agent(self, canvas: pygame.Surface, agent_pos: np.ndarray, cell_size: int):
        """Draw red agent circle."""
        agent_center = (int(agent_pos[0] * cell_size + cell_size // 2),
                        int(agent_pos[1] * cell_size + cell_size // 2))
        pygame.draw.circle(canvas, (255, 0, 0), agent_center, cell_size // 4)
    
    def render_info_panel(self, canvas: pygame.Surface, grid_size: int, cell_size: int, 
                          engagement: float, time_spent: int, artifacts_count: int, current_lang: int):
        """Draw bottom info panel."""
        info_y = grid_size * cell_size
        pygame.draw.rect(canvas, (240, 240, 240), (0, info_y, grid_size * cell_size, 100))
        texts = [
            f"Engagement: {engagement:.2f}",
            f"Time: {time_spent}/200 min",
            f"Artifacts: {artifacts_count}/9",
            f"Language: {'EN' if current_lang == 0 else 'RW'}"
        ]
        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, (0, 0, 0))
            canvas.blit(text_surface, (10 + i * 150, info_y + 40))
    
    def render_human(self, canvas: pygame.Surface, crowding: np.ndarray, exhibits: dict, 
                     artifacts_viewed: np.ndarray, rest_areas: list, entry_pos: tuple, 
                     exit_pos: tuple, agent_pos: np.ndarray, engagement: float, 
                     time_spent: int, artifacts_count: int, current_lang: int):
        """Full human render (to window)."""
        self.init_pygame()
        canvas.fill((255, 255, 255))
        
        self.render_grid(canvas, crowding)
        self.render_exhibits(canvas, exhibits, artifacts_viewed)
        self.render_rest_areas(canvas, rest_areas)
        self.render_entry_exit(canvas, entry_pos, exit_pos, self.grid_size, self.cell_size)
        self.render_agent(canvas, agent_pos, self.cell_size)
        self.render_info_panel(canvas, self.grid_size, self.cell_size, engagement, time_spent, artifacts_count, current_lang)
        
        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)
    
    def render_rgb_array(self, crowding: np.ndarray, exhibits: dict, artifacts_viewed: np.ndarray, 
                         rest_areas: list, entry_pos: tuple, exit_pos: tuple, agent_pos: np.ndarray, 
                         engagement: float, time_spent: int, artifacts_count: int, current_lang: int):
        """Render to RGB array (no window)."""
        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size + 100))
        canvas.fill((255, 255, 255))
        
        self.render_grid(canvas, crowding)
        self.render_exhibits(canvas, exhibits, artifacts_viewed)
        self.render_rest_areas(canvas, rest_areas)
        self.render_entry_exit(canvas, entry_pos, exit_pos, self.grid_size, self.cell_size)
        self.render_agent(canvas, agent_pos, self.cell_size)
        self.render_info_panel(canvas, self.grid_size, self.cell_size, engagement, time_spent, artifacts_count, current_lang)
        
        array = pygame.surfarray.array3d(canvas)
        return np.transpose(array, axes=(1, 0, 2))  # To HWC
    
    def close(self):
        """Close Pygame."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

# Usage in custom_env.py:
# self.renderer = Renderer(grid_size, cell_size)
# In render():
# if self.render_mode == 'human':
#     self.renderer.render_human(canvas, self.crowding, self.exhibits, self.artifacts_viewed, self.rest_areas, self.entry_pos, self.exit_pos, self.agent_pos, self.visitor_engagement, self.time_spent, self.artifacts_viewed.sum(), self.current_language)
# elif self.render_mode == 'rgb_array':
#     return self.renderer.render_rgb_array(...)
# self.renderer.close() in close()