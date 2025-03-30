# --- Imports and Pygame Init ---
import pygame
print("Pygame Version:", pygame.ver) # Check your version!
import sys
import random
import math
import time
import threading
import queue
import socket
import json
import traceback # Added for better error reporting

pygame.init()

# --- Get Screen Info and Set Fullscreen Mode ---
try:
    info = pygame.display.Info()
    actual_screen_width = info.current_w
    actual_screen_height = info.current_h
    print(f"Detected screen resolution: {actual_screen_width}x{actual_screen_height}")
except pygame.error as e:
    print(f"Warning: Could not get screen info ({e}). Using default 800x450.")
    actual_screen_width = 800
    actual_screen_height = 450

try:
    screen = pygame.display.set_mode((actual_screen_width, actual_screen_height), pygame.FULLSCREEN | pygame.SCALED | pygame.SRCALPHA) # Added SCALED and SRCALPHA potentially
    # screen = pygame.display.set_mode((actual_screen_width, actual_screen_height), pygame.FULLSCREEN)
    print("Fullscreen mode set.")
except pygame.error as e:
    print(f"Warning: Could not set fullscreen mode ({e}). Trying windowed mode.")
    screen = pygame.display.set_mode((actual_screen_width, actual_screen_height)) # Fallback

pygame.display.set_caption("SmartBin™ Waste Management System")

# --- Update Global Dimensions USED BY GUI ELEMENTS ---
SCREEN_WIDTH = actual_screen_width
SCREEN_HEIGHT = actual_screen_height

# --- Define colors ---
WHITE = (255, 255, 255); BLACK = (0, 0, 0); CREAM = (248, 246, 235)
LIGHT_CREAM = (252, 250, 245); LEAF_GREEN = (105, 162, 76); LIGHT_GREEN = (183, 223, 177)
DARK_GREEN = (65, 122, 36); SOFT_BROWN = (139, 98, 65); LIGHT_BROWN = (188, 152, 106)
WATER_BLUE = (99, 171, 190); LIGHT_BLUE = (175, 219, 230); SUNSET_ORANGE = (233, 127, 2)
WOOD_BROWN = (160, 120, 85); TEXT_BROWN = (90, 65, 40)

# --- Fonts ---
try:
    # Use common cross-platform fonts first if available
    font_title = pygame.font.SysFont("DejaVu Sans", 36, bold=True) # Or Verdana, Arial
    font_large = pygame.font.SysFont("DejaVu Sans", 30, bold=True)
    font_medium = pygame.font.SysFont("DejaVu Sans", 22)
    font_small = pygame.font.SysFont("DejaVu Sans", 18)
    font_tiny = pygame.font.SysFont("DejaVu Sans", 16)
    print("Loaded DejaVu Sans system font.")
except Exception as e1:
    print(f"Warning: Could not load system font 'DejaVu Sans'. Trying Arial. Error: {e1}")
    try:
        font_title = pygame.font.SysFont("Arial", 36, bold=True)
        font_large = pygame.font.SysFont("Arial", 30, bold=True)
        font_medium = pygame.font.SysFont("Arial", 22)
        font_small = pygame.font.SysFont("Arial", 18)
        font_tiny = pygame.font.SysFont("Arial", 16)
        print("Loaded Arial system font.")
    except Exception as e2:
        print(f"Warning: Could not load system font 'Arial'. Using default font. Error: {e2}")
        font_title = pygame.font.Font(None, 40); font_large = pygame.font.Font(None, 34)
        font_medium = pygame.font.Font(None, 26); font_small = pygame.font.Font(None, 22)
        font_tiny = pygame.font.Font(None, 18)

# --- Communication Setup ---
detection_queue = queue.Queue()
GUI_SERVER_HOST = '0.0.0.0' # Listen on all available network interfaces
GUI_SERVER_PORT = 9999
server_running = True

# --- Pygame GUI Classes ---

class BinStats: # (Unchanged)
    def __init__(self):
        self.recycled_items = 0
        self.landfill_items = 0
        self.total_items = 0
        self.co2_saved = 0 # Example environmental impact metric
        self.water_saved = 0 # Example environmental impact metric
        self.last_updated = time.time()

    def update_stats(self, bin_type):
        """Updates counts based on the type of item processed."""
        self.total_items += 1
        if bin_type.upper() == "RECYCLING":
            self.recycled_items += 1
            # Add rough estimates for environmental impact (adjust as needed)
            self.co2_saved += random.uniform(0.1, 0.3) # kg CO2 eq saved per recycled item
            self.water_saved += random.uniform(0.5, 2.0) # Liters water saved
        elif bin_type.upper() == "TRASH":
            self.landfill_items += 1
        self.last_updated = time.time()

    def get_recycling_percentage(self):
        """Calculates the percentage of items that were recycled."""
        if self.total_items == 0:
            return 0
        return (self.recycled_items / self.total_items) * 100

    def get_landfill_percentage(self):
        """Calculates the percentage of items that went to landfill."""
        if self.total_items == 0:
            return 0
        return (self.landfill_items / self.total_items) * 100

class NatureElement: # (Unchanged)
    def __init__(self):
        self.reset()
        self.y = random.randint(0, SCREEN_HEIGHT) # Start some visible

    def reset(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = -random.randint(10, 50) # Start off-screen top
        self.type = random.choice(["leaf", "drop"])
        self.size = random.randint(3, 8) if self.type == "leaf" else random.randint(2, 5)
        self.speed = random.uniform(0.3, 1.0)
        self.drift = random.uniform(-0.3, 0.3)
        self.rotation = random.randint(0, 360)
        self.rot_speed = random.uniform(-1, 1)
        if self.type == "leaf":
            green_variation = random.randint(-20, 20)
            base_color = LEAF_GREEN
            r = max(0, min(255, base_color[0] + green_variation))
            g = max(0, min(255, base_color[1] + green_variation))
            b = max(0, min(255, base_color[2] - green_variation))
            self.color = (r, g, b)
        else: # drop
            self.color = WATER_BLUE

    def update(self):
        self.y += self.speed
        self.x += self.drift
        self.rotation += self.rot_speed
        # Reset if off-screen
        if self.y > SCREEN_HEIGHT + 20 or self.x < -20 or self.x > SCREEN_WIDTH + 20:
            self.reset()

    def draw(self, surface):
        try:
            if self.type == "leaf":
                # Simple leaf shape
                leaf_surf = pygame.Surface((self.size * 4, self.size * 3), pygame.SRCALPHA)
                leaf_color = (*self.color[:3], 180) # Add alpha
                stem_color = (*SOFT_BROWN[:3], 180)
                pygame.draw.ellipse(leaf_surf, leaf_color, (0, 0, self.size * 3, self.size * 2))
                pygame.draw.line(leaf_surf, stem_color, (self.size * 1.5, self.size * 1), (self.size * 3, self.size * 2), max(1, self.size // 3))
                rotated_leaf = pygame.transform.rotate(leaf_surf, self.rotation)
                leaf_rect = rotated_leaf.get_rect(center=(int(self.x), int(self.y)))
                surface.blit(rotated_leaf, leaf_rect)
            else: # drop
                drop_surf = pygame.Surface((self.size * 2, self.size * 3), pygame.SRCALPHA)
                drop_color = (*WATER_BLUE[:3], 150)
                pygame.draw.circle(drop_surf, drop_color, (self.size, self.size), self.size)
                # Teardrop point
                points = [(self.size - self.size/2, self.size), (self.size + self.size/2, self.size), (self.size, self.size * 2.5)]
                pygame.draw.polygon(drop_surf, drop_color, points)
                # Highlight
                highlight_pos = (int(self.size * 0.7), int(self.size * 0.7))
                highlight_radius = max(1, int(self.size / 3))
                highlight_color = (*LIGHT_BLUE[:3], 180)
                pygame.draw.circle(drop_surf, highlight_color, highlight_pos, highlight_radius)
                rotated_drop = pygame.transform.rotate(drop_surf, self.rotation / 10) # Slower rotation for drops
                drop_rect = rotated_drop.get_rect(center=(int(self.x), int(self.y)))
                surface.blit(rotated_drop, drop_rect)
        except (pygame.error, TypeError, ValueError) as e:
            print(f"Warning: Error drawing nature element: {e}")
            self.reset() # Reset if drawing fails

class NaturalProgressBar: # (Unchanged)
    def __init__(self, x, y, width, height, color, bg_color, max_value=100, style="plant"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.bg_color = bg_color
        self.max_value = max_value
        self.current_value = 0
        self.target_value = 0
        self.animation_speed = 2 # Speed at which the bar visually fills/empties
        self.style = style # "plant" or "water"

        if self.style == "plant":
            self.vine_points = []
            self.leaf_positions = [] # Store indices and side (-1 or 1)
            self.generate_vine_path()

    def generate_vine_path(self):
        """Generates a slightly randomized path for the vine."""
        self.vine_points = []
        segment_height = self.height / 10.0 # Divide height into 10 segments
        if segment_height <= 0: return # Avoid division by zero or negative height

        for i in range(11): # 11 points for 10 segments
            y_pos = self.y + self.height - (i * segment_height)
            # Add horizontal wiggle, increasing towards the top? No, keep it simple.
            wiggle = random.uniform(-self.width/6, self.width/6) if i > 0 else 0
            x_pos = self.x + self.width/2 + wiggle
            self.vine_points.append((x_pos, y_pos))

            # Add potential leaf positions every few segments
            if i > 0 and i % 2 == 0: # e.g., at segment 2, 4, 6, 8, 10
                leaf_side = 1 if i % 4 == 0 else -1 # Alternate sides
                self.leaf_positions.append((i, leaf_side)) # Store segment index and side

    def set_value(self, value):
        """Sets the target value for the progress bar."""
        self.target_value = max(0, min(value, self.max_value))

    def update(self):
        """Updates the current value towards the target value for animation."""
        if abs(self.current_value - self.target_value) < self.animation_speed:
            self.current_value = self.target_value
        elif self.current_value < self.target_value:
            self.current_value += self.animation_speed
        elif self.current_value > self.target_value:
            self.current_value -= self.animation_speed

    def draw(self, surface):
        """Draws the progress bar based on its style."""
        try:
            # Draw background rectangle
            bg_rect = pygame.Rect(self.x, self.y, self.width, self.height)
            pygame.draw.rect(surface, self.bg_color, bg_rect) # Border radius ignored
            pygame.draw.rect(surface, WOOD_BROWN, bg_rect, 2) # Outline

            # Calculate progress ratio (0.0 to 1.0)
            progress_ratio = self.current_value / self.max_value if self.max_value > 0 else 0
            progress_ratio = max(0, min(1, progress_ratio)) # Clamp between 0 and 1

            # Draw based on style
            if self.style == "plant":
                self.draw_plant_progress(surface, progress_ratio)
            else: # water
                self.draw_water_progress(surface, progress_ratio)

            # Draw percentage text
            text_color = TEXT_BROWN
            text = font_medium.render(f"{int(self.current_value)}%", True, text_color)
            text_rect = text.get_rect(center=(self.x + self.width/2, self.y + 25)) # Position near top
            surface.blit(text, text_rect)
        except (pygame.error, ValueError, ZeroDivisionError, IndexError) as e:
            print(f"Warning: Error drawing progress bar: {e}")
            # Draw fallback background if error occurs
            pygame.draw.rect(surface, self.bg_color, (self.x, self.y, self.width, self.height))

    def draw_plant_progress(self, surface, progress_ratio):
         """Draws the plant/vine style progress."""
         visible_segments = math.ceil(progress_ratio * 10) # How many segments to show

         # Draw vine segments
         if visible_segments > 0 and len(self.vine_points) > 1:
             for i in range(min(visible_segments, len(self.vine_points) - 1)):
                 if i + 1 < len(self.vine_points): # Ensure end point exists
                     start_point = self.vine_points[i]
                     end_point = self.vine_points[i + 1]
                     thickness = random.randint(3, 5) # Vary thickness slightly
                     pygame.draw.line(surface, DARK_GREEN, start_point, end_point, thickness)

                     # Draw leaves at designated positions
                     for leaf_idx, leaf_side in self.leaf_positions:
                         if i == leaf_idx - 1: # Draw leaf at the end of this segment
                             # Position leaf midway along the segment
                             leaf_x = (start_point[0] + end_point[0]) / 2
                             leaf_y = (start_point[1] + end_point[1]) / 2
                             leaf_size = random.randint(5, 8)
                             try:
                                 leaf_surf = pygame.Surface((leaf_size * 3, leaf_size * 2), pygame.SRCALPHA)
                                 pygame.draw.ellipse(leaf_surf, LEAF_GREEN, (0, 0, leaf_size * 3, leaf_size * 2))
                                 angle = 45 if leaf_side > 0 else -45 # Angle based on side
                                 rotated_leaf = pygame.transform.rotate(leaf_surf, angle)
                                 leaf_rect = rotated_leaf.get_rect(center=(leaf_x + (leaf_side * 15), leaf_y)) # Offset sideways
                                 surface.blit(rotated_leaf, leaf_rect)
                             except pygame.error as leaf_err:
                                 print(f"Warning: Error drawing leaf: {leaf_err}")


             # Draw flower at the top if progress is high
             if progress_ratio > 0.9 and len(self.vine_points) > 0:
                 # Get the topmost point of the vine
                 if len(self.vine_points) > 0: # Check again just in case
                     top_x, top_y = self.vine_points[-1]
                     petal_color = (255, 200, 100) # Yellow/Orange petals
                     center_color = SUNSET_ORANGE
                     try:
                         # Draw petals
                         for angle_deg in range(0, 360, 60): # 6 petals
                             angle_rad = math.radians(angle_deg)
                             petal_x = top_x + 8 * math.cos(angle_rad)
                             petal_y = top_y + 8 * math.sin(angle_rad)
                             pygame.draw.circle(surface, petal_color, (int(petal_x), int(petal_y)), 7)
                         # Draw center
                         pygame.draw.circle(surface, center_color, (int(top_x), int(top_y)), 5)
                     except pygame.error as flower_err:
                         print(f"Warning: Error drawing flower: {flower_err}")

    def draw_water_progress(self, surface, progress_ratio):
        """Draws the water style progress."""
        fill_height = int(self.height * progress_ratio)
        if fill_height > 0:
            # Draw the main water fill area
            fill_rect = pygame.Rect(self.x, self.y + self.height - fill_height, self.width, fill_height)
            pygame.draw.rect(surface, self.color, fill_rect)

            # Draw a simple wave effect at the top
            wave_height_amp = 3 # Amplitude of the wave
            wave_y_base = self.y + self.height - fill_height - wave_height_amp
            try:
                wave_surface = pygame.Surface((self.width, wave_height_amp * 2), pygame.SRCALPHA)
                lighter_color = LIGHT_BLUE # Color for the wave crest
                num_points = self.width // 5 # Number of points for the polygon wave
                if num_points < 2: return # Need at least 2 points for polygon

                wave_points = [(0, wave_height_amp * 2)] # Start bottom-left
                for i in range(num_points + 1):
                    x = int(i * (self.width / num_points))
                    # Simple sine wave based on time and horizontal position
                    offset = math.sin(time.time() * 3 + x * 0.05) * wave_height_amp
                    wave_points.append((x, wave_height_amp + offset))
                wave_points.append((self.width, wave_height_amp * 2)) # End bottom-right

                pygame.draw.polygon(wave_surface, lighter_color, wave_points)
                surface.blit(wave_surface, (self.x, wave_y_base))
            except pygame.error as wave_err:
                 print(f"Warning: Error drawing wave: {wave_err}")

class EcoButton: # (Unchanged)
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=WHITE, border_radius=10): # border_radius unused visually
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False
        self.animation = 0 # For click feedback

    def draw(self, surface):
        try:
            current_color = self.hover_color if self.hovered else self.color
            # Click feedback visual
            if self.animation > 0:
                self.animation -= 0.1 # Fade out effect
                # Slightly darken color for feedback
                feedback_color = tuple(max(0, c - 30) for c in current_color[:3])
                current_color = feedback_color

            pygame.draw.rect(surface, current_color, self.rect) # No border radius

            # Simple subtle highlight at the top
            highlight_rect = pygame.Rect(self.rect.x + 5, self.rect.y + 3, self.rect.width - 10, self.rect.height // 4)
            highlight_color = (*WHITE[:3], 30) # Very transparent white
            pygame.draw.rect(surface, highlight_color, highlight_rect) # No border radius

            pygame.draw.rect(surface, WOOD_BROWN, self.rect, 2) # Outline

            # Draw text
            text_surf = font_medium.render(self.text, True, self.text_color)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)
        except pygame.error as e:
            print(f"Warning: Error drawing button: {e}")

    def check_hover(self, mouse_pos):
        if self.rect: # Ensure rect is valid
             self.hovered = self.rect.collidepoint(mouse_pos)
             return self.hovered
        return False

    def clicked(self):
        self.animation = 1.0 # Start click animation
        print(f"Button '{self.text}' clicked.")
        return True # Indicate click occurred

class DetectionAnimation: # (Unchanged)
    def __init__(self, item_name, item_type):
        self.item_name = item_name.upper()
        self.item_type = item_type.upper() # "RECYCLING" or "TRASH"
        self.start_time = time.time()
        self.phase = "dropping" # dropping -> scanning -> revealing -> feedback
        self.phase_durations = {"dropping": 1.2, "scanning": 2.0, "revealing": 1.5, "feedback": 4.0}
        self.total_duration = sum(self.phase_durations.values())
        self.stats_updated = False # Flag to ensure stats are updated only once per animation

        # Animation state variables
        self.scan_progress = 0
        self.reveal_alpha = 0
        self.particle_effects = []

        # Item appearance
        self.item_color = LEAF_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
        self.item_image = self.create_item_image() # Generate a simple procedural image
        self.y_pos = -100 # Start above screen
        self.rotation = random.uniform(-15, 15)
        self.rotation_speed = random.uniform(-2, 2)
        self.fall_speed = 8
        self.target_y = SCREEN_HEIGHT // 2 - 60 # Where item stops falling

        # Loading bar properties (for scanning phase)
        self.loading_width = 280
        self.loading_height = 18
        self.loading_x = SCREEN_WIDTH // 2 - self.loading_width // 2
        self.loading_y = self.target_y + 90 # Below the item

        # Buttons (created later, but needed for positioning prompt)
        self.correct_button = None
        self.incorrect_button = None

    def create_item_image(self):
        """Creates a simple procedural image for the detected item."""
        try:
            size = 80
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            base_color = LEAF_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
            border_color = DARK_GREEN if self.item_type == "RECYCLING" else WOOD_BROWN

            if self.item_type == "RECYCLING":
                 # Simple representation (e.g., rectangle with recycle symbol)
                 pygame.draw.rect(surf, base_color, (size*0.2, size*0.1, size*0.6, size*0.8))
                 pygame.draw.rect(surf, border_color, (size*0.2, size*0.1, size*0.6, size*0.8), 2)
                 # Basic recycle symbol lines
                 center_x, center_y = size // 2, size // 2 + 10
                 radius = size // 6
                 pygame.draw.circle(surf, WHITE, (center_x, center_y), radius, 2) # Outer circle guide (optional)
                 for i in range(3): # Draw 3 arrows (simplified)
                     angle = math.radians(i * 120 + 30) # Starting angle offset
                     start_angle = angle - math.radians(15)
                     end_angle = angle + math.radians(15)
                     p1 = (center_x + radius * math.cos(start_angle), center_y + radius * math.sin(start_angle))
                     p2 = (center_x + (radius+5) * math.cos(angle), center_y + (radius+5) * math.sin(angle)) # Arrow tip
                     p3 = (center_x + radius * math.cos(end_angle), center_y + radius * math.sin(end_angle))
                     pygame.draw.line(surf, WHITE, p1, p2, 2)
                     pygame.draw.line(surf, WHITE, p2, p3, 2)
            else: # TRASH
                # Simple representation (e.g., crumpled ball shape)
                center_x, center_y = size // 2, size // 2
                radius = size // 3
                points = []
                num_verts = random.randint(7, 11) # Irregular polygon
                for i in range(num_verts):
                    angle = math.radians(i * (360 / num_verts))
                    radius_var = radius * random.uniform(0.7, 1.3) # Vary radius
                    x = center_x + radius_var * math.cos(angle)
                    y = center_y + radius_var * math.sin(angle)
                    points.append((int(x), int(y)))
                pygame.draw.polygon(surf, base_color, points)
                pygame.draw.polygon(surf, border_color, points, 2) # Outline

            # Simple highlight
            highlight_color = (*WHITE[:3], 60)
            pygame.draw.circle(surf, highlight_color, (size // 3, size // 3), size // 5)

            return surf
        except pygame.error as e:
            print(f"Warning: Error creating item image: {e}")
            return None # Return None if creation fails

    def update(self):
        """Updates the animation state. Returns True if animation is finished."""
        if not self.item_image: return True # Finish immediately if image failed

        try:
            current_time = time.time()
            elapsed_total = current_time - self.start_time

            # --- Phase Transitions ---
            if elapsed_total < self.phase_durations["dropping"]:
                self.phase = "dropping"
                self.y_pos += self.fall_speed
                self.rotation += self.rotation_speed
            elif elapsed_total < self.phase_durations["dropping"] + self.phase_durations["scanning"]:
                self.phase = "scanning"
                # Calculate progress within scanning phase
                elapsed_in_phase = elapsed_total - self.phase_durations["dropping"]
                self.scan_progress = min(100, (elapsed_in_phase / self.phase_durations["scanning"]) * 100)
                # Stop y_pos if it reached target
                if self.y_pos < self.target_y: self.y_pos = self.target_y; self.fall_speed = 0
            elif elapsed_total < self.phase_durations["dropping"] + self.phase_durations["scanning"] + self.phase_durations["revealing"]:
                self.phase = "revealing"
                # Calculate progress within revealing phase for alpha fade-in
                elapsed_in_phase = elapsed_total - (self.phase_durations["dropping"] + self.phase_durations["scanning"])
                progress = min(1.0, elapsed_in_phase / self.phase_durations["revealing"])
                self.reveal_alpha = int(255 * (progress**0.5)) # Ease-in effect
            else:
                self.phase = "feedback"
                self.reveal_alpha = 255 # Fully revealed

            # --- Phase-Specific Logic ---
            # Trigger stats update ONCE when feedback starts
            if self.phase == "feedback" and not self.stats_updated:
                # This is where the GUI knows the animation has reached the point
                # where the item type is confirmed.
                # The actual stats update and progress bar setting happens
                # in the main interface's update_detection method.
                print(f"Animation: Reached feedback phase for {self.item_name} ({self.item_type})")
                # We set self.stats_updated=True here, but the main loop checks this flag
                # self.stats_updated = True # This flag is checked by the main interface update

            # Stop falling when target Y is reached
            if self.phase == "dropping" and self.y_pos >= self.target_y:
                self.y_pos = self.target_y
                self.fall_speed = 0
                self.rotation_speed *= 0.95 # Slow down rotation slightly

            # Add particle effects during scanning/revealing
            if self.phase == "scanning" and random.random() < 0.15:
                self.add_natural_particle("scan")
            if self.phase == "revealing" and random.random() < 0.25:
                self.add_natural_particle("reveal")

            self.update_particles() # Update all active particles

            # Check if the total animation time has elapsed
            return elapsed_total >= self.total_duration
        except (TypeError, ValueError, ZeroDivisionError) as e:
            print(f"Warning: Error updating animation: {e}")
            return True # End animation if error occurs

    def add_natural_particle(self, phase_type):
        """Adds a particle effect based on the animation phase."""
        start_x = SCREEN_WIDTH // 2
        start_y = self.y_pos + (self.item_image.get_height() // 2 if self.item_image else 0)

        if phase_type == "scan":
            # Sparks from the loading bar
            start_x = self.loading_x + (self.loading_width * self.scan_progress / 100)
            start_y = self.loading_y + self.loading_height / 2
            angle = random.uniform(math.pi * 1.2, math.pi * 1.8) # Downward arc
            speed = random.uniform(0.5, 1.5)
            ptype = "sparkle"; color = SUNSET_ORANGE; size = random.uniform(1, 3)
            vy_initial = math.sin(angle) * speed - 0.5 # Slightly upward bias initially?
        else: # reveal
            # Burst from the item center
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(1.5, 3.0)
            ptype = "leaf" if self.item_type == "RECYCLING" else "drop"
            color = LEAF_GREEN if ptype == "leaf" else WATER_BLUE
            size = random.uniform(3, 7)
            vy_initial = math.sin(angle) * speed

        self.particle_effects.append({
            "type": ptype,
            "x": start_x + random.uniform(-5, 5),
            "y": start_y + random.uniform(-5, 5),
            "vx": math.cos(angle) * speed,
            "vy": vy_initial,
            "size": size,
            "color": color,
            "rotation": random.uniform(0, 360),
            "rot_speed": random.uniform(-4, 4),
            "life": random.uniform(0.8, 1.5) # How long particle lives (seconds)
        })

    def update_particles(self):
        """Updates position, rotation, and life of particles."""
        # Iterate backwards to allow safe removal
        for i in range(len(self.particle_effects) - 1, -1, -1):
            # Safety check in case list is modified elsewhere unexpectedly
             if i >= len(self.particle_effects): continue

             p = self.particle_effects[i]
             p["x"] += p["vx"]
             p["y"] += p["vy"]
             p["vy"] += 0.05 # Simple gravity
             p["rotation"] += p["rot_speed"]
             p["life"] -= 0.015 # Decrease life (adjust rate as needed)

             if p["life"] <= 0:
                 # Ensure index is still valid before popping
                 if 0 <= i < len(self.particle_effects):
                      self.particle_effects.pop(i)


    def draw_particles(self, surface):
        """Draws active particles."""
        for p in self.particle_effects:
            try:
                # Calculate alpha based on remaining life
                life_ratio = max(0, p.get('life', 0) / 1.0) # Assuming max life around 1.0-1.5
                alpha = int(255 * min(1, life_ratio * 2)) # Fade out quickly
                if alpha <= 0: continue # Don't draw dead particles

                particle_type = p.get("type")
                size = p.get("size", 0)
                if size <= 0: continue # Skip invalid particles

                if particle_type == "leaf":
                    part_surf = pygame.Surface((size * 3, size * 2), pygame.SRCALPHA)
                    color = (*p.get("color", LEAF_GREEN)[:3], alpha)
                    pygame.draw.ellipse(part_surf, color, (0, 0, size * 2, size * 1.5)) # Simple ellipse leaf
                    rotated_part = pygame.transform.rotate(part_surf, p.get("rotation", 0))
                    part_rect = rotated_part.get_rect(center=(int(p.get("x", 0)), int(p.get("y", 0))))
                    surface.blit(rotated_part, part_rect)
                elif particle_type == "drop":
                    part_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                    color = (*p.get("color", WATER_BLUE)[:3], alpha)
                    pygame.draw.circle(part_surf, color, (int(size), int(size)), int(size))
                    part_rect = part_surf.get_rect(center=(int(p.get("x", 0)), int(p.get("y", 0))))
                    surface.blit(part_surf, part_rect)
                elif particle_type == "sparkle":
                    color = (*p.get("color", SUNSET_ORANGE)[:3], alpha)
                    center_x, center_y = int(p.get("x", 0)), int(p.get("y", 0))
                    num_lines = 5
                    rotation = p.get("rotation", 0)
                    for i in range(num_lines): # Draw lines radiating outwards
                         angle = rotation + i * (360 / num_lines)
                         rad = math.radians(angle)
                         end_x = center_x + math.cos(rad) * size * 1.5
                         end_y = center_y + math.sin(rad) * size * 1.5
                         pygame.draw.line(surface, color, (center_x, center_y), (int(end_x), int(end_y)), 1)
            except (pygame.error, KeyError, TypeError, ValueError, AttributeError) as e:
                print(f"Warning: Error drawing particle (type: {p.get('type', 'unknown')}, life: {p.get('life', '?')}, error: {e}). Skipping draw.")
                continue # Skip this particle if drawing fails

    def draw(self, surface):
        """Draws the current state of the animation."""
        if not self.item_image: return # Don't draw if image failed

        try:
            # Draw item
            rotated_image = pygame.transform.rotate(self.item_image, self.rotation)
            rotated_rect = rotated_image.get_rect(center=(SCREEN_WIDTH // 2, int(self.y_pos)))
            surface.blit(rotated_image, rotated_rect)

            # Draw phase-specific elements
            if self.phase == "scanning":
                # Draw loading bar background
                loading_bg_rect = pygame.Rect(self.loading_x, self.loading_y, self.loading_width, self.loading_height)
                pygame.draw.rect(surface, LIGHT_BROWN, loading_bg_rect) # No BR
                # Draw loading bar fill
                fill_color = LEAF_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
                fill_width = int(self.loading_width * (self.scan_progress / 100))
                if fill_width > 0:
                     fill_rect = pygame.Rect(self.loading_x, self.loading_y, fill_width, self.loading_height)
                     pygame.draw.rect(surface, fill_color, fill_rect) # No BR
                # Draw loading bar outline
                pygame.draw.rect(surface, WOOD_BROWN, loading_bg_rect, 2) # No BR
                # Draw "Analyzing..." text
                text = font_medium.render("Analyzing...", True, TEXT_BROWN)
                text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, self.loading_y - 25))
                surface.blit(text, text_rect)

            elif self.phase in ["revealing", "feedback"]:
                # --- Draw Item Name and Type (Fade In) ---
                text_center_x = SCREEN_WIDTH // 2
                text_base_y = self.y_pos + rotated_rect.height // 2 + 20 # Below the item

                # Item Name
                name_surf = font_large.render(self.item_name, True, TEXT_BROWN)
                name_surf.set_alpha(self.reveal_alpha) # Apply fade-in alpha
                name_rect = name_surf.get_rect(center=(text_center_x, text_base_y + 20))
                # Background for name
                name_bg_rect = name_rect.inflate(20, 10)
                name_bg_surf = pygame.Surface(name_bg_rect.size, pygame.SRCALPHA)
                name_bg_surf.fill((*CREAM[:3], int(180 * (self.reveal_alpha / 255)))) # Background fades in too
                pygame.draw.rect(name_bg_surf, (*WOOD_BROWN[:3], int(200 * (self.reveal_alpha / 255))), name_bg_surf.get_rect(), 2) # No BR
                surface.blit(name_bg_surf, name_bg_rect.topleft)
                surface.blit(name_surf, name_rect)

                # Item Type (Recyclable/Landfill)
                type_color = DARK_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
                type_text = "Recyclable" if self.item_type == "RECYCLING" else "Landfill"
                type_surf = font_medium.render(type_text, True, type_color)
                type_surf.set_alpha(self.reveal_alpha) # Apply fade-in alpha
                type_rect = type_surf.get_rect(center=(text_center_x, text_base_y + 60))
                # Background for type
                type_bg_rect = type_rect.inflate(15, 8)
                type_bg_surf = pygame.Surface(type_bg_rect.size, pygame.SRCALPHA)
                type_bg_surf.fill((*CREAM[:3], int(160 * (self.reveal_alpha / 255)))) # Background fades in too
                pygame.draw.rect(type_bg_surf, (*WOOD_BROWN[:3], int(180 * (self.reveal_alpha / 255))), type_bg_surf.get_rect(), 1) # No BR
                surface.blit(type_bg_surf, type_bg_rect.topleft)
                surface.blit(type_surf, type_rect)

                # --- Draw Feedback Prompt (Only during feedback phase) ---
                if self.phase == "feedback":
                    if self.correct_button is None or self.incorrect_button is None:
                        print("Warning: Buttons not set in DetectionAnimation for positioning feedback prompt.")
                        prompt_center_y = SCREEN_HEIGHT - 150 # Fallback position
                    else:
                        # Position prompt relative to buttons
                        prompt_y_offset = 40 # Space above buttons
                        # Use the top of the correct_button (assuming they are aligned)
                        prompt_center_y = self.correct_button.rect.top - prompt_y_offset

                    time_left = max(0, self.total_duration - (time.time() - self.start_time))
                    prompt_text = f"Is this correct? ({int(time_left)}s)"
                    prompt_surf = font_medium.render(prompt_text, True, TEXT_BROWN)
                    prompt_rect = prompt_surf.get_rect(center=(SCREEN_WIDTH // 2, prompt_center_y))

                    # Background for prompt
                    prompt_bg_rect = prompt_rect.inflate(20, 10)
                    prompt_bg_surf = pygame.Surface(prompt_bg_rect.size, pygame.SRCALPHA)
                    prompt_bg_surf.fill((*LIGHT_CREAM[:3], 190)) # Semi-transparent background
                    pygame.draw.rect(prompt_bg_surf, WOOD_BROWN, prompt_bg_surf.get_rect(), 2) # No BR
                    surface.blit(prompt_bg_surf, prompt_bg_rect.topleft)
                    surface.blit(prompt_surf, prompt_rect)

            # Draw particles on top
            self.draw_particles(surface)
        except (pygame.error, TypeError, ValueError, ZeroDivisionError, AttributeError) as e:
             print(f"Warning: Error drawing detection animation phase '{self.phase}': {e}")
             # Attempt to recover or end animation gracefully
             self.phase = "feedback" # Force to end state
             self.start_time = time.time() - self.total_duration # Make it seem like duration elapsed

class SmartBinInterface:
    def __init__(self):
        self.stats = BinStats()
        self.nature_elements = [NatureElement() for _ in range(15)] # Adjust number as needed
        self.state = "idle" # "idle", "detecting"
        self.detection_animation = None

        # --- Calculate sizes based on fullscreen dimensions ---
        pb_rel_height = 0.55; pb_rel_width = 0.2
        pb_height = int(SCREEN_HEIGHT * pb_rel_height); pb_width = int(SCREEN_WIDTH * pb_rel_width)
        pb_y_offset = 40 # Move bars down slightly from true center
        pb_y = (SCREEN_HEIGHT // 2) - (pb_height // 2) + pb_y_offset
        pb_x_margin = int(SCREEN_WIDTH * 0.08) # Gap from center

        # --- SWAPPED POSITIONS ---
        landfill_pb_x = (SCREEN_WIDTH // 2) - pb_x_margin - pb_width  # Landfill on Left
        recycling_pb_x = (SCREEN_WIDTH // 2) + pb_x_margin           # Recycling on Right

        # --- Instantiate progress bars with SWAPPED positions, styles, and colors ---
        self.landfill_progress = NaturalProgressBar(landfill_pb_x, pb_y, pb_width, pb_height, WATER_BLUE, LIGHT_CREAM, style="water")
        self.recycling_progress = NaturalProgressBar(recycling_pb_x, pb_y, pb_width, pb_height, LEAF_GREEN, LIGHT_CREAM, style="plant")
        self.recycling_progress.set_value(0) # Start empty
        self.landfill_progress.set_value(0) # Start empty

        # --- Button positioning (relative to bottom center) ---
        button_width = int(SCREEN_WIDTH * 0.12); button_height = int(SCREEN_HEIGHT * 0.07)
        button_y = SCREEN_HEIGHT - button_height - int(SCREEN_HEIGHT * 0.08) # Y position from bottom
        button_gap = int(SCREEN_WIDTH * 0.02)

        self.correct_button = EcoButton(SCREEN_WIDTH // 2 - button_width - button_gap // 2, button_y, button_width, button_height, "Correct", LEAF_GREEN, LIGHT_GREEN)
        self.incorrect_button = EcoButton(SCREEN_WIDTH // 2 + button_gap // 2, button_y, button_width, button_height, "Incorrect", SOFT_BROWN, LIGHT_BROWN)

        # --- Hint setup ---
        self.last_hint_time = 0; self.hint_interval = 25 # Time between hints (idle state)
        self.hints = ["SmartBin™ uses AI to sort waste accurately.", "Recycling reduces landfill waste and saves resources.", "Contamination ruins recycling - please sort carefully!", "Over 95% accuracy in waste sorting!", "Thank you for helping keep the planet green!", "Plastic bottles take over 400 years to decompose.", "Glass is infinitely recyclable!", "Recycle paper 5-7 times before fibers weaken.", "Check local guidelines for specific recycling rules.", "Reduce, Reuse, Recycle - in that order!"]
        self.current_hint = random.choice(self.hints); self.hint_alpha = 0; self.hint_fade_in = True; self.hint_display_duration = 8; self.hint_fade_duration = 1.5; self.hint_state = "fading_in"; self.hint_visible_start_time = 0

    def update_nature_elements(self): # (Unchanged)
        for element in self.nature_elements: element.update()
    def draw_nature_elements(self, surface): # (Unchanged)
        for element in self.nature_elements: element.draw(surface)
    def update_progress_bars(self): # (Unchanged)
        # The update logic itself doesn't care about position
        self.recycling_progress.update()
        self.landfill_progress.update()
    def draw_progress_bars(self, surface): # (Drawing is Swapped, see method below)
        try:
            # --- Define Padding ---
            card_padding_x = int(SCREEN_WIDTH * 0.02) # Horizontal padding inside card
            card_padding_y_top = int(SCREEN_HEIGHT * 0.08) # Top padding inside card (for label)
            card_padding_y_bottom = int(SCREEN_HEIGHT * 0.12) # Bottom padding inside card (for counts/stats)

            # --- Calculate Card Rectangles (SWAPPED) ---
            # Landfill Card (Left)
            landfill_card_rect = pygame.Rect(
                self.landfill_progress.x - card_padding_x,
                self.landfill_progress.y - card_padding_y_top,
                self.landfill_progress.width + 2 * card_padding_x,
                self.landfill_progress.height + card_padding_y_top + card_padding_y_bottom
            )
            # Recycling Card (Right)
            recycling_card_rect = pygame.Rect(
                self.recycling_progress.x - card_padding_x,
                self.recycling_progress.y - card_padding_y_top,
                self.recycling_progress.width + 2 * card_padding_x,
                self.recycling_progress.height + card_padding_y_top + card_padding_y_bottom
            )

            # --- Draw Cards ---
            pygame.draw.rect(surface, CREAM, landfill_card_rect) # No border radius needed
            pygame.draw.rect(surface, WOOD_BROWN, landfill_card_rect, 2) # Outline
            pygame.draw.rect(surface, CREAM, recycling_card_rect) # No border radius needed
            pygame.draw.rect(surface, WOOD_BROWN, recycling_card_rect, 2) # Outline

            # --- Draw Progress Bars (SWAPPED ORDER) ---
            self.landfill_progress.draw(surface)    # Draw landfill bar (on the left)
            self.recycling_progress.draw(surface)   # Draw recycling bar (on the right)

            # --- Draw Labels (SWAPPED) ---
            # Landfill Label (Above left bar)
            landfill_label = font_medium.render("Landfill", True, TEXT_BROWN)
            landfill_rect = landfill_label.get_rect(center=(self.landfill_progress.x + self.landfill_progress.width / 2, self.landfill_progress.y - card_padding_y_top / 2))
            surface.blit(landfill_label, landfill_rect)
            # Recycling Label (Above right bar)
            recycling_label = font_medium.render("Recycling", True, DARK_GREEN)
            recycling_rect = recycling_label.get_rect(center=(self.recycling_progress.x + self.recycling_progress.width / 2, self.recycling_progress.y - card_padding_y_top / 2))
            surface.blit(recycling_label, recycling_rect)

            # --- Draw Item Counts (SWAPPED) ---
            # Y position for counts (below the bars)
            count_y = self.landfill_progress.y + self.landfill_progress.height + card_padding_y_bottom / 2
            # Landfill Count (Below left bar)
            landfill_count_text = f"{self.stats.landfill_items} items"
            landfill_count_surf = font_small.render(landfill_count_text, True, TEXT_BROWN)
            landfill_count_rect = landfill_count_surf.get_rect(center=(landfill_rect.centerx, count_y))
            surface.blit(landfill_count_surf, landfill_count_rect)
            # Recycling Count (Below right bar)
            recycling_count_text = f"{self.stats.recycled_items} items"
            recycling_count_surf = font_small.render(recycling_count_text, True, DARK_GREEN)
            recycling_count_rect = recycling_count_surf.get_rect(center=(recycling_rect.centerx, count_y))
            surface.blit(recycling_count_surf, recycling_count_rect)

            # --- Draw Environmental Impact Section (Centered at Bottom - Unchanged) ---
            if self.stats.total_items > 0:
                 impact_card_height = int(SCREEN_HEIGHT * 0.14)
                 impact_card_width = int(SCREEN_WIDTH * 0.45)
                 impact_card_y = SCREEN_HEIGHT - impact_card_height - int(SCREEN_HEIGHT * 0.02)
                 impact_card_x = SCREEN_WIDTH // 2 - impact_card_width // 2
                 impact_card = pygame.Rect(impact_card_x, impact_card_y, impact_card_width, impact_card_height)

                 pygame.draw.rect(surface, CREAM, impact_card) # No border radius
                 pygame.draw.rect(surface, WOOD_BROWN, impact_card, 2) # Outline

                 # Title
                 impact_title = font_medium.render("Environmental Impact", True, TEXT_BROWN)
                 impact_title_rect = impact_title.get_rect(center=(impact_card.centerx, impact_card.y + impact_card_height * 0.25))
                 surface.blit(impact_title, impact_title_rect)

                 # Stats Text and Icons
                 stats_y = impact_card.y + impact_card_height * 0.68 # Y position for stats text
                 icon_size = int(SCREEN_HEIGHT * 0.03) # Size for icons

                 # CO2 Saved (Left side)
                 co2_text = f"{self.stats.co2_saved:.1f} kg CO₂"
                 co2_surf = font_small.render(co2_text, True, DARK_GREEN)
                 co2_rect = co2_surf.get_rect(midright=(impact_card.centerx - int(impact_card_width * 0.05), stats_y))
                 surface.blit(co2_surf, co2_rect)
                 # Simple leaf icon
                 leaf_icon_rect = pygame.Rect(0, 0, icon_size, icon_size * 0.8)
                 leaf_icon_rect.midright = (co2_rect.left - int(impact_card_width * 0.02), stats_y)
                 pygame.draw.ellipse(surface, LEAF_GREEN, leaf_icon_rect)

                 # Water Saved (Right side)
                 water_text = f"{self.stats.water_saved:.1f} L Water"
                 water_surf = font_small.render(water_text, True, WATER_BLUE)
                 water_rect = water_surf.get_rect(midleft=(impact_card.centerx + int(impact_card_width * 0.05), stats_y))
                 surface.blit(water_surf, water_rect)
                 # Simple drop icon
                 drop_icon_rect = pygame.Rect(0, 0, icon_size * 0.7, icon_size)
                 drop_icon_rect.midleft = (water_rect.right + int(impact_card_width * 0.02), stats_y)
                 pygame.draw.ellipse(surface, WATER_BLUE, drop_icon_rect) # Use ellipse for drop shape
        except (pygame.error, TypeError, ValueError) as e:
            print(f"Warning: Error drawing progress bars/stats: {e}")
            # Fallback: Draw a simple box indicating error area
            pygame.draw.rect(surface, CREAM, (50, 50, SCREEN_WIDTH-100, SCREEN_HEIGHT-100), 5)
            error_text = font_small.render("Error drawing stats", True, BLACK)
            screen.blit(error_text, (60, 60))
    def update_hint(self): # (Unchanged)
        current_time = time.time()
        time_since_change = current_time - self.last_hint_time

        if self.hint_state == "fading_in":
            if time_since_change < self.hint_fade_duration:
                # Calculate alpha based on fade duration
                self.hint_alpha = int(255 * (time_since_change / self.hint_fade_duration))
            else:
                # Fade in complete
                self.hint_alpha = 255
                self.hint_state = "visible"
                self.hint_visible_start_time = current_time # Record when it became fully visible
        elif self.hint_state == "visible":
            time_visible = current_time - self.hint_visible_start_time
            if time_visible >= self.hint_display_duration:
                # Time to start fading out
                self.hint_state = "fading_out"
                self.last_hint_time = current_time # Reset timer for fade out
        elif self.hint_state == "fading_out":
            time_fading = current_time - self.last_hint_time
            if time_fading < self.hint_fade_duration:
                # Calculate alpha based on fade out progress
                self.hint_alpha = int(255 * (1 - (time_fading / self.hint_fade_duration)))
            else:
                # Fade out complete, prepare for next hint
                self.hint_alpha = 0
                self.current_hint = random.choice(self.hints) # Choose a new hint
                self.hint_state = "fading_in"
                self.last_hint_time = current_time # Reset timer for next fade in

        # Ensure alpha stays within bounds
        self.hint_alpha = max(0, min(255, self.hint_alpha))
    def draw_hint(self, surface): # (Unchanged)
        if self.hint_alpha <= 0: return # Don't draw if fully faded out

        try:
            hint_surf = font_small.render(self.current_hint, True, TEXT_BROWN)
            # Center hint horizontally, position near bottom vertically
            hint_rect = hint_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - int(SCREEN_HEIGHT*0.08)))

            # Create a background surface for the hint with alpha
            bg_rect = hint_rect.inflate(int(SCREEN_WIDTH*0.04), int(SCREEN_HEIGHT*0.03)) # Add padding
            hint_back = pygame.Surface(bg_rect.size, pygame.SRCALPHA) # Surface supports alpha
            # Calculate background color alpha based on hint alpha
            bg_color_alpha = (*CREAM[:3], int(200 * (self.hint_alpha / 255)))
            pygame.draw.rect(hint_back, bg_color_alpha, hint_back.get_rect()) # No border radius

            # Draw border with corresponding alpha
            border_color_alpha = (*WOOD_BROWN[:3], int(220 * (self.hint_alpha / 255)))
            pygame.draw.rect(hint_back, border_color_alpha, hint_back.get_rect(), 2) # No border radius

            # Blit background and then the text (with its own alpha)
            surface.blit(hint_back, bg_rect.topleft)
            hint_surf.set_alpha(self.hint_alpha) # Apply alpha to the text itself
            surface.blit(hint_surf, hint_rect)
        except pygame.error as e:
            print(f"Warning: Error drawing hint: {e}")
    def process_camera_detection(self, detection_data): # (Unchanged)
        """Starts the detection animation when data is received."""
        if self.state == "idle" and isinstance(detection_data, dict):
            item_name = detection_data.get("name", "Unknown Item")
            item_type = detection_data.get("type", "TRASH") # Default to TRASH if type is missing

            # Validate item_type
            if item_type.upper() not in ["RECYCLING", "TRASH"]:
                print(f"Warning: Received invalid item type '{item_type}'. Defaulting to TRASH.")
                item_type = "TRASH"

            print(f"GUI: Received detection - Item: {item_name}, Type: {item_type}")
            self.detection_animation = DetectionAnimation(item_name, item_type)
            # Pass button instances to animation for positioning feedback prompt
            self.detection_animation.correct_button = self.correct_button
            self.detection_animation.incorrect_button = self.incorrect_button
            self.state = "detecting"
        elif not isinstance(detection_data, dict):
            print(f"Warning: Received invalid detection data format: {type(detection_data)}")

    # <<< --- START OF MODIFIED update_detection --- >>>
    def update_detection(self):
        """Checks queue for new detections and updates animation/state."""
        try:
            # Check for new detections only when idle
            if self.state == "idle":
                 if not detection_queue.empty():
                     detection_data = detection_queue.get()
                     self.process_camera_detection(detection_data)
                     detection_queue.task_done() # Mark task as done

            # Update active animation
            if self.detection_animation:
                animation_finished = self.detection_animation.update()

                # --- NEW LOGIC: Update stats & progress bars when feedback starts ---
                # Check the animation's flag, not its phase directly
                # self.detection_animation.stats_updated is set inside its update method
                # when the feedback phase *begins*. We check it here.
                if self.detection_animation.phase == "feedback" and not self.detection_animation.stats_updated:
                    print(f"GUI: Feedback phase started for {self.detection_animation.item_name}.")
                    # 1. Update the core statistics
                    self.stats.update_stats(self.detection_animation.item_type)
                    print(f"GUI: BinStats updated - Total: {self.stats.total_items}, Rec: {self.stats.recycled_items}, Land: {self.stats.landfill_items}")

                    # 2. Calculate percentages based on updated counts
                    total_items = self.stats.total_items
                    if total_items > 0:
                        # Use the specific percentage methods from BinStats
                        recycling_percentage = self.stats.get_recycling_percentage()
                        landfill_percentage = self.stats.get_landfill_percentage()

                        # 3. Set the target values for EACH progress bar individually
                        self.recycling_progress.set_value(recycling_percentage)
                        self.landfill_progress.set_value(landfill_percentage)

                        print(f"GUI: Progress bars updated - Rec: {recycling_percentage:.1f}%, Land: {landfill_percentage:.1f}%")
                    else:
                        # Reset bars if total becomes zero (e.g., after incorrect feedback resets stats?)
                        self.recycling_progress.set_value(0)
                        self.landfill_progress.set_value(0)
                        print("GUI: Total items is zero, resetting progress bars.")

                    # 4. Mark stats as updated *for this specific animation instance*
                    self.detection_animation.stats_updated = True
                # --- END OF NEW LOGIC ---

                # Check if the entire animation sequence (including feedback) is finished
                if animation_finished:
                    print("GUI: Detection animation finished.")
                    self.detection_animation = None # Clear the animation object
                    self.state = "idle" # Return to idle state
        except queue.Empty:
            pass # It's normal for the queue to be empty
        except Exception as e:
            print(f"Error in update_detection: {e}")
            traceback.print_exc() # Print detailed error stack
            # Reset state in case of error during animation update
            self.detection_animation = None
            self.state = "idle"
    # <<< --- END OF MODIFIED update_detection --- >>>

    def draw_detection(self, surface): # (Unchanged)
        """Draws the detection animation overlay and elements."""
        try:
            if self.detection_animation:
                # Draw semi-transparent overlay
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay_alpha = 100 # Base alpha
                # Adjust alpha based on phase for effect
                if self.detection_animation.phase == "dropping":
                     # Fade in overlay as item drops
                     overlay_alpha = int(100 * (self.detection_animation.y_pos / self.detection_animation.target_y)) if self.detection_animation.target_y > 0 else 0
                elif self.detection_animation.phase == "scanning":
                     overlay_alpha = 100 + int(50 * (self.detection_animation.scan_progress / 100)) # Darken slightly during scan
                elif self.detection_animation.phase == "revealing":
                     overlay_alpha = 150 + int(105 * (self.detection_animation.reveal_alpha / 255)) # Darken more as result revealed
                elif self.detection_animation.phase == "feedback":
                     overlay_alpha = 200 # Darkest during feedback

                overlay.fill((0, 0, 0, max(0, min(200, overlay_alpha)))) # Black overlay with calculated alpha
                surface.blit(overlay, (0, 0))

                # Draw the animation itself (item, text, particles, buttons)
                self.detection_animation.draw(surface)

                # Draw buttons specifically during the feedback phase
                # (They are drawn *by* the detection_animation.draw method now)
                # if self.detection_animation.phase == "feedback":
                #     self.correct_button.draw(surface)
                #     self.incorrect_button.draw(surface)
        except (pygame.error, AttributeError) as e:
            print(f"Warning: Error drawing detection overlay/animation: {e}")

    def handle_button_clicks(self, mouse_pos): # (Unchanged)
        """Handles clicks on the feedback buttons."""
        clicked = False
        # Only handle clicks during the feedback phase of an active animation
        if self.state == "detecting" and self.detection_animation and self.detection_animation.phase == "feedback":
            # Check which button was hovered (already updated by check_button_hover)
            # The clicked() method provides visual feedback and returns True
            if self.correct_button.hovered and self.correct_button.clicked():
                 print("Feedback: Correct button clicked.")
                 # Potentially send 'correct' feedback somewhere? Not implemented.
                 clicked = True
            elif self.incorrect_button.hovered and self.incorrect_button.clicked():
                 print("Feedback: Incorrect button clicked.")
                 # Potentially send 'incorrect' feedback or adjust stats? Not implemented.
                 clicked = True

            if clicked:
                # End the animation and return to idle state immediately after feedback
                self.detection_animation = None
                self.state = "idle"
                # Optional short delay to prevent immediate re-triggering if needed
                # pygame.time.wait(150)
        return clicked # Indicate if a feedback button was processed

    def check_button_hover(self, mouse_pos): # (Unchanged)
        """Updates the hover state of feedback buttons."""
        # Only check hover if in the correct state/phase
        if self.state == "detecting" and self.detection_animation and self.detection_animation.phase == "feedback":
            try:
                self.correct_button.check_hover(mouse_pos)
                self.incorrect_button.check_hover(mouse_pos)
            except (AttributeError, TypeError) as e:
                 # Handle potential errors if buttons aren't fully initialized somehow
                 print(f"Warning: Error checking button hover state: {e}")
        else:
            # Ensure buttons are not marked as hovered when not active
            self.correct_button.hovered = False
            self.incorrect_button.hovered = False


# --- Helper Functions --- (Unchanged)
def draw_background(surface):
    """Draws a gradient background."""
    try:
        # Simple vertical gradient
        for y in range(SCREEN_HEIGHT):
            ratio = y / SCREEN_HEIGHT
            # Interpolate between light cream and cream
            r = int(LIGHT_CREAM[0] * (1 - ratio) + CREAM[0] * ratio)
            g = int(LIGHT_CREAM[1] * (1 - ratio) + CREAM[1] * ratio)
            b = int(LIGHT_CREAM[2] * (1 - ratio) + CREAM[2] * ratio)
            pygame.draw.line(surface, (r, g, b), (0, y), (SCREEN_WIDTH, y))
    except pygame.error as e:
        print(f"Warning: Error drawing background: {e}")
        surface.fill(CREAM) # Fallback to solid color

def draw_header(surface):
    """Draws the header bar with title and decorative elements."""
    try:
        header_height = int(SCREEN_HEIGHT * 0.12) # Relative header height
        header_rect = pygame.Rect(0, 0, SCREEN_WIDTH, header_height)

        # Draw base header color
        pygame.draw.rect(surface, CREAM, header_rect)

        # Add subtle wood grain effect
        for y in range(0, header_height, 3):
            grain_width = random.randint(int(SCREEN_WIDTH * 0.8), SCREEN_WIDTH)
            grain_x = random.randint(0, int(SCREEN_WIDTH * 0.2))
            grain_alpha = random.randint(8, 18) # Very subtle
            grain_line = pygame.Surface((grain_width, 1), pygame.SRCALPHA)
            grain_line.fill((*SOFT_BROWN[:3], grain_alpha))
            surface.blit(grain_line, (grain_x, y))

        # Bottom border line
        pygame.draw.rect(surface, WOOD_BROWN, (0, header_height - 2, SCREEN_WIDTH, 2))

        # Draw Title
        title_text = font_title.render("SmartBin™ Waste Management", True, TEXT_BROWN)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, header_height//2))
        surface.blit(title_text, title_rect)

        # Decorative leaves
        leaf_size = int(header_height * 0.3)
        leaf_surf = pygame.Surface((leaf_size * 1.5, leaf_size), pygame.SRCALPHA)
        pygame.draw.ellipse(leaf_surf, LEAF_GREEN, (0, 0, leaf_size * 1.5, leaf_size))
        pygame.draw.line(leaf_surf, DARK_GREEN, (leaf_size * 0.5, leaf_size / 2), (leaf_size * 1.5, leaf_size / 2), 2) # Simple vein

        leaf_left_rect = leaf_surf.get_rect(center=(title_rect.left - int(SCREEN_WIDTH*0.05), title_rect.centery))
        surface.blit(leaf_surf, leaf_left_rect)
        # Flip leaf for the right side
        leaf_right_surf = pygame.transform.flip(leaf_surf, True, False)
        leaf_right_rect = leaf_right_surf.get_rect(center=(title_rect.right + int(SCREEN_WIDTH*0.05), title_rect.centery))
        surface.blit(leaf_right_surf, leaf_right_rect)

    except pygame.error as e:
        print(f"Warning: Error drawing header: {e}")
        # Fallback simple header
        pygame.draw.rect(surface, CREAM, (0, 0, SCREEN_WIDTH, 65))
        pygame.draw.line(surface, WOOD_BROWN, (0, 63), (SCREEN_WIDTH, 63), 2)

def simulate_detection(queue): # (Unchanged)
    """Puts a random simulated detection event into the queue."""
    items = [
        {"name": "Plastic Bottle", "type": "RECYCLING"},
        {"name": "Apple Core", "type": "TRASH"},
        {"name": "Aluminum Can", "type": "RECYCLING"},
        {"name": "Paper Towel", "type": "TRASH"},
        {"name": "Newspaper", "type": "RECYCLING"},
        {"name": "Coffee Cup", "type": "TRASH"}, # Often TRASH due to lining
        {"name": "Glass Jar", "type": "RECYCLING"}
    ]
    chosen_item = random.choice(items)
    print(f"\n--- SIMULATING DETECTION: {chosen_item['name']} ({chosen_item['type']}) ---")
    queue.put(chosen_item)

# --- Server Thread Function --- (Unchanged)
def gui_server_thread(host, port, data_queue, running_flag_func):
    """Listens for incoming TCP connections and puts received JSON data into the queue."""
    # Imports moved inside for thread safety/clarity if needed, though likely fine here
    # import socket, json, time
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow address reuse quickly after server restart
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((host, port))
        server_socket.listen(1) # Listen for one connection at a time
        print(f"[Server Thread] Listening on {host}:{port}")
        server_socket.settimeout(1.0) # Non-blocking check for shutdown flag

        while running_flag_func(): # Check if main loop wants us to stop
            try:
                conn, addr = server_socket.accept()
                with conn:
                    print(f"[Server Thread] Accepted connection from {addr}")
                    data = b""
                    conn.settimeout(2.0) # Timeout for receiving data
                    while True:
                        try:
                            chunk = conn.recv(1024) # Read up to 1KB
                            if not chunk: break # Connection closed by client
                            data += chunk
                            # Basic check if a potential JSON object might be complete
                            # This isn't foolproof but prevents infinite loops on partial data
                            if data.strip().endswith(b'}'):
                                break
                        except socket.timeout:
                             print("[Server Thread] Receive timeout.")
                             break # Stop receiving if timeout occurs
                    # Process received data
                    if data:
                        try:
                            message_str = data.decode('utf-8').strip()
                            # Handle multiple JSON objects sent in one go (split by '}')
                            potential_jsons = message_str.split('}')
                            processed_count = 0
                            for part in potential_jsons:
                                if not part.strip(): continue # Skip empty parts
                                json_str = part + '}' # Re-add the closing brace
                                try:
                                    detection_data = json.loads(json_str)
                                    # Basic validation of received data structure
                                    if (isinstance(detection_data, dict) and
                                        "type" in detection_data and
                                        "name" in detection_data):
                                        data_queue.put(detection_data)
                                        processed_count += 1
                                        print(f"[Server Thread] Valid data added to queue: {detection_data}")
                                    else:
                                        print(f"[Server Thread] Warning: Parsed data part is not in expected format: {json_str}")
                                except json.JSONDecodeError as e:
                                     # Ignore decode errors on partial JSONs, might be okay if next part completes it
                                     if part != potential_jsons[-1]: # Check if it's not the last part
                                          pass # Likely an incomplete JSON segment
                                     else:
                                          print(f"[Server Thread] Error decoding JSON part: {e} - Part: {json_str}")
                            # if processed_count == 0 and message_str:
                            #      print("[Server Thread] No valid JSON objects found in received data.")

                        except UnicodeDecodeError as e:
                            print(f"[Server Thread] Error decoding UTF-8: {e}")
                        except Exception as e:
                            print(f"[Server Thread] Error processing received data: {e}")
                    else:
                        print("[Server Thread] No data received or connection closed early.")
            except socket.timeout:
                continue # No connection attempt, loop again to check running_flag
            except Exception as e:
                print(f"[Server Thread] Error accepting connection: {e}")
                time.sleep(0.5) # Short pause before retrying accept
    except OSError as e:
         # Handle specific errors like "address already in use"
         print(f"[Server Thread] CRITICAL OS Error binding socket (e.g., address in use?): {e}")
         # Consider signaling main thread to exit?
    except Exception as e:
        print(f"[Server Thread] Error setting up server socket: {e}")
    finally:
        print("[Server Thread] Shutting down...")
        server_socket.close()
        print("[Server Thread] Socket closed.")


# --- Main Function --- (Unchanged)
def main():
    global server_running
    server_thread = None
    try:
        interface = SmartBinInterface()
        clock = pygame.time.Clock()
        running = True

        # Start the server thread
        # Pass the lambda function to check the global server_running flag
        server_thread = threading.Thread(target=gui_server_thread,
                                         args=(GUI_SERVER_HOST, GUI_SERVER_PORT, detection_queue, lambda: server_running),
                                         daemon=True) # Daemon allows main thread to exit even if server hangs
        server_thread.start()
        print("GUI Server thread started.")

        print("-" * 30)
        print("SmartBin GUI Initialized.")
        print(f"Listening for detections on {GUI_SERVER_HOST}:{GUI_SERVER_PORT}")
        print("Press 'D' to simulate a detection event.")
        print("Press 'ESC' to quit.")
        print("-" * 30)

        last_update_time = time.time() # For delta time calculation (optional)

        while running:
            mouse_pos = pygame.mouse.get_pos() # Get mouse position for hover checks

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    server_running = False # Signal server thread to stop
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        server_running = False # Signal server thread to stop
                    if event.key == pygame.K_d: # Simulate detection
                         if interface.state == "idle":
                             simulate_detection(detection_queue)
                         else:
                             print("Simulation ignored: Detection already in progress.")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                     if event.button == 1: # Left mouse button
                         interface.handle_button_clicks(event.pos) # Pass click position

            # --- Update Game State ---
            current_time = time.time()
            delta_time = current_time - last_update_time # Optional: for frame-rate independent physics/animation
            last_update_time = current_time

            interface.check_button_hover(mouse_pos) # Update button hover states
            interface.update_nature_elements()    # Update background animations
            interface.update_progress_bars()      # Update visual fill animation
            interface.update_hint()               # Update hint fade/text
            interface.update_detection()          # Check queue, update animation/state, update stats & bar values

            # --- Drawing ---
            try:
                draw_background(screen)             # Draw gradient background
                interface.draw_nature_elements(screen) # Draw leaves/drops
                draw_header(screen)                 # Draw top title bar

                # Draw main content based on state
                if interface.state == "idle":
                    interface.draw_progress_bars(screen) # Draw the two main progress bars
                    # Draw hint only if buttons aren't animating from a recent click
                    correct_animating = hasattr(interface.correct_button, 'animation') and interface.correct_button.animation > 0;
                    incorrect_animating = hasattr(interface.incorrect_button, 'animation') and interface.incorrect_button.animation > 0
                    if not (correct_animating or incorrect_animating):
                         interface.draw_hint(screen)
                else: # state is "detecting"
                    interface.draw_detection(screen) # Draw the overlay and animation

            except Exception as draw_err:
                # Catch errors during the drawing phase to prevent crash
                print(f"CRITICAL error during drawing phase: {draw_err}")
                traceback.print_exc()
                # Try to display a simple error message on screen
                screen.fill(CREAM)
                try:
                    err_font = pygame.font.SysFont("Arial", 20)
                    err_text = err_font.render("Drawing Error Occurred - Check Console", True, BLACK)
                    screen.blit(err_text, (50, 50))
                except Exception as font_err:
                    # If even rendering basic text fails, log it
                    print(f"Could not render error text to screen: {font_err}")

            pygame.display.flip() # Update the full screen
            clock.tick(60) # Limit frame rate to 60 FPS

    except Exception as main_err:
        print(f"CRITICAL error in main loop: {main_err}")
        traceback.print_exc() # Print detailed stack trace
    finally:
        # --- Cleanup ---
        print("Shutting down Pygame and Server...")
        server_running = False # Ensure flag is false
        if server_thread and server_thread.is_alive():
            print("Waiting for server thread to join...")
            server_thread.join(timeout=2.0) # Wait max 2 seconds for thread to finish
            if server_thread.is_alive():
                 print("Warning: Server thread did not join cleanly.")
        else:
             print("Server thread already stopped or not started.")

        pygame.quit()
        print("Application terminated.")
        sys.exit()

# --- Script Entry Point ---
if __name__ == "__main__":
    main()