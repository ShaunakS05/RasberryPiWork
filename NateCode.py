import pygame
import sys
import random
import math
import time
import threading
import queue
# --- Removed OpenCV, numpy, base64, os, serial, OpenAI ---
# These belong in the separate detection script

# --- Pygame Initialization ---
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 450
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("SmartBin™ Waste Management System")

# Define colors - natural, earthy sustainability palette
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CREAM = (248, 246, 235)
LIGHT_CREAM = (252, 250, 245)
LEAF_GREEN = (105, 162, 76)
LIGHT_GREEN = (183, 223, 177)
DARK_GREEN = (65, 122, 36)
SOFT_BROWN = (139, 98, 65)
LIGHT_BROWN = (188, 152, 106)
WATER_BLUE = (99, 171, 190)
LIGHT_BLUE = (175, 219, 230)
SUNSET_ORANGE = (233, 127, 2)
WOOD_BROWN = (160, 120, 85)
TEXT_BROWN = (90, 65, 40)

# Fonts
font_title = pygame.font.SysFont("Arial", 36, bold=True)
font_large = pygame.font.SysFont("Arial", 30, bold=True)
font_medium = pygame.font.SysFont("Arial", 22)
font_small = pygame.font.SysFont("Arial", 18)
font_tiny = pygame.font.SysFont("Arial", 16)

# --- Global Variables for Communication Between Threads ---
detection_queue = queue.Queue()  # For receiving detection results from the *external* camera process
# --- Removed camera_active and ser ---

# --- Removed Camera Thread Functions ---
# (encode_image, ask_chatgpt, initialize_serial, process_capture, camera_detection_thread, backSub)
# These functions belong in your separate detection script (raspberry_pi.py)

# --- Pygame GUI Classes ---

# Smart bin statistics (Unchanged)
class BinStats:
    def __init__(self):
        self.recycled_items = 0
        self.landfill_items = 0
        self.total_items = 0
        # CO2/Water saving logic might need refinement or actual data sources
        self.co2_saved = 0
        self.water_saved = 0
        self.last_updated = time.time()

    def update_stats(self, bin_type):
        self.total_items += 1
        if bin_type.upper() == "RECYCLING":
            self.recycled_items += 1
            # Use more realistic or configurable values
            self.co2_saved += random.uniform(0.1, 0.3) # Example: kg CO2 per item
            self.water_saved += random.uniform(0.5, 2.0) # Example: Liters water per item
        elif bin_type.upper() == "TRASH":
            self.landfill_items += 1
        self.last_updated = time.time()

    def get_recycling_percentage(self):
        if self.total_items == 0:
            return 0
        return (self.recycled_items / self.total_items) * 100

# Nature-inspired animated elements (leaves, water drops) (Unchanged, mostly graphical)
class NatureElement:
    def __init__(self):
        self.reset()
        self.y = random.randint(0, SCREEN_HEIGHT)

    def reset(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = -10
        self.type = random.choice(["leaf", "drop"])
        self.size = random.randint(3, 8) if self.type == "leaf" else random.randint(2, 5)
        self.speed = random.uniform(0.3, 1.0)
        self.drift = random.uniform(-0.3, 0.3)
        self.rotation = random.randint(0, 360)
        self.rot_speed = random.uniform(-1, 1)

        # Color based on element type
        if self.type == "leaf":
            green_variation = random.randint(-20, 20)
            base_color = LEAF_GREEN
            r = max(0, min(255, base_color[0] + green_variation))
            g = max(0, min(255, base_color[1] + green_variation))
            b = max(0, min(255, base_color[2] - green_variation))
            self.color = (r, g, b)
        else:  # water drop
            self.color = WATER_BLUE

    def update(self):
        self.y += self.speed
        self.x += self.drift
        self.rotation += self.rot_speed

        if self.y > SCREEN_HEIGHT + 20 or self.x < -20 or self.x > SCREEN_WIDTH + 20:
            self.reset()

    def draw(self, surface):
        try:
            if self.type == "leaf":
                leaf_surf = pygame.Surface((self.size * 4, self.size * 3), pygame.SRCALPHA)
                leaf_color = (*self.color, 180)
                stem_color = (*SOFT_BROWN[:3], 180)
                pygame.draw.ellipse(leaf_surf, leaf_color, (0, 0, self.size * 3, self.size * 2))
                pygame.draw.line(leaf_surf, stem_color,
                              (self.size * 1.5, self.size * 1),
                              (self.size * 3, self.size * 2), max(1, self.size // 3)) # Adjusted thickness
                rotated_leaf = pygame.transform.rotate(leaf_surf, self.rotation)
                leaf_rect = rotated_leaf.get_rect(center=(int(self.x), int(self.y)))
                surface.blit(rotated_leaf, leaf_rect)

            else:  # water drop
                drop_surf = pygame.Surface((self.size * 2, self.size * 3), pygame.SRCALPHA)
                drop_color = (*WATER_BLUE[:3], 150)
                pygame.draw.circle(drop_surf, drop_color, (self.size, self.size), self.size)
                points = [
                    (self.size - self.size/2, self.size),
                    (self.size + self.size/2, self.size),
                    (self.size, self.size * 2.5)
                ]
                pygame.draw.polygon(drop_surf, drop_color, points)
                highlight_pos = (int(self.size * 0.7), int(self.size * 0.7))
                highlight_radius = max(1, int(self.size / 3))
                highlight_color = (*LIGHT_BLUE[:3], 180)
                pygame.draw.circle(drop_surf, highlight_color, highlight_pos, highlight_radius)
                rotated_drop = pygame.transform.rotate(drop_surf, self.rotation / 10)
                drop_rect = rotated_drop.get_rect(center=(int(self.x), int(self.y)))
                surface.blit(rotated_drop, drop_rect)
        except (pygame.error, TypeError, ValueError) as e:
            print(f"Warning: Error drawing nature element: {e}")
            self.reset()

# Natural-looking progress bar
class NaturalProgressBar:
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
        self.animation_speed = 2 # Speed of bar filling/emptying
        self.style = style

        if self.style == "plant":
            self.vine_points = []
            self.leaf_positions = []
            self.generate_vine_path()

    def generate_vine_path(self):
        self.vine_points = []
        segment_height = self.height / 10.0 # Use float division

        if segment_height <= 0: return # Avoid division by zero if height is too small

        for i in range(11):
            y_pos = self.y + self.height - (i * segment_height)
            wiggle = random.uniform(-self.width/6, self.width/6) if i > 0 else 0
            x_pos = self.x + self.width/2 + wiggle
            self.vine_points.append((x_pos, y_pos))

            if i > 0 and i % 2 == 0:
                leaf_side = 1 if i % 4 == 0 else -1
                self.leaf_positions.append((i, leaf_side))

    def set_value(self, value):
        self.target_value = max(0, min(value, self.max_value)) # Clamp value

    def update(self):
        if abs(self.current_value - self.target_value) < self.animation_speed:
            self.current_value = self.target_value
        elif self.current_value < self.target_value:
            self.current_value += self.animation_speed
        elif self.current_value > self.target_value:
            self.current_value -= self.animation_speed

    def draw(self, surface):
        try:
            # Draw background frame
            bg_rect = pygame.Rect(self.x, self.y, self.width, self.height)
            # Fixed: Pass width=0 and border_radius positionally
            pygame.draw.rect(surface, self.bg_color, bg_rect, 0, 10)
            # Fixed: Pass width=2 and border_radius positionally
            pygame.draw.rect(surface, WOOD_BROWN, bg_rect, 2, 10) # Border

            progress_ratio = self.current_value / self.max_value if self.max_value > 0 else 0
            progress_ratio = max(0, min(1, progress_ratio)) # Ensure ratio is between 0 and 1

            if self.style == "plant":
                self.draw_plant_progress(surface, progress_ratio)
            else:
                self.draw_water_progress(surface, progress_ratio)

            # Draw percentage text (centered horizontally, near top)
            text_color = TEXT_BROWN # Use consistent text color
            text = font_medium.render(f"{int(self.current_value)}%", True, text_color)
            text_rect = text.get_rect(center=(self.x + self.width/2, self.y + 25))
            surface.blit(text, text_rect)
        except (pygame.error, ValueError, ZeroDivisionError, IndexError) as e:
            print(f"Warning: Error drawing progress bar: {e}")
            # Draw a simple fallback rectangle
            pygame.draw.rect(surface, self.bg_color, (self.x, self.y, self.width, self.height)) # No border radius on fallback


    def draw_plant_progress(self, surface, progress_ratio):
        visible_segments = math.ceil(progress_ratio * 10)

        if visible_segments > 0 and len(self.vine_points) > 1:
            # Draw vine
            for i in range(min(visible_segments, len(self.vine_points) - 1)):
                 # Check index bounds
                if i + 1 < len(self.vine_points):
                    start_point = self.vine_points[i]
                    end_point = self.vine_points[i + 1]
                    thickness = random.randint(3, 5)
                    pygame.draw.line(surface, DARK_GREEN, start_point, end_point, thickness)

                    # Draw leaves
                    for leaf_idx, leaf_side in self.leaf_positions:
                        if i == leaf_idx - 1:
                            leaf_x = (start_point[0] + end_point[0]) / 2
                            leaf_y = (start_point[1] + end_point[1]) / 2
                            leaf_size = random.randint(5, 8)
                            try:
                                leaf_surf = pygame.Surface((leaf_size * 3, leaf_size * 2), pygame.SRCALPHA)
                                pygame.draw.ellipse(leaf_surf, LEAF_GREEN, (0, 0, leaf_size * 3, leaf_size * 2))
                                angle = 45 if leaf_side > 0 else -45
                                rotated_leaf = pygame.transform.rotate(leaf_surf, angle)
                                leaf_rect = rotated_leaf.get_rect(center=(leaf_x + (leaf_side * 15), leaf_y))
                                surface.blit(rotated_leaf, leaf_rect)
                            except pygame.error as leaf_err:
                                print(f"Warning: Error drawing leaf: {leaf_err}")


            # Draw flower top
            if progress_ratio > 0.9 and len(self.vine_points) > 0:
                 # Check index bounds before accessing vine_points[-1]
                if len(self.vine_points) > 0:
                    top_x, top_y = self.vine_points[-1]
                    petal_color = (255, 200, 100)
                    center_color = SUNSET_ORANGE
                    try:
                        for angle_deg in range(0, 360, 60):
                            angle_rad = math.radians(angle_deg)
                            petal_x = top_x + 8 * math.cos(angle_rad)
                            petal_y = top_y + 8 * math.sin(angle_rad) # Use top_y here
                            pygame.draw.circle(surface, petal_color, (int(petal_x), int(petal_y)), 7)
                        pygame.draw.circle(surface, center_color, (int(top_x), int(top_y)), 5)
                    except pygame.error as flower_err:
                         print(f"Warning: Error drawing flower: {flower_err}")


    def draw_water_progress(self, surface, progress_ratio):
        fill_height = int(self.height * progress_ratio)

        if fill_height > 0:
            # Draw water fill with clipping for rounded corners
            water_clip_rect = pygame.Rect(self.x, self.y, self.width, self.height) # Full background area
            temp_surf = pygame.Surface(water_clip_rect.size, pygame.SRCALPHA)
            # Fixed: Pass width=0 and border_radius positionally
            pygame.draw.rect(temp_surf, self.color, (0, self.height - fill_height, self.width, fill_height), 0, 10) # Draw fill on temp surface

            # Apply clipping based on the rounded background rect
            clip_mask = pygame.Surface(water_clip_rect.size, pygame.SRCALPHA)
            # Fixed: Pass width=0 and border_radius positionally
            pygame.draw.rect(clip_mask, (255, 255, 255, 255), (0, 0, self.width, self.height), 0, 10)
            temp_surf.blit(clip_mask, (0,0), special_flags=pygame.BLEND_RGBA_MULT) # Keep only where mask is white

            surface.blit(temp_surf, water_clip_rect.topleft)


            # Add wave effect
            wave_height_amp = 3
            wave_y_base = self.y + self.height - fill_height - wave_height_amp
            try:
                wave_surface = pygame.Surface((self.width, wave_height_amp * 2), pygame.SRCALPHA)
                lighter_color = LIGHT_BLUE
                num_points = self.width // 5 # Control wave detail
                if num_points < 2: return # Need at least 2 points for polygon

                wave_points = []
                wave_points.append((0, wave_height_amp * 2)) # Bottom left for filling
                for i in range(num_points + 1):
                    x = int(i * (self.width / num_points))
                    offset = math.sin(time.time() * 3 + x * 0.05) * wave_height_amp # Faster wave
                    wave_points.append((x, wave_height_amp + offset))
                wave_points.append((self.width, wave_height_amp * 2)) # Bottom right for filling

                pygame.draw.polygon(wave_surface, lighter_color, wave_points)
                surface.blit(wave_surface, (self.x, wave_y_base))
            except pygame.error as wave_err:
                 print(f"Warning: Error drawing wave: {wave_err}")


# Eco-friendly UI button
class EcoButton:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=WHITE, border_radius=10):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_radius = border_radius
        self.hovered = False
        self.animation = 0 # Click animation timer

    def draw(self, surface):
        try:
            current_color = self.hover_color if self.hovered else self.color
            # Click feedback: slightly darken
            if self.animation > 0:
                self.animation -= 0.1
                # Create a slightly darker color for feedback
                feedback_color = tuple(max(0, c - 30) for c in current_color[:3])
                current_color = feedback_color

            # Main button rectangle
            # Fixed: Pass width=0 and border_radius positionally
            pygame.draw.rect(surface, current_color, self.rect, 0, self.border_radius)

            # Subtle top highlight
            highlight_rect = pygame.Rect(self.rect.x + 5, self.rect.y + 3, self.rect.width - 10, self.rect.height // 4)
            highlight_color = (*WHITE[:3], 30) # Subtle white highlight
            # Fixed: Pass width=0 and border_radius positionally
            pygame.draw.rect(surface, highlight_color, highlight_rect, 0, self.border_radius // 2)


            # Border
            # Fixed: Pass width=2 and border_radius positionally
            pygame.draw.rect(surface, WOOD_BROWN, self.rect, 2, self.border_radius)

            # Text
            text_surf = font_medium.render(self.text, True, self.text_color)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)
        except pygame.error as e:
             print(f"Warning: Error drawing button: {e}")


    def check_hover(self, mouse_pos):
        if self.rect:
            self.hovered = self.rect.collidepoint(mouse_pos)
            return self.hovered
        return False

    def clicked(self):
        self.animation = 1.0 # Start click animation
        print(f"Button '{self.text}' clicked.") # Add feedback
        return True

# Redesigned detection animation
class DetectionAnimation:
    def __init__(self, item_name, item_type):
        self.item_name = item_name.upper() # Ensure uppercase
        self.item_type = item_type.upper() # Ensure uppercase
        self.start_time = time.time()
        self.phase = "dropping"
        self.phase_durations = {
            "dropping": 1.2, # Slightly faster drop
            "scanning": 2.0, # Faster scan
            "revealing": 1.5, # Faster reveal
            "feedback": 4.0  # Shorter feedback time
        }
        # Use a simple elapsed time check instead of complicated phase booleans
        self.total_duration = sum(self.phase_durations.values())
        self.stats_updated = False

        self.scan_progress = 0
        self.reveal_alpha = 0
        self.particle_effects = []

        self.item_color = LEAF_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
        self.item_image = self.create_item_image() # Can return None if error occurs

        self.y_pos = -100
        self.rotation = random.uniform(-15, 15) # Start with slight rotation
        self.rotation_speed = random.uniform(-2, 2)
        self.fall_speed = 8 # Consistent fall speed
        self.target_y = SCREEN_HEIGHT // 2 - 60 # Slightly higher target

        self.loading_width = 280 # Slightly smaller bar
        self.loading_height = 18
        self.loading_x = SCREEN_WIDTH // 2 - self.loading_width // 2
        self.loading_y = self.target_y + 90 # Position below item

    def create_item_image(self):
        try:
            size = 80
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            base_color = LEAF_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
            border_color = DARK_GREEN if self.item_type == "RECYCLING" else WOOD_BROWN

            # Simplified generic shapes based on type
            if self.item_type == "RECYCLING":
                # Rounded rectangle like a bottle/can
                 # Fixed: Pass width=0 and border_radius positionally
                 pygame.draw.rect(surf, base_color, (size*0.2, size*0.1, size*0.6, size*0.8), 0, 15)
                 # Fixed: Pass width=2 and border_radius positionally
                 pygame.draw.rect(surf, border_color, (size*0.2, size*0.1, size*0.6, size*0.8), 2, 15)
                 # Simple recycling arrow symbol
                 center_x, center_y = size // 2, size // 2 + 10
                 radius = size // 6
                 pygame.draw.circle(surf, WHITE, (center_x, center_y), radius, 2)
                 # Add small arrows (simplified)
                 for i in range(3):
                     angle = math.radians(i * 120 + 30) # Offset start angle
                     start_angle = angle - math.radians(15)
                     end_angle = angle + math.radians(15)
                     # Calculate points for a small arrowhead
                     p1 = (center_x + radius * math.cos(start_angle), center_y + radius * math.sin(start_angle))
                     p2 = (center_x + (radius+5) * math.cos(angle), center_y + (radius+5) * math.sin(angle)) # Tip
                     p3 = (center_x + radius * math.cos(end_angle), center_y + radius * math.sin(end_angle))
                     pygame.draw.line(surf, WHITE, p1, p2, 2)
                     pygame.draw.line(surf, WHITE, p2, p3, 2)


            else: # TRASH
                # Irregular blob shape
                center_x, center_y = size // 2, size // 2
                radius = size // 3
                points = []
                num_verts = random.randint(7, 11)
                for i in range(num_verts):
                    angle = math.radians(i * (360 / num_verts))
                    radius_var = radius * random.uniform(0.7, 1.3)
                    x = center_x + radius_var * math.cos(angle)
                    y = center_y + radius_var * math.sin(angle)
                    points.append((int(x), int(y)))
                pygame.draw.polygon(surf, base_color, points)
                pygame.draw.polygon(surf, border_color, points, 2)

            # Subtle highlight
            highlight_color = (*WHITE[:3], 60)
            pygame.draw.circle(surf, highlight_color, (size // 3, size // 3), size // 5)

            return surf
        except pygame.error as e:
             print(f"Warning: Error creating item image: {e}")
             return None # Return None if creation fails

    def update(self):
        if not self.item_image: # Stop if image failed to create
             return True # End animation immediately

        try:
            current_time = time.time()
            elapsed_total = current_time - self.start_time

            # Determine current phase based on elapsed time
            if elapsed_total < self.phase_durations["dropping"]:
                self.phase = "dropping"
                self.y_pos += self.fall_speed
                self.rotation += self.rotation_speed
                # Clamp position and stop falling
                if self.y_pos >= self.target_y:
                    self.y_pos = self.target_y
                    self.fall_speed = 0 # Stop falling
                    self.rotation_speed *= 0.95 # Slow down rotation
            elif elapsed_total < self.phase_durations["dropping"] + self.phase_durations["scanning"]:
                self.phase = "scanning"
                elapsed_in_phase = elapsed_total - self.phase_durations["dropping"]
                self.scan_progress = min(100, (elapsed_in_phase / self.phase_durations["scanning"]) * 100)
                # Add scanning particles
                if random.random() < 0.15:
                    self.add_natural_particle("scan")
            elif elapsed_total < self.phase_durations["dropping"] + self.phase_durations["scanning"] + self.phase_durations["revealing"]:
                self.phase = "revealing"
                elapsed_in_phase = elapsed_total - (self.phase_durations["dropping"] + self.phase_durations["scanning"])
                progress = min(1.0, elapsed_in_phase / self.phase_durations["revealing"])
                self.reveal_alpha = int(255 * (progress**0.5)) # Ease-in alpha
                # Add reveal particles
                if random.random() < 0.25:
                    self.add_natural_particle("reveal")
            else:
                self.phase = "feedback"
                self.reveal_alpha = 255 # Ensure fully visible
                # Update stats only once
                if not self.stats_updated:
                    # This is where stats would be updated based on the result
                    self.stats_updated = True
                    print(f"Animation: Stats update trigger for {self.item_name} ({self.item_type})")

            self.update_particles()

            # Check if total animation time has passed
            return elapsed_total >= self.total_duration

        except (TypeError, ValueError, ZeroDivisionError) as e:
            print(f"Warning: Error updating animation: {e}")
            return True # End animation on error

    def add_natural_particle(self, phase_type):
        start_x = SCREEN_WIDTH // 2
        start_y = self.y_pos + (self.item_image.get_height() // 2 if self.item_image else 0)

        if phase_type == "scan":
            start_x = self.loading_x + (self.loading_width * self.scan_progress / 100)
            start_y = self.loading_y + self.loading_height / 2
            angle = random.uniform(math.pi * 1.2, math.pi * 1.8) # Emit downwards
            speed = random.uniform(0.5, 1.5)
            ptype = "sparkle"
            color = SUNSET_ORANGE
            size = random.uniform(1, 3)
        else: # reveal
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(1.5, 3.0) # Faster particles
            ptype = "leaf" if self.item_type == "RECYCLING" else "drop"
            color = LEAF_GREEN if ptype == "leaf" else WATER_BLUE
            size = random.uniform(3, 7)


        self.particle_effects.append({
            "type": ptype,
            "x": start_x + random.uniform(-5, 5),
            "y": start_y + random.uniform(-5, 5),
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed - (0.5 if phase_type=="scan" else 0), # Add slight downward push for scan
            "size": size,
            "color": color,
            "rotation": random.uniform(0, 360),
            "rot_speed": random.uniform(-4, 4),
            "life": random.uniform(0.8, 1.5) # Vary lifetime
        })

    def update_particles(self):
        for i in range(len(self.particle_effects) - 1, -1, -1):
            # Check index validity before accessing
            if i >= len(self.particle_effects):
                continue
            p = self.particle_effects[i]
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.05 # Simple gravity
            p["rotation"] += p["rot_speed"]
            p["life"] -= 0.015 # Faster fade
            # Optional: fade color
            # alpha = max(0, int(255 * (p["life"] / 1.0))) # Assuming initial life is ~1.0
            # p["current_color"] = (*p["color"][:3], alpha)

            if p["life"] <= 0:
                 # Check index again before popping, in case list was modified elsewhere (less likely here)
                 if 0 <= i < len(self.particle_effects):
                     self.particle_effects.pop(i)


    def draw_particles(self, surface):
        for p in self.particle_effects:
            try:
                # Calculate alpha based on remaining life
                # Use full alpha initially, fade quickly at the end
                life_ratio = max(0, p.get('life', 0) / 1.0) # Use .get with default
                alpha = int(255 * min(1, life_ratio * 2)) # Fade faster towards end

                if alpha <= 0: continue # Don't draw invisible particles

                particle_type = p.get("type")
                size = p.get("size", 0)
                if size <= 0: continue

                if particle_type == "leaf":
                    part_surf = pygame.Surface((size * 3, size * 2), pygame.SRCALPHA)
                    color = (*p.get("color", LEAF_GREEN)[:3], alpha)
                    pygame.draw.ellipse(part_surf, color, (0, 0, size * 2, size * 1.5)) # Simpler leaf
                    rotated_part = pygame.transform.rotate(part_surf, p.get("rotation", 0))
                    part_rect = rotated_part.get_rect(center=(int(p.get("x", 0)), int(p.get("y", 0))))
                    surface.blit(rotated_part, part_rect)
                elif particle_type == "drop":
                    part_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                    color = (*p.get("color", WATER_BLUE)[:3], alpha)
                    pygame.draw.circle(part_surf, color, (size, size), size) # Simple circle drop
                    part_rect = part_surf.get_rect(center=(int(p.get("x", 0)), int(p.get("y", 0))))
                    surface.blit(part_surf, part_rect)
                elif particle_type == "sparkle": # sparkle
                    color = (*p.get("color", SUNSET_ORANGE)[:3], alpha)
                    center_x, center_y = int(p.get("x", 0)), int(p.get("y", 0))
                    # Draw radiating lines for sparkle
                    num_lines = 5
                    rotation = p.get("rotation", 0)
                    for i in range(num_lines):
                        angle = rotation + i * (360 / num_lines)
                        rad = math.radians(angle)
                        end_x = center_x + math.cos(rad) * size * 1.5
                        end_y = center_y + math.sin(rad) * size * 1.5
                        pygame.draw.line(surface, color, (center_x, center_y), (int(end_x), int(end_y)), 1)

            except (pygame.error, KeyError, TypeError, ValueError) as e:
                print(f"Warning: Error drawing particle: {e}")
                # Attempt to remove the problematic particle
                try:
                    self.particle_effects.remove(p)
                except ValueError:
                    pass # Already removed

    def draw(self, surface):
        if not self.item_image: return # Don't draw if image failed

        try:
            # Draw Item
            rotated_image = pygame.transform.rotate(self.item_image, self.rotation)
            rotated_rect = rotated_image.get_rect(center=(SCREEN_WIDTH // 2, int(self.y_pos)))
            surface.blit(rotated_image, rotated_rect)

            # Draw Phase-Specific Elements
            if self.phase == "scanning":
                # Loading Bar Background
                loading_bg_rect = pygame.Rect(self.loading_x, self.loading_y, self.loading_width, self.loading_height)
                # Fixed: Pass width=0 and border_radius positionally
                pygame.draw.rect(surface, LIGHT_BROWN, loading_bg_rect, 0, 8)
                # Loading Bar Fill
                fill_color = LEAF_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
                fill_width = int(self.loading_width * (self.scan_progress / 100))
                if fill_width > 0:
                    fill_rect = pygame.Rect(self.loading_x, self.loading_y, fill_width, self.loading_height)
                    # Fixed: Pass width=0 and border_radius positionally
                    pygame.draw.rect(surface, fill_color, fill_rect, 0, 8)
                 # Loading Bar Border
                # Fixed: Pass width=2 and border_radius positionally
                pygame.draw.rect(surface, WOOD_BROWN, loading_bg_rect, 2, 8)
                # Status Text
                text = font_medium.render("Analyzing...", True, TEXT_BROWN)
                text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, self.loading_y - 25)) # Position above bar
                surface.blit(text, text_rect)

            elif self.phase in ["revealing", "feedback"]:
                # Common drawing for revealing/feedback phases

                # Calculate center positions for text based on current item position
                text_center_x = SCREEN_WIDTH // 2
                text_base_y = self.y_pos + rotated_rect.height // 2 + 20 # Below item image

                # Item Name Text (with subtle background)
                name_surf = font_large.render(self.item_name, True, TEXT_BROWN)
                name_surf.set_alpha(self.reveal_alpha)
                name_rect = name_surf.get_rect(center=(text_center_x, text_base_y + 20))
                 # Background panel for name
                name_bg_rect = name_rect.inflate(20, 10) # Add padding
                name_bg_surf = pygame.Surface(name_bg_rect.size, pygame.SRCALPHA)
                name_bg_surf.fill((*CREAM[:3], int(180 * (self.reveal_alpha / 255)))) # Semi-transparent cream
                # Fixed: Pass width=2 and border_radius positionally
                pygame.draw.rect(name_bg_surf, (*WOOD_BROWN[:3], int(200 * (self.reveal_alpha / 255))), name_bg_surf.get_rect(), 2, 5) # Border
                surface.blit(name_bg_surf, name_bg_rect.topleft)
                surface.blit(name_surf, name_rect)


                # Item Type Text (Recyclable/Non-Recyclable)
                type_color = DARK_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
                type_text = "Recyclable" if self.item_type == "RECYCLING" else "Landfill"
                type_surf = font_medium.render(type_text, True, type_color)
                type_surf.set_alpha(self.reveal_alpha)
                type_rect = type_surf.get_rect(center=(text_center_x, text_base_y + 60))
                # Background panel for type
                type_bg_rect = type_rect.inflate(15, 8)
                type_bg_surf = pygame.Surface(type_bg_rect.size, pygame.SRCALPHA)
                type_bg_surf.fill((*CREAM[:3], int(160 * (self.reveal_alpha / 255))))
                 # Fixed: Pass width=1 and border_radius positionally
                pygame.draw.rect(type_bg_surf, (*WOOD_BROWN[:3], int(180 * (self.reveal_alpha / 255))), type_bg_surf.get_rect(), 1, 5)
                surface.blit(type_bg_surf, type_bg_rect.topleft)
                surface.blit(type_surf, type_rect)


                # Feedback Phase Specifics
                if self.phase == "feedback":
                    time_left = max(0, self.total_duration - (time.time() - self.start_time))
                    prompt_text = f"Is this correct? ({int(time_left)}s)"
                    prompt_surf = font_medium.render(prompt_text, True, TEXT_BROWN)
                    prompt_rect = prompt_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150)) # Above buttons
                    # Background for prompt
                    prompt_bg_rect = prompt_rect.inflate(20, 10)
                    prompt_bg_surf = pygame.Surface(prompt_bg_rect.size, pygame.SRCALPHA)
                    prompt_bg_surf.fill((*LIGHT_CREAM[:3], 190))
                    # Fixed: Pass width=2 and border_radius positionally
                    pygame.draw.rect(prompt_bg_surf, WOOD_BROWN, prompt_bg_surf.get_rect(), 2, 8)
                    surface.blit(prompt_bg_surf, prompt_bg_rect.topleft)
                    surface.blit(prompt_surf, prompt_rect)


            # Draw particles on top of everything else in this animation
            self.draw_particles(surface)

        except (pygame.error, TypeError, ValueError, ZeroDivisionError, AttributeError) as e:
             print(f"Warning: Error drawing detection animation phase '{self.phase}': {e}")
             # Attempt to gracefully end the animation if drawing fails badly
             self.phase = "feedback" # Jump to end state visually
             self.start_time = time.time() - self.total_duration # Mark as finished


# Modern UI with sustainability theme
class SmartBinInterface:
    def __init__(self):
        self.stats = BinStats()
        self.nature_elements = [NatureElement() for _ in range(15)]
        self.state = "idle"
        self.detection_animation = None

        pb_width, pb_height = 140, 200
        pb_y = SCREEN_HEIGHT // 2 - pb_height // 2 + 20 # Centered vertically offset by header
        pb_x_margin = 180 # Distance from center

        self.recycling_progress = NaturalProgressBar(
            SCREEN_WIDTH // 2 - pb_x_margin - pb_width // 2, pb_y, pb_width, pb_height,
            LEAF_GREEN, LIGHT_CREAM, style="plant"
        )
        self.landfill_progress = NaturalProgressBar(
            SCREEN_WIDTH // 2 + pb_x_margin - pb_width // 2, pb_y, pb_width, pb_height,
            WATER_BLUE, LIGHT_CREAM, style="water"
        )
        # Start progress bars empty or load from saved state if implemented
        self.recycling_progress.set_value(0)
        self.landfill_progress.set_value(0)

        button_width = 160
        button_height = 50
        button_y = SCREEN_HEIGHT - 80 # Position buttons lower
        self.correct_button = EcoButton(
            SCREEN_WIDTH // 2 - button_width - 20, button_y, button_width, button_height,
            "Correct", LEAF_GREEN, LIGHT_GREEN
        )
        self.incorrect_button = EcoButton(
            SCREEN_WIDTH // 2 + 20, button_y, button_width, button_height,
            "Incorrect", SOFT_BROWN, LIGHT_BROWN
        )

        self.last_hint_time = 0 # Show first hint immediately
        self.hint_interval = 25 # Shorter hint interval
        self.hints = [
            "SmartBin™ uses AI to sort waste accurately.",
            "Recycling reduces landfill waste and saves resources.",
            "Contamination ruins recycling - please sort carefully!",
            "Over 95% accuracy in waste sorting!",
            "Thank you for helping keep the planet green!",
            "Plastic bottles take over 400 years to decompose.",
            "Glass is infinitely recyclable!",
            "Recycle paper 5-7 times before fibers weaken.",
            "Check local guidelines for specific recycling rules.",
            "Reduce, Reuse, Recycle - in that order!",
        ]
        self.current_hint = random.choice(self.hints)
        self.hint_alpha = 0
        self.hint_fade_in = True
        self.hint_display_duration = 8 # How long hint stays fully visible
        self.hint_fade_duration = 1.5 # How long fade takes
        self.hint_state = "fading_in" # fading_in, visible, fading_out
        self.hint_visible_start_time = 0


    def update_nature_elements(self):
        for element in self.nature_elements:
            element.update()

    def draw_nature_elements(self, surface):
        for element in self.nature_elements:
            element.draw(surface)

    def update_progress_bars(self):
        self.recycling_progress.update()
        self.landfill_progress.update()

    def draw_progress_bars(self, surface):
        try:
            # Define card areas slightly larger than progress bars
            card_padding = 20
            recycling_card_rect = pygame.Rect(
                self.recycling_progress.x - card_padding,
                self.recycling_progress.y - 40, # Extend upwards for label
                self.recycling_progress.width + 2 * card_padding,
                self.recycling_progress.height + 60 # Room for label and count
            )
            landfill_card_rect = pygame.Rect(
                self.landfill_progress.x - card_padding,
                self.landfill_progress.y - 40,
                self.landfill_progress.width + 2 * card_padding,
                self.landfill_progress.height + 60
            )

            # Draw cards with background and border
            for card_rect in [recycling_card_rect, landfill_card_rect]:
                 # Fixed: Pass width=0 and border_radius positionally
                pygame.draw.rect(surface, CREAM, card_rect, 0, 15)
                # Optional: Add subtle texture here if needed
                # Fixed: Pass width=2 and border_radius positionally
                pygame.draw.rect(surface, WOOD_BROWN, card_rect, 2, 15)

            # Draw progress indicators on top of cards
            self.recycling_progress.draw(surface)
            self.landfill_progress.draw(surface)

            # Labels (Positioned above progress bars within the card)
            recycling_label = font_medium.render("Recycling", True, DARK_GREEN)
            recycling_rect = recycling_label.get_rect(center=(self.recycling_progress.x + self.recycling_progress.width / 2, self.recycling_progress.y - 20))
            surface.blit(recycling_label, recycling_rect)

            landfill_label = font_medium.render("Landfill", True, TEXT_BROWN)
            landfill_rect = landfill_label.get_rect(center=(self.landfill_progress.x + self.landfill_progress.width / 2, self.landfill_progress.y - 20))
            surface.blit(landfill_label, landfill_rect)

            # Item Counts (Positioned below progress bars)
            count_y = self.recycling_progress.y + self.recycling_progress.height + 15
            recycling_count_text = f"{self.stats.recycled_items} items"
            recycling_count_surf = font_small.render(recycling_count_text, True, DARK_GREEN)
            recycling_count_rect = recycling_count_surf.get_rect(center=(recycling_rect.centerx, count_y))
            surface.blit(recycling_count_surf, recycling_count_rect)

            landfill_count_text = f"{self.stats.landfill_items} items"
            landfill_count_surf = font_small.render(landfill_count_text, True, TEXT_BROWN)
            landfill_count_rect = landfill_count_surf.get_rect(center=(landfill_rect.centerx, count_y))
            surface.blit(landfill_count_surf, landfill_count_rect)


            # Draw Environmental Impact Section (Only if items have been processed)
            if self.stats.total_items > 0:
                 impact_card_height = 65
                 impact_card_y = SCREEN_HEIGHT - impact_card_height - 10 # Position at bottom
                 impact_card = pygame.Rect(SCREEN_WIDTH // 2 - 180, impact_card_y, 360, impact_card_height)

                 # Draw card background and border
                 # Fixed: Pass width=0 and border_radius positionally
                 pygame.draw.rect(surface, CREAM, impact_card, 0, 10)
                 # Fixed: Pass width=2 and border_radius positionally
                 pygame.draw.rect(surface, WOOD_BROWN, impact_card, 2, 10)

                 # Impact Title
                 impact_title = font_medium.render("Environmental Impact", True, TEXT_BROWN)
                 impact_title_rect = impact_title.get_rect(center=(impact_card.centerx, impact_card.y + 18))
                 surface.blit(impact_title, impact_title_rect)

                 # Impact Stats (CO2 and Water Saved)
                 stats_y = impact_card.y + 45
                 icon_size = 15

                 # CO2 Saved
                 co2_text = f"{self.stats.co2_saved:.1f} kg CO₂"
                 co2_surf = font_small.render(co2_text, True, DARK_GREEN)
                 co2_rect = co2_surf.get_rect(midright=(impact_card.centerx - 10, stats_y))
                 surface.blit(co2_surf, co2_rect)
                 # Leaf Icon
                 leaf_icon_rect = pygame.Rect(0, 0, icon_size, icon_size * 0.8)
                 leaf_icon_rect.midright = (co2_rect.left - 5, stats_y)
                 pygame.draw.ellipse(surface, LEAF_GREEN, leaf_icon_rect)

                 # Water Saved
                 water_text = f"{self.stats.water_saved:.1f} L Water"
                 water_surf = font_small.render(water_text, True, WATER_BLUE)
                 water_rect = water_surf.get_rect(midleft=(impact_card.centerx + 10, stats_y))
                 surface.blit(water_surf, water_rect)
                 # Drop Icon
                 drop_icon_rect = pygame.Rect(0, 0, icon_size * 0.7, icon_size)
                 drop_icon_rect.midleft = (water_rect.right + 5, stats_y)
                 pygame.draw.ellipse(surface, WATER_BLUE, drop_icon_rect)


        except (pygame.error, TypeError, ValueError) as e:
            print(f"Warning: Error drawing progress bars/stats: {e}")


    def update_hint(self):
        current_time = time.time()
        time_since_change = current_time - self.last_hint_time

        if self.hint_state == "fading_in":
            if time_since_change < self.hint_fade_duration:
                self.hint_alpha = int(255 * (time_since_change / self.hint_fade_duration))
            else:
                self.hint_alpha = 255
                self.hint_state = "visible"
                self.hint_visible_start_time = current_time # Record when it became fully visible
        elif self.hint_state == "visible":
            time_visible = current_time - self.hint_visible_start_time
            if time_visible >= self.hint_display_duration:
                self.hint_state = "fading_out"
                self.last_hint_time = current_time # Reset timer for fade out start
        elif self.hint_state == "fading_out":
            time_fading = current_time - self.last_hint_time
            if time_fading < self.hint_fade_duration:
                 self.hint_alpha = int(255 * (1 - (time_fading / self.hint_fade_duration)))
            else:
                self.hint_alpha = 0
                # Change hint and start fading in the new one
                self.current_hint = random.choice(self.hints)
                self.hint_state = "fading_in"
                self.last_hint_time = current_time # Reset timer for fade in start

        self.hint_alpha = max(0, min(255, self.hint_alpha)) # Clamp alpha


    def draw_hint(self, surface):
        if self.hint_alpha <= 0: return # Don't draw if fully faded out

        try:
            hint_surf = font_small.render(self.current_hint, True, TEXT_BROWN)
            hint_rect = hint_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 45)) # Position near bottom center

            # Dynamic background based on text size
            bg_rect = hint_rect.inflate(30, 15)
            hint_back = pygame.Surface(bg_rect.size, pygame.SRCALPHA)

            # Background color with alpha
            bg_color_alpha = (*CREAM[:3], int(200 * (self.hint_alpha / 255)))
            # Fixed: Pass width=0 and border_radius positionally
            pygame.draw.rect(hint_back, bg_color_alpha, hint_back.get_rect(), 0, 10)

             # Border with alpha
            border_color_alpha = (*WOOD_BROWN[:3], int(220 * (self.hint_alpha / 255)))
             # Fixed: Pass width=2 and border_radius positionally
            pygame.draw.rect(hint_back, border_color_alpha, hint_back.get_rect(), 2, 10)

            # Blit background first
            surface.blit(hint_back, bg_rect.topleft)

            # Blit text with alpha
            hint_surf.set_alpha(self.hint_alpha)
            surface.blit(hint_surf, hint_rect)

        except pygame.error as e:
             print(f"Warning: Error drawing hint: {e}")


    def process_camera_detection(self, detection_data):
        """Process detection data received from the external camera process"""
        if self.state == "idle" and isinstance(detection_data, dict):
            item_name = detection_data.get("name", "Unknown Item")
            item_type = detection_data.get("type", "TRASH") # Default to TRASH if type missing

            # Validate type
            if item_type.upper() not in ["RECYCLING", "TRASH"]:
                 print(f"Warning: Received invalid item type '{item_type}'. Defaulting to TRASH.")
                 item_type = "TRASH"

            print(f"GUI: Received detection - Item: {item_name}, Type: {item_type}")
            self.detection_animation = DetectionAnimation(item_name, item_type)
            self.state = "detecting"
        elif not isinstance(detection_data, dict):
            print(f"Warning: Received invalid detection data format: {type(detection_data)}")


    def update_detection(self):
        try:
            # Check queue for new data ONLY IF idle
            if self.state == "idle":
                 if not detection_queue.empty():
                    detection_data = detection_queue.get()
                    self.process_camera_detection(detection_data)
                    detection_queue.task_done() # Mark task as done

            # Update ongoing animation
            if self.detection_animation:
                animation_finished = self.detection_animation.update()

                # Update BinStats when animation reaches feedback phase (and hasn't already)
                if self.detection_animation.phase == "feedback" and not self.detection_animation.stats_updated:
                    # Ensure stats are updated only once per animation
                    self.stats.update_stats(self.detection_animation.item_type)
                    print(f"GUI: Stats updated for {self.detection_animation.item_name}")

                    # Update progress bars based on new stats
                    if self.stats.total_items > 0:
                        recycling_percentage = self.stats.get_recycling_percentage()
                        self.recycling_progress.set_value(recycling_percentage)
                        self.landfill_progress.set_value(100 - recycling_percentage)
                        print(f"GUI: Progress bars updated - Rec: {recycling_percentage:.1f}%, Land: {100-recycling_percentage:.1f}%")
                    else:
                        self.recycling_progress.set_value(0)
                        self.landfill_progress.set_value(0)

                    self.detection_animation.stats_updated = True # Mark as updated


                # If animation update() returns True, it's finished
                if animation_finished:
                    print("GUI: Detection animation finished.")
                    self.detection_animation = None
                    self.state = "idle" # Return to idle state

        except queue.Empty:
             pass # Normal if queue is empty
        except Exception as e:
            print(f"Error in update_detection: {e}")
            # Attempt to recover gracefully
            self.detection_animation = None
            self.state = "idle"


    def draw_detection(self, surface):
        try:
            if self.detection_animation:
                # Draw a semi-transparent overlay during detection to focus attention
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay_alpha = 100 # Adjust transparency
                if self.detection_animation.phase == "dropping":
                     overlay_alpha = int(100 * (self.detection_animation.y_pos / self.detection_animation.target_y)) if self.detection_animation.target_y > 0 else 0
                elif self.detection_animation.phase == "scanning":
                     overlay_alpha = 100 + int(50 * (self.detection_animation.scan_progress / 100)) # Darken during scan
                elif self.detection_animation.phase == "revealing":
                     overlay_alpha = 150 + int(105 * (self.detection_animation.reveal_alpha / 255)) # Darkest during reveal/feedback
                elif self.detection_animation.phase == "feedback":
                    overlay_alpha = 200

                overlay.fill((0, 0, 0, max(0, min(200, overlay_alpha)))) # Black overlay
                surface.blit(overlay, (0, 0))


                # Draw the animation itself
                self.detection_animation.draw(surface)

                # Draw feedback buttons ONLY during the feedback phase
                if self.detection_animation.phase == "feedback":
                    self.correct_button.draw(surface)
                    self.incorrect_button.draw(surface)

        except (pygame.error, AttributeError) as e:
            # Handle rendering errors gracefully
            print(f"Warning: Error drawing detection overlay/animation: {e}")
            # Don't crash, maybe skip drawing this frame for detection elements


    def handle_button_clicks(self, mouse_pos):
        clicked = False
        # Only handle clicks if in feedback state and animation exists
        if self.state == "detecting" and self.detection_animation and self.detection_animation.phase == "feedback":
            # Use get_pressed()[0] for left mouse button click event
            # Check hover first before checking click state
            correct_hover = self.correct_button.check_hover(mouse_pos)
            incorrect_hover = self.incorrect_button.check_hover(mouse_pos)

            if pygame.mouse.get_pressed()[0]: # Check if left button is currently down
                if correct_hover:
                    if self.correct_button.clicked(): # clicked() returns True
                        # TODO: Optionally send feedback ('correct') back to the detection script via network/queue
                        print("Feedback: Correct")
                        clicked = True
                elif incorrect_hover:
                    if self.incorrect_button.clicked():
                        # TODO: Optionally send feedback ('incorrect') back to the detection script
                        print("Feedback: Incorrect")
                        clicked = True

                # If a button was clicked, reset the state immediately
                if clicked:
                    self.detection_animation = None
                    self.state = "idle"
                    # Debounce: Short delay to prevent multiple clicks processing
                    pygame.time.wait(150) # Increased slightly

        return clicked # Return whether a click was handled


    def check_button_hover(self, mouse_pos):
        # Update hover state for buttons only when they are potentially visible
        if self.state == "detecting" and self.detection_animation and self.detection_animation.phase == "feedback":
            try:
                self.correct_button.check_hover(mouse_pos)
                self.incorrect_button.check_hover(mouse_pos)
            except (AttributeError, TypeError) as e:
                 print(f"Warning: Error checking button hover state: {e}")
        else:
            # Ensure buttons are not marked as hovered when not visible
            self.correct_button.hovered = False
            self.incorrect_button.hovered = False


# Draw natural, earthy background
def draw_background(surface):
    try:
        # Subtle vertical gradient
        for y in range(SCREEN_HEIGHT):
            ratio = y / SCREEN_HEIGHT
            r = int(LIGHT_CREAM[0] * (1 - ratio) + CREAM[0] * ratio)
            g = int(LIGHT_CREAM[1] * (1 - ratio) + CREAM[1] * ratio)
            b = int(LIGHT_CREAM[2] * (1 - ratio) + CREAM[2] * ratio)
            pygame.draw.line(surface, (r, g, b), (0, y), (SCREEN_WIDTH, y))

        # Optional: Add very subtle pattern overlay (can impact performance)
        # pattern_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        # for _ in range(50): # Draw fewer elements
        #     x, y = random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)
        #     size = random.randint(2, 4)
        #     alpha = random.randint(3, 8)
        #     color = random.choice([(*LEAF_GREEN[:3], alpha), (*WATER_BLUE[:3], alpha), (*SOFT_BROWN[:3], alpha)])
        #     pygame.draw.circle(pattern_surf, color, (x, y), size)
        # surface.blit(pattern_surf, (0,0))

    except pygame.error as e:
        print(f"Warning: Error drawing background: {e}")
        surface.fill(CREAM) # Fallback


# Draw natural, eco-friendly header
def draw_header(surface):
    try:
        header_height = 65 # Slightly smaller header
        header_rect = pygame.Rect(0, 0, SCREEN_WIDTH, header_height)
        # Background with wood texture
        pygame.draw.rect(surface, CREAM, header_rect)
        for y in range(0, header_height, 3):
             grain_width = random.randint(int(SCREEN_WIDTH * 0.8), SCREEN_WIDTH)
             grain_x = random.randint(0, int(SCREEN_WIDTH * 0.2))
             grain_alpha = random.randint(8, 18)
             grain_line = pygame.Surface((grain_width, 1), pygame.SRCALPHA)
             grain_line.fill((*SOFT_BROWN[:3], grain_alpha))
             surface.blit(grain_line, (grain_x, y))
        # Bottom border line
        pygame.draw.rect(surface, WOOD_BROWN, (0, header_height - 2, SCREEN_WIDTH, 2))

        # Title Text
        title_text = font_title.render("SmartBin™ Waste Management", True, TEXT_BROWN)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, header_height//2))
        surface.blit(title_text, title_rect)

        # Leaf Decorations
        leaf_size = 18
        leaf_surf = pygame.Surface((leaf_size * 1.5, leaf_size), pygame.SRCALPHA)
        pygame.draw.ellipse(leaf_surf, LEAF_GREEN, (0, 0, leaf_size * 1.5, leaf_size))
        pygame.draw.line(leaf_surf, DARK_GREEN, (leaf_size * 0.5, leaf_size / 2), (leaf_size * 1.5, leaf_size / 2), 2) # Stem

        leaf_left_rect = leaf_surf.get_rect(center=(title_rect.left - 30, title_rect.centery))
        surface.blit(leaf_surf, leaf_left_rect)

        leaf_right_surf = pygame.transform.flip(leaf_surf, True, False) # Flip for right side
        leaf_right_rect = leaf_right_surf.get_rect(center=(title_rect.right + 30, title_rect.centery))
        surface.blit(leaf_right_surf, leaf_right_rect)

    except pygame.error as e:
        print(f"Warning: Error drawing header: {e}")
        # Minimal fallback header
        pygame.draw.rect(surface, CREAM, (0, 0, SCREEN_WIDTH, 65))
        pygame.draw.line(surface, WOOD_BROWN, (0, 63), (SCREEN_WIDTH, 63), 2)


# --- SIMULATION FUNCTION (for testing GUI without external process) ---
def simulate_detection(queue):
    """Puts a simulated detection event into the queue."""
    items = [
        {"name": "Plastic Bottle", "type": "RECYCLING"},
        {"name": "Apple Core", "type": "TRASH"},
        {"name": "Aluminum Can", "type": "RECYCLING"},
        {"name": "Paper Towel", "type": "TRASH"},
        {"name": "Newspaper", "type": "RECYCLING"},
        {"name": "Coffee Cup", "type": "TRASH"}, # Often coated, not recyclable
        {"name": "Glass Jar", "type": "RECYCLING"},
    ]
    chosen_item = random.choice(items)
    print(f"\n--- SIMULATING DETECTION: {chosen_item['name']} ({chosen_item['type']}) ---")
    queue.put(chosen_item)


# Main function
def main():
    try:
        interface = SmartBinInterface()
        clock = pygame.time.Clock()
        running = True

        # --- Removed Serial Initialization ---
        # --- Removed Camera Thread Start ---
        print("GUI Initialized. Waiting for detection data...")
        print("Press 'D' to simulate a detection event.")

        last_update_time = time.time()

        while running:
            # --- Event Handling ---
            mouse_pos = pygame.mouse.get_pos() # Get mouse pos once per frame for hover checks

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # --- Simulation Trigger ---
                    if event.key == pygame.K_d:
                         if interface.state == "idle": # Only simulate if idle
                             simulate_detection(detection_queue)
                         else:
                             print("Simulation ignored: Detection already in progress.")
                # Handle mouse clicks specifically here using MOUSEBUTTONDOWN
                elif event.type == pygame.MOUSEBUTTONDOWN:
                     if event.button == 1: # Left mouse button
                         # Pass the mouse position from the event
                         interface.handle_button_clicks(event.pos)


            # --- Updates ---
            current_time = time.time()
            delta_time = current_time - last_update_time # Can be used for time-based physics if needed
            last_update_time = current_time

            interface.check_button_hover(mouse_pos) # Update hover states continuously

            interface.update_nature_elements()
            interface.update_progress_bars()
            interface.update_hint()
            interface.update_detection() # Checks queue and updates animation/state

            # --- Drawing ---
            try:
                draw_background(screen)
                interface.draw_nature_elements(screen) # Draw background elements first
                draw_header(screen)

                # Draw main content based on state
                if interface.state == "idle":
                    interface.draw_progress_bars(screen)
                    # Draw hint only when idle and not during button feedback time
                    if not (interface.correct_button.animation > 0 or interface.incorrect_button.animation > 0):
                         interface.draw_hint(screen)
                else: # "detecting" state
                    interface.draw_detection(screen) # Draws overlay, animation, and buttons

            except Exception as draw_err:
                print(f"Critical error during drawing: {draw_err}")
                # Simple fallback display to avoid crash loop
                screen.fill(CREAM)
                try:
                    err_font = pygame.font.SysFont("Arial", 20)
                    err_text = err_font.render("Drawing Error Occurred", True, BLACK)
                    screen.blit(err_text, (50, 50))
                except: pass # Ignore errors during error display

            pygame.display.flip() # Use flip instead of update for full screen redraws
            clock.tick(60) # Limit frame rate


    except Exception as main_err:
        print(f"Critical error in main loop: {main_err}")
    finally:
        # --- Cleanup ---
        print("Shutting down Pygame...")
        pygame.quit()
        print("Application terminated.")
        sys.exit()


if __name__ == "__main__":
    main()