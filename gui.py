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
import traceback  # Added for better error reporting

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
    screen = pygame.display.set_mode((actual_screen_width, actual_screen_height), pygame.FULLSCREEN | pygame.SCALED | pygame.SRCALPHA)
    print("Fullscreen mode set.")
except pygame.error as e:
    print(f"Warning: Could not set fullscreen mode ({e}). Trying windowed mode.")
    screen = pygame.display.set_mode((actual_screen_width, actual_screen_height))

pygame.display.set_caption("Consolimate™ Waste Management System")

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
    font_title = pygame.font.SysFont("DejaVu Sans", 100, bold=True)
    font_large = pygame.font.SysFont("DejaVu Sans", 90, bold=True)
    font_medium = pygame.font.SysFont("DejaVu Sans", 70)
    font_small = pygame.font.SysFont("DejaVu Sans", 45)
    font_tiny = pygame.font.SysFont("DejaVu Sans", 36)
    print("Loaded DejaVu Sans system font.")
except Exception as e1:
    print(f"Warning: Could not load system font 'DejaVu Sans'. Trying Arial. Error: {e1}")
    try:
        font_title = pygame.font.SysFont("Arial", 45, bold=True)
        font_large = pygame.font.SysFont("Arial", 40, bold=True)
        font_medium = pygame.font.SysFont("Arial", 30)
        font_small = pygame.font.SysFont("Arial", 28)
        font_tiny = pygame.font.SysFont("Arial", 22)
        print("Loaded Arial system font.")
    except Exception as e2:
        print(f"Warning: Could not load system font 'Arial'. Using default font. Error: {e2}")
        font_title = pygame.font.Font(None, 60); font_large = pygame.font.Font(None, 50)
        font_medium = pygame.font.Font(None, 45); font_small = pygame.font.Font(None, 36)
        font_tiny = pygame.font.Font(None, 25)

# --- Communication Setup ---
detection_queue = queue.Queue()
GUI_SERVER_HOST = '0.0.0.0'  # Listen on all available network interfaces
GUI_SERVER_PORT = 9999
server_running = True

# --- Pygame GUI Classes ---

class BinStats:
    def __init__(self):
        self.recycled_items = 0
        self.landfill_items = 0
        self.total_items = 0
        self.co2_saved = 0
        self.water_saved = 0
        self.last_updated = time.time()

    def update_stats(self, bin_type):
        self.total_items += 1
        if bin_type.upper() == "RECYCLING":
            self.recycled_items += 1
            print(self.recycled_items)
            self.co2_saved += random.uniform(0.1, 0.3)
            self.water_saved += random.uniform(0.5, 2.0)
        elif bin_type.upper() == "TRASH":
            self.landfill_items += 1
        self.last_updated = time.time()

    def get_recycling_percentage(self):
        if self.total_items == 0:
            return 0
        return (self.recycled_items / self.total_items) * 100

    def get_landfill_percentage(self):
        if self.total_items == 0:
            return 0
        return (self.landfill_items / self.total_items) * 100

class NatureElement:
    def __init__(self):
        self.reset()
        self.y = random.randint(0, SCREEN_HEIGHT)

    def reset(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = -random.randint(10, 50)
        self.type = random.choice(["leaf", "drop"])
        self.size = random.randint(6, 15) if self.type == "leaf" else random.randint(6, 15)
        self.speed = random.uniform(1, 2)
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
        else:
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
                leaf_color = (*self.color[:3], 180)
                stem_color = (*SOFT_BROWN[:3], 180)
                pygame.draw.ellipse(leaf_surf, leaf_color, (0, 0, self.size * 3, self.size * 2))
                pygame.draw.line(leaf_surf, stem_color, (self.size * 1.5, self.size * 1), (self.size * 3, self.size * 2), max(1, self.size // 3))
                rotated_leaf = pygame.transform.rotate(leaf_surf, self.rotation)
                leaf_rect = rotated_leaf.get_rect(center=(int(self.x), int(self.y)))
                surface.blit(rotated_leaf, leaf_rect)
            else:
                drop_surf = pygame.Surface((self.size * 2, self.size * 3), pygame.SRCALPHA)
                drop_color = (*WATER_BLUE[:3], 150)
                pygame.draw.circle(drop_surf, drop_color, (self.size, self.size), self.size)
                points = [(self.size - self.size/2, self.size), (self.size + self.size/2, self.size), (self.size, self.size * 2.5)]
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
        self.animation_speed = 2
        self.style = style

        if self.style == "plant":
            self.vine_points = []
            self.leaf_positions = []
            self.generate_vine_path()

    def generate_vine_path(self):
        self.vine_points = []
        segment_height = self.height / 10.0
        if segment_height <= 0: return

        for i in range(11):
            y_pos = self.y + self.height - (i * segment_height)
            wiggle = random.uniform(-self.width/6, self.width/6) if i > 0 else 0
            x_pos = self.x + self.width/2 + wiggle
            self.vine_points.append((x_pos, y_pos))
            if i > 0 and i % 2 == 0:
                leaf_side = 1 if i % 4 == 0 else -1
                self.leaf_positions.append((i, leaf_side))

    def set_value(self, value):
        self.target_value = max(0, min(value, self.max_value))

    def update(self):
        if abs(self.current_value - self.target_value) < self.animation_speed:
            self.current_value = self.target_value
        elif self.current_value < self.target_value:
            self.current_value += self.animation_speed
        elif self.current_value > self.target_value:
            self.current_value -= self.animation_speed

    def draw(self, surface):
        try:
            bg_rect = pygame.Rect(self.x, self.y, self.width, self.height)
            pygame.draw.rect(surface, self.bg_color, bg_rect)
            pygame.draw.rect(surface, WOOD_BROWN, bg_rect, 2)
            progress_ratio = self.current_value / self.max_value if self.max_value > 0 else 0
            progress_ratio = max(0, min(1, progress_ratio))
            if self.style == "plant":
                self.draw_plant_progress(surface, progress_ratio)
            else:
                self.draw_water_progress(surface, progress_ratio)
            text_color = TEXT_BROWN
            text = font_medium.render(f"{int(self.current_value)}%", True, text_color)
            text_rect = text.get_rect(center=(self.x + self.width/2, self.y + 55))
            surface.blit(text, text_rect)
        except (pygame.error, ValueError, ZeroDivisionError, IndexError) as e:
            print(f"Warning: Error drawing progress bar: {e}")
            pygame.draw.rect(surface, self.bg_color, (self.x, self.y, self.width, self.height))

    def draw_plant_progress(self, surface, progress_ratio):
         visible_segments = math.ceil(progress_ratio * 10)
         if visible_segments > 0 and len(self.vine_points) > 1:
             for i in range(min(visible_segments, len(self.vine_points) - 1)):
                 if i + 1 < len(self.vine_points):
                     start_point = self.vine_points[i]
                     end_point = self.vine_points[i + 1]
                     thickness = random.randint(3, 5)
                     pygame.draw.line(surface, DARK_GREEN, start_point, end_point, thickness)
                     for leaf_idx, leaf_side in self.leaf_positions:
                         if i == leaf_idx - 1:
                             leaf_x = (start_point[0] + end_point[0]) / 2
                             leaf_y = (start_point[1] + end_point[1]) / 2
                             leaf_size = random.randint(15, 18)
                             try:
                                 leaf_surf = pygame.Surface((leaf_size * 3, leaf_size * 2), pygame.SRCALPHA)
                                 pygame.draw.ellipse(leaf_surf, LEAF_GREEN, (0, 0, leaf_size * 3, leaf_size * 2))
                                 angle = 45 if leaf_side > 0 else -45
                                 rotated_leaf = pygame.transform.rotate(leaf_surf, angle)
                                 leaf_rect = rotated_leaf.get_rect(center=(leaf_x + (leaf_side * 15), leaf_y))
                                 surface.blit(rotated_leaf, leaf_rect)
                             except pygame.error as leaf_err:
                                 print(f"Warning: Error drawing leaf: {leaf_err}")
             if progress_ratio > 0.9 and len(self.vine_points) > 0:
                 if len(self.vine_points) > 0:
                     top_x, top_y = self.vine_points[-1]
                     petal_color = (255, 200, 100)
                     center_color = SUNSET_ORANGE
                     try:
                         for angle_deg in range(0, 360, 60):
                             angle_rad = math.radians(angle_deg)
                             petal_x = top_x + 8 * math.cos(angle_rad)
                             petal_y = top_y + 8 * math.sin(angle_rad)
                             pygame.draw.circle(surface, petal_color, (int(petal_x), int(petal_y)), 7)
                         pygame.draw.circle(surface, center_color, (int(top_x), int(top_y)), 5)
                     except pygame.error as flower_err:
                         print(f"Warning: Error drawing flower: {flower_err}")

    def draw_water_progress(self, surface, progress_ratio):
        fill_height = int(self.height * progress_ratio)
        if fill_height > 0:
            fill_rect = pygame.Rect(self.x, self.y + self.height - fill_height, self.width, fill_height)
            pygame.draw.rect(surface, self.color, fill_rect)
            wave_height_amp = 3
            wave_y_base = self.y + self.height - fill_height - wave_height_amp
            try:
                wave_surface = pygame.Surface((self.width, wave_height_amp * 2), pygame.SRCALPHA)
                lighter_color = LIGHT_BLUE
                num_points = self.width // 5
                if num_points < 2: return
                wave_points = [(0, wave_height_amp * 2)]
                for i in range(num_points + 1):
                    x = int(i * (self.width / num_points))
                    offset = math.sin(time.time() * 3 + x * 0.05) * wave_height_amp
                    wave_points.append((x, wave_height_amp + offset))
                wave_points.append((self.width, wave_height_amp * 2))
                pygame.draw.polygon(wave_surface, lighter_color, wave_points)
                surface.blit(wave_surface, (self.x, wave_y_base))
            except pygame.error as wave_err:
                 print(f"Warning: Error drawing wave: {wave_err}")

class EcoButton:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=WHITE, border_radius=10):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False
        self.animation = 0

    def draw(self, surface):
        try:
            current_color = self.hover_color if self.hovered else self.color
            if self.animation > 0:
                self.animation -= 0.1
                feedback_color = tuple(max(0, c - 30) for c in current_color[:3])
                current_color = feedback_color
            pygame.draw.rect(surface, current_color, self.rect)
            highlight_rect = pygame.Rect(self.rect.x + 5, self.rect.y + 3, self.rect.width - 10, self.rect.height // 4)
            highlight_color = (*WHITE[:3], 30)
            pygame.draw.rect(surface, highlight_color, highlight_rect)
            pygame.draw.rect(surface, WOOD_BROWN, self.rect, 2)
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
        self.animation = 1.0
        print(f"Button '{self.text}' clicked.")
        return True

class DetectionAnimation:
    def __init__(self, item_name, item_type):
        self.item_name = item_name.upper()
        self.item_type = item_type.upper()
        self.start_time = time.time()
        self.phase = "dropping"
        self.phase_durations = {"dropping": 1.2, "scanning": 2.0, "revealing": 1.5, "feedback": 4.0}
        self.total_duration = sum(self.phase_durations.values())
        self.stats_updated = False
        self.scan_progress = 0
        self.reveal_alpha = 0
        self.particle_effects = []
        self.item_color = LEAF_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
        self.item_image = self.create_item_image()
        self.y_pos = -100
        self.rotation = random.uniform(-15, 15)
        self.rotation_speed = random.uniform(-2, 2)
        self.fall_speed = 8
        self.target_y = SCREEN_HEIGHT // 2 - 60
        self.loading_width = 280
        self.loading_height = 18
        self.loading_x = SCREEN_WIDTH // 2 - self.loading_width // 2
        self.loading_y = self.target_y + 90
        self.correct_button = None
        self.incorrect_button = None

    def create_item_image(self):
        try:
            size = 80
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            base_color = LEAF_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
            border_color = DARK_GREEN if self.item_type == "RECYCLING" else WOOD_BROWN
            if self.item_type == "RECYCLING":
                 pygame.draw.rect(surf, base_color, (size*0.2, size*0.1, size*0.6, size*0.8))
                 pygame.draw.rect(surf, border_color, (size*0.2, size*0.1, size*0.6, size*0.8), 2)
                 center_x, center_y = size // 2, size // 2 + 10
                 radius = size // 6
                 pygame.draw.circle(surf, WHITE, (center_x, center_y), radius, 2)
                 for i in range(3):
                     angle = math.radians(i * 120 + 30)
                     start_angle = angle - math.radians(15)
                     end_angle = angle + math.radians(15)
                     p1 = (center_x + radius * math.cos(start_angle), center_y + radius * math.sin(start_angle))
                     p2 = (center_x + (radius+5) * math.cos(angle), center_y + (radius+5) * math.sin(angle))
                     p3 = (center_x + radius * math.cos(end_angle), center_y + radius * math.sin(end_angle))
                     pygame.draw.line(surf, WHITE, p1, p2, 2)
                     pygame.draw.line(surf, WHITE, p2, p3, 2)
            else:
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
            highlight_color = (*WHITE[:3], 60)
            pygame.draw.circle(surf, highlight_color, (size // 3, size // 3), size // 5)
            return surf
        except pygame.error as e:
            print(f"Warning: Error creating item image: {e}")
            return None

    def update(self):
        if not self.item_image: return True
        try:
            current_time = time.time()
            elapsed_total = current_time - self.start_time
            if elapsed_total < self.phase_durations["dropping"]:
                self.phase = "dropping"
                self.y_pos += self.fall_speed
                self.rotation += self.rotation_speed
            elif elapsed_total < self.phase_durations["dropping"] + self.phase_durations["scanning"]:
                self.phase = "scanning"
                elapsed_in_phase = elapsed_total - self.phase_durations["dropping"]
                self.scan_progress = min(100, (elapsed_in_phase / self.phase_durations["scanning"]) * 100)
                if self.y_pos < self.target_y:
                    self.y_pos = self.target_y
                    self.fall_speed = 0
            elif elapsed_total < self.phase_durations["dropping"] + self.phase_durations["scanning"] + self.phase_durations["revealing"]:
                self.phase = "revealing"
                elapsed_in_phase = elapsed_total - (self.phase_durations["dropping"] + self.phase_durations["scanning"])
                progress = min(1.0, elapsed_in_phase / self.phase_durations["revealing"])
                self.reveal_alpha = int(255 * (progress**0.5))
            else:
                self.phase = "feedback"
                self.reveal_alpha = 255
            if self.phase == "feedback" and not self.stats_updated:
                print(f"GUI: Feedback phase started for {self.item_name}.")
                self.stats_updated = True  # Mark as updated after processing
                # Update core stats for record keeping
                # (you may still want to update your BinStats)
                # self.stats.update_stats(self.item_type)  # Uncomment if needed
                if self.item_type.upper() == "RECYCLING":
                     new_value = min(smart_bin_interface.recycling_progress.current_value + 25, 100)
                     smart_bin_interface.recycling_progress.set_value(new_value)
                     smart_bin_interface.stats.recycled_items += 1
                     print(f"GUI: Recycling progress updated to {new_value}%")
                elif self.item_type.upper() == "TRASH":
                     new_value = min(smart_bin_interface.landfill_progress.current_value + 25, 100)
                     smart_bin_interface.landfill_progress.set_value(new_value)
                     smart_bin_interface.stats.landfill_items += 1
                     print(f"GUI: Landfill progress updated to {new_value}%")
            if self.phase == "dropping" and self.y_pos >= self.target_y:
                self.y_pos = self.target_y
                self.fall_speed = 0
                self.rotation_speed *= 0.95
            if self.phase == "scanning" and random.random() < 0.15:
                self.add_natural_particle("scan")
            if self.phase == "revealing" and random.random() < 0.25:
                self.add_natural_particle("reveal")
            self.update_particles()
            return elapsed_total >= self.total_duration
        except (TypeError, ValueError, ZeroDivisionError) as e:
            print(f"Warning: Error updating animation: {e}")
            return True

    def add_natural_particle(self, phase_type):
        start_x = SCREEN_WIDTH // 2
        start_y = self.y_pos + (self.item_image.get_height() // 2 if self.item_image else 0)
        if phase_type == "scan":
            start_x = self.loading_x + (self.loading_width * self.scan_progress / 100)
            start_y = self.loading_y + self.loading_height / 2
            angle = random.uniform(math.pi * 1.2, math.pi * 1.8)
            speed = random.uniform(0.5, 1.5)
            ptype = "sparkle"; color = SUNSET_ORANGE; size = random.uniform(1, 3)
            vy_initial = math.sin(angle) * speed - 0.5
        else:
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
            "life": random.uniform(0.8, 1.5)
        })

    def update_particles(self):
        for i in range(len(self.particle_effects) - 1, -1, -1):
             if i >= len(self.particle_effects): continue
             p = self.particle_effects[i]
             p["x"] += p["vx"]
             p["y"] += p["vy"]
             p["vy"] += 0.05
             p["rotation"] += p["rot_speed"]
             p["life"] -= 0.015
             if p["life"] <= 0:
                 if 0 <= i < len(self.particle_effects):
                      self.particle_effects.pop(i)

    def draw_particles(self, surface):
        for p in self.particle_effects:
            try:
                life_ratio = max(0, p.get('life', 0) / 1.0)
                alpha = int(255 * min(1, life_ratio * 2))
                if alpha <= 0: continue
                particle_type = p.get("type")
                size = p.get("size", 0)
                if size <= 0: continue
                if particle_type == "leaf":
                    part_surf = pygame.Surface((size * 3, size * 2), pygame.SRCALPHA)
                    color = (*p.get("color", LEAF_GREEN)[:3], alpha)
                    pygame.draw.ellipse(part_surf, color, (0, 0, size * 2, size * 1.5))
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
                    for i in range(num_lines):
                         angle = rotation + i * (360 / num_lines)
                         rad = math.radians(angle)
                         end_x = center_x + math.cos(rad) * size * 1.5
                         end_y = center_y + math.sin(rad) * size * 1.5
                         pygame.draw.line(surface, color, (center_x, center_y), (int(end_x), int(end_y)), 1)
            except (pygame.error, KeyError, TypeError, ValueError, AttributeError) as e:
                print(f"Warning: Error drawing particle (type: {p.get('type', 'unknown')}, life: {p.get('life', '?')}, error: {e}). Skipping draw.")
                continue

    def draw(self, surface):
        if not self.item_image: return
        try:
            rotated_image = pygame.transform.rotate(self.item_image, self.rotation)
            rotated_rect = rotated_image.get_rect(center=(SCREEN_WIDTH // 2, int(self.y_pos)))
            surface.blit(rotated_image, rotated_rect)
            if self.phase == "scanning":
                loading_bg_rect = pygame.Rect(self.loading_x, self.loading_y+25, self.loading_width, self.loading_height)
                pygame.draw.rect(surface, LIGHT_BROWN, loading_bg_rect)
                fill_color = LEAF_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
                fill_width = int(self.loading_width * (self.scan_progress / 100))
                if fill_width > 0:
                     fill_rect = pygame.Rect(self.loading_x, self.loading_y+25, fill_width, self.loading_height)
                     pygame.draw.rect(surface, fill_color, fill_rect)
                pygame.draw.rect(surface, WOOD_BROWN, loading_bg_rect, 2)
                text = font_medium.render("Analyzing...", True, TEXT_BROWN)
                text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, self.loading_y - 20))
                surface.blit(text, text_rect)
            elif self.phase in ["revealing", "feedback"]:
                text_center_x = SCREEN_WIDTH // 2
                text_base_y = self.y_pos + rotated_rect.height // 2 + 20
                name_surf = font_large.render(self.item_name, True, TEXT_BROWN)
                name_surf.set_alpha(self.reveal_alpha)
                name_rect = name_surf.get_rect(center=(text_center_x, text_base_y + 40))
                name_bg_rect = name_rect.inflate(20, 10)
                name_bg_surf = pygame.Surface(name_bg_rect.size, pygame.SRCALPHA)
                name_bg_surf.fill((*CREAM[:3], int(180 * (self.reveal_alpha / 255))))
                pygame.draw.rect(name_bg_surf, (*WOOD_BROWN[:3], int(200 * (self.reveal_alpha / 255))), name_bg_surf.get_rect(), 2)
                surface.blit(name_bg_surf, name_bg_rect.topleft)
                surface.blit(name_surf, name_rect)
                type_color = DARK_GREEN if self.item_type == "RECYCLING" else SOFT_BROWN
                type_text = "Recyclable" if self.item_type == "RECYCLING" else "Landfill"
                type_surf = font_medium.render(type_text, True, type_color)
                type_surf.set_alpha(self.reveal_alpha)
                type_rect = type_surf.get_rect(center=(text_center_x, text_base_y + 150))
                type_bg_rect = type_rect.inflate(15, 8)
                type_bg_surf = pygame.Surface(type_bg_rect.size, pygame.SRCALPHA)
                type_bg_surf.fill((*CREAM[:3], int(160 * (self.reveal_alpha / 255))))
                pygame.draw.rect(type_bg_surf, (*WOOD_BROWN[:3], int(180 * (self.reveal_alpha / 255))), type_bg_surf.get_rect(), 1)
                surface.blit(type_bg_surf, type_bg_rect.topleft)
                surface.blit(type_surf, type_rect)
                if self.phase == "feedback":
                    if self.correct_button is None or self.incorrect_button is None:
                        print("Warning: Buttons not set in DetectionAnimation for positioning feedback prompt.")
                        prompt_center_y = SCREEN_HEIGHT - 150
                    else:
                        prompt_y_offset = 40
                        prompt_center_y = self.correct_button.rect.top - prompt_y_offset
                    time_left = max(0, self.total_duration - (time.time() - self.start_time))
                    prompt_text = f"Is this correct? ({int(time_left)}s)"
                    prompt_surf = font_medium.render(prompt_text, True, TEXT_BROWN)
                    prompt_rect = prompt_surf.get_rect(center=(SCREEN_WIDTH // 2, prompt_center_y))
                    prompt_bg_rect = prompt_rect.inflate(20, 10)
                    prompt_bg_surf = pygame.Surface(prompt_bg_rect.size, pygame.SRCALPHA)
                    prompt_bg_surf.fill((*LIGHT_CREAM[:3], 190))
                    pygame.draw.rect(prompt_bg_surf, WOOD_BROWN, prompt_bg_surf.get_rect(), 2)
                    surface.blit(prompt_bg_surf, prompt_bg_rect.topleft)
                    surface.blit(prompt_surf, prompt_rect)
            self.draw_particles(surface)
        except (pygame.error, TypeError, ValueError, ZeroDivisionError, AttributeError) as e:
             print(f"Warning: Error drawing detection animation phase '{self.phase}': {e}")
             self.phase = "feedback"
             self.start_time = time.time() - self.total_duration

class SmartBinInterface:
    def __init__(self):
        self.stats = BinStats()
        self.nature_elements = [NatureElement() for _ in range(15)]
        self.state = "idle"
        self.detection_animation = None
        pb_rel_height = 0.55; pb_rel_width = 0.2
        pb_height = int(SCREEN_HEIGHT * pb_rel_height); pb_width = int(SCREEN_WIDTH * pb_rel_width)
        pb_y_offset = 40
        pb_y = (SCREEN_HEIGHT // 2) - (pb_height // 2) + pb_y_offset
        pb_x_margin = int(SCREEN_WIDTH * 0.08)
        landfill_pb_x = (SCREEN_WIDTH // 2) - pb_x_margin - pb_width
        recycling_pb_x = (SCREEN_WIDTH // 2) + pb_x_margin
        self.landfill_progress = NaturalProgressBar(landfill_pb_x, pb_y, pb_width, pb_height, WATER_BLUE, LIGHT_CREAM, style="water")
        self.recycling_progress = NaturalProgressBar(recycling_pb_x, pb_y, pb_width, pb_height, LEAF_GREEN, LIGHT_CREAM, style="plant")
        self.recycling_progress.set_value(0)
        self.landfill_progress.set_value(0)
        button_width = int(SCREEN_WIDTH * 0.12); button_height = int(SCREEN_HEIGHT * 0.07)
        button_y = SCREEN_HEIGHT - button_height - int(SCREEN_HEIGHT * 0.08)
        button_gap = int(SCREEN_WIDTH * 0.02)
        self.correct_button = EcoButton(SCREEN_WIDTH // 2 - button_width - button_gap // 2, button_y, button_width, button_height, "Correct", LEAF_GREEN, LIGHT_GREEN)
        self.incorrect_button = EcoButton(SCREEN_WIDTH // 2 + button_gap // 2, button_y, button_width, button_height, "Incorrect", SOFT_BROWN, LIGHT_BROWN)
        self.last_hint_time = 0; self.hint_interval = 25
        self.hints = ["Consolimate™ uses AI to sort waste accurately.", "Recycling reduces landfill waste and saves resources.", "Contamination ruins recycling - please sort carefully!", "Over 95% accuracy in waste sorting!", "Thank you for helping keep the planet green!", "Plastic bottles take over 400 years to decompose.", "Glass is infinitely recyclable!", "Recycle paper 5-7 times before fibers weaken.", "Check local guidelines for specific recycling rules.", "Reduce, Reuse, Recycle - in that order!"]
        self.current_hint = random.choice(self.hints); self.hint_alpha = 0; self.hint_fade_in = True; self.hint_display_duration = 8; self.hint_fade_duration = 1.5; self.hint_state = "fading_in"; self.hint_visible_start_time = 0

    def update_nature_elements(self):
        for element in self.nature_elements: element.update()
    def draw_nature_elements(self, surface):
        for element in self.nature_elements: element.draw(surface)
    def update_progress_bars(self):
        self.recycling_progress.update()
        self.landfill_progress.update()
    def draw_progress_bars(self, surface):
        try:
            card_padding_x = int(SCREEN_WIDTH * 0.02)
            card_padding_y_top = int(SCREEN_HEIGHT * 0.08)
            card_padding_y_bottom = int(SCREEN_HEIGHT * 0.12)
            landfill_card_rect = pygame.Rect(
                self.landfill_progress.x - card_padding_x,
                self.landfill_progress.y - card_padding_y_top,
                self.landfill_progress.width + 2 * card_padding_x,
                self.landfill_progress.height + card_padding_y_top + card_padding_y_bottom
            )
            recycling_card_rect = pygame.Rect(
                self.recycling_progress.x - card_padding_x,
                self.recycling_progress.y - card_padding_y_top,
                self.recycling_progress.width + 2 * card_padding_x,
                self.recycling_progress.height + card_padding_y_top + card_padding_y_bottom
            )
            pygame.draw.rect(surface, CREAM, landfill_card_rect)
            pygame.draw.rect(surface, WOOD_BROWN, landfill_card_rect, 2)
            pygame.draw.rect(surface, CREAM, recycling_card_rect)
            pygame.draw.rect(surface, WOOD_BROWN, recycling_card_rect, 2)
            self.landfill_progress.draw(surface)
            self.recycling_progress.draw(surface)
            landfill_label = font_medium.render("Landfill", True, TEXT_BROWN)
            landfill_rect = landfill_label.get_rect(center=(self.landfill_progress.x + self.landfill_progress.width / 2, self.landfill_progress.y - card_padding_y_top / 2))
            surface.blit(landfill_label, landfill_rect)
            recycling_label = font_medium.render("Recycling", True, DARK_GREEN)
            recycling_rect = recycling_label.get_rect(center=(self.recycling_progress.x + self.recycling_progress.width / 2, self.recycling_progress.y - card_padding_y_top / 2))
            surface.blit(recycling_label, recycling_rect)
            count_y = self.landfill_progress.y + self.landfill_progress.height + card_padding_y_bottom / 2
            landfill_count_text = f"{self.stats.landfill_items} items"
            landfill_count_surf = font_small.render(landfill_count_text, True, TEXT_BROWN)
            landfill_count_rect = landfill_count_surf.get_rect(center=(landfill_rect.centerx, count_y))
            surface.blit(landfill_count_surf, landfill_count_rect)
            recycling_count_text = f"{self.stats.recycled_items} items"
            recycling_count_surf = font_small.render(recycling_count_text, True, DARK_GREEN)
            recycling_count_rect = recycling_count_surf.get_rect(center=(recycling_rect.centerx, count_y))
            surface.blit(recycling_count_surf, recycling_count_rect)
            if self.stats.total_items > 0:
                 impact_card_height = int(SCREEN_HEIGHT * 0.14)
                 impact_card_width = int(SCREEN_WIDTH * 0.45)
                 impact_card_y = SCREEN_HEIGHT - impact_card_height - int(SCREEN_HEIGHT * 0.02)
                 impact_card_x = SCREEN_WIDTH // 2 - impact_card_width // 2
                 impact_card = pygame.Rect(impact_card_x, impact_card_y, impact_card_width, impact_card_height)
                 pygame.draw.rect(surface, CREAM, impact_card)
                 pygame.draw.rect(surface, WOOD_BROWN, impact_card, 2)
                 impact_title = font_medium.render("Environmental Impact", True, TEXT_BROWN)
                 impact_title_rect = impact_title.get_rect(center=(impact_card.centerx, impact_card.y + impact_card_height * 0.25))
                 surface.blit(impact_title, impact_title_rect)
                 stats_y = impact_card.y + impact_card_height * 0.68
                 icon_size = int(SCREEN_HEIGHT * 0.03)
                 co2_text = f"{self.stats.co2_saved:.1f} kg CO₂"
                 co2_surf = font_small.render(co2_text, True, DARK_GREEN)
                 co2_rect = co2_surf.get_rect(midright=(impact_card.centerx - int(impact_card_width * 0.05), stats_y))
                 surface.blit(co2_surf, co2_rect)
                 leaf_icon_rect = pygame.Rect(0, 0, icon_size, int(icon_size * 0.8))
                 leaf_icon_rect.midright = (co2_rect.left - int(impact_card_width * 0.02), stats_y)
                 pygame.draw.ellipse(surface, LEAF_GREEN, leaf_icon_rect)
                 water_text = f"{self.stats.water_saved:.1f} L Water"
                 water_surf = font_small.render(water_text, True, WATER_BLUE)
                 water_rect = water_surf.get_rect(midleft=(impact_card.centerx + int(impact_card_width * 0.05), stats_y))
                 surface.blit(water_surf, water_rect)
                 drop_icon_rect = pygame.Rect(0, 0, int(icon_size * 0.7), icon_size)
                 drop_icon_rect.midleft = (water_rect.right + int(impact_card_width * 0.02), stats_y)
                 pygame.draw.ellipse(surface, WATER_BLUE, drop_icon_rect)
        except (pygame.error, TypeError, ValueError) as e:
            print(f"Warning: Error drawing progress bars/stats: {e}")
            pygame.draw.rect(surface, CREAM, (50, 50, SCREEN_WIDTH-100, SCREEN_HEIGHT-100), 5)
            error_text = font_small.render("Error drawing stats", True, BLACK)
            screen.blit(error_text, (60, 60))
    def update_hint(self):
        current_time = time.time()
        time_since_change = current_time - self.last_hint_time
        if self.hint_state == "fading_in":
            if time_since_change < self.hint_fade_duration:
                self.hint_alpha = int(255 * (time_since_change / self.hint_fade_duration))
            else:
                self.hint_alpha = 255
                self.hint_state = "visible"
                self.hint_visible_start_time = current_time
        elif self.hint_state == "visible":
            time_visible = current_time - self.hint_visible_start_time
            if time_visible >= self.hint_display_duration:
                self.hint_state = "fading_out"
                self.last_hint_time = current_time
        elif self.hint_state == "fading_out":
            time_fading = current_time - self.last_hint_time
            if time_fading < self.hint_fade_duration:
                self.hint_alpha = int(255 * (1 - (time_fading / self.hint_fade_duration)))
            else:
                self.hint_alpha = 0
                self.current_hint = random.choice(self.hints)
                self.hint_state = "fading_in"
                self.last_hint_time = current_time
        self.hint_alpha = max(0, min(255, self.hint_alpha))
        '''
    def draw_hint(self, surface):
        if self.hint_alpha <= 0: return
        try:
            hint_surf = font_small.render(self.current_hint, True, TEXT_BROWN)
            hint_rect = hint_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - int(SCREEN_HEIGHT*0.08)))
            bg_rect = hint_rect.inflate(int(SCREEN_WIDTH*0.04), int(SCREEN_HEIGHT*0.03))
            hint_back = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_color_alpha = (*CREAM[:3], int(200 * (self.hint_alpha / 255)))
            pygame.draw.rect(hint_back, bg_color_alpha, hint_back.get_rect())
            border_color_alpha = (*WOOD_BROWN[:3], int(220 * (self.hint_alpha / 255)))
            pygame.draw.rect(hint_back, border_color_alpha, hint_back.get_rect(), 2)
            surface.blit(hint_back, bg_rect.topleft)
            hint_surf.set_alpha(self.hint_alpha)
            surface.blit(hint_surf, hint_rect)
        except pygame.error as e:
            print(f"Warning: Error drawing hint: {e}")'
            '''
    def process_camera_detection(self, detection_data):
        if self.state == "idle" and isinstance(detection_data, dict):
            item_name = detection_data.get("name", "Unknown Item")
            item_type = detection_data.get("type", "TRASH")
            if item_type.upper() not in ["RECYCLING", "TRASH"]:
                print(f"Warning: Received invalid item type '{item_type}'. Defaulting to TRASH.")
                item_type = "TRASH"
            print(f"GUI: Received detection - Item: {item_name}, Type: {item_type}")
            self.detection_animation = DetectionAnimation(item_name, item_type)
            self.detection_animation.correct_button = self.correct_button
            self.detection_animation.incorrect_button = self.incorrect_button
            self.state = "detecting"
        elif not isinstance(detection_data, dict):
            print(f"Warning: Received invalid detection data format: {type(detection_data)}")
    def update_detection(self):
        try:
            if self.state == "idle":
                 if not detection_queue.empty():
                     detection_data = detection_queue.get()
                     self.process_camera_detection(detection_data)
                     detection_queue.task_done()
            if self.detection_animation:
                animation_finished = self.detection_animation.update()
                # --- NEW LOGIC: Increment progress bar by 10% per detection ---
                # This logic now adds 10 to the corresponding progress bar when the detection animation reaches feedback
                # The update in the DetectionAnimation.update() method now calls the progress bar update (via smart_bin_interface)
                if animation_finished:
                    print("GUI: Detection animation finished.")
                    self.detection_animation = None
                    self.state = "idle"
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in update_detection: {e}")
            traceback.print_exc()
            self.detection_animation = None
            self.state = "idle"
    def draw_detection(self, surface):
        try:
            if self.detection_animation:
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay_alpha = 100
                if self.detection_animation.phase == "dropping":
                     overlay_alpha = int(100 * (self.detection_animation.y_pos / self.detection_animation.target_y)) if self.detection_animation.target_y > 0 else 0
                elif self.detection_animation.phase == "scanning":
                     overlay_alpha = 100 + int(50 * (self.detection_animation.scan_progress / 100))
                elif self.detection_animation.phase == "revealing":
                     overlay_alpha = 150 + int(105 * (self.detection_animation.reveal_alpha / 255))
                elif self.detection_animation.phase == "feedback":
                     overlay_alpha = 200
                overlay.fill((0, 0, 0, max(0, min(200, overlay_alpha))))
                surface.blit(overlay, (0, 0))
                self.detection_animation.draw(surface)
        except (pygame.error, AttributeError) as e:
            print(f"Warning: Error drawing detection overlay/animation: {e}")
    def handle_button_clicks(self, mouse_pos):
        clicked = False
        if self.state == "detecting" and self.detection_animation and self.detection_animation.phase == "feedback":
            if self.correct_button.hovered and self.correct_button.clicked():
                 print("Feedback: Correct button clicked.")
                 clicked = True
            elif self.incorrect_button.hovered and self.incorrect_button.clicked():
                 print("Feedback: Incorrect button clicked.")
                 clicked = True
            if clicked:
                self.detection_animation = None
                self.state = "idle"
        return clicked
    def check_button_hover(self, mouse_pos):
        if self.state == "detecting" and self.detection_animation and self.detection_animation.phase == "feedback":
            try:
                self.correct_button.check_hover(mouse_pos)
                self.incorrect_button.check_hover(mouse_pos)
            except (AttributeError, TypeError) as e:
                 print(f"Warning: Error checking button hover state: {e}")
        else:
            self.correct_button.hovered = False
            self.incorrect_button.hovered = False

def draw_background(surface):
    try:
        for y in range(SCREEN_HEIGHT):
            ratio = y / SCREEN_HEIGHT
            r = int(LIGHT_CREAM[0] * (1 - ratio) + CREAM[0] * ratio)
            g = int(LIGHT_CREAM[1] * (1 - ratio) + CREAM[1] * ratio)
            b = int(LIGHT_CREAM[2] * (1 - ratio) + CREAM[2] * ratio)
            pygame.draw.line(surface, (r, g, b), (0, y), (SCREEN_WIDTH, y))
    except pygame.error as e:
        print(f"Warning: Error drawing background: {e}")
        surface.fill(CREAM)

def draw_header(surface):
    try:
        header_height = int(SCREEN_HEIGHT * 0.12)
        header_rect = pygame.Rect(0, 0, SCREEN_WIDTH, header_height)
        pygame.draw.rect(surface, CREAM, header_rect)
        for y in range(0, header_height, 3):
            grain_width = random.randint(int(SCREEN_WIDTH * 0.8), SCREEN_WIDTH)
            grain_x = random.randint(0, int(SCREEN_WIDTH * 0.2))
            grain_alpha = random.randint(8, 18)
            grain_line = pygame.Surface((grain_width, 1), pygame.SRCALPHA)
            grain_line.fill((*SOFT_BROWN[:3], grain_alpha))
            surface.blit(grain_line, (grain_x, y))
        pygame.draw.rect(surface, WOOD_BROWN, (0, header_height - 2, SCREEN_WIDTH, 2))
        title_text = font_title.render("Consolimate™ Waste Management", True, TEXT_BROWN)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, header_height//2))
        surface.blit(title_text, title_rect)
        leaf_size = int(header_height * 0.3)
        leaf_surf = pygame.Surface((leaf_size * 1.5, leaf_size), pygame.SRCALPHA)
        pygame.draw.ellipse(leaf_surf, LEAF_GREEN, (0, 0, leaf_size * 1.5, leaf_size))
        pygame.draw.line(leaf_surf, DARK_GREEN, (leaf_size * 0.5, leaf_size / 2), (leaf_size * 1.5, leaf_size / 2), 2)
        leaf_left_rect = leaf_surf.get_rect(center=(title_rect.left - int(SCREEN_WIDTH*0.05), title_rect.centery))
        surface.blit(leaf_surf, leaf_left_rect)
        leaf_right_surf = pygame.transform.flip(leaf_surf, True, False)
        leaf_right_rect = leaf_right_surf.get_rect(center=(title_rect.right + int(SCREEN_WIDTH*0.05), title_rect.centery))
        surface.blit(leaf_right_surf, leaf_right_rect)
    except pygame.error as e:
        print(f"Warning: Error drawing header: {e}")
        pygame.draw.rect(surface, CREAM, (0, 0, SCREEN_WIDTH, 65))
        pygame.draw.line(surface, WOOD_BROWN, (0, 63), (SCREEN_WIDTH, 63), 2)

def simulate_detection(queue):
    items = [
        {"name": "Plastic Bottle", "type": "RECYCLING"},
        {"name": "Apple Core", "type": "TRASH"},
        {"name": "Aluminum Can", "type": "RECYCLING"},
        {"name": "Paper Towel", "type": "TRASH"},
        {"name": "Newspaper", "type": "RECYCLING"},
        {"name": "Coffee Cup", "type": "TRASH"},
        {"name": "Glass Jar", "type": "RECYCLING"}
    ]
    chosen_item = random.choice(items)
    print(f"\n--- SIMULATING DETECTION: {chosen_item['name']} ({chosen_item['type']}) ---")
    queue.put(chosen_item)

def gui_server_thread(host, port, data_queue, running_flag_func):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"[Server Thread] Listening on {host}:{port}")
        server_socket.settimeout(1.0)
        while running_flag_func():
            try:
                conn, addr = server_socket.accept()
                with conn:
                    print(f"[Server Thread] Accepted connection from {addr}")
                    data = b""
                    conn.settimeout(2.0)
                    while True:
                        try:
                            chunk = conn.recv(1024)
                            if not chunk: break
                            data += chunk
                            if data.strip().endswith(b'}'):
                                break
                        except socket.timeout:
                             print("[Server Thread] Receive timeout.")
                             break
                    if data:
                        try:
                            message_str = data.decode('utf-8').strip()
                            potential_jsons = message_str.split('}')
                            processed_count = 0
                            for part in potential_jsons:
                                if not part.strip(): continue
                                json_str = part + '}'
                                try:
                                    detection_data = json.loads(json_str)
                                    if (isinstance(detection_data, dict) and
                                        "type" in detection_data and
                                        "name" in detection_data):
                                        data_queue.put(detection_data)
                                        processed_count += 1
                                        print(f"[Server Thread] Valid data added to queue: {detection_data}")
                                    else:
                                        print(f"[Server Thread] Warning: Parsed data part is not in expected format: {json_str}")
                                except json.JSONDecodeError as e:
                                     if part != potential_jsons[-1]:
                                          pass
                                     else:
                                          print(f"[Server Thread] Error decoding JSON part: {e} - Part: {json_str}")
                        except UnicodeDecodeError as e:
                            print(f"[Server Thread] Error decoding UTF-8: {e}")
                        except Exception as e:
                            print(f"[Server Thread] Error processing received data: {e}")
                    else:
                        print("[Server Thread] No data received or connection closed early.")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Server Thread] Error accepting connection: {e}")
                time.sleep(0.5)
    except OSError as e:
         print(f"[Server Thread] CRITICAL OS Error binding socket: {e}")
    except Exception as e:
        print(f"[Server Thread] Error setting up server socket: {e}")
    finally:
        print("[Server Thread] Shutting down...")
        server_socket.close()
        print("[Server Thread] Socket closed.")

def main():
    global server_running, smart_bin_interface
    server_thread = None
    try:
        smart_bin_interface = SmartBinInterface()
        clock = pygame.time.Clock()
        running = True
        server_thread = threading.Thread(target=gui_server_thread,
                                         args=(GUI_SERVER_HOST, GUI_SERVER_PORT, detection_queue, lambda: server_running),
                                         daemon=True)
        server_thread.start()
        print("GUI Server thread started.")
        print("-" * 30)
        print("SmartBin GUI Initialized.")
        print(f"Listening for detections on {GUI_SERVER_HOST}:{GUI_SERVER_PORT}")
        print("Press 'D' to simulate a detection event.")
        print("Press 'ESC' to quit.")
        print("-" * 30)
        last_update_time = time.time()
        while running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    server_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        server_running = False
                    if event.key == pygame.K_d:
                         if smart_bin_interface.state == "idle":
                             simulate_detection(detection_queue)
                         else:
                             print("Simulation ignored: Detection already in progress.")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                     if event.button == 1:
                         smart_bin_interface.handle_button_clicks(event.pos)
            current_time = time.time()
            delta_time = current_time - last_update_time
            last_update_time = current_time
            smart_bin_interface.check_button_hover(mouse_pos)
            smart_bin_interface.update_nature_elements()
            smart_bin_interface.update_progress_bars()
            smart_bin_interface.update_hint()
            smart_bin_interface.update_detection()
            try:
                draw_background(screen)
                smart_bin_interface.draw_nature_elements(screen)
                draw_header(screen)
                if smart_bin_interface.state == "idle":
                    smart_bin_interface.draw_progress_bars(screen)
                    correct_animating = hasattr(smart_bin_interface.correct_button, 'animation') and smart_bin_interface.correct_button.animation > 0
                    incorrect_animating = hasattr(smart_bin_interface.incorrect_button, 'animation') and smart_bin_interface.incorrect_button.animation > 0

                else:
                    smart_bin_interface.draw_detection(screen)
            except Exception as draw_err:
                print(f"CRITICAL error during drawing phase: {draw_err}")
                traceback.print_exc()
                screen.fill(CREAM)
                try:
                    err_font = pygame.font.SysFont("Arial", 20)
                    err_text = err_font.render("Drawing Error Occurred - Check Console", True, BLACK)
                    screen.blit(err_text, (50, 50))
                except Exception as font_err:
                    print(f"Could not render error text to screen: {font_err}")
            pygame.display.flip()
            clock.tick(60)
    except Exception as main_err:
        print(f"CRITICAL error in main loop: {main_err}")
        traceback.print_exc()
    finally:
        print("Shutting down Pygame and Server...")
        server_running = False
        if server_thread and server_thread.is_alive():
            print("Waiting for server thread to join...")
            server_thread.join(timeout=2.0)
            if server_thread.is_alive():
                 print("Warning: Server thread did not join cleanly.")
        else:
             print("Server thread already stopped or not started.")
        pygame.quit()
        print("Application terminated.")
        sys.exit()

if __name__ == "__main__":
    main()
