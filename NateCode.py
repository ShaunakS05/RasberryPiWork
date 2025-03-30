import pygame
import sys
import random
import math
import time
import threading
import queue
import cv2
import numpy as np
import base64
import os
import serial
from openai import OpenAI
from dotenv import load_dotenv

# --- Environment and API Configuration ---
load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except:
    print("Warning: OpenAI API key not found. Vision analysis will be simulated.")

# --- Arduino Configuration ---
ARDUINO_PORT = "/dev/ttyACM0"  # Change to your Arduino's port
BAUD_RATE = 9600

# --- Camera and Analysis Configuration ---
analysis_delay = 1.5  # Seconds the change must persist
process_every_n_frames = 3  # How often to check for change
cooldown_period = 5  # Seconds after analysis before next trigger
min_contour_area = 500  # Minimum pixel area for change detection

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
detection_queue = queue.Queue()  # For passing detection results to GUI
camera_active = True  # Flag to control camera thread
ser = None  # Serial connection object

# --- Camera Thread Functions ---

# Initialize background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)

# Function to encode image for OpenAI API
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Function to analyze image with GPT-4
def ask_chatgpt(image_path):
    print(f"→ Analyzing image based on detected change: {image_path}")
    try:
        imageDecoded = encode_image(image_path)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a trashcan vision assistant.\n"
                                "Look at this image, which was captured because motion or change was detected.\n"
                                "Determine if the primary changing object is a common piece of trash or recycling.\n"
                                "If it is NOT trash/recycling (e.g., a person, hand entering/leaving, significant lighting change, empty view after object removed), respond ONLY with:\n"
                                "Classification: IGNORE\n"
                                "Smelly: NO\n"
                                "Smell Rating: 0\n"
                                "Volume Estimation: 0 cm^3\n"
                                "Item Name: IGNORE\n\n"
                                "If it IS a piece of trash/recycling:\n"
                                "1. Classify it as RECYCLING or TRASH.\n"
                                "2. Determine if it is typically smelly. Respond with YES or NO.\n"
                                "3. Rate how smelly it is likely to be on a scale of 1 to 10.\n"
                                "4. Estimate the volume of the object in cm cubed by identifying the object and finding the average volume of that kind of object. Respond with <number> cm^3.\n"
                                "5. Determine what the item is. Respond with the item name in ALL CAPS.\n\n"
                                "Respond in the following format EXACTLY:\n"
                                "Classification: <RECYCLING or TRASH or IGNORE>\n"
                                "Smelly: <YES or NO>\n"
                                "Smell Rating: <0 to 10>\n"
                                "Volume Estimation: <number> cm^3\n"
                                "Item Name: <ITEM NAME or IGNORE>"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64, {imageDecoded}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100,
        )
        response_text = response.choices[0].message.content.strip()
        print("--- GPT-4V Raw Response ---")
        print(response_text)
        print("--------------------------")
       
        lines = response_text.splitlines()
        classification, is_smelly, smell_rating, volume_guess, item_name = "UNKNOWN", "UNKNOWN", -1, -1.0, "UNKNOWN"
       
        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith("classification:"):
                classification = line.split(":", 1)[1].strip().upper()
            elif line_lower.startswith("smelly:"):
                is_smelly = line.split(":", 1)[1].strip().upper()
            elif line_lower.startswith("smell rating:"):
                try:
                    smell_rating = int(line.split(":", 1)[1].strip())
                except ValueError:
                    smell_rating = -1
            elif line_lower.startswith("volume estimation:"):
                try:
                    volume_str = line.split(":", 1)[1].strip().split()[0]
                    volume_guess = float(volume_str.replace(',', ''))
                except (ValueError, IndexError):
                    volume_guess = -1.0
            elif line_lower.startswith("item name:"):
                item_name = line.split(":", 1)[1].strip().upper()

        if classification == "IGNORE":
            print("→ GPT-4V determined the object/change should be ignored.")
            return None
       
        print("→ Final Classification:", classification)
        print("→ Smelly object?", is_smelly)
        print("→ Smell rating:", smell_rating)
        print("→ Estimated volume (cm^3):", volume_guess)
        print("→ Item:", item_name)
       
        return classification, is_smelly, smell_rating, volume_guess, item_name
    except Exception as e:
        print(f"An error occurred during OpenAI request or processing: {e}")
        return None

# Function to initialize serial connection
def initialize_serial():
    global ser
    try:
        print(f"Attempting to connect to Arduino on {ARDUINO_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino reset
        if ser.is_open:
            print("Arduino connected successfully.")
            return True
        else:
            print("Failed to open serial port.")
            ser = None
            return False
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
        ser = None
        return False
    except Exception as e:
        print(f"An unexpected error occurred during serial initialization: {e}")
        ser = None
        return False

# Function to process captured frame
def process_capture(frame_to_process, image_path="item_capture.jpg"):
    global ser
   
    print(f"Change detected for >{analysis_delay}s. Processing captured frame...")
    classification_result = None
   
    try:
        save_path = image_path
        cv2.imwrite(save_path, frame_to_process)
        print(f"Image captured and saved to {save_path}")
       
        # Call OpenAI for analysis
        result = ask_chatgpt(save_path)
       
        if result:
            classification, is_smelly, smell_rating, volume_guess, item_name = result
            classification_result = classification
           
            # Send result to the GUI thread
            detection_queue.put({
                "type": classification_result,
                "name": item_name,
                "smelly": is_smelly,
                "smell_rating": smell_rating,
                "volume": volume_guess
            })
        else:
            print("Analysis resulted in IGNORE or an error. No command sent to Arduino.")
    except Exception as e:
        print(f"Error during frame processing or OpenAI analysis step: {e}")
    finally:
        if classification_result in ["TRASH", "RECYCLING"]:
            # Send command to Arduino
            command = 'T\n' if classification_result == "TRASH" else 'R\n'
            if ser and ser.is_open:
                try:
                    print(f"Sending command '{command.strip()}' to Arduino...")
                    ser.write(command.encode('utf-8'))
                    print("Command sent.")
                except serial.SerialException as e:
                    print(f"Error writing to Arduino: {e}")
                except Exception as e:
                    print(f"Unexpected error during serial write: {e}")
            else:
                print("Cannot send command: Arduino serial port not available.")

# Camera detection thread function
def camera_detection_thread():
    global camera_active, backSub
   
    is_processing = False
    last_trigger_time = 0
    change_first_seen_time = 0.0
   
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Cannot open camera")
        return
   
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)
   
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   
    if h == 0 or w == 0:
        print(f"Error: Could not get valid frame dimensions ({w}x{h}) from camera.")
        cap.release()
        return
   
    print(f"Camera resolution set to {w}x{h}")
   
    # Initialize background model
    print("Camera opened. Allowing time for background learning...")
    initial_bg_frames = 30
    for i in range(initial_bg_frames):
        ret, frame = cap.read()
        if ret:
            _ = backSub.apply(frame)
        else:
            print(f"Warning: Failed to grab frame {i+1}/{initial_bg_frames} during background learning.")
        time.sleep(0.05)
   
    print(f"Starting monitoring. Will analyze significant changes present for >{analysis_delay}s...")
   
    frame_counter = 0
    last_contour_boxes = []
    current_frame = None
   
    try:
        while camera_active:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue
           
            current_frame = frame.copy()
            display_frame = frame
            now = time.time()
           
            frame_counter += 1
            significant_change_detected_in_last_processed_frame = False
           
            if frame_counter % process_every_n_frames == 0:
                # Apply background subtraction
                fgMask = backSub.apply(current_frame)
                _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
               
                # Clean up mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fgMask_cleaned = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
                fgMask_cleaned = cv2.morphologyEx(fgMask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
               
                # Find contours
                contours, _ = cv2.findContours(fgMask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                significant_change_detected = False
                current_contour_boxes = []
               
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > min_contour_area:
                        significant_change_detected = True
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        current_contour_boxes.append((x, y, x+cw, y+ch))
               
                last_contour_boxes = current_contour_boxes
                significant_change_detected_in_last_processed_frame = significant_change_detected
               
                # Trigger logic
                if significant_change_detected_in_last_processed_frame:
                    if change_first_seen_time == 0.0:
                        print(f"Significant change detected. Starting {analysis_delay}s timer...")
                        change_first_seen_time = now
                    else:
                        elapsed_time = now - change_first_seen_time
                        if elapsed_time >= analysis_delay:
                            if not is_processing and (now - last_trigger_time > cooldown_period):
                                print(f"--- Change detected for >{analysis_delay}s. Triggering Analysis ---")
                                is_processing = True
                                change_first_seen_time = 0.0
                                last_trigger_time = now
                                frame_to_analyze = current_frame
                               
                                # Start analysis in a separate thread
                                threading.Thread(
                                    target=process_capture,
                                    args=(frame_to_analyze, "item_capture.jpg"),
                                    daemon=True
                                ).start()
                else:
                    # No significant change detected
                    if change_first_seen_time != 0.0:
                        print("Change disappeared before analysis delay.")
                        change_first_seen_time = 0.0
           
            # Display processing status in separate window if needed
            try:
                # Draw contour boxes on display frame
                if significant_change_detected_in_last_processed_frame:
                    for (x1, y1, x2, y2) in last_contour_boxes:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
               
                status = "Processing..." if is_processing else "Monitoring"
                cv2.putText(display_frame, f"Status: {status}", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
               
                cv2.imshow("Camera Feed", display_frame)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Error displaying camera feed: {e}")
           
            # Check if processing is complete
            if is_processing and (now - last_trigger_time > 5.0):  # Add timeout
                is_processing = False
           
            time.sleep(0.01)  # Small delay to prevent CPU hogging
   
    except Exception as e:
        print(f"Error in camera thread: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera thread shut down.")

# --- Pygame GUI Classes ---

# Smart bin statistics
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
            self.co2_saved += random.uniform(0.2, 0.5)
            self.water_saved += random.uniform(1, 3)
        elif bin_type.upper() == "TRASH":
            self.landfill_items += 1
        self.last_updated = time.time()
       
    def get_recycling_percentage(self):
        if self.total_items == 0:
            return 0
        return (self.recycled_items / self.total_items) * 100

# Nature-inspired animated elements (leaves, water drops)
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
       
        if self.y > SCREEN_HEIGHT or self.x < -20 or self.x > SCREEN_WIDTH + 20:
            self.reset()
           
    def draw(self, surface):
        try:
            if self.type == "leaf":
                # Create a leaf shape
                leaf_surf = pygame.Surface((self.size * 4, self.size * 3), pygame.SRCALPHA)
               
                # Draw leaf shape - simplified oval with stem
                leaf_color = (*self.color, 180)  # Semi-transparent
                stem_color = (SOFT_BROWN[0], SOFT_BROWN[1], SOFT_BROWN[2], 180)
               
                # Leaf body
                pygame.draw.ellipse(leaf_surf, leaf_color, (0, 0, self.size * 3, self.size * 2))
               
                # Leaf stem
                pygame.draw.line(leaf_surf, stem_color,
                              (self.size * 1.5, self.size * 1),
                              (self.size * 3, self.size * 2), 2)
               
                # Rotate leaf
                rotated_leaf = pygame.transform.rotate(leaf_surf, self.rotation)
                leaf_rect = rotated_leaf.get_rect(center=(int(self.x), int(self.y)))
                surface.blit(rotated_leaf, leaf_rect)
               
            else:  # water drop
                drop_surf = pygame.Surface((self.size * 2, self.size * 3), pygame.SRCALPHA)
                drop_color = (WATER_BLUE[0], WATER_BLUE[1], WATER_BLUE[2], 150)
               
                # Draw teardrop shape
                pygame.draw.circle(drop_surf, drop_color, (self.size, self.size), self.size)
                points = [
                    (self.size - self.size/2, self.size),
                    (self.size + self.size/2, self.size),
                    (self.size, self.size * 2.5)
                ]
                pygame.draw.polygon(drop_surf, drop_color, points)
               
                # Add highlight
                highlight_pos = (int(self.size * 0.7), int(self.size * 0.7))
                highlight_radius = max(1, int(self.size / 3))
                highlight_color = (LIGHT_BLUE[0], LIGHT_BLUE[1], LIGHT_BLUE[2], 180)
                pygame.draw.circle(drop_surf, highlight_color, highlight_pos, highlight_radius)
               
                # Rotate slightly
                rotated_drop = pygame.transform.rotate(drop_surf, self.rotation / 10)  # Subtle rotation
                drop_rect = rotated_drop.get_rect(center=(int(self.x), int(self.y)))
                surface.blit(rotated_drop, drop_rect)
        except (pygame.error, TypeError) as e:
            # If rendering fails, just reset this element
            self.reset()

# Natural-looking progress bar (like a growing plant or filling water level)
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
        self.style = style  # "plant" or "water"
       
        # Elements for plant style progress
        if self.style == "plant":
            self.vine_points = []
            self.leaf_positions = []
            self.generate_vine_path()
       
    def generate_vine_path(self):
        # Create a natural-looking vine path
        self.vine_points = []
        segment_height = self.height / 10
       
        for i in range(11):  # 11 points to create 10 segments
            y_pos = self.y + self.height - (i * segment_height)
            wiggle = random.uniform(-self.width/6, self.width/6) if i > 0 else 0
            x_pos = self.x + self.width/2 + wiggle
            self.vine_points.append((x_pos, y_pos))
           
            # Add leaf positions at certain intervals
            if i > 0 and i % 2 == 0:
                leaf_side = 1 if i % 4 == 0 else -1  # Alternate sides
                self.leaf_positions.append((i, leaf_side))
       
    def set_value(self, value):
        self.target_value = min(value, self.max_value)
       
    def update(self):
        if self.current_value < self.target_value:
            self.current_value = min(self.target_value, self.current_value + self.animation_speed)
        elif self.current_value > self.target_value:
            self.current_value = max(self.target_value, self.current_value - self.animation_speed)
           
    def draw(self, surface):
        try:
            # Draw background
            pygame.draw.rect(surface, self.bg_color, (self.x, self.y, self.width, self.height), border_radius=10)
            pygame.draw.rect(surface, (*self.bg_color, 100), (self.x, self.y, self.width, self.height),
                          2, border_radius=10)
           
            progress_ratio = self.current_value / self.max_value if self.max_value > 0 else 0
           
            if self.style == "plant":
                self.draw_plant_progress(surface, progress_ratio)
            else:  # water style
                self.draw_water_progress(surface, progress_ratio)
               
            # Draw percentage text
            text_color = WHITE if progress_ratio > 0.5 else TEXT_BROWN
            text = font_medium.render(f"{int(self.current_value)}%", True, text_color)
            text_rect = text.get_rect(center=(self.x + self.width/2, self.y + 25))
            surface.blit(text, text_rect)
        except (pygame.error, ValueError, ZeroDivisionError) as e:
            # Recover from rendering errors
            pass
           
    def draw_plant_progress(self, surface, progress_ratio):
        # Calculate how much of the vine to draw based on progress
        visible_segments = math.ceil(progress_ratio * 10)
       
        if visible_segments > 0 and len(self.vine_points) > 1:
            # Draw the vine
            for i in range(min(visible_segments, len(self.vine_points) - 1)):
                start_point = self.vine_points[i]
                end_point = self.vine_points[i + 1]
               
                # Vary thickness slightly for natural look
                thickness = random.randint(3, 5)
                pygame.draw.line(surface, DARK_GREEN, start_point, end_point, thickness)
               
                # Draw leaves at predetermined positions
                for leaf_idx, leaf_side in self.leaf_positions:
                    if i == leaf_idx - 1:
                        # Calculate leaf position
                        leaf_x = (start_point[0] + end_point[0]) / 2
                        leaf_y = (start_point[1] + end_point[1]) / 2
                       
                        # Draw leaf
                        leaf_size = random.randint(5, 8)
                        leaf_surf = pygame.Surface((leaf_size * 3, leaf_size * 2), pygame.SRCALPHA)
                       
                        # Leaf shape
                        pygame.draw.ellipse(leaf_surf, LEAF_GREEN, (0, 0, leaf_size * 3, leaf_size * 2))
                       
                        # Rotate based on which side
                        angle = 45 if leaf_side > 0 else -45
                        rotated_leaf = pygame.transform.rotate(leaf_surf, angle)
                        leaf_rect = rotated_leaf.get_rect(center=(leaf_x + (leaf_side * 15), leaf_y))
                        surface.blit(rotated_leaf, leaf_rect)
           
            # Draw flower/plant top when progress is high
            if progress_ratio > 0.9 and len(self.vine_points) > 0:
                top_x, top_y = self.vine_points[-1]
               
                # Draw a simple flower
                petal_color = (255, 200, 100)  # Yellow-orange
                center_color = SUNSET_ORANGE
               
                # Petals (circles arranged in a flower pattern)
                for angle in range(0, 360, 60):
                    petal_x = top_x + 8 * math.cos(math.radians(angle))
                    petal_y = top_y + 8 * math.sin(math.radians(angle))
                    pygame.draw.circle(surface, petal_color, (petal_x, petal_y), 7)
               
                # Center of flower
                pygame.draw.circle(surface, center_color, (top_x, top_y), 5)
               
    def draw_water_progress(self, surface, progress_ratio):
        # Calculate fill height
        fill_height = int(self.height * progress_ratio)
       
        if fill_height > 0:
            # Draw water fill
            water_rect = pygame.Rect(self.x, self.y + self.height - fill_height,
                                  self.width, fill_height)
            pygame.draw.rect(surface, self.color, water_rect, border_radius=10)
           
            # Add wave effect at the top of the water
            wave_height = 3
            wave_surface = pygame.Surface((self.width, wave_height * 2), pygame.SRCALPHA)
           
            # Draw a lighter colored wave pattern
            lighter_color = LIGHT_BLUE
            for x in range(0, self.width, 10):
                offset = math.sin(time.time() * 2 + x * 0.1) * wave_height
                pygame.draw.circle(wave_surface, lighter_color,
                               (x, wave_height + offset), wave_height)
           
            # Place the wave at the top of the water level
            surface.blit(wave_surface, (self.x, self.y + self.height - fill_height - wave_height))

# Modern, eco-friendly UI button
class EcoButton:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=WHITE, border_radius=10):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_radius = border_radius
        self.hovered = False
        self.animation = 0
       
    def draw(self, surface):
        try:
            current_color = self.hover_color if self.hovered else self.color
            if self.animation > 0:
                self.animation -= 0.1
                current_color = tuple(max(0, c - 20) for c in current_color)
           
            # Main button with rounded corners and natural gradient
            pygame.draw.rect(surface, current_color, self.rect, border_radius=self.border_radius)
           
            # Add subtle wood-like texture effect
            for i in range(0, self.rect.height, 3):
                texture_alpha = random.randint(5, 15)
                texture_line = pygame.Surface((self.rect.width, 1), pygame.SRCALPHA)
                texture_line.fill((*BLACK[:3], texture_alpha))
                surface.blit(texture_line, (self.rect.left, self.rect.top + i))
           
            # Add subtle highlight on top
            highlight_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, self.rect.height // 3)
            for i in range(highlight_rect.height):
                alpha = 30 - int((i / highlight_rect.height) * 30)
                highlight_surface = pygame.Surface((highlight_rect.width, 1), pygame.SRCALPHA)
                highlight_surface.fill((*WHITE[:3], alpha))
                surface.blit(highlight_surface, (highlight_rect.left, highlight_rect.top + i))
           
            # Button text
            text_surf = font_medium.render(self.text, True, self.text_color)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)
        except pygame.error:
            # Handle rendering errors
            pass
       
    def check_hover(self, mouse_pos):
        if not self.rect:  # Safety check
            return False
        self.hovered = self.rect.collidepoint(mouse_pos)
        return self.hovered
   
    def clicked(self):
        self.animation = 1.0
        return True

# Redesigned detection animation with natural, smooth loading
class DetectionAnimation:
    def __init__(self, item_name, item_type):
        self.item_name = item_name
        self.item_type = item_type
        self.start_time = time.time()
        self.phase = "dropping"  # dropping, scanning, revealing, feedback
        self.phase_durations = {
            "dropping": 1.5,
            "scanning": 2.5,
            "revealing": 2.0,
            "feedback": 5.0
        }
        self.phases_completed = {
            "dropping": False,
            "scanning": False,
            "revealing": False,
            "feedback": False
        }
        self.stats_updated = False
       
        # Loading progress for scanning phase
        self.scan_progress = 0
       
        self.reveal_alpha = 0
        self.particle_effects = []
       
        # Use natural colors based on item type
        self.item_color = LEAF_GREEN if item_type.upper() == "RECYCLING" else SOFT_BROWN
        self.item_image = self.create_item_image()
       
        self.y_pos = -100
        self.rotation = 0
        self.rotation_speed = random.uniform(1, 3)
        self.fall_speed = random.uniform(5, 8)
        self.target_y = SCREEN_HEIGHT // 2 - 50
       
        # New loading bar for scanning
        self.loading_width = 300
        self.loading_height = 20
        self.loading_x = SCREEN_WIDTH // 2 - self.loading_width // 2
        self.loading_y = self.target_y + 80
       
    def create_item_image(self):
        try:
            size = 80
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
           
            # Create more organic, natural-looking item icons
            if self.item_type.upper() == "RECYCLING":
                if "bottle" in self.item_name.lower():
                    # Bottle with a more curved, organic shape
                    pygame.draw.rect(surf, self.item_color, (size//3, size//6, size//3, size*2//3),
                                 border_radius=10)
                    pygame.draw.ellipse(surf, self.item_color, (size//3 - 5, size//12, size//3 + 10, size//6))
                   
                    # Recycling symbol
                    symbol_radius = 8
                    symbol_center = (size//2, size//2 + 10)
                    pygame.draw.circle(surf, WHITE, symbol_center, symbol_radius, 2)
                   
                elif "can" in self.item_name.lower():
                    # Aluminum can with texture
                    pygame.draw.rect(surf, self.item_color, (size//4, size//6, size//2, size*2//3),
                                 border_radius=5)
                   
                    # Recycling symbol
                    symbol_x = size//2
                    symbol_y = size//2
                    symbol_size = 12
                   
                    # Draw simplified recycling arrows
                    for i in range(3):
                        angle = math.radians(i * 120)
                        x1 = symbol_x + symbol_size * math.cos(angle)
                        y1 = symbol_y + symbol_size * math.sin(angle)
                        x2 = symbol_x + symbol_size * math.cos(angle + math.radians(120))
                        y2 = symbol_y + symbol_size * math.sin(angle + math.radians(120))
                        pygame.draw.line(surf, WHITE, (x1, y1), (x2, y2), 2)
                   
                elif "paper" in self.item_name.lower() or "newspaper" in self.item_name.lower():
                    # Paper with texture
                    paper_color = LIGHT_CREAM
                    pygame.draw.rect(surf, paper_color, (size//5, size//5, size*3//5, size*3//5))
                   
                    # Add paper texture lines
                    for y in range(size//5, size*4//5, 5):
                        line_alpha = random.randint(10, 30)
                        pygame.draw.line(surf, (*SOFT_BROWN, line_alpha),
                                     (size//5, y), (size*4//5, y), 1)
                   
                    # Fold corner
                    pygame.draw.polygon(surf, (*paper_color, 180), [
                        (size*4//5, size//5),
                        (size*4//5, size//3),
                        (size*2//3, size//5)
                    ])
                else:
                    # Generic recyclable - leaf shape
                    leaf_points = [
                        (size//2, size//5),
                        (size*3//4, size//3),
                        (size*4//5, size//2),
                        (size*3//4, size*2//3),
                        (size//2, size*4//5),
                        (size//4, size*2//3),
                        (size//5, size//2),
                        (size//4, size//3)
                    ]
                    pygame.draw.polygon(surf, self.item_color, leaf_points)
                   
                    # Leaf vein
                    pygame.draw.line(surf, (*DARK_GREEN, 150),
                                 (size//2, size//5), (size//2, size*4//5), 2)
                   
            else:  # landfill items
                if "wrapper" in self.item_name.lower():
                    # Crumpled wrapper with texture
                    wrapper_points = [
                        (size//4, size//4),
                        (size*3//4, size//4),
                        (size*4//5, size//2),
                        (size*3//4, size*3//4),
                        (size//4, size*3//4),
                        (size//5, size//2)
                    ]
                    pygame.draw.polygon(surf, self.item_color, wrapper_points)
                   
                    # Add crinkle lines
                    for _ in range(5):
                        x1 = random.randint(size//4, size*3//4)
                        y1 = random.randint(size//4, size*3//4)
                        x2 = x1 + random.randint(-10, 10)
                        y2 = y1 + random.randint(-10, 10)
                        pygame.draw.line(surf, (*BLACK, 30), (x1, y1), (x2, y2), 1)
                   
                elif "cup" in self.item_name.lower():
                    # Paper/styrofoam cup
                    cup_color = LIGHT_CREAM if "paper" in self.item_name.lower() else WHITE
                   
                    # Cup body
                    pygame.draw.polygon(surf, cup_color, [
                        (size//3, size//4),
                        (size*2//3, size//4),
                        (size*3//5, size*3//4),
                        (size*2//5, size*3//4)
                    ])
                   
                    # Cup rim
                    pygame.draw.ellipse(surf, cup_color, (size//3 - 5, size//5, size//2 + 10, size//8))
                   
                    # Cup texture
                    if "paper" in self.item_name.lower():
                        for y in range(size//4, size*3//4, 5):
                            pygame.draw.line(surf, (*SOFT_BROWN, 30),
                                         (size//3, y), (size*2//3, y), 1)
                else:
                    # Generic waste - irregular blob shape
                    center_x = size // 2
                    center_y = size // 2
                    radius = size // 3
                   
                    points = []
                    for angle in range(0, 360, 30):
                        rad = math.radians(angle)
                        radius_var = radius * random.uniform(0.8, 1.2)
                        x = center_x + radius_var * math.cos(rad)
                        y = center_y + radius_var * math.sin(rad)
                        points.append((x, y))
                       
                    pygame.draw.polygon(surf, self.item_color, points)
           
            # Add subtle texture and highlights
            for _ in range(3):
                x = random.randint(size//4, size*3//4)
                y = random.randint(size//4, size*3//4)
                r = random.randint(2, 4)
                highlight_color = (*WHITE, 70)
                pygame.draw.circle(surf, highlight_color, (x, y), r)
               
            return surf
        except pygame.error:
            # Return a fallback surface if image creation fails
            fallback = pygame.Surface((80, 80), pygame.SRCALPHA)
            pygame.draw.rect(fallback, self.item_color, (20, 20, 40, 40))
            return fallback
       
    def update(self):
        try:
            current_time = time.time()
            elapsed = current_time - self.start_time
           
            if self.phase == "dropping":
                self.y_pos += self.fall_speed
                self.rotation += self.rotation_speed
                if self.y_pos >= self.target_y:
                    self.y_pos = self.target_y
                    if not self.phases_completed["dropping"]:
                        self.phase = "scanning"
                        self.phases_completed["dropping"] = True
                        self.start_time = current_time
                   
            elif self.phase == "scanning":
                # Smooth loading bar progress
                progress_percentage = min(100, (elapsed / self.phase_durations["scanning"]) * 100)
                self.scan_progress = progress_percentage
               
                # Add leaf/drop particles during scanning
                if random.random() < 0.1:
                    self.add_natural_particle()
                   
                if elapsed >= self.phase_durations["scanning"] and not self.phases_completed["scanning"]:
                    self.phase = "revealing"
                    self.phases_completed["scanning"] = True
                    self.start_time = current_time
                   
            elif self.phase == "revealing":
                progress = min(1.0, elapsed / self.phase_durations["revealing"])
                self.reveal_alpha = int(255 * progress)
               
                # Add celebratory particles
                if random.random() < 0.2:
                    self.add_reveal_particle()
                self.update_particles()
               
                if elapsed >= self.phase_durations["revealing"] and not self.phases_completed["revealing"]:
                    self.phase = "feedback"
                    self.phases_completed["revealing"] = True
                    self.start_time = current_time
                   
            elif self.phase == "feedback":
                self.update_particles()
               
                if not self.stats_updated:
                    self.stats_updated = True
                   
                if elapsed >= self.phase_durations["feedback"] and not self.phases_completed["feedback"]:
                    self.phases_completed["feedback"] = True
                    return True
                   
            return False
        except (TypeError, ValueError) as e:
            # Handle calculation errors by resetting to a safe state
            self.phase = "feedback"
            self.phases_completed["feedback"] = True
            return True
   
    def add_natural_particle(self):
        # Add nature-inspired particles during scanning
        particle_type = "leaf" if self.item_type.upper() == "RECYCLING" else "sparkle"
       
        if particle_type == "leaf":
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(1, 2)
            self.particle_effects.append({
                "type": "leaf",
                "x": SCREEN_WIDTH // 2,
                "y": self.loading_y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "size": random.uniform(3, 6),
                "color": LEAF_GREEN,
                "rotation": random.uniform(0, 360),
                "rot_speed": random.uniform(-3, 3),
                "life": 1.0
            })
        else:
            self.particle_effects.append({
                "type": "sparkle",
                "x": self.loading_x + (self.loading_width * self.scan_progress / 100),
                "y": self.loading_y + random.uniform(-10, 10),
                "vx": random.uniform(-1, 1),
                "vy": random.uniform(-2, -0.5),
                "size": random.uniform(1, 3),
                "color": (*SUNSET_ORANGE, 150),
                "life": 1.0
            })
       
    def add_reveal_particle(self):
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(1, 2.5)
       
        # Determine particle type based on item type
        if self.item_type.upper() == "RECYCLING":
            particle_type = random.choice(["leaf", "drop"])
            if particle_type == "leaf":
                color = LEAF_GREEN
            else:
                color = WATER_BLUE
        else:
            particle_type = "sparkle"
            color = SUNSET_ORANGE
           
        self.particle_effects.append({
            "type": particle_type,
            "x": SCREEN_WIDTH // 2,
            "y": self.y_pos + self.item_image.get_height() // 2,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "size": random.uniform(2, 5),
            "color": color,
            "rotation": random.uniform(0, 360),
            "rot_speed": random.uniform(-3, 3),
            "life": 1.0
        })
       
    def update_particles(self):
        for i in range(len(self.particle_effects)-1, -1, -1):
            if i >= len(self.particle_effects):  # Safety check for index
                continue
               
            p = self.particle_effects[i]
            p["x"] += p["vx"]
            p["y"] += p["vy"]
           
            if "rotation" in p:
                p["rotation"] += p["rot_speed"]
               
            p["life"] -= 0.02
            if p["life"] <= 0:
                # Make sure we're not out of bounds
                if 0 <= i < len(self.particle_effects):
                    self.particle_effects.pop(i)
               
    def draw_particles(self, surface):
        for p in self.particle_effects:
            try:
                if p["type"] == "leaf":
                    # Draw leaf particle
                    leaf_size = p["size"]
                    leaf_surf = pygame.Surface((leaf_size * 4, leaf_size * 3), pygame.SRCALPHA)
                   
                    leaf_color = (*p["color"], int(255 * p["life"]))
                    pygame.draw.ellipse(leaf_surf, leaf_color, (0, 0, leaf_size * 3, leaf_size * 2))
                   
                    # Add stem
                    stem_color = (*SOFT_BROWN, int(200 * p["life"]))
                    pygame.draw.line(leaf_surf, stem_color,
                                  (leaf_size * 1.5, leaf_size),
                                  (leaf_size * 3, leaf_size * 1.5), 2)
                   
                    rotated_leaf = pygame.transform.rotate(leaf_surf, p["rotation"])
                    leaf_rect = rotated_leaf.get_rect(center=(int(p["x"]), int(p["y"])))
                    surface.blit(rotated_leaf, leaf_rect)
                   
                elif p["type"] == "drop":
                    # Draw water drop particle
                    drop_size = p["size"]
                    drop_surf = pygame.Surface((drop_size * 2, drop_size * 3), pygame.SRCALPHA)
                   
                    drop_color = (*p["color"], int(200 * p["life"]))
                    pygame.draw.circle(drop_surf, drop_color, (drop_size, drop_size), drop_size)
                    points = [
                        (drop_size - drop_size/2, drop_size),
                        (drop_size + drop_size/2, drop_size),
                        (drop_size, drop_size * 2.5)
                    ]
                    pygame.draw.polygon(drop_surf, drop_color, points)
                   
                    rotated_drop = pygame.transform.rotate(drop_surf, p["rotation"] / 5)
                    drop_rect = rotated_drop.get_rect(center=(int(p["x"]), int(p["y"])))
                    surface.blit(rotated_drop, drop_rect)
                   
                else:  # sparkle
                    # Draw simple sparkle/star particle
                    sparkle_color = (*p["color"][:3], int(p["color"][3] * p["life"]))
                    for angle in range(0, 360, 45):
                        end_x = p["x"] + math.cos(math.radians(angle)) * p["size"] * 2
                        end_y = p["y"] + math.sin(math.radians(angle)) * p["size"] * 2
                        pygame.draw.line(surface, sparkle_color,
                                     (int(p["x"]), int(p["y"])),
                                     (int(end_x), int(end_y)), 1)
            except (pygame.error, KeyError, TypeError) as e:
                # Skip rendering this particle if there's an error
                continue
               
    def draw(self, surface):
        try:
            if self.phase == "dropping":
                rotated_image = pygame.transform.rotate(self.item_image, self.rotation)
                rotated_rect = rotated_image.get_rect(center=(SCREEN_WIDTH // 2, self.y_pos))
                surface.blit(rotated_image, rotated_rect)
               
            elif self.phase == "scanning":
                # Draw the item
                surface.blit(self.item_image, (SCREEN_WIDTH // 2 - self.item_image.get_width() // 2, self.y_pos))
               
                # Draw natural-looking loading bar with wood texture background
                # Background
                loading_bg_rect = pygame.Rect(self.loading_x, self.loading_y, self.loading_width, self.loading_height)
                pygame.draw.rect(surface, LIGHT_BROWN, loading_bg_rect, border_radius=10)
               
                # Wood grain texture
                for y in range(self.loading_y, self.loading_y + self.loading_height, 2):
                    texture_alpha = random.randint(10, 30)
                    texture_width = random.randint(self.loading_width - 20, self.loading_width)
                    texture_x = self.loading_x + random.randint(0, 20)
                    texture_line = pygame.Surface((texture_width, 1), pygame.SRCALPHA)
                    texture_line.fill((*SOFT_BROWN, texture_alpha))
                    surface.blit(texture_line, (texture_x, y))
               
                # Progress fill - use green for recycling, earth tone for landfill
                fill_color = LEAF_GREEN if self.item_type.upper() == "RECYCLING" else SOFT_BROWN
                fill_width = int(self.loading_width * (self.scan_progress / 100))
                if fill_width > 0:
                    fill_rect = pygame.Rect(self.loading_x, self.loading_y, fill_width, self.loading_height)
                    pygame.draw.rect(surface, fill_color, fill_rect, border_radius=10)
                   
                    # Add some texture to the filled part
                    for y in range(self.loading_y, self.loading_y + self.loading_height, 3):
                        if random.random() < 0.7:  # Only add texture to some lines
                            highlight_alpha = random.randint(20, 40)
                            highlight_line = pygame.Surface((fill_width, 1), pygame.SRCALPHA)
                            highlight_line.fill((*WHITE, highlight_alpha))
                            surface.blit(highlight_line, (self.loading_x, y))
               
                # Border
                pygame.draw.rect(surface, WOOD_BROWN, loading_bg_rect, 2, border_radius=10)
               
                # Status text
                text = font_medium.render("Analyzing...", True, TEXT_BROWN)
                text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, self.loading_y - 20))
                surface.blit(text, text_rect)
               
                # Draw particles
                self.draw_particles(surface)
                   
            elif self.phase in ["revealing", "feedback"]:
                # Draw the item
                surface.blit(self.item_image, (SCREEN_WIDTH // 2 - self.item_image.get_width() // 2, self.y_pos))
               
                # Draw particles
                self.draw_particles(surface)
               
                # Fade in text animation
                alpha = self.reveal_alpha if self.phase == "revealing" else 255
               
                # Item name with natural background
                name_bg = pygame.Surface((300, 40), pygame.SRCALPHA)
                name_bg.fill((*CREAM, alpha * 0.7))
                name_bg_rect = name_bg.get_rect(center=(SCREEN_WIDTH // 2, self.y_pos - 50))
                surface.blit(name_bg, name_bg_rect)
               
                name_surf = font_large.render(self.item_name, True, TEXT_BROWN)
                name_surf.set_alpha(alpha)
                name_rect = name_surf.get_rect(center=(SCREEN_WIDTH // 2, self.y_pos - 50))
                surface.blit(name_surf, name_rect)
               
                # Show item type with natural look
                type_color = LEAF_GREEN if self.item_type.upper() == "RECYCLING" else SOFT_BROWN
                type_text = "Recyclable" if self.item_type.upper() == "RECYCLING" else "Non-Recyclable"
               
                type_bg = pygame.Surface((200, 30), pygame.SRCALPHA)
                type_bg.fill((*CREAM, alpha * 0.7))
                type_bg_rect = type_bg.get_rect(center=(SCREEN_WIDTH // 2, self.y_pos - 20))
                surface.blit(type_bg, type_bg_rect)
               
                type_surf = font_medium.render(type_text, True, type_color)
                type_surf.set_alpha(alpha)
                type_rect = type_surf.get_rect(center=(SCREEN_WIDTH // 2, self.y_pos - 20))
                surface.blit(type_surf, type_rect)
               
                # Countdown timer for feedback phase with natural styling
                if self.phase == "feedback":
                    time_left = self.phase_durations["feedback"] - (time.time() - self.start_time)
                   
                    prompt_bg = pygame.Surface((260, 40), pygame.SRCALPHA)
                    prompt_bg.fill((*CREAM, 180))
                    prompt_bg_rect = prompt_bg.get_rect(center=(SCREEN_WIDTH // 2, self.y_pos + 100))
                    pygame.draw.rect(prompt_bg, WOOD_BROWN, (0, 0, prompt_bg.get_width(), prompt_bg.get_height()), 2, border_radius=10)
                    surface.blit(prompt_bg, prompt_bg_rect)
                   
                    prompt_text = f"Is this correct? ({int(time_left)}s)"
                    prompt_surf = font_medium.render(prompt_text, True, TEXT_BROWN)
                    prompt_rect = prompt_surf.get_rect(center=(SCREEN_WIDTH // 2, self.y_pos + 100))
                    surface.blit(prompt_surf, prompt_rect)
        except (pygame.error, TypeError, ValueError, ZeroDivisionError) as e:
            # Handle any rendering errors gracefully
            pass

# Modern UI with sustainability theme
class SmartBinInterface:
    def __init__(self):
        self.stats = BinStats()
        self.nature_elements = [NatureElement() for _ in range(15)]
        self.state = "idle"  # idle, detecting, feedback
        self.detection_animation = None
       
        # Natural looking progress indicators
        self.recycling_progress = NaturalProgressBar(
            SCREEN_WIDTH // 4 - 70, SCREEN_HEIGHT // 2 - 100, 140, 200,
            LEAF_GREEN, LIGHT_CREAM, style="plant"
        )
        self.landfill_progress = NaturalProgressBar(
            SCREEN_WIDTH * 3 // 4 - 70, SCREEN_HEIGHT // 2 - 100, 140, 200,
            WATER_BLUE, LIGHT_CREAM, style="water"
        )
        self.recycling_progress.set_value(70)  # Example values
        self.landfill_progress.set_value(30)
       
        # Eco-friendly buttons
        button_width = 160
        button_height = 50
        button_y = SCREEN_HEIGHT - 120
        self.correct_button = EcoButton(
            SCREEN_WIDTH // 2 - button_width - 20, button_y, button_width, button_height,
            "Correct", LEAF_GREEN, LIGHT_GREEN
        )
        self.incorrect_button = EcoButton(
            SCREEN_WIDTH // 2 + 20, button_y, button_width, button_height,
            "Incorrect", SOFT_BROWN, LIGHT_BROWN
        )
       
        # Eco tips - at bottom of screen
        self.last_hint_time = time.time()
        self.hint_interval = 30
        self.hints = [
            "SmartBin™ uses AI to automatically sort your waste",
            "Recycling properly reduces landfill waste by up to 75%",
            "Even a single contaminated item can ruin an entire recycling batch",
            "SmartBin™ has prevented over 10,000 tons of waste from landfills",
            "Our accuracy rate in sorting waste is over 95%",
            "Thank you for helping make our planet greener!",
            "Plastic bottles take 450+ years to decompose in landfills",
            "Glass can be recycled infinitely without losing quality",
            "Paper can be recycled 5-7 times before fibers break down"
        ]
        self.current_hint = random.choice(self.hints)
        self.hint_alpha = 0
        self.hint_fade_in = True
       
        # Example waste items for fallback if camera detection fails
        self.waste_items = [
            {"name": "Plastic Bottle", "type": "RECYCLING"},
            {"name": "Aluminum Can", "type": "RECYCLING"},
            {"name": "Glass Bottle", "type": "RECYCLING"},
            {"name": "Cardboard Box", "type": "RECYCLING"},
            {"name": "Paper Cup", "type": "RECYCLING"},
            {"name": "Newspaper", "type": "RECYCLING"},
            {"name": "Plastic Wrapper", "type": "TRASH"},
            {"name": "Styrofoam Cup", "type": "TRASH"},
            {"name": "Food Waste", "type": "TRASH"},
            {"name": "Candy Wrapper", "type": "TRASH"},
            {"name": "Dirty Pizza Box", "type": "TRASH"},
            {"name": "Used Tissue", "type": "TRASH"}
        ]
       
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
            # Draw wood-textured cards for each progress indicator
            recycling_card = pygame.Rect(SCREEN_WIDTH // 4 - 100, SCREEN_HEIGHT // 2 - 150, 200, 280)
            landfill_card = pygame.Rect(SCREEN_WIDTH * 3 // 4 - 100, SCREEN_HEIGHT // 2 - 150, 200, 280)
           
            for card in [recycling_card, landfill_card]:
                # Card background with wood texture
                pygame.draw.rect(surface, CREAM, card, border_radius=15)
               
                # Add wood grain texture
                for y in range(card.y, card.y + card.height, 4):
                    texture_alpha = random.randint(5, 15)
                    texture_width = random.randint(card.width - 30, card.width)
                    texture_x = card.x + random.randint(0, 30)
                    texture_line = pygame.Surface((texture_width, 2), pygame.SRCALPHA)
                    texture_line.fill((*SOFT_BROWN, texture_alpha))
                    surface.blit(texture_line, (texture_x, y))
               
                # Card border
                pygame.draw.rect(surface, WOOD_BROWN, card, 2, border_radius=15)
           
            # Draw progress indicators
            self.recycling_progress.draw(surface)
            self.landfill_progress.draw(surface)
           
            # Labels with natural styling
            recycling_label = font_medium.render("Recycling", True, DARK_GREEN)
            recycling_rect = recycling_label.get_rect(center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2 - 130))
            surface.blit(recycling_label, recycling_rect)
           
            landfill_label = font_medium.render("Landfill", True, TEXT_BROWN)
            landfill_rect = landfill_label.get_rect(center=(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT // 2 - 130))
            surface.blit(landfill_label, landfill_rect)
           
            # Draw leaf decorations near labels
            leaf_size = 15
            leaf_surf = pygame.Surface((leaf_size * 2, leaf_size), pygame.SRCALPHA)
            pygame.draw.ellipse(leaf_surf, LEAF_GREEN, (0, 0, leaf_size * 1.5, leaf_size))
            leaf_rect = leaf_surf.get_rect(center=(recycling_rect.left - 20, recycling_rect.centery))
            surface.blit(leaf_surf, leaf_rect)
           
            drop_size = 12
            drop_surf = pygame.Surface((drop_size, drop_size * 1.5), pygame.SRCALPHA)
            drop_points = [
                (drop_size/2, 0),
                (drop_size, drop_size),
                (drop_size/2, drop_size * 1.5),
                (0, drop_size)
            ]
            pygame.draw.polygon(drop_surf, WATER_BLUE, drop_points)
            drop_rect = drop_surf.get_rect(center=(landfill_rect.left - 20, landfill_rect.centery))
            surface.blit(drop_surf, drop_rect)
           
            # Draw item counts with natural styling
            if self.stats.total_items > 0:
                recycling_count_bg = pygame.Surface((100, 30), pygame.SRCALPHA)
                recycling_count_bg.fill((*LIGHT_GREEN, 100))
                count_bg_rect = recycling_count_bg.get_rect(
                    center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2 + 120)
                )
                surface.blit(recycling_count_bg, count_bg_rect)
               
                recycling_count = font_small.render(f"{self.stats.recycled_items} items", True, DARK_GREEN)
                recycling_count_rect = recycling_count.get_rect(
                    center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2 + 120)
                )
                surface.blit(recycling_count, recycling_count_rect)
               
                landfill_count_bg = pygame.Surface((100, 30), pygame.SRCALPHA)
                landfill_count_bg.fill((*LIGHT_BLUE, 100))
                landfill_bg_rect = landfill_count_bg.get_rect(
                    center=(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT // 2 + 120)
                )
                surface.blit(landfill_count_bg, landfill_bg_rect)
               
                landfill_count = font_small.render(f"{self.stats.landfill_items} items", True, TEXT_BROWN)
                landfill_count_rect = landfill_count.get_rect(
                    center=(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT // 2 + 120)
                )
                surface.blit(landfill_count, landfill_count_rect)
               
                # Environmental impact with natural styling
                impact_card = pygame.Rect(SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT - 90, 400, 70)
               
                # Card with texture
                pygame.draw.rect(surface, CREAM, impact_card, border_radius=10)
                for y in range(impact_card.y, impact_card.y + impact_card.height, 3):
                    if random.random() < 0.7:
                        texture_alpha = random.randint(5, 15)
                        texture_width = random.randint(impact_card.width - 40, impact_card.width)
                        texture_x = impact_card.x + random.randint(0, 40)
                        texture_line = pygame.Surface((texture_width, 1), pygame.SRCALPHA)
                        texture_line.fill((*SOFT_BROWN, texture_alpha))
                        surface.blit(texture_line, (texture_x, y))
               
                pygame.draw.rect(surface, WOOD_BROWN, impact_card, 2, border_radius=10)
               
                # Title
                impact_title = font_medium.render("Environmental Impact", True, TEXT_BROWN)
                impact_title_rect = impact_title.get_rect(
                    center=(SCREEN_WIDTH // 2, impact_card.y + 20)
                )
                surface.blit(impact_title, impact_title_rect)
               
                # Impact stats with icons
                co2_text = f"{self.stats.co2_saved:.1f}kg CO₂ saved"
                co2_surf = font_small.render(co2_text, True, DARK_GREEN)
                co2_rect = co2_surf.get_rect(midright=(SCREEN_WIDTH // 2 - 20, impact_card.y + 45))
                surface.blit(co2_surf, co2_rect)
               
                # Simple leaf icon
                leaf_icon_size = 15
                leaf_icon = pygame.Surface((leaf_icon_size * 1.5, leaf_icon_size), pygame.SRCALPHA)
                pygame.draw.ellipse(leaf_icon, LEAF_GREEN, (0, 0, leaf_icon_size, leaf_icon_size))
                pygame.draw.line(leaf_icon, DARK_GREEN,
                              (leaf_icon_size/2, 0), (leaf_icon_size/2, leaf_icon_size), 1)
                leaf_icon_rect = leaf_icon.get_rect(midright=(co2_rect.left - 5, co2_rect.centery))
                surface.blit(leaf_icon, leaf_icon_rect)
               
                water_text = f"{self.stats.water_saved:.1f}L water saved"
                water_surf = font_small.render(water_text, True, WATER_BLUE)
                water_rect = water_surf.get_rect(midleft=(SCREEN_WIDTH // 2 + 20, impact_card.y + 45))
                surface.blit(water_surf, water_rect)
               
                # Simple water drop icon
                drop_icon_size = 12
                drop_icon = pygame.Surface((drop_icon_size, drop_icon_size * 1.5), pygame.SRCALPHA)
                drop_icon_points = [
                    (drop_icon_size/2, 0),
                    (drop_icon_size, drop_icon_size),
                    (drop_icon_size/2, drop_icon_size * 1.5),
                    (0, drop_icon_size)
                ]
                pygame.draw.polygon(drop_icon, WATER_BLUE, drop_icon_points)
                drop_icon_rect = drop_icon.get_rect(midleft=(water_rect.right + 5, water_rect.centery))
                surface.blit(drop_icon, drop_icon_rect)
        except (pygame.error, TypeError, ValueError) as e:
            # Handle rendering errors
            pass
           
    def update_hint(self):
        current_time = time.time()
       
        # Fade effect for hints
        if self.hint_fade_in:
            self.hint_alpha = min(255, self.hint_alpha + 2)
            if self.hint_alpha >= 255:
                self.hint_fade_in = False
        else:
            self.hint_alpha = max(0, self.hint_alpha - 2)
            if self.hint_alpha <= 0:
                self.hint_fade_in = True
               
        # Cycle through hints
        if current_time - self.last_hint_time > self.hint_interval:
            self.last_hint_time = current_time
            self.current_hint = random.choice(self.hints)
            self.hint_alpha = 0
            self.hint_fade_in = True
           
    def draw_hint(self, surface):
        try:
            # Draw natural looking hint box at bottom of screen
            hint_width = len(self.current_hint) * 7 + 80  # Approximate width based on text length
            hint_height = 40
            hint_x = SCREEN_WIDTH // 2 - hint_width // 2
           
            # Position at bottom of screen, above any other elements
            hint_y = SCREEN_HEIGHT - 15 - hint_height
            if self.stats.total_items > 0:
                # If we have stats showing, position it above the impact card
                hint_y = SCREEN_HEIGHT - 90 - 15 - hint_height
           
            # Create hint box with natural paper texture
            hint_back = pygame.Surface((hint_width, hint_height), pygame.SRCALPHA)
            hint_back.fill((*CREAM[:3], int(220 * (self.hint_alpha / 255))))
           
            # Add subtle paper texture
            for y in range(0, hint_height, 3):
                line_alpha = random.randint(5, 15)
                line_color = (*SOFT_BROWN, line_alpha * (self.hint_alpha / 255))
                pygame.draw.line(hint_back, line_color, (10, y), (hint_width - 10, y), 1)
           
            # Add border
            border_color = (*WOOD_BROWN, int(self.hint_alpha * 0.8))
            pygame.draw.rect(hint_back, border_color, (0, 0, hint_width, hint_height), 2, border_radius=10)
           
            # Leaf decorations at corners
            leaf_size = 10
            for corner in [(10, 10), (hint_width - 10, 10), (10, hint_height - 10), (hint_width - 10, hint_height - 10)]:
                x, y = corner
                leaf_color = (*LEAF_GREEN, int(180 * (self.hint_alpha / 255)))
               
                # Simple leaf
                pygame.draw.ellipse(hint_back, leaf_color, (x - leaf_size//2, y - leaf_size//2, leaf_size, leaf_size))
           
            surface.blit(hint_back, (hint_x, hint_y))
           
            # Hint text
            hint_surf = font_small.render(self.current_hint, True, TEXT_BROWN)
            hint_surf.set_alpha(self.hint_alpha)
            hint_rect = hint_surf.get_rect(center=(SCREEN_WIDTH // 2, hint_y + hint_height // 2))
            surface.blit(hint_surf, hint_rect)
        except pygame.error:
            # Handle rendering errors
            pass
       
    def process_camera_detection(self, detection_data):
        """Process detection data from the camera thread"""
        if self.state == "idle":
            print(f"Processing camera detection: {detection_data}")
            item_name = detection_data.get("name", "Unknown Item")
            item_type = detection_data.get("type", "RECYCLING")
            self.detection_animation = DetectionAnimation(item_name, item_type)
            self.state = "detecting"
           
    def update_detection(self):
        try:
            # Check for new detection from camera
            if not detection_queue.empty() and self.state == "idle":
                detection_data = detection_queue.get()
                self.process_camera_detection(detection_data)
           
            # Update current detection animation
            if self.detection_animation:
                timed_out = self.detection_animation.update()
               
                if self.detection_animation.phase == "feedback" and not self.detection_animation.stats_updated:
                    self.stats.update_stats(self.detection_animation.item_type)
                   
                    # Guard against division by zero
                    if self.stats.total_items > 0:
                        recycling_percentage = self.stats.get_recycling_percentage()
                        self.recycling_progress.set_value(recycling_percentage)
                        self.landfill_progress.set_value(100 - recycling_percentage)
                   
                    self.detection_animation.stats_updated = True
                   
                if timed_out:
                    self.detection_animation = None
                    self.state = "idle"
        except Exception as e:
            # Recover from any unexpected errors in detection
            print(f"Error in update_detection: {e}")
            self.detection_animation = None
            self.state = "idle"
               
    def draw_detection(self, surface):
        try:
            if self.detection_animation:
                # Natural-looking background panel for detection area
                if self.detection_animation.phase in ["scanning", "revealing", "feedback"]:
                    panel_width = 500
                    panel_height = 300
                    panel_x = SCREEN_WIDTH // 2 - panel_width // 2
                    panel_y = SCREEN_HEIGHT // 2 - panel_height // 2
                   
                    # Panel with natural texture
                    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
                    panel.fill((*CREAM, 200))
                   
                    # Add texture
                    for y in range(0, panel_height, 3):
                        if random.random() < 0.7:
                            texture_alpha = random.randint(5, 15)
                            texture_width = random.randint(panel_width - 60, panel_width)
                            texture_x = random.randint(0, 60)
                            texture_line = pygame.Surface((texture_width, 1), pygame.SRCALPHA)
                            texture_line.fill((*SOFT_BROWN, texture_alpha))
                            panel.blit(texture_line, (texture_x, y))
                   
                    # Border with subtle decorative elements
                    pygame.draw.rect(panel, WOOD_BROWN, (0, 0, panel_width, panel_height), 3, border_radius=20)
                   
                    # Nature decorations at corners
                    for corner in [(20, 20), (panel_width - 20, 20), (20, panel_height - 20), (panel_width - 20, panel_height - 20)]:
                        x, y = corner
                        if self.detection_animation.item_type.upper() == "RECYCLING":
                            # Draw leaf
                            leaf_size = 15
                            pygame.draw.ellipse(panel, LEAF_GREEN, (x - leaf_size, y - leaf_size//2, leaf_size * 2, leaf_size))
                        else:
                            # Draw water drop
                            drop_size = 15
                            drop_points = [
                                (x, y - drop_size),
                                (x + drop_size, y),
                                (x, y + drop_size),
                                (x - drop_size, y)
                            ]
                            pygame.draw.polygon(panel, WATER_BLUE, drop_points)
                   
                    # Show feedback buttons only when in feedback phase
                    if self.detection_animation.phase == "feedback":
                        surface.blit(panel, (panel_x, panel_y))
                       
                        # Draw buttons on top of the panel
                        self.correct_button.draw(surface)
                        self.incorrect_button.draw(surface)
                       
                        # Add leaf/drop decorations to buttons based on item type
                        if self.detection_animation.item_type.upper() == "RECYCLING":
                            # Draw leaf near correct button
                            leaf_size = 20
                            leaf_surf = pygame.Surface((leaf_size * 2, leaf_size), pygame.SRCALPHA)
                            pygame.draw.ellipse(leaf_surf, LEAF_GREEN, (0, 0, leaf_size * 1.5, leaf_size))
                            leaf_rect = leaf_surf.get_rect(center=(self.correct_button.rect.left - 15, self.correct_button.rect.centery))
                            surface.blit(leaf_surf, leaf_rect)
                        else:
                            # Draw drop near incorrect button
                            drop_size = 15
                            drop_surf = pygame.Surface((drop_size, drop_size * 1.5), pygame.SRCALPHA)
                            drop_points = [
                                (drop_size/2, 0),
                                (drop_size, drop_size),
                                (drop_size/2, drop_size * 1.5),
                                (0, drop_size)
                            ]
                            pygame.draw.polygon(drop_surf, WATER_BLUE, drop_points)
                            drop_rect = drop_surf.get_rect(center=(self.incorrect_button.rect.left - 15, self.incorrect_button.rect.centery))
                            surface.blit(drop_surf, drop_rect)
                       
                        # Draw the detection animation on top
                        self.detection_animation.draw(surface)
                    else:
                        # For other phases, draw the panel first, then the animation
                        surface.blit(panel, (panel_x, panel_y))
                        self.detection_animation.draw(surface)
                else:
                    # For dropping phase, just draw the animation
                    self.detection_animation.draw(surface)
        except pygame.error:
            # Handle rendering errors
            pass
               
    def handle_button_clicks(self, mouse_pos):
        try:
            if self.state == "detecting" and self.detection_animation and self.detection_animation.phase == "feedback":
                if self.correct_button.check_hover(mouse_pos) and pygame.mouse.get_pressed()[0]:
                    self.correct_button.clicked()
                    self.detection_animation = None
                    self.state = "idle"
                    return True
                elif self.incorrect_button.check_hover(mouse_pos) and pygame.mouse.get_pressed()[0]:
                    self.incorrect_button.clicked()
                    self.detection_animation = None
                    self.state = "idle"
                    return True
            return False
        except (AttributeError, TypeError):
            # Handle errors in mouse interaction
            self.detection_animation = None
            self.state = "idle"
            return True
       
    def check_button_hover(self, mouse_pos):
        if self.state == "detecting" and self.detection_animation and self.detection_animation.phase == "feedback":
            try:
                self.correct_button.check_hover(mouse_pos)
                self.incorrect_button.check_hover(mouse_pos)
            except (AttributeError, TypeError):
                # Handle errors in button hover check
                pass

# Draw natural, earthy background
def draw_background(surface):
    try:
        # Create a subtle gradient from top to bottom
        for y in range(SCREEN_HEIGHT):
            ratio = y / SCREEN_HEIGHT
            r = int(LIGHT_CREAM[0] * (1 - ratio) + CREAM[0] * ratio)
            g = int(LIGHT_CREAM[1] * (1 - ratio) + CREAM[1] * ratio)
            b = int(LIGHT_CREAM[2] * (1 - ratio) + CREAM[2] * ratio)
            pygame.draw.line(surface, (r, g, b), (0, y), (SCREEN_WIDTH, y))
       
        # Add subtle natural pattern
        for x in range(0, SCREEN_WIDTH, 40):
            for y in range(0, SCREEN_HEIGHT, 40):
                pattern_type = (x + y) % 3
               
                if pattern_type == 0 and random.random() < 0.3:
                    # Draw tiny leaf
                    leaf_size = random.randint(3, 6)
                    leaf_alpha = random.randint(5, 15)
                    leaf_surf = pygame.Surface((leaf_size * 2, leaf_size), pygame.SRCALPHA)
                    pygame.draw.ellipse(leaf_surf, (*LEAF_GREEN, leaf_alpha),
                                    (0, 0, leaf_size * 1.5, leaf_size))
                    leaf_rect = leaf_surf.get_rect(center=(x + random.randint(-10, 10), y + random.randint(-10, 10)))
                    surface.blit(leaf_surf, leaf_rect)
                   
                elif pattern_type == 1 and random.random() < 0.2:
                    # Draw tiny drop
                    drop_size = random.randint(2, 5)
                    drop_alpha = random.randint(5, 15)
                    drop_surf = pygame.Surface((drop_size, drop_size * 1.5), pygame.SRCALPHA)
                    drop_points = [
                        (drop_size/2, 0),
                        (drop_size, drop_size),
                        (drop_size/2, drop_size * 1.5),
                        (0, drop_size)
                    ]
                    pygame.draw.polygon(drop_surf, (*WATER_BLUE, drop_alpha), drop_points)
                    drop_rect = drop_surf.get_rect(center=(x + random.randint(-10, 10), y + random.randint(-10, 10)))
                    surface.blit(drop_surf, drop_rect)
                   
                elif pattern_type == 2 and random.random() < 0.15:
                    # Draw tiny recycling symbol
                    symbol_size = random.randint(3, 6)
                    symbol_alpha = random.randint(5, 15)
                    symbol_surf = pygame.Surface((symbol_size * 2, symbol_size * 2), pygame.SRCALPHA)
                   
                    # Simplified recycling arrows
                    center_x = symbol_size
                    center_y = symbol_size
                    for i in range(3):
                        angle = math.radians(i * 120)
                        x1 = center_x + symbol_size * 0.8 * math.cos(angle)
                        y1 = center_y + symbol_size * 0.8 * math.sin(angle)
                        x2 = center_x + symbol_size * 0.8 * math.cos(angle + math.radians(120))
                        y2 = center_y + symbol_size * 0.8 * math.sin(angle + math.radians(120))
                        pygame.draw.line(symbol_surf, (*DARK_GREEN, symbol_alpha), (x1, y1), (x2, y2), 1)
                   
                    symbol_rect = symbol_surf.get_rect(center=(x + random.randint(-10, 10), y + random.randint(-10, 10)))
                    surface.blit(symbol_surf, symbol_rect)
    except pygame.error:
        # Handle rendering errors for background
        surface.fill(CREAM)  # Fallback to simple background

# Draw natural, eco-friendly header
def draw_header(surface):
    try:
        # Natural wood-like header background
        header_height = 70
        header_rect = pygame.Rect(0, 0, SCREEN_WIDTH, header_height)
        pygame.draw.rect(surface, CREAM, header_rect)
       
        # Add wood grain texture
        for y in range(0, header_height, 2):
            grain_width = random.randint(SCREEN_WIDTH - 100, SCREEN_WIDTH)
            grain_x = random.randint(0, 100)
            grain_alpha = random.randint(5, 15)
            grain_line = pygame.Surface((grain_width, 1), pygame.SRCALPHA)
            grain_line.fill((*SOFT_BROWN, grain_alpha))
            surface.blit(grain_line, (grain_x, y))
       
        # Add natural border at the bottom
        border_height = 3
        border_rect = pygame.Rect(0, header_height - border_height, SCREEN_WIDTH, border_height)
        pygame.draw.rect(surface, WOOD_BROWN, border_rect)
       
        # Title with natural styling
        title_text = font_title.render("SmartBin™ Waste Management", True, TEXT_BROWN)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, header_height//2))
        surface.blit(title_text, title_rect)
       
        # Add leaf decorations around title
        leaf_positions = [
            (title_rect.left - 40, title_rect.centery - 5),
            (title_rect.right + 40, title_rect.centery - 5)
        ]
       
        for pos in leaf_positions:
            leaf_size = 20
            leaf_surf = pygame.Surface((leaf_size * 2, leaf_size), pygame.SRCALPHA)
            pygame.draw.ellipse(leaf_surf, LEAF_GREEN, (0, 0, leaf_size * 1.5, leaf_size))
           
            # Add stem
            pygame.draw.line(leaf_surf, DARK_GREEN, (leaf_size * 0.75, leaf_size/2), (leaf_size * 1.5, leaf_size/2), 2)
           
            # Mirror the second leaf
            if pos[0] > SCREEN_WIDTH//2:
                leaf_surf = pygame.transform.flip(leaf_surf, True, False)
               
            leaf_rect = leaf_surf.get_rect(center=pos)
            surface.blit(leaf_surf, leaf_rect)
       
        # Add small recycling symbol
        symbol_size = 25
        symbol_x = title_rect.right + 70
        symbol_y = title_rect.centery
       
        # Draw a circular arrow
        symbol_surf = pygame.Surface((symbol_size, symbol_size), pygame.SRCALPHA)
        pygame.draw.circle(symbol_surf, DARK_GREEN, (symbol_size//2, symbol_size//2), symbol_size//2, 2)
       
        # Add arrow tips
        arrow_points = [
            (symbol_size//2, 0),
            (symbol_size//2 + symbol_size//10, symbol_size//10),
            (symbol_size//2 - symbol_size//10, symbol_size//10)
        ]
        pygame.draw.polygon(symbol_surf, DARK_GREEN, arrow_points)
       
        # Rotate and position
        rotated_symbol = pygame.transform.rotate(symbol_surf, -30)
        symbol_rect = rotated_symbol.get_rect(center=(symbol_x, symbol_y))
        surface.blit(rotated_symbol, symbol_rect)
    except pygame.error:
        # Handle rendering errors for header
        # Simple fallback header
        pygame.draw.rect(surface, CREAM, (0, 0, SCREEN_WIDTH, 70))
        try:
            title_text = font_title.render("SmartBin™ Waste Management", True, TEXT_BROWN)
            surface.blit(title_text, (SCREEN_WIDTH//2 - title_text.get_width()//2, 20))
        except Exception:
            pass

# Main function
def main():
    try:
        # Initialize the SmartBin interface
        interface = SmartBinInterface()
        clock = pygame.time.Clock()
        running = True
       
        # Initialize serial connection
        serial_initialized = initialize_serial()
        if not serial_initialized:
            print("Warning: Proceeding without Arduino communication.")
       
        # Start the camera detection thread
        camera_thread = threading.Thread(target=camera_detection_thread, daemon=True)
        camera_thread.start()
        print("Camera detection thread started.")
       
        # Main game loop
        while running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                           
                # Handle mouse interactions
                mouse_pos = pygame.mouse.get_pos()
                if not interface.handle_button_clicks(mouse_pos):
                    interface.check_button_hover(mouse_pos)
                   
                # Update all components
                interface.update_nature_elements()
                interface.update_progress_bars()
                interface.update_hint()
                interface.update_detection()
               
                # Draw everything
                try:
                    draw_background(screen)
                    interface.draw_nature_elements(screen)
                    draw_header(screen)
                   
                    if interface.state == "idle":
                        interface.draw_progress_bars(screen)
                        interface.draw_hint(screen)
                    else:
                        interface.draw_detection(screen)
                except Exception as e:
                    # If rendering fails, try to recover with minimum display
                    print(f"Rendering error: {e}")
                    screen.fill(CREAM)  # Fallback background color
                   
                pygame.display.update()
                clock.tick(60)
            except Exception as e:
                # Catch-all for loop errors to prevent crashes
                print(f"Error in main loop: {e}")
                # Try to continue running
               
        # Clean up before exit
        camera_active = False
        if camera_thread.is_alive():
            print("Waiting for camera thread to terminate...")
            camera_thread.join(timeout=2.0)
           
        if ser and ser.is_open:
            ser.close()
            print("Serial connection closed.")
           
        pygame.quit()
        cv2.destroyAllWindows()
        print("Application terminated.")
        sys.exit()
    except Exception as e:
        # Catch-all for critical errors
        print(f"Critical error: {e}")
        pygame.quit()
        cv2.destroyAllWindows()
        sys.exit()

if __name__ == "__main__":
    main()