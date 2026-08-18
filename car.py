"""
Car Entity Module.
Handles physics movement, ray casting (sensors), track collision detection,
skid mark rendering, and neural network driving logic.
"""

import time
import numpy as np
import pygame as pg
from neural_network import NeuralNetwork


class Car:
    """Represents an individual vehicle (Human or AI driven)."""

    def __init__(self, name="AI", image_path="car_sprite_yellow.png", size=(72, 36)):
        self.name = name
        self.size = size  # (width, height)
        self.position = np.array([400.0, 80.0])
        self.orientation = 0.0  # Angle in degrees

        # Physics Properties
        self.speed = 0.0
        self.max_speed = 8.0
        self.acceleration = 0.2
        self.friction = 0.05
        self.turn_speed = 3.0

        # Sprite Initialization
        self.image_path = image_path
        raw_image = pg.image.load(self.image_path).convert_alpha()
        self.original_image = pg.transform.scale(raw_image, self.size)
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect()

        # Neural Network & Fitness Tracking
        self.nn = NeuralNetwork()
        self.is_alive = True
        self.time_alive = 0
        self.fitness = 0.0

        # Checkpoint & Timing Tracking
        self.checkpoint_index = 0
        self.time_since_checkpoint = 0
        self.prev_distance_to_checkpoint = None

        self.lap_start_time = time.time()
        self.current_lap_time = 0.0
        self.best_lap_time = float("inf")
        self.laps_completed = 0

        # Skidmark Wheel Anchors
        self.prev_left_wheel = None
        self.prev_right_wheel = None

    def get_wheel_positions(self):
        """Calculates rear wheel world coordinates based on vehicle orientation."""
        rad = np.radians(self.orientation)
        length, width = max(self.size), min(self.size)

        rear_offset = -length * 0.33
        side_offset = width * 0.38

        if self.size[0] > self.size[1]:
            forward = np.array([np.cos(rad), np.sin(rad)])
            right = np.array([-np.sin(rad), np.cos(rad)])
        else:
            forward = np.array([np.sin(rad), -np.cos(rad)])
            right = np.array([np.cos(rad), np.sin(rad)])

        rear_center = self.position + forward * rear_offset
        left_wheel = rear_center - right * side_offset
        right_wheel = rear_center + right * side_offset

        return left_wheel, right_wheel

    def draw_skidmarks(self, skidmark_surface, is_turning_hard):
        """Draws continuous line segments on the persistent skidmark surface."""
        if not self.is_alive:
            return

        high_speed = abs(self.speed) > (self.max_speed * 0.5)

        if high_speed and is_turning_hard:
            left_wheel, right_wheel = self.get_wheel_positions()

            if self.prev_left_wheel is not None and self.prev_right_wheel is not None:
                skid_color = (30, 30, 30, 140)
                skid_width = 4

                pg.draw.line(skidmark_surface, skid_color, self.prev_left_wheel, left_wheel, skid_width)
                pg.draw.line(skidmark_surface, skid_color, self.prev_right_wheel, right_wheel, skid_width)

            self.prev_left_wheel = left_wheel
            self.prev_right_wheel = right_wheel
        else:
            self.prev_left_wheel = None
            self.prev_right_wheel = None

    def update_lap_timer(self):
        """Updates elapsed time for current lap."""
        if self.is_alive:
            self.current_lap_time = time.time() - self.lap_start_time

    def check_checkpoint(self, checkpoints, leaderboard):
        """Checks if vehicle reached current target checkpoint and records lap times."""
        if not self.is_alive or not checkpoints:
            return

        target_cp = checkpoints[self.checkpoint_index % len(checkpoints)]
        distance = np.linalg.norm(target_cp - self.position)

        if distance < 50:
            self.checkpoint_index += 1

            # Full Lap Completed
            if self.checkpoint_index > 0 and self.checkpoint_index % len(checkpoints) == 0:
                lap_time = time.time() - self.lap_start_time
                self.laps_completed += 1

                if lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time

                leaderboard.add_entry(self.name, lap_time)
                self.lap_start_time = time.time()

    def draw(self, screen, checkpoints=None):
        """Renders rotated car sprite and target line on screen."""
        rotated_image = pg.transform.rotate(self.original_image, -self.orientation)
        new_rect = rotated_image.get_rect(center=self.position)
        screen.blit(rotated_image, new_rect.topleft)

        if checkpoints and self.is_alive:
            target_cp = checkpoints[self.checkpoint_index % len(checkpoints)]
            pg.draw.line(screen, (0, 255, 0), self.position, target_cp, 1)

    def cast_rays(self, screen, track_mask):
        """Casts distance measurement rays to detect track walls."""
        offsets = [-60, -30, 0, 30, 60]
        sensor_distances = []

        for offset in offsets:
            angle = np.radians(self.orientation + offset)
            max_len = 200
            step_size = 2

            x, y = int(self.position[0]), int(self.position[1])

            for d in range(0, max_len, step_size):
                x = int(self.position[0] + d * np.cos(angle))
                y = int(self.position[1] + d * np.sin(angle))

                # Boundary Check
                if x >= screen.get_width() or x < 0 or y >= screen.get_height() or y < 0:
                    break

                # Track Mask Collision Check
                if track_mask.get_at((x, y)) == 0:
                    break

            sensor_distances.append(d)
            pg.draw.line(screen, (0, 255, 0), (int(self.position[0]), int(self.position[1])), (x, y), 2)

        return sensor_distances

    def ai_update(self, sensor_distances, track_mask, checkpoints):
        """Processes AI sensor inputs through Neural Network and updates movement."""
        if not self.is_alive:
            return

        target_cp = checkpoints[self.checkpoint_index % len(checkpoints)]
        vector = target_cp - self.position
        distance_to_checkpoint = np.linalg.norm(vector)

        angle_to_checkpoint = np.degrees(np.arctan2(vector[1], vector[0]))
        relative_angle = (angle_to_checkpoint - self.orientation + 180) % 360 - 180

        if self.prev_distance_to_checkpoint is None:
            self.prev_distance_to_checkpoint = distance_to_checkpoint

        # Prepare Inputs for Neural Network (7 Inputs)
        inputs = [
            *[d / 200.0 for d in sensor_distances],
            self.speed / self.max_speed,
            relative_angle / 180.0,
        ]
        outputs = self.nn.forward(inputs)

        # Output Decisions
        turned = False
        if outputs[0] > 0.5 and self.speed < self.max_speed:
            self.speed += self.acceleration
        elif outputs[1] > 0.5 and self.speed > -self.max_speed:
            self.speed -= self.acceleration

        if outputs[2] > 0.5:
            self.orientation -= self.turn_speed
            turned = True
        elif outputs[3] > 0.5:
            self.orientation += self.turn_speed
            turned = True

        # Friction
        if self.speed > 0:
            self.speed -= self.friction
        elif self.speed < 0:
            self.speed += self.friction

        if abs(self.speed) < self.friction:
            self.speed = 0

        # Update Position
        rad = np.radians(self.orientation)
        self.position += np.array([self.speed * np.cos(rad), self.speed * np.sin(rad)])

        self.time_alive += 1
        self.time_since_checkpoint += 1

        # Calculate Fitness Reward
        progress_delta = self.prev_distance_to_checkpoint - distance_to_checkpoint
        self.fitness += progress_delta * 2.0
        self.prev_distance_to_checkpoint = distance_to_checkpoint

        if self.speed > 0:
            self.fitness += (self.speed / self.max_speed) * 0.5

        if turned:
            self.fitness -= 0.1  # Slight penalty for excess steering

        if distance_to_checkpoint < 50:
            self.checkpoint_index += 1
            self.time_since_checkpoint = 0
            self.fitness += 1000  # Major reward for reaching checkpoint
            new_target = checkpoints[self.checkpoint_index % len(checkpoints)]
            self.prev_distance_to_checkpoint = np.linalg.norm(self.position - new_target)

        # Wall Collision Check
        ix, iy = int(self.position[0]), int(self.position[1])
        if (
            ix < 0
            or ix >= track_mask.get_size()[0]
            or iy < 0
            or iy >= track_mask.get_size()[1]
            or track_mask.get_at((ix, iy)) == 0
        ):
            self.is_alive = False
            self.fitness -= 50  # Crash Penalty

        # Inactivity Timeout Penalty
        if self.time_since_checkpoint > 180 or (self.time_alive > 60 and self.speed < 0.2):
            self.is_alive = False
            self.fitness -= 20

    def human_update(self, track_mask, checkpoints):
        """Handles manual driving controls for player car."""
        if not self.is_alive:
            return

        keys = pg.key.get_pressed()

        # Throttle / Brake
        if (keys[pg.K_w] or keys[pg.K_UP]) and self.speed < self.max_speed:
            self.speed += self.acceleration
        elif (keys[pg.K_s] or keys[pg.K_DOWN]) and self.speed > -self.max_speed / 2:
            self.speed -= self.acceleration

        # Steering
        if keys[pg.K_a] or keys[pg.K_LEFT]:
            self.orientation -= self.turn_speed
        if keys[pg.K_d] or keys[pg.K_RIGHT]:
            self.orientation += self.turn_speed

        # Apply Friction
        if self.speed > 0:
            self.speed -= self.friction
        elif self.speed < 0:
            self.speed += self.friction

        if abs(self.speed) < self.friction:
            self.speed = 0

        # Update Position
        rad = np.radians(self.orientation)
        self.position += np.array([self.speed * np.cos(rad), self.speed * np.sin(rad)])

        # Check Checkpoints
        if checkpoints:
            target_cp = checkpoints[self.checkpoint_index % len(checkpoints)]
            if np.linalg.norm(target_cp - self.position) < 50:
                self.checkpoint_index += 1

        # Check Wall Collision
        ix, iy = int(self.position[0]), int(self.position[1])
        if (
            ix < 0
            or ix >= track_mask.get_size()[0]
            or iy < 0
            or iy >= track_mask.get_size()[1]
            or track_mask.get_at((ix, iy)) == 0
        ):
            self.is_alive = False