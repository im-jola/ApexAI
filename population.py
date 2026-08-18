"""
Population Manager Module for Genetic Algorithm.
Handles generation updates, selection, crossover, mutation, and real-time telemetry rendering.
"""

import random
import numpy as np
import pygame as pg
from car import Car


class Population:
    """Manages the generation of AI cars and executes genetic algorithm steps."""

    def __init__(self, size=30):
        self.size = size
        self.generation = 1
        self.cars = []
        self.starting_position = np.array([400.0, 80.0])
        self.best_fitness = 0.0
        self.fitness_history = []

        for _ in range(size):
            car = Car()
            car.position = np.copy(self.starting_position)
            self.cars.append(car)

    def update_all(self, screen, track_mask, checkpoints):
        """Updates all living AI cars in current generation."""
        any_alive = False
        for car in self.cars:
            if car.is_alive:
                distances = car.cast_rays(screen, track_mask)
                car.ai_update(distances, track_mask, checkpoints)
                car.draw(screen, checkpoints)
                any_alive = True
        return any_alive

    def evolve(self):
        """Evolves population using natural selection, crossover, and mutation."""
        # Sort cars by fitness (highest first)
        self.cars.sort(key=lambda car: car.fitness, reverse=True)

        best_gen_fitness = self.cars[0].fitness
        avg_gen_fitness = sum(c.fitness for c in self.cars) / self.size
        self.fitness_history.append((best_gen_fitness, avg_gen_fitness))

        # Select Top 20% Parents
        num_parents = max(1, len(self.cars) // 5)
        parents = self.cars[:num_parents]

        # Save Best Network to File
        if parents[0].fitness > self.best_fitness:
            self.best_fitness = parents[0].fitness
            parents[0].nn.save("best_car.npz")
            print(f"New High Score: {self.best_fitness:.2f} - Saved to best_car.npz!")

        # Create Next Generation
        new_generation = []

        # Elitism: Retain Top Champion Unchanged
        champion = Car()
        champion.position = np.copy(self.starting_position)
        champion.nn.w1 = np.copy(parents[0].nn.w1)
        champion.nn.b1 = np.copy(parents[0].nn.b1)
        champion.nn.w2 = np.copy(parents[0].nn.w2)
        champion.nn.b2 = np.copy(parents[0].nn.b2)
        new_generation.append(champion)

        # Fill Rest of Population with Offspring
        for _ in range(self.size - 1):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            child = Car()
            child.position = np.copy(self.starting_position)
            child.nn = parent1.nn.crossover(parent2.nn)
            child.nn.mutate(rate=0.1, scale=0.2)
            new_generation.append(child)

        self.cars = new_generation
        self.generation += 1

    def load_best_car(self, filename="best_car.npz"):
        """Loads pretrained neural network weights into current population."""
        try:
            for car in self.cars:
                car.nn.load(filename)
                car.position = np.copy(self.starting_position)
                car.is_alive = True
                car.orientation = 0.0
                car.speed = 0.0
                car.fitness = 0.0
                car.checkpoint_index = 0
                car.time_since_checkpoint = 0
                car.time_alive = 0
            print(f"Loaded weights from {filename}. Population reset.")
        except FileNotFoundError:
            print(f"File {filename} not found.")

    def draw_telemetry_graph(self, screen, x=570, y=10, width=220, height=120):
        """Renders live fitness progress telemetry graph."""
        panel = pg.Surface((width, height), pg.SRCALPHA)
        panel.fill((20, 20, 20, 200))
        screen.blit(panel, (x, y))
        pg.draw.rect(screen, (100, 100, 100), (x, y, width, height), 1)

        if len(self.fitness_history) < 2:
            return

        max_val = max(max(best, avg) for best, avg in self.fitness_history)
        if max_val <= 0:
            max_val = 1.0

        best_points, avg_points = [], []
        n_gens = len(self.fitness_history)
        padding = 8
        graph_w = width - (padding * 2)
        graph_h = height - (padding * 2)

        for i, (best_fit, avg_fit) in enumerate(self.fitness_history):
            px = x + padding + (i / (n_gens - 1)) * graph_w
            py_best = (y + height - padding) - (max(0, best_fit) / max_val) * graph_h
            py_avg = (y + height - padding) - (max(0, avg_fit) / max_val) * graph_h

            best_points.append((px, py_best))
            avg_points.append((px, py_avg))

        if len(avg_points) > 1:
            pg.draw.lines(screen, (255, 215, 0), False, avg_points, 2)  # Gold: Average
        if len(best_points) > 1:
            pg.draw.lines(screen, (0, 255, 128), False, best_points, 2)  # Green: Best

    def draw_nn_visualizer(self, screen, x=570, y=140, width=220, height=220, font=None):
        """Draws live visual representation of best active car's neural network."""
        best_car = next((c for c in self.cars if c.is_alive), self.cars[0])
        nn = best_car.nn

        panel = pg.Surface((width, height), pg.SRCALPHA)
        panel.fill((20, 20, 20, 200))
        screen.blit(panel, (x, y))
        pg.draw.rect(screen, (100, 100, 100), (x, y, width, height), 1)

        x_in, x_hid, x_out = x + 30, x + 110, x + 190

        in_pos = [(x_in, int(y + 20 + i * ((height - 40) / 6))) for i in range(7)]
        hid_pos = [(x_hid, int(y + 25 + i * ((height - 50) / 5))) for i in range(6)]
        out_pos = [(x_out, int(y + 35 + i * ((height - 70) / 3))) for i in range(4)]

        # Render Synaptic Connections
        for i, p1 in enumerate(in_pos):
            for j, p2 in enumerate(hid_pos):
                weight = nn.w1[i, j]
                color = (0, 220, 0) if weight > 0 else (220, 0, 0)
                thickness = max(1, min(4, int(abs(weight) * 2.5)))
                pg.draw.line(screen, color, p1, p2, thickness)

        for j, p1 in enumerate(hid_pos):
            for k, p2 in enumerate(out_pos):
                weight = nn.w2[j, k]
                color = (0, 220, 0) if weight > 0 else (220, 0, 0)
                thickness = max(1, min(4, int(abs(weight) * 2.5)))
                pg.draw.line(screen, color, p1, p2, thickness)

        # Render Layer Nodes
        def draw_layer_nodes(positions, activations):
            for i, pos in enumerate(positions):
                act = np.clip(activations[i], 0.0, 1.0)
                color = (int(50 + 205 * act), int(50 + 205 * act), int(50 * (1 - act)))
                pg.draw.circle(screen, color, pos, 6)
                pg.draw.circle(screen, (255, 255, 255), pos, 6, 1)

        draw_layer_nodes(in_pos, nn.last_inputs)
        draw_layer_nodes(hid_pos, nn.last_hidden)
        draw_layer_nodes(out_pos, nn.last_outputs)

        # Output Action Labels
        if font:
            labels = ["Accel", "Brake", "Left", "Right"]
            for k, pos in enumerate(out_pos):
                is_active = nn.last_outputs[k] > 0.5
                lbl_color = (0, 255, 0) if is_active else (150, 150, 150)
                txt = font.render(labels[k], True, lbl_color)
                screen.blit(txt, (pos[0] - 38, pos[1] - 6))