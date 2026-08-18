"""
Main Execution Script for Self-Driving Car Simulation with Genetic Algorithm.
Includes Phase 1 (Checkpoint Editor) and Phase 2 (Real-Time AI & Player Simulation).
"""

import sys
import time
import numpy as np
import pygame as pg

from car import Car
from leaderboard import Leaderboard
from population import Population


def main():
    # ==========================================
    # INITIALIZATION & SETUP
    # ==========================================
    pg.init()
    pg.font.init()

    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pg.display.set_caption("2D Self-Driving Car - AI Genetic Algorithm")
    clock = pg.time.Clock()

    font = pg.font.SysFont("Arial", 16)
    cp_font = pg.font.SysFont("Arial", 14)

    # Load Track and Create Collision Mask
    track_surface = pg.image.load("custom_track.png").convert()
    track_mask = pg.mask.from_surface(track_surface)

    # Create Transparent Surface for Tire Skid Marks
    skidmark_surface = pg.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pg.SRCALPHA)

    # Initialize Entities
    population = Population(size=30)
    leaderboard = Leaderboard()
    player_car = Car(name="Player", image_path="car_sprite_red.png")

    checkpoints = []
    edit_mode = True
    running = True
    speed_multiplier = 1

    editor_instructions = [
        "--- CHECKPOINT EDITOR ---",
        "Left Click   : Add Checkpoint",
        "Right Click : Remove Checkpoint",
        "C               : Clear All Checkpoints",
        "P               : Print Array to Console",
        "SPACE      : Finish & Start Simulation",
    ]

    # ==========================================
    # PHASE 1: CHECKPOINT EDITOR LOOP
    # ==========================================
    while running and edit_mode:
        screen.blit(track_surface, (0, 0))

        for i, cp in enumerate(checkpoints):
            pg.draw.circle(screen, (0, 255, 0), cp.astype(int), 8)
            txt = cp_font.render(str(i + 1), True, (255, 255, 255))
            screen.blit(txt, (cp[0] + 10, cp[1] - 10))

        panel_width, panel_height = 270, 135
        panel = pg.Surface((panel_width, panel_height), pg.SRCALPHA)
        panel.fill((20, 20, 20, 210))
        screen.blit(panel, (10, 10))
        pg.draw.rect(screen, (0, 255, 0), (10, 10, panel_width, panel_height), 1)

        for idx, line in enumerate(editor_instructions):
            color = (255, 215, 0) if idx == 0 else (255, 255, 255)
            txt_surf = font.render(line, True, color)
            screen.blit(txt_surf, (20, 15 + idx * 20))

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                edit_mode = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    checkpoints.append(np.array([float(event.pos[0]), float(event.pos[1])]))
                elif event.button == 3:
                    mouse_pos = np.array([float(event.pos[0]), float(event.pos[1])])
                    removed = False
                    for i, cp in enumerate(checkpoints):
                        if np.linalg.norm(cp - mouse_pos) < 15:
                            checkpoints.pop(i)
                            removed = True
                            break
                    if not removed and checkpoints:
                        checkpoints.pop()

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_z or event.key == pg.K_BACKSPACE:
                    if checkpoints:
                        checkpoints.pop()
                elif event.key == pg.K_c:
                    checkpoints.clear()
                elif event.key == pg.K_p:
                    print("\ncheckpoints = [")
                    for cp in checkpoints:
                        print(f"    np.array([{cp[0]}, {cp[1]}]),")
                    print("]\n")
                elif event.key == pg.K_SPACE:
                    if len(checkpoints) >= 2:
                        edit_mode = False
                    else:
                        print("Please place at least 2 checkpoints before starting.")

        pg.display.flip()
        clock.tick(60)

    # ==========================================
    # SIMULATION PREPARATION
    # ==========================================
    if running and len(checkpoints) >= 2:
        start_pos = np.copy(checkpoints[0])

        population.starting_position = np.copy(start_pos)
        for car in population.cars:
            car.position = np.copy(start_pos)

        def reset_player():
            player_car.position = np.copy(start_pos)
            player_car.orientation = 0
            player_car.speed = 0.0
            player_car.is_alive = True
            player_car.checkpoint_index = 0
            player_car.lap_start_time = time.time()

        reset_player()

    # ==========================================
    # PHASE 2: MAIN SIMULATION LOOP
    # ==========================================
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_l:
                    population.load_best_car("best_car.npz")
                elif event.key == pg.K_r:
                    reset_player()
                elif event.key == pg.K_1:
                    speed_multiplier = 1
                elif event.key == pg.K_2:
                    speed_multiplier = 5
                elif event.key == pg.K_3:
                    speed_multiplier = 20
                elif event.key == pg.K_4:
                    speed_multiplier = 100

        screen.fill((30, 30, 30))
        screen.blit(track_surface, (0, 0))

        for cp in checkpoints:
            pg.draw.circle(screen, (0, 255, 0), cp.astype(int), 6)

        for _ in range(speed_multiplier):
            keys = pg.key.get_pressed()
            player_turning = keys[pg.K_a] or keys[pg.K_d] or keys[pg.K_LEFT] or keys[pg.K_RIGHT]

            player_car.human_update(track_mask, checkpoints)
            player_car.draw_skidmarks(skidmark_surface, player_turning)
            player_car.update_lap_timer()
            player_car.check_checkpoint(checkpoints, leaderboard)

            if not population.update_all(screen, track_mask, checkpoints):
                population.evolve()
                for car in population.cars:
                    car.lap_start_time = time.time()
                break

            for idx, car in enumerate(population.cars):
                if car.is_alive:
                    car.name = f"AI #{idx+1}"
                    ai_turning = abs(car.speed) > 2.0
                    car.draw_skidmarks(skidmark_surface, ai_turning)
                    car.update_lap_timer()
                    car.check_checkpoint(checkpoints, leaderboard)

        screen.blit(skidmark_surface, (0, 0))

        if player_car.is_alive:
            player_car.cast_rays(screen, track_mask)
            player_car.draw(screen, checkpoints)

        gen_text = font.render(f"Generation: {population.generation}", True, (255, 255, 0))
        alive_count = sum(1 for c in population.cars if c.is_alive)
        alive_text = font.render(f"Cars Alive: {alive_count}/{population.size}", True, (0, 255, 0))
        ff_text = font.render(f"Speed: {speed_multiplier}x (Keys 1-4)", True, (255, 255, 255))

        screen.blit(gen_text, (10, 10))
        screen.blit(alive_text, (10, 35))
        screen.blit(ff_text, (10, 60))

        leaderboard.draw(screen, font=font, player_car=player_car)
        population.draw_telemetry_graph(screen)
        population.draw_nn_visualizer(screen, font=font)

        pg.display.flip()
        clock.tick(60 if speed_multiplier == 1 else 0)

    pg.quit()
    sys.exit()


if __name__ == "__main__":
    main()