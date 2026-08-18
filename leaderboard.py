"""
Leaderboard Display Module.
Renders fast lap records and tracks live human player stats on screen.
"""

import pygame as pg


class Leaderboard:
    """Stores lap records and renders formatted standings box on screen."""

    def __init__(self, max_entries=5):
        self.best_times = {}  # {driver_name: best_lap_time}
        self.max_entries = max_entries

    def add_entry(self, driver_name, lap_time):
        """Updates driver's score if lap time is faster than previous best."""
        if driver_name not in self.best_times or lap_time < self.best_times[driver_name]:
            self.best_times[driver_name] = lap_time

    def draw(self, screen, x=10, y=100, width=220, height=150, font=None, player_car=None):
        """Renders leaderboard overlay panel on screen."""
        panel = pg.Surface((width, height), pg.SRCALPHA)
        panel.fill((20, 20, 20, 210))
        screen.blit(panel, (x, y))
        pg.draw.rect(screen, (100, 100, 100), (x, y, width, height), 1)

        if not font:
            return

        # Title Header
        title = font.render("LEADERBOARD (Best Laps)", True, (255, 215, 0))
        screen.blit(title, (x + 10, y + 8))

        # Sorted Leaderboard Entries
        sorted_entries = sorted(self.best_times.items(), key=lambda item: item[1])[: self.max_entries]

        if not sorted_entries:
            empty_txt = font.render("No completed laps yet", True, (150, 150, 150))
            screen.blit(empty_txt, (x + 10, y + 35))
        else:
            for i, (name, lap_time) in enumerate(sorted_entries):
                color = (0, 220, 255) if "Player" in name else (225, 225, 225)
                row_text = f"{i+1}. {name}: {lap_time:.2f}s"
                txt_surf = font.render(row_text, True, color)
                screen.blit(txt_surf, (x + 10, y + 32 + i * 18))

        # Human Player Live Tracker Footer
        if player_car:
            pg.draw.line(screen, (80, 80, 80), (x + 5, y + height - 32), (x + width - 5, y + height - 32))
            p_best = f"{player_car.best_lap_time:.2f}s" if player_car.best_lap_time < float("inf") else "N/A"
            p_text = font.render(
                f"Your Best: {p_best} | Current: {player_car.current_lap_time:.1f}s",
                True,
                (0, 220, 255),
            )
            screen.blit(p_text, (x + 10, y + height - 24))