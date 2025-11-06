"""Plane Jumpers - an endless runner inspired by Subway Surfers.

This module creates a simple 2D arcade game using pygame. The player pilots a
paper airplane through three sky lanes, dodging storm clouds and collecting
rings. Speed ramps up over time, and the run ends when the plane collides with a
cloud. The script can be executed directly:

    python plane_jumpers.py

Controls
--------
LEFT / RIGHT (or A / D): Shift lanes
SPACE (or W / UP): Burst upward to dodge low hazards
ESC: Quit

The code is structured so that core game components are encapsulated inside
classes, which makes further customization easier for designers. Assets are
simple procedurally drawn surfaces, keeping the example dependency free beyond
pygame itself.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pygame

# Window configuration
WIDTH, HEIGHT = 480, 720
FPS = 60
LANES = (WIDTH * 0.25, WIDTH * 0.5, WIDTH * 0.75)

# Player constants
PLANE_WIDTH, PLANE_HEIGHT = 70, 46
PLANE_LIFT = -13
GRAVITY = 0.75
MAX_DESCENT = 12

# Gameplay constants
SCROLL_SPEED = 6
SPEED_INCREMENT = 0.0008
OBSTACLE_FREQUENCY = 1400  # milliseconds
RING_FREQUENCY = 900       # milliseconds
INVULN_TIME = 650          # milliseconds after a ring pickup

BACKGROUND_COLOR = (15, 32, 64)
SKY_COLOR = (32, 64, 128)
RING_COLOR = (255, 207, 74)
CLOUD_COLOR = (220, 235, 245)
PLANE_COLOR = (110, 200, 255)
BOOST_COLOR = (255, 80, 120)


@dataclass
class Actor:
    image: pygame.Surface
    rect: pygame.Rect
    velocity: pygame.Vector2

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.image, self.rect)

    def update(self, dt: float) -> None:
        self.rect.x += int(self.velocity.x * dt)
        self.rect.y += int(self.velocity.y * dt)


class Plane(Actor):
    def __init__(self) -> None:
        image = pygame.Surface((PLANE_WIDTH, PLANE_HEIGHT), pygame.SRCALPHA)
        self._draw_plane(image)
        rect = image.get_rect()
        rect.centerx = LANES[1]
        rect.bottom = HEIGHT - 40
        super().__init__(image, rect, pygame.Vector2(0, 0))
        self.lane_index = 1
        self.invulnerable_until = 0

    @staticmethod
    def _draw_plane(surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        body = [(0, h // 2), (int(w * 0.65), 0), (w, h // 2), (int(w * 0.65), h)]
        pygame.draw.polygon(surface, PLANE_COLOR, body)
        pygame.draw.polygon(surface, (255, 255, 255), body, 3)
        wing = [(int(w * 0.25), h // 2), (int(w * 0.6), int(h * 0.1)),
                (int(w * 0.55), int(h * 0.9))]
        pygame.draw.polygon(surface, BOOST_COLOR, wing)

    def switch_lane(self, direction: int) -> None:
        new_index = self.lane_index + direction
        new_index = max(0, min(len(LANES) - 1, new_index))
        if new_index != self.lane_index:
            self.lane_index = new_index
            self.rect.centerx = int(LANES[self.lane_index])

    def lift(self) -> None:
        self.velocity.y = PLANE_LIFT

    def apply_gravity(self) -> None:
        self.velocity.y = min(self.velocity.y + GRAVITY, MAX_DESCENT)

    def update(self, dt: float, time_now: int) -> None:
        super().update(dt)
        self.rect.bottom = min(self.rect.bottom, HEIGHT - 40)
        self.rect.top = max(self.rect.top, HEIGHT // 4)
        if time_now > self.invulnerable_until:
            self.image.set_alpha(255)
        else:
            # Blink while invulnerable.
            blink = 200 + int(55 * math.sin(time_now / 100))
            self.image.set_alpha(blink)


class Cloud(Actor):
    def __init__(self) -> None:
        radius = random.randint(28, 42)
        image = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(image, CLOUD_COLOR, (radius, radius), radius)
        offset = random.choice((-12, 12))
        pygame.draw.circle(image, CLOUD_COLOR, (radius + offset, radius - 8), radius - 6)
        pygame.draw.circle(image, CLOUD_COLOR, (radius - offset, radius + 4), radius - 10)
        rect = image.get_rect()
        lane = random.choice(LANES)
        rect.centerx = lane
        rect.y = -rect.height
        velocity = pygame.Vector2(0, 0)
        super().__init__(image, rect, velocity)


class Ring(Actor):
    def __init__(self) -> None:
        radius = 22
        thickness = 6
        image = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(image, RING_COLOR, (radius, radius), radius)
        pygame.draw.circle(image, (255, 239, 200), (radius, radius), radius - thickness)
        rect = image.get_rect()
        rect.centerx = random.choice(LANES)
        rect.y = -rect.height
        super().__init__(image, rect, pygame.Vector2(0, 0))


class Starfield:
    def __init__(self, density: int = 90) -> None:
        self.stars: List[Tuple[float, float, float]] = []
        for _ in range(density):
            x = random.random() * WIDTH
            y = random.random() * HEIGHT
            depth = random.random()
            self.stars.append((x, y, depth))

    def update(self, dt: float, speed: float) -> None:
        for i, (x, y, depth) in enumerate(self.stars):
            y += dt * speed * (0.3 + depth)
            if y > HEIGHT:
                y -= HEIGHT
                x = random.random() * WIDTH
            self.stars[i] = (x, y, depth)

    def draw(self, surface: pygame.Surface) -> None:
        for x, y, depth in self.stars:
            brightness = 150 + int(105 * depth)
            pygame.draw.circle(surface, (brightness, brightness, 255), (int(x), int(y)), 2)


def spawn_actor(group: List[Actor], actor: Actor) -> None:
    group.append(actor)


def update_and_prune(group: List[Actor], dt: float, speed: float) -> None:
    for actor in group:
        actor.rect.y += int(speed * dt)
        actor.update(dt)
    group[:] = [a for a in group if a.rect.top <= HEIGHT]


def detect_collision(plane: Plane, obstacles: Iterable[Actor]) -> bool:
    for obstacle in obstacles:
        if plane.rect.colliderect(obstacle.rect):
            return True
    return False


def collect_rings(plane: Plane, rings: List[Ring], time_now: int) -> int:
    collected = 0
    remaining: List[Ring] = []
    for ring in rings:
        if plane.rect.colliderect(ring.rect):
            collected += 1
            plane.invulnerable_until = time_now + INVULN_TIME
            plane.image.set_alpha(180)
        else:
            remaining.append(ring)
    rings[:] = remaining
    return collected


def draw_gradient_background(surface: pygame.Surface, offset: float) -> None:
    surface.fill(BACKGROUND_COLOR)
    steps = HEIGHT // 4
    for i in range(steps):
        ratio = i / steps
        color = (
            int(SKY_COLOR[0] * ratio + BACKGROUND_COLOR[0] * (1 - ratio)),
            int(SKY_COLOR[1] * ratio + BACKGROUND_COLOR[1] * (1 - ratio)),
            int(SKY_COLOR[2] * ratio + BACKGROUND_COLOR[2] * (1 - ratio)),
        )
        y = (i * 4 + offset) % HEIGHT
        pygame.draw.rect(surface, color, (0, y, WIDTH, 4))


def render_text(surface: pygame.Surface, text: str, size: int, pos: Tuple[int, int], *,
                center: bool = False, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    font = pygame.font.Font(None, size)
    surf = font.render(text, True, color)
    rect = surf.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(surf, rect)


def main(*, max_frames: Optional[int] = None, screenshot_path: Optional[str] = None) -> None:
    pygame.init()
    pygame.display.set_caption("Plane Jumpers")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    plane = Plane()
    clouds: List[Cloud] = []
    rings: List[Ring] = []
    starfield = Starfield()

    last_cloud = pygame.time.get_ticks()
    last_ring = pygame.time.get_ticks()
    base_speed = SCROLL_SPEED
    score = 0
    best = 0
    running = True
    game_over = False

    frame_count = 0
    screenshot_saved = False

    while running:
        dt = clock.tick(FPS)
        time_now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                if not game_over:
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        plane.switch_lane(-1)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        plane.switch_lane(1)
                    elif event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                        plane.lift()
                else:
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        # Restart game
                        plane = Plane()
                        clouds.clear()
                        rings.clear()
                        starfield = Starfield()
                        base_speed = SCROLL_SPEED
                        score = 0
                        game_over = False
                        last_cloud = last_ring = time_now
        if not running:
            break

        pressed = pygame.key.get_pressed()
        if not game_over:
            if pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
                plane.switch_lane(-1)
            if pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
                plane.switch_lane(1)
            if pressed[pygame.K_SPACE] or pressed[pygame.K_UP] or pressed[pygame.K_w]:
                plane.lift()

        # Spawn logic
        if not game_over and time_now - last_cloud > OBSTACLE_FREQUENCY:
            spawn_actor(clouds, Cloud())
            last_cloud = time_now
        if not game_over and time_now - last_ring > RING_FREQUENCY:
            spawn_actor(rings, Ring())
            last_ring = time_now

        if not game_over:
            base_speed += SPEED_INCREMENT * dt
            plane.apply_gravity()
            plane.update(dt / 16.0, time_now)
            update_and_prune(clouds, dt / 16.0, base_speed)
            update_and_prune(rings, dt / 16.0, base_speed)
            starfield.update(dt / 16.0, base_speed)

            score += base_speed * dt / 600
            score += collect_rings(plane, rings, time_now) * 10

            if time_now > plane.invulnerable_until:
                if detect_collision(plane, clouds):
                    game_over = True
                    best = max(best, int(score))
        else:
            plane.image.set_alpha(180)

        # Rendering
        draw_gradient_background(screen, base_speed * time_now / 2000)
        starfield.draw(screen)
        for cloud in clouds:
            cloud.draw(screen)
        for ring in rings:
            ring.draw(screen)
        plane.draw(screen)

        render_text(screen, "Plane Jumpers", 48, (WIDTH // 2, 60), center=True)
        render_text(screen, f"Score: {int(score)}", 36, (20, 20))
        render_text(screen, f"Best: {int(best)}", 36, (WIDTH - 160, 20))

        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 10, 30, 180))
            screen.blit(overlay, (0, 0))
            render_text(screen, "Crash!", 72, (WIDTH // 2, HEIGHT // 2 - 60), center=True)
            render_text(screen, f"Final Score: {int(score)}", 48, (WIDTH // 2, HEIGHT // 2 + 10), center=True)
            render_text(screen, "Press SPACE to retry", 36, (WIDTH // 2, HEIGHT // 2 + 80), center=True)

        pygame.display.flip()

        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            if screenshot_path and not screenshot_saved:
                pygame.image.save(screen, screenshot_path)
                screenshot_saved = True
            break

    if screenshot_path and not screenshot_saved:
        pygame.image.save(screen, screenshot_path)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Plane Jumpers endless runner.")
    parser.add_argument("--frames", type=int, default=None,
                        help="Run the game loop for a limited number of frames (for testing).")
    parser.add_argument("--screenshot", type=str, default=None,
                        help="Path to save a screenshot of the final frame (useful for headless runs).")
    parser.add_argument("--headless", action="store_true",
                        help="Use the SDL dummy video driver to render without a visible window.")

    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    main(max_frames=args.frames, screenshot_path=args.screenshot)
