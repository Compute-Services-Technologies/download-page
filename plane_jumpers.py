"""Plane Jumpers - cinematic endless runner with a pseudo-3D skyline.

This module uses pygame to render an endless runner inspired by Subway Surfers
but themed around aerobatic planes weaving through floating rings and traffic.
Highlights include camera-swayed perspective lines, a luminous runway grid,
parallax skyscrapers, ambient air traffic, and a redesigned hero jet with
afterburn pulses and procedural exhaust. Everything is still drawn on the fly
so no external art assets are required.

Run the game directly to open the interactive window:

    python plane_jumpers.py

Command line flags make it suitable for automation:

    python plane_jumpers.py --headless --frames 600 --screenshot capture.png

You can also tailor a session directly from the CLI:

    python plane_jumpers.py --difficulty ace --seed 42

Controls
========
LEFT / RIGHT (or A / D)
    Bank the plane between the three aerial lanes.
SPACE (or W / UP)
    Pop upward to vault over low obstacles.
ESC or Q
    Quit the game.
ENTER (when crashed)
    Restart the run.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame

# Window configuration
WIDTH, HEIGHT = 560, 840
FPS = 60

# Perspective constants
HORIZON_Y = 180
GROUND_Y = HEIGHT - 140
LANE_NEAR = (WIDTH * 0.22, WIDTH * 0.5, WIDTH * 0.78)
LANE_FAR = (WIDTH * 0.44, WIDTH * 0.5, WIDTH * 0.56)

# Player constants
PLANE_ALTITUDE_LIMIT = 140.0
PLANE_LIFT = -22.0
GRAVITY = 1.6
MAX_DESCENT = 24.0
INVULN_TIME = 900  # milliseconds after a ring pickup

# Gameplay constants
SCROLL_SPEED = 240.0  # pixels per second
SPEED_INCREMENT = 18.0  # acceleration per second
OBSTACLE_INTERVAL = (820, 1340)  # ms range between spawns
RING_INTERVAL = (520, 920)
SPEEDLINE_SPAWN = 0.12  # chance per frame

# Difficulty tuning presets (multipliers are applied in main).
DIFFICULTY_PRESETS: Dict[str, Dict[str, float]] = {
    "relaxed": {
        "speed": SCROLL_SPEED * 0.82,
        "increment": SPEED_INCREMENT * 0.7,
        "obstacle_factor": 1.28,
        "ring_factor": 0.9,
    },
    "classic": {
        "speed": SCROLL_SPEED,
        "increment": SPEED_INCREMENT,
        "obstacle_factor": 1.0,
        "ring_factor": 1.0,
    },
    "ace": {
        "speed": SCROLL_SPEED * 1.22,
        "increment": SPEED_INCREMENT * 1.35,
        "obstacle_factor": 0.78,
        "ring_factor": 1.15,
    },
}


def _scale_interval(interval: Tuple[int, int], factor: float) -> Tuple[int, int]:
    """Scale an interval but clamp it to keep chronological ordering."""

    base_low, base_high = interval
    low = max(120, int(base_low * factor))
    high = max(low + 60, int(base_high * factor))
    return low, high


def _save_screenshot(surface: pygame.Surface, path: str) -> None:
    """Persist a screenshot ensuring parent directories exist."""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pygame.image.save(surface, path)


# Colours
BG_GRADIENT_TOP = (8, 28, 56)
BG_GRADIENT_BOTTOM = (96, 168, 236)
SUN_COLOR = (252, 196, 122)
LANE_GLOW = (64, 152, 255)
HUD_COLOR = (235, 244, 255)
WARNING_COLOR = (255, 88, 120)
TRAIL_COLOR = (148, 218, 255)


class Perspective:
    """Project lane indices into screen coordinates with camera sway."""

    def __init__(self) -> None:
        self.horizon = HORIZON_Y
        self.ground = GROUND_Y
        self.lane_near = LANE_NEAR
        self.lane_far = LANE_FAR
        self.camera_offset = 0.0
        self._target_offset = 0.0

    def _depth_t(self, y: float) -> float:
        if y <= self.horizon:
            return 0.0
        if y >= self.ground:
            return 1.0
        return (y - self.horizon) / (self.ground - self.horizon)

    def lane_x(self, lane: int, y: float) -> float:
        t = self._depth_t(y)
        far = self.lane_far[lane]
        near = self.lane_near[lane]
        base = far * (1.0 - t) + near * t
        sway = self.camera_offset * (1.0 - 0.75 * t) * WIDTH * 0.24
        lane_splay = (lane - 1) * self.camera_offset * (1.0 - t) * WIDTH * 0.08
        return base + sway + lane_splay

    def scale(self, y: float, *, bias: float = 1.0) -> float:
        t = self._depth_t(y)
        return 0.34 + 0.92 * t * bias

    def update_camera(self, dt_ms: int, lane_index: int, roll: float) -> None:
        dt = dt_ms / 1000.0
        lane_target = (lane_index - 1) * 0.18
        roll_target = max(-1.0, min(1.0, roll / 32.0)) * 0.1
        self._target_offset = lane_target + roll_target
        blend = min(1.0, 6.5 * dt)
        self.camera_offset += (self._target_offset - self.camera_offset) * blend


class SkyBackdrop:
    def __init__(self) -> None:
        self.clouds = [self._spawn_cloud() for _ in range(6)]
        self.cloud_speed = 22

    @staticmethod
    def _spawn_cloud() -> Tuple[float, float, float, float]:
        x = random.uniform(-120, WIDTH + 120)
        y = random.uniform(20, HORIZON_Y + 80)
        scale = random.uniform(0.8, 1.6)
        drift = random.uniform(10, 30)
        return x, y, scale, drift

    def update(self, dt_ms: int) -> None:
        dt = dt_ms / 1000.0
        updated = []
        for x, y, scale, drift in self.clouds:
            x += drift * dt
            if x - 200 > WIDTH:
                x = -200
                y = random.uniform(40, HORIZON_Y + 90)
                scale = random.uniform(0.9, 1.8)
                drift = random.uniform(14, 34)
            updated.append((x, y, scale, drift))
        self.clouds = updated

    def draw(self, surface: pygame.Surface) -> None:
        self._draw_gradient(surface)
        self._draw_sun(surface)
        for x, y, scale, _ in self.clouds:
            cloud_surface = pygame.Surface((220, 120), pygame.SRCALPHA)
            pygame.draw.ellipse(cloud_surface, (240, 250, 255, 210), (0, 20, 160, 80))
            pygame.draw.ellipse(cloud_surface, (210, 230, 255, 160), (60, 0, 160, 90))
            pygame.draw.ellipse(cloud_surface, (240, 250, 255, 190), (90, 28, 120, 70))
            scaled = pygame.transform.smoothscale(
                cloud_surface,
                (int(cloud_surface.get_width() * scale), int(cloud_surface.get_height() * scale)),
            )
            rect = scaled.get_rect(center=(int(x), int(y)))
            surface.blit(scaled, rect)

    @staticmethod
    def _draw_gradient(surface: pygame.Surface) -> None:
        for i in range(HEIGHT):
            lerp = i / HEIGHT
            r = int(BG_GRADIENT_TOP[0] * (1 - lerp) + BG_GRADIENT_BOTTOM[0] * lerp)
            g = int(BG_GRADIENT_TOP[1] * (1 - lerp) + BG_GRADIENT_BOTTOM[1] * lerp)
            b = int(BG_GRADIENT_TOP[2] * (1 - lerp) + BG_GRADIENT_BOTTOM[2] * lerp)
            surface.fill((r, g, b), rect=pygame.Rect(0, i, WIDTH, 1))

    @staticmethod
    def _draw_sun(surface: pygame.Surface) -> None:
        sun = pygame.Surface((240, 240), pygame.SRCALPHA)
        pygame.draw.circle(sun, SUN_COLOR + (140,), (120, 120), 120)
        pygame.draw.circle(sun, (255, 230, 170, 180), (120, 120), 90)
        pygame.draw.circle(sun, (255, 255, 255, 200), (120, 120), 46)
        surface.blit(sun, (WIDTH - 300, 40), special_flags=pygame.BLEND_ADD)


@dataclass
class TrailParticle:
    x: float
    y: float
    life: float
    drift: float
    scale: float
    max_life: float

    def update(self, dt_ms: int) -> None:
        frame = dt_ms / 16.0
        self.life -= dt_ms
        self.y += 32 * frame
        self.x += self.drift * frame
        self.scale *= 1.0 + 0.015 * frame

    def draw(self, surface: pygame.Surface) -> None:
        if self.life <= 0:
            return
        alpha = max(0, min(255, int(255 * (self.life / self.max_life))))
        radius = max(2, int(12 * self.scale))
        pygame.draw.circle(surface, TRAIL_COLOR + (alpha,), (int(self.x), int(self.y)), radius)


@dataclass
class SpeedLine:
    lane: int
    y: float
    length: float
    opacity: float

    def update(self, dt_ms: int, speed: float) -> None:
        dt = dt_ms / 1000.0
        self.y += (speed * 1.6) * dt
        self.opacity = max(0.0, self.opacity - 0.65 * dt)

    def draw(self, surface: pygame.Surface, perspective: Perspective) -> None:
        if self.opacity <= 0:
            return
        y1 = self.y
        y2 = self.y + self.length
        x1 = perspective.lane_x(self.lane, y1)
        x2 = perspective.lane_x(self.lane, y2)
        color = (180, 220, 255, int(180 * self.opacity))
        line_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.aaline(line_surface, color, (x1, y1), (x2, y2))
        surface.blit(line_surface, (0, 0))


class GroundPlane:
    def __init__(self, perspective: Perspective) -> None:
        self.perspective = perspective
        self.flow = 0.0

    def update(self, dt_ms: int, speed: float) -> None:
        self.flow = (self.flow + speed * (dt_ms / 1000.0) * 0.0024) % 1.0

    def draw(self, surface: pygame.Surface) -> None:
        ground = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        left_horizon = self.perspective.lane_x(0, HORIZON_Y) - 240
        right_horizon = self.perspective.lane_x(2, HORIZON_Y) + 240
        base_polygon = [
            (0, HEIGHT),
            (WIDTH, HEIGHT),
            (right_horizon, HORIZON_Y),
            (left_horizon, HORIZON_Y),
        ]
        pygame.draw.polygon(ground, (16, 34, 68, 200), base_polygon)

        depth = -self.flow
        stripe_spacing = 0.12
        while depth < 1.15:
            y1_depth = max(0.0, depth)
            y2_depth = depth + stripe_spacing
            if y2_depth <= 0.0:
                depth += stripe_spacing
                continue
            if y1_depth > 1.0:
                break
            y1 = HORIZON_Y + (GROUND_Y - HORIZON_Y) * y1_depth
            y2 = HORIZON_Y + (GROUND_Y - HORIZON_Y) * min(1.0, y2_depth)
            width1 = 24 + 240 * y1_depth
            width2 = 24 + 240 * min(1.0, y2_depth)
            left1 = self.perspective.lane_x(1, y1) - width1
            right1 = self.perspective.lane_x(1, y1) + width1
            left2 = self.perspective.lane_x(1, y2) - width2
            right2 = self.perspective.lane_x(1, y2) + width2
            alpha = max(0, min(200, int(200 * (1.0 - y1_depth * 0.82))))
            color = (180, 220, 255, alpha)
            pygame.draw.polygon(ground, color, [(left1, y1), (right1, y1), (right2, y2), (left2, y2)])
            depth += stripe_spacing

        edge_color = (36, 80, 140, 130)
        edge_width = 180
        pygame.draw.polygon(
            ground,
            edge_color,
            [
                (self.perspective.lane_x(0, GROUND_Y) - edge_width, GROUND_Y),
                (self.perspective.lane_x(0, HORIZON_Y) - 140, HORIZON_Y),
                (left_horizon, HORIZON_Y),
                (0, HEIGHT),
            ],
        )
        pygame.draw.polygon(
            ground,
            edge_color,
            [
                (WIDTH, HEIGHT),
                (right_horizon, HORIZON_Y),
                (self.perspective.lane_x(2, HORIZON_Y) + 140, HORIZON_Y),
                (self.perspective.lane_x(2, GROUND_Y) + edge_width, GROUND_Y),
            ],
        )

        surface.blit(ground, (0, 0))


class CityLayer:
    def __init__(self, *, offset: float, speed_scale: float, color: Tuple[int, int, int]) -> None:
        self.offset = offset
        self.speed_scale = speed_scale
        self.color = color
        self.structures: List[Tuple[float, float, float, float]] = []
        self._bootstrap()

    def _bootstrap(self) -> None:
        x = -WIDTH * 0.1
        while x < WIDTH * 1.2:
            width = random.uniform(60, 160)
            height = random.uniform(70, 190)
            taper = random.uniform(0.0, 14.0)
            self.structures.append((x, width, height, taper))
            x += width * 0.7

    def update(self, dt_ms: int, base_speed: float) -> None:
        dt = dt_ms / 1000.0
        travel = base_speed * self.speed_scale * dt
        updated: List[Tuple[float, float, float, float]] = []
        for x, width, height, taper in self.structures:
            x -= travel
            if x + width < -160:
                continue
            updated.append((x, width, height, taper))
        self.structures = updated
        while self.structures and self.structures[-1][0] + self.structures[-1][1] < WIDTH * 1.2:
            width = random.uniform(60, 180)
            height = random.uniform(90, 220)
            taper = random.uniform(0.0, 16.0)
            last_x = self.structures[-1][0] + self.structures[-1][1]
            self.structures.append((last_x + random.uniform(24, 80), width, height, taper))
        if not self.structures:
            self._bootstrap()

    def draw(self, surface: pygame.Surface) -> None:
        skyline = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        base_y = HORIZON_Y + self.offset
        for x, width, height, taper in self.structures:
            rect = pygame.Rect(int(x), int(base_y - height), int(width), int(height))
            pygame.draw.rect(skyline, (*self.color, 180), rect)
            if taper > 1.0:
                top = pygame.Surface((int(width), int(taper)), pygame.SRCALPHA)
                pygame.draw.polygon(
                    top,
                    (min(255, self.color[0] + 20), min(255, self.color[1] + 20), min(255, self.color[2] + 20), 160),
                    [(0, int(taper)), (int(width / 2), 0), (int(width), int(taper))],
                )
                skyline.blit(top, (int(x), int(base_y - height - taper)))
        surface.blit(skyline, (0, 0))


@dataclass
class AmbientPlane:
    x: float
    y: float
    speed: float
    scale: float
    phase: float = field(default_factory=lambda: random.uniform(0.0, math.tau))

    def update(self, dt_ms: int) -> bool:
        dt = dt_ms / 1000.0
        self.phase += 0.8 * dt
        self.x -= self.speed * dt
        self.y += math.sin(self.phase) * 8.0 * dt
        return self.x > -160

    def draw(self, surface: pygame.Surface) -> None:
        base = pygame.Surface((140, 64), pygame.SRCALPHA)
        pygame.draw.polygon(base, (200, 230, 255), [(12, 32), (68, 18), (128, 32), (68, 44)])
        pygame.draw.polygon(base, (140, 190, 255), [(36, 32), (74, 24), (112, 32), (74, 40)])
        pygame.draw.polygon(base, (255, 140, 110), [(42, 36), (64, 8), (54, 36)])
        pygame.draw.circle(base, (255, 255, 255), (104, 30), 6)
        size = (max(2, int(base.get_width() * self.scale)), max(2, int(base.get_height() * self.scale)))
        scaled = pygame.transform.smoothscale(base, size)
        rotated = pygame.transform.rotozoom(scaled, math.sin(self.phase) * 6.0, 1.0)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, rect)


@dataclass
class LaneActor:
    lane: int
    pos_y: float
    base_image: pygame.Surface
    wobble_speed: float = 0.0
    wobble_magnitude: float = 0.0
    render_image: pygame.Surface = field(init=False)
    render_rect: pygame.Rect = field(init=False)
    wobble_phase: float = field(default_factory=lambda: random.uniform(0, math.tau))

    def update(self, dt_ms: int, speed: float, perspective: Perspective) -> None:
        self.pos_y += speed * (dt_ms / 1000.0)
        if self.wobble_speed:
            self.wobble_phase += self.wobble_speed * (dt_ms / 1000.0)
        self.update_geometry(perspective)

    def update_geometry(self, perspective: Perspective) -> None:
        y = self.pos_y + math.sin(self.wobble_phase) * self.wobble_magnitude
        x = perspective.lane_x(self.lane, y)
        scale = perspective.scale(y)
        width = max(2, int(self.base_image.get_width() * scale))
        height = max(2, int(self.base_image.get_height() * scale))
        self.render_image = pygame.transform.smoothscale(self.base_image, (width, height))
        self.render_rect = self.render_image.get_rect(center=(x, y))

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.render_image, self.render_rect)

    def offscreen(self) -> bool:
        return self.render_rect.top > HEIGHT + 60


class SkyRing(LaneActor):
    def __init__(self, lane: int, pos_y: float) -> None:
        base = pygame.Surface((120, 120), pygame.SRCALPHA)
        pygame.draw.circle(base, (255, 205, 84), (60, 60), 58)
        pygame.draw.circle(base, (255, 245, 200), (60, 60), 36)
        pygame.draw.circle(base, (40, 120, 255), (60, 60), 28, 6)
        super().__init__(lane, pos_y, base, wobble_speed=2.6, wobble_magnitude=6)


class SkyDrone(LaneActor):
    def __init__(self, lane: int, pos_y: float) -> None:
        base = pygame.Surface((140, 96), pygame.SRCALPHA)
        pygame.draw.ellipse(base, (38, 46, 68), (10, 36, 120, 42))
        pygame.draw.ellipse(base, (80, 120, 180), (10, 26, 120, 52))
        pygame.draw.rect(base, (24, 28, 40), (58, 24, 24, 48))
        pygame.draw.circle(base, (255, 110, 96), (28, 52), 12)
        pygame.draw.circle(base, (255, 180, 80), (112, 52), 12)
        pygame.draw.line(base, (160, 200, 255), (0, 22), (40, 8), 6)
        pygame.draw.line(base, (160, 200, 255), (140, 22), (100, 8), 6)
        super().__init__(lane, pos_y, base, wobble_speed=1.8, wobble_magnitude=10)


class TurbulencePod(LaneActor):
    def __init__(self, lane: int, pos_y: float) -> None:
        base = pygame.Surface((160, 120), pygame.SRCALPHA)
        pygame.draw.ellipse(base, (210, 228, 255), (0, 20, 160, 80))
        pygame.draw.ellipse(base, (120, 168, 255), (0, 30, 160, 70), 4)
        pygame.draw.circle(base, (255, 255, 255, 180), (80, 60), 40, 4)
        pygame.draw.circle(base, (108, 176, 255, 120), (80, 60), 56, 4)
        super().__init__(lane, pos_y, base, wobble_speed=2.1, wobble_magnitude=14)


class SkyFreighter(LaneActor):
    def __init__(self, lane: int, pos_y: float) -> None:
        base = pygame.Surface((220, 160), pygame.SRCALPHA)
        pygame.draw.polygon(base, (180, 208, 230), [(30, 88), (180, 48), (210, 60), (180, 108), (30, 96)])
        pygame.draw.polygon(base, (120, 160, 210), [(30, 88), (180, 48), (210, 60), (180, 108), (30, 96)], 6)
        pygame.draw.ellipse(base, (220, 244, 255), (96, 44, 80, 48))
        pygame.draw.rect(base, (255, 140, 110), (18, 86, 24, 18))
        pygame.draw.polygon(base, (255, 220, 180), [(42, 82), (70, 52), (84, 86)])
        pygame.draw.polygon(base, (255, 220, 180), [(42, 102), (70, 132), (84, 100)])
        pygame.draw.ellipse(base, (20, 28, 46, 180), (24, 88, 180, 18))
        super().__init__(lane, pos_y, base, wobble_speed=1.2, wobble_magnitude=6)


class Plane:
    def __init__(self, perspective: Perspective) -> None:
        self.perspective = perspective
        self.base_image = self._build_sprite()
        self.lane_index = 1
        self.base_y = GROUND_Y + 54
        self.altitude = 0.0
        self.velocity = 0.0
        self.invulnerable_until = 0
        self.roll = 0.0
        self.target_roll = 0.0
        self.shadow_scale = 1.2
        self.trail: List[TrailParticle] = []
        self.trail_timer = 0.0
        self.render_image = self.base_image
        self.render_rect = self.base_image.get_rect()
        self.shadow_rect = pygame.Rect(0, 0, 0, 0)
        self.glow_phase = 0.0

    def _build_sprite(self) -> pygame.Surface:
        surface = pygame.Surface((220, 140), pygame.SRCALPHA)
        fuselage = [(24, 68), (142, 34), (204, 52), (142, 102), (24, 72)]
        pygame.draw.polygon(surface, (188, 228, 255), fuselage)
        pygame.draw.polygon(surface, (86, 134, 216), fuselage, 6)
        nose = pygame.Surface((120, 76), pygame.SRCALPHA)
        pygame.draw.ellipse(nose, (210, 240, 255), (0, 0, 120, 76))
        pygame.draw.ellipse(nose, (120, 170, 240), (12, 12, 96, 48))
        surface.blit(nose, (82, 32))
        canopy = pygame.Surface((90, 54), pygame.SRCALPHA)
        pygame.draw.ellipse(canopy, (36, 64, 120), (0, 0, 90, 54))
        pygame.draw.ellipse(canopy, (130, 200, 255, 220), (10, 6, 70, 36))
        pygame.draw.ellipse(canopy, (255, 255, 255, 150), (18, 10, 42, 22))
        surface.blit(canopy, (70, 38))
        tail = pygame.Surface((82, 80), pygame.SRCALPHA)
        pygame.draw.polygon(tail, (72, 122, 210), [(0, 60), (64, 12), (64, 108)])
        pygame.draw.polygon(tail, (180, 220, 255), [(8, 60), (48, 34), (48, 86)])
        surface.blit(tail, (18, 26))
        wing = pygame.Surface((200, 120), pygame.SRCALPHA)
        pygame.draw.polygon(wing, (255, 112, 96), [(24, 72), (110, 16), (160, 24), (116, 72), (160, 120), (110, 128), (24, 84)])
        pygame.draw.polygon(wing, (255, 190, 160), [(28, 72), (110, 28), (150, 34), (112, 74), (150, 118), (110, 122), (28, 82)], 4)
        pygame.draw.polygon(wing, (255, 220, 180, 90), [(40, 74), (110, 36), (142, 42), (110, 74), (142, 112), (110, 118), (40, 82)])
        surface.blit(wing, (0, 0))
        stabilizer = pygame.Surface((120, 70), pygame.SRCALPHA)
        pygame.draw.polygon(stabilizer, (240, 180, 120), [(8, 36), (94, 22), (112, 36), (94, 48), (8, 42)])
        pygame.draw.polygon(stabilizer, (255, 220, 180), [(14, 36), (90, 28), (102, 36), (90, 44), (14, 40)], 4)
        surface.blit(stabilizer, (36, 76))
        jet = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(jet, (255, 200, 160), (30, 30), 14)
        pygame.draw.circle(jet, (255, 120, 80), (30, 30), 8)
        pygame.draw.circle(jet, (255, 240, 220, 120), (30, 30), 20, 4)
        surface.blit(jet, (18, 50))
        return surface

    def reset(self, time_now: int) -> None:
        self.lane_index = 1
        self.altitude = 0.0
        self.velocity = 0.0
        self.invulnerable_until = time_now
        self.roll = 0.0
        self.target_roll = 0.0
        self.trail.clear()

    def switch_lane(self, direction: int) -> None:
        new_lane = max(0, min(2, self.lane_index + direction))
        if new_lane != self.lane_index:
            self.lane_index = new_lane
            self.target_roll = direction * 18.0

    def lift(self) -> None:
        if self.altitude < PLANE_ALTITUDE_LIMIT * 0.92:
            self.velocity = PLANE_LIFT

    def apply_gravity(self, frame_scale: float) -> None:
        self.velocity = min(self.velocity + GRAVITY * frame_scale, MAX_DESCENT)
        self.altitude = max(0.0, min(PLANE_ALTITUDE_LIMIT, self.altitude + self.velocity * frame_scale))

    def update(self, dt_ms: int, time_now: int) -> None:
        frame_scale = dt_ms / 16.0
        self.apply_gravity(frame_scale)
        self.roll += (self.target_roll - self.roll) * min(1.0, 0.16 * frame_scale)
        self.target_roll *= 0.7
        self.perspective.update_camera(dt_ms, self.lane_index, self.roll)
        self.trail_timer += dt_ms
        self.glow_phase = (self.glow_phase + dt_ms / 320.0) % math.tau
        if self.trail_timer > 45:
            self.trail_timer = 0
            tail_x, tail_y = self._tail_world()
            drift = random.uniform(-12, 12)
            particle = TrailParticle(tail_x, tail_y, 420.0, drift, random.uniform(0.6, 0.9), 420.0)
            self.trail.append(particle)
        for particle in list(self.trail):
            particle.update(dt_ms)
            if particle.life <= 0:
                self.trail.remove(particle)
        self.update_geometry(time_now)

    def update_geometry(self, time_now: int) -> None:
        y = self.base_y - self.altitude
        x = self.perspective.lane_x(self.lane_index, y)
        scale = self.perspective.scale(y, bias=1.18)
        image = pygame.transform.rotozoom(self.base_image, self.roll, scale)
        if time_now < self.invulnerable_until:
            image.set_alpha(160 + int(80 * math.sin(time_now / 60)))
        else:
            image.set_alpha(255)
        self.render_image = image
        self.render_rect = image.get_rect(center=(x, y))
        shadow_width = int(120 * scale * self.shadow_scale)
        shadow_height = int(36 * scale * self.shadow_scale)
        self.shadow_rect = pygame.Rect(0, 0, shadow_width, shadow_height)
        self.shadow_rect.center = (x, self.base_y + 30)

    def _tail_world(self) -> Tuple[float, float]:
        return float(self.render_rect.centerx - self.render_rect.width * 0.36), float(self.render_rect.centery + self.render_rect.height * 0.08)

    def draw(self, surface: pygame.Surface) -> None:
        # Draw shadow
        shadow_surface = pygame.Surface(self.shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, (20, 34, 54, 140), shadow_surface.get_rect())
        surface.blit(shadow_surface, self.shadow_rect.topleft)
        # Draw jet trail
        trail_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for particle in self.trail:
            particle.draw(trail_surface)
        surface.blit(trail_surface, (0, 0), special_flags=pygame.BLEND_ADD)
        # Engine glow
        glow_surface = pygame.Surface((160, 160), pygame.SRCALPHA)
        pulse = 160 + int(60 * math.sin(self.glow_phase))
        pygame.draw.circle(glow_surface, (255, 160, 120, pulse), (80, 120), 36)
        pygame.draw.circle(glow_surface, (255, 220, 180, pulse // 2), (80, 120), 58, 6)
        glow_rect = glow_surface.get_rect(center=(int(self.render_rect.centerx - self.render_rect.width * 0.44), int(self.render_rect.centery + self.render_rect.height * 0.12)))
        surface.blit(glow_surface, glow_rect, special_flags=pygame.BLEND_ADD)
        # Draw plane
        surface.blit(self.render_image, self.render_rect)

    def is_invulnerable(self, time_now: int) -> bool:
        return time_now < self.invulnerable_until


def render_hud(
    surface: pygame.Surface,
    score: int,
    best: int,
    combo: int,
    time_now: int,
    plane: Plane,
    air_speed: float,
) -> None:
    font_big = pygame.font.Font(None, 48)
    font_small = pygame.font.Font(None, 30)
    hud_surface = pygame.Surface((WIDTH, 110), pygame.SRCALPHA)
    pygame.draw.rect(hud_surface, (6, 20, 40, 180), (20, 16, WIDTH - 40, 76), border_radius=24)
    text_score = font_big.render(f"Score {score}", True, HUD_COLOR)
    text_best = font_small.render(f"Best {best}", True, HUD_COLOR)
    hud_surface.blit(text_score, (40, 30))
    hud_surface.blit(text_best, (WIDTH - 200, 40))
    altitude_ratio = plane.altitude / PLANE_ALTITUDE_LIMIT
    altitude_text = font_small.render(f"Altitude {int(altitude_ratio * 100):02d}%", True, (180, 220, 255))
    speed_text = font_small.render(f"IAS {int(air_speed):03d} kt", True, (180, 220, 255))
    hud_surface.blit(speed_text, (40, 66))
    hud_surface.blit(altitude_text, (WIDTH - 220, 66))
    if combo > 1:
        combo_text = font_small.render(f"x{combo} Air Combo!", True, (255, 210, 120))
        hud_surface.blit(combo_text, (WIDTH // 2 - combo_text.get_width() // 2, 66))
    if plane.is_invulnerable(time_now):
        invuln_text = font_small.render("Aegis Shield", True, (130, 220, 255))
        hud_surface.blit(invuln_text, (WIDTH - 220, 82))
    surface.blit(hud_surface, (0, 0))


def draw_lane_guides(surface: pygame.Surface, perspective: Perspective) -> None:
    lane_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    segments = 14
    for lane in range(3):
        for step in range(segments):
            t1 = step / segments
            t2 = (step + 1) / segments
            y1 = HORIZON_Y + (GROUND_Y - HORIZON_Y) * t1
            y2 = HORIZON_Y + (GROUND_Y - HORIZON_Y) * t2
            x1 = perspective.lane_x(lane, y1)
            x2 = perspective.lane_x(lane, y2)
            intensity = int(120 + 110 * t1)
            color = (*LANE_GLOW[:3], intensity)
            width = max(2, int(3 + 6 * t2))
            pygame.draw.line(lane_surface, color, (x1, y1), (x2, y2), width)
    surface.blit(lane_surface, (0, 0), special_flags=pygame.BLEND_ADD)

def main(
    *,
    max_frames: Optional[int] = None,
    screenshot_path: Optional[str] = None,
    difficulty: str = "classic",
    seed: Optional[int] = None,
) -> None:
    pygame.init()
    pygame.display.set_caption("Plane Jumpers")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    perspective = Perspective()

    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"Unknown difficulty '{difficulty}'")

    if seed is not None:
        random.seed(seed)

    preset = DIFFICULTY_PRESETS[difficulty]
    obstacle_interval = _scale_interval(OBSTACLE_INTERVAL, preset["obstacle_factor"])
    ring_interval = _scale_interval(RING_INTERVAL, preset["ring_factor"])
    speed_increment = preset["increment"]

    backdrop = SkyBackdrop()
    city_far = CityLayer(offset=6, speed_scale=0.14, color=(24, 58, 112))
    city_near = CityLayer(offset=28, speed_scale=0.28, color=(38, 88, 162))
    ground = GroundPlane(perspective)
    plane = Plane(perspective)
    rings: List[SkyRing] = []
    obstacles: List[LaneActor] = []
    speed_lines: List[SpeedLine] = []
    ambient_planes: List[AmbientPlane] = []

    score = 0
    best = 0
    combo = 0
    combo_timer = 0
    base_speed = preset["speed"]
    running = True
    game_over = False

    time_now = pygame.time.get_ticks()
    next_obstacle = time_now + random.randint(*obstacle_interval)
    next_ring = time_now + random.randint(*ring_interval)

    frame_count = 0
    screenshot_saved = False

    while running:
        dt_ms = clock.tick(FPS)
        time_now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif game_over and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    # Restart
                    plane.reset(time_now)
                    rings.clear()
                    obstacles.clear()
                    speed_lines.clear()
                    score = 0
                    combo = 0
                    combo_timer = 0
                    base_speed = preset["speed"]
                    game_over = False
                    next_obstacle = time_now + random.randint(*obstacle_interval)
                    next_ring = time_now + random.randint(*ring_interval)

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
        if not game_over and time_now >= next_obstacle:
            actor_cls = random.choices(
                [SkyDrone, TurbulencePod, SkyFreighter],
                weights=(0.44, 0.32, 0.24),
                k=1,
            )[0]
            spawn = actor_cls(random.randrange(3), HORIZON_Y - 140)
            spawn.update_geometry(perspective)
            obstacles.append(spawn)
            next_obstacle = time_now + random.randint(*obstacle_interval)
        if not game_over and time_now >= next_ring:
            ring = SkyRing(random.randrange(3), HORIZON_Y - 200)
            ring.update_geometry(perspective)
            rings.append(ring)
            next_ring = time_now + random.randint(*ring_interval)

        # Update game state
        backdrop.update(dt_ms)
        scene_speed = base_speed if not game_over else base_speed * 0.18
        city_far.update(dt_ms, scene_speed)
        city_near.update(dt_ms, scene_speed)
        ground.update(dt_ms, scene_speed)
        if not game_over:
            base_speed += speed_increment * (dt_ms / 1000.0)
            plane.update(dt_ms, time_now)
            for actor in obstacles:
                actor.update(dt_ms, base_speed, perspective)
            for ring in rings:
                ring.update(dt_ms, base_speed * 0.92, perspective)
            if random.random() < SPEEDLINE_SPAWN:
                speed_lines.append(SpeedLine(random.randrange(3), HORIZON_Y + 40, random.uniform(60, 160), 1.0))
            for line in list(speed_lines):
                line.update(dt_ms, base_speed)
                if line.y > HEIGHT or line.opacity <= 0:
                    speed_lines.remove(line)

            if random.random() < 0.006:
                spawn_y = random.uniform(80, HORIZON_Y - 20)
                ambient_planes.append(
                    AmbientPlane(
                        x=WIDTH + 120,
                        y=spawn_y,
                        speed=random.uniform(40.0, 110.0),
                        scale=random.uniform(0.4, 0.8),
                    )
                )

            # Cleanup actors
            obstacles[:] = [actor for actor in obstacles if not actor.offscreen()]
            rings[:] = [ring for ring in rings if not ring.offscreen()]

            # Scoring & collisions
            collected = 0
            for ring in list(rings):
                if ring.render_rect.colliderect(plane.render_rect):
                    rings.remove(ring)
                    collected += 1
            if collected:
                combo = min(9, combo + collected)
                combo_timer = 1400
                score += 150 * combo
                plane.invulnerable_until = time_now + INVULN_TIME
            else:
                combo_timer = max(0, combo_timer - dt_ms)
                if combo_timer == 0:
                    combo = 0

            if not plane.is_invulnerable(time_now):
                for obstacle in obstacles:
                    if obstacle.render_rect.colliderect(plane.render_rect.inflate(-40, -30)):
                        game_over = True
                        best = max(best, score)
                        break
            score += int(base_speed * (dt_ms / 1200.0))
        else:
            plane.update(dt_ms, time_now)

        for bg_plane in list(ambient_planes):
            if not bg_plane.update(dt_ms):
                ambient_planes.remove(bg_plane)

        # Drawing
        backdrop.draw(screen)
        city_far.draw(screen)
        city_near.draw(screen)
        ground.draw(screen)
        draw_lane_guides(screen, perspective)
        for bg_plane in ambient_planes:
            bg_plane.draw(screen)
        for line in speed_lines:
            line.draw(screen, perspective)
        for ring in rings:
            ring.draw(screen)
        for obstacle in obstacles:
            obstacle.draw(screen)
        plane.draw(screen)
        render_hud(
            screen,
            score,
            best,
            combo if combo_timer else 0,
            time_now,
            plane,
            base_speed * 0.92,
        )

        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((6, 12, 28, 180))
            screen.blit(overlay, (0, 0))
            font_big = pygame.font.Font(None, 84)
            font_small = pygame.font.Font(None, 42)
            text = font_big.render("Flight Over", True, WARNING_COLOR)
            rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
            screen.blit(text, rect)
            text_score = font_small.render(f"Final Score {score}", True, HUD_COLOR)
            rect_score = text_score.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30))
            screen.blit(text_score, rect_score)
            restart = font_small.render("Press ENTER to relaunch", True, (180, 210, 255))
            rect_restart = restart.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 36))
            screen.blit(restart, rect_restart)

        pygame.display.flip()

        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            if screenshot_path and not screenshot_saved:
                _save_screenshot(screen, screenshot_path)
                screenshot_saved = True
            break

    if screenshot_path and not screenshot_saved:
        _save_screenshot(screen, screenshot_path)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Plane Jumpers pseudo-3D endless runner.")
    parser.add_argument("--frames", type=int, default=None,
                        help="Run the game loop for a limited number of frames (useful for testing).")
    parser.add_argument("--screenshot", type=str, default=None,
                        help="Path to save a screenshot of the final frame (works with --headless).")
    parser.add_argument("--headless", action="store_true",
                        help="Use the SDL dummy video driver to render without opening a window.")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTY_PRESETS.keys()), default="classic",
                        help="Preset pacing: relaxed, classic, or ace.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed Python's RNG for deterministic obstacle and ring layouts.")

    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    main(
        max_frames=args.frames,
        screenshot_path=args.screenshot,
        difficulty=args.difficulty,
        seed=args.seed,
    )
