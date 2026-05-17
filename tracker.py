import cv2
import mediapipe as mp
import random
import time
import os
import math

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.expanduser("~/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Done!")

# ── MediaPipe Setup ────────────────────────────────────────────────────────
BaseOptions       = mp.tasks.BaseOptions
HandLandmarker    = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# ── Game Constants ─────────────────────────────────────────────────────────
W, H         = 960, 540
PLAYER_R     = 22          # player circle radius
OBS_W        = 55          # obstacle width
OBS_MIN_H    = 40
OBS_MAX_H    = 160
OBS_SPEED    = 5           # starting speed
SPAWN_EVERY  = 1.2         # seconds between spawns
HIT_SHRINK   = 8           # how many px to shrink on near-miss warning

# Colours (BGR)
C_BG         = (15, 15, 30)
C_PLAYER     = (0, 230, 255)
C_PLAYER_HIT = (0, 80, 255)
C_OBS        = (60, 60, 220)
C_OBS_FAST   = (30, 30, 180)
C_TEXT       = (255, 255, 255)
C_SCORE      = (0, 230, 180)
C_OVER       = (50, 50, 255)
C_WARN       = (0, 140, 255)

# ── Helpers ────────────────────────────────────────────────────────────────
def draw_rounded_rect(img, x, y, w, h, r, color, alpha=1.0):
    overlay = img.copy()
    cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), color, -1)
    cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), color, -1)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    else:
        img[:] = overlay

def rect_circle_collision(rx, ry, rw, rh, cx, cy, cr):
    nearest_x = max(rx, min(cx, rx + rw))
    nearest_y = max(ry, min(cy, ry + rh))
    dx = cx - nearest_x
    dy = cy - nearest_y
    return math.hypot(dx, dy) < cr

# ── Game State ─────────────────────────────────────────────────────────────
class Obstacle:
    def __init__(self, speed):
        self.h    = random.randint(OBS_MIN_H, OBS_MAX_H)
        # gap in the obstacle — player passes through it
        self.gap  = random.randint(80, 180)
        self.y    = random.randint(0, H - self.h - self.gap)
        self.x    = W + OBS_W
        self.speed = speed
        self.passed = False

    def update(self):
        self.x -= self.speed

    def draw(self, frame):
        col = C_OBS_FAST if self.speed > 9 else C_OBS
        # top block
        draw_rounded_rect(frame, self.x, self.y, OBS_W, self.h, 6, col)
        # bottom block
        bot_y = self.y + self.h + self.gap
        bot_h = H - bot_y
        if bot_h > 0:
            draw_rounded_rect(frame, self.x, bot_y, OBS_W, bot_h, 6, col)

    def collides(self, px, py, pr):
        # top block
        if rect_circle_collision(self.x, self.y, OBS_W, self.h, px, py, pr):
            return True
        # bottom block
        bot_y = self.y + self.h + self.gap
        bot_h = H - bot_y
        if bot_h > 0 and rect_circle_collision(self.x, bot_y, OBS_W, bot_h, px, py, pr):
            return True
        return False

    def off_screen(self):
        return self.x + OBS_W < 0


def run_game():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    # Game vars
    score       = 0
    best        = 0
    lives       = 3
    obstacles   = []
    last_spawn  = time.time()
    start_time  = time.time()
    game_over   = False
    hit_flash   = 0          # frames to flash red
    px, py      = W // 4, H // 2   # player position

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (W, H))

            # ── Hand Detection ──────────────────────────────────────────
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            hand_detected = False
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                # Use index finger tip (landmark 8) to control player
                raw_x = int(lms[8].x * W)
                raw_y = int(lms[8].y * H)
                # Smooth movement
                px = int(px * 0.4 + raw_x * 0.6)
                py = int(py * 0.4 + raw_y * 0.6)
                hand_detected = True

                # Draw subtle hand dots
                for lm in lms:
                    lx, ly = int(lm.x * W), int(lm.y * H)
                    cv2.circle(frame, (lx, ly), 3, (100, 100, 100), -1)

            # ── Dark overlay on camera feed ─────────────────────────────
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), C_BG, -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            if not game_over:
                elapsed = time.time() - start_time
                speed   = OBS_SPEED + int(elapsed / 8)   # speed up every 8s

                # ── Spawn obstacles ─────────────────────────────────────
                if time.time() - last_spawn > SPAWN_EVERY:
                    obstacles.append(Obstacle(speed))
                    last_spawn = time.time()

                # ── Update & draw obstacles ─────────────────────────────
                for obs in obstacles[:]:
                    obs.update()
                    obs.draw(frame)
                    if obs.off_screen():
                        obstacles.remove(obs)
                        continue
                    if not obs.passed and obs.x + OBS_W < px - PLAYER_R:
                        obs.passed = True
                        score += 1
                    if obs.collides(px, py, PLAYER_R):
                        lives -= 1
                        hit_flash = 12
                        obstacles.remove(obs)
                        if lives <= 0:
                            game_over = True
                            best = max(best, score)

                # ── Draw player ─────────────────────────────────────────
                p_color = C_PLAYER_HIT if hit_flash > 0 else C_PLAYER
                if hit_flash > 0:
                    hit_flash -= 1
                cv2.circle(frame, (px, py), PLAYER_R + 4, (*p_color[:2], 60), -1)
                cv2.circle(frame, (px, py), PLAYER_R, p_color, -1)
                cv2.circle(frame, (px - 6, py - 6), 5, (255, 255, 255), -1)  # eye

                # ── HUD ─────────────────────────────────────────────────
                # Score
                cv2.putText(frame, f"Score: {score}", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, C_SCORE, 2)
                # Lives as hearts
                for i in range(lives):
                    hx = W - 50 - i * 38
                    cv2.putText(frame, "♥", (hx, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 80, 255), 2)
                # Speed indicator
                cv2.putText(frame, f"Speed x{speed}", (20, H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

                # No-hand warning
                if not hand_detected:
                    cv2.putText(frame, "Show your hand!", (W//2 - 130, H//2),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, C_WARN, 2)

            else:
                # ── Game Over Screen ────────────────────────────────────
                draw_rounded_rect(frame, W//2-220, H//2-120, 440, 240, 18,
                                  (20, 20, 50), alpha=0.85)
                cv2.putText(frame, "GAME OVER", (W//2 - 160, H//2 - 55),
                            cv2.FONT_HERSHEY_DUPLEX, 1.8, C_OVER, 3)
                cv2.putText(frame, f"Score : {score}", (W//2 - 100, H//2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, C_TEXT, 2)
                cv2.putText(frame, f"Best  : {best}", (W//2 - 100, H//2 + 45),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, C_SCORE, 2)
                cv2.putText(frame, "Press R to restart  |  Q to quit",
                            (W//2 - 210, H//2 + 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1)

            cv2.imshow("Hand Dodge Game", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r') and game_over:
                # Restart
                score      = 0
                lives      = 3
                obstacles  = []
                last_spawn = time.time()
                start_time = time.time()
                game_over  = False
                hit_flash  = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_game()