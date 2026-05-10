import math
import random
import json
import time

NUM_SIMULATIONS = 8000
FRAMES_PER_SIM = 300
LOG_INTERVAL = 5
LOOKAHEAD_FRAMES = 60  # 1 full second of lookahead

WIDTH, HEIGHT = 800, 600
JET_RADIUS = 11
MISSILE_RADIUS = 3
FLARE_RADIUS = 6
HIT_DISTANCE = JET_RADIUS + MISSILE_RADIUS

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def generate_headless_data():
    print(f"Spinning up {NUM_SIMULATIONS} headless rounds...")
    start_time = time.time()
    dataset = []

    for sim_id in range(NUM_SIMULATIONS):
        if sim_id % 1000 == 0 and sim_id > 0:
            print(f"  {sim_id} sims done...")

        # --- JET INIT (constrained to center zone) ---
        jet = {
            "x": random.uniform(200, WIDTH - 200),
            "y": random.uniform(150, HEIGHT - 150),
            "vx": 0.0, "vy": 0.0,
            "heading": random.uniform(0, 2 * math.pi),
            "speed": 5.0,
            "drag": 0.92
        }

        # MISSILES: 30% of sims are "hot" scenarios with close spawns
        num_missiles = random.randint(1, 5)
        missiles = []
        is_hot_scenario = (random.random() < 0.30)

        for _ in range(num_missiles):
            # Strongly prefer homing (75%) — ballistic almost never hits
            m_type = 0 if random.random() < 0.75 else 1

            if is_hot_scenario:
                # Spawn close, guaranteed pressure situation
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(80, 180)
                mx = clamp(jet["x"] + math.cos(angle) * dist, 20, WIDTH - 20)
                my = clamp(jet["y"] + math.sin(angle) * dist, 20, HEIGHT - 20)
            else:
                # Normal: spawn from edges
                mx = random.choice([
                    random.uniform(0, 50),
                    random.uniform(WIDTH - 50, WIDTH)
                ])
                my = random.choice([
                    random.uniform(0, 50),
                    random.uniform(HEIGHT - 50, HEIGHT)
                ])

            missiles.append({
                "x": mx, "y": my,
                "vx": 0.0, "vy": 0.0,
                "heading": math.atan2(jet["y"] - my, jet["x"] - mx),
                # Hot scenarios: faster missiles
                "speed": random.uniform(5.0, 8.0) if is_hot_scenario else random.uniform(3.5, 7.0),
                "type": m_type,
                "dead": False
            })

        flares = []
        next_flare_frame = random.randint(120, 240)
        frame_history = []
        collision_history = []

        for frame in range(FRAMES_PER_SIM):
            # JET UPDATE with wall bounce (keeps it in the arena)
            jet["heading"] += random.uniform(-0.1, 0.1)
            thrust_x = math.cos(jet["heading"]) * jet["speed"]
            thrust_y = math.sin(jet["heading"]) * jet["speed"]
            jet["vx"] = jet["vx"] * jet["drag"] + thrust_x * 0.1
            jet["vy"] = jet["vy"] * jet["drag"] + thrust_y * 0.1
            jet["x"] += jet["vx"]
            jet["y"] += jet["vy"]

            # Soft wall bounce — flip velocity, push back inside
            if jet["x"] < 30:
                jet["x"] = 30
                jet["vx"] = abs(jet["vx"])
                jet["heading"] = normalize_angle(-jet["heading"] + math.pi)
            elif jet["x"] > WIDTH - 30:
                jet["x"] = WIDTH - 30
                jet["vx"] = -abs(jet["vx"])
                jet["heading"] = normalize_angle(-jet["heading"] + math.pi)
            if jet["y"] < 30:
                jet["y"] = 30
                jet["vy"] = abs(jet["vy"])
                jet["heading"] = normalize_angle(-jet["heading"])
            elif jet["y"] > HEIGHT - 30:
                jet["y"] = HEIGHT - 30
                jet["vy"] = -abs(jet["vy"])
                jet["heading"] = normalize_angle(-jet["heading"])

            # --- FLARES ---
            if frame == next_flare_frame:
                start_angle = jet["heading"] + math.radians(45)
                angle_step = math.radians(270) / 7
                speed = jet["speed"] + 2.0
                for i in range(8):
                    angle = start_angle + (i * angle_step)
                    flares.append({
                        "x": jet["x"], "y": jet["y"],
                        "vx": math.cos(angle) * speed,
                        "vy": math.sin(angle) * speed,
                        "drag": 0.85,
                        "life": 120
                    })
                next_flare_frame += random.randint(120, 240)

            active_flares = []
            for f in flares:
                f["vx"] *= f["drag"]
                f["vy"] *= f["drag"]
                f["x"] += f["vx"]
                f["y"] += f["vy"]
                f["life"] -= 1
                if f["life"] > 0:
                    active_flares.append(f)
            flares = active_flares

            # --- MISSILES ---
            hit_this_frame = False
            for m in missiles:
                if m["dead"]:
                    continue

                if m["type"] == 0:  # Homing — much tighter turn rate
                    dx, dy = jet["x"] - m["x"], jet["y"] - m["y"]
                    desired = math.atan2(dy, dx)
                    diff = normalize_angle(desired - m["heading"])
                    # 0.15 instead of 0.05 — actually dangerous now
                    m["heading"] += max(-0.15, min(0.15, diff))

                m["vx"] = math.cos(m["heading"]) * m["speed"]
                m["vy"] = math.sin(m["heading"]) * m["speed"]
                m["x"] += m["vx"]
                m["y"] += m["vy"]

                if m["x"] < 0 or m["x"] > WIDTH or m["y"] < 0 or m["y"] > HEIGHT:
                    m["dead"] = True
                    continue

                dist = math.hypot(jet["x"] - m["x"], jet["y"] - m["y"])
                if dist < HIT_DISTANCE:
                    hit_this_frame = True
                    m["dead"] = True
                    continue

                for f in flares:
                    if math.hypot(f["x"] - m["x"], f["y"] - m["y"]) < (MISSILE_RADIUS + FLARE_RADIUS):
                        m["dead"] = True
                        f["life"] = 0
                        break

            snapshot = {
                "jet": {
                    "pos": [round(jet["x"], 2), round(jet["y"], 2)],
                    "vel": [round(jet["vx"], 2), round(jet["vy"], 2)]
                },
                "missiles": [
                    {
                        "pos": [round(m["x"], 2), round(m["y"], 2)],
                        "vel": [round(m["vx"], 2), round(m["vy"], 2)],
                        "type": m["type"]
                    }
                    for m in missiles if not m["dead"]
                ],
                "flares": [
                    {
                        "pos": [round(f["x"], 2), round(f["y"], 2)],
                        "vel": [round(f["vx"], 2), round(f["vy"], 2)]
                    }
                    for f in flares
                ]
            }
            frame_history.append(snapshot)
            collision_history.append(hit_this_frame)

        # --- RETROACTIVE LABELING ---
        for t in range(0, FRAMES_PER_SIM - LOOKAHEAD_FRAMES, LOG_INTERVAL):
            future_hit = any(collision_history[t: t + LOOKAHEAD_FRAMES])
            record = frame_history[t]
            record["label"] = 1 if future_hit else 0
            if len(record["missiles"]) > 0:
                dataset.append(record)

    label_1 = sum(1 for r in dataset if r["label"] == 1)
    label_0 = sum(1 for r in dataset if r["label"] == 0)
    print(f"\nDone in {round(time.time() - start_time, 2)}s")
    print(f"Total frames: {len(dataset)}")
    print(f"Label 0 (safe):   {label_0} ({100*label_0/len(dataset):.1f}%)")
    print(f"Label 1 (danger): {label_1} ({100*label_1/len(dataset):.1f}%)")
    print(f"Effective pos_weight: {label_0/max(label_1,1):.2f}x")

    with open("data.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    print("Exported to data.jsonl")

if __name__ == "__main__":
    generate_headless_data()