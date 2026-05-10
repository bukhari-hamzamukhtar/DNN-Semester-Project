import math
import random
import json
import time

CURRICULUM_STAGES = [
    {"name": "slow_singles",  "num_m": (1, 2), "speed": (2.0, 4.0), "hot_prob": 0.05, "turn_rate": 0.06},
    {"name": "fast_homing",   "num_m": (1, 3), "speed": (4.0, 6.0), "hot_prob": 0.15, "turn_rate": 0.10},
    {"name": "multi_threat",  "num_m": (2, 4), "speed": (4.5, 7.0), "hot_prob": 0.25, "turn_rate": 0.12},
    {"name": "cluster_swarm", "num_m": (3, 6), "speed": (5.0, 8.0), "hot_prob": 0.40, "turn_rate": 0.15},
    {"name": "adversarial",   "num_m": (4, 8), "speed": (5.5, 9.0), "hot_prob": 0.60, "turn_rate": 0.18},
]
NUM_SIMULATIONS  = 2000   # per stage
FRAMES_PER_SIM   = 300
LOG_INTERVAL     = 5
WIDTH, HEIGHT    = 800, 600
LOOKAHEAD_FRAMES = 60  # 1 full second of lookahead

JET_RADIUS = 11
MISSILE_RADIUS = 3
FLARE_RADIUS = 6
HIT_DISTANCE = JET_RADIUS + MISSILE_RADIUS

MANEUVER_NAMES = ["NORMAL", "BARREL_ROLL", "JINKING", "FALLING_LEAF", "COBRA", "IMMELMANN"]
MANEUVER_IDX   = {name: i for i, name in enumerate(MANEUVER_NAMES)}

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def oracle_maneuver(jet_pos, missiles, arena_w=800, arena_h=600, margin=45):
    if not missiles:
        return "NORMAL"
    jx, jy = jet_pos
    near = [(m, math.hypot(m["pos"][0]-jx, m["pos"][1]-jy)) for m in missiles]
    if min(jx, arena_w-jx, jy, arena_h-jy) < margin:
        return "IMMELMANN"
    close_80  = [m for m,d in near if d < 80]
    close_150 = [m for m,d in near if d < 150]
    close_200 = [m for m,d in near if d < 200]
    if len(close_80) == 1:
        m = close_80[0]
        dx, dy = m["pos"][0]-jx, m["pos"][1]-jy
        if m["vel"][0]*dx + m["vel"][1]*dy < 0:
            return "COBRA"
    if len(close_150) >= 2:
        return "FALLING_LEAF"
    if len(missiles) >= 3 and len(close_200) >= 1:
        return "JINKING"
    if len(close_200) == 1 and len(missiles) == 1 and min(jx, arena_w-jx) > 160 and min(jy, arena_h-jy) > 160:
        return "BARREL_ROLL"
    return "NORMAL"

def generate_headless_data(stage_cfg, output_file="data.jsonl"):
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

        # --- MISSILES: 30% of sims are "hot" scenarios with close spawns ---
        num_missiles = random.randint(*stage_cfg["num_m"])
        is_hot_scenario = (random.random() < stage_cfg["hot_prob"])
        missiles = []  
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
                "speed": random.uniform(*stage_cfg["speed"]),
                "turn_rate": stage_cfg["turn_rate"],  # NEW — used in homing update
                "type": m_type,
                "dead": False
            })

        flares = []
        next_flare_frame = random.randint(120, 240)
        frame_history = []
        collision_history = []

        for frame in range(FRAMES_PER_SIM):
            # --- JET UPDATE: 50% random, 50% flee-closest-missile ---
            alive_missiles = [m for m in missiles if not m["dead"]]
            if alive_missiles and random.random() < 0.50:
                # Simple flee: steer away from closest missile
                closest = min(alive_missiles, key=lambda m: math.hypot(m["x"]-jet["x"], m["y"]-jet["y"]))
                flee_angle = math.atan2(jet["y"]-closest["y"], jet["x"]-closest["x"])
                diff = normalize_angle(flee_angle - jet["heading"])
                jet["heading"] += clamp(diff, -0.18, 0.18)
            else:
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
                    m["heading"] += max(-m["turn_rate"], min(m["turn_rate"], diff))  # was hardcoded 0.15

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
            record["maneuver"] = MANEUVER_IDX[oracle_maneuver(
                record["jet"]["pos"], record["missiles"]
            )]
            if len(record["missiles"]) > 0:
                dataset.append(record)

    label_1 = sum(1 for r in dataset if r["label"] == 1)
    label_0 = sum(1 for r in dataset if r["label"] == 0)
    print(f"\nDone in {round(time.time() - start_time, 2)}s")
    print(f"Total frames: {len(dataset)}")
    print(f"Label 0 (safe):   {label_0} ({100*label_0/len(dataset):.1f}%)")
    print(f"Label 1 (danger): {label_1} ({100*label_1/len(dataset):.1f}%)")
    print(f"Effective pos_weight: {label_0/max(label_1,1):.2f}x")

    with open(output_file, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"Exported to {output_file}")

if __name__ == "__main__":
    for stage in CURRICULUM_STAGES:
        print(f"\n=== {stage['name']} ===")
        generate_headless_data(stage, output_file=f"data_{stage['name']}.jsonl")