import pygame
import math
import random
from collections import deque
import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data
import threading
import queue

import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

class EdgeModel(nn.Module):
    def __init__(self, node_dim=64, edge_dim=32, msg_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, msg_dim), nn.ReLU(), nn.LayerNorm(msg_dim)
        )
    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], dim=-1)
        return self.mlp(out)

class NodeModel(nn.Module):
    def __init__(self, node_dim=64, msg_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + msg_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, node_dim), nn.ReLU(), nn.LayerNorm(node_dim)
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        agg = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce='sum')
        out = torch.cat([x, agg], dim=-1)
        return self.mlp(out)

class ThreatGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.LayerNorm(64))
        self.edge_enc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.LayerNorm(32))
        self.layers = nn.ModuleList([
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32)),
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32)),
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32))
        ])
        self.pool_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU())

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_enc(x)
        edge_attr = self.edge_enc(edge_attr)
        for layer in self.layers:
            x, edge_attr, _ = layer(x, edge_index, edge_attr, None, batch)
        pooled = scatter(x, batch, dim=0, reduce='mean')
        return self.pool_head(pooled)

class CrossThreatAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_scorer = nn.Sequential(
            nn.Linear(34, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, 1)
        )
        self.danger_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.LayerNorm(16),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, threat_embeds, contexts):
        if threat_embeds.size(0) == 0:
            return torch.tensor([0.0], device=threat_embeds.device), torch.empty(0, device=threat_embeds.device)
        combined = torch.cat([threat_embeds, contexts], dim=-1)
        scores = self.attn_scorer(combined)
        attn_weights = torch.softmax(scores / 1.5, dim=0)
        z = torch.sum(attn_weights * threat_embeds, dim=0, keepdim=True)
        danger = self.danger_head(z).squeeze(-1)
        return danger, attn_weights.squeeze(-1)

class NeuralEvasionBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.threat_gnn = ThreatGNN()
        self.attention = CrossThreatAttention()

    def forward(self, sub_graphs_list, contexts):
        if not sub_graphs_list:
            device = contexts.device if contexts is not None else torch.device('cpu')
            return torch.tensor([0.0], device=device), torch.empty(0, device=device)
        batched_data = Batch.from_data_list(sub_graphs_list)
        threat_embeds = self.threat_gnn(batched_data)
        danger, attn_weights = self.attention(threat_embeds, contexts)
        return danger, attn_weights

# --- 2. LOAD THE BRAIN ---
print("Loading Neural Engine...")
device = torch.device('cpu')
model = NeuralEvasionBrain().to(device) # <-- Updated Class
model.load_state_dict(torch.load('jet_brain_v1.pt', map_location=device), strict=False) # <-- Updated File
model.eval()
print("Engine Online.")

# --- 3. THREADING QUEUES & GRAPH BUILDER ---
state_queue = queue.Queue(maxsize=1)  
decision_queue = queue.Queue(maxsize=1) 

def build_candidate_sub_graphs(jet_sim, missiles_sim, flares_sim):
    sub_graphs = []
    for m in missiles_sim:
        nearby_flares = [f for f in flares_sim if math.hypot(f['pos'][0] - m['pos'][0], f['pos'][1] - m['pos'][1]) < 200][:6]
        
        nodes = [jet_sim, m] + nearby_flares
        x_rows, pos, vel = [], [], []
        
        for node in nodes:
            p = [node['pos'][0], node['pos'][1]]
            v = [node['vel'][0], node['vel'][1]]
            x_rows.append(p + v + [float(node['type'])])
            pos.append(torch.tensor(p))
            vel.append(torch.tensor(v))
            
        x = torch.tensor(x_rows, dtype=torch.float)
        num_nodes = len(nodes)
        
        edge_index_list, edge_attr_list = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue
                edge_index_list.append([i, j])
                rel_pos = pos[i] - pos[j]
                rel_vel = vel[i] - vel[j]
                edge_attr_list.append(torch.cat([rel_pos, rel_vel], dim=0))
                
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr_list, dim=0).float()
        
        dist = math.hypot(jet_sim['pos'][0] - m['pos'][0], jet_sim['pos'][1] - m['pos'][1])
        norm_dist = min(dist / 800.0, 1.0)
        rel_vx = m['vel'][0] - jet_sim['vel'][0]
        rel_vy = m['vel'][1] - jet_sim['vel'][1]
        closing_speed = max(((jet_sim['pos'][0]-m['pos'][0])*rel_vx + (jet_sim['pos'][1]-m['pos'][1])*rel_vy) / (dist + 1e-5), 0.1)
        tti = min((dist / closing_speed) / 5.0, 1.0)
        
        context = torch.tensor([norm_dist, tti], dtype=torch.float)
        sub_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, context=context))
        
    return sub_graphs

def gnn_inference_worker():
    while True:
        state = state_queue.get()
        if state is None:
            break

        jet_x = state['jet']['pos'][0]
        jet_y = state['jet']['pos'][1]

        best_heading  = state['jet']['heading']
        lowest_total  = 999.0
        best_attention = []

        candidates = [0, 0.26, -0.26, 0.52, -0.52, 0.8, -0.8, 1.2, -1.2]

        for delta in candidates:
            test_heading = state['jet']['heading'] + delta
            test_vx = math.cos(test_heading) * 4.0
            test_vy = math.sin(test_heading) * 4.0

            # Project position 30 frames ahead (was 15 — too short)
            proj_x = jet_x + test_vx * 30
            proj_y = jet_y + test_vy * 30

            # Wide wall exclusion zone: 100px padding
            if (proj_x < 100 or proj_x > 700 or
                    proj_y < 100 or proj_y > 450):
                total_danger = 10.0
                current_attn = []
            else:
                jet_sim = {
                    'pos': [jet_x, jet_y],
                    'vel': [test_vx, test_vy],
                    'type': 0
                }
                missiles_sim = [
                    {'pos': m['pos'], 'vel': m['vel'], 'type': 1}
                    for m in state['missiles']
                ]
                flares_sim = [
                    {'pos': f['pos'], 'vel': f['vel'], 'type': 2}
                    for f in state['flares']
                ]

                if missiles_sim:
                    sub_graphs = build_candidate_sub_graphs(
                        jet_sim, missiles_sim, flares_sim)
                    contexts = torch.stack(
                        [g.context for g in sub_graphs]).to(device)
                    with torch.no_grad():
                        dnn_danger, attn_weights = model(sub_graphs, contexts)
                    gnn_danger   = dnn_danger.item()
                    current_attn = attn_weights.cpu().numpy().tolist()
                else:
                    gnn_danger   = 0.0
                    current_attn = []

                # Distance-maintenance bonus: reward moves that increase
                # separation from ALL active missiles (uses jet speed advantage)
                escape_bonus = 0.0
                for m in missiles_sim:
                    cur_dist  = math.hypot(jet_x       - m['pos'][0],
                                           jet_y       - m['pos'][1])
                    proj_dist = math.hypot(proj_x      - m['pos'][0],
                                           proj_y      - m['pos'][1])
                    # negative bonus (reward) if we're moving away
                    if proj_dist > cur_dist:
                        escape_bonus -= 0.04
                    else:
                        escape_bonus += 0.04

                # Center gravity: mild pull toward arena center
                center_penalty = (math.hypot(proj_x - 400,
                                             proj_y - 275) / 800.0) * 0.10

                total_danger = gnn_danger + escape_bonus + center_penalty

            if total_danger < lowest_total:
                lowest_total   = total_danger
                best_heading   = test_heading
                best_attention = current_attn

        # Report actual raw gnn danger (not including bonuses) for HUD/FSM
        drop_flare = lowest_total > 0.42

        if not decision_queue.full():
            decision_queue.put(
                (best_heading, drop_flare, best_attention, lowest_total))

# --- 4. GAME ENGINE & CLASSES ---
WIDTH, HEIGHT = 800, 600
FPS = 60
BG_COLOR = (5, 10, 20)      
GRID_COLOR = (10, 31, 58)   
JET_COLOR = (0, 255, 200)   
MISSILE_COLOR = (255, 68, 68) 
FLARE_COLOR = (255, 176, 32)  

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Evasion - Tactical 1v1")
clock = pygame.time.Clock()

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

class Jet:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx, self.vy = 0, 0
        self.heading = 0 
        self.speed = 5.4
        self.max_speed = 5.4
        self.drag = 0.92
        self.radius = 11
        self.last_flare_time = 0
        self.last_attack_time = pygame.time.get_ticks()
        self.trail = deque(maxlen=20) # 20 positions for the jet

    def update(self):
        thrust_x = math.cos(self.heading) * self.speed
        thrust_y = math.sin(self.heading) * self.speed
        self.vx = self.vx * self.drag + thrust_x * 0.1
        self.vy = self.vy * self.drag + thrust_y * 0.1
        self.x += self.vx
        self.y += self.vy
        self.trail.append((self.x, self.y)) # Save position for trail

    def draw(self, surface):
        # Draw motion trail
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int((i / max(len(self.trail), 1)) * 160)
            r_t   = max(1, int((i / max(len(self.trail), 1)) * 3))
            pygame.draw.circle(surface, (0, 255, 200, alpha),
                               (int(tx), int(ty)), r_t)

        # Fighter jet polygon — sharp delta-wing silhouette
        # tip: forward  back_l/back_r: swept-back wings  notch: fuselage center
        R = self.radius
        tip       = (R * 2.2,  0.0)
        back_l    = (-R * 1.0,  R * 1.2)
        back_r    = (-R * 1.0, -R * 1.2)
        notch     = (-R * 0.2,  0.0)   # rear center notch for delta shape

        raw_pts = [tip, back_l, notch, back_r]

        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        def rot(px, py):
            return (
                self.x + px * cos_h - py * sin_h,
                self.y + px * sin_h + py * cos_h
            )

        pts = [rot(px, py) for px, py in raw_pts]

        # Barrel roll bank: compress Y axis
        if globals().get('ai_maneuver') == "BARREL_ROLL":
            roll_scale = max(0.15, abs(math.cos(
                pygame.time.get_ticks() * 0.012)))
            cy_avg = sum(p[1] for p in pts) / len(pts)
            pts = [(p[0], cy_avg + (p[1] - cy_avg) * roll_scale)
                   for p in pts]

        # Glow layer (large, translucent)
        pygame.draw.polygon(surface, (0, 255, 200, 35), pts)
        # Mid layer
        pygame.draw.polygon(surface, (0, 255, 200, 100), pts)
        # Sharp outline
        pygame.draw.polygon(surface, (0, 255, 200), pts)
        pygame.draw.polygon(surface, (255, 255, 255), pts, 1)

        # Engine glow at tail
        tail_x, tail_y = rot(-R * 0.9, 0)
        pygame.draw.circle(surface, (0, 255, 200, 180),
                           (int(tail_x), int(tail_y)), 4)
        pygame.draw.circle(surface, (255, 255, 200, 80),
                           (int(tail_x), int(tail_y)), 7)

class JetMissile:
    def __init__(self, x, y, target_x, target_y):
        self.x, self.y = x, y
        self.heading = math.atan2(target_y - y, target_x - x)
        self.speed = 6.0 
        self.radius = 3
        self.spawn_time = pygame.time.get_ticks()
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed
        self.dead = False
        self.trail = deque(maxlen=12)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.trail.append((self.x, self.y))
        if self.x < 0 or self.x > WIDTH or self.y < 0 or self.y > HEIGHT:
            self.dead = True

    def is_dead(self):
        return self.dead or (pygame.time.get_ticks() - self.spawn_time > 4000)

    def draw(self, surface):
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int((i / len(self.trail)) * 180)
            width = int((i / len(self.trail)) * 2) + 1
            pygame.draw.circle(surface, (*JET_COLOR, alpha), (int(tx), int(ty)), width)
        pygame.draw.circle(surface, JET_COLOR, (int(self.x), int(self.y)), self.radius)

class FlareParticle:
    def __init__(self, x, y, angle, speed):
        self.x, self.y = x, y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.drag = 0.85
        self.radius = 6
        self.spawn_time = pygame.time.get_ticks()

    def update(self):
        self.vx *= self.drag
        self.vy *= self.drag
        self.x += self.vx
        self.y += self.vy

    def is_dead(self):
        return pygame.time.get_ticks() - self.spawn_time > 2000 

    def draw(self, surface):
        pygame.draw.circle(surface, FLARE_COLOR, (int(self.x), int(self.y)), self.radius)

# Apply the same deque logic to all your missiles
class HomingMissile:
    def __init__(self, x, y, target, target_x, target_y):
        self.x, self.y = x, y
        self.target = target
        self.heading = math.atan2(target_y - y, target_x - x)
        self.speed = 5.0
        self.turn_rate = 0.15
        self.max_speed = self.speed  # needed for cluster missile speed scaling
        self.radius = 2
        self.spawn_time = pygame.time.get_ticks()
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed
        self.dead = False 
        self.trail = deque(maxlen=12) # Shorter trail for missiles
        self.spawn_delay = 0

    def update(self):
        if self.spawn_delay > 0:
            self.spawn_delay -= 1
            return
        dx, dy = self.target.x - self.x, self.target.y - self.y
        desired_heading = math.atan2(dy, dx)
        diff = normalize_angle(desired_heading - self.heading)
        self.heading += max(-self.turn_rate, min(self.turn_rate, diff))
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed
        self.x += self.vx
        self.y += self.vy
        self.trail.append((self.x, self.y))
        
        if self.x < 0 or self.x > WIDTH or self.y < 0 or self.y > HEIGHT:
            self.dead = True

    def is_dead(self):
        return self.dead or (pygame.time.get_ticks() - self.spawn_time > 8500)

    def draw(self, surface):
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int((i / len(self.trail)) * 180)
            width = int((i / len(self.trail)) * 2) + 1
            pygame.draw.circle(surface, (*MISSILE_COLOR, alpha), (int(tx), int(ty)), width)
        pygame.draw.circle(surface, MISSILE_COLOR, (int(self.x), int(self.y)), self.radius)

class BallisticMissile:
    def __init__(self, x, y, target_x, target_y):
        self.x, self.y = x, y
        self.heading = math.atan2(target_y - y, target_x - x)
        self.speed = 7.0 
        self.radius = 2
        self.spawn_time = pygame.time.get_ticks()
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed
        self.dead = False
        self.trail = deque(maxlen=12)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.trail.append((self.x, self.y))
        if self.x < 0 or self.x > WIDTH or self.y < 0 or self.y > HEIGHT:
            self.dead = True

    def is_dead(self):
        return self.dead or (pygame.time.get_ticks() - self.spawn_time > 3000)

    def draw(self, surface):
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int((i / len(self.trail)) * 180)
            width = int((i / len(self.trail)) * 2) + 1
            pygame.draw.circle(surface, (255, 100, 255, alpha), (int(tx), int(ty)), width)
        pygame.draw.circle(surface, (255, 100, 255), (int(self.x), int(self.y)), self.radius)

class ExplosionParticle:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(2, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.radius = random.uniform(2, 5)
        self.color = color
        self.life = 255 

    def update(self):
        self.vx *= 0.9  # Adds some air friction so they slow down naturally
        self.vy *= 0.9
        self.x += self.vx
        self.y += self.vy
        self.life -= 15
        self.radius -= 0.15

    def is_dead(self):
        return self.life <= 0 or self.radius <= 0

    def draw(self, surface):
        if self.life > 0 and self.radius > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.radius))

class Shockwave:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.radius = 8
        self.color = color
        self.alpha = 200

    def update(self):
        self.radius += 2.5 # Expands over ~12 frames
        self.alpha -= 15   # Fades out

    def is_dead(self):
        return self.alpha <= 0

    def draw(self, surface):
        if self.alpha > 0:
            pygame.draw.circle(surface, (*self.color, int(self.alpha)), (int(self.x), int(self.y)), int(self.radius), 2)

class Launcher:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.color = color
        self.last_fire = pygame.time.get_ticks() - 2001  # make the first shot available immediately
        self.alive = True

    def can_fire(self):
        return self.alive and (pygame.time.get_ticks() - self.last_fire >= 2000)

    def fire(self):
        self.last_fire = pygame.time.get_ticks()

    def draw(self, surface):
        if not self.alive:
            # Draw a busted, destroyed launcher so you know it's dead
            pygame.draw.rect(surface, (40, 40, 40), (self.x - 15, self.y - 10, 30, 20), 2)
            pygame.draw.line(surface, (150, 40, 40), (self.x - 10, self.y - 15), (self.x + 10, self.y + 5), 2)
            pygame.draw.line(surface, (150, 40, 40), (self.x - 10, self.y + 5), (self.x + 10, self.y - 15), 2)
            return
            
        pygame.draw.rect(surface, self.color, (self.x - 15, self.y - 10, 30, 20), 2)
        pygame.draw.circle(surface, self.color, (self.x, self.y), 5)

def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = math.hypot(x2 - x1, y2 - y1)
    if dl == 0: return
    dashes = int(dl / dash_length)
    for i in range(dashes):
        if i % 2 == 0:
            px1, py1 = x1 + (x2 - x1) * i / dashes, y1 + (y2 - y1) * i / dashes
            px2, py2 = x1 + (x2 - x1) * (i + 1) / dashes, y1 + (y2 - y1) * (i + 1) / dashes
            pygame.draw.line(surf, color, (int(px1), int(py1)), (int(px2), int(py2)), width)

# --- 5. MAIN LOOP SETUP ---
jet = Jet(WIDTH//2, HEIGHT//2)
missiles = []
ai_missiles = []
flares = []
explosions = [] # Our new particle container

# Sitting perfectly on the floor now
launcher_homing = Launcher(100, HEIGHT - 50, MISSILE_COLOR)
launcher_ballistic = Launcher(WIDTH - 100, HEIGHT - 50, (255, 100, 255))

glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
font = pygame.font.SysFont("impact", 64)

gnn_thread = threading.Thread(target=gnn_inference_worker, daemon=True)
gnn_thread.start()

frame_count = 0
running = True
game_over = False
user_won = False
shockwaves = []
radar_angle = 0.0
current_hud_attention = []
current_hud_danger = 0.0

# --- FSM STATE VARIABLES ---
# --- FSM STATE VARIABLES ---
ai_maneuver = "NORMAL"
maneuver_timer = 0
cobra_cooldown = 0
cluster_cooldown = 0
general_cooldown = 0 
maneuver_memory = {} 
global_gnn_heading = 0.0
global_gnn_flare = False

while running:
    frame_count += 1
    
    # Drain the cluster cooldown every frame
    if cluster_cooldown > 0:
        cluster_cooldown -= 1
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            state_queue.put(None)
            
        if not game_over:
            if event.type == pygame.KEYDOWN:
                # H — single homing missile from left launcher
                if event.key == pygame.K_h and launcher_homing.can_fire():
                    launcher_homing.fire()
                    missiles.append(HomingMissile(
                        launcher_homing.x, launcher_homing.y, jet, jet.x, jet.y))

                # B — predictive ballistic from right launcher
                elif event.key == pygame.K_b and launcher_ballistic.can_fire():
                    launcher_ballistic.fire()
                    missiles.append(BallisticMissile(
                        launcher_ballistic.x, launcher_ballistic.y, jet.x, jet.y))

                # C — cluster swarm (5 homing missiles, staggered speed + turn rate)
                elif event.key == pygame.K_c and cluster_cooldown == 0:
                    cluster_cooldown = 360  # 6-second cooldown at 60fps
                    base_angle = math.atan2(
                        jet.y - launcher_homing.y, jet.x - launcher_homing.x)
                    offsets = [-20, -10, 0, 10, 20]
                    speeds  = [3.5, 3.7, 3.9, 4.1, 4.3]   # 0.2 apart, none the same
                    turn_rates = [0.06, 0.05, 0.04, 0.03, 0.025]  # slower = holds fan longer
                    for i, (offset, spd, tr) in enumerate(
                            zip(offsets, speeds, turn_rates)):
                        rad_offset = math.radians(offset)
                        tx = launcher_homing.x + math.cos(base_angle + rad_offset) * 120
                        ty = launcher_homing.y + math.sin(base_angle + rad_offset) * 120
                        m = HomingMissile(launcher_homing.x, launcher_homing.y,
                                          jet, tx, ty)
                        m.speed     = spd
                        m.max_speed = spd
                        m.turn_rate = tr
                        m.spawn_delay = i * 3
                        missiles.append(m)

    if not game_over:
        # --- THE BRAIN CONNECTION ---
        if frame_count % 6 == 0 and state_queue.empty():
            state = {
                'jet': {'pos': [jet.x, jet.y], 'vel': [jet.vx, jet.vy], 'heading': jet.heading},
                'missiles': [{'pos': [m.x, m.y], 'vel': [m.vx, m.vy]} for m in missiles],
                'flares': [{'pos': [f.x, f.y], 'vel': [f.vx, f.vy]} for f in flares]
            }
            state_queue.put(state)

        # OLD (lines 617–626) — DELETE BOTH BLOCKS, replace with this:
        if not decision_queue.empty():
            gnn_heading, gnn_flare, current_hud_attention, current_hud_danger = decision_queue.get()
            global_gnn_heading = gnn_heading
            global_gnn_flare = gnn_flare
        
        # ---------------------------------------------------------
        # --- THE FSM MANEUVER BRAIN (Anti-Spam Edition) ---
        # ---------------------------------------------------------
        if cobra_cooldown > 0:
            cobra_cooldown -= 1
        if general_cooldown > 0:
            general_cooldown -= 1

        # ── Helper: heading toward arena center from current position ──
        def heading_to_center():
            return math.atan2(300 - jet.y, 400 - jet.x)

        # ── Wall proximity on each axis ─────────────────────────────────
        dist_left   = jet.x
        dist_right  = WIDTH  - jet.x
        dist_top    = jet.y
        dist_bottom = (HEIGHT - 50) - jet.y   # 50px for HUD bar at bottom

        wall_dist_x = min(dist_left, dist_right)
        wall_dist_y = min(dist_top,  dist_bottom)

        # Determine which specific wall is closest and the escape heading
        if dist_left < dist_right and dist_left < dist_top and dist_left < dist_bottom:
            wall_escape_heading = 0.0        # face right
        elif dist_right < dist_left and dist_right < dist_top and dist_right < dist_bottom:
            wall_escape_heading = math.pi    # face left
        elif dist_top < dist_bottom:
            wall_escape_heading = math.pi / 2  # face down
        else:
            wall_escape_heading = -math.pi / 2 # face up

        # ── Is jet MOVING toward the nearest wall? ──────────────────────
        # Only trigger IMMELMANN if we're actually heading into danger
        heading_toward_wall = False
        if dist_left   < 110 and jet.vx < -1.0: heading_toward_wall = True
        if dist_right  < 110 and jet.vx >  1.0: heading_toward_wall = True
        if dist_top    < 110 and jet.vy < -1.0: heading_toward_wall = True
        if dist_bottom < 110 and jet.vy >  1.0: heading_toward_wall = True

        # ── 1. IMMELMANN — wall escape, absolute override ───────────────
        if heading_toward_wall and ai_maneuver != "IMMELMANN":
            ai_maneuver = "IMMELMANN"
            maneuver_timer = 35
            # Directly set heading toward arena center at the START of maneuver
            jet.heading = wall_escape_heading

        elif ai_maneuver == "NORMAL" and general_cooldown == 0:

            close_missiles = [
                (m, math.hypot(m.x - jet.x, m.y - jet.y))
                for m in missiles
            ]

            # ── 2. COBRA — head-on single missile, very close ───────────
            head_on = False
            if sum(1 for _, d in close_missiles if d < 100) == 1:
                for m, dist in close_missiles:
                    if dist < 100:
                        dx = m.x - jet.x
                        dy = m.y - jet.y
                        if m.vx * dx + m.vy * dy < 0:
                            head_on = True
                            break

            if (current_hud_danger > 0.72 and head_on
                    and cobra_cooldown == 0
                    and wall_dist_x > 150 and wall_dist_y > 150):
                ai_maneuver  = "COBRA"
                maneuver_timer = 40
                cobra_cooldown = 480
                maneuver_memory['start_speed'] = jet.speed

            # ── 3. FALLING LEAF — multi-missile saturated ───────────────
            elif (current_hud_danger > 0.65
                    and len(missiles) >= 2
                    and sum(1 for _, d in close_missiles if d < 180) >= 2
                    and wall_dist_x > 120 and wall_dist_y > 120):
                ai_maneuver  = "FALLING_LEAF"
                maneuver_timer = 90

            # ── 4. JINKING — 3+ missiles, medium danger ─────────────────
            elif (0.48 < current_hud_danger < 0.68
                    and len(missiles) >= 3
                    and wall_dist_x > 100 and wall_dist_y > 100):
                ai_maneuver  = "JINKING"
                maneuver_timer = 120

            # ── 5. BARREL ROLL — single missile, open space ─────────────
            elif (0.50 < current_hud_danger < 0.65
                    and sum(1 for _, d in close_missiles if d < 200) == 1
                    and wall_dist_x > 150 and wall_dist_y > 150):
                ai_maneuver  = "BARREL_ROLL"
                maneuver_timer = 72

        # ── EXECUTE ACTIVE MANEUVER ─────────────────────────────────────
        if ai_maneuver == "IMMELMANN":
            jet.speed    = jet.max_speed * 1.3
            # Gradually blend toward wall_escape_heading over 35 frames
            diff = normalize_angle(wall_escape_heading - jet.heading)
            jet.heading += diff * 0.12          # smooth interpolation
            maneuver_timer -= 1
            if maneuver_timer <= 0:
                jet.heading  = heading_to_center()  # snap to center heading
                ai_maneuver  = "NORMAL"
                general_cooldown = 45

        elif ai_maneuver == "COBRA":
            if maneuver_timer > 20:
                jet.speed    = max(0.1, jet.speed -
                                   maneuver_memory['start_speed'] / 20)
                jet.heading += math.pi / 20
            elif maneuver_timer == 20:
                for i in range(14):
                    angle = i * (2 * math.pi / 14)
                    flares.append(FlareParticle(jet.x, jet.y, angle, 5.0))
            else:
                jet.speed    = min(jet.max_speed * 1.2,
                                   jet.speed + jet.max_speed * 1.2 / 20)
            maneuver_timer -= 1
            if maneuver_timer <= 0:
                ai_maneuver  = "NORMAL"
                general_cooldown = 60

        elif ai_maneuver == "FALLING_LEAF":
            jet.speed    = jet.max_speed * 0.2
            fc           = 90 - maneuver_timer
            jet.heading += math.sin(fc * 0.4) * 0.6
            if fc % 18 == 0:
                base = jet.heading + math.pi
                for i in range(6):
                    angle = base - math.pi/2 + i * math.pi / 5
                    flares.append(FlareParticle(jet.x, jet.y, angle, 4.0))
            maneuver_timer -= 1
            if (maneuver_timer <= 0 or not missiles
                    or current_hud_danger < 0.2):
                ai_maneuver  = "NORMAL"
                general_cooldown = 60

        elif ai_maneuver == "JINKING":
            jet.speed    = jet.max_speed * 0.85
            if maneuver_timer % 10 == 0:
                # Wall-safe jink: clamp proposed heading away from walls
                jink_delta   = random.choice([-0.45, -0.3, 0.3, 0.45])
                proposed     = jet.heading + jink_delta
                # Don't jink into a wall quadrant
                proposed_vx  = math.cos(proposed) * jet.speed
                proposed_vy  = math.sin(proposed) * jet.speed
                if not (
                    (jet.x + proposed_vx * 20 < 80) or
                    (jet.x + proposed_vx * 20 > 720) or
                    (jet.y + proposed_vy * 20 < 80) or
                    (jet.y + proposed_vy * 20 > 470)
                ):
                    jet.heading = proposed
            maneuver_timer -= 1
            if maneuver_timer <= 0:
                ai_maneuver  = "NORMAL"
                general_cooldown = 60

        elif ai_maneuver == "BARREL_ROLL":
            jet.speed    = jet.max_speed
            # Wall-safe roll: only continue spinning if not near wall
            if wall_dist_x > 100 and wall_dist_y > 100:
                jet.heading += 0.15
            else:
                # Near wall: abort roll, face center
                jet.heading  = heading_to_center()
                ai_maneuver  = "NORMAL"
                general_cooldown = 60
            maneuver_timer -= 1
            if maneuver_timer <= 0:
                ai_maneuver  = "NORMAL"
                general_cooldown = 90

        elif ai_maneuver == "NORMAL":
            jet.speed = jet.max_speed
            if global_gnn_heading != 0.0:
                jet.heading = global_gnn_heading

            if (global_gnn_flare
                    and pygame.time.get_ticks() - jet.last_flare_time > 2500
                    and wall_dist_x > 80 and wall_dist_y > 80):
                jet.last_flare_time = pygame.time.get_ticks()
                start_angle = jet.heading + math.radians(45)
                angle_step  = math.radians(270) / 7
                for i in range(8):
                    angle = start_angle + i * angle_step
                    flares.append(
                        FlareParticle(jet.x, jet.y, angle, jet.speed + 2.0))
                global_gnn_flare = False

        # --- JET ATTACK LOGIC ---
        if pygame.time.get_ticks() - jet.last_attack_time > 4000:
            targets = [l for l in [launcher_homing, launcher_ballistic] if l.alive]
            if targets:
                target = random.choice(targets)
                ai_missiles.append(JetMissile(jet.x, jet.y, target.x, target.y))
                jet.last_attack_time = pygame.time.get_ticks()

        # --- UPDATE PHYSICS & COLLISIONS ---
        jet.update()
        
        # Jet Wall Collision (Lethal)
        if jet.x < jet.radius or jet.x > WIDTH - jet.radius or jet.y < jet.radius or jet.y > HEIGHT - 50 - jet.radius:
            shockwaves.append(Shockwave(jet.x, jet.y, JET_COLOR))
            game_over = True
            user_won = True
            game_over_time = pygame.time.get_ticks()
            state_queue.put(None)
            # Massive cyan explosion for the jet
            for _ in range(50):
                shockwaves.append(Shockwave(jet.x, jet.y, JET_COLOR))
        
        # Flare Updates
        alive_flares = []
        for f in flares:
            f.update()
            if not f.is_dead():
                alive_flares.append(f)
        flares = alive_flares
        
        # Explosion Updates
        alive_explosions = []
        for e in explosions:
            e.update()
            if not e.is_dead():
                alive_explosions.append(e)
        explosions = alive_explosions
        
        # AI Missiles Updates
        alive_ai_missiles = []
        for aim in ai_missiles:
            aim.update()
            # Check collision with Launchers
            for launcher in [launcher_homing, launcher_ballistic]:
                if launcher.alive:
                    if math.hypot(launcher.x - aim.x, launcher.y - aim.y) < (15 + aim.radius):
                        launcher.alive = False
                        aim.dead = True
                        # Launcher gets blown to bits
                        for _ in range(30):
                            explosions.append(ExplosionParticle(launcher.x, launcher.y, (255, 80, 50)))
                        shockwaves.append(Shockwave(launcher.x, launcher.y, (255, 80, 50)))
                        break
            if not aim.is_dead():
                alive_ai_missiles.append(aim)
        ai_missiles = alive_ai_missiles
        
        # User Missiles Updates
        alive_missiles = []
        for m in missiles:
            m.update()
            
            # 1. Jet vs Missile
            if math.hypot(jet.x - m.x, jet.y - m.y) < (jet.radius + m.radius):
                game_over = True
                user_won = True
                game_over_time = pygame.time.get_ticks()
                state_queue.put(None)
                # Blow up the jet
                for _ in range(50):
                    explosions.append(ExplosionParticle(jet.x, jet.y, JET_COLOR))
                shockwaves.append(Shockwave(jet.x, jet.y, JET_COLOR))
                continue
                
            # 2. Flare vs Missile
            for f in flares:
                if math.hypot(f.x - m.x, f.y - m.y) < (f.radius + m.radius):
                    f.spawn_time -= 5000 
                    m.dead = True 
                    # Tiny pop for hitting a flare
                    for _ in range(5):
                        explosions.append(ExplosionParticle(f.x, f.y, FLARE_COLOR))
                    shockwaves.append(Shockwave(f.x, f.y, FLARE_COLOR))
                    break
                    
            # 3. Intercept! User Missile vs AI Missile
            for aim in ai_missiles:
                if math.hypot(aim.x - m.x, aim.y - m.y) < (aim.radius + m.radius):
                    m.dead = True
                    aim.dead = True
                    # Mid-air collision sparks
                    for _ in range(20):
                        explosions.append(ExplosionParticle(aim.x, aim.y, (255, 255, 255)))
                    shockwaves.append(Shockwave(aim.x, aim.y, (255, 255, 255)))
                    break
            
            if not m.is_dead():
                alive_missiles.append(m)
                
        missiles = alive_missiles
        
        # Check AI Win Condition
        if not launcher_homing.alive and not launcher_ballistic.alive:
            game_over = True
            user_won = False
            game_over_time = pygame.time.get_ticks()
            state_queue.put(None)

    # ═══════════════════════════════════════════════════════
    # RENDER — Full redesigned UI
    # ═══════════════════════════════════════════════════════
    screen.fill((4, 8, 16))          # deep space black
    glow_surface.fill((0, 0, 0, 0))

    # ── Radar grid: concentric rings + radials ──────────────
    cx, cy = WIDTH // 2, (HEIGHT - 50) // 2
    for r in [60, 120, 180, 240, 300]:
        alpha_val = max(30, 80 - r // 5)
        pygame.draw.circle(screen, (0, 40, 60), (cx, cy), r, 1)
    for i in range(12):
        angle = i * math.pi / 6
        ex = cx + math.cos(angle) * 340
        ey = cy + math.sin(angle) * 340
        pygame.draw.line(screen, (0, 30, 50), (cx, cy), (int(ex), int(ey)), 1)

    # Rotating radar sweep with fading arc
    radar_angle = (radar_angle + 0.025) % (math.pi * 2)
    for sweep_offset in range(0, 40, 4):
        arc_angle = radar_angle - math.radians(sweep_offset)
        sa = (0, max(0, 200 - sweep_offset * 5), max(0, 160 - sweep_offset * 4),
              max(0, 180 - sweep_offset * 5))
        ex = cx + math.cos(arc_angle) * 340
        ey = cy + math.sin(arc_angle) * 340
        pygame.draw.line(glow_surface, sa, (cx, cy), (int(ex), int(ey)), 2)

    # ── Static objects ──────────────────────────────────────
    launcher_homing.draw(screen)
    launcher_ballistic.draw(screen)

    # ── Moving objects ──────────────────────────────────────
    if not game_over:
        jet.draw(glow_surface)
    for f in flares:
        f.draw(glow_surface)
    for m in missiles:
        m.draw(glow_surface)
    for aim in ai_missiles:
        aim.draw(glow_surface)
    for e in explosions:
        e.draw(screen)
    for s in shockwaves:
        s.update()
        s.draw(glow_surface)

    # Filter dead shockwaves
    shockwaves[:] = [s for s in shockwaves if not s.is_dead()]

    screen.blit(glow_surface, (0, 0))

    # ── Bottom HUD bar ──────────────────────────────────────
    hud_y    = HEIGHT - 50
    hud_font = pygame.font.SysFont("courier", 13, bold=True)
    hud_sm   = pygame.font.SysFont("courier", 11)

    # Background panel
    pygame.draw.rect(screen, (6, 14, 26), (0, hud_y, WIDTH, 50))
    pygame.draw.line(screen, (0, 180, 140), (0, hud_y), (WIDTH, hud_y), 1)

    # Danger meter (left side)
    danger_clamped = min(max(current_hud_danger, 0.0), 1.0)
    meter_w        = 160
    meter_h        = 12
    mx_pos, my_pos = 14, hud_y + 10
    pygame.draw.rect(screen, (20, 30, 40),
                     (mx_pos, my_pos, meter_w, meter_h), 0, 4)
    r_col = int(danger_clamped * 255)
    g_col = int((1.0 - danger_clamped) * 200)
    fill_w = int(danger_clamped * meter_w)
    if fill_w > 0:
        pygame.draw.rect(screen, (r_col, g_col, 0),
                         (mx_pos, my_pos, fill_w, meter_h), 0, 4)
    pygame.draw.rect(screen, (0, 180, 140),
                     (mx_pos, my_pos, meter_w, meter_h), 1, 4)
    dlbl = hud_font.render("THREAT", True, (0, 200, 160))
    screen.blit(dlbl, (mx_pos, my_pos + 16))

    # Maneuver state label (center)
    maneuver_color = {
        "NORMAL":       (80,  180, 140),
        "IMMELMANN":    (255, 220,  50),
        "COBRA":        (255,  80,  80),
        "FALLING_LEAF": (255, 160,  30),
        "JINKING":      (80,  160, 255),
        "BARREL_ROLL":  (180,  80, 255),
    }.get(ai_maneuver, (255, 255, 255))
    mlbl = hud_font.render(f"[ {ai_maneuver} ]", True, maneuver_color)
    screen.blit(mlbl, (WIDTH // 2 - mlbl.get_width() // 2, hud_y + 8))

    # Controls legend (right side)
    ctrl_txt = "H:Homing  B:Ballistic  C:Cluster"
    clbl = hud_sm.render(ctrl_txt, True, (60, 100, 120))
    screen.blit(clbl, (WIDTH - clbl.get_width() - 14, hud_y + 28))

    # ── Attention HUD (top-right panel) ─────────────────────
    attn_font  = pygame.font.SysFont("courier", 12, bold=True)
    attn_label = pygame.font.SysFont("courier", 10)
    panel_x    = WIDTH - 190
    panel_y    = 12
    panel_w    = 178
    panel_h    = 130

    # Panel background + border
    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((6, 14, 30, 210))
    screen.blit(panel_surf, (panel_x, panel_y))
    pygame.draw.rect(screen, (0, 140, 110),
                     (panel_x, panel_y, panel_w, panel_h), 1, 6)

    # Title
    attn_title = attn_font.render("AI ATTENTION", True, (0, 220, 170))
    screen.blit(attn_title, (panel_x + 8, panel_y + 7))

    if current_hud_attention:
        num_bars = min(len(current_hud_attention), 8)
        bar_area_w = panel_w - 16
        bar_w      = max(14, bar_area_w // num_bars - 4)
        bar_max_h  = 60
        bar_base_y = panel_y + 100

        for i in range(num_bars):
            w_val   = current_hud_attention[i]
            bx      = panel_x + 8 + i * (bar_w + 4)
            bh      = int(w_val * bar_max_h)

            # Background track
            pygame.draw.rect(screen, (15, 30, 45),
                             (bx, bar_base_y - bar_max_h, bar_w, bar_max_h),
                             0, 3)

            # Gradient bar: green→yellow→red based on weight
            r_b = int(w_val * 255)
            g_b = int((1.0 - w_val) * 220)
            bar_color = (r_b, g_b, 30)

            # Flash white if near max and high danger
            if w_val > 0.80 and current_hud_danger > 0.55:
                if (pygame.time.get_ticks() // 150) % 2 == 0:
                    bar_color = (255, 255, 255)

            if bh > 0:
                pygame.draw.rect(screen, bar_color,
                                 (bx, bar_base_y - bh, bar_w, bh), 0, 3)
            pygame.draw.rect(screen, (0, 160, 120),
                             (bx, bar_base_y - bar_max_h, bar_w, bar_max_h),
                             1, 3)

            # Label
            m_lbl = attn_label.render(f"M{i+1}", True, (120, 180, 160))
            screen.blit(m_lbl, (bx + bar_w // 2 - 7, bar_base_y + 2))

            # Weight percentage
            pct = attn_label.render(f"{int(w_val*100)}%", True,
                                    (80, 140, 120))
            screen.blit(pct, (bx - 1, bar_base_y - bh - 14))
    else:
        no_thr = attn_label.render("no threats", True, (40, 80, 70))
        screen.blit(no_thr, (panel_x + 45, panel_y + 65))

    # ── Game over overlay ───────────────────────────────────
    if game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT - 50), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        screen.blit(overlay, (0, 0))
        go_font = pygame.font.SysFont("impact", 56)
        sub_font = pygame.font.SysFont("courier", 20, bold=True)
        if user_won:
            msg     = "JET DESTROYED"
            sub_msg = "YOU WIN — AI DEFEATED"
            c1, c2  = (255, 60, 60), (200, 80, 80)
        else:
            msg     = "BASES OBLITERATED"
            sub_msg = "AI WINS — HUMANITY LOSES"
            c1, c2  = (0, 255, 200), (0, 180, 140)
        t1 = go_font.render(msg, True, c1)
        t2 = sub_font.render(sub_msg, True, c2)
        screen.blit(t1, (WIDTH//2 - t1.get_width()//2, HEIGHT//2 - 60))
        screen.blit(t2, (WIDTH//2 - t2.get_width()//2, HEIGHT//2 + 10))
        if pygame.time.get_ticks() - game_over_time > 3000:
            running = False

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()