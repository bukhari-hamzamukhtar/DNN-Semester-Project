import pygame
import math
import random
from collections import deque
import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter
import threading
import queue


# ═══════════════════════════════════════════════════════════════
#  1.  GNN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
class EdgeModel(nn.Module):
    def __init__(self, node_dim=64, edge_dim=32, msg_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, msg_dim), nn.ReLU(), nn.LayerNorm(msg_dim)
        )
    def forward(self, src, dest, edge_attr, u, batch):
        return self.mlp(torch.cat([src, dest, edge_attr], dim=-1))

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
        return self.mlp(torch.cat([x, agg], dim=-1))

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

MANEUVER_NAMES = ["NORMAL", "BARREL_ROLL", "JINKING", "FALLING_LEAF", "COBRA", "IMMELMANN"]

class TemporalThreatAttention(nn.Module):
    def __init__(self, embed_dim=32, context_dim=5, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(embed_dim + context_dim, hidden_dim)
        self.attn_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.LayerNorm(32), nn.Linear(32, 1)
        )
        self.danger_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(), nn.LayerNorm(16),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, threat_embeds, contexts, hidden=None):
        if threat_embeds.size(0) == 0:
            dev = threat_embeds.device
            return torch.tensor([0.0], device=dev), torch.empty(0, device=dev), None
        K = threat_embeds.size(0)
        if hidden is None or hidden.size(0) != K:
            hidden = torch.zeros(K, self.hidden_dim, device=threat_embeds.device)
        combined    = torch.cat([threat_embeds, contexts], dim=-1)
        new_hidden  = self.gru(combined, hidden)
        scores      = self.attn_scorer(new_hidden)
        attn_weights = torch.softmax(scores / 0.5, dim=0)
        z      = torch.sum(attn_weights * new_hidden, dim=0, keepdim=True)
        danger = self.danger_head(z).squeeze(-1)
        return danger, attn_weights.squeeze(-1), new_hidden

class NeuralEvasionBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.threat_gnn = ThreatGNN()
        self.attention  = TemporalThreatAttention()
        # Maneuver head: 64 (GRU hidden pooled) + 1 (danger) → 6 logits
        self.maneuver_head = nn.Sequential(
            nn.Linear(65, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, len(MANEUVER_NAMES))
        )

    def forward(self, sub_graphs_list, contexts, hidden=None):
        if not sub_graphs_list:
            dev = contexts.device if contexts is not None else torch.device('cpu')
            return (torch.tensor([0.0], device=dev),
                    torch.empty(0, device=dev),
                    None,
                    torch.zeros(len(MANEUVER_NAMES), device=dev))
        batched = Batch.from_data_list(sub_graphs_list)
        embeds  = self.threat_gnn(batched)
        danger, attn_weights, new_hidden = self.attention(embeds, contexts, hidden)
        # Pool GRU hidden states for maneuver head
        pooled      = new_hidden.mean(dim=0, keepdim=True)       # [1, 64]
        m_input     = torch.cat([pooled, danger.unsqueeze(-1).unsqueeze(0)
                                 if danger.dim()==0 else danger.unsqueeze(-1).reshape(1,1)], dim=-1)  # [1, 65]
        maneuver_logits = self.maneuver_head(m_input).squeeze(0) # [6]
        return danger, attn_weights, new_hidden, maneuver_logits

# ═══════════════════════════════════════════════════════════════
#  2.  LOAD MODEL
# ═══════════════════════════════════════════════════════════════
print("Loading Neural Engine...")
device = torch.device('cpu')
model = NeuralEvasionBrain().to(device)

try:
    # 1. Load the raw dictionary from the file
    checkpoint = torch.load('jet_brain_v3.pt', map_location=device)
    model_dict = model.state_dict()

    # 2. Filter out anything that doesn't have the exact same shape
    filtered_dict = {k: v for k, v in checkpoint.items() 
                     if k in model_dict and v.shape == model_dict[k].shape}

    # 3. Update the model's dictionary and load it safely
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print("Engine Online. Weights loaded (Mismatched shapes skipped!).")
    
except FileNotFoundError:
    print("WARNING: jet_brain_v3.pt not found — running with random weights.")
    
model.eval()

# ═══════════════════════════════════════════════════════════════
#  3.  THREADING & GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════
state_queue    = queue.Queue(maxsize=1)
decision_queue = queue.Queue(maxsize=1)

def build_candidate_sub_graphs(jet_sim, missiles_sim, flares_sim):
    sub_graphs = []
    for m in missiles_sim:
        nearby_flares = [
            f for f in flares_sim
            if math.hypot(f['pos'][0]-m['pos'][0], f['pos'][1]-m['pos'][1]) < 200
        ][:6]
        nodes = [jet_sim, m] + nearby_flares
        x_rows, pos, vel = [], [], []
        for node in nodes:
            p = [node['pos'][0], node['pos'][1]]
            v = [node['vel'][0], node['vel'][1]]
            x_rows.append(p + v + [float(node['type'])])
            pos.append(torch.tensor(p))
            vel.append(torch.tensor(v))
        x = torch.tensor(x_rows, dtype=torch.float)
        n = len(nodes)
        ei, ea = [], []
        for i in range(n):
            for j in range(n):
                if i == j: continue
                ei.append([i, j])
                ea.append(torch.cat([pos[i]-pos[j], vel[i]-vel[j]], dim=0))
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_attr  = torch.stack(ea, dim=0).float()
        dist = math.hypot(jet_sim['pos'][0]-m['pos'][0], jet_sim['pos'][1]-m['pos'][1])
        nd   = min(dist/800.0, 1.0)
        # Signed closing speed
        dx   = jet_sim['pos'][0] - m['pos'][0]
        dy   = jet_sim['pos'][1] - m['pos'][1]
        rvx  = m['vel'][0] - jet_sim['vel'][0]
        rvy  = m['vel'][1] - jet_sim['vel'][1]
        closing = (dx*rvx + dy*rvy) / (dist + 1e-5)
        closing_norm = max(-1.0, min(1.0, closing / 10.0))
        if closing > 0.05:
            tti  = min((dist / closing) / 60.0, 1.0)
        else:
            tti  = 1.0
        # Angle of attack
        mh_len = math.hypot(rvx, rvy) + 1e-5
        aoa    = max(0.0, (dx*rvx + dy*rvy) / ((dist+1e-5) * mh_len))
        mtype  = float(m.get('type', 1))
        ctx  = torch.tensor([nd, tti, closing_norm, aoa, mtype], dtype=torch.float)
        sub_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, context=ctx))
    return sub_graphs

def gnn_inference_worker():
    gru_hidden         = None   # persists missile trajectory memory across frames
    prev_missile_count = 0      # used to detect when to reset hidden state
    while True:
        state = state_queue.get()
        if state is None:
            break
        jx, jy = state['jet']['pos']
        best_heading  = state['jet']['heading']
        lowest_total  = 999.0
        best_attention = []
        candidates = [0, 0.20, -0.20, 0.40, -0.40, 0.65, -0.65, 1.0, -1.0, 1.4, -1.4, math.pi]
        candidate_scores = []
        best_edges = []
        current_situation_danger = 0.0 
        for delta in candidates:
            th    = state['jet']['heading'] + delta
            tvx   = math.cos(th) * 4.0
            tvy   = math.sin(th) * 4.0
            px    = jx + tvx * 30
            py    = jy + tvy * 30
            if px < 100 or px > 700 or py < 100 or py > 450:
                total_danger  = 10.0
                current_attn  = []
            else:
                js  = {'pos':[jx,jy], 'vel':[tvx,tvy], 'type':0}
                ms  = [{'pos':m['pos'],'vel':m['vel'],'type':1} for m in state['missiles']]
                fs  = [{'pos':f['pos'],'vel':f['vel'],'type':2} for f in state['flares']]
                if ms:
                    sgs  = build_candidate_sub_graphs(js, ms, fs)
                    ctxs = torch.stack([g.context for g in sgs]).to(device)
                    with torch.no_grad():
                        dnn_danger, aw, _, _ = model(sgs, ctxs, gru_hidden)
                    gnn_d        = dnn_danger.item()
                    current_attn = aw.cpu().numpy().tolist()
                else:
                    gnn_d        = 0.0
                    current_attn = []
                esc = 0.0
                for m in ms:
                    cd = math.hypot(jx-m['pos'][0], jy-m['pos'][1])
                    pd = math.hypot(px-m['pos'][0], py-m['pos'][1])
                    esc += -0.04 if pd > cd else 0.04
                cp = (math.hypot(px-400, py-275)/800.0)*0.10
                total_danger = gnn_d + esc + cp
                if delta == 0:  # current heading
                    current_situation_danger = total_danger

                if ms and total_danger < lowest_total:
                    with torch.no_grad():
                        batched = Batch.from_data_list(sgs)
                        x = model.threat_gnn.node_enc(batched.x)
                        ea = model.threat_gnn.edge_enc(batched.edge_attr)
                        for layer in model.threat_gnn.layers:
                            x, ea, _ = layer(x, batched.edge_index, ea, None, batched.batch)
                        magnitudes = ea.norm(dim=1).cpu().numpy()

                    edge_data = []
                    for k, sg in enumerate(sgs):
                        nodes_pos = [
                            [js['pos'][0], js['pos'][1]],
                            [ms[k]['pos'][0], ms[k]['pos'][1]]
                        ] + [[f['pos'][0], f['pos'][1]] for f in fs[:6]]
                        for idx, (i, j) in enumerate(sg.edge_index.t().tolist()):
                            if i < len(nodes_pos) and j < len(nodes_pos):
                                edge_data.append({
                                    'p1': nodes_pos[i],
                                    'p2': nodes_pos[j],
                                    'mag': float(magnitudes[idx]) if idx < len(magnitudes) else 0.0
                                })
                    best_edges = edge_data

            candidate_scores.append((th, total_danger))
            if total_danger < lowest_total:
                lowest_total  = total_danger
                best_heading  = th
                best_attention = current_attn

        # Only drop flare when genuinely hot AND missiles are present
        # ── REAL FORWARD PASS: updates GRU memory + gets maneuver recommendation ──
        ms_count = len(state['missiles'])
        if ms_count != prev_missile_count:
            gru_hidden = None
        prev_missile_count = ms_count

        current_situation_danger = 0.0
        gnn_recommended_maneuver = "NORMAL"

        if state['missiles']:
            js_real = {'pos': state['jet']['pos'], 'vel': state['jet']['vel'], 'type': 0}
            ms_real = [{'pos': m['pos'], 'vel': m['vel'], 'type': 1} for m in state['missiles']]
            fs_real = [{'pos': f['pos'], 'vel': f['vel'], 'type': 2} for f in state['flares']]
            sgs_real = build_candidate_sub_graphs(js_real, ms_real, fs_real)
            if sgs_real:
                ctxs_real = torch.stack([g.context for g in sgs_real]).to(device)
                with torch.no_grad():
                    d_real, _, gru_hidden, m_logits = model(sgs_real, ctxs_real, gru_hidden)
                current_situation_danger     = d_real.item()
                gnn_recommended_maneuver     = MANEUVER_NAMES[m_logits.argmax().item()]

        drop_flare = (current_situation_danger > 0.55) and len(state['missiles']) > 0
        if not decision_queue.full():
            decision_queue.put((best_heading, drop_flare, best_attention,
                                current_situation_danger, best_edges,
                                candidate_scores, gnn_recommended_maneuver))

# ═══════════════════════════════════════════════════════════════
#  4.  CONSTANTS & PYGAME INIT
# ═══════════════════════════════════════════════════════════════
WIDTH, HEIGHT = 800, 600
FPS           = 60
BG_COLOR      = (5, 10, 20)
JET_COLOR     = (0, 255, 200)
MISSILE_COLOR = (255, 68, 68)
FLARE_COLOR   = (255, 176, 32)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Evasion — Tactical AI 1v1")
clock = pygame.time.Clock()

def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

# ═══════════════════════════════════════════════════════════════
#  5.  GAME OBJECT CLASSES
# ═══════════════════════════════════════════════════════════════
class Jet:
    def __init__(self, x, y):
        self.x, self.y     = float(x), float(y)
        self.vx, self.vy   = 0.0, 0.0
        self.heading       = 0.0
        self.speed         = 5.0
        self.max_speed     = 5.0
        self.drag          = 0.92
        self.radius        = 11
        self.last_flare_time  = 0
        self.last_attack_time = pygame.time.get_ticks()
        self.trail = deque(maxlen=20)

    def update(self):
        self.vx = self.vx*self.drag + math.cos(self.heading)*self.speed*0.1
        self.vy = self.vy*self.drag + math.sin(self.heading)*self.speed*0.1
        self.x += self.vx
        self.y += self.vy
        self.trail.append((self.x, self.y))

    def draw(self, surface, ai_maneuver):
        for i, (tx, ty) in enumerate(self.trail):
            a = int((i/max(len(self.trail),1))*160)
            r = max(1, int((i/max(len(self.trail),1))*3))
            pygame.draw.circle(surface, (0,255,200,a), (int(tx),int(ty)), r)
        R = self.radius
        tip    = ( R*2.2,  0.0)
        back_l = (-R*1.0,  R*1.2)
        back_r = (-R*1.0, -R*1.2)
        notch  = (-R*0.2,  0.0)
        raw    = [tip, back_l, notch, back_r]
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        def rot(px,py): return (self.x+px*ch-py*sh, self.y+px*sh+py*ch)
        pts = [rot(px,py) for px,py in raw]
        if ai_maneuver == "BARREL_ROLL":
            rs = max(0.15, abs(math.cos(pygame.time.get_ticks()*0.012)))
            cy_avg = sum(p[1] for p in pts)/len(pts)
            pts = [(p[0], cy_avg+(p[1]-cy_avg)*rs) for p in pts]
        pygame.draw.polygon(surface, (0,255,200,35),  pts)
        pygame.draw.polygon(surface, (0,255,200,100), pts)
        pygame.draw.polygon(surface, (0,255,200),     pts)
        pygame.draw.polygon(surface, (255,255,255),   pts, 1)
        tx2, ty2 = rot(-R*0.9, 0)
        pygame.draw.circle(surface, (0,255,200,180), (int(tx2),int(ty2)), 4)
        pygame.draw.circle(surface, (255,255,200,80), (int(tx2),int(ty2)), 7)

class JetMissile:
    def __init__(self, x, y, tx, ty):
        self.x, self.y   = float(x), float(y)
        self.heading     = math.atan2(ty-y, tx-x)
        self.speed       = 4.0
        self.radius      = 4
        self.spawn_time  = pygame.time.get_ticks()
        self.vx = math.cos(self.heading)*self.speed
        self.vy = math.sin(self.heading)*self.speed
        self.dead  = False
        self.trail = deque(maxlen=12)
    def update(self):
        self.x += self.vx; self.y += self.vy
        self.trail.append((self.x, self.y))
        if self.x<0 or self.x>WIDTH or self.y<0 or self.y>HEIGHT:
            self.dead = True
    def is_dead(self): return self.dead or pygame.time.get_ticks()-self.spawn_time>4000
    def draw(self, s):
        for i,(tx,ty) in enumerate(self.trail):
            a=int((i/len(self.trail))*180)
            pygame.draw.circle(s,(*JET_COLOR,a),(int(tx),int(ty)),max(1,int((i/len(self.trail))*2)+1))
        pygame.draw.circle(s,JET_COLOR,(int(self.x),int(self.y)),self.radius)

class FlareParticle:
    def __init__(self, x, y, angle, speed):
        self.x, self.y  = float(x), float(y)
        self.vx = math.cos(angle)*speed
        self.vy = math.sin(angle)*speed
        self.drag       = 0.85
        self.radius     = 6
        self.spawn_time = pygame.time.get_ticks()
    def update(self):
        self.vx*=self.drag; self.vy*=self.drag
        self.x+=self.vx; self.y+=self.vy
    def is_dead(self): return pygame.time.get_ticks()-self.spawn_time>2000
    def draw(self, s): pygame.draw.circle(s,FLARE_COLOR,(int(self.x),int(self.y)),self.radius)

class HomingMissile:
    def __init__(self, x, y, target, tx, ty):
        self.x, self.y   = float(x), float(y)
        self.target      = target
        self.heading     = math.atan2(ty-y, tx-x)
        self.speed       = 5.0
        self.max_speed   = 5.0
        self.turn_rate   = 0.15
        self.radius      = 2
        self.spawn_time  = pygame.time.get_ticks()
        self.vx = math.cos(self.heading)*self.speed
        self.vy = math.sin(self.heading)*self.speed
        self.dead        = False
        self.trail       = deque(maxlen=12)
        self.spawn_delay = 0
        self.lateral_bias = 0.0   # persistent curve for cluster formation spread
    def update(self):
        if self.spawn_delay>0: self.spawn_delay-=1; return
        dx,dy = self.target.x-self.x, self.target.y-self.y
        dh    = math.atan2(dy,dx)
        diff  = normalize_angle(dh-self.heading)
        self.heading += max(-self.turn_rate, min(self.turn_rate, diff))
        self.heading += self.lateral_bias  # cluster spread: each missile drifts its own way
        self.vx = math.cos(self.heading)*self.speed
        self.vy = math.sin(self.heading)*self.speed
        self.x += self.vx; self.y += self.vy
        self.trail.append((self.x,self.y))
        if self.x<0 or self.x>WIDTH or self.y<0 or self.y>HEIGHT: self.dead=True
    def is_dead(self): return self.dead or pygame.time.get_ticks()-self.spawn_time>8500
    def draw(self, s):
        for i,(tx,ty) in enumerate(self.trail):
            a=int((i/len(self.trail))*180)
            pygame.draw.circle(s,(*MISSILE_COLOR,a),(int(tx),int(ty)),max(1,int((i/len(self.trail))*2)+1))
        pygame.draw.circle(s,MISSILE_COLOR,(int(self.x),int(self.y)),self.radius)

class BallisticMissile:
    def __init__(self, x, y, tx, ty):
        self.x, self.y  = float(x), float(y)
        self.heading    = math.atan2(ty-y, tx-x)
        self.speed      = 7.0
        self.radius     = 2
        self.spawn_time = pygame.time.get_ticks()
        self.vx = math.cos(self.heading)*self.speed
        self.vy = math.sin(self.heading)*self.speed
        self.dead  = False
        self.trail = deque(maxlen=12)
    def update(self):
        self.x+=self.vx; self.y+=self.vy
        self.trail.append((self.x,self.y))
        if self.x<0 or self.x>WIDTH or self.y<0 or self.y>HEIGHT: self.dead=True
    def is_dead(self): return self.dead or pygame.time.get_ticks()-self.spawn_time>3000
    def draw(self, s):
        for i,(tx,ty) in enumerate(self.trail):
            a=int((i/len(self.trail))*180)
            pygame.draw.circle(s,(255,100,255,a),(int(tx),int(ty)),max(1,int((i/len(self.trail))*2)+1))
        pygame.draw.circle(s,(255,100,255),(int(self.x),int(self.y)),self.radius)

class ExplosionParticle:
    def __init__(self, x, y, color):
        self.x,self.y = float(x),float(y)
        a = random.uniform(0,math.pi*2); sp = random.uniform(2,8)
        self.vx=math.cos(a)*sp; self.vy=math.sin(a)*sp
        self.radius=random.uniform(2,5); self.color=color; self.life=255
    def update(self):
        self.vx*=0.9; self.vy*=0.9
        self.x+=self.vx; self.y+=self.vy
        self.life-=15; self.radius-=0.15
    def is_dead(self): return self.life<=0 or self.radius<=0
    def draw(self, s):
        if self.life>0 and self.radius>0:
            pygame.draw.circle(s,self.color,(int(self.x),int(self.y)),int(self.radius))

class Shockwave:
    def __init__(self, x, y, color):
        self.x,self.y=float(x),float(y); self.radius=8; self.color=color; self.alpha=200
    def update(self): self.radius+=2.5; self.alpha-=15
    def is_dead(self): return self.alpha<=0
    def draw(self, s):
        if self.alpha>0:
            pygame.draw.circle(s,(*self.color,int(self.alpha)),(int(self.x),int(self.y)),int(self.radius),2)

class Launcher:
    def __init__(self, x, y, color):
        self.x,self.y=x,y; self.color=color
        self.last_fire=pygame.time.get_ticks()-2001; self.alive=True
    def can_fire(self): return self.alive and pygame.time.get_ticks()-self.last_fire>=2000
    def fire(self): self.last_fire=pygame.time.get_ticks()
    def draw(self, s):
        if not self.alive:
            pygame.draw.rect(s,(40,40,40),(self.x-15,self.y-10,30,20),2)
            pygame.draw.line(s,(150,40,40),(self.x-10,self.y-15),(self.x+10,self.y+5),2)
            pygame.draw.line(s,(150,40,40),(self.x-10,self.y+5),(self.x+10,self.y-15),2)
            return
        pygame.draw.rect(s,self.color,(self.x-15,self.y-10,30,20),2)
        pygame.draw.circle(s,self.color,(self.x,self.y),5)

# ═══════════════════════════════════════════════════════════════
#  6.  BEAUTIFUL SCREEN HELPERS
# ═══════════════════════════════════════════════════════════════
# Font declarations (loaded once)
try:
    font_title  = pygame.font.SysFont("impact", 72)
    font_sub    = pygame.font.SysFont("courier", 18, bold=True)
    font_hud    = pygame.font.SysFont("courier", 13, bold=True)
    font_sm     = pygame.font.SysFont("courier", 11)
    font_tiny   = pygame.font.SysFont("courier", 10)
    font_btn    = pygame.font.SysFont("courier", 16, bold=True)
    font_result = pygame.font.SysFont("impact", 56)
except:
    font_title = font_sub = font_hud = font_sm = font_tiny = font_btn = font_result = pygame.font.SysFont(None, 32)

# Background particle field shared across screens
class BgParticle:
    def __init__(self):
        self.reset()
    def reset(self):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        self.speed = random.uniform(0.2, 0.8)
        self.angle = random.uniform(0, math.pi*2)
        self.alpha = random.randint(30, 120)
        self.size  = random.choice([1,1,1,2])
        self.color = random.choice([(0,255,200),(0,180,140),(255,68,68),(255,176,32)])
    def update(self):
        self.x += math.cos(self.angle)*self.speed
        self.y += math.sin(self.angle)*self.speed
        if self.x<-5 or self.x>WIDTH+5 or self.y<-5 or self.y>HEIGHT+5:
            self.reset()

BG_PARTICLES = [BgParticle() for _ in range(120)]

def draw_radar_bg(surface, radar_angle, glow_surf):
    surface.fill((4, 8, 16))
    glow_surf.fill((0,0,0,0))
    cx, cy = WIDTH//2, (HEIGHT-50)//2
    # Rings
    for r in [60,120,180,240,300]:
        pygame.draw.circle(surface,(0,40,60),(cx,cy),r,1)
    # Radials
    for i in range(12):
        a = i*math.pi/6
        ex,ey = cx+math.cos(a)*340, cy+math.sin(a)*340
        pygame.draw.line(surface,(0,30,50),(cx,cy),(int(ex),int(ey)),1)
    # Sweep
    for off in range(0,40,4):
        aa = radar_angle - math.radians(off)
        sa = (0, max(0,200-off*5), max(0,160-off*4), max(0,180-off*5))
        ex,ey = cx+math.cos(aa)*340, cy+math.sin(aa)*340
        pygame.draw.line(glow_surf, sa, (cx,cy),(int(ex),int(ey)),2)

def draw_glow_text(surface, text, font, x, y, color, center=True, layers=3):
    """Draw text with a multi-layer glow effect."""
    for i in range(layers, 0, -1):
        alpha = int(60 / i)
        size_boost = i * 2
        # Fake glow: render at same pos but with dim color
        glow_c = tuple(min(255, int(c * 0.6)) for c in color)
        surf = font.render(text, True, glow_c)
        if center:
            rx, ry = x - surf.get_width()//2, y - surf.get_height()//2
        else:
            rx, ry = x, y
        # Offset copies for bloom
        for dx, dy in [(-i,0),(i,0),(0,-i),(0,i)]:
            surf2 = font.render(text, True, glow_c)
            surf2.set_alpha(alpha)
            surface.blit(surf2, (rx+dx, ry+dy))
    # Sharp center
    sharp = font.render(text, True, color)
    if center:
        surface.blit(sharp, (x-sharp.get_width()//2, y-sharp.get_height()//2))
    else:
        surface.blit(sharp, (x, y))

def draw_scanlines(surface, alpha=18):
    """Overlay subtle scanlines for CRT feel."""
    for y in range(0, HEIGHT, 3):
        pygame.draw.line(surface, (0,0,0,alpha), (0,y), (WIDTH,y), 1)

class NeonButton:
    """Hover-responsive neon button for menus."""
    def __init__(self, cx, cy, w, h, text, color_primary, color_glow):
        self.rect   = pygame.Rect(cx-w//2, cy-h//2, w, h)
        self.text   = text
        self.cp     = color_primary
        self.cg     = color_glow
        self.hovered = False
        self.pulse   = 0.0

    def update(self, mx, my):
        self.hovered = self.rect.collidepoint(mx, my)
        self.pulse   = (self.pulse + 0.08) % (math.pi*2)

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and
                event.button == 1 and self.rect.collidepoint(event.pos))

    def draw(self, surface):
        t   = pygame.time.get_ticks()
        osc = 0.5 + 0.5*math.sin(self.pulse)

        # outer glow box (multiple layers)
        for expand in ([6,4,2] if self.hovered else [3,1]):
            r = self.rect.inflate(expand*2, expand*2)
            a = int(40*osc) if self.hovered else 20
            glow_s = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
            gc = (*self.cg, a)
            pygame.draw.rect(glow_s, gc, (0,0,r.width,r.height), border_radius=8)
            surface.blit(glow_s, (r.x, r.y))

        # fill
        fill_col = tuple(min(255,int(c*0.25)) for c in self.cp) if not self.hovered else \
                   tuple(min(255,int(c*0.45)) for c in self.cp)
        pygame.draw.rect(surface, fill_col, self.rect, border_radius=6)

        # border
        b_alpha = int(180+75*osc) if self.hovered else 140
        border_s = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(border_s, (*self.cp, b_alpha),
                         (0,0,self.rect.width,self.rect.height), 2, border_radius=6)
        surface.blit(border_s, self.rect.topleft)

        # text
        tc = (255,255,255) if self.hovered else self.cp
        draw_glow_text(surface, self.text, font_btn,
                       self.rect.centerx, self.rect.centery, tc, center=True, layers=2)

        # hover scanline overlay
        if self.hovered:
            hl = pygame.Surface((self.rect.width, self.rect.height//2), pygame.SRCALPHA)
            hl.fill((*self.cp, 15))
            surface.blit(hl, self.rect.topleft)


# ═══════════════════════════════════════════════════════════════
#  7.  START SCREEN
# ═══════════════════════════════════════════════════════════════
def run_start_screen():
    btn_play = NeonButton(WIDTH//2,    HEIGHT//2+30, 220, 52, "[ LAUNCH MISSION ]", JET_COLOR,    JET_COLOR)
    btn_quit = NeonButton(WIDTH//2,    HEIGHT//2+100, 220, 52, "[ ABORT ]",           MISSILE_COLOR, MISSILE_COLOR)
    radar_a  = 0.0
    scan_s   = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    glow_s   = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tick_sin = 0.0

    while True:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if btn_play.is_clicked(event): return "play"
            if btn_quit.is_clicked(event): return "quit"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return "play"

        # Update
        radar_a  = (radar_a + 0.02) % (math.pi*2)
        tick_sin = (tick_sin + 0.05) % (math.pi*2)
        for p in BG_PARTICLES:
            p.update()
        btn_play.update(mx, my)
        btn_quit.update(mx, my)

        # Draw base
        draw_radar_bg(screen, radar_a, glow_s)
        screen.blit(glow_s, (0,0))

        # BG particles
        for p in BG_PARTICLES:
            ps = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(ps, (*p.color, p.alpha), (p.size, p.size), p.size)
            screen.blit(ps, (int(p.x), int(p.y)))

        # Corner decorative brackets
        bc = (0,180,140,120)
        br_s = pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
        sz = 22
        for cx2,cy2 in [(18,18),(WIDTH-18,18),(18,HEIGHT-18),(WIDTH-18,HEIGHT-18)]:
            for dx,dy,ax,ay in [(-sz,-sz,sz,0),(-sz,-sz,0,sz),(sz,-sz,-sz,0),(sz,-sz,0,sz),
                                  (-sz,sz,sz,0),(-sz,sz,0,-sz),(sz,sz,-sz,0),(sz,sz,0,-sz)]:
                pass
        # simple corner lines
        for (x1,y1),(x2,y2) in [((10,10),(40,10)),((10,10),(10,40)),
                                   ((WIDTH-10,10),(WIDTH-40,10)),((WIDTH-10,10),(WIDTH-10,40)),
                                   ((10,HEIGHT-10),(40,HEIGHT-10)),((10,HEIGHT-10),(10,HEIGHT-40)),
                                   ((WIDTH-10,HEIGHT-10),(WIDTH-40,HEIGHT-10)),((WIDTH-10,HEIGHT-10),(WIDTH-10,HEIGHT-40))]:
            pygame.draw.line(screen,(0,180,140),(x1,y1),(x2,y2),2)

        # Title pulse
        title_y  = HEIGHT//2 - 120
        pulse_off = int(math.sin(tick_sin)*3)
        draw_glow_text(screen, "NEURAL", font_title, WIDTH//2, title_y+pulse_off,
                       JET_COLOR, center=True, layers=4)
        draw_glow_text(screen, "EVASION", font_title, WIDTH//2, title_y+78+pulse_off,
                       (0,200,160), center=True, layers=4)

        # Subtitle
        sub_a = int(160+95*math.sin(tick_sin))
        sub_s = font_sub.render("HIERARCHICAL GNN  ·  CROSS-THREAT ATTENTION", True, (0,180,140))
        sub_s.set_alpha(sub_a)
        screen.blit(sub_s, (WIDTH//2-sub_s.get_width()//2, title_y+155))

        # Divider line
        div_w = int(300+80*math.sin(tick_sin*0.7))
        pygame.draw.line(screen, (0,140,110), (WIDTH//2-div_w//2, title_y+178),
                         (WIDTH//2+div_w//2, title_y+178), 1)

        # Controls legend below buttons
        ctrl_lines = [
            ("H", "Homing missile", JET_COLOR),
            ("B", "Ballistic (predictive)", (255,100,255)),
            ("C", "Cluster swarm  (6s cd)", MISSILE_COLOR),
        ]
        leg_y = HEIGHT//2 + 165
        for i,(key,desc,col) in enumerate(ctrl_lines):
            ks  = font_btn.render(f" {key} ", True, (10,20,30))
            bg  = pygame.Surface((ks.get_width(), ks.get_height()), pygame.SRCALPHA)
            bg.fill((*col,220))
            screen.blit(bg, (WIDTH//2-140, leg_y+i*22))
            screen.blit(ks, (WIDTH//2-140, leg_y+i*22))
            ds = font_sm.render(desc, True, (100,160,140))
            screen.blit(ds, (WIDTH//2-140+ks.get_width()+8, leg_y+i*22+2))

        # Version tag
        vs = font_tiny.render("v2.0  |  GNN + ATTENTION  |  5 MANEUVERS", True, (30,70,60))
        screen.blit(vs, (WIDTH//2-vs.get_width()//2, HEIGHT-22))

        # Buttons
        btn_play.draw(screen)
        btn_quit.draw(screen)

        # Scanlines on top
        draw_scanlines(screen)
        pygame.display.flip()
        clock.tick(FPS)


# ═══════════════════════════════════════════════════════════════
#  8.  GAME OVER SCREEN
# ═══════════════════════════════════════════════════════════════
def run_gameover_screen(user_won, stats):
    btn_again = NeonButton(WIDTH//2, HEIGHT//2+60,  220, 52, "[ PLAY AGAIN ]",  JET_COLOR,    JET_COLOR)
    btn_exit  = NeonButton(WIDTH//2, HEIGHT//2+125, 220, 52, "[ EXIT ]",          MISSILE_COLOR, MISSILE_COLOR)
    radar_a   = 0.0
    glow_s    = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tick_sin  = 0.0
    enter_t   = pygame.time.get_ticks()

    if user_won:
        headline = "JET DESTROYED"
        sub_line = "YOU WIN — AI ELIMINATED"
        h_color  = MISSILE_COLOR
        s_color  = (200, 80, 80)
    else:
        headline = "BASES OBLITERATED"
        sub_line = "AI WINS — HUMANITY FALLS"
        h_color  = JET_COLOR
        s_color  = (0, 180, 140)

    while True:
        mx, my = pygame.mouse.get_pos()
        elapsed = pygame.time.get_ticks() - enter_t

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if elapsed > 800:   # brief lockout so accidental click doesn't skip
                if btn_again.is_clicked(event): return "play"
                if btn_exit.is_clicked(event):  return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and elapsed > 800: return "play"
                if event.key == pygame.K_ESCAPE: return "quit"

        radar_a  = (radar_a + 0.025) % (math.pi*2)
        tick_sin = (tick_sin + 0.06) % (math.pi*2)
        for p in BG_PARTICLES: p.update()
        btn_again.update(mx, my)
        btn_exit.update(mx, my)

        draw_radar_bg(screen, radar_a, glow_s)
        screen.blit(glow_s, (0,0))

        for p in BG_PARTICLES:
            ps = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(ps, (*p.color, p.alpha), (p.size, p.size), p.size)
            screen.blit(ps, (int(p.x), int(p.y)))

        # Dark overlay
        ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        a_ov = min(160, int((elapsed/600)*160))
        ov.fill((0,0,0,a_ov))
        screen.blit(ov, (0,0))

        # Headline
        scale = min(1.0, elapsed/400.0)
        pulse = int(math.sin(tick_sin)*4)
        draw_glow_text(screen, headline, font_result, WIDTH//2, HEIGHT//2-90+pulse,
                       h_color, center=True, layers=4)
        draw_glow_text(screen, sub_line, font_sub, WIDTH//2, HEIGHT//2-28,
                       s_color, center=True, layers=2)

        # Stats panel
        stat_y = HEIGHT//2 + 5
        stat_entries = [
            (f"MISSILES FIRED : {stats.get('fired',0)}",  (80,160,140)),
            (f"FLARES DEPLOYED: {stats.get('flares',0)}", (80,130,100)),
            (f"MANEUVERS USED : {stats.get('maneuvers',0)}", (60,100,90)),
        ]
        for i,(txt,col) in enumerate(stat_entries):
            ss = font_hud.render(txt, True, col)
            screen.blit(ss, (WIDTH//2-ss.get_width()//2, stat_y+i*18))

        # Hint
        if elapsed > 900:
            hint_a = int(80+80*math.sin(tick_sin))
            hs = font_sm.render("R — Play Again  |  ESC — Exit", True, (60,100,90))
            hs.set_alpha(hint_a)
            screen.blit(hs, (WIDTH//2-hs.get_width()//2, HEIGHT-22))

        if elapsed > 800:
            btn_again.draw(screen)
            btn_exit.draw(screen)

        draw_scanlines(screen)
        pygame.display.flip()
        clock.tick(FPS)


# ═══════════════════════════════════════════════════════════════
#  9.  MAIN GAME LOOP
# ═══════════════════════════════════════════════════════════════
def reset_game():
    jet  = Jet(WIDTH//2, HEIGHT//2)
    missiles, ai_missiles, flares, explosions, shockwaves = [], [], [], [], []
    launcher_homing    = Launcher(100, HEIGHT-50, MISSILE_COLOR)
    launcher_ballistic = Launcher(WIDTH-100, HEIGHT-50, (255,100,255))
    return jet, missiles, ai_missiles, flares, explosions, shockwaves, launcher_homing, launcher_ballistic

def draw_polar_hud(surface, jet_heading, candidate_scores, cx, cy, r=48):
    """Polar chart of candidate headings — spoke length = safety, color = danger."""
    if not candidate_scores:
        return
    s = pygame.Surface((r*2+20, r*2+20), pygame.SRCALPHA)
    ox, oy = r+10, r+10

    # Background rings
    for ring_r in [r//3, r*2//3, r]:
        pygame.draw.circle(s, (0, 180, 140, 18), (ox, oy), ring_r, 1)
    pygame.draw.circle(s, (0, 180, 140, 35), (ox, oy), 4)  # center dot

    max_d = max(d for _, d in candidate_scores) + 1e-5
    min_d = min(d for _, d in candidate_scores)

    for angle, danger in candidate_scores:
        norm = (danger - min_d) / (max_d - min_d + 1e-5)
        spoke_len = int((1.0 - norm) * r)  # safe = long spoke
        # Screen angle: subtract pi/2 because pygame y-axis is flipped
        screen_angle = angle - math.pi / 2
        ex = int(ox + math.cos(screen_angle) * spoke_len)
        ey = int(oy + math.sin(screen_angle) * spoke_len)
        col_r = int(norm * 255)
        col_g = int((1.0 - norm) * 200)
        alpha = 160 + int(norm * 80)
        width = 2 if danger == min_d else 1
        pygame.draw.line(s, (col_r, col_g, 40, alpha), (ox, oy), (ex, ey), width)
        # Dot at tip
        pygame.draw.circle(s, (col_r, col_g, 40, alpha), (ex, ey), 2)

    # Chosen heading arrow (bright teal)
    ba = jet_heading - math.pi / 2
    bx = int(ox + math.cos(ba) * r)
    by = int(oy + math.sin(ba) * r)
    pygame.draw.line(s, (0, 255, 200, 230), (ox, oy), (bx, by), 2)
    pygame.draw.circle(s, (0, 255, 200, 230), (bx, by), 3)

    surface.blit(s, (cx - ox, cy - oy))

def run_game():
    jet, missiles, ai_missiles, flares, explosions, shockwaves, lh, lb = reset_game()
    glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # Start GNN thread fresh each game
    gnn_thread = threading.Thread(target=gnn_inference_worker, daemon=True)
    gnn_thread.start()

    frame_count = 0
    game_over   = False
    user_won    = False
    game_over_time = 0

    radar_angle = 0.0
    current_hud_attention = []
    current_hud_danger    = 0.0
    current_candidate_scores = []
    current_edges = []
    show_gnn_overlay = False

    # FSM state
    ai_maneuver    = "NORMAL"
    maneuver_timer = 0
    cobra_cooldown = 0
    cluster_cooldown = 0
    general_cooldown = 0
    maneuver_memory  = {}
    global_gnn_heading = 0.0
    global_gnn_flare   = False
    gnn_recommended_maneuver = "NORMAL"

    # Stats tracking
    stats = {'fired': 0, 'flares': 0, 'maneuvers': 0}

    def spawn_flares(n, arc_start, arc_deg, speed_boost=2.0):
        angle_step = math.radians(arc_deg) / max(n-1, 1)
        for i in range(n):
            angle = arc_start + i * angle_step
            flares.append(FlareParticle(jet.x, jet.y, angle, jet.speed + speed_boost))
        stats['flares'] += n

    def predict_intercept_flare(m):
        """
        Computes angle+speed to throw a flare into a missile's predicted path.
        Uses time-of-closest-approach math — flare arrives at the same point
        the missile will be at, making the missile lock onto the flare instead.
        """
        dx  = m.x  - jet.x
        dy  = m.y  - jet.y
        rvx = m.vx - jet.vx   # missile vel relative to jet
        rvy = m.vy - jet.vy
        rel_spd_sq = rvx*rvx + rvy*rvy + 1e-5
        # Time of closest approach (when missile is nearest to jet's current pos)
        t = -(dx*rvx + dy*rvy) / rel_spd_sq
        t = max(5.0, min(50.0, t))   # clamp 5–50 frames
        # Predicted missile world position at time t
        target_x = m.x + m.vx * t
        target_y = m.y + m.vy * t
        # Angle and speed from jet to that prediction
        angle = math.atan2(target_y - jet.y, target_x - jet.x)
        dist  = math.hypot(target_x - jet.x, target_y - jet.y)
        # 1.4x boost so flare beats the missile to the intercept point
        speed = min(max(dist / t * 1.4, 5.0), 14.0)
        return angle, speed
    
    running = True
    while running:
        frame_count += 1
        if cluster_cooldown > 0: cluster_cooldown -= 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state_queue.put(None)
                return "quit"
            if not game_over:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h and lh.can_fire():
                        lh.fire()
                        missiles.append(HomingMissile(lh.x, lh.y, jet, jet.x, jet.y))
                        stats['fired'] += 1
                    elif event.key == pygame.K_b and lb.can_fire():
                        lb.fire()
                        missiles.append(BallisticMissile(lb.x, lb.y, jet.x, jet.y))
                        stats['fired'] += 1
                    elif event.key == pygame.K_c and cluster_cooldown == 0 and lh.alive:
                        cluster_cooldown = 360
                        base_a = math.atan2(jet.y-lh.y, jet.x-lh.x)
                        offsets      = [-20, -10,  0,  10,  20]
                        speeds       = [3.5, 3.7, 3.9, 4.1, 4.3]
                        turn_rates   = [0.06, 0.05, 0.04, 0.03, 0.025]
                        # Each missile gets a gentle persistent curve to hold formation spread
                        lat_biases   = [-0.014, -0.007, 0.0, 0.007, 0.014]
                        for i,(off,spd,tr,lb_val) in enumerate(zip(offsets,speeds,turn_rates,lat_biases)):
                            ra  = math.radians(off)
                            tx2 = lh.x + math.cos(base_a+ra)*120
                            ty2 = lh.y + math.sin(base_a+ra)*120
                            m   = HomingMissile(lh.x, lh.y, jet, tx2, ty2)
                            m.speed = spd; m.max_speed = spd
                            m.turn_rate = tr; m.spawn_delay = i*3
                            m.lateral_bias = lb_val
                            missiles.append(m)
                        stats['fired'] += 5
                    elif event.key == pygame.K_ESCAPE:
                        state_queue.put(None)
                        return "menu"
                    elif event.key == pygame.K_g:
                        show_gnn_overlay = not show_gnn_overlay

        if not game_over:
            # ── Push state to GNN every 6 frames ──────────────────────
            if frame_count % 6 == 0 and state_queue.empty():
                state_queue.put({
                    'jet':     {'pos':[jet.x,jet.y],'vel':[jet.vx,jet.vy],'heading':jet.heading},
                    'missiles':[{'pos':[m.x,m.y],'vel':[m.vx,m.vy],'type':getattr(m,'mtype',0)} for m in missiles],
                    'flares':  [{'pos':[f.x,f.y],'vel':[f.vx,f.vy]} for f in flares]
                })

            if not decision_queue.empty():
                (global_gnn_heading, global_gnn_flare, current_hud_attention,
                current_hud_danger, current_edges, current_candidate_scores,
                gnn_recommended_maneuver) = decision_queue.get()

            # ── Cooldown ticks ─────────────────────────────────────────
            if cobra_cooldown   > 0: cobra_cooldown   -= 1
            if general_cooldown > 0: general_cooldown -= 1
            # general_cooldown removed — let the AI re-enter maneuvers immediately

            # ── Wall proximity ─────────────────────────────────────────
            d_left  = jet.x
            d_right = WIDTH  - jet.x
            d_top   = jet.y
            d_bot   = (HEIGHT-50) - jet.y
            wx = min(d_left,  d_right)
            wy = min(d_top,   d_bot)

            if   d_left  < d_right and d_left  < d_top and d_left  < d_bot: weh = 0.0
            elif d_right < d_left  and d_right < d_top and d_right < d_bot: weh = math.pi
            elif d_top   < d_bot:                                             weh = math.pi/2
            else:                                                             weh = -math.pi/2

            htw = False
            if d_left   < 110 and jet.vx < -1.0: htw = True
            if d_right  < 110 and jet.vx >  1.0: htw = True
            if d_top    < 110 and jet.vy < -1.0: htw = True
            if d_bot    < 110 and jet.vy >  1.0: htw = True

            def heading_to_center():
                return math.atan2(300-jet.y, 400-jet.x)

            # ── HYBRID DISTANCE TRIGGERS (don't rely solely on GNN) ────
            # These fire when missiles are physically close regardless of GNN score
            near = [(m, math.hypot(m.x-jet.x, m.y-jet.y)) for m in missiles]
            close_80  = [(m,d) for m,d in near if d < 80]
            close_150 = [(m,d) for m,d in near if d < 150]
            close_200 = [(m,d) for m,d in near if d < 200]

            # Detect head-on: missile velocity dot (jet→missile vector) < 0
            head_on = False
            for m,d in close_80:
                dx,dy = m.x-jet.x, m.y-jet.y
                if m.vx*dx + m.vy*dy < 0:
                    head_on = True; break

            # ── FSM TRANSITIONS ────────────────────────────────────────
            #  Priority: IMMELMANN > COBRA > FALLING_LEAF > JINKING > BARREL_ROLL

            # 1. IMMELMANN — wall escape, always overrides everything
            if htw and ai_maneuver != "IMMELMANN":
                ai_maneuver    = "IMMELMANN"
                maneuver_timer = 35
                jet.heading    = weh
                stats['maneuvers'] += 1

            elif ai_maneuver == "NORMAL" and general_cooldown == 0:
                near = [(m, math.hypot(m.x-jet.x, m.y-jet.y)) for m in missiles]
                rec  = gnn_recommended_maneuver   # GNN already computed this

                if rec == "COBRA" and cobra_cooldown == 0 and wx > 140 and wy > 140:
                    # Confirm head-on before committing to COBRA
                    close80 = [m for m,d in near if d < 100]
                    if close80:
                        ai_maneuver = "COBRA"; maneuver_timer = 40
                        cobra_cooldown = 360; maneuver_memory['start_speed'] = jet.speed
                        stats['maneuvers'] += 1

                elif rec == "FALLING_LEAF" and wx > 100 and wy > 100 and len(missiles) >= 1:
                    ai_maneuver = "FALLING_LEAF"; maneuver_timer = 90
                    stats['maneuvers'] += 1

                elif rec == "JINKING" and wx > 80 and wy > 80 and len(missiles) >= 1:
                    ai_maneuver = "JINKING"; maneuver_timer = 120
                    stats['maneuvers'] += 1

                elif rec == "BARREL_ROLL" and wx > 150 and wy > 150 and len(missiles) == 1:
                    ai_maneuver = "BARREL_ROLL"; maneuver_timer = 72
                    stats['maneuvers'] += 1

                # Geometry safety net: if GNN says NORMAL but a missile is <80px, force COBRA
                elif rec == "NORMAL" and cobra_cooldown == 0 and wx > 140 and wy > 140:
                    close80 = [m for m,d in near if d < 80]
                    if close80:
                        ai_maneuver = "COBRA"; maneuver_timer = 40
                        cobra_cooldown = 360; maneuver_memory['start_speed'] = jet.speed
                        stats['maneuvers'] += 1

            # ── MANEUVER EXECUTION ─────────────────────────────────────
            if ai_maneuver == "IMMELMANN":
                jet.speed = jet.max_speed * 1.5
                diff = normalize_angle(weh - jet.heading)
                jet.heading += diff * 0.12
                maneuver_timer -= 1
                if maneuver_timer <= 0:
                    jet.heading  = heading_to_center()
                    ai_maneuver  = "NORMAL"

            elif ai_maneuver == "COBRA":
                if maneuver_timer > 20:
                    # Phase 1: hard decelerate + pitch perpendicular to missile approach
                    jet.speed = max(0.3, jet.speed - maneuver_memory['start_speed'] / 20)
                    if missiles:
                        tm = min(missiles, key=lambda m: math.hypot(m.x-jet.x, m.y-jet.y))
                        # Two perpendicular options to missile velocity vector
                        perp1 = math.atan2(-tm.vx, tm.vy)
                        perp2 = math.atan2(tm.vx, -tm.vy)
                        # Pick the one that moves the jet farther from the missile
                        d1 = math.hypot(jet.x+math.cos(perp1)*8 - tm.x,
                                        jet.y+math.sin(perp1)*8 - tm.y)
                        d2 = math.hypot(jet.x+math.cos(perp2)*8 - tm.x,
                                        jet.y+math.sin(perp2)*8 - tm.y)
                        target_h = perp1 if d1 > d2 else perp2
                        diff = normalize_angle(target_h - jet.heading)
                        jet.heading += diff * 0.22   # fast pitch to perpendicular
                elif maneuver_timer == 20:
                    # Phase 2: peak decel — deploy full ring of intercept flares
                    for miss in missiles:
                        angle, speed = predict_intercept_flare(miss)
                        for spread in [-0.25, 0.0, 0.25]:
                            flares.append(FlareParticle(jet.x, jet.y, angle+spread,
                                                        speed * (1.0 - abs(spread)*0.2)))
                            stats['flares'] += 1
                    # 360° emergency ring too
                    for i in range(8):
                        flares.append(FlareParticle(jet.x, jet.y,
                                                    i*(2*math.pi/8), 5.0))
                        stats['flares'] += 1
                else:
                    # Phase 3: re-accelerate, flee direction away from missile
                    if missiles:
                        tm = min(missiles, key=lambda m: math.hypot(m.x-jet.x, m.y-jet.y))
                        flee_h = math.atan2(jet.y-tm.y, jet.x-tm.x)
                        diff   = normalize_angle(flee_h - jet.heading)
                        jet.heading += diff * 0.25
                    jet.speed = min(jet.max_speed*1.4,
                                   jet.speed + jet.max_speed*1.4/20)
                maneuver_timer -= 1
                if maneuver_timer <= 0:
                    ai_maneuver  = "NORMAL"
                    general_cooldown = 60

            elif ai_maneuver == "FALLING_LEAF":
                jet.speed = jet.max_speed * 0.35
                fc = 90 - maneuver_timer
                jet.heading += math.sin(fc*0.35) * 0.35
                # 8 flares every 18 frames in 180° rear arc
                if fc % 18 == 0:
                    spawn_flares(10, jet.heading+math.pi-math.pi/2, 180, 4.0)
                maneuver_timer -= 1
                if maneuver_timer <= 0 or not missiles or current_hud_danger < 0.2:
                    ai_maneuver = "NORMAL"

            elif ai_maneuver == "JINKING":
                jet.speed = jet.max_speed * 0.51
                if maneuver_timer % 10 == 0 and missiles:
                    # Zigzag PERPENDICULAR to nearest missile — not random
                    nm   = min(missiles, key=lambda m: math.hypot(m.x-jet.x, m.y-jet.y))
                    m_angle = math.atan2(nm.vy, nm.vx)
                    # Alternate left/right perpendicular each jink
                    side = 1 if (maneuver_timer // 10) % 2 == 0 else -1
                    prop = m_angle + side * math.pi / 2
                    pvx  = math.cos(prop) * jet.speed
                    pvy  = math.sin(prop) * jet.speed
                    # Wall safety — try other side if this one leads to wall
                    if (jet.x+pvx*20 < 80 or jet.x+pvx*20 > 720 or
                            jet.y+pvy*20 < 80 or jet.y+pvy*20 > 470):
                        prop = m_angle - side * math.pi / 2
                    jet.heading = prop
                # Intercept flares during jinking — aimed at each missile
                if maneuver_timer % 20 == 0:
                    for miss in missiles[:3]:   # cap at 3 missiles for perf
                        angle, speed = predict_intercept_flare(miss)
                        flares.append(FlareParticle(jet.x, jet.y, angle,       speed))
                        flares.append(FlareParticle(jet.x, jet.y, angle - 0.2, speed*0.85))
                        stats['flares'] += 5
                maneuver_timer -= 1
                if maneuver_timer <= 0:
                    ai_maneuver  = "NORMAL"
                    general_cooldown = 45

            elif ai_maneuver == "BARREL_ROLL":
                jet.speed = jet.max_speed
                if wx > 100 and wy > 100:
                    jet.heading += 0.15
                else:
                    jet.heading = heading_to_center()
                    ai_maneuver = "NORMAL"
                # 2 side flares every 24 frames during roll
                if maneuver_timer % 24 == 0:
                    spawn_flares(6, jet.heading+math.pi/2-math.radians(20), 40, 4.0)
                maneuver_timer -= 1
                if maneuver_timer <= 0:
                    ai_maneuver = "NORMAL"

            elif ai_maneuver == "NORMAL":
                jet.speed = jet.max_speed

                if missiles:
                    # Geometric flee: always point away from the nearest missile
                    nearest_m  = min(missiles, key=lambda m: math.hypot(m.x-jet.x, m.y-jet.y))
                    nearest_d  = math.hypot(nearest_m.x-jet.x, nearest_m.y-jet.y)
                    flee_h     = math.atan2(jet.y - nearest_m.y, jet.x - nearest_m.x)

                    if global_gnn_heading != 0.0:
                        # Blend: flee dominates when missile is close (<150 px), GNN wins when far
                        flee_w = max(0.0, min(1.0, (200.0 - nearest_d) / 200.0)) ** 1.5
                        gnn_w  = 1.0 - flee_w
                        diff_gnn  = normalize_angle(global_gnn_heading - jet.heading)
                        diff_flee = normalize_angle(flee_h - jet.heading)
                        combined  = diff_gnn * gnn_w + diff_flee * flee_w
                    else:
                        combined = normalize_angle(flee_h - jet.heading)

                    # Smooth steer — never snap, let physics carry momentum
                    jet.heading += combined * 0.18

                elif global_gnn_heading != 0.0:
                    diff = normalize_angle(global_gnn_heading - jet.heading)
                    jet.heading += diff * 0.18

                # Intercept-aimed flares — throw toward missile's predicted position
                if (global_gnn_flare and missiles
                        and pygame.time.get_ticks()-jet.last_flare_time > 2000
                        and wx > 80 and wy > 80):
                    jet.last_flare_time = pygame.time.get_ticks()
                    for miss in missiles:
                        angle, speed = predict_intercept_flare(miss)
                        flares.append(FlareParticle(jet.x, jet.y, angle,        speed))
                        flares.append(FlareParticle(jet.x, jet.y, angle - 0.20, speed*0.85))
                        flares.append(FlareParticle(jet.x, jet.y, angle + 0.20, speed*0.85))
                        stats['flares'] += 3
                    global_gnn_flare = False

            # ── JET COUNTER-ATTACK (every 4 s, targets nearest alive launcher) ──
            if pygame.time.get_ticks()-jet.last_attack_time > 4000:
                targets = [l for l in [lh, lb] if l.alive]
                if targets:
                    # Prefer the launcher that is closer so the missile is harder to dodge
                    tgt = min(targets, key=lambda l: math.hypot(l.x-jet.x, l.y-jet.y))
                    ai_missiles.append(JetMissile(jet.x, jet.y, tgt.x, tgt.y))
                jet.last_attack_time = pygame.time.get_ticks()  # always reset so timer stays rhythmic

            # ── PHYSICS UPDATE ────────────────────────────────────────
            jet.update()

            # Jet wall collision (lethal)
            if (jet.x < jet.radius or jet.x > WIDTH-jet.radius or
                    jet.y < jet.radius or jet.y > HEIGHT-50-jet.radius):
                for _ in range(50): explosions.append(ExplosionParticle(jet.x,jet.y,JET_COLOR))
                shockwaves.append(Shockwave(jet.x,jet.y,JET_COLOR))
                game_over=True; user_won=True; game_over_time=pygame.time.get_ticks()
                state_queue.put(None)

            # Flares
            flares = [f for f in flares if not f.is_dead() or f.update() or True]
            alive_f = []
            for f in flares:
                f.update()
                if not f.is_dead(): alive_f.append(f)
            flares = alive_f

            # Explosions
            explosions = [e for e in explosions if not e.update() or not e.is_dead()]
            alive_e = []
            for e in explosions:
                e.update()
                if not e.is_dead(): alive_e.append(e)
            explosions = alive_e

            # AI missiles
            alive_aim = []
            for aim in ai_missiles:
                aim.update()
                for lnc in [lh, lb]:
                    if lnc.alive and math.hypot(lnc.x-aim.x, lnc.y-aim.y) < 15+aim.radius:
                        lnc.alive = False; aim.dead = True
                        for _ in range(30): explosions.append(ExplosionParticle(lnc.x,lnc.y,(255,80,50)))
                        shockwaves.append(Shockwave(lnc.x,lnc.y,(255,80,50)))
                        break
                if not aim.is_dead(): alive_aim.append(aim)
            ai_missiles = alive_aim

            # Player missiles
            alive_m = []
            for m in missiles:
                m.update()
                # Jet hit
                if math.hypot(jet.x-m.x, jet.y-m.y) < jet.radius+m.radius:
                    game_over=True; user_won=True; game_over_time=pygame.time.get_ticks()
                    state_queue.put(None)
                    for _ in range(50): explosions.append(ExplosionParticle(jet.x,jet.y,JET_COLOR))
                    shockwaves.append(Shockwave(jet.x,jet.y,JET_COLOR))
                    continue
                # Flare hit
                hit_flare = False
                for f in flares:
                    if math.hypot(f.x-m.x, f.y-m.y) < f.radius+m.radius:
                        f.spawn_time -= 5000; m.dead = True; hit_flare = True
                        for _ in range(5): explosions.append(ExplosionParticle(f.x,f.y,FLARE_COLOR))
                        shockwaves.append(Shockwave(f.x,f.y,FLARE_COLOR))
                        break
                if hit_flare: continue
                # Missile-missile intercept
                hit_aim = False
                for aim in ai_missiles:
                    if math.hypot(aim.x-m.x, aim.y-m.y) < aim.radius+m.radius:
                        m.dead=True; aim.dead=True; hit_aim=True
                        for _ in range(20): explosions.append(ExplosionParticle(aim.x,aim.y,(255,255,255)))
                        shockwaves.append(Shockwave(aim.x,aim.y,(255,255,255)))
                        break
                if hit_aim: continue
                if not m.is_dead(): alive_m.append(m)
            missiles = alive_m

            # ── CLUSTER MISSILE REPULSION ─────────────────────────────
            # Homing missiles push each other apart so they never merge/overlap
            REPULSE_DIST  = 26
            REPULSE_FORCE = 0.20
            homing = [m for m in missiles if isinstance(m, HomingMissile) and not m.dead]
            for i in range(len(homing)):
                for j in range(i+1, len(homing)):
                    m1, m2 = homing[i], homing[j]
                    dx = m2.x - m1.x; dy = m2.y - m1.y
                    d  = math.hypot(dx, dy)
                    if 0 < d < REPULSE_DIST:
                        strength = (REPULSE_DIST - d) / REPULSE_DIST * REPULSE_FORCE
                        away = math.atan2(dy, dx)   # m1→m2 direction
                        # m1 turns away, m2 turns away in opposite direction
                        m1.heading = normalize_angle(m1.heading - strength * math.pi * (1 - d/REPULSE_DIST))
                        m2.heading = normalize_angle(m2.heading + strength * math.pi * (1 - d/REPULSE_DIST))

            # AI win: both launchers dead
            if not lh.alive and not lb.alive:
                game_over=True; user_won=False; game_over_time=pygame.time.get_ticks()
                state_queue.put(None)

        # ── RENDER ────────────────────────────────────────────────────
        radar_angle = (radar_angle + 0.025) % (math.pi*2)
        draw_radar_bg(screen, radar_angle, glow_surface)
        screen.blit(glow_surface, (0,0))

        lh.draw(screen); lb.draw(screen)

        if not game_over:
            jet.draw(glow_surface, ai_maneuver)
        for f in flares:   f.draw(glow_surface)
        for m in missiles: m.draw(glow_surface)
        for aim in ai_missiles: aim.draw(glow_surface)
        for e in explosions: e.draw(screen)
        for s in shockwaves:
            s.update(); s.draw(glow_surface)
        shockwaves[:] = [s for s in shockwaves if not s.is_dead()]
        screen.blit(glow_surface, (0,0))

        if show_gnn_overlay and current_edges:
            mags = [e['mag'] for e in current_edges]
            max_mag = max(mags) + 1e-5
            gnn_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            for e in current_edges:
                norm = e['mag'] / max_mag
                alpha = int(30 + norm * 180)
                width = max(1, int(norm * 3))
                r = int(norm * 255)
                g = int((1 - norm) * 180)
                x1, y1 = int(e['p1'][0]), int(e['p1'][1])
                x2, y2 = int(e['p2'][0]), int(e['p2'][1])
                pygame.draw.line(gnn_surf, (r, g, 50, alpha), (x1, y1), (x2, y2), width)
            screen.blit(gnn_surf, (0, 0))
            lbl = font_sm.render("GNN GRAPH [G]", True, (0, 200, 160))
            screen.blit(lbl, (14, HEIGHT - 95))

        # ── HUD bar ───────────────────────────────────────────────────
        hy = HEIGHT-50
        pygame.draw.rect(screen,(6,14,26),(0,hy,WIDTH,50))
        pygame.draw.line(screen,(0,180,140),(0,hy),(WIDTH,hy),1)

        # Threat meter
        dc = min(max(current_hud_danger,0.0),1.0)
        mx_p,my_p = 14, hy+10
        pygame.draw.rect(screen,(20,30,40),(mx_p,my_p,160,12),0,4)
        fw = int(dc*160)
        if fw > 0:
            pygame.draw.rect(screen,(int(dc*255),int((1-dc)*200),0),(mx_p,my_p,fw,12),0,4)
        pygame.draw.rect(screen,(0,180,140),(mx_p,my_p,160,12),1,4)
        screen.blit(font_hud.render("THREAT",True,(0,200,160)),(mx_p,my_p+16))

        # Maneuver label
        mc = {"NORMAL":(80,180,140),"IMMELMANN":(255,220,50),"COBRA":(255,80,80),
              "FALLING_LEAF":(255,160,30),"JINKING":(80,160,255),"BARREL_ROLL":(180,80,255)}.get(ai_maneuver,(255,255,255))
        ml = font_hud.render(f"[ {ai_maneuver} ]", True, mc)
        screen.blit(ml,(WIDTH//2-ml.get_width()//2, hy+8))

        # Controls
        cl = font_sm.render("H:Homing  B:Ballistic  C:Cluster  ESC:Menu",True,(60,100,120))
        screen.blit(cl,(WIDTH-cl.get_width()-14, hy+28))

        # Attention panel
        px2,py2,pw,ph = WIDTH-190, 12, 178, 130
        ps2 = pygame.Surface((pw,ph),pygame.SRCALPHA); ps2.fill((6,14,30,210))
        screen.blit(ps2,(px2,py2))
        pygame.draw.rect(screen,(0,140,110),(px2,py2,pw,ph),1,6)
        screen.blit(font_hud.render("AI ATTENTION",True,(0,220,170)),(px2+8,py2+7))
        if current_hud_attention:
            nb   = min(len(current_hud_attention),8)
            bw   = max(14,(pw-16)//nb-4)
            bmh  = 60; bby = py2+100
            for i in range(nb):
                wv = current_hud_attention[i]
                bx2 = px2+8+i*(bw+4); bh2 = int(wv*bmh)
                pygame.draw.rect(screen,(15,30,45),(bx2,bby-bmh,bw,bmh),0,3)
                rb=int(wv*255); gb=int((1-wv)*220)
                bc=(rb,gb,30)
                if wv>0.80 and current_hud_danger>0.55 and (pygame.time.get_ticks()//150)%2==0:
                    bc=(255,255,255)
                if bh2>0: pygame.draw.rect(screen,bc,(bx2,bby-bh2,bw,bh2),0,3)
                pygame.draw.rect(screen,(0,160,120),(bx2,bby-bmh,bw,bmh),1,3)
                screen.blit(font_tiny.render(f"M{i+1}",True,(120,180,160)),(bx2+bw//2-7,bby+2))
                screen.blit(font_tiny.render(f"{int(wv*100)}%",True,(80,140,120)),(bx2-1,bby-bh2-14))
        else:
            screen.blit(font_sm.render("no threats",True,(40,80,70)),(px2+45,py2+65))

        # Polar counterfactual heading chart — top-left floating panel
        polar_cx, polar_cy = 60, 65
        draw_polar_hud(screen, jet.heading, current_candidate_scores, polar_cx, polar_cy, r=44)
        lbl_p = font_tiny.render("HEADING RISK", True, (0, 160, 120))
        screen.blit(lbl_p, (polar_cx - lbl_p.get_width()//2, polar_cy + 50))

        # Game over flash
        if game_over:
            ov = pygame.Surface((WIDTH,HEIGHT-50),pygame.SRCALPHA)
            ov.fill((0,0,0,120)); screen.blit(ov,(0,0))
            if pygame.time.get_ticks()-game_over_time > 2200:
                state_queue.put(None)
                # drain queue so next game starts fresh
                while not decision_queue.empty(): decision_queue.get()
                return run_gameover_screen(user_won, stats)

        draw_scanlines(screen)
        pygame.display.flip()
        clock.tick(FPS)

    state_queue.put(None)
    return "quit"


# ═══════════════════════════════════════════════════════════════
#  10.  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result = run_start_screen()
    while result == "play":
        # Fresh queues for each game session
        while not state_queue.empty():    state_queue.get()
        while not decision_queue.empty(): decision_queue.get()
        result = run_game()

    pygame.quit()