"""
MotionFormer & Trajectron++ Hybrid Trajectory Forecasting Engine
Problem Statement ID: 26037 — Team Epsilon
Smart India Hackathon 2026

Non-lane-conditioned, multi-modal Gaussian Mixture Model (GMM) trajectory prediction
with 3D BEV cross-attention between dynamic agents and static road defects (potholes, road edges).
"""

import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 12 India Driving Dataset (IDD) Semantic Classes matching Perception/Sensor Fusion
IDD_CLASSES = [
    "person",          # 0
    "rider",           # 1
    "car",             # 2
    "bus",             # 3
    "truck",           # 4
    "autorickshaw",    # 5
    "motorcycle",      # 6
    "bicycle",         # 7
    "animal",          # 8 (stray cow / dog)
    "traffic sign",    # 9
    "traffic light",   # 10
    "vehicle fallback" # 11 (pushcart / tractor)
]

NUM_CLASSES = len(IDD_CLASSES)


class AgentHistoryEncoder(nn.Module):
    """
    Encodes past trajectory history of surrounding dynamic agents.
    Input shape: (batch_size, num_agents, obs_len, in_dim)
      in_dim: [x, y, vx, vy, speed, heading_sin, heading_cos] (7-dim)
      class_ids: (batch_size, num_agents) (integer 0..11)
    Output shape: (batch_size, num_agents, embed_dim)
    """
    def __init__(self, in_dim=7, embed_dim=128, num_classes=12, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.class_embedding = nn.Embedding(num_classes, 32)
        self.feature_proj = nn.Linear(in_dim + 32, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, history, class_ids):
        # history: (B, N, T_obs, 7)
        B, N, T_obs, F_dim = history.shape
        cls_embed = self.class_embedding(class_ids) # (B, N, 32)
        cls_expanded = cls_embed.unsqueeze(2).expand(B, N, T_obs, 32) # (B, N, T_obs, 32)
        
        feat_in = torch.cat([history, cls_expanded], dim=-1) # (B, N, T_obs, 7 + 32)
        feat_flat = feat_in.view(B * N, T_obs, -1)
        feat_proj = F.relu(self.feature_proj(feat_flat))
        
        gru_out, _ = self.gru(feat_proj) # (B*N, T_obs, hidden_dim*2)
        # Use final step embedding:
        last_step = gru_out[:, -1, :] # (B*N, hidden_dim*2)
        agent_embed = self.out_proj(last_step) # (B*N, embed_dim)
        agent_embed = self.layer_norm(agent_embed).view(B, N, self.embed_dim)
        return agent_embed


class StaticHazardEncoder(nn.Module):
    """
    Encodes static road surface defects (potholes, dips, speed breakers)
    and road boundary coordinates.
    potholes: (batch_size, num_potholes, 5) -> [x, y, depth_cm, radius_m, severity]
    road_bounds: (batch_size, num_bound_points, 4) -> [x_left, y_left, x_right, y_right]
    Output: (batch_size, num_tokens, embed_dim)
    """
    def __init__(self, pothole_dim=5, bound_dim=4, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.pothole_mlp = nn.Sequential(
            nn.Linear(pothole_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.bound_mlp = nn.Sequential(
            nn.Linear(bound_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, potholes, road_bounds):
        # potholes: (B, M_p, 5)
        # road_bounds: (B, M_b, 4)
        p_tokens = self.pothole_mlp(potholes)
        b_tokens = self.bound_mlp(road_bounds)
        hazard_tokens = torch.cat([p_tokens, b_tokens], dim=1) # (B, M_p + M_b, embed_dim)
        return hazard_tokens


class BEVCrossAttentionInteraction(nn.Module):
    """
    Multi-Head Attention Engine operating in metric Bird's-Eye View (BEV).
    1. Agent-to-Agent Self-Attention: Learns gap-filling, following, yielding.
    2. Agent-to-Hazard Cross-Attention: Dynamic agents query static hazards
       to predict causal lateral swerving around potholes or unpaved shoulders.
    """
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.agent_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.hazard_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, agent_tokens, hazard_tokens):
        # 1. Multi-Agent Interaction
        agent_interact, _ = self.agent_self_attn(agent_tokens, agent_tokens, agent_tokens)
        x = self.norm1(agent_tokens + agent_interact)
        
        # 2. Causal Hazard Interaction (Query: agents, Key/Value: hazards)
        hazard_effect, _ = self.hazard_cross_attn(query=x, key=hazard_tokens, value=hazard_tokens)
        x = self.norm2(x + hazard_effect)
        
        # 3. Feedforward Refinement
        x = self.norm3(x + self.ffn(x))
        return x


class MultiModalGMMDecoder(nn.Module):
    """
    Decodes multi-agent context embeddings into K trajectory modes.
    For each mode k in 1..K:
      - Mode categorical probability: pi_k (softmax over K)
      - Mean waypoints: mu_k(t) = [dx(t), dy(t)] for t = 1..H
      - Spatial covariance: [log(sigma_x(t)), log(sigma_y(t)), tanh(rho(t))]
    """
    def __init__(self, embed_dim=128, num_modes=3, pred_len=30):
        super().__init__()
        self.num_modes = num_modes
        self.pred_len = pred_len
        
        # Mode probability head:
        self.prob_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes)
        )
        
        # Trajectory parameters head:
        # Per mode and per timestep: [dx, dy, log_sig_x, log_sig_y, rho] = 5 params
        self.traj_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_modes * pred_len * 5)
        )

    def forward(self, context):
        # context: (B, N, embed_dim)
        B, N, _ = context.shape
        
        # 1. Mode probabilities:
        logits = self.prob_head(context) # (B, N, K)
        probs = F.softmax(logits, dim=-1) # (B, N, K)
        
        # 2. Trajectory waypoints & covariances:
        raw_params = self.traj_head(context) # (B, N, K * H * 5)
        raw_params = raw_params.view(B, N, self.num_modes, self.pred_len, 5)
        
        dx = raw_params[..., 0]
        dy = raw_params[..., 1]
        log_sig_x = torch.clamp(raw_params[..., 2], min=-2.0, max=3.0)
        log_sig_y = torch.clamp(raw_params[..., 3], min=-2.0, max=3.0)
        rho = torch.tanh(raw_params[..., 4]) * 0.95 # keep strictly in (-1, 1)
        
        sig_x = torch.exp(log_sig_x)
        sig_y = torch.exp(log_sig_y)
        
        # Pack predictions:
        # waypoints: (B, N, K, H, 2)
        waypoints = torch.stack([dx, dy], dim=-1)
        # covariances: (B, N, K, H, 3) -> [sig_x^2, sig_y^2, cov_xy]
        cov_xy = rho * sig_x * sig_y
        covariances = torch.stack([sig_x ** 2, sig_y ** 2, cov_xy], dim=-1)
        
        return probs, waypoints, covariances


class MotionFormerPipeline(nn.Module):
    """
    Full End-to-End BEV MotionFormer Architecture for Unstructured Indian Roads.
    Integrates Agent History, Static Road Defects, Cross-Attention, and GMM Decoding.
    """
    def __init__(self, embed_dim=128, num_modes=3, pred_len=30, obs_len=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.pred_len = pred_len
        self.obs_len = obs_len
        
        self.history_encoder = AgentHistoryEncoder(in_dim=7, embed_dim=embed_dim, num_classes=NUM_CLASSES)
        self.hazard_encoder = StaticHazardEncoder(pothole_dim=5, bound_dim=4, embed_dim=embed_dim)
        self.bev_interaction = BEVCrossAttentionInteraction(embed_dim=embed_dim, num_heads=4)
        self.gMM_decoder = MultiModalGMMDecoder(embed_dim=embed_dim, num_modes=num_modes, pred_len=pred_len)
        
        self.init_weights_physically()

    def init_weights_physically(self):
        """
        Initializes prior weights with calibrated physical motion biases:
        Mode 0: Straight / velocity-aligned cruising
        Mode 1: Evasive swerve left / gap-filling
        Mode 2: Evasive swerve right / yield
        This allows the pipeline to run immediately and make sensible predictions
        even before large pre-trained weights are downloaded!
        """
        with torch.no_grad():
            dt = 0.1
            # Base decoder trajectory head bias
            # Format: [dx, dy, log_sig_x, log_sig_y, rho]
            bias = torch.zeros(self.num_modes, self.pred_len, 5)
            for t in range(self.pred_len):
                time_s = (t + 1) * dt
                # Mode 0: Nominal straight forward
                bias[0, t, 0] = 0.0          # lateral offset = 0
                bias[0, t, 1] = time_s        # forward motion unit rate
                bias[0, t, 2] = math.log(0.3 + 0.15 * time_s) # sig_x grows with time
                bias[0, t, 3] = math.log(0.3 + 0.20 * time_s) # sig_y
                bias[0, t, 4] = 0.0
                
                # Mode 1: Swerve Left
                bias[1, t, 0] = -1.5 * (1.0 - math.exp(-0.8 * time_s)) # left swerve up to -1.5m
                bias[1, t, 1] = time_s * 0.95
                bias[1, t, 2] = math.log(0.5 + 0.25 * time_s)
                bias[1, t, 3] = math.log(0.4 + 0.20 * time_s)
                bias[1, t, 4] = -0.2
                
                # Mode 2: Swerve Right / Yield
                bias[2, t, 0] = 1.5 * (1.0 - math.exp(-0.8 * time_s))  # right swerve up to +1.5m
                bias[2, t, 1] = time_s * 0.90
                bias[2, t, 2] = math.log(0.5 + 0.25 * time_s)
                bias[2, t, 3] = math.log(0.4 + 0.20 * time_s)
                bias[2, t, 4] = 0.2
                
            last_linear = self.gMM_decoder.traj_head[-1]
            if last_linear.bias is not None:
                last_linear.bias.data.copy_(bias.view(-1))

    def forward(self, history, class_ids, potholes, road_bounds):
        """
        history:     (B, N, T_obs, 7) [x, y, vx, vy, speed, sin_yaw, cos_yaw]
        class_ids:   (B, N) (int64)
        potholes:    (B, M_p, 5) [x, y, depth_cm, radius_m, severity]
        road_bounds: (B, M_b, 4) [x_left, y_left, x_right, y_right]
        
        Returns:
          probs:       (B, N, K) Mode probabilities summing to 1
          waypoints:   (B, N, K, H, 2) Future trajectory offsets [dx, dy] in meters
          covariances: (B, N, K, H, 3) [var_x, var_y, cov_xy]
        """
        agent_tokens = self.history_encoder(history, class_ids) # (B, N, embed_dim)
        hazard_tokens = self.hazard_encoder(potholes, road_bounds) # (B, M, embed_dim)
        bev_context = self.bev_interaction(agent_tokens, hazard_tokens) # (B, N, embed_dim)
        probs, waypoints, covariances = self.gMM_decoder(bev_context)
        return probs, waypoints, covariances


def export_to_onnx(model, onnx_filepath="motionformer.onnx", num_agents=4, num_potholes=3, num_bounds=10):
    """
    Exports the trained MotionFormer model to standard ONNX format for
    direct ingestion into MATLAB Deep Learning Toolbox via importNetworkFromONNX.
    """
    model.eval()
    obs_len = model.obs_len
    
    dummy_history = torch.randn(1, num_agents, obs_len, 7, dtype=torch.float32)
    dummy_classes = torch.randint(0, NUM_CLASSES, (1, num_agents), dtype=torch.int64)
    dummy_potholes = torch.randn(1, num_potholes, 5, dtype=torch.float32)
    dummy_bounds = torch.randn(1, num_bounds, 4, dtype=torch.float32)
    
    dynamic_axes = {
        'history': {0: 'batch_size', 1: 'num_agents'},
        'class_ids': {0: 'batch_size', 1: 'num_agents'},
        'potholes': {0: 'batch_size', 1: 'num_potholes'},
        'road_bounds': {0: 'batch_size', 1: 'num_bound_points'},
        'probs': {0: 'batch_size', 1: 'num_agents'},
        'waypoints': {0: 'batch_size', 1: 'num_agents'},
        'covariances': {0: 'batch_size', 1: 'num_agents'}
    }
    
    torch.onnx.export(
        model,
        (dummy_history, dummy_classes, dummy_potholes, dummy_bounds),
        onnx_filepath,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['history', 'class_ids', 'potholes', 'road_bounds'],
        output_names=['probs', 'waypoints', 'covariances'],
        dynamic_axes=dynamic_axes
    )
    print(f">> Successfully exported MotionFormer to ONNX: {onnx_filepath}")
    return onnx_filepath


if __name__ == "__main__":
    print("==================================================================")
    print("  MotionFormer & Trajectron++ Hybrid Prediction Engine Initialized")
    print("  Team Epsilon | Problem Statement ID: 26037")
    print("==================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionFormerPipeline(embed_dim=128, num_modes=3, pred_len=30, obs_len=10).to(device)
    model.eval()
    
    # Test batch
    B, N, T_obs = 1, 3, 10
    M_p, M_b = 2, 8
    history = torch.randn(B, N, T_obs, 7).to(device)
    class_ids = torch.tensor([[5, 8, 6]]).to(device) # autorickshaw, animal, motorcycle
    potholes = torch.randn(B, M_p, 5).to(device)
    road_bounds = torch.randn(B, M_b, 4).to(device)
    
    with torch.no_grad():
        probs, waypoints, covariances = model(history, class_ids, potholes, road_bounds)
        
    print(f">> Forward pass successful:")
    print(f"   Mode probabilities shape : {probs.shape} (Sum: {probs.sum(dim=-1).cpu().numpy()})")
    print(f"   Future waypoints shape   : {waypoints.shape} (Horizon: 3.0s, 30 steps)")
    print(f"   Covariance ellipses shape: {covariances.shape}")
    
    onnx_path = os.path.join(os.path.dirname(__file__), "motionformer.onnx")
    export_to_onnx(model, onnx_path)
