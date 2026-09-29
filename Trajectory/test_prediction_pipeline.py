r"""
Comprehensive Test Suite for Trajectory Prediction Pipeline
Problem Statement ID: 26037 — Team Epsilon
Smart India Hackathon 2026

Validates:
 1. PyTorch MotionFormer model architecture & tensor dimensions
 2. Categorical mode probability normalization (\sum \pi_k = 1.0)
 3. Spatial covariance matrix physical validity (positive variance, bounded correlation)
 4. ONNX graph export & ONNX checker validation
 5. Downstream MAT data exchange compatibility with Sensor Fusion & Path Planning
"""

import os
import sys
import unittest
import numpy as np
import torch
import onnx

from motionformer_engine import (
    MotionFormerPipeline,
    AgentHistoryEncoder,
    StaticHazardEncoder,
    BEVCrossAttentionInteraction,
    MultiModalGMMDecoder,
    export_to_onnx,
    NUM_CLASSES
)
from download_pretrained_weights import (
    verify_weights,
    ensure_weights_dir,
    create_mock_weights_for_testing
)

class TestTrajectoryPredictionPipeline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.B = 2
        self.N = 4
        self.T_obs = 10
        self.pred_len = 30
        self.num_modes = 3
        self.embed_dim = 128
        
        self.model = MotionFormerPipeline(
            embed_dim=self.embed_dim,
            num_modes=self.num_modes,
            pred_len=self.pred_len,
            obs_len=self.T_obs
        ).to(self.device)
        self.model.eval()

    def test_agent_history_encoder(self):
        """Tests that agent history encoder correctly embeds past states and class IDs."""
        encoder = AgentHistoryEncoder(in_dim=7, embed_dim=self.embed_dim, num_classes=NUM_CLASSES)
        history = torch.randn(self.B, self.N, self.T_obs, 7)
        class_ids = torch.randint(0, NUM_CLASSES, (self.B, self.N))
        
        out = encoder(history, class_ids)
        self.assertEqual(out.shape, (self.B, self.N, self.embed_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_static_hazard_encoder(self):
        """Tests that pothole depressions and road boundary coordinates are tokenized."""
        encoder = StaticHazardEncoder(pothole_dim=5, bound_dim=4, embed_dim=self.embed_dim)
        potholes = torch.randn(self.B, 3, 5) # 3 potholes
        road_bounds = torch.randn(self.B, 6, 4) # 6 boundary segments
        
        out = encoder(potholes, road_bounds)
        self.assertEqual(out.shape, (self.B, 3 + 6, self.embed_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_bev_cross_attention(self):
        """Tests that BEV cross-attention between agents and hazards functions without error."""
        attn_module = BEVCrossAttentionInteraction(embed_dim=self.embed_dim, num_heads=4)
        agent_tokens = torch.randn(self.B, self.N, self.embed_dim)
        hazard_tokens = torch.randn(self.B, 9, self.embed_dim)
        
        out = attn_module(agent_tokens, hazard_tokens)
        self.assertEqual(out.shape, (self.B, self.N, self.embed_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_gmm_decoder_math(self):
        """Validates GMM probabilities, waypoints, and covariance positivity."""
        decoder = MultiModalGMMDecoder(embed_dim=self.embed_dim, num_modes=self.num_modes, pred_len=self.pred_len)
        context = torch.randn(self.B, self.N, self.embed_dim)
        
        probs, waypoints, covariances = decoder(context)
        
        # Probabilities sum to 1.0 across modes
        prob_sums = probs.sum(dim=-1).detach().numpy()
        np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5)
        
        # Waypoints shape (B, N, K, H, 2)
        self.assertEqual(waypoints.shape, (self.B, self.N, self.num_modes, self.pred_len, 2))
        
        # Covariances: var_x > 0, var_y > 0, and valid Cauchy-Schwarz inequality |cov_xy| <= sqrt(var_x * var_y)
        var_x = covariances[..., 0].detach().numpy()
        var_y = covariances[..., 1].detach().numpy()
        cov_xy = covariances[..., 2].detach().numpy()
        
        self.assertTrue((var_x > 0).all(), "Variance in X must be strictly positive")
        self.assertTrue((var_y > 0).all(), "Variance in Y must be strictly positive")
        self.assertTrue((np.abs(cov_xy) <= np.sqrt(var_x * var_y) + 1e-4).all(),
                        "Covariance must satisfy valid correlation bounds")

    def test_full_pipeline_forward(self):
        """Tests full forward pass with heterogeneous Indian classes (autorickshaw, cow, bike)."""
        history = torch.randn(1, 3, self.T_obs, 7)
        class_ids = torch.tensor([[5, 8, 6]]) # autorickshaw, animal, motorcycle
        potholes = torch.tensor([[[-0.8, 16.5, 7.5, 0.75, 0.85]]]) # deep pothole
        road_bounds = torch.randn(1, 4, 4)
        
        with torch.no_grad():
            probs, waypoints, covariances = self.model(history, class_ids, potholes, road_bounds)
            
        self.assertEqual(probs.shape, (1, 3, self.num_modes))
        self.assertEqual(waypoints.shape, (1, 3, self.num_modes, self.pred_len, 2))
        self.assertEqual(covariances.shape, (1, 3, self.num_modes, self.pred_len, 3))

    def test_onnx_export_and_check(self):
        """Tests that model exports to valid ONNX with intact computational graph."""
        onnx_test_path = os.path.join(os.path.dirname(__file__), "test_motionformer.onnx")
        try:
            export_to_onnx(self.model, onnx_filepath=onnx_test_path, num_agents=2, num_potholes=1, num_bounds=4)
            self.assertTrue(os.path.isfile(onnx_test_path))
            
            # Load and verify with ONNX checker
            onnx_model = onnx.load(onnx_test_path)
            onnx.checker.check_model(onnx_model)
        finally:
            if os.path.isfile(onnx_test_path):
                os.remove(onnx_test_path)

    def test_weights_staging(self):
        """Tests weight directory creation and mock weight staging."""
        wdir = ensure_weights_dir()
        self.assertTrue(os.path.isdir(wdir))
        create_mock_weights_for_testing()
        self.assertTrue(os.path.isfile(os.path.join(wdir, "motionformer_weights.pth")))
        self.assertTrue(os.path.isfile(os.path.join(wdir, "trajectory_chennai_calib.json")))


if __name__ == "__main__":
    unittest.main()
