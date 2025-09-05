import torch
import time
import sys
sys.path.append('/notebooks/PMamba/experiments')
from models.motion import Motion

def benchmark_weight_select():
    print("Benchmarking weight_select function...")
    
    # Create larger sample data for benchmarking
    B, C, TP, K = 8, 7, 128, 16
    position = torch.randn(B, C, TP, K)
    
    # Make first 3 channels realistic coordinates
    position[:, 0, :, :] = position[:, 0, :, :] * 10  # x coordinates
    position[:, 1, :, :] = position[:, 1, :, :] * 10  # y coordinates
    position[:, 2, :, :] = position[:, 2, :, :] * 10  # z coordinates
    
    print(f"Input shape: {position.shape}")
    
    # Warm up
    for _ in range(5):
        _ = Motion.weight_select(position, 64)
    
    # Benchmark
    num_runs = 100
    start_time = time.time()
    
    for _ in range(num_runs):
        indices = Motion.weight_select(position, 64)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    print(f"Average execution time over {num_runs} runs: {avg_time*1000:.2f} ms")
    print(f"Throughput: {num_runs/(end_time-start_time):.2f} calls/second")
    
    # Test correctness of indices
    indices = Motion.weight_select(position, 64)
    print(f"Output indices shape: {indices.shape}")
    print(f"Indices valid range: [{indices.min().item()}, {indices.max().item()}]")
    print(f"All indices in valid range: {torch.all((indices >= 0) & (indices < TP)).item()}")

def compare_strategies():
    print("\nComparing different selection strategies...")
    
    # Create sample data
    B, C, TP, K = 4, 6, 64, 8
    position = torch.randn(B, C, TP, K)
    
    # Make data more realistic
    position[:, 0, :, :] = position[:, 0, :, :] * 10  # x coordinates
    position[:, 1, :, :] = position[:, 1, :, :] * 10  # y coordinates
    position[:, 2, :, :] = position[:, 2, :, :] * 10  # z coordinates
    
    # Add some variance to feature channels
    position[:, 3:, :, :] = position[:, 3:, :, :] * 2
    
    topk = 32
    
    print("Testing different weighting strategies:")
    
    # Strategy 1: Distance only (original)
    distances = torch.max(torch.sum(position[:, :3] ** 2, dim=1), dim=-1)[0]
    dist_min = distances.min(dim=-1, keepdim=True)[0]
    dist_max = distances.max(dim=-1, keepdim=True)[0]
    dist_range = dist_max - dist_min
    dist_range = torch.where(dist_range == 0, torch.ones_like(dist_range), dist_range)
    normalized_distances = (distances - dist_min) / dist_range
    _, idx_dist_only = torch.topk(normalized_distances, min(topk, normalized_distances.shape[-1]), -1, largest=True, sorted=False)
    
    # Strategy 2: Distance + Variance (your improvement)
    if position.shape[1] > 3:
        feature_var = torch.var(position[:, 3:], dim=-1).mean(dim=1)
        var_min = feature_var.min(dim=-1, keepdim=True)[0]
        var_max = feature_var.max(dim=-1, keepdim=True)[0]
        var_range = var_max - var_min
        var_range = torch.where(var_range == 0, torch.ones_like(var_range), var_range)
        normalized_variance = (feature_var - var_min) / var_range
    else:
        normalized_variance = torch.zeros_like(normalized_distances)
    
    weights_var = 0.7 * normalized_distances + 0.3 * normalized_variance
    _, idx_dist_var = torch.topk(weights_var, min(topk, weights_var.shape[-1]), -1, largest=True, sorted=False)
    
    # Strategy 3: Distance + Variance + Spatial Isolation (our new improvement)
    # Compute spatial isolation
    coords = position[:, :3, :, 0]  # (B, 3, T*P)
    coords_expanded_1 = coords.unsqueeze(-1)  # (B, 3, T*P, 1)
    coords_expanded_2 = coords.unsqueeze(-2)  # (B, 3, 1, T*P)
    pairwise_distances = torch.sqrt(torch.sum((coords_expanded_1 - coords_expanded_2) ** 2, dim=1) + 1e-8)
    diag_mask = torch.eye(pairwise_distances.shape[-1], device=pairwise_distances.device).unsqueeze(0)
    pairwise_distances = pairwise_distances + diag_mask * 1e9
    spatial_isolation = torch.min(pairwise_distances, dim=-1)[0]
    iso_min = spatial_isolation.min(dim=-1, keepdim=True)[0]
    iso_max = spatial_isolation.max(dim=-1, keepdim=True)[0]
    iso_range = iso_max - iso_min
    iso_range = torch.where(iso_range == 0, torch.ones_like(iso_range), iso_range)
    normalized_isolation = (spatial_isolation - iso_min) / iso_range
    
    weights_iso = 0.4 * normalized_distances + 0.3 * normalized_variance + 0.3 * normalized_isolation
    _, idx_dist_var_iso = torch.topk(weights_iso, min(topk, weights_iso.shape[-1]), -1, largest=True, sorted=False)
    
    # Strategy 4: Our new implementation (using the actual function)
    idx_new_impl = Motion.weight_select(position, topk)
    
    print(f"Distance-only strategy sample indices:     {idx_dist_only[0, :10]}")
    print(f"Distance+Variance strategy sample indices: {idx_dist_var[0, :10]}")
    print(f"Distance+Variance+Isolation sample indices:{idx_dist_var_iso[0, :10]}")
    print(f"New implementation sample indices:         {idx_new_impl[0, :10]}")
    
    # Check if results are different (they should be)
    diff_var = not torch.equal(idx_dist_only, idx_dist_var)
    diff_iso = not torch.equal(idx_dist_var, idx_dist_var_iso)
    diff_new = not torch.equal(idx_dist_var_iso, idx_new_impl)
    
    print(f"\nStrategies produce different results:")
    print(f"  Distance vs Distance+Variance: {diff_var}")
    print(f"  Distance+Variance vs +Isolation: {diff_iso}")
    print(f"  Isolation vs New Implementation: {diff_new}")

if __name__ == "__main__":
    benchmark_weight_select()
    compare_strategies()