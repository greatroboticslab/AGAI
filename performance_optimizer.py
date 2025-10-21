#!/usr/bin/env python3
"""
Performance optimization utilities for the Plant Diagnostic System.
Includes memory optimization, inference speed improvements, and caching.
"""

import torch
import torch.nn as nn
import time
import psutil
import gc
from functools import lru_cache
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Performance optimization utilities for the Plant Diagnostic System."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimization_settings = {
            'use_amp': True,
            'use_compile': True,
            'use_cache': True,
            'batch_size': 1,
            'max_memory_usage': 0.8  # 80% of available GPU memory
        }
    
    def optimize_model_loading(self, model, model_path: str) -> nn.Module:
        """Optimize model loading for faster inference."""
        logger.info("Optimizing model loading...")
        
        # Load model with optimizations
        if hasattr(torch, 'compile') and self.optimization_settings['use_compile']:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Set model to eval mode
        model.eval()
        
        # Move to device
        model = model.to(self.device)
        
        # Enable optimizations
        if self.device.type == 'cuda':
            # Enable TensorFloat-32 for faster training on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("Flash attention enabled")
            except Exception:
                logger.warning("Flash attention not available")
        
        return model
    
    def optimize_memory_usage(self):
        """Optimize memory usage for better performance."""
        logger.info("Optimizing memory usage...")
        
        if self.device.type == 'cuda':
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory = int(total_memory * self.optimization_settings['max_memory_usage'])
            torch.cuda.set_per_process_memory_fraction(
                self.optimization_settings['max_memory_usage']
            )
            
            logger.info(f"GPU memory limit set to {max_memory / 1024**3:.1f} GB")
        
        # Force garbage collection
        gc.collect()
    
    def create_optimized_dataloader(self, dataset, batch_size: int = 1, num_workers: int = 0):
        """Create an optimized DataLoader for faster data loading."""
        from torch.utils.data import DataLoader
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
    
    @lru_cache(maxsize=128)
    def cached_model_inference(self, model, input_tensor_hash: int, model_state_hash: int):
        """Cached model inference for repeated inputs."""
        # This is a placeholder - actual implementation would need to handle
        # the model and input tensor properly
        pass
    
    def benchmark_inference(self, model, input_tensor, num_runs: int = 100):
        """Benchmark inference speed."""
        logger.info(f"Benchmarking inference with {num_runs} runs...")
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        logger.info(f"Average inference time: {avg_time*1000:.2f} ms")
        logger.info(f"Inference FPS: {fps:.2f}")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
            'total_time': end_time - start_time
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / 1024**3
        }
        
        if self.device.type == 'cuda':
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'gpu_memory_cached_gb': torch.cuda.memory_cached() / 1024**3
            })
        
        return stats
    
    def optimize_for_inference(self, model, input_shape: tuple = (1, 3, 448, 448)):
        """Optimize model specifically for inference."""
        logger.info("Optimizing model for inference...")
        
        # Create dummy input for optimization
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Optimize model
        model = self.optimize_model_loading(model, "")
        
        # Benchmark performance
        benchmark_results = self.benchmark_inference(model, dummy_input)
        
        # Get memory stats
        memory_stats = self.get_memory_stats()
        
        logger.info("Optimization complete!")
        logger.info(f"Memory stats: {memory_stats}")
        
        return model, benchmark_results, memory_stats


class InferenceCache:
    """Caching system for inference results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key: str):
        """Get cached result."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached result."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()


def optimize_resnet_inference(model, device: str = "cuda"):
    """Optimize ResNet model for faster inference."""
    logger.info("Optimizing ResNet for inference...")
    
    # Set to eval mode
    model.eval()
    
    # Move to device
    model = model.to(device)
    
    # Enable optimizations
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    return model


def optimize_minigpt_inference(model, device: str = "cuda"):
    """Optimize MiniGPT model for faster inference."""
    logger.info("Optimizing MiniGPT for inference...")
    
    # Set to eval mode
    model.eval()
    
    # Move to device
    model = model.to(device)
    
    # Enable optimizations
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass
    
    return model


def profile_inference_pipeline(model, input_data, num_runs: int = 50):
    """Profile the complete inference pipeline."""
    logger.info("Profiling inference pipeline...")
    
    device = next(model.parameters()).device
    times = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    # Profile
    for i in range(num_runs):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(input_data)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    logger.info(f"Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    logger.info(f"Inference FPS: {1.0/avg_time:.2f}")
    
    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'fps': 1.0 / avg_time,
        'times': times
    }


if __name__ == "__main__":
    # Example usage
    optimizer = PerformanceOptimizer()
    
    # Get memory stats
    stats = optimizer.get_memory_stats()
    print("Memory stats:", stats)
    
    # Optimize memory
    optimizer.optimize_memory_usage()
    
    print("Performance optimization utilities loaded!")

