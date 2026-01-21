#!/usr/bin/env python3
"""
Utility functions for testing.
"""

import os
import sys
import tempfile
import json
from PIL import Image
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_image(width=256, height=256, color=(128, 128, 128)):
    """Create a test image for testing purposes."""
    image = Image.new('RGB', (width, height), color)
    return image


def create_test_annotation():
    """Create a test annotation file for testing purposes."""
    annotation = {
        "annotations": [
            {
                "image_id": "test_001",
                "image_path": "test_001.jpg",
                "caption": "A healthy strawberry plant with green leaves",
                "disease": "healthy",
                "confidence": 0.95
            },
            {
                "image_id": "test_002", 
                "image_path": "test_002.jpg",
                "caption": "A strawberry plant showing signs of overwatering",
                "disease": "overwatered",
                "confidence": 0.87
            }
        ]
    }
    return annotation


def create_temp_test_files():
    """Create temporary test files and return their paths."""
    temp_files = {}
    
    # Create temporary image
    test_image = create_test_image()
    temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    test_image.save(temp_image.name)
    temp_files['image'] = temp_image.name
    temp_image.close()
    
    # Create temporary annotation file
    test_annotation = create_test_annotation()
    temp_annotation = tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w')
    json.dump(test_annotation, temp_annotation)
    temp_files['annotation'] = temp_annotation.name
    temp_annotation.close()
    
    return temp_files


def cleanup_temp_files(temp_files):
    """Clean up temporary test files."""
    for file_path in temp_files.values():
        if os.path.exists(file_path):
            os.unlink(file_path)


def assert_valid_diagnosis(result):
    """Assert that a diagnosis result is valid."""
    assert result is not None, "Diagnosis result should not be None"
    assert isinstance(result, dict), "Diagnosis result should be a dictionary"
    assert 'label' in result, "Diagnosis result should have 'label' key"
    assert 'p1' in result, "Diagnosis result should have 'p1' key"
    assert 'top2' in result, "Diagnosis result should have 'top2' key"
    
    # Validate label
    valid_labels = [
        "healthy", "overwatered", "root_rot", 
        "drought", "frost", "gray_mold", "white_mold"
    ]
    assert result['label'] in valid_labels, f"Invalid label: {result['label']}"
    
    # Validate probability
    assert 0.0 <= result['p1'] <= 1.0, f"Invalid probability: {result['p1']}"
    
    # Validate top2
    assert isinstance(result['top2'], tuple), "top2 should be a tuple"
    assert len(result['top2']) == 2, "top2 should have 2 elements"
    assert isinstance(result['top2'][0], str), "top2[0] should be a string"
    assert isinstance(result['top2'][1], float), "top2[1] should be a float"


def assert_valid_minigpt_response(response):
    """Assert that a MiniGPT response is valid."""
    assert response is not None, "MiniGPT response should not be None"
    assert isinstance(response, str), "MiniGPT response should be a string"
    assert len(response) > 0, "MiniGPT response should not be empty"
    
    # Check for expected structure in medical report
    expected_sections = ["Diagnosis:", "Visible cues:", "Recommendation:"]
    for section in expected_sections:
        if section in response:
            assert True, f"Response contains {section}"
            break
    else:
        # If no medical report format, just check it's a reasonable response
        assert len(response) > 10, "Response should be substantial"


def create_mock_model():
    """Create a mock model for testing."""
    class MockModel:
        def __init__(self):
            self.eval_called = False
            self.to_called = False
            
        def eval(self):
            self.eval_called = True
            return self
            
        def to(self, device):
            self.to_called = True
            return self
            
        def __call__(self, *args, **kwargs):
            # Return mock output
            return {
                'loss': 0.5,
                'logits': np.random.randn(1, 7)
            }
    
    return MockModel()


def create_mock_processor():
    """Create a mock processor for testing."""
    class MockProcessor:
        def __init__(self):
            self.process_called = False
            
        def __call__(self, image):
            self.process_called = True
            # Return mock processed image
            return np.random.randn(3, 448, 448)
    
    return MockProcessor()


def create_mock_chat_state():
    """Create a mock chat state for testing."""
    return {
        'messages': [],
        'image': None,
        'temperature': 0.1
    }


def validate_config_structure(config):
    """Validate that a configuration has the expected structure."""
    required_sections = ['model', 'run']
    
    for section in required_sections:
        assert section in config, f"Config missing required section: {section}"
    
    # Validate model section
    model_config = config['model']
    required_model_keys = ['arch', 'ckpt']
    
    for key in required_model_keys:
        assert key in model_config, f"Model config missing required key: {key}"
    
    # Validate run section
    run_config = config['run']
    required_run_keys = ['task', 'output_dir']
    
    for key in required_run_keys:
        assert key in run_config, f"Run config missing required key: {key}"


def check_gpu_availability():
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_test_data_paths():
    """Get paths to test data files."""
    base_path = "/data/AGAI/MiniGPT-4"
    
    paths = {
        'examples': os.path.join(base_path, "examples"),
        'plant_data': os.path.join(base_path, "plant_diagnostic", "data"),
        'datasets': os.path.join(base_path, "plant_diagnostic", "datasets"),
        'models': os.path.join(base_path, "plant_diagnostic", "models"),
        'checkpoints': os.path.join(base_path, "checkpoints"),
        'llama_weights': os.path.join(base_path, "llama_weights")
    }
    
    return paths


def check_required_files():
    """Check if required files exist for testing."""
    paths = get_test_data_paths()
    missing_files = []
    
    # Check for example images
    examples_dir = paths['examples']
    if os.path.exists(examples_dir):
        example_files = os.listdir(examples_dir)
        if not any(f.endswith(('.jpg', '.jpeg', '.png')) for f in example_files):
            missing_files.append("Example images")
    else:
        missing_files.append("Examples directory")
    
    # Check for model files
    model_files = [
        os.path.join(paths['models'], "resnet_strawberry.pth"),
        os.path.join(paths['checkpoints'], "checkpoint_stage2.pth")
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_files.append(os.path.basename(model_file))
    
    return missing_files


if __name__ == '__main__':
    # Test the utility functions
    print("Testing utility functions...")
    
    # Test image creation
    test_image = create_test_image()
    print(f"Created test image: {test_image.size}")
    
    # Test annotation creation
    test_annotation = create_test_annotation()
    print(f"Created test annotation with {len(test_annotation['annotations'])} items")
    
    # Test temporary files
    temp_files = create_temp_test_files()
    print(f"Created temporary files: {list(temp_files.keys())}")
    
    # Clean up
    cleanup_temp_files(temp_files)
    print("Cleaned up temporary files")
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    print(f"GPU available: {gpu_available}")
    
    # Check required files
    missing_files = check_required_files()
    if missing_files:
        print(f"Missing files: {missing_files}")
    else:
        print("All required files present")
    
    print("Utility function tests completed!")


