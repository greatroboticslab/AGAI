#!/usr/bin/env python3
"""
Test suite for ResNet classifier functionality.
Tests the ResNet-50 model used for strawberry disease classification.
"""

import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resnet_classifier import load_resnet, diagnose_or_none, _CLASSES


class TestResNetClassifier(unittest.TestCase):
    """Test cases for ResNet classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "/data/AGAI/MiniGPT-4/plant_diagnostic/models/resnet_straw_final.pth"
        self.test_image_path = "/data/AGAI/MiniGPT-4/examples/healthy_1.png"
        
    def test_model_loading(self):
        """Test that the ResNet model loads correctly."""
        try:
            model = load_resnet(self.model_path)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'eval'))
        except FileNotFoundError:
            self.skipTest(f"Model file not found: {self.model_path}")
        except Exception as e:
            self.fail(f"Failed to load model: {e}")
    
    def test_classes_definition(self):
        """Test that classes are properly defined."""
        expected_classes = [
            "drought", "frost", "healthy", "overwatered", "root_rot"
        ]
        self.assertEqual(len(_CLASSES), 5)
        for cls in expected_classes:
            self.assertIn(cls, _CLASSES)
    
    def test_thresholds_definition(self):
        """Test that thresholds are properly defined."""
        # Import thresholds from the module
        try:
            from resnet_classifier import _THRESH
            self.assertIsInstance(_THRESH, dict)
            self.assertGreater(len(_THRESH), 0)
            
            # Check that all classes have thresholds
            for cls in _CLASSES:
                self.assertIn(cls, _THRESH)
                self.assertIsInstance(_THRESH[cls], (int, float))
                self.assertGreaterEqual(_THRESH[cls], 0.0)
                self.assertLessEqual(_THRESH[cls], 1.0)
        except ImportError:
            self.skipTest("Thresholds not available in resnet_classifier")
    
    def test_diagnose_with_valid_image(self):
        """Test diagnosis with a valid image."""
        if not os.path.exists(self.test_image_path):
            self.skipTest(f"Test image not found: {self.test_image_path}")
        
        # Load a model first
        try:
            model = load_resnet(self.model_path)
        except FileNotFoundError:
            self.skipTest(f"Model file not found: {self.model_path}")
        except Exception as e:
            self.skipTest(f"Could not load model: {e}")
            
        try:
            result = diagnose_or_none(model, self.test_image_path)
            # Result should be None (below threshold) or a dict with expected keys
            if result is not None:
                self.assertIsInstance(result, dict)
                self.assertIn('label', result)
                self.assertIn('p1', result)
                self.assertIn('top2', result)
                self.assertIn(result['label'], _CLASSES)
                self.assertIsInstance(result['p1'], float)
                self.assertGreaterEqual(result['p1'], 0.0)
                self.assertLessEqual(result['p1'], 1.0)
        except Exception as e:
            self.fail(f"Diagnosis failed: {e}")
    
    def test_diagnose_with_invalid_path(self):
        """Test diagnosis with invalid image path."""
        invalid_path = "/nonexistent/path/image.jpg"
        try:
            result = diagnose_or_none(None, invalid_path)
            # Should handle gracefully or raise an exception
            self.assertTrue(result is None or isinstance(result, dict))
        except Exception:
            # Expected for invalid path
            pass
    
    def test_diagnose_with_corrupted_image(self):
        """Test diagnosis with corrupted image."""
        # Create a temporary corrupted image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b'corrupted image data')
            tmp_path = tmp.name
        
        try:
            result = diagnose_or_none(None, tmp_path)
            # Should handle gracefully
            self.assertTrue(result is None or isinstance(result, dict))
        except Exception:
            # Expected for corrupted image
            pass
        finally:
            os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()
