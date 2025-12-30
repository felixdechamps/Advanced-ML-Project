"""
Script de test pour vérifier l'installation et le setup
Teste tous les composants avant l'entraînement complet
"""

import sys
import torch
import numpy as np

def test_imports():
    """Test que toutes les dépendances sont installées"""
    print("Testing imports...")
    
    try:
        import torch
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import matplotlib
        import seaborn
        import tqdm
        print("✓ All dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def test_pytorch():
    """Test PyTorch et CUDA"""
    print("\nTesting PyTorch...")
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    
    # Test simple tensor operation
    x = torch.randn(2, 3)
    y = x * 2
    print(f"  ✓ Basic tensor operations work")
    
    return True

def test_model_architecture():
    """Test l'architecture du modèle"""
    print("\nTesting model architecture...")
    
    try:
        from models.resnet1d import ResNet1d
        
        # Create model
        # Hannun et al.: single-lead ECG input, 4 classes for PhysioNet 2017
        model = ResNet1d(
            in_channels=1,
            base_filters=32,
            kernel_size=16,
            stride=2,
            n_classes=4,
            dropout_rate=0.2
        )
        
        # Test forward pass
        # Hannun et al.: 30s ECG at 300Hz = 9000 samples
        batch_size = 4
        signal_length = 9000
        x = torch.randn(batch_size, 1, signal_length)
        
        output = model(x)
        
        assert output.shape == (batch_size, 4), f"Expected shape (4, 4), got {output.shape}"
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")
        print(f"  Expected: ~10.5M (Hannun et al.)")
        
        # Hannun et al. report ~10.5M parameters
        if 9_000_000 < total_params < 12_000_000:
            print(f"  ✓ Parameter count in expected range")
        else:
            print(f"  ⚠ Parameter count differs from Hannun et al.")
        
        print(f"  ✓ Model architecture works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test le chargement du dataset"""
    print("\nTesting dataset loading...")
    
    try:
        from data.dataset import PhysioNet2017Dataset
        import os
        
        # Check if data directory exists
        data_dir = './data/physionet2017'
        if not os.path.exists(data_dir):
            print(f"  ⚠ Data directory not found: {data_dir}")
            print(f"  Please download PhysioNet 2017 dataset first")
            print(f"  See README.md for instructions")
            return False
        
        # Check for required files
        ref_file = os.path.join(data_dir, 'REFERENCE.csv')
        if not os.path.exists(ref_file):
            print(f"  ⚠ REFERENCE.csv not found in {data_dir}")
            return False
        
        # Try to load dataset
        dataset = PhysioNet2017Dataset(
            data_dir=data_dir,
            target_length=9000
        )
        
        print(f"  Dataset size: {len(dataset)}")
        
        # PhysioNet 2017: 8,528 samples
        if len(dataset) != 8528:
            print(f"  ⚠ Expected 8,528 samples, got {len(dataset)}")
        
        # Test loading one sample
        signal, label = dataset[0]
        
        assert signal.shape == (9000,), f"Expected shape (9000,), got {signal.shape}"
        assert 0 <= label <= 3, f"Label should be 0-3, got {label}"
        
        print(f"  Sample signal shape: {signal.shape}")
        print(f"  Sample label: {label}")
        print(f"  ✓ Dataset loading works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics():
    """Test les fonctions de calcul de métriques"""
    print("\nTesting metrics...")
    
    try:
        from utils.metrics import ECGMetrics
        
        # Create dummy predictions
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 1, 1, 2, 2])
        y_prob = np.random.rand(8, 4)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
        
        # Calculate metrics
        metrics_calc = ECGMetrics(n_classes=4)
        metrics = metrics_calc.compute_all_metrics(y_true, y_pred, y_prob)
        
        # Check required metrics exist
        required_metrics = [
            'f1_macro', 'f1_weighted', 'f1_per_class',
            'precision_macro', 'recall_macro',
            'auc_macro', 'accuracy'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ✓ Metrics calculation works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_utilities():
    """Test les utilitaires d'entraînement"""
    print("\nTesting training utilities...")
    
    try:
        from utils.training import initialize_weights
        from models.resnet1d import ResNet1d
        
        # Create model
        model = ResNet1d(
            in_channels=1,
            base_filters=32,
            kernel_size=16,
            n_classes=4
        )
        
        # Initialize weights
        # Hannun et al.: "He initialization"
        model = initialize_weights(model)
        
        # Check that weights are initialized (not all zeros)
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert not torch.all(param == 0), f"Weights not initialized: {name}"
                break
        
        print(f"  ✓ Weight initialization works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Training utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ECG CLASSIFICATION - SETUP TEST")
    print("=" * 60)
    print("Testing installation and configuration...")
    print()
    
    tests = [
        ("Imports", test_imports),
        ("PyTorch", test_pytorch),
        ("Model Architecture", test_model_architecture),
        ("Dataset Loading", test_dataset_loading),
        ("Metrics", test_metrics),
        ("Training Utilities", test_training_utilities),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n✓ All tests passed! Ready to train.")
        print("\nNext steps:")
        print("1. python train.py          # Train the model")
        print("2. python evaluate.py       # Evaluate trained model")
        print("3. python lth_ecg.py        # Compress model (after training)")
    else:
        print("\n⚠ Some tests failed. Please fix issues before training.")
        
        if not results["Dataset Loading"]:
            print("\nDataset not found. To download:")
            print("  mkdir -p data/physionet2017")
            print("  cd data/physionet2017")
            print("  wget -r -N -c -np https://physionet.org/files/challenge-2017/1.0.0/training2017/")
    
    return passed_tests == total_tests

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)