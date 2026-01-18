#!/usr/bin/env python3
"""
Calibration Module - All 4 Methods
Implements: Platt Scaling, Isotonic Regression, Temperature Scaling, Beta Calibration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

# ============================================================================
# Base Calibrator Class
# ============================================================================

class BaseCalibrator:
    """Base class for all calibrators"""
    
    def __init__(self):
        self.is_fitted = False
    
    def fit(self, probs, labels):
        """Fit calibrator on validation data
        
        Args:
            probs: Uncalibrated probabilities (numpy array)
            labels: True labels (numpy array)
        """
        raise NotImplementedError
    
    def transform(self, probs):
        """Transform probabilities using fitted calibrator
        
        Args:
            probs: Uncalibrated probabilities (numpy array)
        
        Returns:
            Calibrated probabilities (numpy array)
        """
        raise NotImplementedError
    
    def fit_transform(self, probs, labels):
        """Fit and transform in one step"""
        self.fit(probs, labels)
        return self.transform(probs)

# ============================================================================
# 1. Platt Scaling (Logistic Calibration)
# ============================================================================

class PlattScaling(BaseCalibrator):
    """Platt Scaling: fits a logistic regression on uncalibrated probabilities"""
    
    def __init__(self):
        super().__init__()
        self.lr = LogisticRegression()
    
    def fit(self, probs, labels):
        # Convert probabilities to logits
        probs = np.clip(probs, 1e-7, 1 - 1e-7)  # Avoid log(0)
        logits = np.log(probs / (1 - probs))
        
        # Fit logistic regression
        self.lr.fit(logits.reshape(-1, 1), labels)
        self.is_fitted = True
    
    def transform(self, probs):
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))
        calibrated = self.lr.predict_proba(logits.reshape(-1, 1))[:, 1]
        return calibrated

# ============================================================================
# 2. Isotonic Regression
# ============================================================================

class IsotonicCalibration(BaseCalibrator):
    """Isotonic Regression: non-parametric monotonic calibration"""
    
    def __init__(self):
        super().__init__()
        self.iso_reg = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, probs, labels):
        self.iso_reg.fit(probs, labels)
        self.is_fitted = True
    
    def transform(self, probs):
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        return self.iso_reg.transform(probs)

# ============================================================================
# 3. Temperature Scaling (Neural Network Calibration)
# ============================================================================

class TemperatureScaling(BaseCalibrator):
    """Temperature Scaling: single parameter that scales logits"""
    
    def __init__(self):
        super().__init__()
        self.temperature = 1.0
    
    def fit(self, probs, labels, max_iter=50):
        """Fit temperature using NLL loss"""
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))
        
        # Convert to torch
        logits_tensor = torch.FloatTensor(logits)
        labels_tensor = torch.FloatTensor(labels)
        
        # Optimize temperature
        def nll_loss(t):
            t = float(t[0])
            scaled_logits = logits_tensor / t
            probs_scaled = torch.sigmoid(scaled_logits)
            loss = -torch.mean(
                labels_tensor * torch.log(probs_scaled + 1e-7) +
                (1 - labels_tensor) * torch.log(1 - probs_scaled + 1e-7)
            )
            return loss.item()
        
        # Find optimal temperature
        result = minimize(nll_loss, [1.0], method='Nelder-Mead', 
                         options={'maxiter': max_iter})
        self.temperature = float(result.x[0])
        self.is_fitted = True
    
    def transform(self, probs):
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))
        scaled_logits = logits / self.temperature
        calibrated = sigmoid(scaled_logits)
        return calibrated

# ============================================================================
# 4. Beta Calibration
# ============================================================================

class BetaCalibration(BaseCalibrator):
    """Beta Calibration: uses beta distribution with 3 parameters"""
    
    def __init__(self):
        super().__init__()
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
    
    def fit(self, probs, labels, max_iter=100):
        """Fit beta parameters using MLE"""
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        
        def beta_calibration_loss(params):
            a, b, c = params
            if a <= 0 or b <= 0:
                return 1e10
            
            # Beta distribution mapping
            calibrated = np.clip(probs ** a / (probs ** a + (1 - probs) ** b), 1e-7, 1 - 1e-7)
            calibrated = calibrated * (1 - c) + c / 2
            
            # Negative log likelihood
            loss = -np.mean(
                labels * np.log(calibrated) + 
                (1 - labels) * np.log(1 - calibrated)
            )
            return loss
        
        # Optimize
        result = minimize(beta_calibration_loss, [1.0, 1.0, 0.0], 
                         method='Nelder-Mead',
                         options={'maxiter': max_iter})
        
        self.a, self.b, self.c = result.x
        self.is_fitted = True
    
    def transform(self, probs):
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        calibrated = probs ** self.a / (probs ** self.a + (1 - probs) ** self.b)
        calibrated = calibrated * (1 - self.c) + self.c / 2
        return np.clip(calibrated, 0, 1)

# ============================================================================
# Calibration Metrics
# ============================================================================

def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins for ECE calculation
    
    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def compute_brier_score(probs, labels):
    """Compute Brier Score
    
    Args:
        probs: Predicted probabilities
        labels: True labels
    
    Returns:
        Brier score (lower is better)
    """
    return np.mean((probs - labels) ** 2)

def get_reliability_diagram_data(probs, labels, n_bins=10):
    """Get data for reliability diagram
    
    Returns:
        bin_centers, bin_accuracies, bin_confidences, bin_counts
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            count_in_bin = np.sum(in_bin)
            
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(count_in_bin)
    
    return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)

# ============================================================================
# Factory Function
# ============================================================================

def get_calibrator(method='temperature'):
    """Get calibrator by name
    
    Args:
        method: One of 'platt', 'isotonic', 'temperature', 'beta'
    
    Returns:
        Calibrator instance
    """
    calibrators = {
        'platt': PlattScaling,
        'isotonic': IsotonicCalibration,
        'temperature': TemperatureScaling,
        'beta': BetaCalibration
    }
    
    if method not in calibrators:
        raise ValueError(f"Unknown calibration method: {method}. Choose from {list(calibrators.keys())}")
    
    return calibrators[method]()
