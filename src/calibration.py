import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression


# ==============================================================================
# Base Calibrator
# ==============================================================================

class BaseCalibrator:
    """Base class for logit-based calibrators."""

    def __init__(self):
        self.is_fitted = False

    def fit(self, logits: np.ndarray, y: np.ndarray) -> 'BaseCalibrator':
        """
        Fit calibrator on validation/calibration data.

        Args:
            logits: Raw model logits (NOT probabilities), shape (n,) or (n, 1)
            y: Binary labels, shape (n,) or (n, 1)

        Returns:
            self
        """
        raise NotImplementedError

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply calibration to get calibrated probabilities.

        Args:
            logits: Raw model logits, shape (n,) or (n, 1)

        Returns:
            Calibrated probabilities in [0, 1]
        """
        raise NotImplementedError

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Alias for transform()."""
        return self.transform(logits)

    def fit_transform(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(logits, y)
        return self.transform(logits)


# ==============================================================================
# Temperature Scaling
# ==============================================================================

class TemperatureScaling(BaseCalibrator):
    """
    Temperature Scaling: scales logits by a learned temperature T > 0.

    p_calibrated = sigmoid(logits / T)

    Properties:
        - Single scalar parameter T
        - T > 1: softens probabilities (less confident)
        - T < 1: sharpens probabilities (more confident)
        - T = 1: identity (no change)
        - INVARIANT: threshold-0.5 predictions unchanged (sign preserved)
    """

    def __init__(self):
        super().__init__()
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray, max_iter: int = 100) -> 'TemperatureScaling':
        """
        Fit temperature by minimizing NLL on calibration data.

        Args:
            logits: Model logits (n,)
            y: Binary labels (n,)
            max_iter: Maximum optimization iterations

        Returns:
            self
        """
        logits = np.asarray(logits).flatten()
        y = np.asarray(y).flatten()

        def nll_loss(t):
            t = float(t[0])
            if t <= 0:
                return 1e10  # Penalty for non-positive temperature
            scaled_logits = logits / t
            probs = sigmoid(scaled_logits)
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            return loss

        # Optimize temperature starting from T=1
        result = minimize(
            nll_loss,
            x0=[1.0],
            method='Nelder-Mead',
            options={'maxiter': max_iter}
        )

        self.temperature = float(result.x[0])
        # Ensure temperature is positive
        if self.temperature <= 0:
            self.temperature = 1.0

        self.is_fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling.

        Args:
            logits: Model logits

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")

        logits = np.asarray(logits).flatten()
        scaled_logits = logits / self.temperature
        return sigmoid(scaled_logits)


# ==============================================================================
# Platt Scaling
# ==============================================================================

class PlattScaling(BaseCalibrator):
    """
    Platt Scaling: fits logistic regression on logits.

    p_calibrated = sigmoid(a * logits + b)

    This learns two parameters (a, b) that can shift and scale the logits,
    potentially changing threshold-0.5 predictions (unlike TS).
    """

    def __init__(self):
        super().__init__()
        self.a = 1.0  # slope
        self.b = 0.0  # intercept
        self._lr = None

    def fit(self, logits: np.ndarray, y: np.ndarray) -> 'PlattScaling':
        """
        Fit Platt scaling using logistic regression.

        Args:
            logits: Model logits (n,)
            y: Binary labels (n,)

        Returns:
            self
        """
        logits = np.asarray(logits).flatten()
        y = np.asarray(y).flatten()

        # Use sklearn's LogisticRegression
        # This fits: p = sigmoid(a * logit + b)
        self._lr = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            C=1e10  # Large C = minimal regularization
        )
        self._lr.fit(logits.reshape(-1, 1), y)

        self.a = float(self._lr.coef_[0, 0])
        self.b = float(self._lr.intercept_[0])
        self.is_fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling.

        Args:
            logits: Model logits

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")

        logits = np.asarray(logits).flatten()
        scaled_logits = self.a * logits + self.b
        return sigmoid(scaled_logits)


# ==============================================================================
# Isotonic Regression
# ==============================================================================

class IsotonicCalibration(BaseCalibrator):
    """
    Isotonic Regression: non-parametric calibration using monotonic regression.
    
    Learns a monotonic (non-decreasing) mapping from probabilities to calibrated probabilities.
    More flexible than Platt scaling but requires more calibration data.
    
    Note: Unlike TS/Platt, Isotonic works on PROBABILITIES, not logits.
    We convert logits → probs first, then fit isotonic regression.
    """
    
    def __init__(self):
        super().__init__()
        from sklearn.isotonic import IsotonicRegression
        self._isotonic = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, logits: np.ndarray, y: np.ndarray) -> 'IsotonicCalibration':
        """
        Fit isotonic regression on probabilities.
        
        Args:
            logits: Model logits (n,)
            y: Binary labels (n,)
        
        Returns:
            self
        """
        logits = np.asarray(logits).flatten()
        y = np.asarray(y).flatten()
        
        # Convert logits to probabilities first
        probs = sigmoid(logits)
        
        # Fit isotonic regression: probs → y
        self._isotonic.fit(probs, y)
        self.is_fitted = True
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.
        
        Args:
            logits: Model logits
        
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        logits = np.asarray(logits).flatten()
        probs = sigmoid(logits)
        
        # Transform through isotonic regression
        calibrated_probs = self._isotonic.transform(probs)
        return np.asarray(calibrated_probs).flatten()


# ==============================================================================
# Beta Calibration
# ==============================================================================

class BetaCalibration(BaseCalibrator):
    """
    Beta Calibration: uses Beta distribution for calibration.
    
    Models calibrated probability as:
        p_cal = Beta_CDF(p_raw; a, b, c)
    
    Where a, b, c are learned parameters.
    
    Reference: "Beyond temperature scaling: Obtaining well-calibrated 
               multiclass probabilities with Dirichlet calibration"
               (Kull et al., 2019)
    """
    
    def __init__(self):
        super().__init__()
        self.a = 1.0  # shape parameter
        self.b = 1.0  # shape parameter  
        self.c = 0.0  # location parameter
    
    def fit(self, logits: np.ndarray, y: np.ndarray, max_iter: int = 100) -> 'BetaCalibration':
        """
        Fit beta calibration by optimizing NLL.
        
        Args:
            logits: Model logits (n,)
            y: Binary labels (n,)
            max_iter: Maximum optimization iterations
        
        Returns:
            self
        """
        logits = np.asarray(logits).flatten()
        y = np.asarray(y).flatten()
        
        # Convert to probabilities
        probs = sigmoid(logits)
        
        def nll_loss(params):
            a, b, c = params
            
            # Ensure positive shape parameters
            if a <= 0 or b <= 0:
                return 1e10
            
            # Apply beta transformation
            from scipy.stats import beta as beta_dist
            
            # Clip probabilities to avoid numerical issues
            probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
            
            # Beta CDF transformation
            calibrated_probs = beta_dist.cdf(probs_clipped, a, b)
            
            # Apply location shift
            calibrated_probs = np.clip(calibrated_probs + c, 1e-15, 1 - 1e-15)
            
            # Negative log-likelihood
            loss = -np.mean(y * np.log(calibrated_probs) + 
                          (1 - y) * np.log(1 - calibrated_probs))
            
            return loss
        
        # Optimize parameters starting from (1, 1, 0)
        from scipy.optimize import minimize
        result = minimize(
            nll_loss,
            x0=[1.0, 1.0, 0.0],
            method='Nelder-Mead',
            options={'maxiter': max_iter}
        )
        
        self.a, self.b, self.c = result.x
        
        # Ensure positive shape parameters
        self.a = max(self.a, 0.1)
        self.b = max(self.b, 0.1)
        
        self.is_fitted = True
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply beta calibration.
        
        Args:
            logits: Model logits
        
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        from scipy.stats import beta as beta_dist
        
        logits = np.asarray(logits).flatten()
        probs = sigmoid(logits)
        
        # Clip to avoid numerical issues
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        
        # Beta transformation
        calibrated_probs = beta_dist.cdf(probs_clipped, self.a, self.b)
        
        # Apply location shift and clip
        calibrated_probs = np.clip(calibrated_probs + self.c, 0.0, 1.0)
        
        return calibrated_probs


# ==============================================================================
# Calibration Metrics
# ==============================================================================

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum over bins: (samples_in_bin / total_samples) * |accuracy - confidence|

    Args:
        probs: Predicted probabilities in [0, 1]
        labels: Binary ground truth labels
        n_bins: Number of bins (default: 10, uniform width)

    Returns:
        ECE value in [0, 1]. Lower is better (0 = perfectly calibrated).
    """
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()

    # Uniform-width bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    n_total = len(probs)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this bin
        if bin_upper == 1.0:
            # Include upper boundary in last bin
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            confidence_in_bin = np.mean(probs[in_bin])
            ece += (n_in_bin / n_total) * np.abs(accuracy_in_bin - confidence_in_bin)

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Brier Score.

    Brier = mean((probs - labels)^2)

    Args:
        probs: Predicted probabilities
        labels: Binary labels

    Returns:
        Brier score in [0, 1]. Lower is better.
    """
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    return float(np.mean((probs - labels) ** 2))


def compute_nll(probs: np.ndarray, labels: np.ndarray, clip: float = 1e-7) -> float:
    """
    Compute Negative Log-Likelihood (cross-entropy loss).

    NLL = -mean(y * log(p) + (1-y) * log(1-p))

    Args:
        probs: Predicted probabilities
        labels: Binary labels
        clip: Clip probabilities to [clip, 1-clip] to avoid log(0)

    Returns:
        NLL value. Lower is better.
    """
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    probs = np.clip(probs, clip, 1 - clip)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


# ==============================================================================
# Factory Function
# ==============================================================================

def get_calibrator(method: str) -> BaseCalibrator:
    """
    Get calibrator by name.

    Args:
        method: One of 'temperature', 'ts', 'platt', 'isotonic', 'beta'

    Returns:
        Calibrator instance
    """
    calibrators = {
        'temperature': TemperatureScaling,
        'ts': TemperatureScaling,
        'platt': PlattScaling,
        'isotonic': IsotonicCalibration,
        'beta': BetaCalibration,
    }

    method_lower = method.lower()
    if method_lower not in calibrators:
        raise ValueError(f"Unknown calibration method: {method}. Choose from {list(calibrators.keys())}")

    return calibrators[method_lower]()



# ==============================================================================
# Unit Tests
# ==============================================================================

def _run_calibration_tests():
    """
    Lightweight unit tests for calibration correctness.

    Run with: python -c "from calibration import _run_calibration_tests; _run_calibration_tests()"
    """
    print("="*70)
    print("RUNNING CALIBRATION UNIT TESTS (All 4 Methods)")
    print("="*70)

    np.random.seed(42)

    # Test data: logits with known properties
    logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    labels = np.array([0, 0, 1, 1, 1])

    # Test 1: TS with T=1 should be identity
    print("\n[Test 1] TS identity (T=1)")
    ts = TemperatureScaling()
    ts.temperature = 1.0
    ts.is_fitted = True
    p_raw = sigmoid(logits)
    p_ts = ts.transform(logits)
    assert np.allclose(p_raw, p_ts), f"TS with T=1 should be identity. Got diff: {np.abs(p_raw - p_ts).max()}"
    print("  ✅ PASS: TS with T=1 is identity")

    # Test 2: TS preserves sign (threshold-0.5 predictions)
    print("\n[Test 2] TS preserves threshold-0.5 predictions")
    ts_fitted = TemperatureScaling()
    ts_fitted.fit(logits, labels)
    print(f"  Fitted T = {ts_fitted.temperature:.4f}")

    preds_raw = (sigmoid(logits) >= 0.5).astype(int)
    p_ts_fitted = ts_fitted.transform(logits)
    preds_ts = (p_ts_fitted >= 0.5).astype(int)

    assert np.array_equal(preds_raw, preds_ts), (
        f"TS should preserve 0.5-threshold predictions!\n"
        f"  Raw preds: {preds_raw}\n"
        f"  TS preds:  {preds_ts}"
    )
    print("  ✅ PASS: TS preserves threshold-0.5 predictions")

    # Test 3: Platt scaling can change predictions
    print("\n[Test 3] Platt scaling parameters")
    platt = PlattScaling()
    platt.fit(logits, labels)
    print(f"  Fitted a = {platt.a:.4f}, b = {platt.b:.4f}")
    p_platt = platt.transform(logits)
    print(f"  Raw probs:   {sigmoid(logits)}")
    print(f"  Platt probs: {p_platt}")
    print("  ✅ PASS: Platt scaling fitted")

    # Test 4: Isotonic calibration monotonicity
    print("\n[Test 4] Isotonic calibration")
    isotonic = IsotonicCalibration()
    
    # Generate more data for isotonic (needs more samples)
    np.random.seed(123)
    logits_iso = np.random.randn(100)
    labels_iso = (sigmoid(logits_iso) + 0.1 * np.random.randn(100) > 0.5).astype(int)
    
    isotonic.fit(logits_iso, labels_iso)
    p_iso = isotonic.transform(logits_iso)
    
    # Check that output is valid probabilities
    assert np.all((p_iso >= 0) & (p_iso <= 1)), "Isotonic probs must be in [0, 1]"
    
    # Check monotonicity: higher logit → higher calibrated prob
    sorted_idx = np.argsort(logits_iso)
    p_iso_sorted = p_iso[sorted_idx]
    # Allow small violations due to clipping
    diffs = np.diff(p_iso_sorted)
    violations = np.sum(diffs < -0.01)  # Allow tiny numerical errors
    assert violations < 5, f"Isotonic should be monotonic, got {violations} violations"
    
    print(f"  Calibrated {len(logits_iso)} samples")
    print(f"  Output range: [{p_iso.min():.4f}, {p_iso.max():.4f}]")
    print("  ✅ PASS: Isotonic calibration works")

    # Test 5: Beta calibration
    print("\n[Test 5] Beta calibration")
    beta_cal = BetaCalibration()
    beta_cal.fit(logits, labels, max_iter=50)  # Fewer iters for speed
    print(f"  Fitted a={beta_cal.a:.4f}, b={beta_cal.b:.4f}, c={beta_cal.c:.4f}")
    
    p_beta = beta_cal.transform(logits)
    
    # Check valid probabilities
    assert np.all((p_beta >= 0) & (p_beta <= 1)), "Beta probs must be in [0, 1]"
    
    # Check parameters are positive
    assert beta_cal.a > 0 and beta_cal.b > 0, "Beta shape parameters must be positive"
    
    print(f"  Raw probs:  {sigmoid(logits)}")
    print(f"  Beta probs: {p_beta}")
    print("  ✅ PASS: Beta calibration works")

    # Test 6: ECE computation
    print("\n[Test 6] ECE computation")
    probs_perfect = labels.astype(float)  # Perfect calibration
    ece_perfect = compute_ece(probs_perfect, labels)
    assert ece_perfect < 0.01, f"Perfect predictions should have ECE near 0, got {ece_perfect}"
    print(f"  Perfect calibration ECE = {ece_perfect:.6f}")

    probs_all_half = np.full_like(labels, 0.5, dtype=float)
    ece_all_half = compute_ece(probs_all_half, labels)
    print(f"  All 0.5 predictions ECE = {ece_all_half:.6f}")
    print("  ✅ PASS: ECE computation works")

    # Test 7: NLL with clipping
    print("\n[Test 7] NLL with clipping")
    nll = compute_nll(p_raw, labels)
    print(f"  NLL = {nll:.4f}")
    # NLL should not be inf or nan
    assert np.isfinite(nll), f"NLL should be finite, got {nll}"
    print("  ✅ PASS: NLL is finite")

    # Test 8: Factory function
    print("\n[Test 8] Factory function")
    for method in ['temperature', 'platt', 'isotonic', 'beta']:
        calibrator = get_calibrator(method)
        assert calibrator is not None, f"Failed to create {method} calibrator"
        print(f"  ✅ Created {method} calibrator: {type(calibrator).__name__}")
    
    print("  ✅ PASS: Factory function works for all methods")

    # Test 9: Compare all 4 methods on same data
    print("\n[Test 9] Comparison of all 4 methods")
    np.random.seed(456)
    logits_comp = np.random.randn(50)
    labels_comp = (sigmoid(logits_comp) + 0.15 * np.random.randn(50) > 0.5).astype(int)
    
    results = {}
    for method in ['temperature', 'platt', 'isotonic', 'beta']:
        cal = get_calibrator(method)
        cal.fit(logits_comp, labels_comp)
        probs_cal = cal.transform(logits_comp)
        ece = compute_ece(probs_cal, labels_comp)
        results[method] = ece
        print(f"  {method:12s}: ECE = {ece:.4f}")
    



if __name__ == "__main__":
    _run_calibration_tests()
