import torch
import torch.nn as nn
import numpy as np
from scipy.stats import johnsonsu, t as t_dist, norm
from scipy.special import erfinv
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_data_advanced(data_type, mu_proj, O_proj, count, n_synthetic, device):
    """
    Generate synthetic data using unified copula framework as described in presentation.
    
    Args:
        data_type: 'tabular', 'image', or 'text'
        mu_proj: Mean vector in projected space (bar_mu_c)
        O_proj: Covariance matrix in projected space (Sigma_c)
        count: Number of real samples (for context)
        n_synthetic: Number of synthetic samples to generate
        device: Device for computation
        
    Returns:
        Synthetic samples in projected space
    """
    # Step 2: General Copula Framework (from presentation)
    # Extract marginal parameters
    dim = mu_proj.shape[0]
    marginal_means = mu_proj  # hat_mu_j
    marginal_vars = torch.diag(O_proj)  # hat_sigma_j^2
    marginal_stds = torch.sqrt(marginal_vars)  # hat_sigma_j
    
    # Compute correlation matrix R
    correlation_matrix = compute_correlation_matrix(O_proj, marginal_stds, device)
    
    # Choose copula family and generate samples based on data type
    if data_type == 'text':
        return generate_text_copula(marginal_means, marginal_stds, correlation_matrix, n_synthetic, dim, device)
    elif data_type == 'tabular':
        return generate_tabular_adaptive_copula(marginal_means, marginal_stds, correlation_matrix, n_synthetic, dim, device)
    elif data_type == 'image':
        return generate_image_spatial_copula(marginal_means, marginal_stds, correlation_matrix, n_synthetic, dim, device)
    else:
        # Fallback to text copula for unknown types
        return generate_text_copula(marginal_means, marginal_stds, correlation_matrix, n_synthetic, dim, device)


def compute_correlation_matrix(O_proj, marginal_stds, device):
    """Compute correlation matrix R from covariance matrix."""
    dim = O_proj.shape[0]
    R = torch.zeros((dim, dim), device=device)
    
    for i in range(dim):
        for j in range(dim):
            if marginal_stds[i] > 1e-8 and marginal_stds[j] > 1e-8:
                R[i, j] = O_proj[i, j] / (marginal_stds[i] * marginal_stds[j])
            else:
                R[i, j] = 1.0 if i == j else 0.0
    
    # Ensure valid correlation matrix
    R = 0.5 * (R + R.T)
    R.diagonal().fill_(1.0)
    
    # Ensure positive definite
    try:
        torch.linalg.cholesky(R)
    except:
        # If not positive definite, use nearest positive definite matrix
        eigenvalues, eigenvectors = torch.linalg.eigh(R)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        R = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        R = 0.5 * (R + R.T)
        R.diagonal().fill_(1.0)
    
    return R


def generate_text_copula(marginal_means, marginal_stds, R, n_synthetic, dim, device):
    """
    Improved text copula using Gaussian mixture model with empirical marginals.
    Better captures multimodal nature of text embeddings.
    """
    # Adaptive number of mixture components based on dimension
    n_components = min(3, max(1, int(np.sqrt(dim) / 5)))
    
    try:
        # Generate mixture weights
        weights = torch.softmax(torch.randn(n_components, device=device), dim=0)
        
        # Allocate samples
        samples = torch.zeros(n_synthetic, dim, device=device)
        current_idx = 0
        
        for k in range(n_components):
            # Number of samples for this component
            if k == n_components - 1:
                n_k = n_synthetic - current_idx
            else:
                n_k = int(weights[k] * n_synthetic)
            
            if n_k == 0:
                continue
            
            # Each component has slightly different correlation structure
            # This captures different modes in the text embedding space
            correlation_scale = 0.7 + 0.3 * k / max(1, n_components - 1)
            R_k = R * correlation_scale
            R_k.diagonal().fill_(1.0)
            
            # Ensure positive definite
            try:
                L_k = torch.linalg.cholesky(R_k)
            except:
                eigenvalues, eigenvectors = torch.linalg.eigh(R_k)
                eigenvalues = torch.clamp(eigenvalues, min=1e-6)
                R_k = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
                R_k = 0.5 * (R_k + R_k.T)
                R_k.diagonal().fill_(1.0)
                L_k = torch.linalg.cholesky(R_k)
            
            # Generate from Gaussian copula
            Z_k = torch.randn(n_k, dim, device=device) @ L_k.T
            
            # Transform to uniform using standard normal CDF
            U_k = torch.zeros_like(Z_k)
            for j in range(dim):
                # Use error function for normal CDF (more stable than scipy on GPU)
                U_k[:, j] = 0.5 * (1 + torch.erf(Z_k[:, j] / np.sqrt(2)))
            
            # Apply empirical-inspired transformation
            # Instead of strict empirical CDF, use a flexible transformation
            # that preserves the mean and std while allowing for multimodality
            for j in range(dim):
                # Base transformation
                base_samples = marginal_means[j] + marginal_stds[j] * torch.erfinv(2 * U_k[:, j] - 1) * np.sqrt(2)
                
                # Add slight mode shift for different components
                mode_shift = (k - n_components/2) * marginal_stds[j] * 0.3
                samples[current_idx:current_idx+n_k, j] = base_samples + mode_shift
                
                # Soft clipping to prevent extreme values
                clip_range = 4.0
                samples[current_idx:current_idx+n_k, j] = torch.tanh(
                    samples[current_idx:current_idx+n_k, j] / (clip_range * marginal_stds[j])
                ) * (clip_range * marginal_stds[j])
            
            current_idx += n_k
        
        # Final adjustment to match target moments
        current_mean = torch.mean(samples, dim=0)
        current_std = torch.std(samples, dim=0)
        
        # Standardize and rescale
        samples = (samples - current_mean) / (current_std + 1e-8)
        samples = samples * marginal_stds + marginal_means
        
        return samples
        
    except Exception as e:
        print(f"Text copula generation failed: {e}, using fallback")
        return generate_gaussian_fallback(marginal_means, R, marginal_stds, n_synthetic, device)


def generate_tabular_adaptive_copula(marginal_means, marginal_stds, R, n_synthetic, dim, device):
    """
    Tabular Data: Adaptive Copula Selection (from presentation)
    - Data-driven copula choice based on correlation pattern
    """
    # Analyze correlation pattern
    R_abs = torch.abs(R)
    avg_abs_corr = (torch.sum(R_abs) - dim) / (dim * (dim - 1))  # bar|rho|
    
    # Tail correlation indicator
    R_squared = R @ R
    tau_tail = torch.trace(R_squared) / dim - 1
    
    # Select copula based on correlation analysis
    if avg_abs_corr < 0.3:
        # Weak dependencies: Use Gaussian copula
        return generate_gaussian_copula(marginal_means, marginal_stds, R, n_synthetic, dim, device)
    elif tau_tail > 0.5:
        # Tail dependence: Use t-copula with nu=5
        return generate_t_copula_fixed_df(marginal_means, marginal_stds, R, n_synthetic, dim, device, df=5.0)
    else:
        # Asymmetric: Use Clayton copula
        theta = 2 * avg_abs_corr / (1 - avg_abs_corr)
        return generate_clayton_copula(marginal_means, marginal_stds, R, n_synthetic, dim, device, theta)


def generate_image_spatial_copula(marginal_means, marginal_stds, R, n_synthetic, dim, device):
    """
    Image Data: Spatial Copula (from presentation)
    - Incorporates spatial structure through distance-based weights
    """
    # Define spatial structure
    # Assume features are arranged in a grid
    grid_size = int(np.sqrt(dim))
    if grid_size * grid_size != dim:
        # If not perfect square, use approximate grid
        grid_size = int(np.ceil(np.sqrt(dim)))
    
    # Construct spatial weight matrix W
    W = torch.zeros((dim, dim), device=device)
    ell = np.sqrt(dim) / 4  # Characteristic length scale
    
    for i in range(dim):
        xi, yi = i % grid_size, i // grid_size
        for j in range(dim):
            xj, yj = j % grid_size, j // grid_size
            d_ij = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            W[i, j] = np.exp(-d_ij / ell)
    
    # Modify correlation: tilde_rho_ij = rho_ij * w_ij
    R_spatial = R * W
    
    # Ensure valid correlation matrix
    R_spatial = 0.5 * (R_spatial + R_spatial.T)
    R_spatial.diagonal().fill_(1.0)
    
    # Ensure positive definite (nearPD if needed)
    try:
        torch.linalg.cholesky(R_spatial)
    except:
        eigenvalues, eigenvectors = torch.linalg.eigh(R_spatial)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        R_spatial = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        R_spatial = 0.5 * (R_spatial + R_spatial.T)
        R_spatial.diagonal().fill_(1.0)
    
    # Use Gaussian copula with spatial correlation
    try:
        L = torch.linalg.cholesky(R_spatial)
        
        # Sample from multivariate normal
        Z = torch.randn(n_synthetic, dim, device=device) @ L.T
        
        # Transform to uniform
        U = torch.zeros_like(Z)
        for j in range(dim):
            U[:, j] = torch.tensor(norm.cdf(Z[:, j].cpu().numpy()), device=device)
        
        # Apply truncated normal marginals for images
        samples = torch.zeros_like(U)
        for j in range(dim):
            # Inverse normal CDF
            z_j = torch.erfinv(2 * U[:, j] - 1) * np.sqrt(2)
            # Scale and shift
            samples[:, j] = marginal_means[j] + marginal_stds[j] * z_j
            # Clip to [0, 1] for image features
            samples[:, j] = torch.clamp(samples[:, j], 0, 1)
        
        return samples
        
    except Exception as e:
        print(f"Spatial copula generation failed: {e}, using fallback")
        return generate_gaussian_fallback(marginal_means, R, marginal_stds, n_synthetic, device)


def generate_gaussian_copula(marginal_means, marginal_stds, R, n_synthetic, dim, device):
    """Generate samples using Gaussian copula with normal marginals."""
    try:
        L = torch.linalg.cholesky(R)
        Z = torch.randn(n_synthetic, dim, device=device) @ L.T
        
        # Apply marginal transformations
        samples = marginal_means.unsqueeze(0) + marginal_stds.unsqueeze(0) * Z
        return samples
        
    except:
        return generate_gaussian_fallback(marginal_means, R, marginal_stds, n_synthetic, device)


def generate_t_copula_fixed_df(marginal_means, marginal_stds, R, n_synthetic, dim, device, df=5.0):
    """Generate samples using t-copula with fixed degrees of freedom."""
    try:
        L = torch.linalg.cholesky(R)
        
        # Generate multivariate t
        Z = torch.randn(n_synthetic, dim, device=device)
        chi2_samples = torch.distributions.chi2.Chi2(df).sample((n_synthetic,)).to(device)
        V = chi2_samples / df
        T = Z @ L.T / torch.sqrt(V).unsqueeze(1)
        
        # Transform to uniform
        t_cdf = t_dist(df=df)
        U = torch.zeros_like(T)
        for j in range(dim):
            U[:, j] = torch.tensor(t_cdf.cdf(T[:, j].cpu().numpy()), device=device)
        
        # Apply normal marginals
        samples = torch.zeros_like(U)
        for j in range(dim):
            z_j = torch.erfinv(2 * U[:, j] - 1) * np.sqrt(2)
            samples[:, j] = marginal_means[j] + marginal_stds[j] * z_j
        
        return samples
        
    except Exception as e:
        print(f"t-copula generation failed: {e}, using fallback")
        return generate_gaussian_fallback(marginal_means, R, marginal_stds, n_synthetic, device)


def generate_clayton_copula(marginal_means, marginal_stds, R, n_synthetic, dim, device, theta):
    """Generate samples using Clayton copula (simplified for computational efficiency)."""
    # For high dimensions, Clayton copula is computationally intensive
    # We use a simplified approach based on the correlation structure
    
    # For now, approximate with t-copula with lower df (captures asymmetric dependence)
    df = 3.0  # Lower df for stronger tail dependence
    return generate_t_copula_fixed_df(marginal_means, marginal_stds, R, n_synthetic, dim, device, df)


def generate_gaussian_fallback(marginal_means, R, marginal_stds, n_synthetic, device):
    """Fallback Gaussian generation when copula methods fail."""
    dim = marginal_means.shape[0]
    
    # Reconstruct covariance from correlation and marginal stds
    O_proj = torch.zeros((dim, dim), device=device)
    for i in range(dim):
        for j in range(dim):
            O_proj[i, j] = R[i, j] * marginal_stds[i] * marginal_stds[j]
    
    try:
        L = torch.linalg.cholesky(O_proj)
        z = torch.randn(n_synthetic, dim, device=device)
        return marginal_means.unsqueeze(0) + z @ L.T
    except:
        # If Cholesky fails, use eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(O_proj)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        L = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))
        z = torch.randn(n_synthetic, dim, device=device)
        return marginal_means.unsqueeze(0) + z @ L.T


# Legacy classes for compatibility
class KDESyntheticGenerator:
    """Deprecated - kept for compatibility."""
    def __init__(self, n_bins=50, bandwidth='scott'):
        pass
    
    def fit_and_generate(self, mu_proj, O_proj, count, n_synthetic, device):
        return generate_synthetic_data_advanced('tabular', mu_proj, O_proj, count, n_synthetic, device)


class GMMSyntheticGenerator:
    """Deprecated - kept for compatibility."""
    def __init__(self, n_components=3, regularization_strength=0.3):
        pass
    
    def fit_and_generate(self, mu_proj, O_proj, count, n_synthetic, device):
        return generate_synthetic_data_advanced('image', mu_proj, O_proj, count, n_synthetic, device)


class FlowSyntheticGenerator:
    """Deprecated - kept for compatibility."""
    def __init__(self, flow_steps=4, hidden_dim=64):
        pass
    
    def fit_and_generate(self, mu_proj, O_proj, count, n_synthetic, device):
        return generate_synthetic_data_advanced('text', mu_proj, O_proj, count, n_synthetic, device)


# Add MMD Goodness-of-Fit Testing Class
class MMDGoodnessOfFit:
    """
    Maximum Mean Discrepancy test for comparing real and synthetic data distributions.
    Works for all data types (tabular, image, text).
    """
    def __init__(self, kernel='gaussian', bandwidth='median', threshold=0.05):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.threshold = threshold
        
    def compute_mmd(self, real_data, synthetic_data, return_details=False):
        """
        Compute MMD between real and synthetic data.
        
        Args:
            real_data: Tensor of real data samples
            synthetic_data: Tensor of synthetic data samples
            return_details: If True, return detailed results
            
        Returns:
            Dictionary with MMD results and pass/fail status
        """
        device = real_data.device
        n_real = real_data.shape[0]
        n_synth = synthetic_data.shape[0]
        
        # For efficiency, subsample if datasets are large
        max_samples = 500
        if n_real > max_samples:
            idx = torch.randperm(n_real, device=device)[:max_samples]
            real_data = real_data[idx]
            n_real = max_samples
        if n_synth > max_samples:
            idx = torch.randperm(n_synth, device=device)[:max_samples]
            synthetic_data = synthetic_data[idx]
            n_synth = max_samples
        
        # Compute bandwidth using median heuristic
        if self.bandwidth == 'median':
            # Combine data for distance computation
            combined = torch.cat([real_data, synthetic_data], dim=0)
            # Subsample for bandwidth computation
            subset_size = min(100, combined.shape[0])
            subset_idx = torch.randperm(combined.shape[0], device=device)[:subset_size]
            subset = combined[subset_idx]
            
            # Compute pairwise distances
            distances = torch.cdist(subset, subset)
            # Get median of non-zero distances
            mask = distances > 0
            if mask.any():
                median_dist = torch.median(distances[mask])
                bandwidth = median_dist / np.sqrt(2)
            else:
                bandwidth = torch.tensor(1.0, device=device)
        else:
            bandwidth = torch.tensor(self.bandwidth, device=device)
        
        # Gaussian kernel function
        def gaussian_kernel(X, Y, bandwidth):
            # Compute squared Euclidean distances
            XX = torch.sum(X**2, dim=1, keepdim=True)
            YY = torch.sum(Y**2, dim=1, keepdim=True)
            XY = torch.matmul(X, Y.T)
            distances = XX + YY.T - 2 * XY
            # Apply Gaussian kernel
            return torch.exp(-distances / (2 * bandwidth**2))
        
        # Compute kernel matrices
        K_rr = gaussian_kernel(real_data, real_data, bandwidth)
        K_ss = gaussian_kernel(synthetic_data, synthetic_data, bandwidth)
        K_rs = gaussian_kernel(real_data, synthetic_data, bandwidth)
        
        # Compute unbiased MMD² estimator
        # Remove diagonal terms for unbiased estimate
        if n_real > 1:
            K_rr_sum = (torch.sum(K_rr) - torch.trace(K_rr)) / (n_real * (n_real - 1))
        else:
            K_rr_sum = torch.tensor(0.0, device=device)
            
        if n_synth > 1:
            K_ss_sum = (torch.sum(K_ss) - torch.trace(K_ss)) / (n_synth * (n_synth - 1))
        else:
            K_ss_sum = torch.tensor(0.0, device=device)
            
        K_rs_sum = torch.sum(K_rs) / (n_real * n_synth)
        
        mmd_squared = K_rr_sum + K_ss_sum - 2 * K_rs_sum
        mmd = torch.sqrt(torch.clamp(mmd_squared, min=0))
        
        # Determine if test passes
        passed = mmd.item() < self.threshold
        
        result = {
            'mmd': mmd.item(),
            'mmd_squared': mmd_squared.item(),
            'threshold': self.threshold,
            'passed': passed,
            'bandwidth': bandwidth.item()
        }
        
        if return_details:
            result['n_real'] = n_real
            result['n_synthetic'] = n_synth
            result['kernel_type'] = self.kernel
            
        return result
    
    def adaptive_test_with_feedback(self, real_data, synthetic_data, data_type):
        """
        Test synthetic data quality and provide feedback for improvement.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            data_type: Type of data ('tabular', 'image', 'text')
            
        Returns:
            Test results with adaptive feedback
        """
        result = self.compute_mmd(real_data, synthetic_data, return_details=True)
        
        if not result['passed']:
            # Provide specific feedback based on MMD value and data type
            mmd_value = result['mmd']
            
            feedback = {
                'severity': 'high' if mmd_value > 0.15 else 'medium' if mmd_value > 0.08 else 'low'
            }
            
            if data_type == 'text':
                if mmd_value > 0.15:
                    feedback['suggestion'] = 'Increase mixture components or use different copula'
                elif mmd_value > 0.08:
                    feedback['suggestion'] = 'Adjust correlation structure'
                else:
                    feedback['suggestion'] = 'Fine-tune marginal distributions'
                    
            elif data_type == 'image':
                if mmd_value > 0.15:
                    feedback['suggestion'] = 'Adjust spatial correlation parameters'
                elif mmd_value > 0.08:
                    feedback['suggestion'] = 'Modify spatial weight decay'
                else:
                    feedback['suggestion'] = 'Fine-tune marginal clipping'
                    
            else:  # tabular
                if mmd_value > 0.15:
                    feedback['suggestion'] = 'Switch copula type (try t-copula)'
                elif mmd_value > 0.08:
                    feedback['suggestion'] = 'Adjust correlation threshold for copula selection'
                else:
                    feedback['suggestion'] = 'Fine-tune marginal distributions'
            
            result['feedback'] = feedback
            
        return result


def generate_synthetic_data_with_mmd_validation(data_type, mu_proj, O_proj, count, n_synthetic, device, 
                                               real_data=None, max_attempts=3):
    """
    Generate synthetic data with MMD validation.
    If real_data is provided, will validate synthetic data quality.
    
    Args:
        data_type: Type of data ('tabular', 'image', 'text')
        mu_proj: Mean vector in projected space
        O_proj: Covariance matrix in projected space
        count: Number of real samples
        n_synthetic: Number of synthetic samples to generate
        device: Computation device
        real_data: Optional real data for validation
        max_attempts: Maximum generation attempts if validation fails
        
    Returns:
        Tuple of (synthetic_data, mmd_result)
    """
    mmd_tester = MMDGoodnessOfFit()
    best_synthetic = None
    best_mmd = float('inf')
    
    for attempt in range(max_attempts):
        # Generate synthetic data
        synthetic = generate_synthetic_data_advanced(
            data_type, mu_proj, O_proj, count, n_synthetic, device
        )
        
        if real_data is None:
            # No validation possible
            return synthetic, None
        
        # Validate with MMD
        mmd_result = mmd_tester.adaptive_test_with_feedback(real_data, synthetic, data_type)
        
        if mmd_result['passed']:
            return synthetic, mmd_result
        
        # Keep best attempt
        if mmd_result['mmd'] < best_mmd:
            best_mmd = mmd_result['mmd']
            best_synthetic = synthetic
        
        # If not last attempt, try to improve based on feedback
        if attempt < max_attempts - 1 and 'feedback' in mmd_result:
            print(f"Attempt {attempt + 1} failed with MMD={mmd_result['mmd']:.4f}. "
                  f"Suggestion: {mmd_result['feedback']['suggestion']}")
    
    # Return best attempt even if it didn't pass
    final_result = mmd_tester.compute_mmd(real_data, best_synthetic)
    final_result['warning'] = f'Failed to pass MMD test after {max_attempts} attempts'
    
    return best_synthetic, final_result 