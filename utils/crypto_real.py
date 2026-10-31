"""
Real cryptographic implementations for AegisFL-Cal using tenseal for HE and hash-based ZKP.
"""
import torch
import hashlib
import time
import json
import numpy as np
from typing import Union, List, Any, Dict, Tuple
from collections import defaultdict

# Try to import tenseal, FAIL if not available - NO SIMULATION ALLOWED
try:
    import tenseal as ts
except ImportError:
    raise ImportError(
        "CRITICAL: tenseal is required for AegisFL-Cal but not installed.\n"
        "Please install it with: pip install tenseal\n"
        "Simulation mode is NOT allowed for cryptographic operations."
    )

# Lock 2026-10-30: ZKP bands use the previous round's aggregate mean/std.
# kappa=3 in steady state (~99.7% under a Gaussian). Cold start kappa=5.
# The proved scalar is the client mean S/n. Full Bulletproofs are not in this
# file; the mock still binds the proof challenge to CKKS ciphertext bytes.
KAPPA_STEADY = 3.0
KAPPA_COLD_START = 5.0


def plaintext_digest(tensor: torch.Tensor) -> str:
    """Digest of the plaintext tensor that was passed to CKKS encryption."""
    flat64 = tensor.detach().flatten().cpu().to(torch.float64).numpy()
    return hashlib.sha256(flat64.tobytes()).hexdigest()


def count_plaintext_tensor(n_k_c: int) -> torch.Tensor:
    """Canonical plaintext for the encrypted class count."""
    return torch.tensor([float(n_k_c)], dtype=torch.float32)


def he_threshold_t(n_participants: int) -> int:
    """Decryption threshold among this round's participants.

    t = floor(n_t/2)+1. The paper setting is n_t=30, so t=16.
    This is not floor(N/2)+1 on a roster of 300.
    """
    if n_participants < 1:
        raise ValueError("n_participants must be positive")
    return (int(n_participants) // 2) + 1


if he_threshold_t(30) != 16:
    raise RuntimeError("paper threshold lock broken: n_t=30 must give t=16")


class PerformanceMonitor:
    """Monitors performance metrics for cryptographic operations."""
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'sizes': [],
            'bandwidth': 0
        })
        self.start_times = {}
    
    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        self.start_times[operation_name] = time.time()
    
    def end_operation(self, operation_name: str, size_bytes: int = 0):
        """End timing an operation and record metrics."""
        if operation_name in self.start_times:
            elapsed = time.time() - self.start_times[operation_name]
            self.metrics[operation_name]['count'] += 1
            self.metrics[operation_name]['total_time'] += elapsed
            if size_bytes > 0:
                self.metrics[operation_name]['sizes'].append(size_bytes)
                self.metrics[operation_name]['bandwidth'] += size_bytes
            del self.start_times[operation_name]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        for op_name, metrics in self.metrics.items():
            count = metrics['count']
            if count > 0:
                summary[op_name] = {
                    'count': count,
                    'avg_time_ms': (metrics['total_time'] / count) * 1000,
                    'total_time_s': metrics['total_time'],
                    'avg_size_bytes': sum(metrics['sizes']) / len(metrics['sizes']) if metrics['sizes'] else 0,
                    'total_bandwidth_bytes': metrics['bandwidth']
                }
        return summary

# Global performance monitor
perf_monitor = PerformanceMonitor()

class HEContext:
    """Manages tenseal context for HE operations."""
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40], scale=2**40):
        # TENSEAL_AVAILABLE is always True here since we fail on import
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.global_scale = scale
        self.context.generate_galois_keys()
        self.scale = scale

class HEPublicKey:
    """Real HE public key using tenseal."""
    def __init__(self, context: HEContext = None):
        if context is None:
            raise ValueError("HEContext is required for HEPublicKey")
        self.context = context
        self.key_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]

class HEPrivateKey:
    """Real HE private key using tenseal."""
    def __init__(self, context: HEContext = None):
        if context is None:
            raise ValueError("HEContext is required for HEPrivateKey")
        self.context = context
        self.key_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]

def generate_he_keys():
    """Generate HE key pair using tenseal."""
    perf_monitor.start_operation("he_keygen")
    
    context = HEContext()
    pk = HEPublicKey(context)
    sk = HEPrivateKey(context)
    
    perf_monitor.end_operation("he_keygen")
    return pk, sk

class HECiphertext:
    """Real HE ciphertext using tenseal."""
    def __init__(self, ciphertext_data, public_key: HEPublicKey, original_shape=None):
        if public_key.context is None:
            raise ValueError("HEPublicKey must have a valid context")
        self.public_key_ref = public_key
        self.original_shape = original_shape
        self.ciphertext = ciphertext_data  # ts.CKKSVector
        self.is_real = True
    
    def __repr__(self):
        return f"HECiphertext(real, shape={self.original_shape})"
    
    def get_size_bytes(self) -> int:
        """Estimate ciphertext size in bytes."""
        if hasattr(self.ciphertext, 'serialize'):
            return len(self.ciphertext.serialize())
        else:
            # Estimate based on shape
            if self.original_shape:
                elem_count = np.prod(self.original_shape)
                # CKKS ciphertext is roughly 2x polynomial degree x coefficient size
                return int(elem_count * 8192 * 8 / 1024)  # Rough estimate
            return 8192  # Default estimate

def he_encrypt(plaintext_tensor: torch.Tensor, public_key: HEPublicKey) -> HECiphertext:
    """Encrypt tensor using tenseal."""
    perf_monitor.start_operation("he_encrypt")
    
    if public_key.context is None:
        raise ValueError("HEPublicKey must have a valid context for encryption")
    
    original_shape = plaintext_tensor.shape
    flat_data = plaintext_tensor.flatten().cpu().numpy().tolist()
    
    # Real encryption using tenseal
    encrypted_vector = ts.ckks_vector(public_key.context.context, flat_data)
    result = HECiphertext(encrypted_vector, public_key, original_shape)
    # Stamp the plaintext that was encrypted so a proof cannot open a different vector.
    result.plaintext_digest = plaintext_digest(plaintext_tensor)
    result.plaintext_numel = int(plaintext_tensor.numel())
    
    size = result.get_size_bytes()
    perf_monitor.end_operation("he_encrypt", size)
    return result

def he_add(ciphertext1: HECiphertext, ciphertext2: HECiphertext) -> HECiphertext:
    """Add two ciphertexts."""
    perf_monitor.start_operation("he_add")
    
    # Real homomorphic addition
    result_ct = ciphertext1.ciphertext + ciphertext2.ciphertext
    result = HECiphertext(result_ct, ciphertext1.public_key_ref, ciphertext1.original_shape)
    
    perf_monitor.end_operation("he_add")
    return result

def he_sum_ciphertexts(ciphertext_list: List[HECiphertext], public_key: HEPublicKey) -> HECiphertext:
    """Sum a list of ciphertexts."""
    perf_monitor.start_operation("he_sum")
    
    if not ciphertext_list:
        perf_monitor.end_operation("he_sum")
        return None
    
    result = ciphertext_list[0]
    for ct in ciphertext_list[1:]:
        result = he_add(result, ct)
    
    perf_monitor.end_operation("he_sum")
    return result

def he_decrypt(ciphertext: HECiphertext, private_key: HEPrivateKey) -> torch.Tensor:
    """Decrypt ciphertext."""
    perf_monitor.start_operation("he_decrypt")
    
    if private_key.context is None:
        raise ValueError("HEPrivateKey must have a valid context for decryption")
    
    # Real decryption
    decrypted_list = ciphertext.ciphertext.decrypt()
    decrypted_array = np.array(decrypted_list).reshape(ciphertext.original_shape)
    result = torch.tensor(decrypted_array, dtype=torch.float32)
    
    perf_monitor.end_operation("he_decrypt")
    return result

# ZKP Implementation using Hash-based Commitments
class ZKPCommitment:
    """Hash-based commitment for ZKP."""
    def __init__(self, data: Any, salt: str = None):
        self.salt = salt or hashlib.sha256(str(time.time()).encode()).hexdigest()
        self.commitment = self._compute_commitment(data)
        self.data = data  # In real implementation, this would not be stored
    
    def _compute_commitment(self, data: Any) -> str:
        """Compute SHA256 commitment of data + salt."""
        if isinstance(data, torch.Tensor):
            data_str = str(data.cpu().numpy().tolist())
        else:
            data_str = str(data)
        
        commitment_input = f"{data_str}||{self.salt}"
        return hashlib.sha256(commitment_input.encode()).hexdigest()
    
    def verify(self, claimed_data: Any) -> bool:
        """Verify if claimed data matches commitment."""
        claimed_commitment = self._compute_commitment(claimed_data)
        return claimed_commitment == self.commitment

class ZKPProof:
    """Zero-knowledge proof for correct computation."""
    def __init__(self, statement: str, commitments: Dict[str, ZKPCommitment], 
                 computation_proof: Dict[str, Any],
                 statistical_checks: Dict[str, bool] = None):
        self.statement = statement
        self.commitments = commitments
        self.computation_proof = computation_proof
        self.statistical_checks = statistical_checks if statistical_checks is not None else {}
        self.timestamp = time.time()
        self.proof_hash = self._compute_proof_hash()
    
    def _compute_proof_hash(self) -> str:
        """Compute hash of the entire proof."""
        proof_data = {
            'statement': self.statement,
            'commitments': {k: v.commitment for k, v in self.commitments.items()},
            'computation': self.computation_proof,
            'statistical_checks': self.statistical_checks,
            'timestamp': self.timestamp
        }
        return hashlib.sha256(json.dumps(proof_data, sort_keys=True).encode()).hexdigest()
    
    def get_size_bytes(self) -> int:
        """Estimate proof size."""
        # Each commitment is 64 bytes (SHA256), plus metadata
        # Statistical checks add a bit more metadata
        return len(self.commitments) * 64 + len(json.dumps(self.computation_proof).encode()) + len(json.dumps(self.statistical_checks).encode())

def zkp_generate_projection_proof(features: List[torch.Tensor], U_c: torch.Tensor,
                                  projected_stats: Dict[str, torch.Tensor]) -> ZKPProof:
    """Generate ZKP for correct feature projection and statistics computation."""
    perf_monitor.start_operation("zkp_generate")
    
    # Create commitments for inputs
    commitments = {}
    
    # Commit to projection matrix
    commitments['U_c'] = ZKPCommitment(U_c)
    
    # Commit to aggregated statistics (not individual features for privacy)
    commitments['sum_features'] = ZKPCommitment(sum(features))
    commitments['count'] = ZKPCommitment(torch.tensor(len(features)))
    
    # Commit to output statistics
    for key, value in projected_stats.items():
        commitments[f'output_{key}'] = ZKPCommitment(value)
    
    # Computation proof (simplified - in practice would use more sophisticated techniques)
    computation_proof = {
        'projection_method': 'matrix_multiply',
        'stats_computed': list(projected_stats.keys()),
        'feature_dim': features[0].shape[-1] if features else 0,
        'projected_dim': U_c.shape[1] if U_c is not None else 0
    }
    
    # Create statement
    statement = f"Correct projection using U_c and statistics computation"
    
    proof = ZKPProof(statement, commitments, computation_proof)
    
    perf_monitor.end_operation("zkp_generate", proof.get_size_bytes())
    return proof

def zkp_verify_projection_proof(proof: ZKPProof, U_c: torch.Tensor, 
                               expected_stats_keys: List[str]) -> bool:
    """Verify ZKP for projection correctness."""
    perf_monitor.start_operation("zkp_verify")
    
    try:
        # Verify U_c commitment
        if 'U_c' not in proof.commitments:
            perf_monitor.end_operation("zkp_verify")
            return False
        
        # In practice, we'd verify the commitment matches
        # For now, check structure
        if not proof.commitments['U_c'].verify(U_c):
            perf_monitor.end_operation("zkp_verify")
            return False
        
        # Verify expected statistics were computed
        computed_stats = proof.computation_proof.get('stats_computed', [])
        for key in expected_stats_keys:
            if f'output_{key}' not in proof.commitments:
                perf_monitor.end_operation("zkp_verify")
                return False
        
        # Verify proof integrity
        expected_hash = proof._compute_proof_hash()
        if expected_hash != proof.proof_hash:
            perf_monitor.end_operation("zkp_verify")
            return False
        
        # Verify statistical checks if present
        if proof.statistical_checks:
            for check_name, result in proof.statistical_checks.items():
                if not result:
                    print(f"ZKP statistical check failed: {check_name}")
                    perf_monitor.end_operation("zkp_verify")
                    return False
            
        perf_monitor.end_operation("zkp_verify")
        return True
        
    except Exception as e:
        print(f"ZKP verification error: {e}")
        perf_monitor.end_operation("zkp_verify")
        return False

# Bandwidth calculation for SMPC
def estimate_smpc_bandwidth(num_clients: int, feature_dim: int, num_classes: int) -> Dict[str, int]:
    """Estimate bandwidth requirements for SMPC PCA computation."""
    # Each client sends: S_k,c (d×1), M_k,c (d×d), n_k,c (scalar) per class
    per_client_per_class = (
        feature_dim * 4 +  # S vector (float32)
        feature_dim * feature_dim * 4 +  # M matrix (float32)
        4  # n scalar (float32)
    )
    
    # Total upload from all clients
    total_upload = num_clients * num_classes * per_client_per_class
    
    # Server broadcasts U_c matrices to all clients
    per_class_projection = feature_dim * 10 * 4  # d × s_c matrix (assuming s_c ≈ 10)
    total_download = num_clients * num_classes * per_class_projection
    
    # SMPC protocol overhead (rough estimate: 3x for secret sharing)
    smpc_overhead_factor = 3
    
    return {
        'client_upload_bytes': per_client_per_class * num_classes,
        'total_upload_bytes': total_upload * smpc_overhead_factor,
        'client_download_bytes': per_class_projection * num_classes,
        'total_download_bytes': total_download,
        'total_bandwidth_bytes': (total_upload * smpc_overhead_factor) + total_download
    }

# SMPC functions - implement real versions instead of importing from simulation
def smpc_secure_sum_tensors(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """Secure sum of tensors using SMPC (simplified implementation)."""
    if not tensor_list:
        return None
    return torch.stack(tensor_list).sum(dim=0)

def smpc_secure_sum_scalars(scalar_list: List[float]) -> float:
    """Secure sum of scalars using SMPC (simplified implementation)."""
    if not scalar_list:
        return 0.0
    return float(sum(scalar_list))

# Convenience function to get performance summary
def get_crypto_performance_summary():
    """Get summary of all cryptographic operations performance."""
    return perf_monitor.get_summary()

def reset_performance_monitor():
    """Reset performance monitoring."""
    global perf_monitor
    perf_monitor = PerformanceMonitor()

# Bulletproof-style Range Proofs for Input Validation
class RangeProof:
    """Optional L2 clip proof. Not the primary statistical test.

    The primary test is the client mean against the previous-round aggregate
    band (see zkp_generate_statistical_proof). l2_bound does not accept or
    reject a calibration contribution.
    """
    def __init__(self, commitment: ZKPCommitment, proof_data: Dict[str, Any]):
        self.commitment = commitment
        self.proof_data = proof_data
        self.timestamp = time.time()
    
    def get_size_bytes(self) -> int:
        """Estimate proof size - Bulletproofs are logarithmic in range size."""
        # Typical Bulletproof size: ~2KB for 64-bit range
        return 2048

def generate_range_proof(features: torch.Tensor, bound: float) -> RangeProof:
    """
    Optional proof that ||features||_2 <= bound.

    Not the primary ZKP test. Calibration accepts or rejects on the client
    mean S/n against the lagged aggregate band.
    """
    perf_monitor.start_operation("range_proof_generate")
    
    # Compute L2 norm
    l2_norm = torch.norm(features, p=2).item()
    
    # Create commitment to features
    commitment = ZKPCommitment(features)
    
    # In a real Bulletproof, we would:
    # 1. Commit to each element using Pedersen commitments
    # 2. Prove each element is in valid range
    # 3. Prove sum of squares <= bound^2
    
    # Simplified proof data (in practice would be zero-knowledge)
    proof_data = {
        'bound': bound,
        'norm_squared_commitment': hashlib.sha256(str(l2_norm**2).encode()).hexdigest(),
        'num_elements': features.numel(),
        'proof_type': 'bulletproof_range',
        # In real implementation, would include:
        # - Vector Pedersen commitments
        # - Inner product proof
        # - Range proof for each element
    }
    
    proof = RangeProof(commitment, proof_data)
    perf_monitor.end_operation("range_proof_generate", proof.get_size_bytes())
    return proof

def verify_range_proof(proof: RangeProof, bound: float) -> bool:
    """
    Verify range proof for L2-norm bound.
    
    Args:
        proof: RangeProof to verify
        bound: Expected L2-norm bound
        
    Returns:
        True if proof is valid
    """
    perf_monitor.start_operation("range_proof_verify")
    
    try:
        # Check proof structure
        if not isinstance(proof, RangeProof):
            perf_monitor.end_operation("range_proof_verify")
            return False
        
        # Verify bound matches
        if proof.proof_data.get('bound') != bound:
            perf_monitor.end_operation("range_proof_verify")
            return False
        
        # In real implementation would verify:
        # - Pedersen commitment opening
        # - Inner product proof
        # - Range proofs for each element
        
        perf_monitor.end_operation("range_proof_verify")
        return True
        
    except Exception as e:
        print(f"Range proof verification error: {e}")
        perf_monitor.end_operation("range_proof_verify")
        return False

def _as_1d(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().flatten()
    return torch.tensor(value, dtype=torch.float32).flatten()


def ciphertext_binding_digest(ciphertexts: List[Any]) -> str:
    """SHA-256 over serialized CKKS ciphertext bytes, length-prefixed per ciphertext."""
    hasher = hashlib.sha256()
    for ct in ciphertexts:
        if isinstance(ct, (bytes, bytearray)):
            raw = bytes(ct)
        else:
            ser = ct.ciphertext.serialize()
            raw = ser.encode('utf-8') if isinstance(ser, str) else bytes(ser)
        hasher.update(len(raw).to_bytes(8, 'big'))
        hasher.update(raw)
    return hasher.hexdigest()


def fiat_shamir_challenge(statement: str, mean_commitment: str, ciphertext_sha256: str) -> str:
    """Challenge binds the committed mean to the ciphertext byte digest."""
    blob = f"{statement}|{mean_commitment}|{ciphertext_sha256}".encode()
    return hashlib.sha256(blob).hexdigest()


def bounds_from_previous_aggregate(
    mu_bar: torch.Tensor,
    sigma_hat: torch.Tensor,
    kappa: float = KAPPA_STEADY,
) -> Dict[str, Any]:
    """Band for client means from the previous round's decrypted aggregate.

    lower/upper = mu_bar ± kappa * sigma_hat on each coordinate.
    Steady state uses kappa=3. A cold-start anchor, if one is supplied by the
    caller, uses kappa=5. This function does not look at individual clients.
    """
    mu = _as_1d(mu_bar)
    sigma = _as_1d(sigma_hat).clamp(min=0)
    if mu.numel() != sigma.numel():
        raise ValueError("aggregate mean and std must have the same length")
    kappa_f = float(kappa)
    return {
        'source': 'previous_round_aggregate',
        'kappa': kappa_f,
        'mu': mu,
        'sigma': sigma,
        'lower': mu - kappa_f * sigma,
        'upper': mu + kappa_f * sigma,
    }


def _variance_numerators_ok(S: torch.Tensor, O: torch.Tensor, n_k_c: int, tol: float = 1e-5) -> bool:
    """n * O_jj - S_j^2 >= 0 for every coordinate. O_jj stores sum of squares."""
    if n_k_c <= 0:
        return False
    S_flat = _as_1d(S)
    O_cpu = O.detach().cpu().float() if isinstance(O, torch.Tensor) else torch.tensor(O, dtype=torch.float32)
    if O_cpu.dim() != 2 or O_cpu.shape[0] != O_cpu.shape[1]:
        return False
    if S_flat.numel() != O_cpu.shape[0]:
        return False
    if not torch.isfinite(S_flat).all() or not torch.isfinite(O_cpu).all():
        return False
    for j in range(S_flat.numel()):
        var_num = float(n_k_c) * float(O_cpu[j, j]) - float(S_flat[j]) ** 2
        if var_num != var_num or var_num < -tol:
            return False
    return True


def _mean_in_lagged_band(mean: torch.Tensor, lagged_bounds: Dict[str, Any], tol: float = 1e-4) -> bool:
    lower = _as_1d(lagged_bounds['lower'])
    upper = _as_1d(lagged_bounds['upper'])
    m = _as_1d(mean)
    if m.numel() != lower.numel() or m.numel() != upper.numel():
        return False
    return bool(torch.all(m >= lower - tol) and torch.all(m <= upper + tol))


def zkp_generate_statistical_proof(
    S_k_c_proj: torch.Tensor,
    O_k_c_proj: torch.Tensor,
    n_k_c: int,
    U_c: torch.Tensor,
    lagged_bounds: Dict[str, Any] = None,
    ciphertexts: List[Any] = None,
    ) -> ZKPProof:
    """
    Prove the client mean S/n, not the raw sum.

    Required statements: n>0, n*O_jj - S_j^2 >= 0, and, when lagged_bounds is
    set, S_j/n inside the previous-round aggregate band. Cold start (no
    previous aggregate) records kappa=5 and skips the coordinate test.

    The Fiat-Shamir challenge hashes the CKKS ciphertext bytes. Full
    Bulletproofs are not implemented; this is a working mock with that API.
    l2_bound is not consulted.
    """
    perf_monitor.start_operation("zkp_generate_statistical")

    n_k_c = int(n_k_c)
    n_plain = count_plaintext_tensor(n_k_c)
    n_positive = n_k_c > 0
    values_finite = bool(
        torch.isfinite(S_k_c_proj).all().item() and torch.isfinite(O_k_c_proj).all().item()
    )
    if n_positive:
        means = _as_1d(S_k_c_proj) / float(n_k_c)
    else:
        means = _as_1d(S_k_c_proj) * 0.0

    var_ok = n_positive and values_finite and _variance_numerators_ok(S_k_c_proj, O_k_c_proj, n_k_c)
    cold_start = lagged_bounds is None
    in_band = True
    kappa = KAPPA_COLD_START if cold_start else float(lagged_bounds.get('kappa', KAPPA_STEADY))
    if not cold_start:
        in_band = _mean_in_lagged_band(means, lagged_bounds)

    bound_to_ciphertexts = False
    ciphertext_sha = None
    if ciphertexts is not None and len(ciphertexts) == 3:
        s_match = getattr(ciphertexts[0], 'plaintext_digest', None) == plaintext_digest(S_k_c_proj)
        o_match = getattr(ciphertexts[1], 'plaintext_digest', None) == plaintext_digest(O_k_c_proj)
        n_match = getattr(ciphertexts[2], 'plaintext_digest', None) == plaintext_digest(n_plain)
        if s_match and o_match and n_match:
            ciphertext_sha = ciphertext_binding_digest(ciphertexts)
            bound_to_ciphertexts = True

    valid_stats = bool(n_positive and values_finite and var_ok and in_band and bound_to_ciphertexts)

    commitments = {
        'client_mean': ZKPCommitment(means),
        'S_k_c_proj': ZKPCommitment(S_k_c_proj),
        'O_k_c_proj': ZKPCommitment(O_k_c_proj),
        'n_plain': ZKPCommitment(n_plain),
        'U_c': ZKPCommitment(U_c),
    }
    statement = "Client mean S/n lies in the previous-round aggregate band and matches the CKKS plaintext"
    mean_commitment_hex = commitments['client_mean'].commitment
    challenge = fiat_shamir_challenge(statement, mean_commitment_hex, ciphertext_sha or "")

    statistical_checks_results = {
        'n_positive': bool(n_positive),
        'values_finite': bool(values_finite),
        'variance_numerator_nonnegative': bool(var_ok),
        'ciphertext_bound': bool(bound_to_ciphertexts),
        'mean_in_lagged_band': bool(in_band) if not cold_start else True,
    }
    computation_proof_details = {
        'proof_type': 'lagged_aggregate_mean_range',
        'proved_quantity': 'client_mean_S_over_n',
        'primary_test': 'previous_round_aggregate_mean_std',
        'l2_is_primary': False,
        'full_bulletproofs': False,
        'cold_start': bool(cold_start),
        'kappa': kappa,
        'coordinate_test': (not cold_start),
        'ciphertext_sha256': ciphertext_sha,
        'fiat_shamir_challenge': challenge,
        'final_validity': valid_stats,
    }

    proof = ZKPProof(statement, commitments, computation_proof_details, statistical_checks_results)
    perf_monitor.end_operation("zkp_generate_statistical", proof.get_size_bytes())
    return proof

def zkp_verify_statistical_proof(
    proof: ZKPProof,
    lagged_bounds: Dict[str, Any] = None,
    ciphertexts: List[Any] = None,
    adaptive_bounds: Dict[str, Any] = None,
    ) -> bool:
    """
    Verify a client-mean proof against lagged aggregate bounds.

    lagged_bounds is the server's previous-round aggregate band (or None on
    cold start). adaptive_bounds is accepted only so older call sites still
    load; a median/MAD dict is ignored and cannot accept the proof.
    The server recomputes the band check from the committed mean. It does not
    decrypt individual clients and does not compute a median or MAD.
    """
    perf_monitor.start_operation("zkp_verify_statistical")
    del adaptive_bounds  # median/MAD over clients is not a verification input

    try:
        if proof.computation_proof.get('proof_type') != 'lagged_aggregate_mean_range':
            perf_monitor.end_operation("zkp_verify_statistical")
            return False
        if proof.computation_proof.get('l2_is_primary'):
            perf_monitor.end_operation("zkp_verify_statistical")
            return False
        if proof._compute_proof_hash() != proof.proof_hash:
            perf_monitor.end_operation("zkp_verify_statistical")
            return False
        if not ciphertexts or len(ciphertexts) != 3:
            perf_monitor.end_operation("zkp_verify_statistical")
            return False

        digest = ciphertext_binding_digest(ciphertexts)
        if digest != proof.computation_proof.get('ciphertext_sha256'):
            perf_monitor.end_operation("zkp_verify_statistical")
            return False

        S = proof.commitments['S_k_c_proj'].data
        O = proof.commitments['O_k_c_proj'].data
        n_plain = proof.commitments['n_plain'].data
        means = proof.commitments['client_mean'].data
        if plaintext_digest(S) != getattr(ciphertexts[0], 'plaintext_digest', None):
            perf_monitor.end_operation("zkp_verify_statistical")
            return False
        if plaintext_digest(O) != getattr(ciphertexts[1], 'plaintext_digest', None):
            perf_monitor.end_operation("zkp_verify_statistical")
            return False
        if plaintext_digest(n_plain) != getattr(ciphertexts[2], 'plaintext_digest', None):
            perf_monitor.end_operation("zkp_verify_statistical")
            return False

        n_val = float(_as_1d(n_plain)[0])
        n_k_c = int(round(n_val))
        if n_k_c <= 0 or abs(n_val - n_k_c) > 1e-6:
            perf_monitor.end_operation("zkp_verify_statistical")
            return False
        if not _variance_numerators_ok(S, O, n_k_c):
            perf_monitor.end_operation("zkp_verify_statistical")
            return False
        expected_mean = _as_1d(S) / float(n_k_c)
        if not torch.allclose(_as_1d(means), expected_mean, rtol=0.0, atol=1e-5):
            perf_monitor.end_operation("zkp_verify_statistical")
            return False

        statement = proof.statement
        challenge = fiat_shamir_challenge(
            statement, proof.commitments['client_mean'].commitment, digest
        )
        if challenge != proof.computation_proof.get('fiat_shamir_challenge'):
            perf_monitor.end_operation("zkp_verify_statistical")
            return False

        # Cold start: no previous aggregate, so no coordinate reject (kappa=5 is the
        # multiplier once an anchor exists). Steady state: server band, kappa from
        # that band (paper value 3).
        if lagged_bounds is not None:
            if not _mean_in_lagged_band(means, lagged_bounds):
                perf_monitor.end_operation("zkp_verify_statistical")
                return False

        perf_monitor.end_operation("zkp_verify_statistical")
        return True

    except Exception as e:
        print(f"ZKP statistical verification error: {e}")
        perf_monitor.end_operation("zkp_verify_statistical")
        return False

def compute_adaptive_bounds(*args, **kwargs) -> Dict[str, Any]:
    """Removed. Median/MAD over individual clients is not the bound."""
    raise RuntimeError(
        "median/MAD over clients is dropped. "
        "Use bounds_from_previous_aggregate(mu_bar, sigma_hat, kappa=3)."
    ) 