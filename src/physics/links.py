# -*- coding: utf-8 -*-
"""
This module implements the communication link models as per the specification.

It includes calculations for path loss, antenna gains, noise, and ultimately
the channel capacity for both Downlink and Inter-Satellite Links (ISL).
"""

import numpy as np
from scipy.stats import rice
from scipy.special import erfinv

from src.physics import constants

# Create a single random number generator for the module
_rng = np.random.default_rng()

def get_distance_km(pos1_km: np.ndarray, pos2_km: np.ndarray) -> float:
    """Calculates the Euclidean distance between two points in km."""
    return np.linalg.norm(pos1_km - pos2_km)

def db_to_linear(db_value: float) -> float:
    """Converts a value from decibels (dB) to a linear scale."""
    return 10 ** (db_value / 10.0)

def linear_to_db(linear_value: float) -> float:
    """Converts a value from a linear scale to decibels (dB)."""
    if linear_value <= 0:
        return -np.inf
    return 10 * np.log10(linear_value)

# --- Inter-Satellite Link (ISL) Models ---

def _get_isl_pointing_loss_db() -> float:
    """
    Calculates a random pointing loss sample based on the specified PDF.
    Uses inverse transform sampling on the provided CDF.
    CDF: P(loss <= vartheta) = 1 - erf(sqrt(-ln(vartheta) / (2*alpha*sigma_p^2)))
    """
    # 1. Calculate alpha
    theta_3db_rad = np.deg2rad(constants.ISL_ANTENNA_3DB_BEAMWIDTH_DEG)
    alpha = (2 * np.log(2)) / (theta_3db_rad**2)
    
    # 2. Inverse transform sampling
    sigma_p = constants.ISL_POINTING_ERROR_STD_DEV_RAD
    u = _rng.uniform(0, 1)
    
    # Handle edge case for erfinv(1)
    if u == 0:
        return 0.0

    term_in_sqrt = erfinv(1 - u)**2 * 2 * alpha * (sigma_p**2)
    vartheta = np.exp(-term_in_sqrt)
    
    # vartheta is the linear loss, convert to dB (will be <= 0)
    return linear_to_db(vartheta)

def calculate_isl_fspl_db(distance_km: float, frequency_ghz: float) -> float:
    c = constants.SPEED_OF_LIGHT_M_S
    d_m = distance_km * 1000
    f_hz = frequency_ghz * 1e9
    if d_m <= 0 or f_hz <= 0: return np.inf
    lambda_m = c / f_hz
    fspl_linear = (4 * np.pi * d_m / lambda_m) ** 2
    return linear_to_db(fspl_linear)

def calculate_isl_antenna_gain_db(lambda_m: float) -> float:
    d_area_m2 = constants.ISL_ANTENNA_EFFECTIVE_APERTURE_M2
    d_diam_m = 2 * np.sqrt(d_area_m2 / np.pi)
    gain_linear = (np.pi * d_diam_m / lambda_m) ** 2
    return linear_to_db(gain_linear)

def calculate_system_noise_power_dbw(bandwidth_hz: float, noise_temp_k: float) -> float:
    noise_power_watts = constants.BOLTZMANN_K * noise_temp_k * bandwidth_hz
    return linear_to_db(noise_power_watts)

def get_isl_capacity_bps(distance_km: float, frequency_ghz: float) -> float:
    snr_db = get_isl_snr_db(distance_km, frequency_ghz)
    if np.isinf(snr_db):
        return 0
    snr_linear = db_to_linear(snr_db)
    bandwidth_hz = constants.ISL_BANDWIDTH_TO_FREQUENCY_RATIO * (frequency_ghz * 1e9)
    capacity_bps = bandwidth_hz * np.log2(1 + snr_linear)
    return capacity_bps

# --- Downlink Models ---

def _get_downlink_rician_loss_db() -> float:
    """
    Generates a random sample for small-scale fading using a Rician model.
    The power is normalized to have a mean of 1 (0 dB).
    """
    K = constants.DOWNLINK_RICIAN_FACTOR_K
    # Normalize power E[r^2] = 1
    sigma_sq = 1 / (2 * K + 2)
    sigma = np.sqrt(sigma_sq)
    A = np.sqrt(2 * K * sigma_sq)
    b = A / sigma
    
    # Sample envelope and calculate power (r^2)
    r = rice.rvs(b, scale=sigma, random_state=_rng)
    r_sq = r**2
    return linear_to_db(r_sq)

def get_isl_snr_db(distance_km: float, frequency_ghz: float) -> float:
    p_t_dbw = linear_to_db(constants.ISL_TRANSMITTER_POWER_W)
    
    fspl_db = calculate_isl_fspl_db(distance_km, frequency_ghz)
    pointing_loss_db = _get_isl_pointing_loss_db()
    l_t_db = fspl_db - pointing_loss_db # loss is subtracted from gain, so add here
    
    lambda_m = constants.SPEED_OF_LIGHT_M_S / (frequency_ghz * 1e9)
    g_db = calculate_isl_antenna_gain_db(lambda_m)
    
    bandwidth_hz = constants.ISL_BANDWIDTH_TO_FREQUENCY_RATIO * (frequency_ghz * 1e9)
    noise_temp = constants.ISL_NOISE_TEMP_RECEIVER_K + constants.ISL_NOISE_TEMP_SPACE_K + constants.ISL_NOISE_TEMP_CMB_K
    noise_power_dbw = calculate_system_noise_power_dbw(bandwidth_hz, noise_temp)
    
    snr_db = p_t_dbw + g_db + g_db - l_t_db - noise_power_dbw
    return snr_db

def get_downlink_capacity_bps(distance_km: float, frequency_ghz: float) -> float:
    """
    Calculates the downlink capacity based on the CNR formula.
    This is a placeholder for a full CNR calculation which would require more constants
    like EIRP, G/T etc. We simulate the CNR directly for now.
    """
    # Simplified CNR calculation
    # A real implementation would require EIRP, G/T, etc.
    # We simulate a base CNR that degrades with distance and random fading.
    base_cnr_db = 50.0 # A strong signal base CNR
    fspl_db = constants.FSPL_BASE_LOSS_DB + 20 * np.log10(distance_km) + 20 * np.log10(frequency_ghz)
    rician_loss_db = _get_downlink_rician_loss_db()
    
    # Final CNR
    cnr_db = base_cnr_db - (fspl_db - constants.FSPL_BASE_LOSS_DB) + rician_loss_db - constants.DOWNLINK_ENV_LOSS_DB
    
    if cnr_db < constants.DOWNLINK_CNR_THRESHOLD_DB:
        return 0.0

    cnr_linear = db_to_linear(cnr_db)
    # Assuming bandwidth is a fraction of frequency for simplicity, similar to ISL
    bandwidth_hz = 0.01 * (frequency_ghz * 1e9)
    capacity_bps = bandwidth_hz * np.log2(1 + cnr_linear)
    return capacity_bps

# --- Uplink Models ---

def get_uplink_capacity_bps(distance_km: float, frequency_ghz: float) -> float:
    """
    Calculates the uplink capacity from a UE to a satellite.
    This is a simplified model similar to the downlink.
    """
    # UEs have lower power, so we assume a lower base CNR.
    base_cnr_db = 30.0 
    fspl_db = constants.FSPL_BASE_LOSS_DB + 20 * np.log10(distance_km) + 20 * np.log10(frequency_ghz)
    # Re-using the same Rician fading model for simplicity
    rician_loss_db = _get_downlink_rician_loss_db()
    
    # Final CNR, using the same environmental loss for simplicity
    cnr_db = base_cnr_db - (fspl_db - constants.FSPL_BASE_LOSS_DB) + rician_loss_db - constants.DOWNLINK_ENV_LOSS_DB

    # Re-using the same CNR threshold for now
    if cnr_db < constants.DOWNLINK_CNR_THRESHOLD_DB:
        return 0.0

    cnr_linear = db_to_linear(cnr_db)
    # Assuming a smaller bandwidth for UE uplink (e.g., 0.1% of frequency)
    bandwidth_hz = 0.001 * (frequency_ghz * 1e9) 
    capacity_bps = bandwidth_hz * np.log2(1 + cnr_linear)
    return capacity_bps
