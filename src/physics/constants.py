# -*- coding: utf-8 -*-
"""
This module defines all physical and simulation constants used in the project,
based on the provided technical specification.

Centralizing constants ensures consistency and simplifies tuning of the simulation.
All constants follow the PEP 8 naming convention (UPPER_SNAKE_CASE).
"""

# I. General Physical Constants
# =============================================================================

# WGS-84 Earth ellipsoid parameters
EARTH_EQUATORIAL_RADIUS_KM = 6378.137  # Equatorial radius in km
EARTH_FLATTENING_F = 1 / 298.257223563 # Flattening factor

# Boltzmann constant in Joules per Kelvin (J/K)
BOLTZMANN_K = 1.380649e-23

# Speed of light in vacuum in meters per second (m/s)
SPEED_OF_LIGHT_M_S = 299792458.0


# II. Communication Link Constants
# =============================================================================

# --- Common Link Constants ---

# Constant part of the Free Space Path Loss (FSPL) formula
FSPL_BASE_LOSS_DB = 92.45

# --- Satellite-to-Ground Link (Downlink) Constants ---

# Environmental attenuation loss in dB. Assumed constant as per the spec.
DOWNLINK_ENV_LOSS_DB = 8.0  # Slightly reduced to improve availability

# Rician factor for the small-scale fading model
DOWNLINK_RICIAN_FACTOR_K = 12.0  # Strong LoS component

# Minimum Carrier-to-Noise Ratio (CNR) threshold for successful demodulation in dB
# Note: The link model will no longer hard-zero capacity below this; instead we use soft floors.
DOWNLINK_CNR_THRESHOLD_DB = 0.0

# --- Inter-Satellite Link (ISL) Constants ---

# Transmitter power in Watts (W) - increased to make long ISL more feasible
ISL_TRANSMITTER_POWER_W = 5.0

# Antenna 3dB beamwidth for both theta and phi axes in degrees
ISL_ANTENNA_3DB_BEAMWIDTH_DEG = 10.0

# Effective aperture of the antenna in square meters (m^2).
# Previously 0.2 cm^2 (~0.00002 m^2) was unrealistically small; increase to 0.01 m^2.
ISL_ANTENNA_EFFECTIVE_APERTURE_M2 = 0.01

# Standard deviation of the pointing error in radians (reduced for stability)
ISL_POINTING_ERROR_STD_DEV_RAD = 0.01

# System noise temperature components in Kelvin (K)
ISL_NOISE_TEMP_RECEIVER_K = 500.0   # Reduced receiver noise temperature
ISL_NOISE_TEMP_SPACE_K = 6000.0     # Ts: Noise from surrounding space objects
ISL_NOISE_TEMP_CMB_K = 2.73         # T_cmb: Cosmic Microwave Background

# Bandwidth as a percentage of the carrier frequency
ISL_BANDWIDTH_TO_FREQUENCY_RATIO = 0.02

# --- Link modeling stabilizers for RL ---
# Toggle small-scale/random effects to stabilize learning
LINKS_USE_RANDOM_POINTING = False
LINKS_USE_RICIAN_FADING = False

# Capacity/SNR floors to avoid infinite latencies
MIN_ISL_SNR_DB = -10.0
MIN_DOWNLINK_CNR_DB = -10.0
MIN_ISL_CAPACITY_BPS = 1e5           # 100 kbps floor
MIN_DOWNLINK_CAPACITY_BPS = 1e5      # 100 kbps floor


# III. Information Value (VoI) Constants
# =============================================================================

# Decay factor for the exponential Age of Information (AoI) model
# A higher value means information becomes stale faster.
AOI_DECAY_RATE_LAMBDA = 0.01 # Placeholder value
