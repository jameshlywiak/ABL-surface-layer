import numpy as np
import matplotlib.pyplot as plt
from single_column_model import plotting

def mellor_yamada_tke(temperature, wind_u, wind_v, pressure, height, tke, dt):
    """
    Mellor-Yamada Level 1.5 boundary layer scheme.

    Parameters:
    - temperature: 1D numpy array of temperature (K) at each vertical level.
    - wind_u: 1D numpy array of u-component of wind (m/s) at each vertical level.
    - wind_v: 1D numpy array of v-component of wind (m/s) at each vertical level.
    - pressure: 1D numpy array of pressure (Pa) at each vertical level.
    - height: 1D numpy array of height (m) at each vertical level.
    - tke: 1D numpy array of turbulence kinetic energy (m^2/s^2) at each vertical level.
    - dt: Time step (s).

    Returns:
    - tke: 1D numpy array of updated turbulence kinetic energy (m^2/s^2) at each vertical level.
    """
    # Constants
    g = 9.81  # gravitational acceleration (m/s^2)
    kappa = 0.4  # von Karman constant
    c_e1, c_e2, c_e3 = 1.44, 0.92, 0.4  # closure constants
    tke_min = 1e-6  # minimum TKE to avoid numerical issues

    # Number of levels
    nz = len(height)

    # Initialize arrays
    mixing_length = np.zeros(nz)  # Mixing length

    # Calculate gradients and buoyancy
    dU_dz = np.gradient(wind_u, height)
    dV_dz = np.gradient(wind_v, height)
    dT_dz = np.gradient(temperature, height)
    dP_dz = np.gradient(pressure, height)

    # Potential temperature (theta)
    theta = temperature * (100000 / pressure) ** (287 / 1004)
    dTheta_dz = np.gradient(theta, height)

    for k in range(1, nz - 1):
        # Compute mixing length
        mixing_length[k] = kappa * (height[k+1] - height[k-1]) / 2

        # Compute shear production
        shear_prod = mixing_length[k]**2 * (dU_dz[k]**2 + dV_dz[k]**2)

        # Compute buoyancy production
        buoy_prod = -g / theta[k] * tke[k] * dTheta_dz[k]

        # Compute dissipation
        dissipation = c_e2 * tke[k]**(3/2) / mixing_length[k]

        # Update TKE using prognostic equation
        tke[k] += dt * (shear_prod + c_e3 * buoy_prod - dissipation)

        # Enforce minimum TKE
        tke[k] = max(tke[k], tke_min)

    return tke

# Example usage
if __name__ == "__main__":
    # Sample atmospheric profile
    height = np.linspace(0, 1000, 20)  # 20 levels from 0 to 1000 m
    temperature = 290 - 0.0065 * height  # Linear temperature profile (K)
    wind_u = np.linspace(5, 10, 20)  # Linear u-wind profile (m/s)
    wind_v = np.zeros(20)  # Constant v-wind profile (m/s)
    pressure = 100000 * (1 - 0.0065 * height / 288.15)**5.255  # Barometric formula (Pa)
    dt = 60  # 1-minute time step (s)
    num_timesteps = 10  # Number of timesteps to run

    # Initialize TKE
    tke = np.full(len(height), 1e-6)  # Initial TKE (m^2/s^2)

    # Store initial wind profile
    initial_wind_u = wind_u.copy()

    # Run the boundary layer scheme over multiple timesteps
    for step in range(num_timesteps):
        # Update TKE
        tke = mellor_yamada_tke(temperature, wind_u, wind_v, pressure, height, tke, dt)
        
        print(f"Timestep {step + 1}, TKE: {tke}")