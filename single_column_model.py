import numpy as np
from my82 import mellor_yamada_tke
from rk3 import rk3

def surface_fluxes(temperature, wind_u, wind_v, height):
    """
    Placeholder for surface flux calculations.
    Returns sensible heat flux, latent heat flux, and momentum fluxes.
    """
    # Example: Simplified fluxes (to be replaced with actual parameterization)
    return 10.0, 5.0, 0.01, 0.01  # Sensible heat flux, latent heat flux, tau_u, tau_v

def vertical_mixing(temperature, wind_u, wind_v, pressure, height, tke, dt):
    """
    Perform vertical mixing using the Mellor-Yamada scheme.
    """
    return mellor_yamada_tke(temperature, wind_u, wind_v, pressure, height, tke, dt)

def update_wind_temperature(temperature, wind_u, wind_v, tke, dt, height, initial_wind_u):
    """
    Update wind and temperature using a Runge-Kutta 3-step time-stepping scheme.

    Parameters:
    - temperature: 1D numpy array of temperature (K) at each vertical level.
    - wind_u: 1D numpy array of u-component of wind (m/s) at each vertical level.
    - wind_v: 1D numpy array of v-component of wind (m/s) at each vertical level.
    - tke: 1D numpy array of turbulence kinetic energy (m^2/s^2) at each vertical level.
    - dt: Time step (s).
    - height: 1D numpy array of height (m) at each vertical level.
    - initial_wind_u: 1D numpy array of u-component of wind at t=0 (m/s).

    Returns:
    - temperature: Updated temperature array.
    - wind_u: Updated u-component of wind.
    - wind_v: Updated v-component of wind.
    """

    return rk3(temperature, wind_u, wind_v, tke, dt, height, initial_wind_u)

def plotting(z, tke, ts, **params):

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.plot(tke, z, label=f't={ts}')
    fig.savefig(f'tke_my82_t{ts}.png')

    return None

if __name__ == "__main__":
    # Initialize model parameters
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

    for step in range(num_timesteps):
        # Calculate surface fluxes
        sensible_heat_flux, latent_heat_flux, tau_u, tau_v = surface_fluxes(temperature, wind_u, wind_v, height)
        
        # Update state variables due to surface fluxes (e.g., modify temperature at the lowest level)
        temperature[0] += dt * sensible_heat_flux / (1004 * 1.2)  # Simplified energy conservation

        # Perform vertical mixing
        tke = vertical_mixing(temperature, wind_u, wind_v, pressure, height, tke, dt)
        
        # Update wind and temperature profiles using turbulent diffusion coefficients
        # (This requires a full mixing scheme, including flux divergence computation.)
        
        # Update wind and temperature using RK3
        temperature, wind_u, wind_v = update_wind_temperature(temperature, wind_u, wind_v, tke, dt, height, initial_wind_u)
        
        print(f"Timestep {step + 1} completed. TKE: {tke}")