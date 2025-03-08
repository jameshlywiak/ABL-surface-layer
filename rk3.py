import numpy as np

def compute_tendencies(temperature, wind_u, wind_v, tke, height, initial_wind_u):
    dU_dz = np.gradient(wind_u, height)
    dV_dz = np.gradient(wind_v, height)
    dT_dz = np.gradient(temperature, height)

    # Compute advection using upstream profiles (wind_u at t=0)
    adv_u = np.gradient(initial_wind_u * wind_u, height)
    adv_v = np.gradient(initial_wind_u * wind_v, height)
    adv_t = np.gradient(initial_wind_u * temperature, height)

    # Compute tendencies
    du_tend = -tke * dU_dz - adv_u
    dv_tend = -tke * dV_dz - adv_v
    dt_tend = -tke * dT_dz - adv_t

    return du_tend, dv_tend, dt_tend

def rk3(temperature, wind_u, wind_v, tke, dt, height, initial_wind_u):
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
    initial_temperature = temperature.copy()
    initial_wind_v = wind_v.copy()

    # Stage 1
    du1, dv1, dt1 = compute_tendencies(temperature, wind_u, wind_v, tke, height, initial_wind_u)
    temp1 = temperature + dt * dt1
    u1 = wind_u + dt * du1
    v1 = wind_v + dt * dv1

    # Stage 2
    du2, dv2, dt2 = compute_tendencies(temp1, u1, v1, tke, height, initial_wind_u)
    temp2 = 0.75 * temperature + 0.25 * (temp1 + dt * dt2)
    u2 = 0.75 * wind_u + 0.25 * (u1 + dt * du2)
    v2 = 0.75 * wind_v + 0.25 * (v1 + dt * dv2)

    # Stage 3
    du3, dv3, dt3 = compute_tendencies(temp2, u2, v2, tke, height, initial_wind_u)
    temperature = (1 / 3) * temperature + (2 / 3) * (temp2 + dt * dt3)
    wind_u = (1 / 3) * wind_u + (2 / 3) * (u2 + dt * du3)
    wind_v = (1 / 3) * wind_v + (2 / 3) * (v2 + dt * dv3)

    return temperature, wind_u, wind_v
