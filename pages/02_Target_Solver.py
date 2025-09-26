import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events
from scipy.optimize import minimize

# ---------------- Page setup ----------------
st.set_page_config(page_title="Target Solver – Tap to Aim", layout="wide")
st.title("🎯 Tap the Goal – Inverse Solver for Aim & RPM")

st.markdown("""
Tap (or click) on the goal to choose where you want the ball to hit.  
The solver will compute:
- **Elevation** and **Azimuth** to aim the launcher  
- **Exit speed** and a suggested **average wheel RPM**  
- A preview marker at the computed impact point
""")

# ---------------- Physics helpers ----------------
def lift_coefficient(omega_mag, v_mag, ball_radius_m, CL_max=0.25, CL_k=2.0):
    v = max(v_mag, 1e-6)
    S = (omega_mag * ball_radius_m) / v
    return CL_max * math.tanh(CL_k * S)

def simulate_flight(
    v0, elev_deg, azim_deg, h0,
    goal_distance,
    ball_mass, ball_diam, Cd, rho_air, g,
    wind_speed, wind_dir_deg,
    use_magnus=False, omega_ball_z=0.0, CL_max=0.25, CL_k=2.0,
    dt=0.003, t_max=5.0
):
    """
    Coordinates:
      x: downfield toward the goal plane, y: up, z: right (center of goal at z=0).
    wind_dir_deg is the direction the wind blows TOWARD in the x–z plane:
      0° tailwind (+x), 90° right (+z), 180° headwind (–x), 270° left (–z).
    Magnus spin is about +z for TOPSPIN (drives down), –z for BACKSPIN (lifts).
    """
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    vx = v0 * math.cos(elev) * math.cos(azim)
    vy = v0 * math.sin(elev)
    vz = v0 * math.cos(elev) * math.sin(azim)

    area = math.pi * (ball_diam/2)**2
    k_drag = 0.5 * rho_air * max(Cd, 0.0) * area  # allow Cd=0 to disable drag

    wth = math.radians(wind_dir_deg)
    wvx = wind_speed * math.cos(wth)
    wvz = wind_speed * math.sin(wth)

    x, y, z = 0.0, h0, 0.0
    t = 0.0

    crossed_goal = False
    y_at_goal = None
    z_at_goal = None

    while t < t_max and y >= 0.0:
        x_prev, y_prev, z_prev = x, y, z

        # Relative air velocity
        vrelx = vx - wvx
        vrely = vy - 0.0
        vrelz = vz - wvz
        vrel = math.sqrt(vrelx*vrelx + vrely*vrely + vrelz*vrelz) + 1e-9

        # Drag
        ax = -k_drag * vrel * vrelx / ball_mass
        ay = -g - k_drag * vrel * vrely / ball_mass
        az = -k_drag * vrel * vrelz / ball_mass

        # Magnus (lift) from spin about z
        if use_magnus and abs(omega_ball_z) > 0:
            cross_x = - omega_ball_z * vrely
            cross_y =   omega_ball_z * vrelx
            cross_mag = math.hypot(cross_x, cross_y) + 1e-12
            CL = lift_coefficient(abs(omega_ball_z), vrel, ball_diam/2, CL_max, CL_k)
            Fm = 0.5 * rho_air * area * CL * (vrel**2)
            ax += (Fm / ball_mass) * (cross_x / cross_mag)
            ay += (Fm / ball_mass) * (cross_y / cross_mag)

        # Semi-implicit Euler
        vx += ax * 0.003
        vy += ay * 0.003
        vz += az * 0.003
        x  += vx * 0.003
        y  += vy * 0.003
        z  += vz * 0.003

        t  += 0.003

        # Goal plane crossing x = goal_distance
        if not crossed_goal and x_prev <= goal_distance <= x:
            a = (goal_distance - x_prev) / (x - x_prev + 1e-12)
            y_at_goal = y_prev + a * (y - y_prev)
            z_at_goal = z_prev + a * (z - z_prev)
            crossed_goal = True

    return {
        "crossed_goal": crossed_goal,
        "y_at_goal": y_at_goal,
        "z_at_goal": z_at_goal,
    }

def rpm_from_v0(v0, wheel_radius_m, slip):
    # v0 = slip * r * omega_avg  =>  omega_avg = v0/(slip*r), rpm = omega*60/(2π)
    omega_avg = (v0 / max(slip, 1e-9)) / max(wheel_radius_m, 1e-9)
    rpm_avg = omega_avg * 60.0 / (2 * math.pi)
    return max(0.0, rpm_avg)

def simulate_to_goal(v0, elev_deg, azim_deg, **kwargs):
    res = simulate_flight(v0, elev_deg, azim_deg, kwargs["h0"], kwargs["goal_distance"],
                          kwargs["ball_mass"], kwargs["ball_diam"], kwargs["Cd"], kwargs["rho_air"], kwargs["g"],
                          kwargs["wind_speed"], kwargs["wind_dir_deg"],
                          use_magnus=kwargs["use_magnus"], omega_ball_z=kwargs["omega_ball_z"],
                          CL_max=kwargs["CL_max"], CL_k=kwargs["CL_k"])
    return res["crossed_goal"], res["y_at_goal"], res["z_at_goal"]

def solve_for_target(target_y, target_z, params):
    """
    Solve for (v0, elev, azim) that hits (target_z, target_y) at x=goal_distance.
    """
    # Decision vars: [v0, elev_deg, azim_deg]
    v0_min, v0_max = 5.0, 45.0
    elev_min, elev_max = 0.0, 40.0
    azim_min, azim_max = -20.0, 20.0

    x0 = np.array([
        np.clip(params.get("v0_guess", 20.0), v0_min, v0_max),
        np.clip(params.get("elev_guess", 12.0), elev_min, elev_max),
        np.clip(params.get("azim_guess", 0.0), azim_min, azim_max),
    ])
    bounds = [(v0_min, v0_max), (elev_min, elev_max), (azim_min, azim_max)]

    def objective(x):
        v0, elev_deg, azim_deg = x
        crossed, y_g, z_g = simulate_to_goal(v0, elev_deg, azim_deg, **params)
        if not crossed or y_g is None or z_g is None:
            return 1e6  # penalize infeasible shots
        dy = (y_g - target_y)
        dz = (z_g - target_z)
        return dy*dy*1.2 + dz*dz  # slightly weight height

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=250))
    v0, elev_deg, azim_deg = res.x
    return v0, elev_deg, azim_deg, float(res.fun), bool(res.success)

# ---------------- Sidebar parameters ----------------
with st.sidebar:
    st.header("Goal (centered at z = 0)")
    goal_width = st.slider("Goal width (m)", 2.0, 10.0, 7.31, 0.01)
    goal_height = st.slider("Goal height (m)", 1.0, 4.0, 2.44, 0.01)
    goal_distance = st.slider("Goal distance (m)", 5.0, 30.0, 11.0, 0.5)

    st.header("Launcher & Physics")
    h0 = st.slider("Release height (m)", 0.2, 2.5, 0.6, 0.05)
    wheel_radius_m = st.slider("Wheel radius (m)", 0.03, 0.20, 0.06, 0.005)
    slip = st.slider("Slip/transfer factor", 0.5, 1.0, 0.9, 0.01)
