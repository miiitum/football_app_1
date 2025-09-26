import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events
from scipy.optimize import minimize

st.set_page_config(page_title="Target Solver – Tap to Aim", layout="wide")
st.title("🎯 Tap the Goal – Inverse Solver for Aim & RPM")

st.markdown("""
Tap (or click) on the goal to choose where you want the ball to hit.  
The solver will compute:
- **Elevation** and **Azimuth** to aim the launcher
- **Exit speed** and a suggested **average wheel RPM**
- An optional **preview** of the resulting shot
""")

# ---------------- Physics helpers (standalone copy) ----------------
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
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    vx = v0 * math.cos(elev) * math.cos(azim)
    vy = v0 * math.sin(elev)
    vz = v0 * math.cos(elev) * math.sin(azim)

    area = math.pi * (ball_diam/2)**2
    k_drag = 0.5 * rho_air * Cd * area

    wth = math.radians(wind_dir_deg)
    wvx = wind_speed * math.cos(wth)
    wvz = wind_speed * math.sin(wth)

    x, y, z = 0.0, h0, 0.0
    t = 0.0
    xs = [x]; ys = [y]; zs = [z]

    crossed_goal = False
    y_at_goal = None
    z_at_goal = None

    while t < t_max and y >= 0.0:
        x_prev, y_prev, z_prev = x, y, z

        vrelx = vx - wvx
        vrely = vy - 0.0
        vrelz = vz - wvz
        vrel = math.sqrt(vrelx*vrelx + vrely*vrely + vrelz*vrelz) + 1e-9

        ax = -k_drag * vrel * vrelx / ball_mass
        ay = -g - k_drag * vrel * vrely / ball_mass
        az = -k_drag * vrel * vrelz / ball_mass

        if use_magnus and abs(omega_ball_z) > 0:
            cross_x = - omega_ball_z * vrely
            cross_y =   omega_ball_z * vrelx
            cross_z = 0.0
            cross_mag = math.sqrt(cross_x*cross_x + cross_y*cross_y) + 1e-12
            CL = lift_coefficient(abs(omega_ball_z), vrel, ball_diam/2, CL_max, CL_k)
            Fm = 0.5 * rho_air * area * CL * (vrel**2)
            ax += (Fm / ball_mass) * (cross_x / cross_mag)
            ay += (Fm / ball_mass) * (cross_y / cross_mag)

        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        x  += vx * dt
        y  += vy * dt
        z  += vz * dt

        t  += dt
        xs.append(x); ys.append(y); zs.append(z)

        if x_prev <= goal_distance <= x and not crossed_goal:
            a = (goal_distance - x_prev) / (x - x_prev + 1e-12)
            y_at_goal = y_prev + a * (y - y_prev)
            z_at_goal = z_prev + a * (z - z_prev)
            crossed_goal = True

    return {
        "xs": np.array(xs), "ys": np.array(ys), "zs": np.array(zs),
        "crossed_goal": crossed_goal,
        "y_at_goal": y_at_goal,
        "z_at_goal": z_at_goal,
    }

def rpm_from_v0(v0, wheel_radius_m, slip):
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
    v0_min, v0_max = 5.0, 45.0
    elev_min, elev_max = 0.0, 40.0
    azim_min, azim_max = -20.0, 20.0

    x0 = np.array([min(max(params.get("v0_guess", 20.0), v0_min), v0_max),
                   min(max(params.get("elev_guess", 12.0), elev_min), elev_max),
                   min(max(params.get("azim_guess", 0.0), azim_min), azim_max)])
    bounds = [(v0_min, v0_max), (elev_min, elev_max), (azim_min, azim_max)]

    def objective(x):
        v0, elev_deg, azim_deg = x
        crossed, y_g, z_g = simulate_to_goal(v0, elev_deg, azim_deg, **params)
        if not crossed or (y_g is None) or (z_g is None):
            return 1e6
        dy = (y_g - target_y)
        dz = (z_g - target_z)
        return dy*dy*1.2 + dz*dz

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=250))
    v0, elev_deg, azim_deg = res.x
    return v0, elev_deg, azim_deg, res.fun, res.success

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
    ball_mass = st.slider("Ball mass (kg)", 0.40, 0.50, 0.43, 0.005)
    ball_diam = st.slider("Ball diameter (m)", 0.20, 0.24, 0.22, 0.005)
    Cd = st.slider("Drag coefficient Cd", 0.20, 0.45, 0.25, 0.01)
    rho_air = st.slider("Air density (kg/m³)", 1.00, 1.35, 1.225, 0.005)
    g = st.slider("Gravity (m/s²)", 9.70, 9.85, 9.81, 0.01)
    use_magnus = st.checkbox("Enable Magnus (spin lift)", True)
    omega_ball_z = st.slider("Spin ω about z (rad/s)", -200.0, 200.0, 0.0, 1.0, help="Backspin < 0, Topspin > 0")
    CL_max = st.slider("CL_max", 0.05, 0.40, 0.25, 0.01)
    CL_k = st.slider("CL_k", 0.5, 5.0, 2.0, 0.1)

    st.header("Wind (x–z plane)")
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 20.0, 0.0, 0.1)
    wind_dir_deg = st.slider("Wind direction (deg, blowing TOWARD)", 0.0, 360.0, 180.0, 1.0,
                             help="0° tailwind (+x), 90° right (+z), 180° headwind (–x), 270° left (–z).")

# ---------------- Goal figure (tap target) ----------------
fig_goal = go.Figure()
fig_goal.add_shape(type="rect",
                   x0=-goal_width/2, y0=0, x1=goal_width/2, y1=goal_height,
                   line=dict(width=4))

# Net grid
nx, ny = 16, 6
for i in range(nx+1):
    z = -goal_width/2 + i * (goal_width/nx)
    fig_goal.add_shape(type="line", x0=z, x1=z, y0=0, y1=goal_height, line=dict(width=1, dash="dot"))
for j in range(ny+1):
    yline = j * (goal_height/ny)
    fig_goal.add_shape(type="line", x0=-goal_width/2, x1=goal_width/2, y0=yline, y1=yline, line=dict(width=1, dash="dot"))

fig_goal.update_xaxes(title_text="z (m) [left -, right +]",
                      range=[-goal_width/2-0.5, goal_width/2+0.5], zeroline=True)
fig_goal.update_yaxes(title_text="y (m)", range=[0, goal_height*1.1])
fig_goal.update_layout(margin=dict(l=10,r=10,t=30,b=10))

clicks = plotly_events(fig_goal, click_event=True, hover_event=False, select_event=False, key="goal_click_page")
st.caption("Tap/click inside the goal to set a target impact point.")

# ---------------- Solve when target selected ----------------
if clicks:
    target_z = clicks[0]["x"]
    target_y = clicks[0]["y"]
    st.markdown(f"**Target selected:** z = {target_z:.2f} m, y = {target_y:.2f} m")

    params = dict(
        h0=h0, goal_distance=goal_distance,
        ball_mass=ball_mass, ball_diam=ball_diam, Cd=Cd, rho_air=rho_air, g=g,
        wind_speed=wind_speed, wind_dir_deg=wind_dir_deg,
        use_magnus=use_magnus, omega_ball_z=omega_ball_z,
        CL_max=CL_max, CL_k=CL_k,
        v0_guess=20.0, elev_guess=12.0, azim_guess=0.0
    )

    v0_sol, elev_sol, azim_sol, err, ok = solve_for_target(target_y, target_z, params)

    if ok and err < 0.05:
        rpm_avg = rpm_from_v0(v0_sol, wheel_radius_m, slip)
        st.success(
            f"**Solution**\n\n"
            f"- Elevation: **{elev_sol:.2f}°**\n"
            f"- Azimuth: **{azim_sol:.2f}°** (right +)\n"
            f"- Exit speed v0: **{v0_sol:.2f} m/s** → avg wheel ~ **{rpm_avg:.0f} RPM**"
        )
        st.info("Tip: map avg RPM to your top/bottom wheels based on desired spin (e.g., backspin for lift).")
    else:
        st.warning("No feasible solution with current bounds/conditions. Try a lower target, less headwind, or enable spin.")
