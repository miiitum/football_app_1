import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events
from scipy.optimize import minimize

# ================= Page setup =================
st.set_page_config(page_title="Target Solver – Tap to Aim", layout="wide")
st.title("🎯 Tap the Goal – Inverse Solver for Aim & RPM")
st.markdown(
    "Tap (or click) inside the goal to choose the impact point. "
    "We’ll solve for **elevation**, **azimuth**, and **exit speed** (→ avg wheel RPM)."
)

# ================= Physics helpers =================
def lift_coefficient(omega_mag, v_mag, ball_radius_m, CL_max=0.25, CL_k=2.0):
    v = max(v_mag, 1e-6)
    S = (omega_mag * ball_radius_m) / v
    return CL_max * math.tanh(CL_k * S)

def simulate_flight(
    v0, elev_deg, azim_deg, h0, goal_distance,
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
    k_drag = 0.5 * rho_air * max(Cd, 0.0) * area

    # wind blows TOWARD wind_dir_deg in x–z plane
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
            cross_mag = math.hypot(cross_x, cross_y) + 1e-12
            CL = lift_coefficient(abs(omega_ball_z), vrel, ball_diam/2, CL_max, CL_k)
            Fm = 0.5 * rho_air * area * CL * (vrel**2)
            ax += (Fm / ball_mass) * (cross_x / cross_mag)
            ay += (Fm / ball_mass) * (cross_y / cross_mag)

        vx += ax * dt; vy += ay * dt; vz += az * dt
        x  += vx * dt; y  += vy * dt; z  += vz * dt
        t  += dt

        if not crossed_goal and x_prev <= goal_distance <= x:
            a = (goal_distance - x_prev) / (x - x_prev + 1e-12)
            y_at_goal = y_prev + a * (y - y_prev)
            z_at_goal = z_prev + a * (z - z_prev)
            crossed_goal = True

    return {"crossed_goal": crossed_goal, "y_at_goal": y_at_goal, "z_at_goal": z_at_goal}

def rpm_from_v0(v0, wheel_radius_m, slip):
    omega_avg = (v0 / max(slip, 1e-9)) / max(wheel_radius_m, 1e-9)
    return max(0.0, omega_avg * 60.0 / (2 * math.pi))

def simulate_to_goal(v0, elev_deg, azim_deg, **kwargs):
    res = simulate_flight(v0, elev_deg, azim_deg, kwargs["h0"], kwargs["goal_distance"],
                          kwargs["ball_mass"], kwargs["ball_diam"], kwargs["Cd"], kwargs["rho_air"], kwargs["g"],
                          kwargs["wind_speed"], kwargs["wind_dir_deg"],
                          use_magnus=kwargs["use_magnus"], omega_ball_z=kwargs["omega_ball_z"],
                          CL_max=kwargs["CL_max"], CL_k=kwargs["CL_k"])
    return res["crossed_goal"], res["y_at_goal"], res["z_at_goal"]

def solve_for_target(target_y, target_z, params):
    # decision vars: v0, elev, azim
    v0_min, v0_max   = 5.0, 50.0     # allow a bit more headroom
    elev_min, elev_max = 0.0, 45.0
    azim_min, azim_max = -25.0, 25.0

    x0 = np.array([
        np.clip(params.get("v0_guess", 20.0), v0_min, v0_max),
        np.clip(params.get("elev_guess", 12.0),  elev_min, elev_max),
        np.clip(params.get("azim_guess",  0.0),  azim_min, azim_max),
    ])
    bounds = [(v0_min, v0_max), (elev_min, elev_max), (azim_min, azim_max)]

    def objective(xx):
        v0, elev_deg, azim_deg = xx
        crossed, y_g, z_g = simulate_to_goal(v0, elev_deg, azim_deg, **params)
        if not crossed or y_g is None or z_g is None:
            return 1e6
        dy = (y_g - target_y); dz = (z_g - target_z)
        return dy*dy*1.1 + dz*dz

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=300))
    v0, elev_deg, azim_deg = res.x
    return v0, elev_deg, azim_deg, float(res.fun), bool(res.success)

# ================= Sidebar =================
with st.sidebar:
    st.header("Goal (centered at z = 0)")
    goal_width  = st.slider("Goal width (m)", 2.0, 10.0, 7.31, 0.01)
    goal_height = st.slider("Goal height (m)", 1.0, 4.0, 2.44, 0.01)
    goal_distance = st.slider("Goal distance (m)", 5.0, 30.0, 11.0, 0.5)

    st.header("Launcher & Physics")
    h0 = st.slider("Release height (m)", 0.2, 2.5, 0.6, 0.05)
    wheel_radius_m = st.slider("Wheel radius (m)", 0.03, 0.20, 0.06, 0.005)
    slip = st.slider("Slip/transfer factor", 0.5, 1.0, 0.9, 0.01)
    ball_mass = st.slider("Ball mass (kg)", 0.40, 0.50, 0.43, 0.005)
    ball_diam  = st.slider("Ball diameter (m)", 0.20, 0.24, 0.22, 0.005)
    Cd = st.slider("Drag coefficient Cd", 0.20, 0.45, 0.25, 0.01)
    rho_air = st.slider("Air density (kg/m³)", 1.00, 1.35, 1.225, 0.005)
    g = st.slider("Gravity (m/s²)", 9.70, 9.85, 9.81, 0.01)
    use_magnus = st.checkbox("Enable Magnus (spin lift)", True)
    omega_ball_z = st.slider("Spin ω about z (rad/s)", -200.0, 200.0, 0.0, 1.0,
                             help="Backspin < 0 (lift), Topspin > 0 (downforce)")
    CL_max = st.slider("CL_max", 0.05, 0.40, 0.25, 0.01)
    CL_k   = st.slider("CL_k", 0.5, 5.0, 2.0, 0.1)

    st.header("Wind (x–z plane)")
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 20.0, 0.0, 0.1)
    wind_dir_deg = st.slider("Wind dir (deg, blowing TOWARD)", 0.0, 360.0, 180.0, 1.0,
                             help="0° tailwind (+x), 90° right (+z), 180° headwind (–x), 270° left (–z).")

# ================= Goal figure helpers =================
def finalize_axes(fig):
    fig.update_xaxes(title_text="z (m) [left −, right +]",
                     range=[-goal_width/2 - 0.5, goal_width/2 + 0.5],
                     zeroline=True, constrain="domain")
    fig.update_yaxes(title_text="y (m)", range=[0, goal_height * 1.1])
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        hovermode="closest",          # <<< 2-D nearest (fixes y=2.44 issue)
        clickmode="event+select",     # <<< emit click + selection events
    )

def add_frame_and_net(fig):
    # Make frame/net non-pickable so clicks prefer the grid points
    frame_x = [-goal_width/2,  goal_width/2,  goal_width/2, -goal_width/2, -goal_width/2]
    frame_y = [0,              0,             goal_height,   goal_height,   0]
    fig.add_trace(go.Scatter(
        x=frame_x, y=frame_y, mode="lines", name="frame",
        hoverinfo="skip", hovertemplate=None
    ))
    nx, ny = 16, 6
    for i in range(nx + 1):
        z = -goal_width/2 + i * (goal_width / nx)
        fig.add_trace(go.Scatter(
            x=[z, z], y=[0, goal_height], mode="lines",
            line=dict(width=1, dash="dot"),
            showlegend=False, hoverinfo="skip", hovertemplate=None
        ))
    for j in range(ny + 1):
        yline = j * (goal_height / ny)
        fig.add_trace(go.Scatter(
            x=[-goal_width/2, goal_width/2], y=[yline, yline], mode="lines",
            line=dict(width=1, dash="dot"),
            showlegend=False, hoverinfo="skip", hovertemplate=None
        ))

# ---------- Static view ----------
st.subheader("Goal (static view)")
fig_static = go.Figure()
add_frame_and_net(fig_static)
finalize_axes(fig_static)
st.plotly_chart(fig_static, use_container_width=True, config={"staticPlot": True})

# ---------- Interactive view with click-grid ----------
st.subheader("Tap here to set target")
fig_interactive = go.Figure()

# 1) Invisible click-grid (first trace)
grid_nz, grid_ny = 81, 51   # denser → better tap precision
zs = np.linspace(-goal_width/2, goal_width/2, grid_nz)
ys = np.linspace(0.0, goal_height,    grid_ny)
ZZ, YY = np.meshgrid(zs, ys)
fig_interactive.add_trace(go.Scatter(
    x=ZZ.ravel(), y=YY.ravel(),
    mode="markers",
    marker=dict(size=10, opacity=0.001),
    name="clickgrid", showlegend=False,
    hoverinfo="skip", hovertemplate=None
))

# 2) Non-pickable frame/net
add_frame_and_net(fig_interactive)

finalize_axes(fig_interactive)

clicks = plotly_events(
    fig_interactive,
    click_event=True,
    select_event=True,
    hover_event=False,
    key="goal_click_page",
    override_height=520,
    override_width="100%",
)
st.caption("Tap anywhere inside the interactive goal above. (Using 2-D nearest + invisible grid).")

# Debug (optional): uncomment to inspect raw clicks
# st.write("Event:", clicks)

# ============== Manual fallback ==============
with st.expander("Or set target manually"):
    target_z_manual = st.slider("Target z (m)", -goal_width/2, goal_width/2, 0.0, 0.01)
    target_y_manual = st.slider("Target y (m)", 0.0, goal_height, goal_height * 0.5, 0.01)
    run_manual = st.button("Solve for manual target")

# Decide whether to solve
solve_now = False
target_z = None
target_y = None

if clicks and isinstance(clicks, list) and len(clicks) > 0 and ("x" in clicks[0]) and ("y" in clicks[0]):
    target_z = float(clicks[0]["x"])
    target_y = float(clicks[0]["y"])
    solve_now = True
elif run_manual:
    target_z = target_z_manual
    target_y = target_y_manual
    solve_now = True

# ============== Solve & show ==============
def do_solve_and_plot(target_z, target_y):
    params = dict(
        h0=h0, goal_distance=goal_distance,
        ball_mass=ball_mass, ball_diam=ball_diam, Cd=Cd, rho_air=rho_air, g=g,
        wind_speed=wind_speed, wind_dir_deg=wind_dir_deg,
        use_magnus=use_magnus, omega_ball_z=omega_ball_z,
        CL_max=CL_max, CL_k=CL_k,
        v0_guess=20.0, elev_guess=12.0, azim_guess=0.0
    )
    v0_sol, elev_sol, azim_sol, err, ok = solve_for_target(target_y, target_z, params)

    if ok and err < 0.06:  # slightly looser tolerance
        rpm_avg = rpm_from_v0(v0_sol, wheel_radius_m, slip)
        st.success(
            f"**Solution**\n\n"
            f"- Elevation: **{elev_sol:.2f}°**\n"
            f"- Azimuth: **{azim_sol:.2f}°** (right +)\n"
            f"- Exit speed v0: **{v0_sol:.2f} m/s** → avg wheel ~ **{rpm_avg:.0f} RPM**"
        )

        crossed, y_g, z_g = simulate_to_goal(v0_sol, elev_sol, azim_sol, **params)

        fig_show = go.Figure()
        # faint grid so it’s still clickable after solution if needed
        fig_show.add_trace(go.Scatter(x=ZZ.ravel(), y=YY.ravel(), mode="markers",
                                      marker=dict(size=10, opacity=0.0005),
                                      showlegend=False, hoverinfo="skip", hovertemplate=None))
        add_frame_and_net(fig_show)
        # target and solution markers
        fig_show.add_trace(go.Scatter(x=[target_z], y=[target_y], mode="markers+text",
                                      text=["target"], textposition="bottom center", name="target"))
        if crossed and (y_g is not None) and (z_g is not None):
            fig_show.add_trace(go.Scatter(x=[z_g], y=[y_g], mode="markers+text",
                                          text=["solution"], textposition="top center", name="solution"))
        finalize_axes(fig_show)
        st.plotly_chart(fig_show, use_container_width=True)
    else:
        st.warning("No feasible solution with current bounds/conditions. Try a lower target, enable spin, reduce headwind, or allow higher v0.")

if solve_now:
    st.markdown(f"**Target selected:** z = {target_z:.2f} m, y = {target_y:.2f} m")
    do_solve_and_plot(target_z, target_y)
