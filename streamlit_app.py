
import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events
from scipy.optimize import minimize


# ----------------------------------
# Helper physics functions
# ----------------------------------
def exit_speed_two_wheels(wheel_radius_m, rpm_top, rpm_bottom, slip_factor):
    """Approximate ball exit speed from two contra-rotating wheels (vertical stack).
    Uses average surface speed times a slip/transfer factor.
    """
    omega_top = 2 * math.pi * rpm_top / 60.0
    omega_bot = 2 * math.pi * rpm_bottom / 60.0
    v_surface_avg = wheel_radius_m * 0.5 * (omega_top + omega_bot)
    return slip_factor * v_surface_avg, omega_top, omega_bot

def ball_spin_from_wheels(omega_top, omega_bot, wheel_radius_m, ball_radius_m, spin_coupling=0.6):
    """Estimate ball spin about z-axis (backspin/topspin) from RPM difference.
    Positive omega_z means TOPSPIN (drives ball down); negative means BACKSPIN (lifts).
    We model: omega_z ≈ spin_coupling * (omega_top - omega_bot) * (wheel_radius / ball_radius).
    """
    return spin_coupling * (omega_top - omega_bot) * (wheel_radius_m / max(ball_radius_m, 1e-6))

def lift_coefficient(omega_mag, v_mag, ball_radius_m, CL_max=0.25, CL_k=2.0):
    """Simple spin parameter model: S = (omega * r) / v, CL = CL_max * tanh(CL_k * S)."""
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
      - x: downfield toward the goal plane
      - y: vertical up
      - z: to shooter's right

    Wind convention: wind_dir_deg is the direction the wind is BLOWING TOWARD, in x–z.
      0°  = tailwind (+x), 90° = +z (right), 180° = headwind (–x), 270° = –z (left).
    Spin (Magnus): omega vector is (0, 0, omega_ball_z). Backspin => omega_ball_z < 0.
    """
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    # Initial velocity components
    vx = v0 * math.cos(elev) * math.cos(azim)
    vy = v0 * math.sin(elev)
    vz = v0 * math.cos(elev) * math.sin(azim)

    # Constants
    area = math.pi * (ball_diam/2)**2
    k_drag = 0.5 * rho_air * Cd * area

    # Wind components
    wth = math.radians(wind_dir_deg)
    wvx = wind_speed * math.cos(wth)
    wvz = wind_speed * math.sin(wth)

    # State
    x, y, z = 0.0, h0, 0.0
    t = 0.0
    xs = [x]; ys = [y]; zs = [z]

    crossed_goal = False
    x_goal_cross = None
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

        # Magnus (if enabled)
        if use_magnus and abs(omega_ball_z) > 0:
            # omega = (0, 0, omega_ball_z)
            # direction ~ (omega × v_rel) / |omega × v_rel|
            cross_x = 0.0 * vrelx - omega_ball_z * vrely      # (ωy*vz - ωz*vy) with ωy=0, ωz=omega_ball_z
            cross_y = omega_ball_z * vrelx - 0.0 * vrelz       # (ωz*vx - ωx*vz) with ωx=0
            cross_z = 0.0                                      # (ωx*vy - ωy*vx) with ωx=ωy=0
            cross_mag = math.sqrt(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z) + 1e-12

            # Lift coefficient based on spin parameter
            CL = lift_coefficient(abs(omega_ball_z), vrel, ball_diam/2, CL_max, CL_k)
            Fm = 0.5 * rho_air * area * CL * (vrel**2)
            ax += (Fm / ball_mass) * (cross_x / cross_mag)
            ay += (Fm / ball_mass) * (cross_y / cross_mag)
            az += (Fm / ball_mass) * (cross_z / cross_mag)

        # Semi-implicit Euler
        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        x  += vx * dt
        y  += vy * dt
        z  += vz * dt

        t  += dt
        xs.append(x); ys.append(y); zs.append(z)

        # Goal plane crossing x = goal_distance
        if (not crossed_goal) and x_prev <= goal_distance <= x:
            a = (goal_distance - x_prev) / (x - x_prev + 1e-12)
            y_cross = y_prev + a * (y - y_prev)
            z_cross = z_prev + a * (z - z_prev)
            x_goal_cross = goal_distance
            y_at_goal = y_cross
            z_at_goal = z_cross
            crossed_goal = True

    # Ground impact interpolation to y=0
    range_ground = None
    if ys[-1] < 0 and len(ys) >= 2:
        y0_prev, y0 = ys[-2], ys[-1]
        x0_prev, x0 = xs[-2], xs[-1]
        a = (0 - y0_prev) / (y0 - y0_prev + 1e-12)
        range_ground = x0_prev + a * (x0 - x0_prev)

    return {
        "xs": np.array(xs), "ys": np.array(ys), "zs": np.array(zs),
        "crossed_goal": crossed_goal,
        "x_goal_cross": x_goal_cross,
        "y_at_goal": y_at_goal,
        "z_at_goal": z_at_goal,
        "range_ground": range_ground,
    }

def goal_hit_verdict(crossed_goal, y_at_goal, z_at_goal, goal_w, goal_h):
    inside_h = (y_at_goal is not None) and (0.0 <= y_at_goal <= goal_h)
    inside_w = (z_at_goal is not None) and (abs(z_at_goal) <= goal_w/2)
    return bool(crossed_goal and inside_h and inside_w)


# ----------------------------------
# UI
# ----------------------------------
st.set_page_config(page_title="Two-Wheel Football Launcher", layout="wide")
st.title("Two-Flywheel Football Launcher – Physics + Goal Visualization")

with st.sidebar:
    st.header("Flywheels (vertical stack)")
    wheel_radius_m = st.slider("Wheel radius (m)", 0.03, 0.20, 0.06, 0.005)
    rpm_top = st.slider("Top wheel RPM", 500, 6000, 3200, 50)
    rpm_bottom = st.slider("Bottom wheel RPM", 500, 6000, 3800, 50)
    slip_factor = st.slider("Slip/transfer factor", 0.5, 1.0, 0.9, 0.01)

    st.header("Launch Geometry")
    elev_deg = st.slider("Elevation angle (deg)", 0.0, 45.0, 14.0, 0.5)
    azim_deg = st.slider("Azimuth (deg, + right, - left)", -20.0, 20.0, 0.0, 0.5)
    h0 = st.slider("Release height (m)", 0.2, 2.5, 0.6, 0.05)

    st.header("Goal (face is 7.31 × 2.44 m by default)")
    goal_distance = st.slider("Goal distance (m)", 5.0, 30.0, 11.0, 0.5)
    goal_width = st.slider("Goal width (m)", 2.0, 10.0, 7.31, 0.01)
    goal_height = st.slider("Goal height (m)", 1.0, 4.0, 2.44, 0.01)

    st.header("Physics")
    use_drag = st.checkbox("Use aerodynamic drag", True)
    ball_mass = st.slider("Ball mass (kg)", 0.40, 0.50, 0.43, 0.005)
    ball_diam = st.slider("Ball diameter (m)", 0.20, 0.24, 0.22, 0.005)
    Cd = st.slider("Drag coefficient Cd", 0.20, 0.45, 0.25, 0.01)
    rho_air = st.slider("Air density (kg/m³)", 1.00, 1.35, 1.225, 0.005)
    g = st.slider("Gravity (m/s²)", 9.70, 9.85, 9.81, 0.01)

    st.header("Wind (x–z plane)")
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 20.0, 0.0, 0.1)
    wind_dir_deg = st.slider("Wind direction (deg, blowing TOWARD)", 0.0, 360.0, 180.0, 1.0,
                             help="0° tailwind (+x), 90° right (+z), 180° headwind (–x), 270° left (–z).")

    st.header("Spin / Magnus")
    enable_magnus = st.checkbox("Enable Magnus (spin lift)", True)
    spin_coupling = st.slider("Spin coupling (0–1)", 0.0, 1.0, 0.6, 0.01)
    CL_max = st.slider("CL_max (lift cap)", 0.05, 0.40, 0.25, 0.01)
    CL_k = st.slider("CL_k (lift response)", 0.5, 5.0, 2.0, 0.1)

# Exit speed & spin
v0, omt, omb = exit_speed_two_wheels(wheel_radius_m, rpm_top, rpm_bottom, slip_factor)
ball_radius = ball_diam/2
omega_ball_z = ball_spin_from_wheels(omt, omb, wheel_radius_m, ball_radius, spin_coupling=spin_coupling)
# Convention reminder: omega_ball_z > 0 => TOPSPIN (downforce), < 0 => BACKSPIN (lift)

st.markdown(f"**Exit speed:** {v0:.2f} m/s  |  **Spin ω** (about z): {omega_ball_z:.1f} rad/s "
            f"({'topspin' if omega_ball_z>0 else 'backspin' if omega_ball_z<0 else 'none'})")

# Simulate
Cd_eff = Cd if use_drag else 0.0
res = simulate_flight(v0, elev_deg, azim_deg, h0, goal_distance,
                      ball_mass, ball_diam, Cd_eff, rho_air, g,
                      wind_speed, wind_dir_deg,
                      use_magnus=enable_magnus, omega_ball_z=omega_ball_z,
                      CL_max=CL_max, CL_k=CL_k)

xs, ys, zs = res["xs"], res["ys"], res["zs"]
crossed_goal = res["crossed_goal"]
y_at_goal = res["y_at_goal"]
z_at_goal = res["z_at_goal"]
range_ground = res["range_ground"]

is_goal = goal_hit_verdict(crossed_goal, y_at_goal, z_at_goal, goal_width, goal_height)

# ------------- Layout -------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trajectory (side view)")
    fig_side = go.Figure()
    fig_side.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="trajectory"))
    # Goal plane (post and crossbar)
    fig_side.add_shape(type="line", x0=goal_distance, x1=goal_distance, y0=0, y1=goal_height, line=dict(width=3))
    fig_side.add_shape(type="line", x0=goal_distance-goal_width*0.005, x1=goal_distance+goal_width*0.005,
                       y0=goal_height, y1=goal_height, line=dict(width=3))
    if crossed_goal and y_at_goal is not None:
        fig_side.add_trace(go.Scatter(x=[goal_distance], y=[y_at_goal], mode="markers+text",
                                      text=[f"y={y_at_goal:.2f} m"], textposition="top right",
                                      name="at goal"))
    fig_side.update_layout(xaxis_title="x (m)", yaxis_title="y (m)", margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_side, use_container_width=True)

with col2:
    st.subheader("Goal Face – Impact Point")
    fig_goal = go.Figure()

    # Draw goal rectangle (posts/crossbar) as border
    fig_goal.add_shape(type="rect",
                       x0=-goal_width/2, y0=0, x1=goal_width/2, y1=goal_height,
                       line=dict(width=4))

    # Net grid as background (visualization only)
    nx = 16  # vertical strands
    ny = 6   # horizontal strands
    for i in range(nx+1):
        z = -goal_width/2 + i * (goal_width/nx)
        fig_goal.add_shape(type="line", x0=z, x1=z, y0=0, y1=goal_height, line=dict(width=1, dash="dot"))
    for j in range(ny+1):
        yline = j * (goal_height/ny)
        fig_goal.add_shape(type="line", x0=-goal_width/2, x1=goal_width/2, y0=yline, y1=yline, line=dict(width=1, dash="dot"))

    # Impact point
    if crossed_goal and (y_at_goal is not None) and (z_at_goal is not None):
        fig_goal.add_trace(go.Scatter(x=[z_at_goal], y=[y_at_goal], mode="markers+text",
                                      text=[f"({z_at_goal:.2f}, {y_at_goal:.2f}) m"],
                                      textposition="top center",
                                      name="impact"))
    fig_goal.update_xaxes(title_text="z (m)  [left  -  right]",
                          range=[-goal_width/2-0.5, goal_width/2+0.5],
                          zeroline=True)
    fig_goal.update_yaxes(title_text="y (m)", range=[0, max(goal_height*1.1, (ys.max() if len(ys)>0 else goal_height))])
    fig_goal.update_layout(margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_goal, use_container_width=True)

st.markdown("---")
if crossed_goal:
    verdict = "✅ **GOAL**" if is_goal else "❌ **Missed goal (crossed plane outside frame)**"
    st.markdown(f"{verdict} at **y = {y_at_goal:.2f} m**, **z = {z_at_goal:.2f} m** (x = {goal_distance:.2f} m).")
else:
    st.markdown("🛑 Ball did **not reach** the goal plane before ground impact.")

if range_ground is not None:
    st.markdown(f"**Ground impact range:** {range_ground:.2f} m")

st.caption("Two-wheel model: exit speed ≈ average surface speed × slip. Spin from RPM difference; backspin (bottom>top) lifts, topspin (top>bottom) drives down via Magnus.")
