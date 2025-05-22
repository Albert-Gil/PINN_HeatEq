import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# === 🔧 CONFIGURATION (Edit these!) ===
ALPHA = 0.01             # Thermal diffusivity
SIM_TIME = 10.0           # Total simulation time in seconds
NUM_FRAMES = 300         # Number of animation frames
ANIMATION_FPS = 30       # FPS of output video
TRAINING_EPOCHS = 30000   # Epochs for Adam optimizer
GRID_SIZE = 100          # Spatial resolution

# === 🔬 PDE DEFINITION ===
def heat_2d(x, u):
    du_t = dde.grad.jacobian(u, x, j=2)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    return du_t - ALPHA * (du_xx + du_yy)

def initial_func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

def analytical_solution(x, y, t):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2 * np.pi ** 2 * ALPHA * t)

# === 📐 DOMAIN AND CONDITIONS ===
geom = dde.geometry.Rectangle([0, 0], [1, 1])
timedomain = dde.geometry.TimeDomain(0, SIM_TIME)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary)
ic = dde.IC(geomtime, initial_func, lambda x, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime, heat_2d, [bc, ic],
    num_domain=4000, num_boundary=100, num_initial=100
)

# === 🧠 NEURAL NETWORK MODEL ===
net = dde.maps.FNN([3] + [50]*3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(epochs=TRAINING_EPOCHS)

model.compile("L-BFGS")
model.train()

# === 🌐 GRID FOR EVALUATION ===
x = np.linspace(0, 1, GRID_SIZE)
y = np.linspace(0, 1, GRID_SIZE)
X, Y = np.meshgrid(x, y)
XY = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

# === 📈 COMPUTE RELATIVE L2 ERROR OVER TIME ===
times = np.linspace(0, SIM_TIME, NUM_FRAMES)
rel_errors = []
for t in times:
    input_data = np.hstack((XY, t * np.ones((XY.shape[0], 1))))
    u_pred = model.predict(input_data).flatten()
    u_true = analytical_solution(X.flatten(), Y.flatten(), t)
    rel_error = np.linalg.norm(u_true - u_pred, 2) / np.linalg.norm(u_true, 2)
    rel_errors.append(rel_error)

# === 🎥 ANIMATION ===
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
plt.subplots_adjust(wspace=0.3)
levels = np.linspace(0, 1, 100)

def animate(i):
    for ax in axs: ax.clear()

    t = times[i]
    input_data = np.hstack((XY, t * np.ones((XY.shape[0], 1))))
    u_pred = model.predict(input_data).reshape(GRID_SIZE, GRID_SIZE)
    u_true = analytical_solution(X, Y, t)
    error = np.abs(u_pred - u_true)

    axs[0].contourf(X, Y, u_pred, levels=levels, cmap="inferno")
    axs[0].set_title(f"PINN Prediction (t = {t:.2f}s)")

    axs[1].contourf(X, Y, u_true, levels=levels, cmap="inferno")
    axs[1].set_title("Analytical Solution")

    axs[2].contourf(X, Y, error, 100, cmap="viridis")
    axs[2].set_title("Absolute Error")

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

ani = animation.FuncAnimation(fig, animate, frames=NUM_FRAMES, interval=1000/ANIMATION_FPS)

# Save animation
writer = FFMpegWriter(fps=ANIMATION_FPS)
ani.save("heat2d_pinn_vs_analytical.mp4", writer=writer)
plt.close(fig)

# === 📊 SUMMARY: RELATIVE ERROR OVER TIME ===
plt.figure(figsize=(7, 4))
plt.plot(times, rel_errors, marker='o')
plt.xlabel("Time [s]")
plt.ylabel("Relative L2 Error")
plt.title("PINN vs. Analytical: Relative Error Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("error_over_time.png")
plt.show()

# === 📷 SNAPSHOT PLOTS AT SELECTED TIMES ===
snapshot_times = [1.0, 3.0, 6.0]
for t_snap in snapshot_times:
    input_data = np.hstack((XY, t_snap * np.ones((XY.shape[0], 1))))
    u_pred = model.predict(input_data).reshape(GRID_SIZE, GRID_SIZE)
    u_true = analytical_solution(X, Y, t_snap)
    error = np.abs(u_pred - u_true)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, u_pred, 100, cmap="inferno")
    plt.title(f"PINN Prediction (t = {t_snap}s)")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, u_true, 100, cmap="inferno")
    plt.title("Analytical Solution")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, error, 100, cmap="viridis")
    plt.title("Absolute Error")
    plt.colorbar()

    plt.suptitle(f"Comparison at t = {t_snap}s")
    plt.tight_layout()
    plt.savefig(f"snapshot_t{int(t_snap*10):02d}.png")
    plt.show()
