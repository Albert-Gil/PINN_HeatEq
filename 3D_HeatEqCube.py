import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# === CONFIGURATION ===
ALPHA = 0.01
SIM_TIME = 5.0
EPOCHS = 20000
GRID_SIZE = 30
NUM_FRAMES = 100
FPS = 20
SLICE_Z = 0.5
SLICE_TIMES = [1.0, 3.0, 5.0]  # for snapshot figures

# === PDE SETUP ===
def heat_3d(x, u):
    du_t = dde.grad.jacobian(u, x, j=3)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    du_zz = dde.grad.hessian(u, x, i=2, j=2)
    return du_t - ALPHA * (du_xx + du_yy + du_zz)

def initial_func(x):
    return (
        np.sin(np.pi * x[:, 0:1])
        * np.sin(np.pi * x[:, 1:2])
        * np.sin(np.pi * x[:, 2:3])
    )

def analytical_solution(x, y, z, t):
    return (
        np.sin(np.pi * x)
        * np.sin(np.pi * y)
        * np.sin(np.pi * z)
        * np.exp(-3 * np.pi**2 * ALPHA * t)
    )

# === DOMAIN & CONDITIONS ===
geom = dde.geometry.Cuboid([0, 0, 0], [1, 1, 1])
timedomain = dde.geometry.TimeDomain(0, SIM_TIME)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary)
ic = dde.IC(geomtime, initial_func, lambda x, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime, heat_3d, [bc, ic],
    num_domain=5000, num_boundary=200, num_initial=200
)

# === NEURAL NETWORK ===
net = dde.maps.FNN([4] + [50] * 4 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(epochs=EPOCHS)
model.compile("L-BFGS")
model.train()

# === GRID ===
x = np.linspace(0, 1, GRID_SIZE)
y = np.linspace(0, 1, GRID_SIZE)
z = np.linspace(0, 1, GRID_SIZE)
X, Y, Z = np.meshgrid(x, y, z)
XYZ = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

times = np.linspace(0, SIM_TIME, NUM_FRAMES)

# === PRECOMPUTE RESULTS ===
u_preds, u_trues, u_errs, rel_errors = [], [], [], []
print("Precomputing predictions...")
for t in times:
    input_data = np.hstack((XYZ, t * np.ones((XYZ.shape[0], 1))))
    u_pred = model.predict(input_data).flatten()
    u_true = analytical_solution(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], t)
    u_err = np.abs(u_true - u_pred)
    rel_error = np.linalg.norm(u_err) / np.linalg.norm(u_true)

    u_preds.append(u_pred)
    u_trues.append(u_true)
    u_errs.append(u_err)
    rel_errors.append(rel_error)
print("Done.")

# === ANIMATION ===
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

def animate(frame):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    u_pred = u_preds[frame]
    u_true = u_trues[frame]
    u_err = u_errs[frame]

    mask1 = u_pred > 0.5
    mask2 = u_true > 0.5
    mask3 = u_err > 0.05

    ax1.scatter(XYZ[mask1, 0], XYZ[mask1, 1], XYZ[mask1, 2], c=u_pred[mask1], cmap="inferno", s=1)
    ax1.set_title(f"PINN Prediction (t={times[frame]:.2f}s)")

    ax2.scatter(XYZ[mask2, 0], XYZ[mask2, 1], XYZ[mask2, 2], c=u_true[mask2], cmap="inferno", s=1)
    ax2.set_title("Analytical Solution")

    ax3.scatter(XYZ[mask3, 0], XYZ[mask3, 1], XYZ[mask3, 2], c=u_err[mask3], cmap="viridis", s=1)
    ax3.set_title("Absolute Error")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

ani = animation.FuncAnimation(fig, animate, frames=NUM_FRAMES, interval=1000/FPS)
writer = FFMpegWriter(fps=FPS)
ani.save("heat3d_volume_animation.mp4", writer=writer)
plt.close(fig)

print("Saved animation: heat3d_volume_animation.mp4")

# === ERROR PLOT ===
plt.figure(figsize=(6, 4))
plt.plot(times, rel_errors, marker='o')
plt.title("Relative L2 Error over Time")
plt.xlabel("Time [s]")
plt.ylabel("Relative L2 Error")
plt.grid(True)
plt.tight_layout()
plt.savefig("error_over_time.png")
plt.show()

# === SLICE SNAPSHOTS ===
Z_index = np.argmin(np.abs(z - SLICE_Z))
for t_snap in SLICE_TIMES:
    input_data = np.hstack((XYZ, t_snap * np.ones((XYZ.shape[0], 1))))
    u_pred = model.predict(input_data).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
    u_true = analytical_solution(X, Y, Z, t_snap).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
    u_err = np.abs(u_true - u_pred)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axs[0].imshow(u_pred[:, :, Z_index], cmap="inferno", origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1)
    axs[0].set_title(f"PINN Prediction (z ≈ {SLICE_Z}, t = {t_snap})")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(u_true[:, :, Z_index], cmap="inferno", origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1)
    axs[1].set_title("Analytical Solution")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(u_err[:, :, Z_index], cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
    axs[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(f"slice_snapshot_t{int(t_snap*100):02d}.png")
    plt.show()
