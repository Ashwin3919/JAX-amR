# ── Domain ──────────────────────────────────────────────────────────────────
Lx: float = 1.0
Ly: float = 1.0
Nx: int   = 128
Ny: int   = 128
dx: float = Lx / (Nx - 1)
dy: float = Ly / (Ny - 1)

# ── Thermal ──────────────────────────────────────────────────────────────────
alpha:  float = 1e-3    # thermal diffusivity [m²/s]
T_wall: float = 0.0     # Dirichlet BC value
T_init: float = 0.0     # initial temperature

# ── Laser source ─────────────────────────────────────────────────────────────
laser_cx:    float = 0.5
laser_cy:    float = 0.5
laser_sigma: float = 0.05
laser_power: float = 500.0

# ── Time stepping ─────────────────────────────────────────────────────────────
dt:          float = 1e-3
n_steps:     int   = 600
save_every:  int   = 10

# ── AMR ──────────────────────────────────────────────────────────────────────
MACRO:          int   = 16          # coarse macro-cells per axis
REFINE_THRESH:  float = 2.0         # level-1 threshold (|∇T| > thresh)
COARSEN_THRESH: float = 0.5
MAX_LEVEL:      int   = 3
# tier thresholds: level 1 if grad > REFINE_THRESH,
#                  level 2 if > REFINE_THRESH*2,
#                  level 3 if > REFINE_THRESH*8
REFINE_TIERS = [REFINE_THRESH, REFINE_THRESH * 2, REFINE_THRESH * 8]

# ── IO ────────────────────────────────────────────────────────────────────────
vtk_every:        int = 50   # write VTK output every N solver steps (0 = disable)
checkpoint_every: int = 100
