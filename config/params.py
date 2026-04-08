# ── Domain ──────────────────────────────────────────────────────────────────
Lx: float = 1.0
Ly: float = 1.0
Nx: int   = 128
Ny: int   = 128
dx: float = Lx / (Nx - 1)
dy: float = Ly / (Ny - 1)

# ── Thermal ──────────────────────────────────────────────────────────────────
alpha:  float = 1e-2    # Increased diffusivity
T_wall: float = 0.0     # Dirichlet BC value
T_init: float = 0.0     # initial temperature

# ── Laser source ─────────────────────────────────────────────────────────────
laser_cx:    float = 0.5
laser_cy:    float = 0.5
laser_sigma: float = 0.05
laser_power: float = 50.0   # Reduced power

# ── Time stepping ─────────────────────────────────────────────────────────────
dt:          float = 1e-5
n_steps:     int   = 100
save_every:  int   = 10

# ── AMR ──────────────────────────────────────────────────────────────────────
# Note: These values are used for the OLD visualization-only AMR.
MACRO:          int   = 16          # coarse macro-cells per axis
REFINE_THRESH:  float = 2.0         # level-1 threshold (|∇T| > thresh)
COARSEN_THRESH: float = 0.5
MAX_LEVEL:      int   = 3
# tier thresholds: level 1 if grad > REFINE_THRESH,
#                  level 2 if > REFINE_THRESH*2,
#                  level 3 if > REFINE_THRESH*8
REFINE_TIERS = [REFINE_THRESH, REFINE_THRESH * 2, REFINE_THRESH * 8]

# ── Composite Grid (JIT-AMR) ──────────────────────────────────────────────────
# Coarse grid (full domain)
Nc_x: int = 32
Nc_y: int = 32
# Fine patch (laser zone)
Nf_x: int = 64
Nf_y: int = 64
# Patch boundaries (centered on laser)
patch_x0: float = 0.3
patch_x1: float = 0.7
patch_y0: float = 0.3
patch_y1: float = 0.7

# ── IO ────────────────────────────────────────────────────────────────────────
vtk_every:        int = 50   # write VTK output every N solver steps (0 = disable)
checkpoint_every: int = 100
