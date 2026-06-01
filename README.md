# lau-numerical-pde

Numerical methods for partial differential equations — finite differences, heat/wave/Poisson/advection-diffusion solvers, flexible boundary conditions, error analysis, and an agent-field dynamics application layer.

Built for large-scale agent simulations where PDE-based models describe how information, opinions, or behaviors spread through populations.

---

## What This Does

This crate is a **from-scratch PDE solver toolkit** in pure Rust:

| Module | What you get |
|---|---|
| **Finite differences** | 1D/2D uniform grids, central/forward/backward differences, 2nd and 4th order |
| **Heat equation** | Forward Euler (explicit), Backward Euler (implicit), Crank-Nicolson |
| **Wave equation** | Leapfrog, Lax-Wendroff with dissipation, energy conservation tracking |
| **Poisson equation** | Jacobi, Gauss-Seidel, SOR (with optimal ω), 2D elliptic solver |
| **Advection-diffusion** | Upwind advection + central diffusion, 1D and 2D, CFL/Peclet analysis |
| **Boundary conditions** | Dirichlet, Neumann, Periodic — composable for 1D and 2D |
| **Error analysis** | L2/L∞/L1 norms, convergence order, Richardson extrapolation, total variation |
| **Agent fields** | Information diffusion simulation, consensus metrics, biased diffusion |

76 unit tests cover stability, convergence, conservation, and accuracy.

---

## Key Idea

Every PDE is discretized on structured grids using classical finite difference stencils. The boundary condition system is type-safe and composable — you mix `Dirichlet`, `Neumann`, and `Periodic` freely on each edge of the domain. The implicit solvers (Backward Euler, Crank-Nicolson) build tridiagonal systems and solve them via `nalgebra` LU decomposition.

The `agent_field` module shows the motivating application: modeling how a property (opinion, knowledge, adoption rate) spreads through a spatially-distributed agent population as a continuous field governed by diffusion or advection-diffusion equations.

---

## Install

```toml
[dependencies]
lau-numerical-pde = "0.1.0"
```

Or as a git dependency:

```toml
[dependencies]
lau-numerical-pde = { git = "https://github.com/SuperInstance/lau-numerical-pde" }
```

Requires **Rust 2021 edition**.

### Dependencies

| Crate | Why |
|---|---|
| `nalgebra` | LU decomposition for implicit heat solvers |
| `serde` | Serialize/deserialize grids and boundary conditions |
| `num-traits` | Numeric trait utilities |

---

## Quick Start

### Heat equation (explicit)

```rust
use lau_numerical_pde::{Grid1D, HeatSolver, HeatMethod, BoundaryPair1D};

let grid = Grid1D::new(0.0, 1.0, 51);
let alpha = 0.01;
let bc = BoundaryPair1D::dirichlet(0.0, 0.0);
let solver = HeatSolver::new(grid, alpha, bc, HeatMethod::ForwardEuler);

let u0: Vec<f64> = (0..solver.grid.n)
    .map(|i| (std::f64::consts::PI * solver.grid.x(i)).sin())
    .collect();

let dt = solver.max_stable_dt() * 0.8;
let u_final = solver.solve_final(&u0, dt, 200);
```

### Heat equation (implicit — unconditionally stable)

```rust
let solver = HeatSolver::new(grid, alpha, bc, HeatMethod::CrankNicolson);
// Can use any dt — no CFL restriction
let u_final = solver.solve_final(&u0, 0.1, 10);
```

### Wave equation

```rust
use lau_numerical_pde::{WaveSolver, WaveMethod};

let grid = Grid1D::new(0.0, 1.0, 101);
let solver = WaveSolver::new(grid, 1.0, BoundaryPair1D::dirichlet(0.0, 0.0), WaveMethod::Leapfrog);

let u0 = vec![/* initial displacement */];
let v0 = vec![/* initial velocity */];

let dt = solver.max_stable_dt() * 0.5;
let history = solver.solve(&u0, &v0, dt, 100);
let energy = solver.energy(&history[50], &history[49], dt);
```

### Poisson equation (2D)

```rust
use lau_numerical_pde::{PoissonSolver, PoissonMethod, Grid2D, BoundaryPair2D};

let grid = Grid2D::new(0.0, 1.0, 41, 0.0, 1.0, 41);
let bc = BoundaryPair2D::dirichlet_2d(0.0, 0.0, 0.0, 0.0);
let omega = PoissonSolver::optimal_sor_omega(41, 41);
let solver = PoissonSolver::new(grid, bc, PoissonMethod::SOR(omega))
    .with_tolerance(1e-8);

// f is the source term: -∇²u = f
let f = vec![vec![0.0; 41]; 41]; // fill with your source
let result = solver.solve(&f, None);
println!("converged: {}, iterations: {}", result.converged, result.iterations);
```

### Advection-diffusion

```rust
use lau_numerical_pde::{AdvectionDiffusionSolver, Grid1D, BoundaryPair1D};

let grid = Grid1D::new(0.0, 1.0, 201);
let solver = AdvectionDiffusionSolver::new(grid, 1.0, 0.01, BoundaryPair1D::periodic());

let dt = solver.max_stable_dt() * 0.5;
let u_final = solver.solve_final(&u0, dt, 100);
```

### Agent field dynamics

```rust
use lau_numerical_pde::agent_field::{AgentField, InformationDiffusion};

let grid = Grid1D::new(0.0, 1.0, 101);
let field = AgentField::gaussian(grid, 0.5, 1.0, 0.1);
println!("initial consensus: {}", field.consensus());

let diffusion = InformationDiffusion::new(0.01, BoundaryPair1D::periodic());
let ttc = diffusion.time_to_consensus(&field, dt, 0.01, 10000);
println!("steps to consensus: {:?}", ttc);
```

### Error analysis and convergence verification

```rust
use lau_numerical_pde::ErrorAnalysis;

let l2 = ErrorAnalysis::l2_error(&numerical, &exact, dx);
let linf = ErrorAnalysis::linf_error(&numerical, &exact);
let order = ErrorAnalysis::convergence_order(&errors);
let improved = ErrorAnalysis::richardson_extrapolation(&u_h, &u_h2, 2.0);
```

---

## API Reference

### `finite_diff` — Grids and Operators

| Type | Description |
|---|---|
| `Grid1D` | Uniform 1D grid: `new(x_min, x_max, n)`, `.x(i)`, `.dx`, `.interior_points()` |
| `Grid2D` | Uniform 2D grid: `new(x_min, x_max, nx, y_min, y_max, ny)`, `.x(i)`, `.y(j)` |

| Operator | Order | Notes |
|---|---|---|
| `FiniteDiff::d1_central` | O(h²) | Central first derivative |
| `FiniteDiff::d1_forward` | O(h) | Forward first derivative |
| `FiniteDiff::d1_backward` | O(h) | Backward first derivative |
| `FiniteDiff::d2_central` | O(h²) | Central second derivative |
| `FiniteDiff::d2_central_4th` | O(h⁴) | Fourth-order second derivative |
| `FiniteDiff::laplacian_2d` | O(h²) | 2D Laplacian stencil |

### `boundary` — Boundary Conditions

| Type | Variants |
|---|---|
| `BoundaryCondition` | `Dirichlet(f64)`, `Neumann(f64)`, `Periodic` |
| `BoundaryPair1D` | `(left, right)` — `.dirichlet(l, r)`, `.neumann(l, r)`, `.periodic()` |
| `BoundaryPair2D` | `(x_pair, y_pair)` — `.dirichlet_2d(...)`, `.periodic_2d()` |

Helper functions: `apply_bc_1d`, `apply_bc_2d`, `apply_bc_2d_mut`, `bc_residual_1d`.

### `heat` — Heat Equation (u_t = α u_xx)

| Method | Stability | Order (time, space) |
|---|---|---|
| `ForwardEuler` | r ≤ 0.5 | O(Δt, Δx²) |
| `BackwardEuler` | Unconditional | O(Δt, Δx²) |
| `CrankNicolson` | Unconditional | O(Δt², Δx²) |

Key methods: `.solve()`, `.solve_final()`, `.is_stable()`, `.max_stable_dt()`, `.stability_param()`.

### `wave` — Wave Equation (u_tt = c² u_xx)

| Method | Stability | Notes |
|---|---|---|
| `Leapfrog` | CFL ≤ 1 | Second-order, energy-conserving |
| `LaxWendroff` | CFL ≤ 1 | Leapfrog + small dissipation |

Key methods: `.solve()`, `.solve_final()`, `.energy()`, `.cfl()`, `.max_stable_dt()`.

### `poisson` — Poisson Equation (-∇²u = f)

| Method | Notes |
|---|---|
| `Jacobi` | Simultaneous update; slowest convergence |
| `GaussSeidel` | Immediate update; ~2× faster |
| `SOR(omega)` | Optimal ω via `PoissonSolver::optimal_sor_omega(nx, ny)` |

Key methods: `.solve(&f, u_init)`, `.jacobi_spectral_radius()`, `.with_tolerance()`, `.with_max_iter()`.

### `advection_diffusion` — Advection-Diffusion (u_t + v·∇u = D·∇²u)

Upwind scheme for advection, central for diffusion. Explicit Euler in time.

Key methods: `.solve()`, `.solve_final()`, `.solve_2d()`, `.cfl_advection()`, `.diffusion_number()`, `.peclet()`, `.is_stable()`, `.max_stable_dt()`.

### `agent_field` — Agent Field Dynamics

| Type | Description |
|---|---|
| `AgentField` | Spatially-distributed field: `.uniform()`, `.gaussian()`, `.heterogeneous()` |
| `InformationDiffusion` | Diffusion/biased-diffusion simulator: `.step()`, `.simulate()`, `.time_to_consensus()` |

Field metrics: `.mean()`, `.std_dev()`, `.consensus()`, `.total_mass()`, `.max_value()`, `.min_value()`.

### `error_analysis` — Error Analysis

| Function | Description |
|---|---|
| `l2_error` | RMS error norm |
| `linf_error` | Maximum error norm |
| `l1_error` | L1 error norm |
| `convergence_order` | log₂(e_coarse/e_fine) |
| `convergence_order_regression` | Least-squares fit of log(err) vs log(h) |
| `verify_convergence_order` | Boolean check against expected order |
| `richardson_extrapolation` | Improve accuracy: u_h2 + (u_h2 - u_h)/(2^p - 1) |
| `richardson_error_estimate` | Error bars from Richardson |
| `total_variation` | TV(u) = Σ|u_{i+1} - u_i| |
| `convergence_study` | Run full grid-refinement study |

---

## How It Works

### Finite Differences (`finite_diff.rs`)

Everything starts from Taylor expansions. The second-order central difference for u'' comes from:

```
u(x+h) - 2u(x) + u(x-h) = h²u''(x) + O(h⁴)
```

The 4th-order version uses a 5-point stencil:

```
u''(x) ≈ (-u(x+2h) + 16u(x+h) - 30u(x) + 16u(x-h) - u(x-2h)) / (12h²)
```

The 2D Laplacian applies the 1D stencil in each direction and sums.

### Heat Equation (`heat.rs`)

The heat equation u_t = α u_xx is discretized as:

- **Forward Euler**: `u^{n+1}_i = u^n_i + r(u^n_{i+1} - 2u^n_i + u^n_{i-1})` where r = αΔt/Δx². Stable only when r ≤ 0.5.
- **Backward Euler**: `(I + rA)u^{n+1} = u^n + bc_terms`. The tridiagonal system is solved via LU. Unconditionally stable but first-order in time.
- **Crank-Nicolson**: `(I + r/2·A)u^{n+1} = (I - r/2·A)u^n + bc_terms`. Unconditionally stable and second-order in time — the gold standard for parabolic problems.

Boundary conditions modify the stencil at edges: Dirichlet sets the value, Neumann uses a ghost point (u_{-1} = u_1 - dx·flux), and Periodic wraps.

### Wave Equation (`wave.rs`)

The wave equation u_tt = c²u_xx uses a three-level scheme:

- **Leapfrog**: `u^{n+1} = 2u^n - u^{n-1} + r²(u^n_{i+1} - 2u^n_i + u^n_{i-1})` where r = cΔt/Δx is the Courant number. Second-order in both time and space. Requires r ≤ 1 for stability (CFL condition).
- The first time step uses a Taylor expansion: `u^1 = u^0 + Δt·v^0 + ½Δt²c²u_xx^0`.
- Energy: E = ½∫(u_t² + c²u_x²)dx is conserved for the leapfrog scheme in exact arithmetic.

### Poisson Equation (`poisson.rs`)

The elliptic PDE -∇²u = f is solved iteratively on a 2D grid:

- **Jacobi**: Updates all interior points simultaneously using old values. Convergence rate depends on spectral radius ρ of the iteration matrix: error reduces by factor ρ per sweep.
- **Gauss-Seidel**: Uses updated values immediately (lexicographic ordering). Converges ~2× faster than Jacobi in iteration count.
- **SOR**: Over-relaxes the Gauss-Seidel update: `u_new = (1-ω)u_old + ω·u_GS`. Optimal ω ≈ 2/(1 + √(1-ρ²)) dramatically accelerates convergence.

### Advection-Diffusion (`advection_diffusion.rs`)

Combines two physical processes:

- **Advection** (transport): Uses upwind differencing — the spatial stencil follows the flow direction (backward difference for v > 0, forward for v < 0). This ensures stability but is only first-order. The CFL condition requires |v|Δt/Δx ≤ 1.
- **Diffusion** (spreading): Central differences for the second derivative, requiring DΔt/Δx² ≤ 0.5.
- **Peclet number** Pe = |v|Δx/D indicates whether advection (Pe >> 1) or diffusion (Pe << 1) dominates.

### Agent Fields (`agent_field.rs`)

Maps PDE solutions to agent simulation concepts:

- An `AgentField` represents a population property (opinion, knowledge level) as a continuous function over space.
- **Consensus** measures agreement: 1 - σ/(range/2), where σ is the standard deviation.
- `InformationDiffusion` wraps the heat or advection-diffusion solver to model how information spreads through the population.
- `time_to_consensus` runs the simulation until the field's standard deviation drops below a threshold.

### Error Analysis (`error_analysis.rs`)

Provides tools to verify that solvers converge at the expected rate:

- **Convergence order**: Refine the grid (halving h) and measure how fast the error shrinks. Second-order methods should show error dropping by ~4×.
- **Richardson extrapolation**: Given solutions on grids h and h/2, combine them to cancel the leading error term: u_better = u_{h/2} + (u_{h/2} - u_h)/(2^p - 1).
- **Total variation**: Measures oscillations — important for checking that advection schemes aren't introducing spurious oscillations.

---

## The Math

### Stability Analysis

For Forward Euler on the heat equation, the amplification factor for Fourier mode k is:

```
g(k) = 1 - 4r sin²(kΔx/2)
```

Stability requires |g| ≤ 1 for all k, giving r ≤ 0.5. Backward Euler has g = 1/(1 + 4r sin²(kΔx/2)) < 1 for all r — unconditionally stable. Crank-Nicolson has g = (1 - 2r sin²)/(1 + 2r sin²), also |g| ≤ 1 for all r.

### CFL Condition

For the wave equation, the Courant-Friedrichs-Lewy condition states that the numerical domain of dependence must contain the physical domain of dependence. For u_tt = c²u_xx with the leapfrog scheme, this requires:

```
cΔt/Δx ≤ 1
```

Physically: the wave can't travel more than one grid cell per time step. Violating this causes the scheme to "miss" information and blow up.

### Optimal SOR Parameter

For the Poisson equation on a uniform grid with Jacobi spectral radius ρ = cos(π/N), the optimal SOR relaxation parameter is:

```
ω_opt = 2 / (1 + √(1 - ρ²))
```

For a 100×100 grid, ρ ≈ 0.998 and ω_opt ≈ 1.97. This reduces the number of iterations from O(N²) with Jacobi to O(N) with optimal SOR.

### Richardson Extrapolation

If a method has order p, then:

```
u_h = u_exact + C·h^p + O(h^{p+1})
u_{h/2} = u_exact + C·(h/2)^p + O(h^{p+1})
```

Subtracting: u_exact ≈ u_{h/2} + (u_{h/2} - u_h)/(2^p - 1), which cancels the O(h^p) error term.

---

## License

MIT
