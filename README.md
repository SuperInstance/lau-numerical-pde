# lau-numerical-pde

Numerical methods for partial differential equations in Rust.

## Features

- **Finite difference methods** — 1D/2D grids, explicit/implicit schemes, central/forward/backward differences, Laplacian
- **Heat equation** — Forward Euler, Backward Euler, Crank-Nicolson with stability analysis
- **Wave equation** — Leapfrog and Lax-Wendroff schemes, energy conservation
- **Poisson equation** — Jacobi, Gauss-Seidel, and SOR iteration with optimal relaxation
- **Boundary conditions** — Dirichlet, Neumann, and periodic for 1D/2D domains
- **Advection-diffusion** — Upwind scheme with CFL condition enforcement
- **Error analysis** — L2/L∞/L1 norms, convergence order verification, Richardson extrapolation
- **Agent field dynamics** — Diffusion of information through agent populations

## Usage

```rust
use lau_numerical_pde::{HeatSolver, heat::HeatMethod, Grid1D, BoundaryPair1D};

let grid = Grid1D::new(0.0, 1.0, 51);
let u0: Vec<f64> = (0..grid.n).map(|i| (std::f64::consts::PI * grid.x(i)).sin()).collect();
let solver = HeatSolver::new(grid, 0.1, BoundaryPair1D::dirichlet(0.0, 0.0), HeatMethod::CrankNicolson);
let result = solver.solve_final(&u0, 0.01, 100);
```

## License

MIT
