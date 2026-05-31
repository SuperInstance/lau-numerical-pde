//! Boundary conditions for PDE solvers.

use serde::{Deserialize, Serialize};

/// Boundary condition types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BoundaryCondition {
    /// Fixed value: u = value at boundary
    Dirichlet(f64),
    /// Fixed derivative: du/dn = value at boundary
    Neumann(f64),
    /// Periodic: values wrap around
    Periodic,
}

/// Pair of boundary conditions for a 1D domain (left, right).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryPair1D {
    pub left: BoundaryCondition,
    pub right: BoundaryCondition,
}

impl BoundaryPair1D {
    pub fn dirichlet(left: f64, right: f64) -> Self {
        Self {
            left: BoundaryCondition::Dirichlet(left),
            right: BoundaryCondition::Dirichlet(right),
        }
    }

    pub fn neumann(left: f64, right: f64) -> Self {
        Self {
            left: BoundaryCondition::Neumann(left),
            right: BoundaryCondition::Neumann(right),
        }
    }

    pub fn periodic() -> Self {
        Self {
            left: BoundaryCondition::Periodic,
            right: BoundaryCondition::Periodic,
        }
    }

    pub fn mixed_dirichlet_neumann(left: f64, right_flux: f64) -> Self {
        Self {
            left: BoundaryCondition::Dirichlet(left),
            right: BoundaryCondition::Neumann(right_flux),
        }
    }
}

/// Boundary conditions for a 2D domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryPair2D {
    pub x: BoundaryPair1D,
    pub y: BoundaryPair1D,
}

impl BoundaryPair2D {
    pub fn dirichlet_2d(
        x_left: f64, x_right: f64,
        y_bottom: f64, y_top: f64,
    ) -> Self {
        Self {
            x: BoundaryPair1D::dirichlet(x_left, x_right),
            y: BoundaryPair1D::dirichlet(y_bottom, y_top),
        }
    }

    pub fn periodic_2d() -> Self {
        Self {
            x: BoundaryPair1D::periodic(),
            y: BoundaryPair1D::periodic(),
        }
    }
}

/// Apply boundary conditions to a 1D solution vector.
/// `u` has length n (interior points). This returns a vector of length n+2
/// including ghost/boundary points.
pub fn apply_bc_1d(u: &[f64], dx: f64, bc: &BoundaryPair1D) -> Vec<f64> {
    let n = u.len();
    let mut full = vec![0.0; n + 2];
    full[1..=n].copy_from_slice(u);

    match &bc.left {
        BoundaryCondition::Dirichlet(val) => full[0] = *val,
        BoundaryCondition::Neumann(flux) => full[0] = full[1] - dx * flux,
        BoundaryCondition::Periodic => full[0] = u[n - 1],
    }

    match &bc.right {
        BoundaryCondition::Dirichlet(val) => full[n + 1] = *val,
        BoundaryCondition::Neumann(flux) => full[n + 1] = full[n] + dx * flux,
        BoundaryCondition::Periodic => full[n + 1] = u[0],
    }

    full
}

/// Apply boundary conditions to a 2D solution grid.
/// `u` is ny x nx (interior). Returns (ny+2) x (nx+2) including boundaries.
pub fn apply_bc_2d(u: &[Vec<f64>], dx: f64, dy: f64, bc: &BoundaryPair2D) -> Vec<Vec<f64>> {
    let ny = u.len();
    let nx = u[0].len();
    let mut full = vec![vec![0.0; nx + 2]; ny + 2];

    for i in 0..ny {
        for j in 0..nx {
            full[i + 1][j + 1] = u[i][j];
        }
    }

    // x-boundaries (columns 0 and nx+1)
    for i in 0..ny {
        match &bc.x.left {
            BoundaryCondition::Dirichlet(val) => full[i + 1][0] = *val,
            BoundaryCondition::Neumann(flux) => full[i + 1][0] = full[i + 1][1] - dx * flux,
            BoundaryCondition::Periodic => full[i + 1][0] = u[i][nx - 1],
        }
        match &bc.x.right {
            BoundaryCondition::Dirichlet(val) => full[i + 1][nx + 1] = *val,
            BoundaryCondition::Neumann(flux) => full[i + 1][nx + 1] = full[i + 1][nx] + dx * flux,
            BoundaryCondition::Periodic => full[i + 1][nx + 1] = u[i][0],
        }
    }

    // y-boundaries (rows 0 and ny+1)
    for j in 0..nx + 2 {
        match &bc.y.left {
            BoundaryCondition::Dirichlet(val) => full[0][j] = *val,
            BoundaryCondition::Neumann(flux) => full[0][j] = full[1][j] - dy * flux,
            BoundaryCondition::Periodic => full[0][j] = full[ny][j],
        }
        match &bc.y.right {
            BoundaryCondition::Dirichlet(val) => full[ny + 1][j] = *val,
            BoundaryCondition::Neumann(flux) => full[ny + 1][j] = full[ny][j] + dy * flux,
            BoundaryCondition::Periodic => full[ny + 1][j] = full[1][j],
        }
    }

    full
}

/// Apply 2D boundary conditions in-place (for solvers that mutate the grid directly).
pub fn apply_bc_2d_mut(u: &mut [Vec<f64>], dx: f64, dy: f64, bc: &BoundaryPair2D) {
    let ny = u.len();
    let nx = u[0].len();

    // x-boundaries
    for i in 0..ny {
        match &bc.x.left {
            BoundaryCondition::Dirichlet(val) => u[i][0] = *val,
            BoundaryCondition::Neumann(flux) => u[i][0] = u[i][1] - dx * flux,
            BoundaryCondition::Periodic => u[i][0] = u[i][nx - 2],
        }
        match &bc.x.right {
            BoundaryCondition::Dirichlet(val) => u[i][nx - 1] = *val,
            BoundaryCondition::Neumann(flux) => u[i][nx - 1] = u[i][nx - 2] + dx * flux,
            BoundaryCondition::Periodic => u[i][nx - 1] = u[i][1],
        }
    }

    // y-boundaries
    for j in 0..nx {
        match &bc.y.left {
            BoundaryCondition::Dirichlet(val) => u[0][j] = *val,
            BoundaryCondition::Neumann(flux) => u[0][j] = u[1][j] - dy * flux,
            BoundaryCondition::Periodic => u[0][j] = u[ny - 2][j],
        }
        match &bc.y.right {
            BoundaryCondition::Dirichlet(val) => u[ny - 1][j] = *val,
            BoundaryCondition::Neumann(flux) => u[ny - 1][j] = u[ny - 2][j] + dy * flux,
            BoundaryCondition::Periodic => u[ny - 1][j] = u[1][j],
        }
    }
}

/// Compute boundary enforcement residual (should be ~0 for correctly applied BCs).
pub fn bc_residual_1d(u_full: &[f64], dx: f64, bc: &BoundaryPair1D) -> (f64, f64) {
    let n = u_full.len();
    let left_res = match &bc.left {
        BoundaryCondition::Dirichlet(val) => (u_full[0] - val).abs(),
        BoundaryCondition::Neumann(flux) => ((u_full[1] - u_full[0]) / dx - flux).abs(),
        BoundaryCondition::Periodic => (u_full[0] - u_full[n - 2]).abs(),
    };
    let right_res = match &bc.right {
        BoundaryCondition::Dirichlet(val) => (u_full[n - 1] - val).abs(),
        BoundaryCondition::Neumann(flux) => ((u_full[n - 1] - u_full[n - 2]) / dx - flux).abs(),
        BoundaryCondition::Periodic => (u_full[n - 1] - u_full[1]).abs(),
    };
    (left_res, right_res)
}
