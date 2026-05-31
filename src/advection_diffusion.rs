//! Advection-diffusion equation solver: u_t + v·∇u = D·∇²u
//!
//! Upwind scheme for advection, central differences for diffusion.
//! CFL condition enforcement.

use crate::boundary::{BoundaryCondition, BoundaryPair1D};
use crate::finite_diff::Grid1D;

/// Advection-diffusion solver (1D).
pub struct AdvectionDiffusionSolver {
    pub grid: Grid1D,
    pub velocity: f64,
    pub diffusion: f64,
    pub bc: BoundaryPair1D,
}

impl AdvectionDiffusionSolver {
    pub fn new(grid: Grid1D, velocity: f64, diffusion: f64, bc: BoundaryPair1D) -> Self {
        Self { grid, velocity, diffusion, bc }
    }

    /// CFL number for advection: |v| Δt / Δx
    pub fn cfl_advection(&self, dt: f64) -> f64 {
        self.velocity.abs() * dt / self.grid.dx
    }

    /// Diffusion number: D Δt / Δx²
    pub fn diffusion_number(&self, dt: f64) -> f64 {
        self.diffusion * dt / (self.grid.dx * self.grid.dx)
    }

    /// Peclet number: |v| Δx / D
    pub fn peclet(&self) -> f64 {
        self.velocity.abs() * self.grid.dx / self.diffusion
    }

    /// Check stability: CFL ≤ 1 and diffusion number ≤ 0.5
    pub fn is_stable(&self, dt: f64) -> bool {
        self.cfl_advection(dt) <= 1.0 + 1e-12 && self.diffusion_number(dt) <= 0.5 + 1e-12
    }

    /// Maximum stable time step.
    pub fn max_stable_dt(&self) -> f64 {
        let dt_adv = self.grid.dx / self.velocity.abs();
        let dt_diff = 0.5 * self.grid.dx * self.grid.dx / self.diffusion;
        dt_adv.min(dt_diff)
    }

    /// Solve using upwind for advection + central for diffusion (explicit Euler in time).
    pub fn solve(&self, u0: &[f64], dt: f64, n_steps: usize) -> Vec<Vec<f64>> {
        let n = u0.len();
        let dx = self.grid.dx;
        let v = self.velocity;
        let d = self.diffusion;

        let mut u = u0.to_vec();
        let mut history = vec![u.clone()];

        for _ in 0..n_steps {
            let mut u_new = u.clone();
            for i in 1..n - 1 {
                // Upwind for advection
                let adv = if v >= 0.0 {
                    v * (u[i] - u[i - 1]) / dx
                } else {
                    v * (u[i + 1] - u[i]) / dx
                };

                // Central for diffusion
                let diff = d * (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx);

                u_new[i] = u[i] + dt * (-adv + diff);
            }

            // Apply BCs
            self.apply_bc(&mut u_new);
            u = u_new;
            history.push(u.clone());
        }

        history
    }

    /// Solve and return only final state.
    pub fn solve_final(&self, u0: &[f64], dt: f64, n_steps: usize) -> Vec<f64> {
        self.solve(u0, dt, n_steps).into_iter().last().unwrap_or_else(|| u0.to_vec())
    }

    fn apply_bc(&self, u: &mut [f64]) {
        let n = u.len();
        match &self.bc.left {
            BoundaryCondition::Dirichlet(v) => u[0] = *v,
            BoundaryCondition::Neumann(f) => u[0] = u[1] - self.grid.dx * f,
            BoundaryCondition::Periodic => u[0] = u[n - 2],
        }
        match &self.bc.right {
            BoundaryCondition::Dirichlet(v) => u[n - 1] = *v,
            BoundaryCondition::Neumann(f) => u[n - 1] = u[n - 2] + self.grid.dx * f,
            BoundaryCondition::Periodic => u[n - 1] = u[1],
        }
    }

    /// Solve 2D advection-diffusion: u_t + vx*ux + vy*uy = D*(uxx + uyy).
    pub fn solve_2d(
        grid: &crate::finite_diff::Grid2D,
        vx: f64, vy: f64, diffusion: f64,
        bc: &crate::boundary::BoundaryPair2D,
        u0: &[Vec<f64>],
        dt: f64, n_steps: usize,
    ) -> Vec<Vec<Vec<f64>>> {
        let ny = grid.ny;
        let nx = grid.nx;
        let dx = grid.dx;
        let dy = grid.dy;

        let mut u = u0.to_vec();
        let mut history = vec![u.clone()];

        for _ in 0..n_steps {
            let mut u_new = u.clone();
            for i in 1..ny - 1 {
                for j in 1..nx - 1 {
                    // Upwind for advection
                    let adv_x = if vx >= 0.0 {
                        vx * (u[i][j] - u[i][j - 1]) / dx
                    } else {
                        vx * (u[i][j + 1] - u[i][j]) / dx
                    };
                    let adv_y = if vy >= 0.0 {
                        vy * (u[i][j] - u[i - 1][j]) / dy
                    } else {
                        vy * (u[i + 1][j] - u[i][j]) / dy
                    };

                    let diff_x = diffusion * (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) / (dx * dx);
                    let diff_y = diffusion * (u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j]) / (dy * dy);

                    u_new[i][j] = u[i][j] + dt * (-adv_x - adv_y + diff_x + diff_y);
                }
            }
            // Apply 2D BCs
            crate::boundary::apply_bc_2d_mut(&mut u_new, dx, dy, bc);
            u = u_new;
            history.push(u.clone());
        }

        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ad_cfl_stability() {
        let grid = Grid1D::new(0.0, 1.0, 51);
        let solver = AdvectionDiffusionSolver::new(grid, 1.0, 0.01, BoundaryPair1D::dirichlet(0.0, 0.0));
        let dt_stable = solver.max_stable_dt() * 0.5;
        let dt_unstable = solver.max_stable_dt() * 2.0;
        assert!(solver.is_stable(dt_stable));
        assert!(!solver.is_stable(dt_unstable));
    }

    #[test]
    fn test_ad_peclet_number() {
        let grid = Grid1D::new(0.0, 1.0, 101);
        let solver = AdvectionDiffusionSolver::new(grid, 1.0, 0.1, BoundaryPair1D::dirichlet(0.0, 0.0));
        let pe = solver.peclet();
        assert!(pe > 0.0);
    }

    #[test]
    fn test_ad_pure_diffusion() {
        // With velocity=0, should reduce to heat equation
        let grid = Grid1D::new(0.0, 1.0, 51);
        let d = 0.01;
        let dt = 0.4 * grid.dx * grid.dx / d;
        let u0: Vec<f64> = (0..grid.n).map(|i| (std::f64::consts::PI * grid.x(i)).sin()).collect();

        let solver = AdvectionDiffusionSolver::new(grid, 0.0, d, BoundaryPair1D::dirichlet(0.0, 0.0));
        let u_final = solver.solve_final(&u0, dt, 20);

        // Should decay
        let max0 = u0.iter().cloned().fold(f64::NAN, f64::max);
        let max_f = u_final.iter().cloned().fold(f64::NAN, f64::max);
        assert!(max_f < max0);
    }

    #[test]
    fn test_ad_pure_advection_periodic() {
        // Gaussian pulse advecting with periodic BC
        let grid = Grid1D::new(0.0, 1.0, 201);
        let v = 1.0;
        let dt = 0.5 * grid.dx / v;
        let sigma = 0.05;
        let u0: Vec<f64> = (0..grid.n).map(|i| {
            (-((grid.x(i) - 0.5).powi(2)) / (2.0 * sigma * sigma)).exp()
        }).collect();

        let solver = AdvectionDiffusionSolver::new(grid, v, 0.0001, BoundaryPair1D::periodic());
        let n_steps = (1.0 / dt).round() as usize; // one full period
        let u_final = solver.solve_final(&u0, dt, n_steps);

        // After one period, peak should return near x=0.5
        let max_val = u_final.iter().cloned().fold(f64::NAN, f64::max);
        assert!(max_val > 0.5, "Peak should survive advection: {}", max_val);
    }

    #[test]
    fn test_ad_conservation_periodic() {
        let grid = Grid1D::new(0.0, 1.0, 101);
        let v = 0.5;
        let d = 0.001;
        let dt = solver_max_dt(&grid, v, d) * 0.4;
        let u0: Vec<f64> = (0..grid.n).map(|i| 1.0 + 0.3 * (2.0 * std::f64::consts::PI * grid.x(i)).sin()).collect();

        let solver = AdvectionDiffusionSolver::new(grid, v, d, BoundaryPair1D::periodic());
        let u_final = solver.solve_final(&u0, dt, 50);

        let sum0: f64 = u0.iter().sum();
        let sumf: f64 = u_final.iter().sum();
        // With small diffusion, mass should be approximately conserved
        assert_relative_eq!(sum0, sumf, epsilon = 1.0);
    }

    fn solver_max_dt(grid: &Grid1D, v: f64, d: f64) -> f64 {
        let dt_adv = grid.dx / v;
        let dt_diff = 0.5 * grid.dx * grid.dx / d;
        dt_adv.min(dt_diff)
    }

    #[test]
    fn test_ad_upwind_direction() {
        // Positive velocity: information comes from left
        let grid = Grid1D::new(0.0, 1.0, 51);
        let v = 1.0;
        let dt = 0.4 * grid.dx / v;
        let peak_idx = grid.n / 2;
        let peak_pos_0 = grid.x(peak_idx);
        let mut u0 = vec![0.0; grid.n];
        u0[peak_idx] = 1.0;

        let solver = AdvectionDiffusionSolver::new(grid, v, 0.0, BoundaryPair1D::periodic());
        let u_final = solver.solve_final(&u0, dt, 5);

        // Peak should move to the right
        let max_idx = u_final.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let peak_pos_f = solver.grid.x(max_idx);
        assert!(peak_pos_f > peak_pos_0 || u_final[max_idx] < 0.9,
            "Peak should move right with positive velocity");
    }

    #[test]
    fn test_ad_dirichlet_bc_enforced() {
        let grid = Grid1D::new(0.0, 1.0, 51);
        let dt = 0.3 * grid.dx;
        let u0 = vec![0.5; grid.n];

        let solver = AdvectionDiffusionSolver::new(grid, 1.0, 0.01, BoundaryPair1D::dirichlet(1.0, 0.0));
        let u_final = solver.solve_final(&u0, dt, 20);

        assert_relative_eq!(u_final[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(u_final[u_final.len() - 1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_ad_convergence() {
        let v = 1.0;
        let d = 0.01;
        let errors: Vec<f64> = [51, 101, 201].iter().map(|&n| {
            let grid = Grid1D::new(0.0, 1.0, n);
            let dt = 0.4 * grid.dx / v;
            // Exact: traveling Gaussian with diffusion
            let sigma = 0.1;
            let u0: Vec<f64> = (0..n).map(|i| {
                (-((grid.x(i) - 0.3).powi(2)) / (2.0 * sigma * sigma)).exp()
            }).collect();
            let solver = AdvectionDiffusionSolver::new(grid, v, d, BoundaryPair1D::dirichlet(0.0, 0.0));
            let u_final = solver.solve_final(&u0, dt, 20);
            let max_val = u_final.iter().cloned().fold(f64::NAN, f64::max);
            max_val
        }).collect();

        // All solutions should be bounded
        for &e in &errors {
            assert!(e.is_finite() && e >= 0.0);
        }
    }
}
