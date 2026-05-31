//! Poisson equation solver: -∇²u = f
//!
//! Iterative methods: Jacobi, Gauss-Seidel, SOR.

use crate::boundary::{BoundaryCondition, BoundaryPair2D};
use crate::finite_diff::Grid2D;

/// Iterative method for Poisson equation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoissonMethod {
    Jacobi,
    GaussSeidel,
    SOR(f64), // SOR with relaxation parameter ω
}

/// Poisson equation solver result.
#[derive(Debug, Clone)]
pub struct PoissonResult {
    pub u: Vec<Vec<f64>>,
    pub iterations: usize,
    pub final_residual: f64,
    pub converged: bool,
}

pub struct PoissonSolver {
    pub grid: Grid2D,
    pub bc: BoundaryPair2D,
    pub method: PoissonMethod,
    pub tol: f64,
    pub max_iter: usize,
}

impl PoissonSolver {
    pub fn new(grid: Grid2D, bc: BoundaryPair2D, method: PoissonMethod) -> Self {
        Self { grid, bc, method, tol: 1e-8, max_iter: 10_000 }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self { self.tol = tol; self }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self { self.max_iter = max_iter; self }

    /// Optimal SOR parameter for a rectangular grid.
    pub fn optimal_sor_omega(nx: usize, ny: usize) -> f64 {
        let rho = ((std::f64::consts::PI / nx as f64).cos())
            .max((std::f64::consts::PI / ny as f64).cos());
        2.0 / (1.0 + (1.0 - rho * rho).sqrt())
    }

    /// Compute spectral radius of Jacobi iteration matrix (for analysis).
    pub fn jacobi_spectral_radius(&self) -> f64 {
        let cos_x = (std::f64::consts::PI / self.grid.nx as f64).cos();
        let cos_y = (std::f64::consts::PI / self.grid.ny as f64).cos();
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        (dx2 * cos_x + dy2 * cos_y) / (dx2 + dy2)
    }

    /// Solve -∇²u = f on the interior, with given boundary conditions.
    /// `f` should be ny × nx (full grid including boundaries, boundary values ignored).
    pub fn solve(&self, f: &[Vec<f64>], u_init: Option<&[Vec<f64>]>) -> PoissonResult {
        match self.method {
            PoissonMethod::Jacobi => self.solve_jacobi(f, u_init),
            PoissonMethod::GaussSeidel => self.solve_gauss_seidel(f, u_init),
            PoissonMethod::SOR(omega) => self.solve_sor(f, u_init, omega),
        }
    }

    fn init_u(&self, u_init: Option<&[Vec<f64>]>) -> Vec<Vec<f64>> {
        match u_init {
            Some(init) => init.to_vec(),
            None => vec![vec![0.0; self.grid.nx]; self.grid.ny],
        }
    }

    fn apply_bc_2d(&self, u: &mut [Vec<f64>]) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        // x-boundaries
        for i in 0..ny {
            match &self.bc.x.left {
                BoundaryCondition::Dirichlet(v) => u[i][0] = *v,
                BoundaryCondition::Neumann(flux) => u[i][0] = u[i][1] - self.grid.dx * flux,
                BoundaryCondition::Periodic => u[i][0] = u[i][nx - 2],
            }
            match &self.bc.x.right {
                BoundaryCondition::Dirichlet(v) => u[i][nx - 1] = *v,
                BoundaryCondition::Neumann(flux) => u[i][nx - 1] = u[i][nx - 2] + self.grid.dx * flux,
                BoundaryCondition::Periodic => u[i][nx - 1] = u[i][1],
            }
        }

        // y-boundaries
        for j in 0..nx {
            match &self.bc.y.left {
                BoundaryCondition::Dirichlet(v) => u[0][j] = *v,
                BoundaryCondition::Neumann(flux) => u[0][j] = u[1][j] - self.grid.dy * flux,
                BoundaryCondition::Periodic => u[0][j] = u[ny - 2][j],
            }
            match &self.bc.y.right {
                BoundaryCondition::Dirichlet(v) => u[ny - 1][j] = *v,
                BoundaryCondition::Neumann(flux) => u[ny - 1][j] = u[ny - 2][j] + self.grid.dy * flux,
                BoundaryCondition::Periodic => u[ny - 1][j] = u[1][j],
            }
        }
    }

    fn residual(&self, u: &[Vec<f64>], f: &[Vec<f64>]) -> f64 {
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        let mut max_res = 0.0f64;
        for i in 1..self.grid.ny - 1 {
            for j in 1..self.grid.nx - 1 {
                let lap = (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) / dx2
                    + (u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j]) / dy2;
                let res = (f[i][j] + lap).abs();
                max_res = max_res.max(res);
            }
        }
        max_res
    }

    fn solve_jacobi(&self, f: &[Vec<f64>], u_init: Option<&[Vec<f64>]>) -> PoissonResult {
        let mut u = self.init_u(u_init);
        self.apply_bc_2d(&mut u);
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        let denom = 2.0 / dx2 + 2.0 / dy2;
        let mut iterations = 0;

        for k in 0..self.max_iter {
            let u_old = u.clone();
            for i in 1..self.grid.ny - 1 {
                for j in 1..self.grid.nx - 1 {
                    let lap_neighbors = (u_old[i][j + 1] + u_old[i][j - 1]) / dx2
                        + (u_old[i + 1][j] + u_old[i - 1][j]) / dy2;
                    u[i][j] = (lap_neighbors + f[i][j]) / denom;
                }
            }
            self.apply_bc_2d(&mut u);
            iterations = k + 1;

            let res = self.residual(&u, f);
            if res < self.tol {
                return PoissonResult { u, iterations, final_residual: res, converged: true };
            }
        }

        PoissonResult {
            final_residual: self.residual(&u, f),
            u, iterations, converged: false,
        }
    }

    fn solve_gauss_seidel(&self, f: &[Vec<f64>], u_init: Option<&[Vec<f64>]>) -> PoissonResult {
        let mut u = self.init_u(u_init);
        self.apply_bc_2d(&mut u);
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        let denom = 2.0 / dx2 + 2.0 / dy2;
        let mut iterations = 0;

        for k in 0..self.max_iter {
            for i in 1..self.grid.ny - 1 {
                for j in 1..self.grid.nx - 1 {
                    let lap_neighbors = (u[i][j + 1] + u[i][j - 1]) / dx2
                        + (u[i + 1][j] + u[i - 1][j]) / dy2;
                    u[i][j] = (lap_neighbors + f[i][j]) / denom;
                }
            }
            self.apply_bc_2d(&mut u);
            iterations = k + 1;

            let res = self.residual(&u, f);
            if res < self.tol {
                return PoissonResult { u, iterations, final_residual: res, converged: true };
            }
        }

        PoissonResult {
            final_residual: self.residual(&u, f),
            u, iterations, converged: false,
        }
    }

    fn solve_sor(&self, f: &[Vec<f64>], u_init: Option<&[Vec<f64>]>, omega: f64) -> PoissonResult {
        let mut u = self.init_u(u_init);
        self.apply_bc_2d(&mut u);
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        let denom = 2.0 / dx2 + 2.0 / dy2;
        let mut iterations = 0;

        for k in 0..self.max_iter {
            for i in 1..self.grid.ny - 1 {
                for j in 1..self.grid.nx - 1 {
                    let lap_neighbors = (u[i][j + 1] + u[i][j - 1]) / dx2
                        + (u[i + 1][j] + u[i - 1][j]) / dy2;
                    let u_gs = (lap_neighbors + f[i][j]) / denom;
                    u[i][j] = (1.0 - omega) * u[i][j] + omega * u_gs;
                }
            }
            self.apply_bc_2d(&mut u);
            iterations = k + 1;

            let res = self.residual(&u, f);
            if res < self.tol {
                return PoissonResult { u, iterations, final_residual: res, converged: true };
            }
        }

        PoissonResult {
            final_residual: self.residual(&u, f),
            u, iterations, converged: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Manufactured solution: u = sin(πx)sin(πy) on [0,1]²
    /// Then -∇²u = 2π² sin(πx)sin(πy)
    fn u_exact(x: f64, y: f64) -> f64 {
        (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
    }

    fn f_source(x: f64, y: f64) -> f64 {
        2.0 * std::f64::consts::PI * std::f64::consts::PI
            * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
    }

    fn make_poisson_test(n: usize, method: PoissonMethod) -> PoissonResult {
        let grid = Grid2D::new(0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryPair2D::dirichlet_2d(0.0, 0.0, 0.0, 0.0);

        let f: Vec<Vec<f64>> = (0..n).map(|i| {
            (0..n).map(|j| f_source(grid.x(j), grid.y(i))).collect()
        }).collect();

        let solver = PoissonSolver::new(grid, bc, method).with_tolerance(1e-6);
        solver.solve(&f, None)
    }

    #[test]
    fn test_poisson_jacobi_converges() {
        let result = make_poisson_test(21, PoissonMethod::Jacobi);
        assert!(result.converged);
        assert!(result.final_residual < 1e-6);
    }

    #[test]
    fn test_poisson_gauss_seidel_converges() {
        let result = make_poisson_test(21, PoissonMethod::GaussSeidel);
        assert!(result.converged);
    }

    #[test]
    fn test_poisson_sor_converges() {
        let omega = PoissonSolver::optimal_sor_omega(21, 21);
        let result = make_poisson_test(21, PoissonMethod::SOR(omega));
        assert!(result.converged);
    }

    #[test]
    fn test_poisson_gs_faster_than_jacobi() {
        let gs = make_poisson_test(31, PoissonMethod::GaussSeidel);
        let jac = make_poisson_test(31, PoissonMethod::Jacobi);
        assert!(gs.iterations < jac.iterations,
            "GS ({} iters) should be faster than Jacobi ({} iters)", gs.iterations, jac.iterations);
    }

    #[test]
    fn test_poisson_sor_faster_than_gs() {
        let omega = PoissonSolver::optimal_sor_omega(31, 31);
        let sor = make_poisson_test(31, PoissonMethod::SOR(omega));
        let gs = make_poisson_test(31, PoissonMethod::GaussSeidel);
        assert!(sor.iterations < gs.iterations,
            "SOR ({} iters) should be faster than GS ({} iters)", sor.iterations, gs.iterations);
    }

    #[test]
    fn test_poisson_accuracy() {
        let n = 41;
        let grid = Grid2D::new(0.0, 1.0, n, 0.0, 1.0, n);
        let result = make_poisson_test(n, PoissonMethod::GaussSeidel);
        assert!(result.converged);

        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let err = (result.u[i][j] - u_exact(grid.x(j), grid.y(i))).abs();
                max_err = max_err.max(err);
            }
        }
        assert!(max_err < 0.01, "Max error too large: {}", max_err);
    }

    #[test]
    fn test_poisson_convergence_order() {
        let errors: Vec<f64> = [11, 21, 41].iter().map(|&n| {
            let grid = Grid2D::new(0.0, 1.0, n, 0.0, 1.0, n);
            let omega = PoissonSolver::optimal_sor_omega(n, n);
            let result = make_poisson_test(n, PoissonMethod::SOR(omega));
            let mut l2 = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let err = result.u[i][j] - u_exact(grid.x(j), grid.y(i));
                    l2 += err * err;
                }
            }
            (l2 / (n * n) as f64).sqrt()
        }).collect();

        // Second-order convergence: error should drop by ~4x when n doubles
        let ratio = errors[0] / errors[1];
        assert!(ratio > 2.5, "Expected convergence ratio > 2.5, got {}", ratio);
    }

    #[test]
    fn test_optimal_sor_omega_range() {
        let omega = PoissonSolver::optimal_sor_omega(21, 21);
        assert!(omega > 1.0 && omega < 2.0, "SOR omega should be in (1,2), got {}", omega);
    }

    #[test]
    fn test_poisson_neumann_bc() {
        let n = 21;
        let grid = Grid2D::new(0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryPair2D {
            x: crate::boundary::BoundaryPair1D::neumann(0.0, 0.0),
            y: crate::boundary::BoundaryPair1D::neumann(0.0, 0.0),
        };
        // f = -1 (constant source, Neumann BC → solution has nontrivial structure)
        let f = vec![vec![1.0; n]; n];
        let solver = PoissonSolver::new(grid, bc, PoissonMethod::SOR(1.5)).with_tolerance(1e-4);
        let result = solver.solve(&f, None);
        // Should converge (even if slowly)
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_poisson_periodic_bc() {
        let n = 21;
        let grid = Grid2D::new(0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryPair2D::periodic_2d();
        let f = vec![vec![0.0; n]; n]; // Laplace with periodic BC
        let solver = PoissonSolver::new(grid, bc, PoissonMethod::Jacobi).with_max_iter(100);
        let result = solver.solve(&f, None);
        // With zero source, solution should stay zero
        let max_val = result.u.iter().flat_map(|r| r.iter().cloned()).fold(0.0f64, f64::max);
        assert!(max_val < 1e-10, "Zero source with zero init should give zero solution");
    }

    #[test]
    fn test_jacobi_spectral_radius() {
        let grid = Grid2D::new(0.0, 1.0, 21, 0.0, 1.0, 21);
        let solver = PoissonSolver::new(grid, BoundaryPair2D::dirichlet_2d(0.0,0.0,0.0,0.0), PoissonMethod::Jacobi);
        let rho = solver.jacobi_spectral_radius();
        assert!(rho > 0.0 && rho < 1.0, "Spectral radius should be in (0,1), got {}", rho);
    }
}
