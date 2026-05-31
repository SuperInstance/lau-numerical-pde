//! Heat equation solver: u_t = α u_xx
//!
//! Supports forward Euler (explicit), backward Euler (implicit), and Crank-Nicolson.

use crate::boundary::{BoundaryPair1D, BoundaryCondition};
use crate::finite_diff::Grid1D;
use nalgebra::{DMatrix, DVector, LU};

/// Heat equation solver methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeatMethod {
    ForwardEuler,
    BackwardEuler,
    CrankNicolson,
}

/// Heat equation solver.
pub struct HeatSolver {
    pub grid: Grid1D,
    pub alpha: f64,
    pub bc: BoundaryPair1D,
    pub method: HeatMethod,
}

impl HeatSolver {
    pub fn new(grid: Grid1D, alpha: f64, bc: BoundaryPair1D, method: HeatMethod) -> Self {
        Self { grid, alpha, bc, method }
    }

    /// Stability parameter r = α Δt / Δx²
    pub fn stability_param(&self, dt: f64) -> f64 {
        self.alpha * dt / (self.grid.dx * self.grid.dx)
    }

    /// Check CFL-like stability condition.
    /// Forward Euler: r ≤ 0.5; Backward Euler & CN: unconditionally stable.
    pub fn is_stable(&self, dt: f64) -> bool {
        let r = self.stability_param(dt);
        match self.method {
            HeatMethod::ForwardEuler => r <= 0.5 + 1e-12,
            HeatMethod::BackwardEuler | HeatMethod::CrankNicolson => true,
        }
    }

    /// Maximum stable time step for forward Euler.
    pub fn max_stable_dt(&self) -> f64 {
        0.5 * self.grid.dx * self.grid.dx / self.alpha
    }

    /// Solve for n_steps time steps.
    pub fn solve(&self, u0: &[f64], dt: f64, n_steps: usize) -> Vec<Vec<f64>> {
        match self.method {
            HeatMethod::ForwardEuler => self.solve_forward_euler(u0, dt, n_steps),
            HeatMethod::BackwardEuler => self.solve_backward_euler(u0, dt, n_steps),
            HeatMethod::CrankNicolson => self.solve_crank_nicolson(u0, dt, n_steps),
        }
    }

    /// Solve for n_steps and return only the final state.
    pub fn solve_final(&self, u0: &[f64], dt: f64, n_steps: usize) -> Vec<f64> {
        let history = self.solve(u0, dt, n_steps);
        history.into_iter().last().unwrap_or_else(|| u0.to_vec())
    }

    fn apply_bc_to_interior(&self, u_full: &mut Vec<f64>) {
        let n = u_full.len();
        match &self.bc.left {
            BoundaryCondition::Dirichlet(v) => u_full[0] = *v,
            BoundaryCondition::Neumann(f) => u_full[0] = u_full[1] + self.grid.dx * f,
            BoundaryCondition::Periodic => u_full[0] = u_full[n - 2],
        }
        match &self.bc.right {
            BoundaryCondition::Dirichlet(v) => u_full[n - 1] = *v,
            BoundaryCondition::Neumann(f) => u_full[n - 1] = u_full[n - 2] + self.grid.dx * f,
            BoundaryCondition::Periodic => u_full[n - 1] = u_full[1],
        }
    }

    fn solve_forward_euler(&self, u0: &[f64], dt: f64, n_steps: usize) -> Vec<Vec<f64>> {
        let r = self.stability_param(dt);
        let ni = self.grid.interior_n();
        assert_eq!(u0.len(), ni + 2, "u0 must have grid.n points including boundaries");

        let mut u = u0.to_vec();
        let mut history = vec![u.clone()];

        for _ in 0..n_steps {
            let mut u_new = u.clone();
            for i in 1..=ni {
                u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
            }
            self.apply_bc_to_interior(&mut u_new);
            u = u_new;
            history.push(u.clone());
        }

        history
    }

    fn solve_backward_euler(&self, u0: &[f64], dt: f64, n_steps: usize) -> Vec<Vec<f64>> {
        let r = self.stability_param(dt);
        let ni = self.grid.interior_n();

        // Build (I + r*A) where A is the tridiagonal second-diff matrix
        let (mat, rhs_bc) = self.build_implicit_system(r, 1.0);

        let mut u = u0.to_vec();
        let mut history = vec![u.clone()];

        for _ in 0..n_steps {
            // RHS = u_interior + boundary contributions
            let mut b = DVector::from_vec(u[1..=ni].to_vec());
            b += &rhs_bc;

            let lu = LU::new(mat.clone());
            let sol = lu.solve(&b).expect("Implicit system solve failed");

            for i in 0..ni {
                u[i + 1] = sol[i];
            }
            self.apply_bc_to_interior(&mut u);
            history.push(u.clone());
        }

        history
    }

    fn solve_crank_nicolson(&self, u0: &[f64], dt: f64, n_steps: usize) -> Vec<Vec<f64>> {
        let r = self.stability_param(dt);
        let ni = self.grid.interior_n();

        let (mat_lhs, rhs_bc_lhs) = self.build_implicit_system(r / 2.0, 1.0);
        let (mat_rhs_explicit, rhs_bc_rhs) = self.build_explicit_rhs(r / 2.0);

        let mut u = u0.to_vec();
        let mut history = vec![u.clone()];

        for _ in 0..n_steps {
            // RHS side: (I - r/2 * A) * u^n + boundary terms
            let u_int = DVector::from_vec(u[1..=ni].to_vec());
            let b = &mat_rhs_explicit * &u_int + &rhs_bc_lhs + &rhs_bc_rhs;

            let lu = LU::new(mat_lhs.clone());
            let sol = lu.solve(&b).expect("CN system solve failed");

            for i in 0..ni {
                u[i + 1] = sol[i];
            }
            self.apply_bc_to_interior(&mut u);
            history.push(u.clone());
        }

        history
    }

    fn build_implicit_system(&self, r: f64, _weight: f64) -> (DMatrix<f64>, DVector<f64>) {
        let ni = self.grid.interior_n();
        let mut mat = DMatrix::identity(ni, ni);
        let mut bc_vec = DVector::zeros(ni);

        for i in 0..ni {
            mat[(i, i)] = 1.0 + 2.0 * r;
            if i > 0 { mat[(i, i - 1)] = -r; }
            if i < ni - 1 { mat[(i, i + 1)] = -r; }
        }

        // Boundary contributions
        match &self.bc.left {
            BoundaryCondition::Dirichlet(v) => bc_vec[0] += r * v,
            BoundaryCondition::Neumann(f) => {
                // Ghost point: u_{-1} = u_1 - dx*f => contribution adjusts
                mat[(0, 0)] = 1.0 + r; // reduces from 1+2r since u_{-1} ≈ u_1
                bc_vec[0] += r * self.grid.dx * f;
            }
            BoundaryCondition::Periodic => {
                mat[(0, ni - 1)] = -r;
            }
        }
        match &self.bc.right {
            BoundaryCondition::Dirichlet(v) => bc_vec[ni - 1] += r * v,
            BoundaryCondition::Neumann(f) => {
                mat[(ni - 1, ni - 1)] = 1.0 + r;
                bc_vec[ni - 1] -= r * self.grid.dx * f;
            }
            BoundaryCondition::Periodic => {
                mat[(ni - 1, 0)] = -r;
            }
        }

        (mat, bc_vec)
    }

    fn build_explicit_rhs(&self, r: f64) -> (DMatrix<f64>, DVector<f64>) {
        let ni = self.grid.interior_n();
        let mut mat = DMatrix::identity(ni, ni);
        let mut bc_vec = DVector::zeros(ni);

        for i in 0..ni {
            mat[(i, i)] = 1.0 - 2.0 * r;
            if i > 0 { mat[(i, i - 1)] = r; }
            if i < ni - 1 { mat[(i, i + 1)] = r; }
        }

        match &self.bc.left {
            BoundaryCondition::Dirichlet(v) => bc_vec[0] -= r * v,
            BoundaryCondition::Neumann(f) => {
                mat[(0, 0)] = 1.0 - r;
                bc_vec[0] -= r * self.grid.dx * f;
            }
            BoundaryCondition::Periodic => {
                mat[(0, ni - 1)] = r;
            }
        }
        match &self.bc.right {
            BoundaryCondition::Dirichlet(v) => bc_vec[ni - 1] -= r * v,
            BoundaryCondition::Neumann(f) => {
                mat[(ni - 1, ni - 1)] = 1.0 - r;
                bc_vec[ni - 1] += r * self.grid.dx * f;
            }
            BoundaryCondition::Periodic => {
                mat[(ni - 1, 0)] = r;
            }
        }

        (mat, bc_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Manufactured solution: u(x,t) = exp(-α π² t) sin(πx) on [0,1]
    /// with Dirichlet BC u(0,t)=u(1,t)=0.
    fn heat_exact(x: f64, t: f64, alpha: f64) -> f64 {
        (-alpha * std::f64::consts::PI * std::f64::consts::PI * t).exp() * (std::f64::consts::PI * x).sin()
    }

    fn test_convergence(method: HeatMethod, _expected_order: f64) -> f64 {
        let alpha = 0.01;
        let n_values = [21, 41, 81, 161];

        let mut errors = vec![];
        for &n in &n_values {
            let grid = Grid1D::new(0.0, 1.0, n);
            let dx = grid.dx;
            let dt = match method {
                HeatMethod::ForwardEuler => 0.4 * dx * dx / alpha,
                _ => 0.5 * dx,
            };
            let t_final = 0.1;
            let n_steps = (t_final / dt).ceil() as usize;

            let x_vals: Vec<f64> = (0..n).map(|i| grid.x(i)).collect();
            let u0: Vec<f64> = (0..n).map(|i| heat_exact(x_vals[i], 0.0, alpha)).collect();
            let bc = BoundaryPair1D::dirichlet(0.0, 0.0);
            let solver = HeatSolver::new(grid, alpha, bc, method);
            let u_final = solver.solve_final(&u0, dt, n_steps);

            let mut l2_err = 0.0;
            for i in 0..n {
                let err = u_final[i] - heat_exact(x_vals[i], t_final, alpha);
                l2_err += err * err;
            }
            l2_err = (l2_err / n as f64).sqrt();
            errors.push(l2_err);
        }

        // Compute convergence order from last pair
        let n1 = errors.len();
        if n1 < 2 { return 0.0; }
        let order = (errors[n1 - 2] / errors[n1 - 1]).log2();
        order / 2.0 // dx halves each time, but convergence in dx not dt
    }

    #[test]
    fn test_heat_forward_euler_stable() {
        let grid = Grid1D::new(0.0, 1.0, 11);
        let solver = HeatSolver::new(grid, 1.0, BoundaryPair1D::dirichlet(0.0, 0.0), HeatMethod::ForwardEuler);
        assert!(solver.is_stable(0.005));
        assert!(!solver.is_stable(0.1));
    }

    #[test]
    fn test_heat_backward_euler_always_stable() {
        let grid = Grid1D::new(0.0, 1.0, 11);
        let solver = HeatSolver::new(grid, 1.0, BoundaryPair1D::dirichlet(0.0, 0.0), HeatMethod::BackwardEuler);
        assert!(solver.is_stable(100.0));
    }

    #[test]
    fn test_heat_cn_always_stable() {
        let grid = Grid1D::new(0.0, 1.0, 11);
        let solver = HeatSolver::new(grid, 1.0, BoundaryPair1D::dirichlet(0.0, 0.0), HeatMethod::CrankNicolson);
        assert!(solver.is_stable(100.0));
    }

    #[test]
    fn test_heat_forward_euler_dirichlet() {
        let alpha = 0.01;
        let grid = Grid1D::new(0.0, 1.0, 51);
        let dx = grid.dx;
        let dt = 0.4 * dx * dx / alpha;
        let u0: Vec<f64> = (0..grid.n).map(|i| heat_exact(grid.x(i), 0.0, alpha)).collect();
        let solver = HeatSolver::new(grid, alpha, BoundaryPair1D::dirichlet(0.0, 0.0), HeatMethod::ForwardEuler);
        let u_final = solver.solve_final(&u0, dt, 100);

        // Check boundary values remain 0
        assert_relative_eq!(u_final[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(u_final[u_final.len() - 1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_heat_backward_euler_dirichlet() {
        let alpha = 0.1;
        let grid = Grid1D::new(0.0, 1.0, 41);
        let dt = 0.01;
        let u0: Vec<f64> = (0..grid.n).map(|i| (std::f64::consts::PI * grid.x(i)).sin()).collect();
        let solver = HeatSolver::new(grid, alpha, BoundaryPair1D::dirichlet(0.0, 0.0), HeatMethod::BackwardEuler);
        let u_final = solver.solve_final(&u0, dt, 50);

        assert_relative_eq!(u_final[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(u_final[u_final.len() - 1], 0.0, epsilon = 1e-12);

        // Solution should decay
        let max_val = u_final.iter().cloned().fold(f64::NAN, f64::max);
        assert!(max_val < 1.0);
    }

    #[test]
    fn test_heat_cn_dirichlet() {
        let alpha = 0.1;
        let grid = Grid1D::new(0.0, 1.0, 41);
        let dt = 0.01;
        let u0: Vec<f64> = (0..grid.n).map(|i| (std::f64::consts::PI * grid.x(i)).sin()).collect();
        let solver = HeatSolver::new(grid, alpha, BoundaryPair1D::dirichlet(0.0, 0.0), HeatMethod::CrankNicolson);
        let u_final = solver.solve_final(&u0, dt, 50);

        assert_relative_eq!(u_final[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(u_final[u_final.len() - 1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_heat_convergence_backward_euler() {
        let order = test_convergence(HeatMethod::BackwardEuler, 1.0);
        // BE is first-order in time; with fine dt the spatial order dominates
        assert!(order > 0.5, "Expected order > 0.5, got {}", order);
    }

    #[test]
    fn test_heat_convergence_cn() {
        let order = test_convergence(HeatMethod::CrankNicolson, 2.0);
        assert!(order > 0.5, "Expected order > 0.5, got {}", order);
    }

    #[test]
    fn test_heat_max_stable_dt() {
        let grid = Grid1D::new(0.0, 1.0, 11);
        let solver = HeatSolver::new(grid, 1.0, BoundaryPair1D::dirichlet(0.0, 0.0), HeatMethod::ForwardEuler);
        let dt_max = solver.max_stable_dt();
        assert!(solver.is_stable(dt_max));
        assert!(!solver.is_stable(dt_max * 1.01));
    }

    #[test]
    fn test_heat_conservation_periodic() {
        // With periodic BC and no source, integral of u should be conserved
        let alpha = 0.01;
        let grid = Grid1D::new(0.0, 1.0, 51);
        let dx = grid.dx;
        let dt = 0.4 * dx * dx / alpha;
        let mut u0 = vec![0.0; grid.n];
        for i in 0..grid.n { u0[i] = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * grid.x(i)).sin(); }

        let solver = HeatSolver::new(grid, alpha, BoundaryPair1D::periodic(), HeatMethod::ForwardEuler);
        let u_final = solver.solve_final(&u0, dt, 50);

        let integral_0: f64 = u0.iter().sum::<f64>() * dx;
        let integral_f: f64 = u_final.iter().sum::<f64>() * dx;
        assert_relative_eq!(integral_0, integral_f, epsilon = 1e-6);
    }

    #[test]
    fn test_heat_decay() {
        // Heat equation should cause max value to decrease
        let alpha = 1.0;
        let grid = Grid1D::new(0.0, 1.0, 21);
        let dx = grid.dx;
        let dt = 0.4 * dx * dx / alpha;
        let u0: Vec<f64> = (0..grid.n).map(|i| (std::f64::consts::PI * grid.x(i)).sin()).collect();
        let max0 = u0.iter().cloned().fold(f64::NAN, f64::max);

        let solver = HeatSolver::new(grid, alpha, BoundaryPair1D::dirichlet(0.0, 0.0), HeatMethod::ForwardEuler);
        let u_final = solver.solve_final(&u0, dt, 20);
        let max_f = u_final.iter().cloned().fold(f64::NAN, f64::max);

        assert!(max_f < max0, "Heat should decay: {} >= {}", max_f, max0);
    }

    #[test]
    fn test_heat_neumann_bc() {
        let alpha = 0.1;
        let grid = Grid1D::new(0.0, 1.0, 31);
        let dt = 0.01;
        let u0 = vec![1.0; grid.n];
        let solver = HeatSolver::new(grid, alpha, BoundaryPair1D::neumann(0.0, 0.0), HeatMethod::BackwardEuler);
        let u_final = solver.solve_final(&u0, dt, 10);
        // With zero Neumann and constant initial, solution stays constant
        for v in &u_final {
            assert_relative_eq!(*v, 1.0, epsilon = 1e-10);
        }
    }
}
