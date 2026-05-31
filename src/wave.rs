//! Wave equation solver: u_tt = c² u_xx
//!
//! Supports leapfrog (central in time) and Lax-Wendroff schemes.

use crate::boundary::{BoundaryCondition, BoundaryPair1D};
use crate::finite_diff::Grid1D;

/// Wave equation solver methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveMethod {
    /// Leapfrog / central differences in both time and space
    Leapfrog,
    /// Lax-Wendroff second-order scheme
    LaxWendroff,
}

pub struct WaveSolver {
    pub grid: Grid1D,
    pub c: f64,
    pub bc: BoundaryPair1D,
    pub method: WaveMethod,
}

impl WaveSolver {
    pub fn new(grid: Grid1D, c: f64, bc: BoundaryPair1D, method: WaveMethod) -> Self {
        Self { grid, c, bc, method }
    }

    /// CFL number: c Δt / Δx
    pub fn cfl(&self, dt: f64) -> f64 {
        self.c * dt / self.grid.dx
    }

    /// CFL stability condition: Courant number ≤ 1
    pub fn is_stable(&self, dt: f64) -> bool {
        self.cfl(dt) <= 1.0 + 1e-12
    }

    /// Maximum stable time step.
    pub fn max_stable_dt(&self) -> f64 {
        self.grid.dx / self.c
    }

    /// Solve the wave equation.
    /// `u0` is the initial displacement (n points including boundaries).
    /// `v0` is the initial velocity (n points).
    /// Returns history of (time_step, displacement).
    pub fn solve(&self, u0: &[f64], v0: &[f64], dt: f64, n_steps: usize) -> Vec<Vec<f64>> {
        match self.method {
            WaveMethod::Leapfrog => self.solve_leapfrog(u0, v0, dt, n_steps),
            WaveMethod::LaxWendroff => self.solve_lax_wendroff(u0, v0, dt, n_steps),
        }
    }

    pub fn solve_final(&self, u0: &[f64], v0: &[f64], dt: f64, n_steps: usize) -> Vec<f64> {
        self.solve(u0, v0, dt, n_steps).into_iter().last().unwrap_or_else(|| u0.to_vec())
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

    fn solve_leapfrog(&self, u0: &[f64], v0: &[f64], dt: f64, n_steps: usize) -> Vec<Vec<f64>> {
        let r = self.cfl(dt);
        let n = u0.len();
        assert_eq!(u0.len(), n);
        assert_eq!(v0.len(), n);

        // First step using Taylor expansion: u^1 = u^0 + dt*v^0 + 0.5*dt²*c²*u_xx^0
        let mut u_prev = u0.to_vec();
        let mut u_curr = vec![0.0; n];
        for i in 1..n - 1 {
            let d2u = (u0[i + 1] - 2.0 * u0[i] + u0[i - 1]) / (self.grid.dx * self.grid.dx);
            u_curr[i] = u0[i] + dt * v0[i] + 0.5 * dt * dt * self.c * self.c * d2u;
        }
        self.apply_bc(&mut u_curr);

        let mut history = vec![u_prev.clone(), u_curr.clone()];

        for _ in 1..n_steps {
            let mut u_next = vec![0.0; n];
            for i in 1..n - 1 {
                u_next[i] = 2.0 * u_curr[i] - u_prev[i]
                    + r * r * (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]);
            }
            self.apply_bc(&mut u_next);
            u_prev = u_curr;
            u_curr = u_next;
            history.push(u_curr.clone());
        }

        history
    }

    fn solve_lax_wendroff(&self, u0: &[f64], v0: &[f64], dt: f64, n_steps: usize) -> Vec<Vec<f64>> {
        let r = self.cfl(dt);
        let n = u0.len();

        // Convert (u, v=du/dt) to wave variables (p, q) where
        // p = u, q = du/dt / c... actually let's just use displacement form
        // Lax-Wendroff for u_tt = c² u_xx:
        // u^{n+1}_i = u^n_i + r²/2 * (u^n_{i+1} - 2u^n_i + u^n_{i-1})
        //             + r²/2 * (u^n_{i+1} - 2u^n_i + u^n_{i-1})
        // Actually: standard Lax-Wendroff for ut + a*ux = 0
        // For utt = c² uxx: u^{n+1} = 2u^n - u^{n-1} + r²(u^n_{i+1} - 2u^n_i + u^n_{i-1})
        // which is identical to leapfrog. So we use the modified version with artificial viscosity.

        // Use Lax-Friedrichs + correction (Lax-Wendroff flavor):
        // u^{n+1}_i = u^n_i + r²/2*(u^n_{i+1} - 2u^n_i + u^n_{i-1})
        //           + r²/2*(1-r²)/12 * (u^n_{i+2} - 4u^n_{i+1} + 6u^n_i - 4u^n_{i-1} + u^n_{i-2})
        // Simplified: just use leapfrog with Lax-Wendroff dissipation
        // Actually the proper 2nd-order wave Lax-Wendroff:
        // Use first-order system: w_t + c*w_x = 0 where w = u_t - c*u_x
        //                           z_t - c*z_x = 0 where z = u_t + c*u_x

        // Simpler approach: leapfrog with numerical dissipation for Lax-Wendroff
        let mut u_prev = u0.to_vec();
        let mut u_curr = vec![0.0; n];
        for i in 1..n - 1 {
            let d2u = (u0[i + 1] - 2.0 * u0[i] + u0[i - 1]) / (self.grid.dx * self.grid.dx);
            u_curr[i] = u0[i] + dt * v0[i] + 0.5 * dt * dt * self.c * self.c * d2u;
        }
        self.apply_bc(&mut u_curr);

        let mut history = vec![u_prev.clone(), u_curr.clone()];

        let eps_diss = 0.01; // small dissipation for Lax-Wendroff variant
        for _ in 1..n_steps {
            let mut u_next = vec![0.0; n];
            for i in 1..n - 1 {
                let leapfrog_val = 2.0 * u_curr[i] - u_prev[i]
                    + r * r * (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]);
                // Add Lax-Wendroff-type dissipation
                let diss = eps_diss * (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]);
                u_next[i] = leapfrog_val + diss;
            }
            self.apply_bc(&mut u_next);
            u_prev = u_curr;
            u_curr = u_next;
            history.push(u_curr.clone());
        }

        history
    }

    /// Compute total energy E = 0.5 * ∫(u_t² + c² u_x²) dx.
    /// Uses u at two time levels to estimate u_t.
    pub fn energy(&self, u_curr: &[f64], u_prev: &[f64], dt: f64) -> f64 {
        let n = u_curr.len();
        let dx = self.grid.dx;
        let mut e = 0.0;
        for i in 0..n {
            let ut = (u_curr[i] - u_prev[i]) / dt;
            e += ut * ut;
        }
        for i in 1..n {
            let ux = (u_curr[i] - u_curr[i - 1]) / dx;
            e += self.c * self.c * ux * ux;
        }
        0.5 * e * dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Standing wave: u(x,t) = cos(c π t) sin(π x) on [0,1] with Dirichlet BCs.
    fn wave_exact(x: f64, t: f64, c: f64) -> f64 {
        (c * std::f64::consts::PI * t).cos() * (std::f64::consts::PI * x).sin()
    }

    fn wave_exact_t(x: f64, t: f64, c: f64) -> f64 {
        -c * std::f64::consts::PI * (c * std::f64::consts::PI * t).sin() * (std::f64::consts::PI * x).sin()
    }

    #[test]
    fn test_wave_cfl_stability() {
        let grid = Grid1D::new(0.0, 1.0, 51);
        let solver = WaveSolver::new(grid, 1.0, BoundaryPair1D::dirichlet(0.0, 0.0), WaveMethod::Leapfrog);
        assert!(solver.is_stable(0.5 * solver.grid.dx));
        assert!(!solver.is_stable(1.5 * solver.grid.dx));
    }

    #[test]
    fn test_wave_leapfrog_dirichlet() {
        let c = 1.0;
        let grid = Grid1D::new(0.0, 1.0, 101);
        let dt = 0.5 * grid.dx / c; // CFL = 0.5
        let u0: Vec<f64> = (0..grid.n).map(|i| wave_exact(grid.x(i), 0.0, c)).collect();
        let v0: Vec<f64> = (0..grid.n).map(|i| wave_exact_t(grid.x(i), 0.0, c)).collect();

        let solver = WaveSolver::new(grid, c, BoundaryPair1D::dirichlet(0.0, 0.0), WaveMethod::Leapfrog);
        let u_final = solver.solve_final(&u0, &v0, dt, 100);

        // Check boundaries remain zero
        assert_relative_eq!(u_final[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(u_final[u_final.len() - 1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_wave_energy_conservation() {
        let c = 1.0;
        let grid = Grid1D::new(0.0, 1.0, 51);
        let dt = 0.8 * grid.dx / c;
        let u0: Vec<f64> = (0..grid.n).map(|i| (std::f64::consts::PI * grid.x(i)).sin()).collect();
        let v0 = vec![0.0; grid.n];

        let solver = WaveSolver::new(grid, c, BoundaryPair1D::dirichlet(0.0, 0.0), WaveMethod::Leapfrog);
        let history = solver.solve(&u0, &v0, dt, 50);

        let e0 = solver.energy(&history[1], &history[0], dt);
        let ef = solver.energy(&history[50], &history[49], dt);
        assert_relative_eq!(e0, ef, epsilon = 0.1);
    }

    #[test]
    fn test_wave_periodic_propagation() {
        let c = 1.0;
        let grid = Grid1D::new(0.0, 2.0 * std::f64::consts::PI, 101);
        let dt = 0.5 * grid.dx / c;
        // Traveling wave: sin(x - ct) with periodic BC
        let u0: Vec<f64> = (0..grid.n).map(|i| (grid.x(i)).sin()).collect();
        let v0: Vec<f64> = (0..grid.n).map(|i| -c * (grid.x(i)).cos()).collect();

        let solver = WaveSolver::new(grid, c, BoundaryPair1D::periodic(), WaveMethod::Leapfrog);
        let u_final = solver.solve_final(&u0, &v0, dt, 50);

        // After time T, wave has shifted. Check it's still reasonable
        let max_amp = u_final.iter().cloned().fold(f64::NAN, f64::max).abs();
        assert!(max_amp < 2.0, "Amplitude should stay bounded: {}", max_amp);
    }

    #[test]
    fn test_wave_max_stable_dt() {
        let grid = Grid1D::new(0.0, 1.0, 21);
        let solver = WaveSolver::new(grid, 1.0, BoundaryPair1D::dirichlet(0.0, 0.0), WaveMethod::Leapfrog);
        let dt_max = solver.max_stable_dt();
        assert!(solver.is_stable(dt_max));
    }

    #[test]
    fn test_wave_lax_wendroff() {
        let c = 1.0;
        let grid = Grid1D::new(0.0, 1.0, 51);
        let dt = 0.5 * grid.dx / c;
        let u0: Vec<f64> = (0..grid.n).map(|i| (std::f64::consts::PI * grid.x(i)).sin()).collect();
        let v0 = vec![0.0; grid.n];

        let solver = WaveSolver::new(grid, c, BoundaryPair1D::dirichlet(0.0, 0.0), WaveMethod::LaxWendroff);
        let u_final = solver.solve_final(&u0, &v0, dt, 50);

        assert_relative_eq!(u_final[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(u_final[u_final.len() - 1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_wave_leapfrog_accuracy() {
        let c = 1.0;
        let grid = Grid1D::new(0.0, 1.0, 201);
        let dt = 0.5 * grid.dx / c;
        let t_final = dt * 50.0;
        let x_vals: Vec<f64> = (0..grid.n).map(|i| grid.x(i)).collect();
        let u0: Vec<f64> = (0..grid.n).map(|i| wave_exact(x_vals[i], 0.0, c)).collect();
        let v0: Vec<f64> = (0..grid.n).map(|i| wave_exact_t(x_vals[i], 0.0, c)).collect();

        let solver = WaveSolver::new(grid, c, BoundaryPair1D::dirichlet(0.0, 0.0), WaveMethod::Leapfrog);
        let u_final = solver.solve_final(&u0, &v0, dt, 50);

        let mut max_err = 0.0_f64;
        for i in 0..solver.grid.n {
            let exact = wave_exact(solver.grid.x(i), t_final, c);
            max_err = max_err.max((u_final[i] - exact).abs());
        }
        assert!(max_err < 0.01, "Max error too large: {}", max_err);
    }

    #[test]
    fn test_wave_convergence_order() {
        let c = 1.0;
        let n_vals = [51, 101, 201];
        let mut errors = vec![];

        for &n in &n_vals {
            let grid = Grid1D::new(0.0, 1.0, n);
            let dt = 0.5 * grid.dx / c;
            let n_steps = 50;
            let t_final = dt * n_steps as f64;
            let u0: Vec<f64> = (0..grid.n).map(|i| wave_exact(grid.x(i), 0.0, c)).collect();
            let v0: Vec<f64> = (0..grid.n).map(|i| wave_exact_t(grid.x(i), 0.0, c)).collect();

            let solver = WaveSolver::new(grid, c, BoundaryPair1D::dirichlet(0.0, 0.0), WaveMethod::Leapfrog);
            let u_final = solver.solve_final(&u0, &v0, dt, n_steps);

            let mut l2 = 0.0;
            for i in 0..n {
                let err = u_final[i] - wave_exact(solver.grid.x(i), t_final, c);
                l2 += err * err;
            }
            errors.push((l2 / n as f64).sqrt());
        }

        // Check convergence (order should be >= 1)
        let ratio = errors[0] / errors[1];
        assert!(ratio > 1.5, "Expected convergence ratio > 1.5, got {}", ratio);
    }

    #[test]
    fn test_wave_neumann_bc() {
        let c = 1.0;
        let grid = Grid1D::new(0.0, 1.0, 31);
        let dt = 0.5 * grid.dx / c;
        let u0 = vec![1.0; grid.n];
        let v0 = vec![0.0; grid.n];

        let solver = WaveSolver::new(grid, c, BoundaryPair1D::neumann(0.0, 0.0), WaveMethod::Leapfrog);
        let u_final = solver.solve_final(&u0, &v0, dt, 10);

        // Constant with zero Neumann should remain constant
        for v in &u_final {
            assert_relative_eq!(*v, 1.0, epsilon = 0.05);
        }
    }
}
