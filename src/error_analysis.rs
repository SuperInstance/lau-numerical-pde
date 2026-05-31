//! Error analysis utilities for numerical PDE solutions.

use crate::finite_diff::Grid1D;

/// Error analysis and convergence verification tools.
pub struct ErrorAnalysis;

impl ErrorAnalysis {
    /// L2 (root mean square) error norm.
    pub fn l2_error(numerical: &[f64], exact: &[f64], dx: f64) -> f64 {
        assert_eq!(numerical.len(), exact.len());
        let n = numerical.len();
        let sum: f64 = numerical.iter().zip(exact.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        (sum * dx / (n as f64 * dx)).sqrt()
    }

    /// L∞ (maximum) error norm.
    pub fn linf_error(numerical: &[f64], exact: &[f64]) -> f64 {
        numerical.iter().zip(exact.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max)
    }

    /// L1 error norm.
    pub fn l1_error(numerical: &[f64], exact: &[f64], dx: f64) -> f64 {
        let sum: f64 = numerical.iter().zip(exact.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        sum * dx
    }

    /// Compute convergence order from a series of errors on successively refined grids.
    /// Returns the estimated order: log2(e_coarse / e_fine).
    pub fn convergence_order(errors: &[f64]) -> f64 {
        assert!(errors.len() >= 2);
        let n = errors.len();
        if errors[n - 1] <= 0.0 || errors[n - 2] <= 0.0 {
            return f64::NAN;
        }
        (errors[n - 2] / errors[n - 1]).log2()
    }

    /// Compute convergence order from a series of (grid_spacing, error) pairs.
    /// Fits log(error) = a + b*log(h), returns |b|.
    pub fn convergence_order_regression(pairs: &[(f64, f64)]) -> f64 {
        let n = pairs.len() as f64;
        let lh: Vec<f64> = pairs.iter().map(|(h, _)| h.ln()).collect();
        let le: Vec<f64> = pairs.iter().map(|(_, e)| e.ln()).collect();

        let sum_lh: f64 = lh.iter().sum();
        let sum_le: f64 = le.iter().sum();
        let sum_lh2: f64 = lh.iter().map(|x| x * x).sum();
        let sum_lhle: f64 = lh.iter().zip(le.iter()).map(|(h, e)| h * e).sum();

        let denom = n * sum_lh2 - sum_lh * sum_lh;
        if denom.abs() < 1e-30 { return f64::NAN; }

        let slope = (n * sum_lhle - sum_lh * sum_le) / denom;
        slope.abs()
    }

    /// Verify that convergence order is at least `expected_order` within `tolerance`.
    pub fn verify_convergence_order(errors: &[f64], expected_order: f64, tolerance: f64) -> bool {
        let computed = Self::convergence_order(errors);
        computed >= expected_order - tolerance
    }

    /// Richardson extrapolation: given solutions on grids with spacing h and h/2,
    /// estimate the error at the finer grid.
    pub fn richardson_extrapolation(u_h: &[f64], u_h2: &[f64], order: f64) -> Vec<f64> {
        // Extrapolated = u_h2 + (u_h2 - u_h) / (2^order - 1)
        let factor = 1.0 / (2.0_f64.powf(order) - 1.0);
        u_h2.iter().zip(u_h.iter())
            .map(|(fine, coarse)| fine + factor * (fine - coarse))
            .collect()
    }

    /// Compute error estimate using Richardson extrapolation.
    pub fn richardson_error_estimate(u_h: &[f64], u_h2: &[f64], order: f64) -> Vec<f64> {
        let factor = 1.0 / (2.0_f64.powf(order) - 1.0);
        u_h2.iter().zip(u_h.iter())
            .map(|(fine, coarse)| (factor * (fine - coarse)).abs())
            .collect()
    }

    /// Total variation of a 1D solution.
    pub fn total_variation(u: &[f64]) -> f64 {
        u.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
    }

    /// Run a convergence study for a PDE solver.
    /// `solver_fn` takes (n_points) and returns (grid, numerical_solution).
    /// `exact_fn` takes x and returns exact value.
    pub fn convergence_study(
        n_values: &[usize],
        solver_fn: &dyn Fn(usize) -> (Grid1D, Vec<f64>),
        exact_fn: &dyn Fn(f64) -> f64,
    ) -> Vec<(usize, f64, f64)> {
        n_values.iter().map(|&n| {
            let (grid, sol) = solver_fn(n);
            let exact: Vec<f64> = (0..n).map(|i| exact_fn(grid.x(i))).collect();
            let l2 = (0..n).map(|i| (sol[i] - exact[i]).powi(2)).sum::<f64>().sqrt();
            let linf = Self::linf_error(&sol, &exact);
            (n, l2, linf)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_l2_error_known() {
        let numerical = vec![1.0, 2.0, 3.0];
        let exact = vec![1.1, 2.1, 3.1];
        let l2 = ErrorAnalysis::l2_error(&numerical, &exact, 0.5);
        assert_relative_eq!(l2, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_linf_error() {
        let numerical = vec![1.0, 2.5, 3.0];
        let exact = vec![1.1, 2.0, 3.2];
        let linf = ErrorAnalysis::linf_error(&numerical, &exact);
        assert_relative_eq!(linf, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_l1_error() {
        let numerical = vec![1.0, 2.0, 3.0];
        let exact = vec![1.0, 2.0, 3.0];
        let l1 = ErrorAnalysis::l1_error(&numerical, &exact, 0.5);
        assert_relative_eq!(l1, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_convergence_order_exact() {
        // Simulate second-order convergence: error = C * h^2
        let errors = vec![0.01, 0.0025, 0.000625]; // h, h/2, h/4 with order 2
        let order = ErrorAnalysis::convergence_order(&errors[0..2]);
        assert_relative_eq!(order, 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_convergence_order_first() {
        let errors = vec![0.1, 0.05];
        let order = ErrorAnalysis::convergence_order(&errors);
        assert_relative_eq!(order, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_convergence_order_regression() {
        let pairs = vec![
            (0.1, 0.01),   // h=0.1, err=0.01
            (0.05, 0.0025), // h=0.05, err=0.0025 (order 2)
            (0.025, 0.000625),
        ];
        let order = ErrorAnalysis::convergence_order_regression(&pairs);
        assert_relative_eq!(order, 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_verify_convergence_pass() {
        let errors = vec![0.01, 0.0025];
        assert!(ErrorAnalysis::verify_convergence_order(&errors, 2.0, 0.1));
    }

    #[test]
    fn test_verify_convergence_fail() {
        let errors = vec![0.01, 0.005]; // order 1
        assert!(!ErrorAnalysis::verify_convergence_order(&errors, 2.0, 0.1));
    }

    #[test]
    fn test_richardson_extrapolation() {
        // If true solution is T, and u_h = T + e*h^2, u_h2 = T + e*(h/2)^2
        // Richardson: T ≈ u_h2 + (u_h2 - u_h) / 3
        let t_exact = 5.0;
        let h = 0.1;
        let e_coeff = 1.0;
        let u_h = vec![t_exact + e_coeff * h * h]; // 5.01
        let u_h2 = vec![t_exact + e_coeff * (h / 2.0_f64).powi(2)]; // 5.0025

        let extrap = ErrorAnalysis::richardson_extrapolation(&u_h, &u_h2, 2.0);
        assert_relative_eq!(extrap[0], t_exact, epsilon = 1e-12);
    }

    #[test]
    fn test_richardson_error_estimate() {
        let u_h = vec![5.01];
        let u_h2 = vec![5.0025];
        let err_est = ErrorAnalysis::richardson_error_estimate(&u_h, &u_h2, 2.0);
        // Error estimate ≈ (5.0025 - 5.01)/3 = 0.0025
        assert_relative_eq!(err_est[0], 0.0025, epsilon = 1e-10);
    }

    #[test]
    fn test_total_variation() {
        let u = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let tv = ErrorAnalysis::total_variation(&u);
        assert_relative_eq!(tv, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_total_variation_constant() {
        let u = vec![3.0; 10];
        let tv = ErrorAnalysis::total_variation(&u);
        assert_relative_eq!(tv, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_convergence_study() {
        let results = ErrorAnalysis::convergence_study(
            &[11, 21, 41],
            &|n| {
                let grid = Grid1D::new(0.0, 1.0, n);
                // Approximate sin(pi*x) with O(h^2) error by using the FD Laplacian
                let sol: Vec<f64> = (0..n).map(|i| (std::f64::consts::PI * grid.x(i)).sin() + 0.01 * grid.dx * grid.dx).collect();
                (grid, sol)
            },
            &|x| (std::f64::consts::PI * x).sin(),
        );
        assert_eq!(results.len(), 3);
        // Errors should decrease as grid refines
        assert!(results[0].1 > results[2].1);
    }

    #[test]
    fn test_l2_error_zero() {
        let u = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(ErrorAnalysis::l2_error(&u, &u, 1.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_convergence_order_third() {
        let errors = vec![0.001, 0.000125]; // ratio 8 = 2^3
        let order = ErrorAnalysis::convergence_order(&errors);
        assert_relative_eq!(order, 3.0, epsilon = 0.01);
    }
}
