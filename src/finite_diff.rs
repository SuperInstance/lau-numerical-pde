//! Finite difference grids and operators.

use serde::{Deserialize, Serialize};

/// 1D uniform grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid1D {
    pub n: usize,
    pub dx: f64,
    pub x_min: f64,
    pub x_max: f64,
}

impl Grid1D {
    pub fn new(x_min: f64, x_max: f64, n: usize) -> Self {
        assert!(n >= 2, "Grid must have at least 2 points");
        let dx = (x_max - x_min) / (n - 1) as f64;
        Self { n, dx, x_min, x_max }
    }

    /// Interior points (excludes boundaries).
    pub fn interior_n(&self) -> usize {
        self.n.saturating_sub(2)
    }

    pub fn x(&self, i: usize) -> f64 {
        self.x_min + i as f64 * self.dx
    }

    pub fn interior_points(&self) -> Vec<f64> {
        (1..self.n - 1).map(|i| self.x(i)).collect()
    }

    pub fn all_points(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.x(i)).collect()
    }
}

/// 2D uniform grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid2D {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl Grid2D {
    pub fn new(x_min: f64, x_max: f64, nx: usize, y_min: f64, y_max: f64, ny: usize) -> Self {
        assert!(nx >= 2 && ny >= 2);
        let dx = (x_max - x_min) / (nx - 1) as f64;
        let dy = (y_max - y_min) / (ny - 1) as f64;
        Self { nx, ny, dx, dy, x_min, x_max, y_min, y_max }
    }

    pub fn interior_nx(&self) -> usize { self.nx.saturating_sub(2) }
    pub fn interior_ny(&self) -> usize { self.ny.saturating_sub(2) }

    pub fn x(&self, i: usize) -> f64 { self.x_min + i as f64 * self.dx }
    pub fn y(&self, j: usize) -> f64 { self.y_min + j as f64 * self.dy }
}

/// Finite difference operators on grids.
pub struct FiniteDiff;

impl FiniteDiff {
    /// First derivative using central difference: O(h^2).
    pub fn d1_central(u: &[f64], dx: f64) -> Vec<f64> {
        let n = u.len();
        assert!(n >= 3);
        let mut du = vec![0.0; n];
        for i in 1..n - 1 {
            du[i] = (u[i + 1] - u[i - 1]) / (2.0 * dx);
        }
        // Forward/backward at boundaries
        du[0] = (u[1] - u[0]) / dx;
        du[n - 1] = (u[n - 1] - u[n - 2]) / dx;
        du
    }

    /// First derivative using forward difference: O(h).
    pub fn d1_forward(u: &[f64], dx: f64) -> Vec<f64> {
        let n = u.len();
        assert!(n >= 2);
        let mut du = vec![0.0; n];
        for i in 0..n - 1 {
            du[i] = (u[i + 1] - u[i]) / dx;
        }
        du[n - 1] = du[n - 2];
        du
    }

    /// First derivative using backward difference: O(h).
    pub fn d1_backward(u: &[f64], dx: f64) -> Vec<f64> {
        let n = u.len();
        assert!(n >= 2);
        let mut du = vec![0.0; n];
        for i in 1..n {
            du[i] = (u[i] - u[i - 1]) / dx;
        }
        du[0] = du[1];
        du
    }

    /// Second derivative using central difference: O(h^2).
    pub fn d2_central(u: &[f64], dx: f64) -> Vec<f64> {
        let n = u.len();
        assert!(n >= 3);
        let mut d2u = vec![0.0; n];
        for i in 1..n - 1 {
            d2u[i] = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx);
        }
        d2u
    }

    /// Laplacian on a 2D grid: O(h^2) central differences.
    pub fn laplacian_2d(u: &[Vec<f64>], dx: f64, dy: f64) -> Vec<Vec<f64>> {
        let ny = u.len();
        let nx = u[0].len();
        let mut lap = vec![vec![0.0; nx]; ny];
        for i in 1..ny - 1 {
            for j in 1..nx - 1 {
                let d2x = (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) / (dx * dx);
                let d2y = (u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j]) / (dy * dy);
                lap[i][j] = d2x + d2y;
            }
        }
        lap
    }

    /// Fourth-order central second derivative: O(h^4).
    pub fn d2_central_4th(u: &[f64], dx: f64) -> Vec<f64> {
        let n = u.len();
        assert!(n >= 5);
        let mut d2u = vec![0.0; n];
        let dx2 = dx * dx;
        for i in 2..n - 2 {
            d2u[i] = (-u[i + 2] + 16.0 * u[i + 1] - 30.0 * u[i] + 16.0 * u[i - 1] - u[i - 2])
                / (12.0 * dx2);
        }
        d2u
    }

    /// Truncation error estimate for second-order central d2.
    pub fn truncation_error_d2(u_exact_fn: &dyn Fn(f64) -> f64, x: &[f64], dx: f64) -> f64 {
        let n = x.len();
        if n < 3 { return f64::NAN; }
        let mut max_err = 0.0f64;
        for i in 1..n - 1 {
            let _numerical = (u_exact_fn(x[i] + dx) - 2.0 * u_exact_fn(x[i]) + u_exact_fn(x[i] - dx)) / (dx * dx);
            // Approximate 4th derivative for truncation
            if i >= 1 && i < n - 1 {
                let h = dx;
                let fourth = (u_exact_fn(x[i] + 2.0*h) - 4.0*u_exact_fn(x[i]+h) + 6.0*u_exact_fn(x[i])
                    - 4.0*u_exact_fn(x[i]-h) + u_exact_fn(x[i]-2.0*h)) / (h*h*h*h);
                let trunc = (dx * dx / 12.0 * fourth).abs();
                max_err = max_err.max(trunc);
            }
        }
        max_err
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_grid1d_construction() {
        let g = Grid1D::new(0.0, 1.0, 11);
        assert_eq!(g.n, 11);
        assert_relative_eq!(g.dx, 0.1);
        assert_eq!(g.interior_n(), 9);
        assert_relative_eq!(g.x(0), 0.0);
        assert_relative_eq!(g.x(5), 0.5);
        assert_relative_eq!(g.x(10), 1.0);
    }

    #[test]
    fn test_grid2d_construction() {
        let g = Grid2D::new(0.0, 1.0, 11, 0.0, 2.0, 21);
        assert_eq!(g.nx, 11);
        assert_eq!(g.ny, 21);
        assert_relative_eq!(g.dx, 0.1);
        assert_relative_eq!(g.dy, 0.1);
    }

    #[test]
    fn test_d1_central_linear() {
        let g = Grid1D::new(0.0, 1.0, 51);
        let u: Vec<f64> = (0..g.n).map(|i| 3.0 * g.x(i) + 2.0).collect();
        let du = FiniteDiff::d1_central(&u, g.dx);
        for i in 1..g.n - 1 {
            assert_relative_eq!(du[i], 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_d1_forward_quadratic() {
        let g = Grid1D::new(0.0, 1.0, 101);
        let u: Vec<f64> = (0..g.n).map(|i| g.x(i).powi(2)).collect();
        let du = FiniteDiff::d1_forward(&u, g.dx);
        // At x=0, forward gives (h^2 - 0)/h = h, exact is 0
        assert_relative_eq!(du[0], g.dx, epsilon = 1e-12);
    }

    #[test]
    fn test_d2_central_quadratic() {
        let g = Grid1D::new(0.0, 1.0, 51);
        let u: Vec<f64> = (0..g.n).map(|i| g.x(i).powi(2)).collect();
        let d2u = FiniteDiff::d2_central(&u, g.dx);
        for i in 1..g.n - 1 {
            assert_relative_eq!(d2u[i], 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_d2_central_4th_order() {
        let g = Grid1D::new(0.0, 1.0, 101);
        let u: Vec<f64> = (0..g.n).map(|i| (g.x(i)).powi(4)).collect();
        let d2u = FiniteDiff::d2_central_4th(&u, g.dx);
        // d2/dx2(x^4) = 12x^2
        for i in 2..g.n - 2 {
            let expected = 12.0 * g.x(i).powi(2);
            assert_relative_eq!(d2u[i], expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_laplacian_2d() {
        let g = Grid2D::new(0.0, 1.0, 21, 0.0, 1.0, 21);
        let mut u = vec![vec![0.0; g.nx]; g.ny];
        for i in 0..g.ny {
            for j in 0..g.nx {
                let x = g.x(j);
                let y = g.y(i);
                u[i][j] = x.powi(2) + y.powi(2);
            }
        }
        let lap = FiniteDiff::laplacian_2d(&u, g.dx, g.dy);
        for i in 1..g.ny - 1 {
            for j in 1..g.nx - 1 {
                assert_relative_eq!(lap[i][j], 4.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_d1_backward() {
        let g = Grid1D::new(0.0, 1.0, 51);
        let u: Vec<f64> = (0..g.n).map(|i| 5.0 * g.x(i) - 1.0).collect();
        let du = FiniteDiff::d1_backward(&u, g.dx);
        for i in 2..g.n {
            assert_relative_eq!(du[i], 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_grid1d_interior_points() {
        let g = Grid1D::new(0.0, 1.0, 5);
        let pts = g.interior_points();
        assert_eq!(pts.len(), 3);
        assert_relative_eq!(pts[0], 0.25);
        assert_relative_eq!(pts[1], 0.5);
        assert_relative_eq!(pts[2], 0.75);
    }
}
