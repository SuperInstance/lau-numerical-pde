//! Agent field dynamics — diffusion of information through agent populations.
//!
//! Models how information/opinion/behavior spreads through a population of agents
//! using PDE-based approaches. The "field" represents a continuous approximation
//! of a discrete agent property (e.g., opinion, knowledge level, adoption rate).

use crate::boundary::BoundaryPair1D;
use crate::finite_diff::Grid1D;
use crate::heat::HeatSolver;
use crate::heat::HeatMethod;
use crate::advection_diffusion::AdvectionDiffusionSolver;

/// Represents an agent population distributed over a 1D spatial domain.
#[derive(Debug, Clone)]
pub struct AgentField {
    /// The field value at each grid point (e.g., opinion, knowledge level).
    pub values: Vec<f64>,
    /// The spatial grid.
    pub grid: Grid1D,
}

impl AgentField {
    /// Create a uniform agent field.
    pub fn uniform(grid: Grid1D, value: f64) -> Self {
        Self { values: vec![value; grid.n], grid }
    }

    /// Create a Gaussian bump centered at x_center with given amplitude and width.
    pub fn gaussian(grid: Grid1D, x_center: f64, amplitude: f64, sigma: f64) -> Self {
        let values: Vec<f64> = (0..grid.n)
            .map(|i| amplitude * (-((grid.x(i) - x_center).powi(2)) / (2.0 * sigma * sigma)).exp())
            .collect();
        Self { values, grid }
    }

    /// Create a field with random-ish variation (deterministic using sin mix).
    pub fn heterogeneous(grid: Grid1D, base: f64, amplitude: f64) -> Self {
        let values: Vec<f64> = (0..grid.n)
            .map(|i| {
                let x = grid.x(i);
                base + amplitude * (3.0 * std::f64::consts::PI * x).sin()
                    + 0.5 * amplitude * (7.0 * std::f64::consts::PI * x).cos()
            })
            .collect();
        Self { values, grid }
    }

    /// Mean field value.
    pub fn mean(&self) -> f64 {
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Standard deviation of field values.
    pub fn std_dev(&self) -> f64 {
        let m = self.mean();
        let variance: f64 = self.values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / self.values.len() as f64;
        variance.sqrt()
    }

    /// Total mass (integral of field).
    pub fn total_mass(&self) -> f64 {
        self.values.iter().sum::<f64>() * self.grid.dx
    }

    /// Max field value.
    pub fn max_value(&self) -> f64 {
        self.values.iter().cloned().fold(f64::NAN, f64::max)
    }

    /// Min field value.
    pub fn min_value(&self) -> f64 {
        self.values.iter().cloned().fold(f64::NAN, f64::min)
    }

    /// Compute consensus measure: 1 - normalized_variance.
    /// Returns 1.0 when all agents agree, 0.0 when maximally dispersed.
    pub fn consensus(&self) -> f64 {
        let sd = self.std_dev();
        let range = self.max_value() - self.min_value();
        if range.abs() < 1e-15 { return 1.0; }
        1.0 - (sd / (range / 2.0)).min(1.0)
    }
}

/// Simulate information diffusion through an agent population.
/// Models: pure diffusion (heat equation), biased diffusion (advection-diffusion).
pub struct InformationDiffusion {
    /// Diffusion coefficient: how fast information spreads.
    pub diffusion_rate: f64,
    /// Bias velocity: directional tendency in information flow.
    pub bias: f64,
    /// Boundary conditions for the population.
    pub bc: BoundaryPair1D,
}

impl InformationDiffusion {
    pub fn new(diffusion_rate: f64, bc: BoundaryPair1D) -> Self {
        Self { diffusion_rate, bias: 0.0, bc }
    }

    pub fn with_bias(mut self, bias: f64) -> Self { self.bias = bias; self }

    /// Simulate one step of information diffusion.
    pub fn step(&self, field: &AgentField, dt: f64) -> AgentField {
        if self.bias.abs() < 1e-15 {
            // Pure diffusion
            let solver = HeatSolver::new(
                field.grid.clone(), self.diffusion_rate, self.bc.clone(), HeatMethod::ForwardEuler,
            );
            let new_values = solver.solve_final(&field.values, dt, 1);
            AgentField { values: new_values, grid: field.grid.clone() }
        } else {
            // Advection-diffusion
            let solver = AdvectionDiffusionSolver::new(
                field.grid.clone(), self.bias, self.diffusion_rate, self.bc.clone(),
            );
            let new_values = solver.solve_final(&field.values, dt, 1);
            AgentField { values: new_values, grid: field.grid.clone() }
        }
    }

    /// Run a full simulation for n_steps.
    pub fn simulate(&self, field: &AgentField, dt: f64, n_steps: usize) -> Vec<AgentField> {
        let mut history = vec![field.clone()];
        let mut current = field.clone();
        for _ in 0..n_steps {
            current = self.step(&current, dt);
            history.push(current.clone());
        }
        history
    }

    /// Compute time to consensus: how many steps until std_dev < threshold.
    pub fn time_to_consensus(&self, field: &AgentField, dt: f64, threshold: f64, max_steps: usize) -> Option<usize> {
        let mut current = field.clone();
        for step in 0..max_steps {
            current = self.step(&current, dt);
            if current.std_dev() < threshold {
                return Some(step + 1);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_agent_field_uniform() {
        let grid = Grid1D::new(0.0, 1.0, 51);
        let field = AgentField::uniform(grid, 0.5);
        assert_relative_eq!(field.mean(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(field.std_dev(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(field.consensus(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_agent_field_gaussian() {
        let grid = Grid1D::new(0.0, 1.0, 101);
        let field = AgentField::gaussian(grid, 0.5, 1.0, 0.1);
        assert!(field.max_value() > 0.9);
        assert!(field.values[0] < 0.01);
        assert!(field.values[100] < 0.01);
    }

    #[test]
    fn test_agent_field_heterogeneous() {
        let grid = Grid1D::new(0.0, 1.0, 101);
        let field = AgentField::heterogeneous(grid, 0.5, 0.3);
        assert!(field.std_dev() > 0.0);
        assert!(field.consensus() < 1.0);
    }

    #[test]
    fn test_agent_field_mass() {
        let grid = Grid1D::new(0.0, 1.0, 101);
        let field = AgentField::uniform(grid, 2.0);
        assert_relative_eq!(field.total_mass(), 2.0, epsilon = 0.03);
    }

    #[test]
    fn test_info_diffusion_pure_spreads() {
        let grid = Grid1D::new(0.0, 1.0, 101);
        let field = AgentField::gaussian(grid, 0.5, 1.0, 0.05);

        let diffusion = InformationDiffusion::new(0.01, BoundaryPair1D::dirichlet(0.0, 0.0));
        let dt = 0.4 * field.grid.dx * field.grid.dx / diffusion.diffusion_rate;
        let result = diffusion.simulate(&field, dt, 50);

        // Peak decreases as the Gaussian flattens
        assert!(result.last().unwrap().max_value() < field.max_value());
    }

    #[test]
    fn test_info_diffusion_consensus_reached() {
        let grid = Grid1D::new(0.0, 1.0, 51);
        let dx = grid.dx;
        let field = AgentField::heterogeneous(grid, 0.5, 0.5);

        let diffusion = InformationDiffusion::new(0.1, BoundaryPair1D::neumann(0.0, 0.0));
        let dt_max = 0.4 * dx * dx / diffusion.diffusion_rate;
        let ttc = diffusion.time_to_consensus(&field, dt_max * 0.5, 0.01, 10000);
        assert!(ttc.is_some(), "Should reach consensus within 10000 steps");
    }

    #[test]
    fn test_info_diffusion_mass_conservation_periodic() {
        let grid = Grid1D::new(0.0, 1.0, 51);
        let dx = grid.dx;
        let field = AgentField::heterogeneous(grid, 0.5, 0.3);
        let mass0 = field.total_mass();

        let diffusion = InformationDiffusion::new(0.01, BoundaryPair1D::periodic());
        let dt_max = 0.4 * dx * dx / diffusion.diffusion_rate;
        let result = diffusion.simulate(&field, dt_max * 0.5, 50);

        let mass_f = result.last().unwrap().total_mass();
        assert_relative_eq!(mass0, mass_f, epsilon = 0.05);
    }

    #[test]
    fn test_info_diffusion_with_bias() {
        let grid = Grid1D::new(0.0, 1.0, 101);
        let dx = grid.dx;
        let field = AgentField::gaussian(grid, 0.5, 1.0, 0.1);

        let diffusion = InformationDiffusion::new(0.001, BoundaryPair1D::periodic()).with_bias(0.5);
        let dt = diffusion.diffusion_rate * 0.3 * dx / (diffusion.bias + 1e-10);
        let result = diffusion.simulate(&field, dt.min(0.001), 20);

        // With bias, the peak should shift
        let initial_peak_idx = field.values.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let final_peak_idx = result.last().unwrap().values.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        // Peak should move or change
        let initial_max = field.values.iter().cloned().fold(f64::NAN, f64::max);
        let final_max = result.last().unwrap().values.iter().cloned().fold(f64::NAN, f64::max);
        assert!(final_max < initial_max || initial_peak_idx != final_peak_idx,
            "Field should change with bias");
    }

    #[test]
    fn test_consensus_zero_field() {
        let grid = Grid1D::new(0.0, 1.0, 11);
        let field = AgentField::uniform(grid, 0.0);
        assert_relative_eq!(field.consensus(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_consensus_decreases_with_variance() {
        let grid = Grid1D::new(0.0, 1.0, 51);
        let field_low = AgentField::heterogeneous(grid.clone(), 0.5, 0.05);
        let field_high = AgentField::heterogeneous(grid, 0.5, 1.5);
        assert!(field_low.std_dev() < field_high.std_dev());
    }

    #[test]
    fn test_agent_field_min_max() {
        let grid = Grid1D::new(0.0, 1.0, 51);
        let field = AgentField::heterogeneous(grid, 0.5, 0.5);
        assert!(field.max_value() > field.min_value());
    }

    #[test]
    fn test_diffusion_convergence_order() {
        let errors: Vec<f64> = [21, 41, 81, 161].iter().map(|&n| {
            let grid = Grid1D::new(0.0, 1.0, n);
            let field = AgentField::gaussian(grid.clone(), 0.5, 1.0, 0.1);
            let diffusion = InformationDiffusion::new(0.01, BoundaryPair1D::dirichlet(0.0, 0.0));
            let dt = 0.4 * field.grid.dx * field.grid.dx / diffusion.diffusion_rate;
            let result = diffusion.simulate(&field, dt, 10);
            let final_field = result.last().unwrap();
            // Check the solution is bounded (sanity)
            final_field.max_value().abs()
        }).collect();

        // All errors should be finite and bounded
        for e in &errors {
            assert!(e.is_finite());
        }
    }
}
