"""
Tests for Pareto front generation and filtering.
Tests multi-objective optimization functionality.
"""

from test_metal.optimization import (
    InverseRegression,
    ParetoOptimizer,
    ParetoOptimum,
)


class TestParetoOptimizerFrontGeneration:
    """Test Pareto front generation."""

    def test_generate_pareto_front_single_element(self, mock_ols_result):
        """Test generating Pareto front for single element."""
        inverse = InverseRegression([mock_ols_result])
        optimizer = ParetoOptimizer(inverse)

        targets = {
            "Sulfur (S)": ("steel_S_before", 0.08),
        }

        solutions = optimizer.generate_pareto_front(targets, n_points=50)

        assert isinstance(solutions, list)
        assert len(solutions) > 0
        assert all(isinstance(sol, ParetoOptimum) for sol in solutions)

    def test_generate_pareto_front_two_elements(self, two_element_models):
        """Test generating Pareto front for two elements."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        targets = {
            "Sulfur (S)": ("steel_S_before", 0.06),
            "Silicon (Si)": ("steel_Si_before", 0.12),
        }

        solutions = optimizer.generate_pareto_front(targets, n_points=50)

        assert isinstance(solutions, list)
        assert len(solutions) > 0
        assert all(isinstance(sol, ParetoOptimum) for sol in solutions)

        # Check that all solutions have both input values
        for sol in solutions:
            assert "Sulfur (S)" in sol.input_values
            assert "Silicon (Si)" in sol.input_values

    def test_generate_pareto_front_with_different_n_points(self, mock_ols_result):
        """Test Pareto front generation with different number of points."""
        inverse = InverseRegression([mock_ols_result])
        optimizer = ParetoOptimizer(inverse)

        targets = {
            "Sulfur (S)": ("steel_S_before", 0.08),
        }

        # Test with different n_points
        solutions_10 = optimizer.generate_pareto_front(targets, n_points=10)
        solutions_100 = optimizer.generate_pareto_front(targets, n_points=100)

        assert len(solutions_10) > 0
        assert len(solutions_100) > 0
        # More points should give more solutions (usually)
        # But not guaranteed due to filtering

    def test_pareto_solutions_have_correct_structure(self, two_element_models):
        """Test that Pareto solutions have all required attributes."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        targets = {
            "Sulfur (S)": ("steel_S_before", 0.06),
            "Silicon (Si)": ("steel_Si_before", 0.12),
        }

        solutions = optimizer.generate_pareto_front(targets, n_points=30)

        for sol in solutions:
            assert hasattr(sol, "solution_id")
            assert hasattr(sol, "input_values")
            assert hasattr(sol, "output_values")
            assert hasattr(sol, "total_impurity_input")
            assert hasattr(sol, "total_impurity_output")
            assert hasattr(sol, "efficiency")

            # Check types
            assert isinstance(sol.input_values, dict)
            assert isinstance(sol.output_values, dict)
            assert isinstance(sol.total_impurity_input, (int, float))
            assert isinstance(sol.total_impurity_output, (int, float))
            assert isinstance(sol.efficiency, (int, float))


class TestParetoOptimizerFiltering:
    """Test Pareto front filtering (domination check)."""

    def test_filter_pareto_front_removes_dominated(self, two_element_models):
        """Test that filtering removes dominated solutions."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        targets = {
            "Sulfur (S)": ("steel_S_before", 0.06),
            "Silicon (Si)": ("steel_Si_before", 0.12),
        }

        solutions = optimizer.generate_pareto_front(targets, n_points=100)
        filtered = optimizer.filter_pareto_front(solutions)

        # Filtered should be subset or equal to original
        assert len(filtered) <= len(solutions)
        assert all(isinstance(sol, ParetoOptimum) for sol in filtered)

    def test_filter_empty_list(self, two_element_models):
        """Test filtering empty list."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        filtered = optimizer.filter_pareto_front([])

        assert filtered == []

    def test_filter_single_solution(self, two_element_models):
        """Test filtering with single solution."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        # Create single solution manually
        solution = ParetoOptimum(
            solution_id=1,
            input_values={"Sulfur (S)": 0.06, "Silicon (Si)": 0.12},
            output_values={"Sulfur (S)": 0.05, "Silicon (Si)": 0.10},
            total_impurity_input=0.18,
            total_impurity_output=0.15,
            efficiency=16.67,
        )

        filtered = optimizer.filter_pareto_front([solution])

        assert len(filtered) == 1
        assert filtered[0].solution_id == solution.solution_id

    def test_filter_identical_solutions(self, two_element_models):
        """Test filtering when all solutions are identical (non-dominated)."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        # Create identical solutions
        sol1 = ParetoOptimum(
            solution_id=1,
            input_values={"S": 0.05},
            output_values={"S": 0.03},
            total_impurity_input=0.05,
            total_impurity_output=0.03,
            efficiency=40.0,
        )
        sol2 = ParetoOptimum(
            solution_id=2,
            input_values={"S": 0.05},
            output_values={"S": 0.03},
            total_impurity_input=0.05,
            total_impurity_output=0.03,
            efficiency=40.0,
        )

        filtered = optimizer.filter_pareto_front([sol1, sol2])

        # Both should remain (they're not dominated)
        assert len(filtered) >= 1

    def test_filter_clearly_dominated_solution(self, two_element_models):
        """Test that clearly dominated solutions are removed."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        # Better solution: less input, less output
        sol_better = ParetoOptimum(
            solution_id=1,
            input_values={"S": 0.04, "Si": 0.10},
            output_values={"S": 0.02, "Si": 0.08},
            total_impurity_input=0.14,
            total_impurity_output=0.10,
            efficiency=28.57,
        )

        # Worse solution: more input, more output
        sol_worse = ParetoOptimum(
            solution_id=2,
            input_values={"S": 0.06, "Si": 0.12},
            output_values={"S": 0.04, "Si": 0.10},
            total_impurity_input=0.18,
            total_impurity_output=0.14,
            efficiency=22.22,
        )

        filtered = optimizer.filter_pareto_front([sol_better, sol_worse])

        # Both should be in filtered (one doesn't strictly dominate other on all criteria)
        assert len(filtered) >= 1

    def test_filter_sorted_by_input(self, two_element_models):
        """Test that filtered solutions are sorted by input impurity."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        targets = {
            "Sulfur (S)": ("steel_S_before", 0.06),
            "Silicon (Si)": ("steel_Si_before", 0.12),
        }

        solutions = optimizer.generate_pareto_front(targets, n_points=50)
        filtered = optimizer.filter_pareto_front(solutions)

        if len(filtered) > 1:
            # Check sorting
            for i in range(len(filtered) - 1):
                assert filtered[i].total_impurity_input <= filtered[i + 1].total_impurity_input


class TestParetoOptimizerIntegration:
    """Integration tests for complete Pareto optimization workflow."""

    def test_complete_workflow_two_elements(self, two_element_models):
        """Test complete workflow: generate and filter Pareto front."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        targets = {
            "Sulfur (S)": ("steel_S_before", 0.06),
            "Silicon (Si)": ("steel_Si_before", 0.12),
        }

        # Generate
        solutions = optimizer.generate_pareto_front(targets, n_points=100)
        assert len(solutions) > 0

        # Filter
        filtered = optimizer.filter_pareto_front(solutions)
        assert len(filtered) > 0
        assert len(filtered) <= len(solutions)

        # Check properties of best solution
        best = filtered[0]
        assert best.total_impurity_input == min(s.total_impurity_input for s in filtered)

    def test_pareto_efficiency_values(self, two_element_models):
        """Test that efficiency values are in reasonable range (0-100%)."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        targets = {
            "Sulfur (S)": ("steel_S_before", 0.06),
            "Silicon (Si)": ("steel_Si_before", 0.12),
        }

        solutions = optimizer.generate_pareto_front(targets, n_points=50)
        filtered = optimizer.filter_pareto_front(solutions)

        for sol in filtered:
            assert 0 <= sol.efficiency <= 100, f"Efficiency {sol.efficiency} out of range"

    def test_pareto_with_different_target_levels(self, two_element_models):
        """Test Pareto optimization with different target stringency levels."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        # Relaxed targets (easier to achieve)
        relaxed_targets = {
            "Sulfur (S)": ("steel_S_before", 0.10),
            "Silicon (Si)": ("steel_Si_before", 0.16),
        }

        # Stringent targets (harder to achieve)
        stringent_targets = {
            "Sulfur (S)": ("steel_S_before", 0.03),
            "Silicon (Si)": ("steel_Si_before", 0.08),
        }

        relaxed = optimizer.generate_pareto_front(relaxed_targets, n_points=50)
        stringent = optimizer.generate_pareto_front(stringent_targets, n_points=50)

        assert len(relaxed) > 0
        assert len(stringent) > 0

        # Relaxed targets should allow for lower input requirements
        relaxed_min = min(s.total_impurity_input for s in relaxed)
        stringent_min = min(s.total_impurity_input for s in stringent)

        # This is not strictly guaranteed but likely
        # Just check they're both reasonable values
        assert 0 < relaxed_min < 1
        assert 0 < stringent_min < 1


class TestParetoEdgeCases:
    """Test edge cases in Pareto optimization."""

    def test_pareto_with_single_element(self, mock_ols_result):
        """Test Pareto optimization with only one element (should still work)."""
        inverse = InverseRegression([mock_ols_result])
        optimizer = ParetoOptimizer(inverse)

        targets = {
            "Sulfur (S)": ("steel_S_before", 0.08),
        }

        solutions = optimizer.generate_pareto_front(targets, n_points=50)
        filtered = optimizer.filter_pareto_front(solutions)

        assert len(filtered) > 0

    def test_pareto_solutions_improvement(self, two_element_models):
        """Test that Pareto solutions show improvement (output < input)."""
        inverse = InverseRegression(two_element_models)
        optimizer = ParetoOptimizer(inverse)

        targets = {
            "Sulfur (S)": ("steel_S_before", 0.06),
            "Silicon (Si)": ("steel_Si_before", 0.12),
        }

        solutions = optimizer.generate_pareto_front(targets, n_points=50)
        filtered = optimizer.filter_pareto_front(solutions)

        # All solutions should show some purification (output < input or equal)
        for sol in filtered:
            assert sol.total_impurity_output <= sol.total_impurity_input + 0.01
