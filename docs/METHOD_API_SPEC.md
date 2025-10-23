# Method API Specification

## Overview

All derivative estimation methods must conform to these standard APIs to ensure:
- Consistent interface across all implementations
- Easy testing and validation
- Straightforward integration into benchmark pipeline

## Python API

### Category Class Requirements

Each category module must define a class inheriting from `MethodEvaluator`:

```python
from methods.python.common import MethodEvaluator
from typing import Dict

class CategoryMethods(MethodEvaluator):
    """
    Category-specific derivative estimation methods.

    Inherits from MethodEvaluator which provides:
    - self.x_train: np.ndarray - training input points
    - self.y_train: np.ndarray - training output values
    - self.x_eval: np.ndarray - evaluation points
    - self.orders: List[int] - derivative orders to compute
    """

    def evaluate_method(self, method_name: str) -> Dict:
        """
        Dispatch to appropriate method based on method_name.

        Args:
            method_name: String identifier for method

        Returns:
            Dictionary with standard format (see below)
        """
        # Dispatcher logic here
        pass

    def _method_implementation(self) -> Dict:
        """Individual method implementation."""
        # Method logic here
        pass
```

### Return Format (Standard)

All methods must return a dictionary with this structure:

```python
{
    "predictions": {
        0: [float, ...],  # 0th derivative (function values)
        1: [float, ...],  # 1st derivative
        2: [float, ...],  # 2nd derivative
        # ... up to max order
    },
    "failures": {
        # Optional: errors encountered
        order: "error message"
    },
    "meta": {
        # Optional: method-specific metadata
        "hyperparameter_name": value
    }
}
```

**Required fields:**
- `predictions`: Dict[int, List[float]] - One list per derivative order
- Each prediction list must have length == len(self.x_eval)
- Use `np.nan` for failed predictions at specific points

**Optional fields:**
- `failures`: Dict[int, str] - Errors by derivative order
- `meta`: Dict[str, Any] - Method-specific metadata (hyperparameters, diagnostics)

### Input Validation

Handled by base `MethodEvaluator.__init__`:
- Checks for NaN/inf in x_train, y_train, x_eval
- Raises ValueError if invalid

### Error Handling

Methods should:
1. Wrap each derivative order in try/except
2. On error: set predictions[order] = [np.nan] * len(x_eval)
3. Record error in failures[order]
4. Continue with remaining orders

## Julia API

### Function Signature (Standard)

```julia
function method_name(
    x_train::Vector{Float64},
    y_train::Vector{Float64},
    x_eval::Vector{Float64},
    orders::Vector{Int};
    kwargs...
) -> Dict{String, Any}
```

### Return Format (Standard)

```julia
Dict(
    "predictions" => Dict(
        0 => Vector{Float64},  # 0th derivative
        1 => Vector{Float64},  # 1st derivative
        # ...
    ),
    "failures" => Dict(
        # Optional: order => error_message
    ),
    "meta" => Dict(
        # Optional: method metadata
        "hyperparameter_name" => value
    )
)
```

### Category Module Structure

Each category file should:
1. Define individual method functions following standard signature
2. Export all method functions
3. Provide a dispatcher function (optional but recommended)

```julia
module CategoryMethods

export method1, method2, method_dispatcher

function method1(x_train, y_train, x_eval, orders; kwargs...)
    # Implementation
    return Dict("predictions" => ..., "failures" => ..., "meta" => ...)
end

function method_dispatcher(method_name::String, x_train, y_train, x_eval, orders; kwargs...)
    if method_name == "method1"
        return method1(x_train, y_train, x_eval, orders; kwargs...)
    # ...
    else
        error("Unknown method: $method_name")
    end
end

end  # module
```

## Testing Requirements

### Validation Checklist

For each extracted method:

1. **Import Check**: Module imports without errors
2. **Instantiation Check**: Can create instance with test data
3. **Method Call Check**: Can call evaluate_method(name)
4. **Output Format Check**: Returns dict with required fields
5. **Output Length Check**: Predictions match x_eval length
6. **Behavior Check**: Results match original implementation

### Test Data

Standard test case:
```python
x_train = np.linspace(0, 1, 50)
y_train = np.sin(2 * np.pi * x_train) + 0.01 * np.random.randn(50)
x_eval = np.linspace(0, 1, 20)
orders = [0, 1, 2]
```

### Acceptance Criteria

- All predictions lists have length == len(x_eval)
- No crashes on standard inputs
- Handles edge cases gracefully (returns NaN, not crashes)
- Results numerically match original implementation (rtol=1e-10)

## File Organization

```
methods/
├── python/
│   ├── common.py                 # Base MethodEvaluator
│   ├── gp/
│   │   └── gaussian_process.py   # GPMethods class
│   ├── splines/
│   │   └── splines.py            # SplineMethods class
│   ├── spectral/
│   │   └── spectral.py           # SpectralMethods class
│   ├── adaptive/
│   │   └── adaptive.py           # AdaptiveMethods class
│   └── filtering/
│       └── filters.py            # FilteringMethods class
└── julia/
    ├── common.jl                 # Shared utilities
    ├── gp/
    │   └── gaussian_process.jl
    ├── splines/
    │   └── splines.jl
    └── ...
```

## Naming Conventions

### Python
- Class names: `CategoryMethods` (e.g., `GPMethods`, `SplineMethods`)
- Method names: Private with underscore (e.g., `_gp_rbf_mean_derivative`)
- File names: Lowercase with underscores (e.g., `gaussian_process.py`)

### Julia
- Function names: Lowercase with underscores (e.g., `fit_gp_rbf`)
- Module names: CamelCase (e.g., `GaussianProcessMethods`)
- File names: Lowercase with underscores (e.g., `gaussian_process.jl`)

## Migration Notes

When extracting from original files:
1. Keep exact same logic - no refactoring yet
2. Preserve all comments and docstrings
3. Update import paths to use new structure
4. Add module-level docstring explaining category
5. Test against original before committing
