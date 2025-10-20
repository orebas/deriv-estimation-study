# Quick test run with reduced parameters
include("derivative_estimation_study.jl")

# Override the constants for quick testing
const MONTE_CARLO_TRIALS = 5  # Reduced from 100
const NOISE_LEVELS = [0.01, 0.1]  # Just 2 levels
const DATA_SIZES = [21, 51]  # Just 2 sizes

println("\n" * "=" ^ 80)
println("RUNNING TEST WITH REDUCED PARAMETERS")
println("  Monte Carlo trials: $MONTE_CARLO_TRIALS")
println("  Noise levels: $NOISE_LEVELS")
println("  Data sizes: $DATA_SIZES")
println("=" ^ 80 * "\n")
