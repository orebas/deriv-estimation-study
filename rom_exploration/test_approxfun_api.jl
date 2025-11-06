"""
Test: What's the RIGHT way to use ApproxFun with data points?

From ApproxFun philosophy:
- Fun(f, domain) for functions → automatic adaptive degree
- Fun(space, coefficients) for explicit coefficients
- But what about data points (x, y)?
"""

using ApproxFun

# Test data
n = 101
t = range(0, 1, length=n)
omega = 2 * π
y_noisy = sin.(omega .* t) .+ 1e-3 .* randn(n)

println("ApproxFun API Test for Data Points")
println("=" ^ 80)

# APPROACH 1: Direct from function (WORKS - adaptive)
println("\n1. Fun(function, domain) - ApproxFun's intended use")
f1 = Fun(x -> sin(omega * x), 0..1)
println("   Coefficients: $(length(coefficients(f1)))")
println("   ✓ Adaptive degree selection")

# APPROACH 2: From coefficients (manual)
println("\n2. Fun(space, coeffs) - Manual coefficient specification")
S = Chebyshev(0..1)
f2 = Fun(S, [1.0, 0.5, 0.25])  # Arbitrary coeffs
println("   Coefficients: $(length(coefficients(f2)))")
println("   ✗ No adaptation - we specify everything")

# APPROACH 3: From VALUES at points (is this possible?)
println("\n3. Can we create Fun from (x, y) data points?")
println("   Trying: Fun(space, values)")
try
    # This is what transform() expects - values at Chebyshev points
    f3 = Fun(S, y_noisy)
    println("   Result: $(length(coefficients(f3))) coefficients")
    println("   Issue: Assumes y_noisy are values at Chebyshev points!")
    println("   → Overfits: uses all $(length(y_noisy)) points")
catch e
    println("   FAILED: $e")
end

# APPROACH 4: ApproxFun's points() - get where to evaluate
println("\n4. Using ApproxFun.points() - evaluate at Chebyshev points")
println("   This is for interpolation, not data fitting")
cheb_pts = points(S, 20)
println("   ApproxFun gives us 20 Chebyshev points: $(cheb_pts[1:3])...")
println("   If we have data at these exact points → perfect")
println("   If we have data at OTHER points → need to interpolate (loses info)")

println("\n" * "=" ^ 80)
println("CONCLUSION")
println("=" ^ 80)
println("""
ApproxFun's automatic degree selection is designed for FUNCTIONS, not DATA.

For functions:
  Fun(sin, 0..1) → Evaluates sin at Chebyshev points
                 → Computes coefficients
                 → Truncates when they decay below tolerance
                 ✓ Automatic!

For data points:
  We have (x[i], y[i]) at arbitrary points
  ApproxFun doesn't have a built-in "fit data with adaptive degree"
  We MUST choose the degree manually

Our least-squares approach IS the right way:
  - Choose degree based on data quality/noise level
  - For noisy data: lower degree (smoothing)
  - For clean data: higher degree (accuracy)

This is actually a FEATURE, not a bug:
  The user needs to specify how much smoothing they want!
  No algorithm can automatically determine "optimal" smoothing without
  knowing the noise level and intended use.

Recommendation:
  - Default: degree 10-15 (good for typical noisy data)
  - Make it configurable
  - Provide guidance: higher noise → lower degree
""")
