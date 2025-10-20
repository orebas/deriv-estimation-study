"""
Test if TaylorDiff actually works
"""

using TaylorDiff
using ForwardDiff

println("=" ^ 70)
println("TESTING TAYLORDIFF vs FORWARDDIFF")
println("=" ^ 70)

# Test 1: Simple polynomial
println("\nTest 1: Polynomial f(x) = x^4 + 2x^3 - x^2 + 5x - 3")
f1(x) = x^4 + 2x^3 - x^2 + 5x - 3

# Analytical derivatives
f1_d1(x) = 4x^3 + 6x^2 - 2x + 5
f1_d2(x) = 12x^2 + 12x - 2
f1_d3(x) = 24x + 12
f1_d4(x) = 24.0

x_test = 2.0
println("At x = $x_test:")
println("  True: f'=$( f1_d1(x_test)), f''=$(f1_d2(x_test)), f'''=$(f1_d3(x_test)), f''''=$(f1_d4(x_test))")

# TaylorDiff
try
    td1 = TaylorDiff.derivative(f1, x_test, 1)
    td2 = TaylorDiff.derivative(f1, x_test, 2)
    td3 = TaylorDiff.derivative(f1, x_test, 3)
    td4 = TaylorDiff.derivative(f1, x_test, 4)
    println("  TaylorDiff: f'=$td1, f''=$td2, f'''=$td3, f''''=$td4")

    err1 = abs(td1 - f1_d1(x_test))
    err2 = abs(td2 - f1_d2(x_test))
    err3 = abs(td3 - f1_d3(x_test))
    err4 = abs(td4 - f1_d4(x_test))
    println("  Errors: $(err1), $(err2), $(err3), $(err4)")

    if err1 < 1e-10 && err2 < 1e-10 && err3 < 1e-10 && err4 < 1e-10
        println("  ✓ TaylorDiff works perfectly!")
    else
        println("  ✗ TaylorDiff has errors")
    end
catch e
    println("  ✗ TaylorDiff FAILED: $e")
end

# Test 2: Closure with branching (like our interpolators)
println("\n" * "=" ^ 70)
println("Test 2: Linear interpolator (has branching)")
x_data = [1.0, 2.0, 3.0]
y_data = [1.0, 4.0, 9.0]

function make_interpolator(x_pts, y_pts)
    return function(x)
        if x <= x_pts[1]
            return y_pts[1]
        elseif x >= x_pts[end]
            return y_pts[end]
        end
        
        idx = searchsortedfirst(x_pts, x)
        if idx > length(x_pts)
            return y_pts[end]
        elseif idx == 1
            return y_pts[1]
        end
        
        x0, x1 = x_pts[idx-1], x_pts[idx]
        y0, y1 = y_pts[idx-1], y_pts[idx]
        t = (x - x0) / (x1 - x0)
        return (1 - t) * y0 + t * y1
    end
end

f2 = make_interpolator(x_data, y_data)
x_test2 = 2.5

println("At x = $x_test2:")
println("  f($x_test2) = $(f2(x_test2))")

try
    td1 = TaylorDiff.derivative(f2, x_test2, 1)
    println("  TaylorDiff f' = $td1")
    if isnan(td1)
        println("  ✗ TaylorDiff returns NaN")
    else
        println("  ✓ TaylorDiff works")
    end
catch e
    println("  ✗ TaylorDiff FAILED: $e")
end

try
    fd1 = ForwardDiff.derivative(f2, x_test2)
    println("  ForwardDiff f' = $fd1")
    println("  ✓ ForwardDiff works")
catch e
    println("  ✗ ForwardDiff FAILED: $e")
end

println("\n" * "=" ^ 70)
println("CONCLUSION: TaylorDiff likely fails on functions with:")
println("  - Branching (if/else)")
println("  - Array indexing with computed indices")
println("  - Calls to external C libraries")
println("=" ^ 70)
