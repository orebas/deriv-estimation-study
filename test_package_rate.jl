using SavitzkyGolay
using Statistics

n = 101
x = collect(range(0, 1, length=n))
dx = mean(diff(x))
y = sin.(2π * x)

println("Testing SavitzkyGolay.jl rate parameter:")
println("dx = $dx")
println()

r_dx = savitzky_golay(y, 15, 7, deriv=1, rate=dx)
r_1 = savitzky_golay(y, 15, 7, deriv=1, rate=1.0)
r_inv = savitzky_golay(y, 15, 7, deriv=1, rate=1/dx)

true_deriv = 2π * cos(2π * x[50])

println("At x=$(x[50]):")
println("  True 1st deriv: $true_deriv")
println("  Package rate=dx:   $(r_dx.y[50])")
println("  Package rate=1.0:  $(r_1.y[50])")
println("  Package rate=1/dx: $(r_inv.y[50])")
println()
println("Which matches true? Ratio to true:")
println("  rate=dx:   $(r_dx.y[50] / true_deriv)")
println("  rate=1.0:  $(r_1.y[50] / true_deriv)")
println("  rate=1/dx: $(r_inv.y[50] / true_deriv)")
