"""
Ground Truth Generation for High-Order Derivative Study

Generates symbolic derivatives and ground truth data from ODE systems
"""

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Symbolics

"""
    lotka_volterra_system()

Define the Lotka-Volterra predator-prey system with symbolic derivatives.

Returns a NamedTuple with:
- system: The ODE system
- params: Parameter values
- ic: Initial conditions
- tspan: Time span
- obs: Observable variables
- name: System name
"""
function lotka_volterra_system()
    @variables x(t) y(t)
    @parameters α β γ δ

    # Equations: dx/dt = α*x - β*x*y,  dy/dt = δ*x*y - γ*y
    eqs = [
        D(x) ~ α * x - β * x * y,
        D(y) ~ δ * x * y - γ * y
    ]

    @named sys = ODESystem(eqs, t)

    return (
        system = structural_simplify(sys),
        params = [α => 1.5, β => 1.0, γ => 3.0, δ => 1.0],
        ic = [x => 1.0, y => 1.0],
        tspan = (0.0, 10.0),
        obs = [x, y],
        name = "Lotka-Volterra"
    )
end


"""
    calculate_symbolic_derivatives(sys_def, max_order)

Calculate symbolic expressions for derivatives up to max_order.

Returns a Vector{Vector{Num}} where derivs[order][obs_idx] gives the
order-th derivative of the obs_idx-th observable.
"""
function calculate_symbolic_derivatives(sys_def, max_order)
    sys = sys_def.system
    obs = sys_def.obs

    # Create dictionary from equations (lhs => rhs)
    eq_dict = Dict(eq.lhs => eq.rhs for eq in equations(sys))

    # derivs[order] = Vector of derivatives for each observable
    derivs = Vector{Vector{Num}}(undef, max_order)

    # First derivative
    derivs[1] = [substitute(expand_derivatives(D(o)), eq_dict) for o in obs]

    # Higher derivatives (recursive substitution)
    for order in 2:max_order
        derivs[order] = [
            substitute(expand_derivatives(D(derivs[order-1][i])), eq_dict)
            for i in 1:length(obs)
        ]
    end

    return derivs
end


"""
    generate_ground_truth(sys_def, times, max_order; solver_tols=(1e-12, 1e-12))

Generate ground truth data by solving an augmented ODE system.

Arguments:
- sys_def: System definition from lotka_volterra_system()
- times: Vector of time points to evaluate
- max_order: Maximum derivative order to compute
- solver_tols: (abstol, reltol) for ODE solver

Returns a Dict with:
- :t => time points
- :obs => Dict mapping obs_idx to Dict(order => values)
"""
function generate_ground_truth(sys_def, times, max_order; solver_tols=(1e-12, 1e-12))
    println("  Computing symbolic derivatives up to order $max_order...")
    derivs = calculate_symbolic_derivatives(sys_def, max_order)

    # Create observed equations for all derivatives
    obs_eqs = []
    d_vars = []  # Store (obs_idx, order, variable) tuples

    for (i, o) in enumerate(sys_def.obs)
        for order in 1:max_order
            # Create variable name: d1_o1, d2_o1, etc.
            var_name = Symbol("d$(order)_o$i")
            v = only(@variables $var_name(t))
            push!(d_vars, (i, order, v))
            push!(obs_eqs, v ~ derivs[order][i])
        end
    end

    # Create augmented system with observed derivatives
    println("  Building augmented ODE system...")
    @named ext_sys = ODESystem(
        equations(sys_def.system),
        t,
        observed=obs_eqs
    )

    # Solve with high precision
    println("  Solving ODE system...")
    prob = ODEProblem(
        structural_simplify(ext_sys),
        sys_def.ic,
        sys_def.tspan,
        sys_def.params
    )

    abstol, reltol = solver_tols
    sol = solve(
        prob,
        AutoVern9(Rodas4P()),
        abstol=abstol,
        reltol=reltol,
        saveat=times
    )

    # Extract data
    println("  Extracting solution data...")
    data = Dict(:t => sol.t, :obs => Dict())

    for (i, o) in enumerate(sys_def.obs)
        # Order 0 = the observable itself
        data[:obs][i] = Dict(0 => sol[o])

        # Orders 1 through max_order
        for order in 1:max_order
            # Find the corresponding derivative variable
            d_var_tuple = d_vars[findfirst(x -> x[1]==i && x[2]==order, d_vars)]
            dv = d_var_tuple[3]
            data[:obs][i][order] = sol[dv]
        end
    end

    return data
end


"""
    print_sample_derivatives(data, obs_idx=1, sample_idx=1)

Print a sample of derivative values for inspection.
"""
function print_sample_derivatives(data, obs_idx=1, sample_idx=1)
    println("\nSample derivative values at t=$(data[:t][sample_idx]):")
    println("Observable $obs_idx:")

    max_order = maximum(keys(data[:obs][obs_idx]))
    for order in 0:max_order
        val = data[:obs][obs_idx][order][sample_idx]
        println("  d^$order/dt^$order: $(round(val, digits=6))")
    end
end


# Test function
function test_ground_truth()
    println("=" ^ 70)
    println("TESTING GROUND TRUTH GENERATION")
    println("=" ^ 70)

    sys = lotka_volterra_system()
    times = range(sys.tspan[1], sys.tspan[2], length=51)

    data = generate_ground_truth(sys, times, 7)

    print_sample_derivatives(data, 1, 1)
    print_sample_derivatives(data, 1, 26)  # Middle point

    println("\n✓ Ground truth generation successful!")
    return data
end

