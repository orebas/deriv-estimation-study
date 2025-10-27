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
    van_der_pol_system()

Define the Van der Pol oscillator with symbolic derivatives.
This is a classic nonlinear oscillator with relaxation dynamics.

The second-order equation ẍ - μ(1-x²)ẋ + x = 0 is rewritten as:
    dx/dt = v
    dv/dt = μ(1-x²)v - x

With μ=3, this produces strong relaxation oscillations with fast/slow dynamics.

Returns a NamedTuple with:
- system: The ODE system
- params: Parameter values
- ic: Initial conditions
- tspan: Time span
- obs: Observable variables
- name: System name
"""
function van_der_pol_system()
    @variables x(t) v(t)
    @parameters μ

    # Van der Pol equations: dx/dt = v,  dv/dt = μ(1-x²)v - x
    eqs = [
        D(x) ~ v,
        D(v) ~ μ * (1 - x^2) * v - x
    ]

    @named sys = ODESystem(eqs, t)

    return (
        system = structural_simplify(sys),
        params = [μ => 3.0],  # μ=3 gives nice relaxation oscillations
        ic = [x => 2.0, v => 0.0],  # Start away from origin for interesting trajectory
        tspan = (0.0, 20.0),  # Longer time to capture several cycles
        obs = [x],  # Observe x position (has sharp peaks/troughs)
        name = "Van-der-Pol"
    )
end


"""
    lorenz_system()

Define the Lorenz system with symbolic derivatives.
This is a classic chaotic system exhibiting sensitive dependence on initial conditions.

The equations are:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

With standard parameters σ=10, ρ=28, β=8/3, the system exhibits chaotic behavior
on the famous "butterfly" attractor.

Returns a NamedTuple with:
- system: The ODE system
- params: Parameter values
- ic: Initial conditions
- tspan: Time span
- obs: Observable variables
- name: System name
"""
function lorenz_system()
    @variables x(t) y(t) z(t)
    @parameters σ ρ β

    # Lorenz equations
    eqs = [
        D(x) ~ σ * (y - x),
        D(y) ~ x * (ρ - z) - y,
        D(z) ~ x * y - β * z
    ]

    @named sys = ODESystem(eqs, t)

    return (
        system = structural_simplify(sys),
        params = [σ => 10.0, ρ => 25.0, β => 8.0/3.0],  # Mildly chaotic parameters (ρ slightly above critical 24.74)
        ic = [x => 1.0, y => 1.0, z => 1.0],  # Generic initial condition
        tspan = (0.0, 5.0),  # Shorter time span for gentler dynamics
        obs = [x],  # Observe x-coordinate (oscillates between wings)
        name = "Lorenz"
    )
end


"""
    get_all_ode_systems()

Get dictionary of all available ODE systems.
Keys are system identifiers (e.g., "lotka_volterra"), values are system definitions.

To filter based on config, use get_all_ode_systems(enabled_keys).

Returns a Dict{String, NamedTuple} where each value has the structure returned by
lotka_volterra_system(), van_der_pol_system(), etc.
"""
function get_all_ode_systems(enabled_keys::Union{Vector{String}, Nothing}=nothing)
    all_systems = Dict(
        "lotka_volterra" => lotka_volterra_system(),
        "van_der_pol" => van_der_pol_system(),
        "lorenz" => lorenz_system()
    )

    # If specific systems are requested, filter
    if enabled_keys !== nothing
        return Dict(k => all_systems[k] for k in enabled_keys if haskey(all_systems, k))
    else
        return all_systems
    end
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

