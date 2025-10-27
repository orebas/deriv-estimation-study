"""
Configuration loader for derivative estimation study.
Provides single source of truth from config.toml file.
"""

using TOML

"""
    load_config()

Load configuration from config.toml in the project root directory.
Returns a Dict with all configuration sections.
"""
function load_config()
    # Find the project root (where config.toml lives)
    project_root = dirname(dirname(@__FILE__))
    config_path = joinpath(project_root, "config.toml")

    if !isfile(config_path)
        error("Configuration file not found: $config_path")
    end

    config = TOML.parsefile(config_path)
    return config
end

"""
    get_comprehensive_config()

Get configuration for comprehensive study.
Returns named tuple with: noise_levels, data_size, max_derivative_order, trials_per_config
"""
function get_comprehensive_config()
    config = load_config()
    comp = config["comprehensive_study"]

    return (
        noise_levels = Float64.(comp["noise_levels"]),
        data_size = Int(comp["data_size"]),
        max_derivative_order = Int(comp["max_derivative_order"]),
        trials_per_config = Int(comp["trials_per_config"])
    )
end

"""
    get_enabled_ode_systems()

Get list of enabled ODE systems from configuration.
Returns Vector{String} of ODE system keys (e.g., ["lotka_volterra", "van_der_pol", "lorenz"])
"""
function get_enabled_ode_systems()
    config = load_config()
    ode_systems = config["ode_systems"]
    return String.(ode_systems["enabled_systems"])
end

"""
    get_pilot_config()

Get configuration for pilot study.
Returns named tuple with: trials, noise_levels, data_sizes, max_derivative_order
"""
function get_pilot_config()
    config = load_config()
    pilot = config["pilot_study"]

    return (
        trials = Int(pilot["trials"]),
        noise_levels = Float64.(pilot["noise_levels"]),
        data_sizes = Int.(pilot["data_sizes"]),
        max_derivative_order = Int(pilot["max_derivative_order"])
    )
end

"""
    get_paths_config()

Get path configuration.
Returns named tuple with: build_dir, results_dir, figures_dir, tables_dir, data_dir
"""
function get_paths_config()
    config = load_config()
    paths = config["paths"]

    return (
        build_dir = paths["build_dir"],
        results_dir = paths["results_dir"],
        figures_dir = paths["figures_dir"],
        tables_dir = paths["tables_dir"],
        data_dir = paths["data_dir"]
    )
end

"""
    print_config_summary(section::Symbol)

Print a summary of configuration for a given section (:comprehensive, :pilot, or :all).
"""
function print_config_summary(section::Symbol=:all)
    if section == :comprehensive || section == :all
        comp = get_comprehensive_config()
        println("=" ^ 80)
        println("COMPREHENSIVE STUDY CONFIGURATION (from config.toml)")
        println("=" ^ 80)
        println("Data size: $(comp.data_size)")
        println("Max derivative order: $(comp.max_derivative_order)")
        println("Trials per configuration: $(comp.trials_per_config)")
        println("Noise levels: $(comp.noise_levels)")
        println()
    end

    if section == :pilot || section == :all
        pilot = get_pilot_config()
        println("=" ^ 80)
        println("PILOT STUDY CONFIGURATION (from config.toml)")
        println("=" ^ 80)
        println("Trials: $(pilot.trials)")
        println("Noise levels: $(pilot.noise_levels)")
        println("Data sizes: $(pilot.data_sizes)")
        println("Max derivative order: $(pilot.max_derivative_order)")
        println()
    end
end
