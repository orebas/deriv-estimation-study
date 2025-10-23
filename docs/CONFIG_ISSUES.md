# Configuration Inconsistencies Found

## Problem Summary

Multiple sources of truth for configuration parameters cause confusion:
- Bash scripts set environment variables that Julia programs ignore
- Each Julia program has its own hardcoded constants
- No central configuration file

## Specific Issues Found

### 1. Comprehensive Study (`02_run_comprehensive.sh` + `comprehensive_study.jl`)

**Bash script says:**
```bash
DATA_SIZE=51 (default)
MAX_DERIV=5 (default)
NUM_TRIALS=10 (default)
```

**Julia actually uses:**
```julia
DATA_SIZE = 101
MAX_DERIV = 7
TRIALS_PER_CONFIG = 3
NOISE_LEVELS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2, 5e-2]
```

**Impact:** Script prints misleading configuration info but runs correct values.

### 2. Pilot Study (`01_run_pilot.sh` + `pilot_study.jl`)

**Bash script says:**
```bash
MAX_DERIV=3
NOISE=0.0
```

**Julia actually uses:**
```julia
PILOT_TRIALS = 2
PILOT_NOISE_LEVELS = [0.0, 0.01]
PILOT_DATA_SIZES = [51, 101]
PILOT_MAX_DERIV = 4
```

**Impact:** Bash script mentions some values but Julia ignores them and uses more complex configuration.

### 3. Other Julia Programs with Hardcoded Constants

**`quick_study.jl`:**
```julia
MC_TRIALS = 10
NOISE_LEVELS = [0.01, 0.05]
DATA_SIZES = [21, 51]
MAX_DERIV = 3
```

**`test_run.jl`:**
```julia
MONTE_CARLO_TRIALS = 5
NOISE_LEVELS = [0.01, 0.1]
DATA_SIZES = [21, 51]
```

**Impact:** Different test/study scripts use different hardcoded values with no consistency.

### 4. Minimal Pilot (`minimal_pilot.jl`)

**Uses environment variables correctly:**
```julia
noise_level = parse(Float64, get(ENV, "NOISE", string(0.0)))
dsize = parse(Int, get(ENV, "DSIZE", string(51)))
max_deriv = parse(Int, get(ENV, "MAX_DERIV", string(3)))
```

**Impact:** This one is actually done right! Should be the model for others.

## Solution

Created `config.toml` as single source of truth. Next steps:

1. Add TOML.jl to Project.toml dependencies
2. Create `src/config_loader.jl` utility to read config
3. Update all Julia programs to use config
4. Update bash scripts to read from config or remove misleading output
5. Document that config.toml is the authoritative source

## Files Needing Updates

**Julia files:**
- [ ] `src/comprehensive_study.jl` - Read from config instead of hardcoded
- [ ] `src/pilot_study.jl` - Read from config instead of hardcoded
- [ ] `src/quick_study.jl` - Read from config or document as test fixture
- [ ] `src/test_run.jl` - Read from config or document as test fixture

**Bash files:**
- [ ] `scripts/02_run_comprehensive.sh` - Fix misleading output
- [ ] `scripts/01_run_pilot.sh` - Either read config or remove output

**New files:**
- [x] `config.toml` - Central configuration (CREATED)
- [ ] `src/config_loader.jl` - Julia utility to load config
- [x] `docs/CONFIG_ISSUES.md` - This document
