# Parallelization Guide

## Summary

The comprehensive study now supports **2 levels of parallelization**:

### Level 1: Julia Threading (Across Configurations) ✅ **IMPLEMENTED**
- **What**: Run 21 configurations (7 noise levels × 3 trials) in parallel
- **Speedup**: ~3-4x on 6 cores (21 configs → ~4-5 running simultaneously)
- **Usage**: `julia --threads=6 src/comprehensive_study.jl`
- **Status**: **Active** (script automatically uses 6 threads)

### Level 2: Python Multiprocessing (Within Each Configuration) 🔧 **OPTIONAL**
- **What**: Run ~25 Python methods in parallel within each configuration call
- **Speedup**: Additional ~2-3x speedup for Python portion
- **Trade-off**: More complexity, potential memory pressure with 6×N processes
- **Status**: **Not yet implemented** (recommend testing Level 1 first)

---

## Current Performance

### Bottlenecks Identified

**Without parallelization (sequential):**
```
Total time: ~45-60 minutes for 21 configurations
  - Per config: ~2-3 minutes
    - Python methods: ~60-90 seconds (25 methods × 2-4s each)
    - Julia methods: ~30-40 seconds (12 methods × 2-3s each)
    - I/O and overhead: ~10-20 seconds
```

**With Julia threading (6 threads):**
```
Total time: ~10-15 minutes for 21 configurations
  - Configs run in batches of 6 in parallel
  - 21 configs ÷ 6 threads = ~4 batches
  - ~2.5 minutes per batch → ~10 minutes total
```

**With both levels (estimated):**
```
Total time: ~5-8 minutes for 21 configurations
  - Python parallelized within each config
  - Both levels stack multiplicatively
```

---

## Level 1: Julia Threading (IMPLEMENTED)

### How It Works

**Before (Sequential):**
```julia
for noise_level in NOISE_LEVELS
    for trial in 1:TRIALS_PER_CONFIG
        # Process one config (2-3 minutes)
        run_python_methods(...)
        run_julia_methods(...)
        compute_errors(...)
    end
end
```

**After (Parallel):**
```julia
@threads for config_idx in 1:21
    noise_level, trial = configs[config_idx]
    # Each thread processes one config independently
    run_python_methods(...)  # Calls Python subprocess
    run_julia_methods(...)
    compute_errors(...)
end
```

### Thread Safety

- **File I/O**: Uses `ReentrantLock` for thread-safe writes
- **Random seeds**: Each config gets unique seed based on `config_idx`
- **Results**: Thread-local arrays, merged after parallel section
- **Python calls**: Each subprocess is independent (no shared state)

### Usage

```bash
# Automatic (script uses 6 threads)
./scripts/02_run_comprehensive.sh

# Manual
julia --project=. --threads=6 src/comprehensive_study.jl

# Check threading
julia --threads=6 -e 'using Base.Threads; println("Threads: ", nthreads())'
```

### Monitoring

The output shows which thread is processing each config:
```
[1/21] Thread 2: Processing noise=1.0e-8, trial=1...
[2/21] Thread 4: Processing noise=1.0e-8, trial=2...
[3/21] Thread 1: Processing noise=1.0e-8, trial=3...
...
```

---

## Level 2: Python Multiprocessing (OPTIONAL)

### When To Use

Consider adding Python parallelization if:
- ✓ You have >12 cores available
- ✓ Julia threading alone isn't fast enough
- ✓ You have sufficient RAM (~16GB+ recommended for 6×4 processes)

**Recommendation**: Test Level 1 first. If still too slow, add Level 2.

### Implementation Strategy

Modify `python/python_methods_integrated.py` line 295:

**Before:**
```python
results = {}
for method in methods:
    print(f"  Evaluating {method}...")
    results[method] = evaluator.evaluate_method(method)
```

**After (using joblib):**
```python
from joblib import Parallel, delayed

def eval_one(evaluator, method):
    return (method, evaluator.evaluate_method(method))

# Run 4 methods in parallel
n_jobs = int(os.environ.get("PYTHON_JOBS", "4"))
results = dict(Parallel(n_jobs=n_jobs)(
    delayed(eval_one)(evaluator, method)
    for method in methods
))
```

### Trade-offs

**Pros:**
- Faster Python method evaluation
- Independent method failures don't block others

**Cons:**
- More memory usage (N copies of data)
- Harder to debug
- Potential process spawning overhead
- With Julia threading, you'd have 6 Julia threads × 4 Python processes = 24 processes

---

## Tuning Recommendations

### Conservative (Default): 6 Julia threads only
```bash
julia --threads=6 src/comprehensive_study.jl
```
**Use when**:
- You have 8 cores or fewer
- You want simplicity and reliability
- Expected time: ~10-15 minutes

### Aggressive: Both levels
```bash
# Set Python to use 2-3 parallel jobs per config
export PYTHON_JOBS=3
julia --threads=6 src/comprehensive_study.jl
```
**Use when**:
- You have 12+ cores
- You have 16GB+ RAM
- You need maximum speed
- Expected time: ~5-8 minutes
- **Warning**: 6×3 = 18 processes simultaneously

### Minimal: Single-threaded (debugging)
```bash
julia --threads=1 src/comprehensive_study.jl
```
**Use when**:
- Debugging issues
- Comparing results
- Limited resources

---

## Monitoring Performance

### Check Thread Usage
```bash
# During run, check CPU usage
htop  # Look for julia processes using 100% on multiple cores

# Check thread count
ps -eLf | grep julia | wc -l
```

### Profile Bottlenecks
```julia
# Add at top of comprehensive_study.jl
using Profile

# Wrap main loop
@profile @threads for config_idx in 1:total_configs
    # ... existing code ...
end

Profile.print()
```

---

## Common Issues

### 1. "Too many open files"
**Cause**: Parallel I/O hitting OS limits
**Fix**: Increase file descriptor limit
```bash
ulimit -n 4096
```

### 2. Memory pressure
**Symptom**: System slowdown, swap usage
**Fix**: Reduce thread count
```bash
julia --threads=4 src/comprehensive_study.jl
```

### 3. Non-deterministic results
**Cause**: Race condition in RNG or I/O
**Check**: Each config uses unique seed (config_idx * 1000 + trial)

### 4. Hanging processes
**Symptom**: Some threads never complete
**Check**: Python subprocess timeouts are working (300s default)

---

## Future Improvements

### GPU Acceleration
Some methods (GP, Fourier) could benefit from GPU:
```julia
using CUDA
# Move expensive operations to GPU
```

### Distributed Computing
For massive studies, use Julia's Distributed module:
```julia
using Distributed
addprocs(24)  # Across multiple machines
@distributed for config in configs
    # Process config
end
```

### Incremental Results
Save results after each config (already happening) to allow:
- Resume from failure
- Monitor progress in real-time
- Early termination

---

## Performance Expectations

| Setup | Cores | Threads | Python Jobs | Time | Speedup |
|-------|-------|---------|-------------|------|---------|
| Sequential | 1 | 1 | 1 | 60 min | 1.0x |
| Julia threading | 8 | 6 | 1 | 12 min | 5.0x |
| Both levels | 12 | 6 | 3 | 6 min | 10.0x |
| Both levels | 16 | 8 | 4 | 4 min | 15.0x |

*Times are estimates. Actual performance depends on CPU, memory, and method selection.*
