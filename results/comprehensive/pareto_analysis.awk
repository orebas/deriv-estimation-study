BEGIN { FS=","; OFS="," }
NR==1 { next }
{
    idx++
    method[idx] = $1
    category[idx] = $2
    nrmse[idx] = $4
    timing[idx] = $5
}
END {
    print "Pareto-optimal methods:"
    print "Method                          Category              nRMSE      Time(s)"
    print "============================================================================="
    
    for (i=1; i<=idx; i++) {
        dominated = 0
        # Check if any other method dominates this one
        for (j=1; j<=idx; j++) {
            if (i != j) {
                # Method j dominates i if j is BOTH faster AND more accurate
                if (timing[j] < timing[i] && nrmse[j] < nrmse[i]) {
                    dominated = 1
                    break
                }
            }
        }
        if (!dominated) {
            # Exclude catastrophically failed methods (nRMSE > 1e6)
            if (nrmse[i] < 1000000) {
                printf "%-32s %-22s %8.4f   %8.6f\n", method[i], category[i], nrmse[i], timing[i]
            }
        }
    }
}
