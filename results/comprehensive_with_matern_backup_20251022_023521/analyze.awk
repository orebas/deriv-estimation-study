BEGIN {
    FS=","
    OFS=","
}
NR==1 {
    # Find column indices
    for (i=1; i<=NF; i++) {
        col[$i] = i
    }
    next
}
{
    method = $col["method"]
    category = $col["category"]
    nrmse = $col["mean_nrmse"]
    timing = $col["mean_timing"]
    
    count[method]++
    sum_nrmse[method] += nrmse
    sum_timing[method] += timing
    cat[method] = category
}
END {
    print "method,category,config_count,mean_nrmse,mean_timing"
    for (m in count) {
        if (count[m] == 56) {
            printf "%s,%s,%d,%.10f,%.6f\n", m, cat[m], count[m], sum_nrmse[m]/count[m], sum_timing[m]/count[m]
        }
    }
}
