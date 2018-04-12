
dt = float.(readcsv("/Users/kristian/Dropbox/PhD/Fo3_tmp.txt")[2:end, :])

println("TE SETUP")
te = te_from_ts(dt[1:50, 1], dt[1:50, 2], binsizes = [20, 30])
te = te_from_ts(dt[1:50, 1], dt[1:50, 2], binsizes = collect(1:50))

println("TE(Fo3 -> Temp)")
te1_to_2_new = te_from_ts(dt[:, 1], dt[:, 2], n_reps = 1, binsizes = [100])
