using Distributions
using StatsBase
include("utils.jl")


# We assume here that proj is implemented symmetrically for primal and dual variables
function extraGrad(A::Array{Float64, 2}, proj::Function, z0::Array{Float64, 1},  τ::Float64, iter::Int64; tol=1e-6)
    m, n = size(A)
    @assert  size(z0)[1] == m + n
    energy = Float64[]
    x, y = z0[1:n], z0[(n+1):end]
    iter = ceil(Int, iter / 2) # since every iteration is 2 epochs.

    for i in 1:iter
        x, y, energy = extraGradUpdate(A, proj, x, y, energy, τ)
        if energy[end] < tol
            println("The extragradient algorithm achieves $tol accuracy in $(i-1) iterations")
            break
        end
    end
    return energy, x, y
end

"""
    extraGradUpdate: main update for extragradient method for matrix games on a simplex.
"""
function extraGradUpdate(A, proj, x, y, energy, τ)

    Ax, ATy =  A * x, A'* y
    gap = maximum(Ax) - minimum(ATy)
    append!(energy, gap)
    x_ = proj(x .- τ .* ATy)
    y_ = proj(y .+ τ .* Ax)
    # AA: added below to avoid multiplying by 2 when plotting
    Ax, ATy =  A * x_, A'* y_
    gap = maximum(Ax) - minimum(ATy)
    append!(energy, gap)
    x = proj(x .- τ .* ATy)
    y = proj(y .+ τ .* Ax)
    return x, y, energy
end

####################### Extragrad with Bregman update #######
function extraGrad_bregman(A::Array{T, 2}, z0::Array{T, 1},  τ::T, iter::Int64; tol=1e-6) where {T<:Real}
    m, n = size(A)
    iter = ceil(Int, iter / 2) # since every iteration is 2 epochs.
    if size(z0)[1] != m + n
        println("The dimension of a starting point doesn't match the ones of the matrix A")
    else
        energy = Float64[]
        x, y = z0[1:n], z0[(n+1):end]
        for i in 1:iter
            Ax, ATy =  A * x, A'* y
            gap = maximum(Ax) - minimum(ATy)
            append!(energy, gap)
            if gap > tol
                x_ = x .* exp.(-τ * ATy)
                x_ ./= sum(x_)
                y_ = y .* exp.(τ * Ax)
                y_ ./= sum(y_)
                Ax, ATy =  A * x_, A'* y_
                gap = maximum(Ax) - minimum(ATy)
                append!(energy, gap)
                x = x .* exp.(-τ .* ATy)
                x ./= sum(x)
                y  = y .* exp.(τ .* Ax)
                y ./= sum(y)
            else
                println("The extragradient-bregman algorithm achieves $tol accuracy in $(i-1) iterations")
                break
            end
        end
        return energy, x, y
    end
end


function extraGradBregman(A::Array{T, 2}, z0::Array{T, 1},
                          τ::T, iter::Int64; tol=1e-6) where {T<:Real}
    m, n = size(A)
    @assert m + n == size(z0)[1]
    energy = Float64[]
    x, y = z0[1:n], z0[(n+1):end]
    X, Y = zeros(Float64, n), zeros(Float64, m)
    for i in 1:iter
        x, y, X, Y, energy = extraGradBregmanUpdate(A, x, y, X, Y, energy, τ)
        if energy[end] < tol
            println("The extragradient algorithm achieves $tol accuracy in $(i-1) iterations")
            break
        end
    end
    return energy, x, y
end


function extraGradBregmanUpdate(A, x, y, X, Y, energy, τ)
    Ax, ATy =  A * x, A'* y
    gap = maximum(Ax) - minimum(ATy)
    append!(energy, gap)
    X_ = X - τ * ATy
    Y_ = Y + τ * Ax
    X = X - τ * (A' * softmax(Y_))
    Y = Y + τ * (A * softmax(X_))
    x = softmax(X)
    y = softmax(Y)
    return x, y, X, Y, energy
end



######################### ExtraGrad ######################################################
"""
    stochastic ExtraGrad with variance reduction, loopless variant.

bregman = true: uses mirror projection automatically. It ignores input function `proj`
bregman = false: uses Euclidean projection, given by `proj`
distr = false: uses uniform sampling for rows and columns of A
distr = true: uses weighted sampling. If `bregma=true` it uses l_1 sampling which is
    computed in every iteration. If `bregman=false` it uses a sampling |A[i,:]|^2/|A|_F and similarly for columns.
"""

function stochExtraGradLoopless(A::Array{Float64, 2}, proj::Function,
                           z0::Array{Float64, 1},  τ::Float64,
                           α::Float64, p::Float64,
                           max_epoch::Int64; bregman=false, distr=false, tol=1e-6)
    m, n = size(A)
    @assert m + n == size(z0)[1]
    iter = ceil(Int, max_epoch * 2 * m * n / (p * 2 * m * n + m + n ))
    cheap_update = (m + n) / (2 * m * n)
    update_wArray = rand(Bernoulli(p), iter)

    # check if sampling is uniform or with weights |A_i|^2/|A|_F
    if distr && !bregman
        arrayI, arrayJ, rows_weights, columns_weights = sample_with_Frobenius(A, iter)
    elseif !bregman
        arrayI = rand(1:m, iter)
        arrayJ = rand(1:n, iter)
    end
    x, y = z0[1:n], z0[(n+1):end]
    wx, wy = x, y
    wx_old, wy_old = x, y
    Awx, Awy = A * wx, A' * wy
    energy = [maximum(Awx) - minimum(Awy)]
    epoch = [0.0]
    epoch_count = 0.0

    for k in 1:iter
        update_w = update_wArray[k]
        # choose a distance for update: KL or euclidean
        if bregman
            x, y, wx, wy,  Awx, Awy, energy, epoch, epoch_count =
                stochExtraGradLooplessBregman_update(A, x, y, wx, wy, Awx, Awy, energy,
                                                     epoch, epoch_count, α, τ, cheap_update, update_w, distr)
        else
            i, j = arrayI[k], arrayJ[k]
            x, y, wx, wy, Awx, Awy, energy, epoch, epoch_count =
                stochExtraGradLoopless_update(A, proj, x, y, wx, wy, Awx, Awy,
                                         energy, epoch, epoch_count, α, τ, i, j, cheap_update,
                                         update_w, rows_weights, columns_weights)
        end
        if energy[end] < tol
            total_epoch = ceil(sum(epoch))
            println("StocExtraGrad-VR achieves $tol accuracy in $(total_epoch) epochs")
            break
        end
    end
    gap = maximum(A * x) - minimum(A' * y)
    append!(energy, gap)
    append!(epoch, 1.0)
    return energy, x, y, cumsum(epoch)
end

function stochExtraGradLoopless_update(A, proj, x, y, wx, wy, Awx, Awy, energy, epoch,
                                       epoch_count, α, τ, i, j, cheap_update, update_w,
                                       rows_weights, columns_weights)

    Ai = A[i, :]
    ATj = A[:, j]
    x_ = α .* x .+ (1-α) .* wx
    y_ = α .* y .+ (1-α) .* wy
    xx = proj(x_ .- τ .* Awy)
    yy = proj(y_ .+ τ .* Awx)
    x = proj(x_ .- τ .* (Awy  .+ (1/rows_weights[i] * (yy[i] - wy[i])) .* Ai))
    y = proj(y_ .+ τ .* (Awx  .+ (1/columns_weights[i] * (xx[j] - wx[j])) .* ATj))
    epoch_count += cheap_update
    if update_w
        wx, wy = x, y
        Awx, Awy = A * wx, A' * wy
        gap = maximum(Awx) - minimum(Awy)
        append!(energy, gap)
        epoch_count += 1.0
        append!(epoch, epoch_count)
        epoch_count = 0.0
    end

    return x, y, wx, wy,  Awx, Awy, energy, epoch, epoch_count
end


function stochExtraGradLooplessBregman_update(A, x, y, wx, wy, Awx, Awy, energy, epoch,
                                              epoch_count, α, τ, cheap_update, update_w, distr)
    # Main update
    x_ = x.^α .* wx.^(1-α)
    y_ = y.^α .* wy.^(1-α)
    xx = x_ .* exp.(-τ .*  Awy)
    yy = y_ .* exp.(τ .* Awx)
    xx ./= sum(xx)
    yy ./= sum(yy)

    i, weights_yi = sampling_Bregman(yy, wy, distr)
    j, weights_xj = sampling_Bregman(xx, wx, distr)
    Ai = A[i, :]
    ATj = A[:, j]

    x = x_ .* exp.(-τ .* (Awy  .+ ((yy[i] - wy[i]) / weights_yi) .* Ai))
    y = y_ .* exp.(τ .* (Awx  .+ ((xx[j] - wx[j]) / weights_xj) .* ATj))
    x ./= sum(x)
    y ./= sum(y)
    epoch_count += cheap_update
    if update_w
        wx, wy = x, y
        Awx, Awy = A * wx, A' * wy
        gap = maximum(Awx) - minimum(Awy)
        append!(energy, gap)
        epoch_count += 1
        append!(epoch, epoch_count)
        epoch_count = 0.0
    end
    return x, y, wx, wy, Awx, Awy, energy, epoch, epoch_count
end

##########################################################################################

"""
    stochastic FoRB with variance reduction, loopless variant.

bregman = true: uses mirror projection automatically. It ignores input function `proj`
bregman = false: uses Euclidean projection, given by `proj`
distr = false: uses uniform sampling for rows and columns of A
distr = true: uses weighted sampling. If `bregma=true` it uses l_1 sampling which is
    computed in every iteration. If `bregman=false` it uses a sampling |A[i,:]|^2/|A|_F and similarly for columns.
"""

function stochForbLoopless(A::Array{Float64, 2}, proj::Function,
                           z0::Array{Float64, 1},  τ::Float64,
                           α::Float64, p::Float64,
                           max_epoch::Int64; bregman=false, distr=false, tol=1e-6)
    m, n = size(A)
    @assert m + n == size(z0)[1]
    x_weights_ar, y_weights_ar = Float64[], Float64[]
    # Input is max_epoch now, let us compute approximate # of iterations
    # every iteration costs  p * 2mn + m+n and our overall budget is iter * 2mn
    # This will not be exact, but it seems close enough
    iter = ceil(Int, max_epoch * 2 * m * n / (p * 2 * m * n + m + n ))
    cheap_update = (m + n) / (2 * m * n)

    update_wArray = rand(Bernoulli(p), iter)
    # check if sampling is uniform or with weights |A_i|^2/|A|_F
    if distr && !bregman
        arrayI, arrayJ, rows_weights, columns_weights = sample_with_Frobenius(A, iter)
    elseif !bregman
        arrayI = rand(1:m, iter)
        arrayJ = rand(1:n, iter)
    end
    x, y = z0[1:n], z0[(n+1):end]
    wx, wy = x, y
    wx_old, wy_old = x, y
    Awx, Awy = A * wx, A' * wy
    energy = [maximum(Awx) - minimum(Awy)]
    epoch = [0.0]
    epoch_count = 0.0

    for k in 1:iter
        update_w = update_wArray[k]
        # choose a distance for update: KL or euclidean
        if bregman
            x, y, wx, wy, wx_old, wy_old, Awx, Awy, energy, epoch, epoch_count, x_weights_ar, y_weights_ar =
                stochForbLooplessBregman_update(A, x, y, wx, wy, wx_old, wy_old,
                                                Awx, Awy, energy, epoch, epoch_count,
                                                α, τ, cheap_update, update_w, distr,x_weights_ar, y_weights_ar)
        else
            i, j = arrayI[k], arrayJ[k]
            x, y, wx, wy, wx_old, wy_old, Awx, Awy, energy, epoch, epoch_count =
                stochForbLoopless_update(A, proj, x, y, wx, wy, wx_old, wy_old, Awx, Awy,
                                         energy, epoch, epoch_count, α, τ, i, j, cheap_update,
                                         update_w, rows_weights, columns_weights)
        end
        if energy[end] < tol
            total_epoch = ceil(sum(epoch))
            println("StochFoRB-VR algorithm achieves $tol accuracy in $(total_epoch) epochs")
            break
        end
    end
    gap = maximum(A * x) - minimum(A' * y)
    append!(energy, gap)
    append!(epoch, 1.0) ## not sure about this
    return energy, x, y, cumsum(epoch), x_weights_ar, y_weights_ar
end

function sample_with_Frobenius(A, iter)
    m, n = size(A)
    frobenius_norm = norm(A)
    rows_norm = [norm(A[i, :]) for i in 1:m]
    columns_norm = [norm(A[:, j]) for j in 1:n]
    rows_weights = rows_norm.^2 / frobenius_norm^2
    columns_weights = columns_norm.^2 / frobenius_norm^2
    arrayI = sample((1:m), Weights(rows_weights), iter)
    arrayJ = sample((1:n), Weights(columns_weights), iter)
    return arrayI, arrayJ, rows_weights, columns_weights
end

function stochForbLoopless_update(A, proj, x, y, wx, wy, wx_old, wy_old, Awx, Awy,
                                  energy, epoch, epoch_count, α, τ, i, j, cheap_update,
                                  update_w, rows_weights, columns_weights)
    Ai = A[i, :]
    ATj = A[:, j]
    x = proj(α .* x .+ (1-α) .* wx .- τ .* (Awy  .+ (1/rows_weights[i] * (y[i] - wy_old[i])) .* Ai))
    y = proj(α .* y .+ (1-α) .* wy .+ τ .* (Awx  .+ (1/columns_weights[i] * (x[j] - wx_old[j])) .* ATj))

    # Cost is (m+n) / (2mn). ONLY for dense case, I will do sparse case later.
    #cheap_update = (m + n) / (2 * m * n) I defined it in the main program
    epoch_count += cheap_update
    wx_old, wy_old = wx, wy
    if update_w
        wx, wy = x, y
        Awx, Awy = A * wx, A' * wy
        gap = maximum(Awx) - minimum(Awy)
        append!(energy, gap)
        epoch_count += 1.0
        append!(epoch, epoch_count)
        epoch_count = 0.0
    end
    return x, y, wx, wy, wx_old, wy_old, Awx, Awy, energy, epoch, epoch_count
end

function sampling_Bregman(x, wx, distr)
    # Sampling procedure
    n = length(x)
    # what is cheaper: to pass n as an argument or every time take length?
    if distr == true
        diff_x = abs.(x .- wx)
        sum_diff_x = sum(diff_x)
        if sum_diff_x < 1e-12
            j = 1 # anyway the vector x - wx is zero
            weights_xj = 1.0 / n
        else
            weights_x = diff_x ./ sum_diff_x
            j = sample((1:n), Weights(weights_x))
            weights_xj = weights_x[j]
        end
        @assert weights_xj != 0
    else # Uniform sampling
        j = sample((1:n))
        weights_xj = 1.0 / n
    end
    return j, weights_xj
end


function stochForbLooplessBregman_update(A, x, y, wx, wy, wx_old, wy_old,
                                         Awx, Awy, energy, epoch, epoch_count,
                                         α, τ, cheap_update, update_w, distr, x_weights_ar, y_weights_ar)

    i, weights_yi = sampling_Bregman(y, wy_old, distr)
    j, weights_xj = sampling_Bregman(x, wx_old, distr)
    append!(x_weights_ar, weights_xj)
    append!(y_weights_ar, weights_yi)

    # Main update
    Ai = A[i, :]
    ATj = A[:, j]
    xx = x.^α .* wx.^(1-α) .* exp.(-τ .* (Awy  .+ ((y[i] - wy_old[i]) / weights_yi) .* Ai))
    yy = y.^α .* wy.^(1-α) .* exp.(τ .* (Awx  .+ ((x[j] - wx_old[j]) / weights_xj) .* ATj))
    x = xx ./ sum(xx)
    y = yy ./ sum(yy)
    epoch_count +=  cheap_update
    wx_old, wy_old = wx, wy
    if update_w
        wx, wy = x, y
        Awx, Awy = A * wx, A' * wy
        gap = maximum(Awx) - minimum(Awy)
        append!(energy, gap)
        epoch_count += 1
        append!(epoch, epoch_count)
        epoch_count = 0.0
    end
    return x, y, wx, wy, wx_old, wy_old, Awx, Awy, energy, epoch, epoch_count, x_weights_ar, y_weights_ar
end

"""
Carmon et al. variant
"""
function stochMPCarmon(A::Array{Float64, 2}, proj::Function,
                       z0::Array{Float64, 1},  α::Float64,
                       η::Float64, max_epoch::Int64;
                       bregman=false, distr=false, tol=1e-6)

    m, n = size(A)
    @assert m + n == size(z0)[1]
    T = ceil(Int, 4 / (η * α))
    outer_iter = ceil(Int, max_epoch * 2 * m * n / (4 * m * n + T * (m + n)))
    cheap_update = (m + n) / (2 * m * n)
    # averaged cost for each time we save data. In [Carmon et al], every outer iteration
    # does two full updates, thus 2 times save data. Cost of one outer iteration
    # is (2 + K*cheap_update) so cost per save is (2 + K*cheap_update)/2
    cost_per_save = (2 + T * cheap_update)/2

    # check if sampling is uniform or with weights |A_i|^2/|A|_F
    if distr && !bregman
        arrayI, arrayJ, rows_weights, columns_weights = sample_with_Frobenius(A, T * outer_iter)
    elseif !bregman
        rows_weights = 1 / m * ones(Float64, n)
        columns_weights = 1 / n * ones(Float64, m)
        arrayI = rand(1:m, T * outer_iter) # AA: fixing error for iter
        arrayJ = rand(1:n, T * outer_iter)
    end
    x, y = z0[1:n], z0[(n+1):end]
    if bregman
        Awx, Awy = A * x, A' * y
        x0, y0 = ones(n) + log.(x), ones(m) + log.(y)
        x, y = x0, y0
    else
        x0, y0 = x, y
        Awx, Awy = A * x, A' * y
    end
    energy = [maximum(Awx) - minimum(Awy)]

    x_avg, y_avg = zeros(Float64, n), zeros(Float64, m)
    for k in 1:outer_iter
        if bregman
            Awx, Awy, energy = compute_full_operator(A, softmax(x0), softmax(y0), energy)
        else
            Awx, Awy, energy = compute_full_operator(A, x0, y0, energy)
        end
        if energy[end] < tol
            break
        end
        if bregman
            x, y, x_avg, y_avg =
                stochMPCarmon_update_bregman(A, x, y, x_avg, y_avg, x0, y0, Awx, Awy,
                                     α, η, m, n, T, k)
        else
            x, y, x_avg, y_avg =
                stochMPCarmon_update_euclidean(A, proj, x, y, x_avg, y_avg, x0, y0, Awx, Awy,
                                     α, η, arrayI, arrayJ, m, n, rows_weights, columns_weights, T, k)
        end
        # x_avg is in the primal space, so no need to softmax here.
        Awx, Awy, energy = compute_full_operator(A, x_avg, y_avg, energy)

        if energy[end] < tol
            #total_epoch = ceil(sum(epoch))
            println("Stochastic Carmon algorithm achieved $tol accuracy")
            break
        end

        if bregman
            # x0 in the dual space
            x0, y0 = stochMPCarmon_fullupdate_bregman(x0, y0, Awx, Awy, α)
        else
            x0, y0 = stochMPCarmon_fullupdate_euclidean(proj, x0, y0, Awx, Awy, α)
        end
    end
    running_cost = Array(1:length(energy)) * cost_per_save
    return energy, x, y, running_cost
end


function stochMPCarmon_update_euclidean(A, proj, x, y, x_avg, y_avg, x0, y0, Awx, Awy,
                              α, η, arrayI, arrayJ, m, n, rows_weights, columns_weights, T, k)
    x, y = x0, y0
    for t in 1:T
        i = arrayI[(k - 1) * T + t]
        j = arrayJ[(k - 1) * T + t]
        Ai = A[i, :]
        ATj = A[:, j]

        x = proj((x .+ (η * α / 2) .* x0 .- η .* (Awy .+ (1/rows_weights[i] * (y[i] - y0[i])) .* Ai)) ./ (1 + η*α/2) )
        y = proj((y .+ (η * α / 2) .* y0 .+ η .* (Awx .+ (1/columns_weights[j] * (x[j] - x0[j])) .* ATj)) ./ (1 + η*α/2) )
        x_avg = (1 / t) .* x .+ (1 - 1 / t) .* x_avg
        y_avg = (1 / t) .* y .+ (1 - 1 / t) .* y_avg
    end
    return x, y, x_avg, y_avg
end

function stochMPCarmon_update_bregman(A, x, y, x_avg, y_avg, x0, y0, Awx, Awy,
                                    α, η, m, n, T, k)
    x, y = x0, y0
    for t in 1:T
        oracle_x = MirProxVR_stoch_oracle(y, softmax(y0), A, 1)
        oracle_y = MirProxVR_stoch_oracle(x, softmax(x0), A, 2)

        x = (x .+ (η * α / 2) .* x0 .- η .* (Awy .+ oracle_x)) ./ (1 + η * α / 2)
        y = (y .+ (η * α / 2) .* y0 .+ η .* (Awx .+ oracle_y)) ./ (1 + η * α / 2)
        x_avg = (1 / t) .* softmax(x) .+ (1 - 1 / t) .* x_avg
        y_avg = (1 / t) .* softmax(y) .+ (1 - 1 / t) .* y_avg
    end
    return x, y, x_avg, y_avg
end

function stochMPCarmon_fullupdate_euclidean(proj, x0, y0, Awx, Awy, α)
    x = proj(x0 .- (Awy ./ α))
    y = proj(y0 .+ (Awx ./ α))
return x, y
end

function stochMPCarmon_fullupdate_bregman(x0, y0, Awx, Awy, α)
    x = x0 .- (Awy ./ α)
    y = y0 .+ (Awx ./ α)
return x, y
end

function compute_full_operator(A, x_avg, y_avg, energy)
    Awx, Awy = A * x_avg, A' * y_avg
    append!(energy, maximum(Awx) - minimum(Awy))
    return Awx, Awy, energy
end


#### Stochastic Bregman Extragradient-VR (mirror prox) for Matrix Games.

"""
Stochastic Extragradient with variance reduction, two-loop variant.
X = nabla h_1(x)
u, v = w_k
u_new, v_new = w_{k+1}

U, V = nabla h(w_bar_k)
U_new, V_new = nabla h(w_bar_{k+1})
"""
function MirProxVR_stoch_oracle(X, u, A, flag)
    N = length(X)
    x = softmax(X)
    diff = x .- u
    abs_diff = abs.(diff)
    norm_diff = sum(abs_diff)
    i = sample((1:N), Weights(abs_diff/norm_diff))
    if flag == 1
        row_or_column = A[i, :]
    else
        row_or_column = A[:, i]
    end
    return (norm_diff * sign(diff[i])) .* row_or_column
end


function MirProxVR_update(Z, w, W_, w_next, W_next, Fw, A, α, τ, k, n)
    Z_ = α .* Z .+ (1 - α) .* W_ .- τ .* Fw
    oracle_x = MirProxVR_stoch_oracle(Z_[(n+1):end], w[(n+1):end], A, 1)
    oracle_y = MirProxVR_stoch_oracle(Z_[1:n], w[1:n], A, 2)
    Z .= Z_ .- τ .* [oracle_x; -oracle_y]
    z = [softmax(Z[1:n]); softmax(Z[(n+1):end])]

    w_next = (z .+ (k-1) .* w_next) ./ k
    W_next = (Z .+ (k-1) .* W_next) ./ k
    return Z, w_next, W_next
end

function MirProxVR(A::Array{Float64, 2}, z0::Array{Float64, 1},
                        τ::Float64, α::Float64, K::Int64, max_epoch::Int64;  tol=1e-6)
    m, n = size(A)
    @assert m + n == size(z0)[1]
    cheap_update = (m + n) / (2 * m * n)
    cost_per_epoch = 1 + K * cheap_update
    S = ceil(Int, max_epoch * 2 * m * n / (2 * m * n + K * (m + n)))
    z = copy(z0)
    Z = log.(z) + ones(m+n)
    w = z
    W_ = Z
    w_next = zeros(Float64, m+n)
    W_next = zeros(Float64, m+n)
    Fw = [A' * w[(n+1):end];  -A * w[1:n]]
    energy = [maximum(-Fw[(n+1):end]) - minimum(Fw[1:n])]
    for s in 1:S
        for k in 1:K
            Z, w_next, W_next =
                MirProxVR_update(Z, w, W_, w_next, W_next, Fw, A, α, τ, k, n)
        end
        W_ = W_next
        w = w_next
        # no need to update w_next, W_next, since for k=1, they are just zero anyways
        Fw = [A' * w[(n+1):end];  -A * w[1:n]]
        gap = maximum(-Fw[(n+1):end]) - minimum(Fw[1:n])
        append!(energy, gap)
        if gap < tol
            println("StochMirProx-VR algorithm achieved $tol accuracy")
            break
        end
    end
    # compute how much we spent
    running_cost = Array(1:length(energy)) * cost_per_epoch
    z = softmax(Z)
    return energy, z[1:n], z[n:end], running_cost
end

######################################################
######################################################
######################################################
################# Extragrad - Looped #################
######################################################
######################################################
######################################################
function stochExtraGradLooped(A::Array{Float64, 2}, proj::Function,
                           z0::Array{Float64, 1},  τ::Float64,
                           α::Float64, T::Int64,
                           max_epoch::Int64; bregman=false, distr=false, tol=1e-6)
    m, n = size(A)
    @assert m + n == size(z0)[1]
    outer_iter = ceil(Int, max_epoch * 2 * m * n / (2 * m * n + T * (m + n)))
    cheap_update = (m + n) / (2 * m * n)
    cost_per_epoch = 1 + T * cheap_update

    if distr && !bregman
        arrayI, arrayJ, rows_weights, columns_weights = sample_with_Frobenius(A, T * outer_iter)
    elseif !bregman
        rows_weights = 1 / m * ones(Float64, n)
        columns_weights = 1 / n * ones(Float64, m)
        arrayI = rand(1:m, T * outer_iter)
        arrayJ = rand(1:n, T * outer_iter)
    end

    x, y = z0[1:n], z0[(n+1):end]
    x0, y0 = x, y
    wx, wy = x, y
    Awx, Awy = A * x, A' * y
    epoch = [0.0]
    energy = [maximum(Awx) - minimum(Awy)]
    epoch_count = 0.0

    x_avg, y_avg = zeros(Float64, n), zeros(Float64, m)

    for k in 1:outer_iter
        x, y, x_avg, y_avg =
            stochExtraGradLooped_update(A, proj, x, y, x_avg, y_avg, wx, wy, Awx, Awy,
                                 α, τ, arrayI, arrayJ, m, n, rows_weights, columns_weights, T, k)

        Awx, Awy, energy = compute_full_operator(A, x_avg, y_avg, energy)
        wx, wy = x_avg, y_avg
        if energy[end] < tol
            println("StocExtraGrad-VR-Looped achieved $tol accuracy")
            break
        end
    end
    # x_avg is on the primal space, so no need to softmax here.
    running_cost = Array(1:length(energy)) * cost_per_epoch
    return energy, x, y, running_cost
end

function stochExtraGradLooped_update(A, proj, x, y, x_avg, y_avg, wx, wy, Awx, Awy, α, τ,
                                        arrayI, arrayJ, m, n, rows_weights, columns_weights, T, k)
    for t in 1:T
        i = arrayI[(k - 1) * T + t]
        j = arrayJ[(k - 1) * T + t]
        Ai = A[i, :]
        ATj = A[:, j]
        x_ = α .* x .+ (1-α) .* wx
        y_ = α .* y .+ (1-α) .* wy
        xx = proj(x_ .- τ .* Awy)
        yy = proj(y_ .+ τ .* Awx)
        x = proj(x_ .- τ .* (Awy  .+ (1/rows_weights[i] * (yy[i] - wy[i])) .* Ai))
        y = proj(y_ .+ τ .* (Awx  .+ (1/columns_weights[i] * (xx[j] - wx[j])) .* ATj))

        x_avg = 1 / t .* x .+ (1 - 1 / t) .* x_avg
        y_avg = 1 / t .* y .+ (1 - 1 / t) .* y_avg
    end

    return x, y, x_avg, y_avg
end
