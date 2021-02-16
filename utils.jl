using DataStructures: MutableBinaryMaxHeap, top_with_handle, delete!

function proj_simplex0(v::Array{T, 1}, s::T=one(T)) where {T<:Real}
    n = length(v)
    if sum(v) == s && all(v .≥ 0)
        w = v
    else
        u = sort(v, rev=true)
        cssv = cumsum(u)
        ρ = findall(>(0), u .* Array((1:n)) .> (cssv .- s))[end]
        θ = (cssv[ρ] - s) / (ρ)
        w = max.(v .- θ, zero(T))
    end
    return w
end


""" Algorithm from Held, M., Wolfe, P., Crowder, H.: 'Validation of
subgradient optimization'. The description is from the Condat L.:
'Fast Projection onto the Simplex and the l1 Ball'. (Algorithm 1)"""

function proj_simplex1(y::Array{T, 1}, a::T=one(T)) where {T<:Real}
    n = length(y)
    if sum(y) == a && all(y .≥ 0)
        x = y
    else
        let τ
            u = sort(y, rev=true)
            cumsum_u = zero(T)
            for k in 1:n
                if cumsum_u + u[k] < k * u[k] + a
                    cumsum_u += u[k]
                    τ = (cumsum_u - a) / k
                else
                    break
                end
            end
            x = max.(y .- τ, zero(T))
        end
    end
    return x
end


function proj_simplex12(y::Array{T, 1}, a::T=one(T)) where {T<:Real}
    n = length(y)
    if sum(y) == a && all(y .≥ 0)
        return y
    else
        τ = 0.0
        u = sort(y, rev=true)
        cumsum_u = zero(T)
        for k in 1:n
            if cumsum_u + u[k] < k * u[k] + a
                cumsum_u += u[k]
                τ = (cumsum_u - a) / k
            else
                break
            end
        end
        y = max.(y .- τ, zero(T))
    end
    return y
end




""" Algorithm from van den Berg, E., Friedlander, M.P.: 'Probing the
Pareto frontier for basis pursuit solution'. The description is from
Condat L: 'Fast Projection onto the Simplex and the l1
Ball'. (Algorithm 2)"""


function proj_simplex2(y::Array{T, 1}, a::T=one(T)) where {T<:Real}
    N = length(y)
    if sum(y) == a && all(y .≥ 0)
        x = y
    else
        τ = zero(T)
        v = MutableBinaryMaxHeap(y)
        cumsum_u = zero(T)
        for k in 1:N
            u = first(v)
            if cumsum_u + u < k * u + a
                cumsum_u += u
                i = top_with_handle(v)[2]
                delete!(v, i)
                τ = (cumsum_u - a) / k
            else
                break
            end
        end
        x = max.(y .- τ, zero(T))
    end
    return x
end

function proj_simplex22(y::Array{Float64, 1}, a=1.0)
    N = length(y)
    if sum(y) == a && all(y .≥ 0)
        x = y
    else
        τ = 0.
        v = MutableBinaryMaxHeap(y)
        cumsum_u = 0.
        for k in 1:N
            u = first(v)
            if cumsum_u + u < k * u + a
                cumsum_u += u
                i = top_with_handle(v)[2]
                delete!(v, i)
                τ = (cumsum_u - a) / k
            else
                break
            end
        end
        x = max.(y .- τ, 0.)
    end
    return x
end

"""
From Condat “Fast projection onto the simplex and the l_1 ball”. In: Mathematical Programming 158.1
(2016), pp. 575–585
"""
function proj_simplex_condat(y::Array{Float64, 1}, a=Float64(1.0))
    N = length(y)
    v = [y[1]]
    v_tilde = Float64[]
    ρ = y[1] - a
    for n in 2:N
        yn = y[n]
        if yn > ρ
            ρ += (yn - ρ) / (length(v) + 1)
            if ρ > yn - a
                append!(v, y[n])
            else
                append!(v_tilde, v)
                v = [yn]
                ρ = yn - a
            end
        end
    end

    if !isempty(v_tilde)
        for yi in v_tilde
            if yi > ρ
                append!(v, yi)
                ρ += (yi - ρ) / length(v)
            end
        end
    end
    # if during the loop rho is increases at least once, then the flag is true. Otherwise we stop
    flag = true
    while flag
        flag = false
        for (i, yi) in enumerate(v)
            if yi ≤ ρ
                deleteat!(v, i)
                ρ += (ρ - yi) / length(v)
                flag = true
            end
        end
    end
    τ = ρ
    x = max.(y .- τ, zero(y))
    return x
end

function proj_simplex3(v, z=1.)
    n = length(v)
    U = Array((1:n))
    s = 0
    ρ = 0
    while length(U) > 0
        G = []
        L = []
        k = U[rand(1:length(U))]
        ds = v[k]
        for j in U
            if v[j] >= v[k]
                if j != k
                    ds += v[j]
                    append!(G, j)
                end
            elseif v[j] < v[k]
                append!(L, j)
            end
        end
        drho = length(G) + 1
        if s + ds - (ρ + drho) * v[k] < z
            s += ds
            ρ += drho
            U = L
        else
            U = G
        end
    end
    theta = (s - z) / ρ
    return max.(v .- theta, 0.)

end




function proj_simplex4(v, z=1., τ=1e-7, max_iter=1000)
    lower = 0
    upper = maximum(v)
    current = Inf
    w = zeros(n)
    for it in 1:max_iter
        if abs(current) / z < τ && current < 0.
            break
        end
        theta = (upper + lower) / 2.0
        w = max.(v .- theta, 0.)
        current = sum(w) - z
        if current <= 0.
            upper = theta
        else
            lower = theta
        end
    end
    return w
end




function softmax(x::Array{T, 1}) where {T<:Real}
    # compare with NNlib implementation
    res = exp.(x .- maximum(x))
    res ./= sum(res)
    return res
end
