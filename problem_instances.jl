
module Problems
export gen_matching_pennies, manh_dist, hide_and_seek, nemirovski1, nemirovski2, policeman_and_burglar_matrix


import LinearAlgebra
const LA = LinearAlgebra
using Distributions
using Random





"""
   Two test problems from Nemirovski et al. "Robust stochastic approximation approach to stochastic programming"
"""
function nemirovski1(n, α=1)
    A = zeros(n, n)
    for i in 1:n
        for j in 1:n
            A[i, j] = ((i + j - 1) / ( 2n - 1 ))^α
        end
    end
    return A
end

function nemirovski2(n, α=1)
    A = zeros(n, n)
    for i in 1:n
        for j in 1:n
            A[i, j] = ((abs(i - j) + 1) / ( 2n - 1 ))^α
        end
    end
    return A
end


"""
    From Juditski & Nemirovki tutorial. Problem Policemen and Burglar.
"""
function policeman_and_burglar_matrix(n, th=0.8; seed="false")
    if seed != "false"
        Random.seed!(parse(Int, seed))
    end
    w = abs.(randn(n))
    th = 0.8
    C = reshape(abs.([i - j for i in 1:n, j in 1:n]), (n, n))
    A = w .*(1 .- exp.(-th .* C))
    return A
end

function randunif(m, n; seed="false")
    if seed != "false"
        Random.seed!(parse(Int, seed))
    end

    A = rand(m, n)
    return A
end


"""
Problem 2.24
"""
function manh_dist(m=100, n=100)
    A = zeros(m, n)
    for i in 1:m
        for j in 1:n
            y = (i - 1) / (m - 1)
            x = (j - 1) / (n - 1)
            A[i,j] = abs(x - y)
        end
    end
    return A
end


"""
    'Hide and Seek' game
"""
function hide_and_seek(m, n, param=0.2; seed="false")
    if seed != "false"
        Random.seed!(parse(Int, seed))
    end
    A = 1.0 * reshape(rand(Bernoulli(param),  m * n), (m,n))
    return A
end


end
