using ForwardDiff
using LinearAlgebra

# Cayley Map
cayleymap(g) = 1/sqrt(1+g'g) * [1; g]
∇cayleymap(g) = 1/sqrt(1+g'g)^3*[-g'; -g*g' + (1+g'g)*I]
∇²cayleymap(g,b) = begin
    s,v = b[1], b[2:4]
    ((g*s + (g*g'v - (1+g'g)*v))*3*g'/(1+g'g) - 
        (I*s + I*g'v + g*v' - 2*v*g'))/sqrt(1+g'g)^3 
end
ForwardDiff.jacobian(x->∇cayleymap(x)'q, g) ≈ ∇²cayleymap(g,q)
cayleyinv(q) = q[2:4]/q[1]
∇cayleyinv(q) = [-1/q[1]^2*q[2:4] 1/q[1]*I]
∇cayleyinv(q) ≈ [-v/s^2 1/s * I]

q = normalize(randn(4))
s = q[1]
v = q[2:4]
g = randn(3)

ForwardDiff.jacobian(cayleymap, g) ≈ ∇cayleymap(g)
ForwardDiff.jacobian(x->∇cayleymap(x)'q, g) ≈ ∇²cayleymap(g,q)
ForwardDiff.jacobian(cayleyinv, q) ≈ ∇cayleyinv(q)


# MRP Map
mrpmap(p) = 1/(1+p'p/4)*[1-p'p/4; p]
∇mrpmap(p) = 1/(1+p'p/4)^2 * [-p'; -0.5*p*p' + (1+p'p/4)*I]
∇²mrpmap(p,b) = begin
    s,v = b[1], b[2:4]
    ((2p*s + p*p'v - 2(1+p'p/4)v)*p' - 
         (1+p'p/4)*(2I*s + I*p'v + p*v' - v*p')) / 2(1+p'p/4)^3
end
ForwardDiff.jacobian(x->∇mrpmap(x)'q, g) ≈ ∇²mrpmap(g,q)
mrpinv(q) = 2/(1+q[1]) * q[2:4]
∇mrpinv(q) = 2*[-q[2:4] (1+q[1]) * I] / (1+q[1])^2

p = rand(3) 
mrpinv(mrpmap(p)) ≈ p
mrpmap(mrpinv(q)) ≈ q
ForwardDiff.jacobian(mrpmap, p) ≈ ∇mrpmap(p)
ForwardDiff.jacobian(x->∇mrpmap(x)'q, g) ≈ ∇²mrpmap(g,q)
ForwardDiff.jacobian(mrpinv, q) ≈ ∇mrpinv(q)


# Vec map
vecmap(c) = [sqrt(1-c'c); c]
∇vecmap(c) = [-1/sqrt(1-c'c)*c'; I]
∇²vecmap(c,b) = begin
    s,v = b[1], b[2:4]
    -(c*c' + (1-c'c)*I)*s / sqrt(1-c'c)^3
end
ForwardDiff.jacobian(x->∇vecmap(x)'q, c) ≈ ∇²vecmap(c,q)
vecinv(q) = q[2:4]
∇vecinv(q) = [zeros(3) I(3)]

if q[1] < 0
    q *= -1
end
s,v = q[1], q[2:4]
c = normalize(randn(3)) * 0.9
vecinv(vecmap(c)) ≈ c
vecmap(vecinv(q)) ≈ q
ForwardDiff.jacobian(vecmap, c) ≈ ∇vecmap(c)
ForwardDiff.jacobian(x->∇vecmap(x)'q, c) ≈ ∇²vecmap(c,q)
ForwardDiff.jacobian(vecinv, q) ≈ ∇vecinv(q)


# Exponential map
expmap(x) = [cos(norm(x)/2); x/norm(x)*sin(norm(x)/2)]
logmap(q) = 2*q[2:4] * atan(norm(q[2:4]), q[1]) / norm(q[2:4])
∇expmap(x) = [
    -x'/(2*norm(x)) * sin(norm(x)/2); 
    1/norm(x) * (I - x*x'/(x'x))*sin(norm(x)/2) + x*x'/(2x'x)*cos(norm(x)/2)
]
∇logmap(q) = begin
    s,v = q[1], q[2:4]
    ds = -2v/(v'v + s^2)
    dv = 2/norm(v) *(I - v*v'/(v'v)) * atan(norm(v), s) + 2*v*v'/(v'v) * s/(v'v + s^2)
    [ds dv]
end
∇logmap(q)

x = randn(3)
logmap(expmap(x)) ≈ x
expmap(logmap(q)) ≈ q
ForwardDiff.jacobian(expmap, c) ≈ ∇expmap(c)
# ForwardDiff.jacobian(x->∇expmap(x)'q, c) ≈ ∇²expmap(c,q)
ForwardDiff.jacobian(logmap, q) ≈ ∇logmap(q)

expmap2(x) = [cos(norm(x)); x/norm(x)*sin(norm(x))]
∇expmap2(x) = [
    -x' * sin(norm(x)); 
    (I - x*x'/(x'x))*sin(norm(x)) + x*x'/norm(x)*cos(norm(x))
] / norm(x)
∇²expmap2(x,b) = begin
    s,v = b[1],b[2:4] 
    d1 = -1/norm(x)*(I - (x*x')/(x'x))*s *sin(norm(x)) - (x*x')/(x'x) * cos(norm(x))*s
    d2 = (-(I - (x*x')/(x'x))*v*x' - (I *(x'v) + x*v' - (x*x')/(x'x) * (2*x'v))) / norm(x)^3 * sin(norm(x)) + (I-(x*x')/(x'x))*v*x'/norm(x)^2 * cos(norm(x))
    d3 = 1/norm(x)^2*(I*(x'v) + x*v' - (x*x')/(x'x)*(2x'v))*cos(norm(x)) - (x*x')/norm(x)^3 * v*x' * sin(norm(x))
    d1 + d2 + d3
end
logmap2(q) = q[2:4] * atan(norm(q[2:4]), q[1]) / norm(q[2:4])
∇logmap2(q) = begin
    s,v = q[1], q[2:4]
    ds = -v 
    dv = (v'v + s^2) * (I - v*v'/(v'v)) * atan(norm(v), s) / norm(v) + v*v'/(v'v)* s
    [ds dv]/(v'v + s^2) 
end

x = randn(3)
b = normalize(randn(4))
logmap2(expmap2(x)) ≈ x
expmap2(logmap2(q)) ≈ q
ForwardDiff.jacobian(expmap2, x) ≈ ∇expmap2(x)
ForwardDiff.jacobian(x->∇expmap2(x)'b, x) ≈ ∇²expmap2(x,b)
ForwardDiff.jacobian(logmap2, q) ≈ ∇logmap2(q)
∇²expmap2(x*1e-10,b)

t1(x) = -x/norm(x)*sin(norm(x))*s
t2(x) = 1/norm(x) * (I - (x*x')/(x'x)) * sin(norm(x))*v
t3(x) = (x*x')/(x'x) * cos(norm(x))*v

d1 = -1/norm(x)*(I - (x*x')/(x'x))*s *sin(norm(x)) - (x*x')/(x'x) * cos(norm(x))*s
d2 = (-(I - (x*x')/(x'x))*v*x' - (I *(x'v) + x*v' - (x*x')/(x'x) * (2*x'v))) / norm(x)^3 * sin(norm(x)) + (I-(x*x')/(x'x))*v*x'/norm(x)^2 * cos(norm(x))
d3 = 1/norm(x)^2*(I*(x'v) + x*v' - (x*x')/(x'x)*(2x'v))*cos(norm(x)) - (x*x')/norm(x)^3 * v*x' * sin(norm(x))
ForwardDiff.jacobian(t1, x) ≈ d1
ForwardDiff.jacobian(t2, x) ≈ d2
ForwardDiff.jacobian(t3, x) ≈ d3

# Approximate Exponential map
alog(q) = begin
    s,v = q[1],q[2:4]
    v/s*(1 - norm(v)^2/(3s^2))
end
∇alog(q) = begin
    s,v = q[1],q[2:4]
    ds = v/s^2*(3norm(v)^2/(3s^2) - 1)
    dv = (1-norm(v)^2/(3s^2))/s * I - 2/(3s^3)*v*v'
    [ds dv]
end

q0 = expmap(rand(3)*1e-6)  # small quaternion
norm(alog(q0) - logmap2(q0)) < 1e-6

ForwardDiff.jacobian(alog, q) ≈ ∇alog(q)
