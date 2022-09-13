import Pkg; Pkg.activate(@__DIR__)
using Plots
using ForwardDiff

f(x,y) = (x-0.5)^2 + y^2
c(x,y) = x^2 + y^2 - 1 
X = range(-1,1,length=101)*1.1
Y = range(-1,1,length=101)*1.1

∇f(x,y) = [20x,2y]
∇²f(x,y) = [20 0; 0 2]
L(x,y,λ,ρ) = f(x,y) + λ*c(x,y) + 0.5*ρ*c(x,y)^2
∇L(x,y,λ,ρ) = ForwardDiff.gradient(z->L(z[1],z[2],λ,ρ),[x,y])
∇²L(x,y,λ,ρ) = ForwardDiff.hessian(z->L(z[1],z[2],λ,ρ),[x,y])

ForwardDiff.gradient(z->L(z[1],z[2],λ,ρ),[x,y]) ≈ ∇L(x,y,λ,ρ)
ForwardDiff.hessian(z->L(z[1],z[2],λ,ρ),[x,y]) ≈ ∇²L(x,y,λ,ρ)

x = 1.0
y = 1.0
λ = 0.0
ρ = 10.0

## Original Problem
p = contourf(X,Y, f.(X',Y), level=20, bg=:transparent)
Cx = cos.(range(0,2pi,length=101))
Cy = sin.(range(0,2pi,length=101))
plot!(Cx,Cy,lw=3,c="red",label="")
scatter!([(1,0)], c=:green, label="")
savefig(p, joinpath(@__DIR__,"al_contour.png"))

ρ = 1
λ = 0.0

## Pure Penalty
p = contourf(X,Y, L.(X',Y,λ,ρ), levels=20, bg=:transparent)
plot!(Cx,Cy,lw=3,c="red",label="")
scatter!([(1,0)], c=:green, label="")

for i = 1:20
    dz = -∇²L(x,y,λ,ρ) \ ∇L(x,y,λ,ρ)
    @show norm(dz)
    f(x+dz[1], y+dz[2])
    c(x+dz[1], y+dz[2])
    x += dz[1]
    y += dz[2]
    if norm(dz) < 1e-6
        break
    end
end
scatter!([(x,y)], c=:red, label="")
savefig(joinpath(@__DIR__,"penalty_$ρ.png"))
ρ *= 2

## With Multiplier
ρ = 1
λ = -0.5
p = contourf(X,Y, L.(X',Y,λ,ρ), levels=20, bg=:transparent)
plot!(Cx,Cy,lw=3,c="red",label="")
scatter!([(1,0)], c=:green, label="")
for i = 1:20
    dz = -∇²L(x,y,λ,ρ) \ ∇L(x,y,λ,ρ)
    @show norm(dz)
    f(x+dz[1], y+dz[2])
    c(x+dz[1], y+dz[2])
    x += dz[1]
    y += dz[2]
    if norm(dz) < 1e-6
        break
    end
end
scatter!([(x,y)], c=:red, label="")
savefig(joinpath(@__DIR__,"multiplier.png"))

##
Xv = range(0.8,1.1,length=101)
Yv = range(-.2,0.2,length=101)
contourf(Xv,Yv, L.(Xv',Yv,λ*0,ρ), levels=20)
scatter!([(1,0)], c=:green, label="")

λ += ρ*c(x,y)
ρ *= 2 

∇L(1,0,0,ρ*1e2)
contour(X,Y, L.(X',Y,λ,ρ*10))
surface(X,Y, L.(X',Y,λ,ρ))
L(1,0,λ,1000)