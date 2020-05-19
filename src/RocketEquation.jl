module RocketEquation


using DifferentialEquations
using StaticArrays
using LinearAlgebra
using MultiScaleArrays

import Gloria: onevent!, render!, update!
using Gloria:   Gloria, Window, AbstractObject, Event, Layer, Scene, iskey
using Gloria.Shapes: Vertex, Point, circle, Polygon, intersects

include("SmoothStep.jl")


const width, height = 1600, 900

function attactor(du,u,p,t)
    α, β = p
    n = length(u.nodes)
    for k in 1:n
        du.nodes[k]=zero(du.nodes[k])
        for j in 1:n
            if (k==j)
                du.nodes[k] .+= [u.nodes[k][ 3], u.nodes[k][ 4], -β*u.nodes[k][3],-β*u.nodes[k][4]]
            else
                du.nodes[k][3:4] .+= α*(u.nodes[j][1:2]-u.nodes[k][1:2])
            end
        end
    end


#    [u[ 3], u[ 4], (-u[1])*α-β*u[ 3], (-u[2])*α-β*u[ 4],
#     u[ 7], u[ 8], (-u[5])*α-β*u[ 7], (-u[6])*α-β*u[ 8],
#     u[11], u[12], (-u[9])*α-β*u[11], (-u[10])*α-β*u[12]]
end

struct Thingy{B} <: AbstractMultiScaleArrayLeaf{B}
    values::Vector{B}
end

struct PhysicsLaw{T<:AbstractMultiScaleArray,B<:Number} <: AbstractMultiScaleArrayHead{B}
    nodes::Vector{T}
    values::Vector{B}
    end_idxs::Vector{Int}
end

struct Universe{T<:AbstractMultiScaleArray,B<:Number} <: AbstractMultiScaleArray{B}
    nodes::Vector{T}
    values::Vector{B}
    end_idxs::Vector{Int}
end

Newton = construct(PhysicsLaw,
    [Thingy([-700.,-350.,0.,0.]),
    Thingy([700.,350.,0.,0.]),
    Thingy([-600.,15.,0.,10.]),
    Thingy([200.,-200.,5.,-5.])])

parameters = [1e-2,0.06]

function condition(out,u,t,integrator)
    i = 0
    n = length(u.nodes)
    for k in 1:n
        for l in (k+1):n
            i += 1;
            out[i] = sqrt(sum(abs2,u.nodes[k][1:2].-u.nodes[l][1:2]))-100
        end
    end
end

function affect!(integrator,idx)
    i = 0
    u = integrator.u
    n = length(u.nodes)
    for k in 1:n
        for l in (k+1):n
            i+=1;
            if idx == i
                x₁ = u.nodes[k][1:2]
                v₁ = u.nodes[k][3:4]
                x₂ = u.nodes[l][1:2]
                v₂ = u.nodes[l][3:4]
                # https://stackoverflow.com/a/35212639
                v₁ = (v₁ - 2/(1+1)*(dot(v₁-v₂,x₁-x₂)/sum(abs2,x₁-x₂)*(x₁-x₂)))
                v₂ = -(v₂ - 2/(1+1)*(dot(v₂-v₁,x₂-x₁)/sum(abs2,x₂-x₁)*(x₂-x₁)))

                println("Collision handeled.")

                m = (x₁+x₂)/2


                u.nodes[k][3:4] .= v₁
                u.nodes[l][3:4] .= v₂

                set_u!(integrator, u)
                println(sqrt(sum(abs2,x₁.-x₂))-100,":",v₁./v₂,":", length(u) )
                break
            end
        end
    end
end

cback = VectorContinuousCallback(condition,affect!,(x->Int(((x+1)*x)/2))(length(Newton.nodes)))


problemp = ODEProblem(attactor, Newton, (0.,Inf), parameters)


mutable struct Controls <: AbstractObject
    xb::Float64
    yb::Float64
    pause::Bool
end

mutable struct World{Integrator} <: AbstractObject
    Integ::Integrator
    xoff::Float64
    yoff::Float64
end

const world = World(
                init(problemp,BS3(); save_everystep=false, callback=cback),
                0.,0.)
const controls = Controls(0.,0.,true)
const scene = Scene(Layer([world]),
                    Layer([controls], show=false))



const window = Window("ParticleDemo", width, height, scene, fullscreen=false)

function Gloria.onevent!(C::Controls, e::Event{:mousebutton_down})
    mouse = Gloria.getmousestate()
    C.xb = mouse.x
    C.yb = mouse.y
    C.pause = true
end

function Gloria.onevent!(C::Controls, e::Event{:mousebutton_up})
    mouse = Gloria.getmousestate()
    world.xoff = mouse.x - C.xb
    world.yoff = mouse.y - C.yb
    C.pause = false
end

function Gloria.onevent!(::Controls, e::Event{:key_down})
    iskey(e, "escape") && Gloria.quit!(window, e)
end


function Gloria.update!(world::World, ::Gloria.AbstractLayer, t, dt)
    if !controls.pause
        if (world.xoff != 0) || (world.yoff != 0) #mouse vector exists
            u = world.Integ.u
            u.nodes[1][3:4] .+= [world.xoff, world.yoff]
            set_u!(world.Integ, u)

            world.xoff = 0.
            world.yoff = 0.
        end
        step!(world.Integ,dt)
    end
end

function Gloria.render!(L::Layer, obj::World, frame, fps)
    Gloria.setcolor!(window, 100, 100, 100, 255)
    Gloria.clear!(window)
    Gloria.setcolor!(window, 255, 255, 255, 255)
    i = 2;
    for x in obj.Integ.u.nodes
        i += 1;
        Gloria.render!(L,circle(Vertex(0., 0.), 50;samples=i),800+x[1],450+x[2])
    end
    Gloria.present!(window)
end

Gloria.run!(window)
wait(window)


end # module