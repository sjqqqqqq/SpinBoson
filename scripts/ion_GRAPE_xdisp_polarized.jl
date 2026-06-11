# ion_GRAPE_xdisp_polarized.jl
# GRAPE state transfer for the squeeze-(π/2 rotation)-x-displacement target
# starting from |0⟩_b ⊗ |+J⟩:
#
#   ψ_target = D_x · R(π/2) · S · |0⟩_b ⊗ |+J⟩
#
# where the strobo squeeze and free-segment x-displacement are propagated via
# the protocol Hamiltonians (`make_H_dynamic` and `make_H_dynamic_xdisp`),
# and the π/2 boson rotation is applied analytically between them. Compared
# to the previous polarized sweep, this places the cat lobes along x with
# squeezing perpendicular to the displacement.
#
# Outputs files prefixed `ion_GRAPE_xdisp_pol_*`.
#
# Usage: julia --project=. ion_GRAPE_xdisp_polarized.jl

include(joinpath(@__DIR__, "..", "src", "ion_GRAPE_displace.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    sweep_T_frac([1.0, 0.9, 0.75, 0.5, 1/3];
                 N=1, nmax=20, z_target=0.5, P=1,
                 init=:polarized,
                 t_free_frac=1.0, nt=400, iter_stop=400,
                 xdisp=true,
                 save_prefix="ion_GRAPE_xdisp_pol")
end
