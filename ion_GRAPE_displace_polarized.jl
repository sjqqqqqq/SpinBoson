# ion_GRAPE_displace_polarized.jl
# Polarized-input variant of ion_GRAPE_displace.jl: start at |0⟩_b ⊗ |+J⟩
# instead of |0⟩_b ⊗ |GHZ⟩, otherwise identical (same extended squeeze +
# displacement protocol, same time-compression sweep over T_fracs).
#
# Target  = U_extended(t_strobo + t_free) · |0⟩_b ⊗ |+J⟩
#         = D(α(Jx+Jy)) · S(ζJz) · |0,+J⟩      (numerically extracted)
#
# Outputs files prefixed `ion_GRAPE_displace_pol_*`.
#
# Usage: julia --project=. ion_GRAPE_displace_polarized.jl

include("ion_GRAPE_displace.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    sweep_T_frac([1.0, 0.9, 0.75, 0.5, 1/3];
                 N=1, nmax=20, z_target=0.5, P=1,
                 init=:polarized,
                 t_free_frac=1.0, nt=400, iter_stop=400,
                 save_prefix="ion_GRAPE_displace_pol")
end
