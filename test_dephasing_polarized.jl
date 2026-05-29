# test_dephasing_polarized.jl
# Polarized-input variant of test_dephasing.jl: evolves |0⟩_b ⊗ |+J⟩ under
# bosonic dephasing
#
#     dρ/dt = −i[H(t), ρ] + γ · (n̂ ρ n̂ − ½ {n̂², ρ}),   n̂ = a†a,
#
# for the analytic squeeze+displacement protocol and for the GRAPE pulses
# saved by ion_GRAPE_displace_polarized.jl. Target is the noiseless final
# state of the analytic protocol started from |0⟩_b ⊗ |+J⟩.
#
# Usage: julia --project=. test_dephasing_polarized.jl

include("test_dephasing.jl")   # main(), helpers, includes ion_GRAPE_displace.jl

if abspath(PROGRAM_FILE) == @__FILE__
    pulses = [
        ("protocol",         nothing,                                          1.0,   :black),
        ("GRAPE T_frac=1",   "ion_GRAPE_displace_pol_controls_Tfrac1000.jld2", 1.0,   :purple),
        ("GRAPE T_frac=0.9", "ion_GRAPE_displace_pol_controls_Tfrac900.jld2",  0.9,   :blue),
        ("GRAPE T_frac=0.75","ion_GRAPE_displace_pol_controls_Tfrac750.jld2",  0.75,  :green),
        ("GRAPE T_frac=0.5", "ion_GRAPE_displace_pol_controls_Tfrac500.jld2",  0.5,   :orange),
        ("GRAPE T_frac=1/3", "ion_GRAPE_displace_pol_controls_Tfrac333.jld2",  1/3,   :red),
    ]
    main(; init=:polarized, pulses=pulses, save_prefix="test_dephasing_polarized")
end
