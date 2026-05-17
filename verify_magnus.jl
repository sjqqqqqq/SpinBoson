# verify_magnus.jl
# Verify the analytic claim of Eq.(24)+Eq.(28) in arXiv:2510.25870:
# truncating the Magnus expansion of H(t) (Eq.23) at second order over
# the full stroboscopic protocol [0,T=4Pτ] yields the target spin-dependent
# squeezing unitary (Eq.5/Eq.30) up to higher-order corrections.
#
#   U_trunc = exp( -i Θ̂₁[0,T] - (1/2) Θ̂₂[0,T] )
#   Θ̂₁[0,T] = ∫₀ᵀ H(t) dt
#   Θ̂₂[0,T] = ∫₀ᵀ dt₁ ∫₀^{t₁} dt₂ [H(t₁), H(t₂)]
#            = ∫₀ᵀ [H(t), Θ̂₁[0,t]] dt
#
# The integrals are evaluated segment-by-segment (composite Simpson), so the
# discontinuities of (g, Δ, ϕ) at segment boundaries are not smoothed across.

include("SpinBoson_sim.jl")

using LinearAlgebra
using Printf

function verify_magnus_truncation(; N::Int=1, nmax::Int=20, z_target::Float64=0.5,
                                    P::Int=1, ℓ::Int=1,
                                    ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                                    init::Symbol=:GHZ, nq_per_seg::Int=400)
    @assert iseven(nq_per_seg) "nq_per_seg must be even for composite Simpson."
    sb  = build_spinboson(N, nmax)
    pp  = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ, tf) = pp
    Hf  = make_H_dynamic(sb, Δ_abs, ϕ1, ϕ2, g0, τ)
    d   = length(sb.b_full)
    Hmat(t) = Hf(t, nothing).data

    @printf("=== Verify second-order Magnus truncation ===\n")
    @printf("N = %d, nmax = %d, z = %.3f, P = %d, ℓ = %d\n",
            N, nmax, z_target, P, ℓ)
    @printf("g = 2π × %.3f kHz, |Δ| = 2π × %.3f kHz\n", g0/(2π), Δ_abs/(2π))
    @printf("τ = %.6f ms, T = 4Pτ = %.6f ms, dim(H) = %d\n", τ, tf, d)
    @printf("Quadrature: composite Simpson with %d points per segment.\n",
            nq_per_seg + 1)

    # --- Compute Θ̂₁[0,T] and Θ̂₂[0,T] segment by segment -----------------
    n_segments = 4P
    A   = zeros(ComplexF64, d, d)              # running Θ̂₁[0,t]
    Θ₂  = zeros(ComplexF64, d, d)              # running Θ̂₂[0,t]
    A_seg_norms   = Float64[]
    Θ2_seg_norms  = Float64[]

    for s in 0:(n_segments - 1)
        t_start = s * τ
        t_end   = (s + 1) * τ
        nq = nq_per_seg
        ts = range(t_start, t_end, length = nq + 1)
        h  = (t_end - t_start) / nq

        # H(t_i) at each quadrature point
        Hs = [Hmat(t) for t in ts]

        # Build running A(t) = Θ̂₁[0,t] at each interior point of this segment.
        # Use composite Simpson on subintervals [t_start, ts[i]].
        # Easiest: cumulative trapezoid (smooth within a segment) — error at
        # interior points is O(h²), which is fine for the *outer* commutator
        # integral evaluated by Simpson (overall error still O(h⁴) for Θ₂
        # is overkill; trapezoid for A introduces O(h²·‖Θ₂‖) which we keep
        # negligible by using nq=400).
        As = Vector{Matrix{ComplexF64}}(undef, nq + 1)
        As[1] = copy(A)
        for i in 1:nq
            As[i+1] = As[i] + 0.5 * h * (Hs[i] + Hs[i+1])
        end

        # Outer integral ∫_{t_start}^{t_end} [H(t), A(t)] dt via Simpson 1/3.
        comms = [Hs[i] * As[i] - As[i] * Hs[i] for i in 1:nq+1]
        Θseg = comms[1] + comms[end]
        for i in 2:nq
            Θseg += (iseven(i) ? 4.0 : 2.0) * comms[i]
        end
        Θseg *= h / 3

        A   = As[end]
        Θ₂ += Θseg

        push!(A_seg_norms,  opnorm(A))
        push!(Θ2_seg_norms, opnorm(Θ₂))

        @printf("seg %2d  [%6.4f, %6.4f] ms   ‖Θ̂₁(t)‖ = %.3e   ‖Θ̂₂(t)‖ = %.3e\n",
                s + 1, t_start, t_end, opnorm(A), opnorm(Θ₂))
    end

    @printf("\nFinal:  ‖Θ̂₁[0,T]‖_op = %.3e   ‖Θ̂₂[0,T]‖_op = %.3e\n",
            opnorm(A), opnorm(Θ₂))

    # --- Build U_trunc and U_target ----------------------------------------
    M       = -1im * A - 0.5 * Θ₂                           # generator
    U_trunc = exp(M)
    U_tgt   = build_target_unitary(ζ, N, sb).data

    # Unitarity of U_trunc (will be exact since M is anti-Hermitian iff Θ₁
    # Hermitian and Θ₂ anti-Hermitian — both true analytically).
    udev = opnorm(U_trunc' * U_trunc - I)
    @printf("\nU_trunc unitarity ‖U†U − I‖ = %.2e\n", udev)

    # --- State fidelity on the protocol's input ----------------------------
    psi0       = build_initial(N, sb; init)
    psi_target = build_target(ζ, N, sb; init)
    ψf_data    = U_trunc * psi0.data
    F_state    = abs2(dot(psi_target.data, ψf_data))
    @printf("\nState fidelity  |⟨ψ_target|exp(-iΘ̂₁-½Θ̂₂)|ψ₀⟩|² = %.10f\n", F_state)

    # --- Process fidelity vs U_target on a small low-Fock subspace ---------
    n_test_max = 4
    dim_s = N + 1
    test_inds = Int[]
    for idx_b in 1:(n_test_max + 1), idx_s in 1:dim_s
        push!(test_inds, (idx_b - 1) * dim_s + idx_s)
    end
    d_V = length(test_inds)
    score      = ComplexF64(0)
    target_dev = 0.0
    for k in test_inds
        ek = zeros(ComplexF64, d); ek[k] = 1.0
        score      += dot(U_tgt * ek, U_trunc * ek)
        target_dev  = max(target_dev, abs(1.0 - norm(U_tgt * ek)^2))
    end
    F_proc = abs2(score) / d_V^2
    F_avg  = (d_V * F_proc + 1) / (d_V + 1)
    @printf("\nProcess fidelity on |n⟩_b⊗|m⟩_s, n ≤ %d  (d_V=%d):\n",
            n_test_max, d_V)
    @printf("  F_proc                  = %.10f\n", F_proc)
    @printf("  F_avg                   = %.10f\n", F_avg)
    @printf("  |score|/d_V             = %.10f\n", abs(score) / d_V)
    @printf("  arg(score)              = %+.4f rad   (global phase)\n",
            angle(score))
    @printf("  Max ‖U_target|k⟩‖² − 1  = %.2e   (target truncation leak)\n",
            target_dev)

    # --- Direct full unitary distance --------------------------------------
    # Compare U_trunc to U_target over the test subspace (mat-norm, with
    # global phase divided out).
    P_V = zeros(ComplexF64, d, d)
    for k in test_inds; P_V[k,k] = 1.0; end
    Δ_op = opnorm(P_V * (U_trunc - cis(angle(score)) * U_tgt) * P_V)
    @printf("  ‖P_V (U_trunc − e^{iφ} U_target) P_V‖_op = %.3e\n", Δ_op)

    return (; A, Θ₂, U_trunc, U_target = U_tgt,
              F_state, F_proc, F_avg, score, ζ, τ, tf)
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify_magnus_truncation(N=1, nmax=20, z_target=0.5, P=1, ℓ=1)
end
