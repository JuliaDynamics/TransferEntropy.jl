using DSP
export Amplitude, Phase, Hilbert

abstract type InstantaneousSignalProperty end

""" 
    Amplitude <: InstantaneousSignalProperty

Indicates that the instantaneous amplitudes of a signal should be used. """
struct Amplitude <: InstantaneousSignalProperty end 

""" 
    Phase <: InstantaneousSignalProperty

Indicates that the instantaneous phases of a signal should be used. """
struct Phase <: InstantaneousSignalProperty end 

"""
    Hilbert(est; 
        source::InstantaneousSignalProperty = Phase(),
        target::InstantaneousSignalProperty = Phase(),
        cond::InstantaneousSignalProperty = Phase())
    ) <: TransferEntropyEstimator

Compute transfer entropy on instantaneous phases/amplitudes of relevant signals, which are 
obtained by first applying the Hilbert transform to each signal, then extracting the 
phases/amplitudes of the resulting complex numbers[^Palus2014]. Original time series are 
thus transformed to instantaneous phase/amplitude time series. Transfer 
entropy is then estimated using the provided `est` on those phases/amplitudes (use e.g. 
[`VisitationFrequency`](@ref), or [`SymbolicPermutation`](@ref)).

!!! info
    Details on estimation of the transfer entropy (conditional mutual information) 
    following the phase/amplitude extraction step is not given in Palus (2014). Here, 
    after instantaneous phases/amplitudes have been obtained, these are treated as regular 
    time series, from which transfer entropy is then computed as usual.

See also: [`Phase`](@ref), [`Amplitude`](@ref).

[^Palus2014]: Paluš, M. (2014). Cross-scale interactions and information transfer. Entropy, 16(10), 5263-5289.
"""
struct Hilbert <: TransferEntropyEstimator
    source::InstantaneousSignalProperty
    target::InstantaneousSignalProperty
    cond::InstantaneousSignalProperty
    est
    
    function Hilbert(est::VisitationFrequency;
            source::InstantaneousSignalProperty = Phase(),
            target::InstantaneousSignalProperty = Phase(),
            cond::InstantaneousSignalProperty = Phase())
        new(source, target, cond, est)
    end
end


function transferentropy(source, target, est::Hilbert; base = 2, q = 1, 
        τT = -1, τS = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1)
        
    hil_s = DSP.hilbert(source)
    hil_t = DSP.hilbert(target)
    
    if est.source isa Phase
        s = angle.(hil_s)
    elseif est.source isa Amplitude
        s = abs.(hil_s)
    else
        error("est.source must be either Phase or Amplitude instance")
    end
    
    if est.target isa Phase
        t = angle.(hil_t)
    elseif est.target isa Amplitude
        t = abs.(hil_t)
    else
        error("est.target must be either Phase or Amplitude instance")
    end
    
    # Now, estimate transfer entropy on the phases/amplitudes with the given estimator.
    transferentropy(s, t, est.est, τT = τT, τS = τS, η𝒯 = η𝒯, dT = dT, dS = dS, d𝒯 = d𝒯)
end


function transferentropy(source, target, cond, est::Hilbert; base = 2, q = 1,
        τT = -1, τS = -1, τC = -1, η𝒯 = 1, dT = 1, dS = 1, dC = 1, d𝒯 = 1)
        
    hil_s = DSP.hilbert(source)
    hil_t = DSP.hilbert(target)
    hil_c = DSP.hilbert(cond)

    if est.source isa Phase
        s = angle.(hil_s)
    elseif est.source isa Amplitude
        s = abs.(hil_s)
    else
        error("est.source must be either Phase or Amplitude instance")
    end
    
    if est.target isa Phase
        t = angle.(hil_t)
    elseif est.target isa Amplitude
        t = abs.(hil_t)
    else
        error("est.target must be either Phase or Amplitude instance")
    end

    if est.cond isa Phase
        c = angle.(hil_c)
    elseif est.cond isa Amplitude
        c = abs.(hil_c)
    else
        error("est.cond must be either Phase or Amplitude instance")
    end
    
    transferentropy(s, t, c, est.est, 
        τT = τT, τS = τS, τC = τC, η𝒯 = η𝒯, dT = dT, dS = dS, dC = dC, d𝒯 = d𝒯)
end