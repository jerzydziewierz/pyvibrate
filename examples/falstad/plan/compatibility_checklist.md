# Falstad Circuit Compatibility Checklist

**Total circuits:** 279
**Compatible:** 63 (22.6%)
**Implemented:** 5

## Implemented Circuits

| Falstad File | Our Implementation | Status |
|--------------|-------------------|--------|
| `amp-invert.txt` | `opamp/inverting.py` | IMPLEMENTED |
| `filt-lopass.txt` | `filters/lowpass_rc.py` | IMPLEMENTED |
| `inductkick.txt` | `rlc/inductor_kick.py` | IMPLEMENTED |
| `lrc.txt` | `rlc/lrc_resonance.py` | IMPLEMENTED |
| `voltdivide.txt` | `basic/voltage_divider.py` | IMPLEMENTED |

## Compatible (Not Yet Implemented)

| Falstad File | Status |
|--------------|--------|
| `3way.txt` | Compatible |
| `4way.txt` | Compatible |
| `allpass1.txt` | Compatible |
| `allpass2.txt` | Compatible |
| `amp-dfdx.txt` | Compatible |
| `amp-follower.txt` | Compatible |
| `amp-noninvert.txt` | Compatible |
| `amp-sum.txt` | Compatible |
| `bandpass.txt` | Compatible |
| `besselbutter.txt` | Compatible |
| `blank.txt` | Compatible |
| `butter10lo.txt` | Compatible |
| `cap.txt` | Compatible |
| `capac.txt` | Compatible |
| `capmult.txt` | Compatible |
| `capmultcaps.txt` | Compatible |
| `capmultfreq.txt` | Compatible |
| `cappar.txt` | Compatible |
| `capseries.txt` | Compatible |
| `coupled1.txt` | Compatible |
| `coupled3.txt` | Compatible |
| `crossover.txt` | Compatible |
| `diff.txt` | Compatible |
| `filt-hipass-l.txt` | Compatible |
| `filt-hipass.txt` | Compatible |
| `filt-lopass-l.txt` | Compatible |
| `grid.txt` | Compatible |
| `grid2.txt` | Compatible |
| `gyrator.txt` | Compatible |
| `howland.txt` | Compatible |
| `impedance.txt` | Compatible |
| `indmultfreq.txt` | Compatible |
| `indmultind.txt` | Compatible |
| `indpar.txt` | Compatible |
| `indseries.txt` | Compatible |
| `induct.txt` | Compatible |
| `inductac.txt` | Compatible |
| `inductkick-snub.txt` | Compatible |
| `ladder.txt` | Compatible |
| `lrc-critical.txt` | Compatible |
| `nic-r.txt` | Compatible |
| `notch.txt` | Compatible |
| `phaseseq.txt` | Compatible |
| `phaseshiftosc.txt` | Compatible |
| `pot.txt` | Compatible |
| `potdivide.txt` | Compatible |
| `powerfactor1.txt` | Compatible |
| `powerfactor2.txt` | Compatible |
| `relaxosc.txt` | Compatible |
| `res-par.txt` | Compatible |
| `res-series.txt` | Compatible |
| `resistors.txt` | Compatible |
| `ringing.txt` | Compatible |
| `sine.txt` | Compatible |
| `thevenin.txt` | Compatible |
| `triangle.txt` | Compatible |
| `twint.txt` | Compatible |
| `wheatstone.txt` | Compatible |

## Incompatible Circuits

| Falstad File | Blocking Element | Element Name |
|--------------|------------------|--------------|
| `3-cgand.txt` | `f` | MOSFET (N-ch) |
| `3-cgor.txt` | `f` | MOSFET (N-ch) |
| `3-f211.txt` | `f` | MOSFET (N-ch) |
| `3-f220.txt` | `f` | MOSFET (N-ch) |
| `3-f221.txt` | `f` | MOSFET (N-ch) |
| `3-invert.txt` | `f` | MOSFET (N-ch) |
| `555int.txt` | `153` | Logic AND |
| `555lowduty.txt` | `165` | 555 Timer |
| `555missing.txt` | `165` | 555 Timer |
| `555monostable.txt` | `165` | 555 Timer |
| `555pulsemod.txt` | `165` | 555 Timer |
| `555saw.txt` | `t` | Transistor (NPN) |
| `555schmitt.txt` | `165` | 555 Timer |
| `555sequencer.txt` | `165` | 555 Timer |
| `555square.txt` | `165` | 555 Timer |
| `7segdecoder.txt` | `L` | Unknown element (L) |
| `amdetect.txt` | `A` | Antenna |
| `amp-diff.txt` | `p` | Potentiometer |
| `amp-fullrect.txt` | `d` | Diode |
| `amp-integ.txt` | `p` | Potentiometer |
| `amp-rect.txt` | `d` | Diode |
| `amp-schmitt.txt` | `p` | Potentiometer |
| `cc2.txt` | `179` | Triode |
| `cc2imp.txt` | `f` | MOSFET (N-ch) |
| `cc2impn.txt` | `f` | MOSFET (N-ch) |
| `cc2n.txt` | `179` | Triode |
| `ccdiff.txt` | `179` | Triode |
| `cciamp.txt` | `179` | Triode |
| `ccinductor.txt` | `179` | Triode |
| `ccint.txt` | `179` | Triode |
| `ccitov.txt` | `i` | Current source |
| `ccvccs.txt` | `179` | Triode |
| `ceamp.txt` | `t` | Transistor (NPN) |
| `classd.txt` | `f` | MOSFET (N-ch) |
| `clockedsrff.txt` | `151` | Logic NAND |
| `cmosff.txt` | `f` | MOSFET (N-ch) |
| `cmosinverter.txt` | `f` | MOSFET (N-ch) |
| `cmosinvertercap.txt` | `f` | MOSFET (N-ch) |
| `cmosinverterslow.txt` | `f` | MOSFET (N-ch) |
| `cmosmsff.txt` | `159` | Logic counter |
| `cmosnand.txt` | `f` | MOSFET (N-ch) |
| `cmosnor.txt` | `f` | MOSFET (N-ch) |
| `cmostransgate.txt` | `L` | Unknown element (L) |
| `cmosxor.txt` | `f` | MOSFET (N-ch) |
| `colpitts.txt` | `t` | Transistor (NPN) |
| `counter.txt` | `156` | Logic D-FF |
| `counter8.txt` | `156` | Logic D-FF |
| `coupled2.txt` | `p` | Potentiometer |
| `cube.txt` | `82` | Unknown element (82) |
| `currentsrc.txt` | `t` | Transistor (NPN) |
| `currentsrcelm.txt` | `i` | Current source |
| `currentsrcramp.txt` | `t` | Transistor (NPN) |
| `dac.txt` | `L` | Unknown element (L) |
| `darlington.txt` | `t` | Transistor (NPN) |
| `dcrestoration.txt` | `d` | Diode |
| `deccounter.txt` | `156` | Logic D-FF |
| `decoder.txt` | `150` | Logic inverter |
| `delayrc.txt` | `f` | MOSFET (N-ch) |
| `deltasigma.txt` | `160` | Logic DAC |
| `digcompare.txt` | `L` | Unknown element (L) |
| `digsine.txt` | `155` | Logic XOR |
| `diodeclip.txt` | `d` | Diode |
| `diodecurve.txt` | `d` | Diode |
| `diodelimit.txt` | `d` | Diode |
| `diodevar.txt` | `172` | Current conveyor |
| `divideby2.txt` | `155` | Logic XOR |
| `divideby3.txt` | `155` | Logic XOR |
| `dram.txt` | `f` | MOSFET (N-ch) |
| `dtlinverter.txt` | `M` | Memristor |
| `dtlnand.txt` | `M` | Memristor |
| `dtlnor.txt` | `t` | Transistor (NPN) |
| `eclnor.txt` | `t` | Transistor (NPN) |
| `eclosc.txt` | `t` | Transistor (NPN) |
| `edgedff.txt` | `151` | Logic NAND |
| `filt-vcvs-hipass.txt` | `p` | Potentiometer |
| `filt-vcvs-lopass.txt` | `p` | Potentiometer |
| `flashadc.txt` | `154` | Logic OR |
| `follower.txt` | `t` | Transistor (NPN) |
| `freqdouble.txt` | `158` | Logic T-FF |
| `fulladd.txt` | `154` | Logic OR |
| `fullrect.txt` | `d` | Diode |
| `fullrectf.txt` | `d` | Diode |
| `graycode.txt` | `154` | Logic OR |
| `halfadd.txt` | `154` | Logic OR |
| `hartley.txt` | `t` | Transistor (NPN) |
| `hfadc.txt` | `166` | Phase comparator |
| `inductkick-block.txt` | `d` | Diode |
| `inv-osc.txt` | `I` | Unknown element (I) |
| `invertamp.txt` | `f` | MOSFET (N-ch) |
| `itov.txt` | `i` | Current source |
| `jfetamp.txt` | `j` | JFET |
| `jfetcurrentsrc.txt` | `j` | JFET |
| `jfetfollower-nooff.txt` | `j` | JFET |
| `jfetfollower.txt` | `j` | JFET |
| `jkff.txt` | `151` | Logic NAND |
| `johnsonctr.txt` | `155` | Logic XOR |
| `leadingedge.txt` | `f` | MOSFET (N-ch) |
| `ledflasher.txt` | `163` | Logic decoder |
| `lissa.txt` | `118` | Unknown element (118) |
| `logconvert.txt` | `d` | Diode |
| `longdist.txt` | `T` | Transistor (PNP) |
| `majority.txt` | `L` | Unknown element (L) |
| `masterslaveff.txt` | `151` | Logic NAND |
| `mirror.txt` | `t` | Transistor (NPN) |
| `moscurrentramp.txt` | `f` | MOSFET (N-ch) |
| `moscurrentsrc.txt` | `f` | MOSFET (N-ch) |
| `mosfetamp.txt` | `f` | MOSFET (N-ch) |
| `mosfollower.txt` | `f` | MOSFET (N-ch) |
| `mosmirror.txt` | `f` | MOSFET (N-ch) |
| `mosswitch.txt` | `f` | MOSFET (N-ch) |
| `mr-crossbar.txt` | `m` | Unknown element (m) |
| `mr-sine.txt` | `m` | Unknown element (m) |
| `mr-sine2.txt` | `m` | Unknown element (m) |
| `mr-sine3.txt` | `m` | Unknown element (m) |
| `mr-square.txt` | `m` | Unknown element (m) |
| `mr-triangle.txt` | `m` | Unknown element (m) |
| `mr.txt` | `m` | Unknown element (m) |
| `multivib-a.txt` | `t` | Transistor (NPN) |
| `multivib-bi.txt` | `t` | Transistor (NPN) |
| `multivib-mono.txt` | `t` | Transistor (NPN) |
| `mux.txt` | `f` | MOSFET (N-ch) |
| `mux3state.txt` | `151` | Logic NAND |
| `nandff.txt` | `151` | Logic NAND |
| `nmosfet.txt` | `f` | MOSFET (N-ch) |
| `nmosinverter.txt` | `L` | Unknown element (L) |
| `nmosinverter2.txt` | `f` | MOSFET (N-ch) |
| `nmosnand.txt` | `f` | MOSFET (N-ch) |
| `norton.txt` | `i` | Current source |
| `npn.txt` | `172` | Current conveyor |
| `ohms.txt` | `172` | Current conveyor |
| `opamp-regulator.txt` | `t` | Transistor (NPN) |
| `opamp.txt` | `172` | Current conveyor |
| `opampfeedback.txt` | `172` | Current conveyor |
| `opint-current.txt` | `t` | Transistor (NPN) |
| `opint-invert-amp.txt` | `t` | Transistor (NPN) |
| `opint-slew.txt` | `t` | Transistor (NPN) |
| `opint.txt` | `t` | Transistor (NPN) |
| `peak-detect.txt` | `d` | Diode |
| `phasecomp.txt` | `161` | Logic ADC |
| `phasecompint.txt` | `155` | Logic XOR |
| `phasesplit.txt` | `t` | Transistor (NPN) |
| `pll.txt` | `158` | Logic T-FF |
| `pll2.txt` | `158` | Logic T-FF |
| `pll2a.txt` | `158` | Logic T-FF |
| `pmosfet.txt` | `172` | Current conveyor |
| `pnp.txt` | `172` | Current conveyor |
| `pushpull.txt` | `t` | Transistor (NPN) |
| `pushpullxover.txt` | `t` | Transistor (NPN) |
| `r2rladder.txt` | `160` | Logic DAC |
| `rectify.txt` | `d` | Diode |
| `relay.txt` | `178` | Op-amp (rail) |
| `relayand.txt` | `178` | Op-amp (rail) |
| `relayctr.txt` | `178` | Op-amp (rail) |
| `relayff.txt` | `178` | Op-amp (rail) |
| `relaymux.txt` | `178` | Op-amp (rail) |
| `relayor.txt` | `178` | Op-amp (rail) |
| `relaytff.txt` | `178` | Op-amp (rail) |
| `relayxor.txt` | `178` | Op-amp (rail) |
| `ringmod.txt` | `d` | Diode |
| `rossler.txt` | `d` | Diode |
| `rtlinverter.txt` | `t` | Transistor (NPN) |
| `rtlnand.txt` | `t` | Transistor (NPN) |
| `rtlnor.txt` | `t` | Transistor (NPN) |
| `samplenhold.txt` | `f` | MOSFET (N-ch) |
| `sawtooth.txt` | `d` | Diode |
| `schmitt.txt` | `t` | Transistor (NPN) |
| `scr.txt` | `177` | CCCS |
| `scractrig.txt` | `177` | CCCS |
| `sinediode.txt` | `d` | Diode |
| `spark-marx.txt` | `187` | Varactor |
| `spark-sawtooth.txt` | `187` | Varactor |
| `spikegen.txt` | `d` | Diode |
| `switchedcap.txt` | `159` | Logic counter |
| `switchfilter.txt` | `f` | MOSFET (N-ch) |
| `swtreedac.txt` | `160` | Logic DAC |
| `synccounter.txt` | `156` | Logic D-FF |
| `tdiode.txt` | `175` | VCCS |
| `tdosc.txt` | `175` | VCCS |
| `tdrelax.txt` | `175` | VCCS |
| `tesla.txt` | `T` | Transistor (PNP) |
| `tl.txt` | `171` | Square wave src |
| `tlfreq.txt` | `p` | Potentiometer |
| `tllight.txt` | `171` | Square wave src |
| `tllopass.txt` | `171` | Square wave src |
| `tlmatch1.txt` | `171` | Square wave src |
| `tlmatch2.txt` | `171` | Square wave src |
| `tlmis1.txt` | `171` | Square wave src |
| `tlmismatch.txt` | `171` | Square wave src |
| `tlstand.txt` | `171` | Square wave src |
| `tlterm.txt` | `171` | Square wave src |
| `traffic.txt` | `163` | Logic decoder |
| `trans-diffamp-common.txt` | `t` | Transistor (NPN) |
| `trans-diffamp-cursrc.txt` | `t` | Transistor (NPN) |
| `trans-diffamp.txt` | `t` | Transistor (NPN) |
| `transformer.txt` | `T` | Transistor (PNP) |
| `transformerdc.txt` | `T` | Transistor (PNP) |
| `transformerdown.txt` | `T` | Transistor (PNP) |
| `transformerup.txt` | `T` | Transistor (PNP) |
| `transswitch.txt` | `t` | Transistor (NPN) |
| `triode.txt` | `172` | Current conveyor |
| `triodeamp.txt` | `173` | Comparator |
| `ttlinverter.txt` | `t` | Transistor (NPN) |
| `ttlnand.txt` | `t` | Transistor (NPN) |
| `ttlnor.txt` | `t` | Transistor (NPN) |
| `vco.txt` | `f` | MOSFET (N-ch) |
| `voltdouble.txt` | `d` | Diode |
| `voltdouble2.txt` | `d` | Diode |
| `voltinvert.txt` | `159` | Logic counter |
| `voltquad.txt` | `d` | Diode |
| `volttriple.txt` | `d` | Diode |
| `volume.txt` | `j` | JFET |
| `xor.txt` | `151` | Logic NAND |
| `xorphasedet.txt` | `154` | Logic OR |
| `zeneriv.txt` | `z` | Zener diode |
| `zenerref.txt` | `z` | Zener diode |
| `zenerreffollow.txt` | `t` | Transistor (NPN) |