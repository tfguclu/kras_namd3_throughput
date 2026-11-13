# VMD script: load KRAS production trajectories at 1 ns per frame
# Usage: run from KRAS_unbound_basarim/   ->   vmd -e load_kras.tcl
#
# Frame spacing on disk:
#   HMR (namd_3*)      dcdfreq 250000 x 4 fs = 1 ns   -> stride 1
#   Reference (namd/)  dcdfreq  50000 x 2 fs = 0.1 ns -> stride 10
#   PME-4 (namd_2*)    assumed as reference   -> stride 10

set psf  step3_input.psf
set pdb  step3_input.pdb
set bench KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1

# ---- choose what to load ---------------------------------------------------
set load_hmr 1
set load_ref 1
set load_pme 0

# ---- helpers ---------------------------------------------------------------
# numeric sort on the trailing _N of step5_production_N.dcd
proc piece_sort {a b} {
    set na 0; set nb 0
    regexp {_(\d+)\.dcd$} $a -> na
    regexp {_(\d+)\.dcd$} $b -> nb
    return [expr {$na - $nb}]
}

proc load_rep {name dir pattern stride} {
    global psf
    set files [lsort -command piece_sort [glob -nocomplain -directory $dir $pattern]]
    if {[llength $files] == 0} { puts "WARNING: no DCDs in $dir matching $pattern"; return }
    set mol [mol new $psf type psf waitfor all]
    mol rename $mol $name
    foreach f $files {
        puts "  $name <- [file tail $f]"
        mol addfile $f type dcd step $stride waitfor all molid $mol
    }
    style_mol $mol
    align_mol $mol
    puts "$name: [molinfo $mol get numframes] frames (1 ns/frame)"
}

proc align_mol {mol} {
    set ref [atomselect $mol "protein and name CA" frame 0]
    set sel [atomselect $mol "protein and name CA"]
    set all [atomselect $mol all]
    set n [molinfo $mol get numframes]
    for {set i 0} {$i < $n} {incr i} {
        $sel frame $i; $all frame $i
        $all move [measure fit $sel $ref]
    }
    $ref delete; $sel delete; $all delete
}

proc style_mol {mol} {
    mol delrep 0 $mol
    # protein
    mol representation NewCartoon
    mol color ColorID 8
    mol selection "protein"
    mol addrep $mol
    # Switch I (30-40)
    mol representation NewCartoon
    mol color ColorID 1
    mol selection "protein and resid 30 to 40"
    mol addrep $mol
    # GTP
    mol representation Licorice 0.25
    mol color Name
    mol selection "resname GTP"
    mol addrep $mol
    # Mg
    mol representation VDW 0.6
    mol color ColorID 7
    mol selection "resname MG"
    mol addrep $mol
    # Mg ligands: Ser17, Thr35 side chains
    mol representation Licorice 0.25
    mol color Name
    mol selection "protein and resid 17 35 and sidechain and noh"
    mol addrep $mol
    # first-shell waters (updated every frame)
    mol representation VDW 0.4
    mol color Name
    mol selection "water and name OH2 and within 2.6 of resname MG"
    mol addrep $mol
    mol selupdate 5 $mol 1
}

# ---- HMR, 3 x 10 us --------------------------------------------------------
if {$load_hmr} {
    load_rep HMR_rep1 $bench/namd_3   "step5_production_\[0-9\]*.dcd" 1
    load_rep HMR_rep2 $bench/namd_3_2 "step5_production_\[0-9\]*.dcd" 1
    load_rep HMR_rep3 $bench/namd_3_3 "step5_production_\[0-9\]*.dcd" 1
}

# ---- Reference, 4 x 1.64 us ------------------------------------------------
if {$load_ref} {
    foreach r {1 2 3 4} {
        load_rep REF_rep$r KRAS_unbound_mg_gtp_$r/namd "step5_production_\[0-9\]*.dcd" 10
    }
}

# ---- PME order 4, 3 x 100 ns -----------------------------------------------
if {$load_pme} {
    load_rep PME4_rep1 $bench/namd_2   "step5_production_100ns_4pme*.dcd" 10
    load_rep PME4_rep2 $bench/namd_2_2 "step5_production_100ns_4pme*.dcd" 10
    load_rep PME4_rep3 $bench/namd_2_3 "step5_production_100ns_4pme*.dcd" 10
}

display projection Orthographic
color Display Background white
axes location Off
