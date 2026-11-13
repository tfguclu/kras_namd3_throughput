# VMD: load one HMR replicate at 1 ns/frame
# HMR dcdfreq 250000 x 4 fs = 1 ns per saved frame -> read every frame
# Usage: run inside the replicate folder (namd_3, namd_3_2 or namd_3_3)
#        vmd -e load_kras_hmr.tcl

set psf step3_input.psf

set mol [mol new $psf type psf waitfor all]
set files [list step5_production.dcd]
for {set i 1} {$i <= 8} {incr i} { lappend files step5_production_$i.dcd }
foreach f $files {
    if {[file exists $f]} {
        mol addfile $f type dcd step 1 waitfor all molid $mol
        puts "loaded $f"
    } else {
        puts "WARNING: $f not found"
    }
}
puts "frames: [molinfo $mol get numframes] (1 ns/frame)"

# wrap: protein centered, whole residues kept together (needs unit cell in DCD)
package require pbctools
pbc wrap -molid $mol -all -compound residue -center com -centersel "protein"

# align on protein CA to frame 0
set ref [atomselect $mol "protein and name CA" frame 0]
set sel [atomselect $mol "protein and name CA"]
set all [atomselect $mol all]
for {set i 0} {$i < [molinfo $mol get numframes]} {incr i} {
    $sel frame $i; $all frame $i
    $all move [measure fit $sel $ref]
}

# representations (resolution 50)
mol delrep 0 $mol
mol representation NewCartoon 0.3 50 4.1 0; mol color ColorID 16; mol selection "protein";     mol addrep $mol
mol representation Licorice 0.3 50 50;      mol color Name;       mol selection "resname GTP"; mol addrep $mol
mol representation VDW 1.0 50;              mol color Name;       mol selection "resname MG";  mol addrep $mol
# Mg first-shell waters, re-selected every frame (rep 3)
mol representation Licorice 0.3 50 50;      mol color Name;       mol selection "same residue as (water and name OH2 and within 2.6 of resname MG)"; mol addrep $mol
mol selupdate 3 $mol 1

display projection Orthographic
axes location Off
display depthcue off
color Display Background white
