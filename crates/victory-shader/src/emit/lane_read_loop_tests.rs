//! A lane read inside a loop the lanes leave at different iterations.
//!
//! On the console a wave runs a loop while any of its lanes does, and
//! `v_readlane` reads the lane it names whatever EXEC holds. Lane t of
//! the program below runs the loop t times and reads lane `L`'s thread id
//! into s9 in every iteration; after the loop s9 is `L` in every lane, so
//! lane t stores t + L.
//!
//! The per-lane form ran the loop per lane: a lane that had left it was
//! not there for the later iterations' lane reads (`WaveReadLaneAt` of an
//! inactive lane), and lane 0, which skips the loop, never saw the s9 the
//! loop set (12 of 32 and 24 of 64 lanes wrong, #225). Such programs now
//! run in the console flow (`lower::CONSOLE_FLOW`) on every host. Run on
//! every hardware adapter; each wrong lane is listed.

use super::host::{Dispatch, Host, Root, View, adapters};
use crate::native::{ComputeIo, RootParam, RootType, StageIo};

/// The program for waves of `wave` lanes, one wave per group, reading
/// lane `lane`.
fn program(wave: u32, lane: u32, joined_latch: bool) -> crate::NativeProgram {
    let wave64 = wave == 64;
    let k = |n: u32| 128 + n; // inline integer constant
    let (saveexec, and_exec, or_exec) = if wave64 {
        (
            0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
            0x87FE_6A7E, // s_and_b64 exec, exec, vcc
            0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        )
    } else {
        (
            0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
            0x877E_6A7E, // s_and_b32 exec_lo, exec_lo, vcc_lo
            0x887E_0A7E, // s_or_b32 exec_lo, exec_lo, s10
        )
    };
    let mut words = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE89_0380, // s_mov_b32 s9, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D8A_0080, // v_cmp_ne_u32 vcc, 0, v0
        saveexec,
        if joined_latch {
            0xBF88_0009
        } else {
            0xBF88_0006
        }, // s_cbranch_execz end
        // loop:
        0xD760_0009,
        k(lane) << 9 | 0x100, // v_readlane_b32 s9, v0, lane
        0x4A04_0481,          // v_add_nc_u32 v2, 1, v2
        0x7D82_0102,          // v_cmp_lt_u32 vcc, v2, v0
        and_exec,
    ];
    if joined_latch {
        // Keep a second predecessor of the latch in the CFG. The latch's
        // EXEC phi feeds the loop header's loop-carried EXEC phi.
        words.extend([
            0xBF88_0003, // s_cbranch_execz end
            0xBF85_0001, // s_cbranch_scc1 latch (and_exec set SCC)
            0xBEFE_0480, // s_mov_b64 exec, 0, alternate CFG edge
            0xBF82_FFF7, // latch: s_branch loop
        ]);
    } else {
        words.push(0xBF89_FFFA); // s_cbranch_execnz loop
    }
    words.extend([
        // end:
        or_exec,
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    crate::NativeProgram {
        stage: crate::HwStage::Cs,
        wave64,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1, // one user SGPR (the UAV table)
            threads: [wave, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// Every hardware adapter's device that runs SM 6.6, by name. (Without
/// the Agility SDK, Windows 10 offers none: the test then has nothing to
/// run on, as in `split_tests`.)
fn devices() -> Vec<(String, Host)> {
    let Ok(names) = adapters() else {
        eprintln!("no DXGI; skipped");
        return Vec::new();
    };
    names
        .into_iter()
        .filter_map(|name| match Host::on(&name) {
            Ok(host) => Some((name, host)),
            Err(e) => {
                eprintln!("{name}: no SM 6.6 device ({e}); skipped");
                None
            }
        })
        .collect()
}

fn emit(native: &crate::NativeProgram, target: super::Target) -> super::HostShader {
    let c = crate::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![],
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = crate::io::raise(&p, native).unwrap();
    super::emit(&c, native, &p, &st, &io, &res, target).unwrap()
}

#[test]
fn lane_reads_in_a_loop_lanes_leave_at_different_iterations() {
    let mut wrong = Vec::new();
    let mut ran = 0;
    for (wave, lane) in [(32, 20), (64, 40)] {
        for (name, host) in devices() {
            // For the host's waves: a wave64 program runs in two halves
            // where there are no 64-lane waves.
            let cs = emit(&program(wave, lane, false), host.target());
            let (root, pso) = host
                .compute_pipeline(&cs)
                .unwrap_or_else(|e| panic!("{name}: wave{wave}: {e}"));
            let views = [View::RawUav(wave)];
            let data = vec![0; (cs.user_data_dwords() as usize).max(1)];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(data))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap()
                .remove(0);
            ran += 1;
            let bad: Vec<String> = (0..wave)
                .filter(|&t| out[t as usize] != t + lane)
                .map(|t| format!("{t}: {} (want {})", out[t as usize], t + lane))
                .collect();
            if bad.is_empty() {
                eprintln!("{name}: wave{wave} exact");
            } else {
                eprintln!(
                    "{name}: wave{wave}: {} lanes wrong: {}",
                    bad.len(),
                    bad.join(", ")
                );
                wrong.push(format!("{name} wave{wave}"));
            }
        }
    }
    if ran == 0 {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    }
    assert!(wrong.is_empty(), "wrong on {wrong:?}");
}

/// The same loop with a latch block two edges reach, so the EXEC the loop
/// header joins on its back edge is itself a join. Lane t still runs the
/// loop t times and stores t + 40.
#[test]
fn lane_read_loop_with_a_joined_exec_back_edge() {
    const WAVE: u32 = 64;
    const LANE: u32 = 40;
    let native = program(WAVE, LANE, true);
    // Emission for a host without 64-lane waves, where no device runs it.
    emit(
        &native,
        super::Target {
            lanes_min: 32,
            lanes_max: 32,
        },
    );
    let want: Vec<u32> = (LANE..LANE + WAVE).collect();
    for (name, host) in devices() {
        let mut cs = emit(&native, host.target());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut cs.dxil)
                .unwrap_or_else(|e| panic!("{name}: validation: {e}"));
        }
        let (root, pso) = host
            .compute_pipeline(&cs)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(
                        0,
                        Root::Constants(vec![0; (cs.user_data_dwords() as usize).max(1)]),
                    )],
                    views: &[View::RawUav(WAVE)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(out[0], want, "{name}");
        eprintln!("{name}: joined back edge exact");
    }
}
