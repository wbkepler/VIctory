//! End-to-end tests of descriptors whose emission needs more than one
//! binding: a buffer read by element through the resource heap (no
//! used-resource entry, so no stride; the view-stride table gives it), and
//! one of two immediate samplers chosen by uniform control flow. Synthetic
//! GFX10 programs lowered to DXIL, validated by `dxil.dll` when present,
//! and run on the host GPU and WARP.

use windows::Win32::Graphics::Direct3D12::D3D12_SAMPLER_DESC;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{
    HostShader, StaticSampler, VIEW_SKIP_SHIFT, VIEW_SLOTS, VIEW_STRIDES, lower,
    validator,
};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// User data: s0 the heap index of a structured buffer (a root constant),
/// s1 the resource heap, s2 a table with a raw UAV. Each lane stores dword
/// 1 of element `lane` to dword `lane` of the UAV.
const HEAP_ELEMENT_WAVE32: &[u32] = &[
    0x8F04_8500, // s_lshl_b32 s4, s0, 5
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0203,
    0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE030_2004,
    0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen offset:4
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a 2x1 texture, s1 a root constant, s2 a table
/// with a raw UAV. The root constant picks the immediate sampler by a
/// branch, clamping (non-zero) or wrapping (zero); the program samples the
/// texture at u = 1.25 and stores red to dword 0 of the UAV.
const CHOSEN_SAMPLER_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF40C_0202,
    0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
    0xBE86_0302, // s_mov_b32 s6, s2
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
    0xBF06_8001, // s_cmp_eq_u32 s1, 0
    0xBF85_0006, // s_cbranch_scc1 -> 0x40
    0xBE94_03FF,
    0x0000_0092, // s_mov_b32 s20, 0x92 (clamp x, y, z)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0xBF82_0004, // s_branch -> 0x50
    0xBE94_0380, // 0x40: s_mov_b32 s20, 0 (wrap)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0x7E00_02FF,
    0x3FA0_0000, // 0x50: v_mov_b32 v0, 1.25
    0x7E02_02FF,
    0x3F00_0000, // v_mov_b32 v1, 0.5
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF09C_0108,
    0x00A2_0200, // image_sample_lz v2, v[0:1], s[8:15], s[20:23] dmask:0x1 dim:2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E06_0280, // v_mov_b32 v3, 0
    0xE070_1000,
    0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 the heap index of a constant buffer (a root constant), s1
/// the resource heap, s2 a table with a raw UAV. Each lane stores dword 2
/// of the constant buffer (a scalar buffer load) to dword `lane` of the UAV.
const HEAP_CONSTANT_WAVE32: &[u32] = &[
    0x8F04_8500, // s_lshl_b32 s4, s0, 5
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0203,
    0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF420_0504,
    0xFA00_0008, // s_buffer_load_dword s20, s[8:11], 0x8
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_0214, // v_mov_b32 v1, s20
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0xE070_1000,
    0x8004_0102, // buffer_store_dword v1, v2, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a raw SRV, s2 a table with a raw UAV. Each
/// lane loads the half at byte `2 × lane` and stores it to dword `lane`.
const HALF_LOAD_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0081, // v_lshlrev_b32 v1, 1, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE028_1000,
    0x8002_0201, // buffer_load_ushort v2, v1, s[8:11], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0xE070_1000,
    0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s2 a table with a raw UAV. Each lane stores its index as the
/// half at byte `2 × lane` (two lanes share each dword).
const HALF_STORE_WAVE32: &[u32] = &[
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0081, // v_lshlrev_b32 v1, 1, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE068_1000,
    0x8004_0001, // buffer_store_short v0, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a structured SRV (12-byte elements), s2 a
/// table with a structured UAV (8-byte elements). Each lane loads dword 1
/// of SRV element `lane` and adds it atomically to dword 1 of UAV element
/// `lane`: indexed accesses the translation reads through raw handles at
/// `index * stride + offset`, as the host makes structured views raw.
const STRUCTURED_LOAD_ATOMIC_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE030_2004,
    0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen offset:4
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE0C8_2004,
    0x8004_0200, // buffer_atomic_add v2, v0, s[16:19], 0 idxen offset:4
    0xBF81_0000, // s_endpgm
];

fn param(root: RootType, user_data: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    }
}

/// An entry at offset 0 of the table in user data `user_data`.
fn entry(kind: ResourceType, dwords: u8, user_data: u8) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    }
}

fn build(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    threads: u32,
) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
        target: super::Target::ANY,
        console_cf: false,
        per_lane_only: false,
        gs: None,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// A buffer descriptor's dword 2 (its record count) read as data, beside a
/// load or store through the same descriptor: records of the structure
/// stride for structured views however the code addresses them (by index,
/// at byte offsets, by scalar loads), bytes for raw views, texels for typed
/// ones; and the right element through the raw views the host makes. The
/// cases are AlanQiang0418's regression from the review of #1762: on an
/// RTX 5080 byte-offset accesses counted 192 bytes instead of 16 records,
/// and indexed loads through structured handles of raw views read element 0
/// or 1 for strides 8 to 24.
#[test]
fn a_buffer_descriptors_record_count_reads_as_its_views_kind() {
    use Stride::{Raw, Structured, Typed};
    // 48 dwords, floats 100.0, 101.0, ...: 16 records of 12 bytes, 12
    // RGBA32F texels.
    let data: Vec<u32> = (0..48).map(|i| (100.0f32 + i as f32).to_bits()).collect();
    // v3 = 1 (index) or 12 (byte); the access leaves its value in v2.
    let load_index = [0x7E06_0281, 0xE030_2000, 0x8002_0203]; // buffer_load_dword v2, v3, s[8:11], 0 idxen
    let load_byte = [0x7E06_028C, 0xE030_1000, 0x8002_0203]; // buffer_load_dword v2, v3, s[8:11], 0 offen
    let load_scalar = [0xF420_0504, 0xFA00_000C, 0xBF8C_C07F, 0x7E04_0214]; // s_buffer_load_dword s20, s[8:11], 12; v2 = s20
    let load_typed = [0x7E06_0281, 0xE000_2000, 0x8002_0203]; // buffer_load_format_x v2, v3, s[8:11], 0 idxen
    // v2 = 42 stored to the first dword of record 1.
    let store_index = [0x7E04_02AA, 0x7E06_0281, 0xE070_2000, 0x8002_0203];
    let store_byte = [0x7E04_02AA, 0x7E06_028C, 0xE070_1000, 0x8002_0203];
    // (what, the access, the view, unordered, [record count, value]).
    type Case<'a> = (&'a str, &'a [u32], Stride, bool, [u32; 2]);
    let cases: [Case; 11] = [
        ("4-byte records by index", &load_index, Structured(4), false, [48, data[1]]),
        ("8-byte records by index", &load_index, Structured(8), false, [24, data[2]]),
        ("12-byte records by index", &load_index, Structured(12), false, [16, data[3]]),
        ("16-byte records by index", &load_index, Structured(16), false, [12, data[4]]),
        ("24-byte records by index", &load_index, Structured(24), false, [8, data[6]]),
        ("12-byte records at a byte offset", &load_byte, Structured(12), false, [16, data[3]]),
        ("12-byte records by a scalar load", &load_scalar, Structured(12), false, [16, data[3]]),
        ("raw bytes", &load_byte, Raw, false, [192, data[3]]),
        ("typed texels", &load_typed, Typed, false, [12, data[4]]),
        ("12-byte unordered records by index", &store_index, Structured(12), true, [16, 42]),
        ("12-byte unordered records at a byte offset", &store_byte, Structured(12), true, [16, 42]),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts: Vec<_> = [("hardware", false), ("warp", true)]
        .into_iter()
        .filter_map(|(label, warp)| Host::new(warp).ok().map(|h| (label, h)))
        .collect();
    assert!(!hosts.is_empty(), "neither hardware nor WARP SM 6.6 device available");
    let mut wrong = Vec::new();
    for (name, access, view, unordered, [count, value]) in cases {
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF408_0402,
            0xFA00_0020, // s_load_dwordx4 s[16:19], s[4:5], 0x20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x7E02_020A, // v_mov_b32 v1, s10 (dword 2: the record count)
        ];
        words.extend_from_slice(access);
        words.extend([
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xE070_1000,
            0x8004_0100, // buffer_store_dword v1, v0, s[16:19], 0 offen
            0xE070_1004,
            0x8004_0200, // buffer_store_dword v2, v0, s[16:19], 0 offen offset:4
            0xBF81_0000, // s_endpgm
        ]);
        let kind = if unordered { ResourceType::Uav } else { ResourceType::Srv };
        let stride = match view {
            Structured(bytes) => bytes,
            Typed => 16,
            _ => 0,
        };
        let output = UsedResource {
            table_offset_dwords: 8,
            ..entry(ResourceType::Uav, 4, 0)
        };
        let shader = build(
            &words,
            vec![param(RootType::Table, 0)],
            vec![UsedResource { stride, ..entry(kind, 4, 0) }, output],
            1,
        );
        let input = if unordered {
            View::RawUav(48)
        } else {
            View::Buffer { data: &data, stride: view }
        };
        let views = [input, View::RawUav(2)];
        for (label, host) in &hosts {
            let (root, pso) = host
                .compute_pipeline(&shader)
                .unwrap_or_else(|e| panic!("{label} {name}: compute_pipeline failed: {e}"));
            let ud = vec![0; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} {name}: dispatch failed: {e}"));
            let got = out.last().cloned().unwrap_or_default();
            if got != [count, value] {
                wrong.push(format!("{label} {name}: {got:?}, want [{count}, {value}]"));
            }
            if unordered && out[0].get(3) != Some(&value) {
                wrong.push(format!("{label} {name}: byte 12 holds {:?}, want {value}", out[0].get(3)));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_heap_buffer_read_by_element_takes_the_views_stride() {
    let shader = build(
        HEAP_ELEMENT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    assert_eq!(shader.root_srvs, [VIEW_STRIDES]);
    // 32 elements of 12 bytes: (100 + e, 200 + e, 300 + e).
    let data: Vec<u32> = (0..32).flat_map(|e| [100 + e, 200 + e, 300 + e]).collect();
    let views = [
        View::RawUav(32),
        View::Buffer {
            data: &data,
            stride: Stride::Structured(12),
        },
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                // The buffer is at heap index 1, the UAV's table at 0.
                RootType::Constants => 1,
                RootType::Table => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // User data, then the stride table: heap index 1 holds
                    // 12-byte elements.
                    roots: vec![
                        (0, Root::Constants(ud.clone())),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: vec![4, 12],
                            },
                        ),
                    ],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..32).map(|l| 200 + l).collect();
        assert_eq!(words[0], want, "{label}");
        // The view starts 8 bytes early (astrosteveo's review of #1762):
        // the stride is the entry's low bits, the skip its top ones.
        let early: Vec<u32> = [7, 7].into_iter().chain(data.iter().copied()).collect();
        let views = [
            View::RawUav(32),
            View::Buffer {
                data: &early,
                stride: Stride::Structured(12),
            },
        ];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: vec![4, 12 | 8 << VIEW_SKIP_SHIFT],
                            },
                        ),
                    ],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], want, "{label}, 8 bytes early");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// From rminzao's review of #1762: indexed loads and atomics of structured
/// buffers (strides from their used-resource entries) read and write the
/// right bytes through the raw views the host makes of structured views,
/// and declare raw handles (a structured handle on a raw view is undefined
/// in D3D12). From astrosteveo's review: also when the raw views start 4,
/// 8 or 12 bytes before the title's (a view off 16 bytes of its buffer's
/// copy), as their view-stride entries say.
#[test]
fn indexed_loads_and_atomics_of_structured_buffers_read_raw_views() {
    let structured = |kind, user_data, stride| UsedResource {
        stride,
        ..entry(kind, 4, user_data)
    };
    let shader = build(
        STRUCTURED_LOAD_ATOMIC_WAVE32,
        vec![param(RootType::Table, 0), param(RootType::Table, 2)],
        vec![
            structured(ResourceType::Srv, 0, 12),
            structured(ResourceType::Uav, 2, 8),
        ],
        32,
    );
    assert_eq!(shader.root_srvs, [VIEW_STRIDES], "the views' skips");
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The SRV's table at heap index 1 (byte 32), the UAV's at 0.
            ud[u.index as usize] = if u.slot == 0 { 32 } else { 0 };
        }
        for skip in [0u32, 4, 8, 12] {
            // 32 elements of 12 bytes, (100 + e, 200 + e, 300 + e), after
            // `skip` bytes the view does not have.
            let early = (skip / 4) as usize;
            let data: Vec<u32> = std::iter::repeat_n(0xDEAD, early)
                .chain((0..32).flat_map(|e| [100 + e, 200 + e, 300 + e]))
                .collect();
            let views = [
                View::RawUav(68),
                View::Buffer {
                    data: &data,
                    stride: Stride::Structured(12),
                },
            ];
            let words = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        // User data, then the view-stride table.
                        roots: vec![
                            (0, Root::Constants(ud.clone())),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: vec![8 | skip << VIEW_SKIP_SHIFT, 12 | skip << VIEW_SKIP_SHIFT],
                                },
                            ),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let mut want = vec![0u32; 68];
            for l in 0..32 {
                want[early + 2 * l + 1] = 200 + l as u32;
            }
            assert_eq!(words[0], want, "{label}, views {skip} bytes early");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// User data: s0 a table with a structured SRV (12-byte elements), s2 a
/// table with a raw UAV. Each lane stores dword 4 of the SRV (element 1's
/// dword 1, a scalar buffer load) to dword `lane` of the UAV.
const STRUCTURED_SCALAR_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF420_0504,
    0xFA00_0010, // s_buffer_load_dword s20, s[8:11], 0x10
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_0214, // v_mov_b32 v1, s20
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0xE070_1000,
    0x8004_0102, // buffer_store_dword v1, v2, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// From astrosteveo's review of #1762: a scalar load of a structured view
/// (a stride in its used-resource entry) whose raw view starts early reads
/// from the title's first byte, as its view-stride entry says; a raw view
/// (no stride) is read from its own first byte.
#[test]
fn scalar_loads_of_structured_views_add_the_bytes_their_views_start_early() {
    let structured = |kind, user_data, stride| UsedResource {
        stride,
        ..entry(kind, 4, user_data)
    };
    let shader = build(
        STRUCTURED_SCALAR_WAVE32,
        vec![param(RootType::Table, 0), param(RootType::Table, 2)],
        vec![structured(ResourceType::Srv, 0, 12), entry(ResourceType::Uav, 4, 2)],
        32,
    );
    assert_eq!(shader.root_srvs, [VIEW_STRIDES], "the view's skip");
    let raw = build(
        STRUCTURED_SCALAR_WAVE32,
        vec![param(RootType::Table, 0), param(RootType::Table, 2)],
        vec![entry(ResourceType::Srv, 4, 0), entry(ResourceType::Uav, 4, 2)],
        32,
    );
    assert!(raw.root_srvs.is_empty(), "a raw view starts at its first byte");
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The SRV's table at heap index 1 (byte 32), the UAV's at 0.
            ud[u.index as usize] = if u.slot == 0 { 32 } else { 0 };
        }
        for skip in [0u32, 4, 8, 12] {
            let data: Vec<u32> = std::iter::repeat_n(0xDEAD, (skip / 4) as usize)
                .chain((0..32).flat_map(|e| [100 + e, 200 + e, 300 + e]))
                .collect();
            let views = [
                View::RawUav(32),
                View::Buffer {
                    data: &data,
                    stride: Stride::Structured(12),
                },
            ];
            let words = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud.clone())),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: vec![4, 12 | skip << VIEW_SKIP_SHIFT],
                                },
                            ),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(words[0], vec![201; 32], "{label}, view {skip} bytes early");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A buffer read at a heap index from data whose slot holds no buffer view
/// (the view-stride table holds 0 there; a texture here) reads the host's
/// null buffer view instead (`emit`, "Buffer views at computed heap
/// indices"): a texture descriptor read as a buffer one is a wild address.
#[test]
fn a_heap_buffer_read_where_no_buffer_view_is_reads_zeros() {
    let texel = [[1.0f32, 2.0, 3.0, 4.0]];
    let views = [View::RawUav(32), View::Texture(1, 1, &texel), View::NullBuffers];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        // The null views lie at the end of the host's view heap, which is
        // smaller on a host that caps it: translated for each host (the
        // test host's heap for `View::NullBuffers` is sized the same way).
        super::shader::TEST_VIEW_HEAP.set(Some((host.view_heap_limit(), false)));
        let shader = build(
            HEAP_ELEMENT_WAVE32,
            vec![
                param(RootType::Constants, 0),
                param(RootType::ResourceHeap, 1),
                param(RootType::Table, 2),
            ],
            vec![entry(ResourceType::Uav, 4, 2)],
            32,
        );
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The texture at heap index 1, the UAV's table at 0.
            ud[u.index as usize] = u32::from(u.root == RootType::Constants);
        }
        let dispatch = Dispatch {
            // Heap index 1 holds no buffer view: stride 0.
            roots: vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: vec![4, 0] })],
            views: &views,
            samplers: &[],
            groups: [1, 1, 1],
        };
        let words = host
            .dispatch(&root, &pso, &dispatch)
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        // As other tests translate: for a heap of the default size.
        super::shader::TEST_VIEW_HEAP.set(None);
        assert_eq!(words[0], [0; 32], "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Constant data through the resource heap itself is read from a raw view
/// (the host's constant buffer views need an alignment the title's lack).
#[test]
fn a_constant_buffer_through_the_heap_reads_as_a_raw_buffer() {
    let shader = build(
        HEAP_CONSTANT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    assert!(shader.heap && !shader.heap_writes, "reads the heap itself, writes through a table");
    assert_eq!(shader.root_srvs, [VIEW_STRIDES], "the guard's stride table");
    let data: Vec<u32> = vec![7, 8, 9, 10];
    let views = [
        View::RawUav(32),
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                // The constants are at heap index 1, the UAV's table at 0.
                RootType::Constants => 1,
                RootType::Table => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // User data, then the stride table: heap index 1 holds a
                    // buffer view.
                    roots: vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: vec![4, 4] })],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], vec![9; 32], "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// On a host whose view heaps hold the title's views wherever a slot was
/// free (`emit`, "Heap indices"), a view heap index a shader computes is the
/// title's, from the resource heap itself or through a table: the view-slot
/// table leads it to the host's slot. Here the view of title slot 5 lies at
/// host slot 1, and that of the table's slot 0 at host slot 0.
#[test]
fn a_heap_index_goes_through_the_view_slot_table() {
    let layout = || {
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ]
    };
    let plain = build(HEAP_CONSTANT_WAVE32, layout(), vec![entry(ResourceType::Uav, 4, 2)], 32);
    super::shader::TEST_VIEW_HEAP.set(Some((super::HEAP_VIEWS, true)));
    let shader = build(HEAP_CONSTANT_WAVE32, layout(), vec![entry(ResourceType::Uav, 4, 2)], 32);
    super::shader::TEST_VIEW_HEAP.set(None);
    assert_eq!(shader.root_srvs, [VIEW_SLOTS, VIEW_STRIDES]);
    assert_ne!(shader.dxil, plain.dxil);
    assert_eq!(shader.user_data, plain.user_data, "the same root arguments");
    let data: Vec<u32> = vec![7, 8, 9, 10];
    let views = [
        View::RawUav(32),
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
    ];
    let title = 5;
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                // The constants are at the title's heap index `title`; the
                // UAV's table is at the title's index 0.
                RootType::Constants => title,
                RootType::Table => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // User data, the slot table (the title's slot 5 lies at
                    // host slot 1, its slot 0 at host slot 0; the other
                    // entries lead nowhere a view is), then the stride
                    // table: host slot 1 holds a buffer view.
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (1, Root::View { srv: true, data: vec![0, 2, 2, 2, 2, 1] }),
                        (2, Root::View { srv: true, data: vec![4, 4] }),
                    ],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], vec![9; 32], "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// `buffer_load_ushort`: the half at its byte address, zero-extended (the
/// BVH builder reads 16-bit fields this way).
#[test]
fn a_half_buffer_load_reads_the_half_at_its_address() {
    let shader = build(
        HALF_LOAD_WAVE32,
        vec![param(RootType::Table, 0), param(RootType::Table, 2)],
        vec![entry(ResourceType::Srv, 4, 0), entry(ResourceType::Uav, 4, 2)],
        32,
    );
    // 32 halves 0x8000 + 3 × lane, two per dword (low half first).
    let half = |l: u32| 0x8000 + 3 * l;
    let data: Vec<u32> = (0..16).map(|k| half(2 * k) | half(2 * k + 1) << 16).collect();
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The SRV's table at heap index 0, the UAV's at 1.
            ud[u.index as usize] = if u.slot == 0 { 0 } else { 32 };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..32).map(half).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// `buffer_store_short`: each lane's half lands at its byte address and
/// leaves the other half of the dword alone.
#[test]
fn half_buffer_stores_leave_the_other_half_alone() {
    let shader = build(
        HALF_STORE_WAVE32,
        vec![param(RootType::Table, 2)],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    let views = [View::RawUav(16)];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..16).map(|k| (2 * k) | ((2 * k + 1) << 16)).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn a_sampler_chosen_by_a_uniform_branch_samples_as_chosen() {
    let shader = build(
        CHOSEN_SAMPLER_WAVE32,
        vec![
            param(RootType::Table, 0),
            param(RootType::Constants, 1),
            param(RootType::Table, 2),
        ],
        vec![
            entry(ResourceType::Srv, 8, 0),
            entry(ResourceType::Uav, 4, 2),
        ],
        1,
    );
    assert_eq!(shader.samplers.len(), 2);
    // Red 1 at u < 0.5, 2 from there: u = 1.25 wraps to 0.25, clamps to 1.
    let texels = [[1.0f32, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]];
    let views = [View::Texture(2, 1, &texels), View::RawUav(1)];
    let samplers: [D3D12_SAMPLER_DESC; 0] = [];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (clamp, want) in [(1, 2.0f32), (0, 1.0)] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Table, 0) => 0,
                    (RootType::Constants, 1) => clamp,
                    // The UAV at heap index 1: 32 bytes in.
                    (RootType::Table, 2) => 32,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let words = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &samplers,
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(f32::from_bits(words[0][0]), want, "{label}, clamp {clamp}");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}