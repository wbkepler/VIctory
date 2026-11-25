//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! One clamp per execution costs the world's frames a compute dispatch, two
//! barriers, two transitions and a pipeline change between every two of
//! their thousands of indirect draws. With `VICTORY_INDIRECT_CLAMP=batch`
//! consecutive indirect draws share one dispatch ([`bounded`]): it is recorded before the first of them and
//! walks a table of jobs (offset, kind, stride, records) in the list's
//! upload memory, which the draws after it add theirs to while the list is
//! recorded. A record may join only while nothing recorded since the
//! dispatch can have written it: compute work ends the batch ([`end`]),
//! and so does a record in a range the list's draws declared they write
//! since ([`wrote`], [`wrote_anywhere`]). Indirect dispatches are clamped
//! alone, as before.
//!
//! The clamp's barriers also put a draw's shader writes before the next
//! draw's reads. A draw that joins a batch after a draw that declared
//! writes keeps that: a barrier alone is recorded before it.
//!
//! `VICTORY_INDIRECT_CLAMP`: `always` (the default) one clamp per
//! execution, `batch` as above, `never` none (diagnostics: garbage
//! arguments then hang the device).
//!
//! shortcut: batches are opt-in. They cut the world's GPU time by about a
//! third on an RTX 5090, but one world run showed a skinned character with
//! stretched triangles that per-execution clamps never showed, and what
//! the batch misses has not been found; clamping before the draws that
//! precede a record is the suspect.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.
//! shortcut: arguments the CPU wrote are clamped on the GPU too. Telling
//! them apart needs the submission's order (lists are recorded before the
//! submissions ahead of them have run, so title memory may still hold the
//! bytes from before a shader's write), which a recorded list cannot act
//! on any more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
#define RS_BATCH "UAV(u0), UAV(u1), SRV(t0)"
cbuffer Record : register(b0) { uint one_kind; uint one_stride; uint one_records; uint one_regroup; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
// A batch's jobs: their count, then (16 bytes in) per job the offset of its
// first record in `args`, its kind, stride and records.
ByteAddressBuffer jobs : register(t0);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record of `kind` at `at` in place; its cost. A dispatch whose
// console groups run as `regroup` host groups each has its X multiplied
// (`ungroup` puts it back after the execution).
uint clamped(uint kind, uint at, uint regroup) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        g.x = min(g.x * regroup, 65535u);
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
void walk(uint lane, uint base, uint kind, uint stride, uint records, uint regroup, inout uint left) {
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = base + record * stride;
        uint cost = 0;
        if (record < records) {
            cost = clamped(kind, at, regroup);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            args.Store(at, 0);
        }
    }
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    walk(lane, 0, one_kind, one_stride, one_records, max(one_regroup, 1u), left);
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
// After a regrouped dispatch's execution: each record's X as the title
// wrote it (clamped).
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void ungroup(uint lane : SV_GroupIndex) {
    for (uint record = lane; record < one_records; record += 64) {
        uint at = record * one_stride;
        args.Store(at, args.Load(at) / max(one_regroup, 1u));
    }
}
// The same for every job of a batch, in the order their executions run.
[RootSignature(RS_BATCH)]
[numthreads(64, 1, 1)]
void batch(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    uint count = jobs.Load(0);
    for (uint job = 0; job < count; job++) {
        uint4 j = jobs.Load4(16 + job * 16);
        walk(lane, j.x, j.y, j.z, j.w, 1, left);
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// The same for a batch's jobs.
    batch_root: ID3D12RootSignature,
    batch_pso: ID3D12PipelineState,
    /// Puts back the X of a regrouped dispatch's records (`root`'s layout).
    ungroup_pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let (root, pso) = compile(s!("main"))?;
    let (batch_root, batch_pso) = compile(s!("batch"))?;
    let (_, ungroup_pso) = compile(s!("ungroup"))?;
    Ok(Clamp { root, pso, batch_root, batch_pso, ungroup_pso })
}

/// The clamp's entry point `entry` as a pipeline, with its root signature.
fn compile(entry: windows::core::PCSTR) -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            entry,
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok((root, pso))
    }
}

/// How records are clamped (`VICTORY_INDIRECT_CLAMP`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// One clamp per execution (the default).
    Always,
    /// One per batch of indirect draws.
    Batch,
    /// None (diagnostics).
    Never,
}

fn mode() -> Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = mode_of(std::env::var("VICTORY_INDIRECT_CLAMP").ok().as_deref());
        if mode != Mode::Always {
            tracing::info!(?mode, "indirect arguments' clamp (VICTORY_INDIRECT_CLAMP)");
        }
        mode
    })
}

fn mode_of(value: Option<&str>) -> Mode {
    match value.map(str::trim) {
        Some(v) if v.eq_ignore_ascii_case("batch") => Mode::Batch,
        Some(v) if v.eq_ignore_ascii_case("never") => Mode::Never,
        _ => Mode::Always,
    }
}

/// Clamp dispatches recorded and the executions they bound, since the last
/// call (for the frame-rate line).
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (CLAMPS.swap(0, Relaxed), BOUNDED.swap(0, Relaxed))
}
static CLAMPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BOUNDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Why batches were opened (none open: the list's first indirect draw or
/// the first after compute work; another argument buffer; shader writes
/// anywhere; a written range over the record; the job tables' memory
/// full), then the barriers recorded alone, since the last call.
pub(crate) fn take_reasons() -> [u64; 6] {
    std::array::from_fn(|i| REASONS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}
static REASONS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
/// [`REASONS`]' entry for barriers recorded alone.
const SYNCED: usize = 5;

/// A list's batch of indirect draws under one clamp dispatch, and what its
/// draws may have written since that dispatch.
#[derive(Default)]
pub(crate) struct Batches {
    open: Option<Open>,
    /// Title ranges `[from, to)` shaders may write that were declared since
    /// the open batch's dispatch.
    written: Vec<(u64, u64)>,
    /// Shaders that write wherever they compute ran since (or more ranges
    /// than are worth keeping): no record joins the batch any more.
    anywhere: bool,
    /// A draw declared shader writes since the last barrier: the next draw
    /// that joins the batch gets one, as each had with a clamp of its own.
    unsynced: bool,
    /// Upload memory of the list the job tables come from: the CPU pointer,
    /// the host address, and the bytes used and there.
    block: (usize, u64, u64, u64),
}

/// The batch a list's indirect draws are joining.
struct Open {
    /// The argument buffer, in the indirect-argument state.
    buffer: ID3D12Resource,
    /// The job table (CPU): the count, then the jobs from 16 bytes in.
    table: usize,
    jobs: u32,
}

/// Ranges kept per batch; past them, every record starts a new batch.
const WRITTEN: usize = 16;
/// A job of a batch, and its table's header.
const JOB: u64 = 16;
/// The upload memory job tables are cut from, taken from the list.
const BLOCK: u64 = 16 << 10;

/// How an execution's records were bounded ([`bounded`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Bound {
    /// By a clamp of their own: [`Bound::done`] returns the buffer.
    Alone,
    /// By a batch's clamp, recorded for them.
    First,
    /// By the open batch's clamp, recorded earlier.
    Joined,
    /// Not at all (`VICTORY_INDIRECT_CLAMP=never`).
    Unclamped,
    /// By a clamp of their own that multiplied each dispatch's X by
    /// `regroup` (host groups per console group); [`Bound::done`] puts it
    /// back. The records are at host address `gpu`.
    Regrouped { gpu: u64, stride: u32, records: u32, regroup: u32 },
}

impl Bound {
    /// Whether a clamp was recorded: the list's compute root signature and
    /// pipeline are ours now (callers bind theirs after).
    pub(crate) fn replaced_compute(self) -> bool {
        matches!(self, Bound::Alone | Bound::First | Bound::Regrouped { .. })
    }

    /// After the execution (or instead of it). After [`Bound::Regrouped`]
    /// the list's compute root signature and pipeline are ours again.
    pub(crate) fn done(self, list: &HostList, buffer: &ID3D12Resource) {
        match self {
            Bound::Alone => done(list, buffer),
            Bound::Regrouped { gpu, stride, records, regroup } => {
                done(list, buffer);
                ungroup(list, gpu, (stride, records), regroup);
            }
            _ => {}
        }
    }
}

/// Shaders of a draw recorded in `list` may write title memory
/// `[address, address + len)`: records there join no batch opened before.
pub(crate) fn wrote(list: &HostList, address: u64, len: u64) {
    let mut b = list.indirect();
    if b.open.is_none() || len == 0 {
        return;
    }
    b.unsynced = true;
    if b.anywhere {
        return;
    }
    let range = (address, address.saturating_add(len));
    if b.written.contains(&range) {
        return;
    }
    if b.written.len() == WRITTEN {
        b.anywhere = true;
    } else {
        b.written.push(range);
    }
}

/// Shaders of a draw recorded in `list` write wherever they compute.
pub(crate) fn wrote_anywhere(list: &HostList) {
    let mut b = list.indirect();
    if b.open.is_some() {
        b.anywhere = true;
        b.unsynced = true;
    }
}

/// Ends the list's batch, before compute work is recorded (it may write the
/// arguments of draws after it) and before the list closes: the argument
/// buffer goes back to unordered access.
pub(crate) fn end(list: &HostList) {
    let open = list.indirect().open.take();
    if let Some(open) = open {
        done(list, &open.buffer);
    }
}

/// Bounds, in `list`, `records` records of `kind` (PC argument numbering)
/// `stride` bytes apart, `offset` bytes into `buffer` (whose host address
/// is `gpu`), title memory from `address` on, and leaves the buffer ready
/// to be read as arguments. Draws join the list's batch or open one;
/// `alone` (dispatches) get a clamp of their own. Dispatches whose console
/// groups run as `regroup` > 1 host groups each get one in any mode, which
/// multiplies their X ([`Bound::Regrouped`]). None if the clamp is not
/// available: the execution must be dropped.
pub(crate) fn bounded(
    list: &HostList,
    buffer: &ID3D12Resource,
    at: (u64, u64, u64),
    spec: (u32, u32, u32),
    alone: bool,
    regroup: u32,
) -> Option<Bound> {
    if regroup > 1 {
        end(list);
        let gpu = at.0 + at.1;
        let (_, stride, records) = spec;
        return bound(list, buffer, gpu, spec, regroup).then_some(Bound::Regrouped { gpu, stride, records, regroup });
    }
    bounded_as(mode(), list, buffer, at, spec, alone)
}

fn bounded_as(
    mode: Mode,
    list: &HostList,
    buffer: &ID3D12Resource,
    (gpu, offset, address): (u64, u64, u64),
    spec: (u32, u32, u32),
    alone: bool,
) -> Option<Bound> {
    use std::sync::atomic::Ordering::Relaxed;
    let (kind, stride, records) = spec;
    let fits = u32::try_from(offset)
        .ok()
        .filter(|o| o.is_multiple_of(4) && gpu.is_multiple_of(4) && stride.is_multiple_of(4));
    let offset32 = match (mode, fits) {
        (Mode::Never, _) => {
            end(list);
            return Some(Bound::Unclamped);
        }
        (Mode::Batch, Some(o)) if !alone => o,
        _ => {
            end(list);
            return bound(list, buffer, gpu + offset, spec, 1).then_some(Bound::Alone);
        }
    };
    let (c, budget) = (clamp()?, list.budget()?);
    let (from, to) = (address, address.saturating_add(u64::from(stride) * u64::from(records.max(1))));
    let mut b = list.indirect();
    // The open batch's clamp ran before whatever the list recorded since:
    // it bounds this record only if none of that can have written it.
    let opened = match &b.open {
        None => Some(0),
        Some(o) if o.buffer != *buffer => Some(1),
        _ if b.anywhere => Some(2),
        _ if b.written.iter().any(|&(start, end)| start < to && from < end) => Some(3),
        _ if b.block.2 + JOB > b.block.3 => Some(4),
        _ => None,
    };
    let joins = opened.is_none();
    if let Some(reason) = opened {
        REASONS[reason].fetch_add(1, Relaxed);
        if let Some(open) = b.open.take() {
            done(list, &open.buffer);
        }
        // Room for the table's header and a job at least.
        if b.block.2 + 2 * JOB > b.block.3 {
            let (at, data) = list.scratch(BLOCK)?;
            b.block = (data as usize, at, 0, BLOCK);
        }
        let (data, at, used, _) = b.block;
        let table = data + used as usize;
        // SAFETY: the list's own upload memory, inside the block; an open
        // list, our root signature and pipeline, a live buffer the caller's
        // list holds and list memory alive until it is reset.
        unsafe {
            (table as *mut [u32; 4]).write_volatile([0; 4]);
            let l = &list.list;
            // Earlier shader writes (the title's argument passes) first.
            l.ResourceBarrier(&[uav_barrier()]);
            l.SetComputeRootSignature(&c.batch_root);
            l.SetPipelineState(&c.batch_pso);
            l.SetComputeRootUnorderedAccessView(0, gpu);
            l.SetComputeRootUnorderedAccessView(1, budget);
            l.SetComputeRootShaderResourceView(2, at + used);
            // One group walks every job's records.
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[
                uav_barrier(),
                transition(
                    buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                ),
            ]);
        }
        b.block.2 += JOB;
        b.open = Some(Open { buffer: buffer.clone(), table, jobs: 0 });
        b.written.clear();
        b.anywhere = false;
        b.unsynced = false;
        CLAMPS.fetch_add(1, Relaxed);
    } else if b.unsynced {
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[uav_barrier()]) };
        b.unsynced = false;
        REASONS[SYNCED].fetch_add(1, Relaxed);
    }
    let at = b.block.0 + b.block.2 as usize;
    b.block.2 += JOB;
    let open = b.open.as_mut()?;
    open.jobs += 1;
    // SAFETY: the list's own upload memory, inside the block (checked
    // above); the GPU reads it when the list runs.
    unsafe {
        (at as *mut [u32; 4]).write_volatile([offset32, kind, stride, records]);
        (open.table as *mut u32).write_volatile(open.jobs);
    }
    BOUNDED.fetch_add(1, Relaxed);
    Some(if joins { Bound::Joined } else { Bound::First })
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments; dispatches
/// get their X multiplied by `regroup`. Replaces the list's compute root
/// signature and pipeline (callers bind theirs after). False if the clamp is
/// not available.
fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    regroup: u32,
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    let constants = [kind, stride, records, regroup];
    CLAMPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    BOUNDED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        // One group walks every record.
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// After a regrouped dispatch's execution, the buffer back in unordered
/// access: the X of each of `records` records `stride` bytes apart at host
/// address `gpu` divided by `regroup` again, before the title's passes
/// read them. Replaces the list's compute root signature and pipeline.
fn ungroup(list: &HostList, gpu: u64, (stride, records): (u32, u32), regroup: u32) {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return;
    };
    let l = &list.list;
    let constants = [2, stride, records, regroup];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.ungroup_pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[uav_barrier()]);
    }
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word after (zero when
    /// it runs empty). One record more follows them and must stay as
    /// written. `None` without a host GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            for _ in 0..calls {
                assert!(super::bound(&list, &buffer, gpu, spec, 1));
                super::done(&list, &buffer);
            }
            list.close();
            host.execute(&[&list]);
            for at in all - per..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()]);
            }
            let first = |i: usize| data.add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// A regrouped dispatch's records run with X times the host groups per
    /// console group (clamped to the host's limit), and get their X back
    /// after; Y and Z and the record after them stay as written.
    #[test]
    fn regrouped_dispatches_multiply_x_and_put_it_back() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let words = [3, 2, 1, 20_000, 1, 1, 7, 7, 7];
        let (buffer, data) = host.shared_buffer(words.len() as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let spec = (2, 12, 2);
        // SAFETY: our mapped buffer, written before the lists run and read
        // after they have; a live buffer.
        unsafe {
            for (at, &w) in words.iter().enumerate() {
                data.add(at).write_volatile(w);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let read = || (0..words.len()).map(|i| data.add(i).read_volatile()).collect::<Vec<_>>();
            // As the execution reads them.
            let list = host.new_list().unwrap();
            assert!(super::bound(&list, &buffer, gpu, spec, 4));
            super::done(&list, &buffer);
            list.close();
            host.execute(&[&list]);
            assert_eq!(read(), [12, 2, 1, 65_535, 1, 1, 7, 7, 7]);
            // After it.
            let list = host.new_list().unwrap();
            super::ungroup(&list, gpu, (12, 2), 4);
            list.close();
            host.execute(&[&list]);
            assert_eq!(read(), [3, 2, 1, 16_383, 1, 1, 7, 7, 7]);
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    /// Consecutive indirect draws share one clamp, which bounds all their
    /// records; a draw whose records the list's shaders may have written
    /// since that clamp gets a new one, and so does the first after compute
    /// work. A record no draw names stays as written.
    #[test]
    fn consecutive_draws_share_one_clamp() {
        use super::{Bound, Mode};
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        const RECORDS: usize = 8;
        let (buffer, data) = host.shared_buffer(RECORDS as u64 * 16).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // Title addresses of the records (any: only their order matters).
        let title = |record: u64| 0x10_0000 + record * 16;
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            // Garbage vertex counts, one instance each.
            for at in 0..RECORDS * 4 {
                data.add(at).write_volatile(if at % 4 == 1 { 1 } else { u32::MAX });
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let draw = |record: u64, records: u32| {
                super::bounded_as(Mode::Batch, &list, &buffer, (gpu, record * 16, title(record)), (0, 16, records), false)
            };
            assert_eq!(draw(0, 1), Some(Bound::First));
            assert_eq!(draw(1, 2), Some(Bound::Joined));
            // A draw's shaders may write the fourth record: its clamp ran
            // before they do.
            super::wrote(&list, title(3), 16);
            assert_eq!(draw(4, 1), Some(Bound::Joined));
            assert_eq!(draw(3, 1), Some(Bound::First));
            // Compute work ends the batch.
            super::end(&list);
            assert_eq!(draw(5, 1), Some(Bound::First));
            assert_eq!(draw(6, 1), Some(Bound::Joined));
            list.close();
            host.execute(&[&list]);
            for record in 0..RECORDS {
                let words = [0, 1].map(|i| data.add(record * 4 + i).read_volatile());
                let expected = if record < 7 { [1 << 24, 1] } else { [u32::MAX, 1] };
                assert_eq!(words, expected, "record {record}");
            }
        }
    }

    #[test]
    fn the_clamp_switch_reads_its_three_values() {
        use super::{Mode, mode_of};
        assert_eq!(mode_of(None), Mode::Always);
        assert_eq!(mode_of(Some("Always")), Mode::Always);
        assert_eq!(mode_of(Some("batch")), Mode::Batch);
        assert_eq!(mode_of(Some(" never ")), Mode::Never);
    }
}
