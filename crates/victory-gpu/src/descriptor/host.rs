//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.
//!
//! A host that caps its view heaps below what translated shaders index by
//! default (NVIDIA: 1,000,000 slots) cannot keep that order for a title
//! heap larger than its own: there a view heap holds each distinct view of
//! the title's once, wherever a slot was free ([`Slots`]), and shaders find
//! a title slot's view through a table of the title's slots.

use std::collections::HashMap;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at, with_views};
use crate::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots for the title's views; view heaps have `dynamic` more.
    count: u32,
    /// The title heap's slots: `count`, or more where the host caps its
    /// view heaps and holds each distinct view of the title's once.
    title: u32,
    /// Where the title's views lie, in a view heap of a host that caps
    /// them, for a title heap with more slots than fit; elsewhere title
    /// slot `i` is host slot `i`.
    slots: Option<Slots>,
    /// The view-slot table of a title heap that fits whole in a view heap
    /// of a host that caps them: dword `i` is `i`, for the title's slots
    /// and the host's own past them (shaders translated for such a host
    /// read one whatever the heap).
    order: Option<ID3D12Resource>,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, [`CAPPED_DYNAMIC`] in a capped heap
    /// ([`capped_layout`]), else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each of the title's slots was
    /// last made from: by title slot, which is the host's except in a heap
    /// that shares its views ([`Slots`]).
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// With it the view's own copy of title memory, if it has one.
    kept: Mutex<Vec<Kept>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// Slot pairs kept for host copies' views (`crate::memory`), past the
    /// dynamic slots and below the null buffer views: the next never
    /// taken, those given back, and where they end.
    pairs: Mutex<(u32, Vec<u32>)>,
    pairs_end: u32,
    /// View heaps: the view-stride table translated shaders read
    /// (`victory_shader::emit::VIEW_STRIDES`), a dword per title slot: the
    /// byte stride of the buffer view there, 0 for none. Mapped at the
    /// address kept beside it.
    strides: Option<(ID3D12Resource, usize)>,
    /// View heaps: the view-mark table translated shaders read
    /// (`victory_shader::emit`, "Write marks"), two dwords per slot as the
    /// stride table has one: the mark bit and phase of the unordered buffer
    /// view there, `VIEW_MARK_NONE` for none. Mapped at the address kept
    /// beside it.
    marks: Option<(ID3D12Resource, usize)>,
    /// Per title slot, the title address of the buffer view it holds, as a
    /// buffer descriptor's dwords 0 and 1 (shaders reading a descriptor's
    /// address as data, `crate::memory`): the buffer, its mapping and host
    /// address, made on first use.
    addresses: std::sync::OnceLock<Option<(ID3D12Resource, usize, u64)>>,
    /// What the title slots view that title memory feeds (`crate::bindless`).
    pub(crate) viewed: std::sync::Arc<crate::bindless::Viewed>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// on top of every title slot and [`victory_shader::emit::HEAP_VIEWS`]: the
/// title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

/// The dynamic slots of a heap on a host that caps its view heaps
/// ([`capped_layout`]): each one is a slot fewer for the title's views.
/// The world on an RTX 5090 took at most ~111,500 at once while tables
/// were made afresh there; shaders now find a table's entries through the
/// view-slot table, which leaves the dynamic slots to root views.
const CAPPED_DYNAMIC: u32 = 1 << 17;

/// The slots a capped heap keeps for host copies' view pairs
/// (`crate::memory`) when the title's slots do not all fit: pairs live as
/// long as their copies, and taken from the dynamic slots instead they
/// would leave no long runs free there (root views take 256 at a time).
const CAPPED_PAIRS: u32 = 1 << 14;

/// How the `views` slots of a view heap are shared on a host that makes no
/// larger one (1,000,000 on NVIDIA): the slots for the title's views at
/// most, its dynamic slots, and the slots kept for view pairs past them.
/// The last three stay null: the null buffer views and the slot shaders'
/// heap indices are clamped to (`victory_shader::emit::null_buffer_view`),
/// which is also where a title slot without a view leads.
///
/// The dynamic slots come first: tables whose views the heap does not hold
/// in one run are made afresh in dynamic slots for every draw, and 2^16 of
/// those ran out in seconds ([`CAPPED_DYNAMIC`]).
fn capped_layout(views: u32) -> (u32, u32, u32) {
    let room = views.saturating_sub(3);
    let dynamic = CAPPED_DYNAMIC.min(room / 2);
    let pairs = CAPPED_PAIRS.min(room / 8) & !1;
    (room - dynamic - pairs, dynamic, pairs)
}

/// What the host's view heaps are like, for the translator, when the host
/// device is made (before anything is translated): `views` is the most
/// slots one may have (`victory_shader::emit::host::Host::view_heap_limit`).
pub(crate) fn host_views(views: u32) {
    if views >= victory_shader::emit::HEAP_VIEWS {
        return;
    }
    let (shared, dynamic, _) = capped_layout(views);
    tracing::info!(
        views,
        shared,
        dynamic,
        "the host caps shader-visible view heaps: shaders are translated for heaps of that size, which hold each distinct view of a title heap once; the title's slots lead to them through a table"
    );
    victory_shader::emit::set_view_heap(views, true);
}

/// Where the title's views lie in a view heap of a host that caps its view
/// heaps (NVIDIA: 1,000,000 slots; the title's heap may have more, and in
/// the world holds views in most of them): each distinct view once,
/// wherever a slot was free, however many of the title's slots hold it. A
/// title's records name its distinct views already (`super::Views`), and
/// its slots repeat them several times over, so the views fit where the
/// slots did not. Shaders look every view heap index up (`table`), those
/// they take from the heap itself and those of descriptor tables alike: a
/// table is read where the title has it, though its views lie apart.
struct Slots {
    /// The view-slot table translated shaders read
    /// (`victory_shader::emit::VIEW_SLOTS`): per title slot, the host slot
    /// of the view it holds; `null` for a slot without one. One dword for
    /// every index a shader can come to, in video memory the CPU writes
    /// where the host has that (shaders read it for every descriptor they
    /// take, and NVIDIA reads system memory far slower),
    /// mapped at the address kept beside it and never read back.
    table: (ID3D12Resource, usize),
    /// Where a title slot without a view leads: the heap's last slot,
    /// which stays null.
    null: u32,
    map: Mutex<SlotMap>,
}

/// A distinct view of the title's in a heap that shares them.
struct Shared {
    slot: u32,
    /// The title slots that hold it. With none it stays until the sync is
    /// over ([`HostHeap::settle`]): a view that only moves between title
    /// slots is not made twice.
    held: u32,
    /// Made on the host; a view of a texture the title placed itself is
    /// left for a draw ([`HostHeap::make_slot`]).
    made: bool,
    /// A buffer view, and the generation of the host copies
    /// ([`crate::shadow::generation`]) it was last known to read the right
    /// one at.
    buffer: bool,
    stamp: u64,
}

/// A record key (view index, generation) mixed into a hash.
#[derive(Default)]
struct KeyHasher(u64);

impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }
    fn write_u64(&mut self, key: u64) {
        let mixed = (self.0 ^ key).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = mixed ^ mixed >> 32;
    }
}

struct SlotMap {
    /// Each title slot's host slot: the table's dwords.
    host: Vec<u32>,
    /// The distinct views the title's slots hold, by record key.
    views: HashMap<u64, Shared, std::hash::BuildHasherDefault<KeyHasher>>,
    /// The host slots taken, a bit each; those there are; where the search
    /// for a free one starts.
    taken: Vec<u64>,
    slots: u32,
    next: u32,
    used: u32,
    /// The title slots that hold a view.
    live: u32,
    /// Views whose last title slot gave them up since the last
    /// [`HostHeap::settle`].
    idle: Vec<u64>,
    /// Title views that found no host slot free, so far.
    refused: u64,
    /// The most host slots taken at once so far (logged as it grows).
    peak: u32,
}

/// Counts one of [`stats`] (diagnostics builds).
macro_rules! count {
    ($name:ident) => {
        #[cfg(feature = "diagnostics")]
        stats::$name.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    };
}

/// Diagnostics: how the heaps' slots are used.
#[cfg(feature = "diagnostics")]
pub(crate) mod stats {
    use std::sync::atomic::AtomicU64;
    /// Microseconds in `sync_heaps`.
    pub(crate) static SYNC_US: AtomicU64 = AtomicU64::new(0);
    /// Title slots brought up to date (`set_slot`).
    pub(crate) static VISITS: AtomicU64 = AtomicU64::new(0);
    /// Dynamic slots taken now, and the most since the last log.
    pub(crate) static DYNAMIC_USED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static DYNAMIC_PEAK: AtomicU64 = AtomicU64::new(0);
    /// In heaps that share their views: dwords written to the view-slot
    /// table; views made, made again (their host resource changed, or a
    /// draw came for one left for it) and given back.
    pub(crate) static WRITES: AtomicU64 = AtomicU64::new(0);
    pub(crate) static CREATED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static REMADE: AtomicU64 = AtomicU64::new(0);
    pub(crate) static FREED: AtomicU64 = AtomicU64::new(0);
    /// Views made in dynamic slots (tables made afresh, and the like).
    pub(crate) static TABLE_VIEWS: AtomicU64 = AtomicU64::new(0);
}

impl SlotMap {
    /// A free host slot, the next at or after the last one taken.
    fn take(&mut self) -> Option<u32> {
        if self.used >= self.slots {
            return None;
        }
        let words = self.taken.len();
        let mut word = (self.next / 64) as usize;
        // The first word from `next` on; it comes round again whole.
        let mut free = !self.taken[word] & (!0u64 << (self.next % 64));
        for _ in 0..=words {
            if free != 0 {
                let slot = word as u32 * 64 + free.trailing_zeros();
                self.taken[word] |= 1 << (slot % 64);
                self.used += 1;
                self.next = (slot + 1) % self.slots;
                if self.used > self.peak && (self.used.is_power_of_two() || self.used.is_multiple_of(1 << 16)) {
                    tracing::info!(
                        used = self.used,
                        of = self.slots,
                        title_slots = self.live,
                        "host view heap: distinct title views in use"
                    );
                }
                self.peak = self.peak.max(self.used);
                return Some(slot);
            }
            word = (word + 1) % words;
            free = !self.taken[word];
        }
        None
    }

    /// Host slot `slot` is free again.
    fn give(&mut self, slot: u32) {
        self.taken[(slot / 64) as usize] &= !(1 << (slot % 64));
        self.used -= 1;
    }

    /// A title slot gave up the view of record `key`.
    fn release(&mut self, key: u64) {
        if let Some(shared) = self.views.get_mut(&key) {
            shared.held = shared.held.saturating_sub(1);
            if shared.held == 0 {
                self.idle.push(key);
            }
        }
    }
}

impl Slots {
    /// Title slot `index` now leads to host slot `slot`.
    fn write(&self, index: u32, slot: u32) {
        // SAFETY: a dword of the mapped table, which has one for each of
        // the title's slots.
        unsafe { (self.table.1 as *mut u32).add(index as usize).write_volatile(slot) };
        count!(WRITES);
    }
}

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        // The host first: making it says what its view heaps are like.
        host()?;
        Self::sized(sampler, count, victory_shader::emit::heap_views())
    }

    /// [`HostHeap::new`] on a host whose view heaps have `views` slots at
    /// most (what translated shaders index: `victory_shader::emit::heap_views`).
    fn sized(sampler: bool, count: u32, views: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // Every slot a shader can index (title slots, also past a capped
        // heap, and dynamic ones): shaders read the table as a root view,
        // which has no bounds, at indices clamped below `HEAP_VIEWS`.
        let strides = (!sampler)
            .then(|| host.upload_buffer(u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 4))
            .flatten()
            .map(|(buffer, data)| (buffer, data as usize));
        let marks = (!sampler)
            .then(|| {
                let bytes = u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 8;
                let (buffer, data) = host.upload_buffer(bytes)?;
                crate::host::name(&buffer, || "view marks".into());
                // Every slot unmarked (all ones) until a view is made there.
                // SAFETY: our mapped buffer of `bytes` bytes.
                unsafe { std::ptr::write_bytes(data, 0xFF, bytes as usize) };
                Some((buffer, data as usize))
            })
            .flatten();
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        // Translated shaders clamp their indices below `views` or
        // `HEAP_SAMPLERS`: slots up to there exist and stay null.
        let samplers = victory_shader::emit::HEAP_SAMPLERS;
        let title = count;
        let (mut slots, mut order) = (None, None);
        let (heap, count, extra) = if sampler {
            match make(count.max(samplers)) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if views < victory_shader::emit::HEAP_VIEWS {
            // A host that makes no larger view heap (NVIDIA): every heap
            // has all the slots there are, so each index a shader can come
            // to exists. Below the dynamic slots and the pairs it holds
            // the title's slots if they fit, in the title's order, and
            // else the title's distinct views.
            let (room, dynamic, _) = capped_layout(views);
            if room == 0 {
                tracing::warn!(count, views, "no host descriptor heap: no room for the title's views");
                return None;
            }
            let fits = count <= room;
            // The heap's last slot: never made, without a stride or mark.
            let null = views - 1;
            // An entry for every index a shader can come to, and for the
            // title's: the slot itself where the heap keeps the title's
            // order (a table made afresh in dynamic slots is read there),
            // else without a view at first.
            let entries = count.max(victory_shader::emit::HEAP_VIEWS);
            let bytes = u64::from(entries) * 4;
            let Some((table, data)) = host.gpu_upload_buffer(bytes).or_else(|| host.upload_buffer(bytes)) else {
                tracing::warn!(count, views, "no host descriptor heap: no view-slot table");
                return None;
            };
            crate::host::name(&table, || "view slots".into());
            for index in 0..entries {
                let slot = if fits && index < views { index } else { null };
                // SAFETY: a dword of the mapped table, which has `entries`.
                unsafe { data.cast::<u32>().add(index as usize).write_volatile(slot) };
            }
            let shared = if fits {
                order = Some(table);
                count
            } else {
                tracing::info!(count, shared = room, dynamic, "host view heap capped at {views}: each distinct view of the title's gets a host slot, as the title writes it");
                let mut taken = vec![0u64; room.div_ceil(64) as usize];
                // No slots past the last.
                if !room.is_multiple_of(64) {
                    taken[(room / 64) as usize] = !0 << (room % 64);
                }
                slots = Some(Slots {
                    table: (table, data as usize),
                    null,
                    map: Mutex::new(SlotMap {
                        host: vec![null; count as usize],
                        views: HashMap::default(),
                        taken,
                        slots: room,
                        next: 0,
                        used: 0,
                        live: 0,
                        idle: Vec::new(),
                        refused: 0,
                        peak: 0,
                    }),
                });
                room
            };
            match make(views) {
                Ok(heap) => (heap, shared, dynamic),
                Err(e) => {
                    tracing::warn!(error = %e, count, views, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make((count + DYNAMIC).max(views)) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (make(PORTABLE_VIEWS).ok()?, PORTABLE_VIEWS - MIN_DYNAMIC, MIN_DYNAMIC)
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let (gpu, total) = unsafe {
            (heap.GetGPUDescriptorHandleForHeapStart().ptr, heap.GetDesc().NumDescriptors)
        };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        // The title's slots as `made` tracks them: all of them where they
        // lead to shared views, else those the heap has.
        let tracked = if slots.is_some() { title } else { count };
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            title,
            made: Mutex::new(vec![0; tracked as usize]),
            slots,
            order,
            dynamic: extra,
            sampler,
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
            pairs: Mutex::new((count + extra, Vec::new())),
            pairs_end: match views - 2 {
                null if !sampler && null > count + extra && null < total => null - 1,
                _ => total,
            },
            strides,
            marks,
            addresses: std::sync::OnceLock::new(),
            viewed: Default::default(),
        };
        if !sampler {
            crate::bindless::register(&h.viewed);
        }
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        // What shaders read instead of a slot without a buffer view where
        // they take one at a computed index (`victory_shader::emit`, "Buffer
        // views at computed heap indices"): a null raw SRV, and UAV below it.
        // (`victory_shader::emit::null_buffer_view`.)
        let null = views - 2;
        if !sampler && null > count + extra && null < total {
            h.raw_view_at(null, None, 0, 0, false);
            h.raw_view_at(null - 1, None, 0, 0, true);
        } else if !sampler {
            tracing::warn!(count, total, "no room for the null buffer views: computed indices of slots without a buffer view read what is there");
        }
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

/// A sampler heap of our own, for the lists that set a root signature whose
/// sampler heap is indexed directly (`SAMPLER_HEAP_DIRECTLY_INDEXED`) while
/// the title binds no sampler heap: D3D12 requires one to be bound first
/// (the debug layer's id 1419), and breaking that rule removes the device on
/// NVIDIA. Slot 0 holds a point sampler, so an indexed read of it returns
/// something defined rather than whatever the memory held.
pub(crate) fn fallback_samplers() -> Option<&'static HostHeap> {
    static HEAP: std::sync::OnceLock<Option<HostHeap>> = std::sync::OnceLock::new();
    HEAP.get_or_init(|| {
        let heap = HostHeap::new(true, 1)?;
        null_sampler(heap.cpu(0));
        Some(heap)
    })
    .as_ref()
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start (extending past cursor by
        // n - 1 so contiguous runs crossing the cursor are not missed).
        let cursor = *next as usize;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        let starts = (cursor..end).chain(0..second_end);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let used = stats::DYNAMIC_USED.fetch_add(n as u64, Relaxed) + n as u64;
            stats::DYNAMIC_PEAK.fetch_max(used, Relaxed);
        }
        Some(self.count + first as u32)
    }

    /// Two consecutive slots (view heaps only) kept until
    /// [`HostHeap::free_pair`]: past the dynamic slots where the heap has
    /// room (a host copy's views, `crate::memory`, for as long as it
    /// lives), else dynamic ones.
    pub(crate) fn pair(&self) -> Option<u32> {
        if self.sampler {
            return None;
        }
        {
            let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(index) = pairs.1.pop() {
                return Some(index);
            }
            if pairs.0 + 2 <= self.pairs_end {
                pairs.0 += 2;
                return Some(pairs.0 - 2);
            }
        }
        self.dynamic(2)
    }

    /// Gives back a pair [`HostHeap::pair`] returned.
    pub(crate) fn free_pair(&self, index: u32) {
        if index >= self.count + self.dynamic {
            self.pairs.lock().unwrap_or_else(|e| e.into_inner()).1.push(index);
        } else {
            self.release(index, 2);
        }
    }

    /// The slots this heap has for the title's views (the dynamic ones
    /// follow them); fewer than the title's slots when the host caps its
    /// size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The host slot of title slot `index`, if it has one: its own, or
    /// that of the view it holds in a heap that shares its views.
    #[cfg(any(test, feature = "diagnostics"))]
    fn slot(&self, index: u32) -> Option<u32> {
        let Some(slots) = &self.slots else {
            return (index < self.count).then_some(index);
        };
        let map = slots.map.lock().unwrap_or_else(|e| e.into_inner());
        let host = *map.host.get(index as usize)?;
        (host != slots.null).then_some(host)
    }

    /// The heap index to give a shader for title slot `first` if it finds
    /// the title slots from there to `last` (a table's entries) counting
    /// on from it: the title's own, where the heap has those slots in the
    /// title's order, and where shaders look each index up in the
    /// view-slot table (a heap that shares its views: they lie apart).
    pub(crate) fn run(&self, first: u32, last: u32) -> Option<u32> {
        let slots = if self.slots.is_some() { self.title } else { self.count };
        (first <= last && last < slots).then_some(first)
    }

    /// After a sync, in a heap that shares its views: the views no title
    /// slot holds any more give their host slots back (made null: a stale
    /// index reads nothing, and what the view kept alive may go).
    pub(crate) fn settle(&self) {
        if let Some(slots) = &self.slots {
            self.settle_in(&mut slots.map.lock().unwrap_or_else(|e| e.into_inner()));
        }
    }

    fn settle_in(&self, map: &mut SlotMap) {
        for key in std::mem::take(&mut map.idle) {
            // Taken up again since, or gone already (it was given up twice).
            let Some(slot) = map.views.get(&key).filter(|s| s.held == 0).map(|s| s.slot) else {
                continue;
            };
            map.views.remove(&key);
            map.give(slot);
            self.make_null(slot);
            self.note_address(slot, None);
            count!(FREED);
        }
    }

    /// Diagnostics: logs how the title's views and the dynamic slots are
    /// used, over the last `frames` frames.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn log_stats(&self, frames: f64) {
        use std::sync::atomic::Ordering::Relaxed;
        if self.sampler {
            return;
        }
        let frames = frames.max(1.0);
        let per = |a: &std::sync::atomic::AtomicU64| a.swap(0, Relaxed) as f64 / frames;
        let (sync_ms, visits, table_views) = (
            per(&stats::SYNC_US) / 1000.0,
            per(&stats::VISITS),
            per(&stats::TABLE_VIEWS),
        );
        let used_now = stats::DYNAMIC_USED.load(Relaxed);
        let dyn_peak = stats::DYNAMIC_PEAK.swap(used_now, Relaxed);
        let Some(slots) = &self.slots else {
            tracing::info!(
                sync_ms,
                visits,
                table_views,
                dyn_used = used_now,
                dyn_peak,
                of = self.dynamic,
                "host view heap stats"
            );
            return;
        };
        let map = slots.map.lock().unwrap_or_else(|e| e.into_inner());
        tracing::info!(
            sync_ms,
            visits,
            views = map.used,
            of = map.slots,
            peak = map.peak,
            title_slots = map.live,
            shares = f64::from(map.live) / f64::from(map.used.max(1)),
            table_writes = per(&stats::WRITES),
            created = per(&stats::CREATED),
            remade = per(&stats::REMADE),
            freed = per(&stats::FREED),
            refused = map.refused,
            table_views,
            dyn_used = used_now,
            dyn_peak,
            dyn_of = self.dynamic,
            "host view heap stats (distinct host views of capacity, title slots holding one; per frame: sync_ms, visits, table_writes, created, remade, freed, table_views)"
        );
    }

    /// The view-slot table, for root SRV `VIEW_SLOTS`: every view heap of
    /// a host whose shaders read one has it.
    pub(crate) fn slots(&self) -> Option<&ID3D12Resource> {
        self.slots.as_ref().map(|s| &s.table.0).or(self.order.as_ref())
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            #[cfg(feature = "diagnostics")]
            stats::DYNAMIC_USED.fetch_sub(
                slots.iter().filter(|&&t| t).count() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        count!(TABLE_VIEWS);
        let (kept, mark, skip) = make_marked(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        self.set_stride(index, Some(view), skip);
        self.set_mark(index, mark);
    }

    /// Makes slot `index` (a dynamic one) a raw view, shader resource or
    /// unordered access (`uav`), of `bytes` of `buffer` from byte `first`
    /// (both multiples of 4), which it holds; a null one for no buffer, or
    /// for unordered access to a buffer shaders cannot write.
    pub(crate) fn raw_view_at(&self, index: u32, buffer: Option<&ID3D12Resource>, first: u64, bytes: u64, uav: bool) {
        let Some(host) = host() else { return };
        let writable = |b: &ID3D12Resource| {
            // SAFETY: a live buffer.
            unsafe { b.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
        };
        let buffer = buffer.filter(|&b| !uav || writable(b));
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw).
        let (first, elements) = match buffer {
            Some(_) => (first / 4, (bytes / 4).clamp(1, 1 << 27) as u32),
            // A null view still names its size.
            None => (0, 1),
        };
        // SAFETY: a slot of our heap; the buffer is live (or none: a null view).
        unsafe {
            if uav {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: first,
                            NumElements: elements,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateUnorderedAccessView(buffer, None, Some(&desc), self.cpu(index));
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: first,
                            NumElements: elements,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index));
            }
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.set_mark(index, None);
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
        self.set_stride(index, None, 0);
        // Unmarked by host slot: a slot given back leaves none behind.
        self.set_mark(index, None);
        // A slot of the title's views (what they view is kept by host
        // slot).
        if index < self.count {
            self.viewed.set(index, None, false);
        }
    }

    /// Enters the byte stride of `view` (0 for none or not a buffer view)
    /// for slot `index` in the view-stride table, with the bytes the host's
    /// view starts before it (`skip`, [`raw_window`]) in its top bits.
    fn set_stride(&self, index: u32, view: Option<&View>, skip: u32) {
        let stride = view.map_or(0, view_stride);
        self.write_stride(index, stride | skip << victory_shader::emit::VIEW_SKIP_SHIFT);
    }

    fn write_stride(&self, index: u32, stride: u32) {
        let Some((_, data)) = &self.strides else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        // SAFETY: a dword of the mapped table, which has one per slot below
        // the larger of the title's count and `HEAP_VIEWS`.
        unsafe { (*data as *mut u32).add(index as usize).write_volatile(stride) };
    }

    /// The view-stride table, for root SRV `VIEW_STRIDES`.
    pub(crate) fn strides(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|(buffer, _)| buffer)
    }

    /// Enters slot `index`'s view-mark entry: the mark bit and phase of the
    /// unordered buffer view there ([`crate::shadow::Shadow::mark_entry`]),
    /// none for anything else.
    pub(crate) fn set_mark(&self, index: u32, entry: Option<(u32, u32)>) {
        let Some((_, data)) = &self.marks else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        let entry = entry.unwrap_or((victory_shader::emit::VIEW_MARK_NONE, 0));
        // SAFETY: two dwords of the mapped table, which has two per slot
        // below the larger of the title's count and `HEAP_VIEWS`.
        unsafe {
            let at = (*data as *mut u32).add(2 * index as usize);
            at.write_volatile(entry.0);
            at.add(1).write_volatile(entry.1);
        }
    }

    /// The view-mark table, for the root SRV of shaders marking their writes.
    pub(crate) fn marks(&self) -> Option<&ID3D12Resource> {
        self.marks.as_ref().map(|(buffer, _)| buffer)
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Kept) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            let old = std::mem::replace(slot, resource);
            #[cfg(feature = "diagnostics")]
            if let Some((old, _)) = &old
                && !self.sampler
                && slot.as_ref().map(|s| windows::core::Interface::as_raw(&s.0)) != Some(windows::core::Interface::as_raw(old))
                // SAFETY: a live resource.
                && unsafe { old.GetDesc() }.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
            {
                use windows::core::Interface;
                let left = crate::resource::life::holders(old);
                crate::resource::life::note(old.as_raw() as usize, crate::resource::life::SLOT, u64::from(index), left);
            }
            // Title threads rewrite slots while the GPU runs a submission;
            // one of its shaders may hold the old descriptor (or read the
            // slot at an index from wrong data): what it views is freed
            // only once that work has run.
            if let Some((old, _)) = old {
                crate::host::retire(old);
            }
        }
    }

    /// The host address of the slots' view addresses (see the field), made
    /// on first use from the slots made so far.
    pub(crate) fn view_addresses(&self) -> Option<u64> {
        let made = self.addresses.get_or_init(|| {
            // By host slot. Where shaders find the title's views through
            // the view-slot table an index from data leads to any slot,
            // the null one too: an entry for every slot of the heap.
            let entries = if self.slots.is_some() || self.order.is_some() {
                // SAFETY: a live heap.
                unsafe { self.heap.GetDesc() }.NumDescriptors
            } else {
                self.count
            };
            let (buffer, data) = host()?.upload_buffer(u64::from(entries) * 8)?;
            crate::host::name(&buffer, || "view addresses".into());
            let data = data as usize;
            // No slot is made meanwhile.
            let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            let view = |key: u64| (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten();
            match &self.slots {
                Some(slots) => {
                    let map = slots.map.lock().unwrap_or_else(|e| e.into_inner());
                    for (&key, shared) in &map.views {
                        write_address(data, shared.slot as usize, view(key).as_ref());
                    }
                }
                None => {
                    for (i, &key) in made.iter().enumerate() {
                        write_address(data, i, view(key).as_ref());
                    }
                }
            }
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data, gpu))
        });
        made.as_ref().map(|m| m.2)
    }

    /// Slot `index` now holds `view` (or no view): its view address.
    fn note_address(&self, index: u32, view: Option<&View>) {
        if let Some(Some((_, data, _))) = self.addresses.get() {
            write_address(*data, index as usize, view);
        }
    }

    /// Makes slots `index` and `index + 1` (dynamic ones) a raw shader
    /// resource view and a raw unordered access view of the first `bytes`
    /// (a multiple of 4) of `buffer`, which they hold; null views for no
    /// buffer.
    pub(crate) fn raw_pair_at(&self, index: u32, buffer: Option<&ID3D12Resource>, bytes: u64) {
        let Some(host) = host() else { return };
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw);
        // a longer copy's rest reads 0 through it.
        let elements = (bytes / 4).min(1 << 27) as u32;
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: elements,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: slots of our heap; the buffer is live (or none: null views).
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&srv), self.cpu(index));
            host.device
                .CreateUnorderedAccessView(buffer, None, Some(&uav), self.cpu(index + 1));
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.keep(index + 1, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.write_stride(index + 1, 4);
        // Writes through these are title memory's, marked in its page table.
        self.set_mark(index, None);
        self.set_mark(index + 1, None);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Brings title slot `index` up to date with the record the title has
    /// at `handle` if it changed since, and returns the view recorded
    /// there. Buffer views are made only for draws (`for_draw`), which
    /// also copy their data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        let mut raw = [0u8; 12];
        let [tag, i, generation] = if crate::host::read_into(handle as *const u8, &mut raw) {
            read_header(&raw)
        } else {
            [0; 3]
        };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let placed = self.set_slot(&mut made, index, key, view.as_ref(), for_draw);
        if placed { view } else { None }
    }

    /// Makes host slot `slot` from `view`: its descriptor (a null one of
    /// the view's kind if the host cannot make it, or for a view left for
    /// later), its stride, mark and address entries, what it views that
    /// title memory feeds, and the host resource it holds.
    // shortcut: views of textures the title placed itself are made when a
    // draw reads them through a table (`later` until then): a host texture
    // (and a staging copy) for each of the ~20,000 the title has views of
    // ran the host out of memory (run b2).
    fn make_slot(&self, slot: u32, view: &View, later: bool) {
        self.note_address(slot, Some(view));
        // Shaders index the heap itself (`crate::bindless`): buffer views
        // are made when written.
        let (kept, mark, skip) = if later {
            (None, None, 0)
        } else {
            make_marked(view, self.cpu(slot))
        };
        // By host slot, as shaders read it: their heap index is the host's
        // (through the view-slot table where there is one).
        self.set_stride(slot, Some(view), skip);
        self.set_mark(slot, mark);
        if kept.is_none() && !self.sampler {
            null_for(view, self.cpu(slot));
        }
        if !self.sampler {
            self.viewed.set(slot, Some(view), kept.is_some());
        }
        self.keep(slot, kept.flatten());
    }

    /// Brings title slot `index` (its record's `key`, `view` the view it
    /// names if any) up to date in `made`, which is by title slot. Whether
    /// the slot holds a view that has a host slot.
    fn set_slot(&self, made: &mut [u64], index: u32, key: u64, view: Option<&View>, for_draw: bool) -> bool {
        count!(VISITS);
        if let Some(slots) = &self.slots {
            return self.share(slots, made, index, key, view, for_draw);
        }
        // The title's slot is the host's.
        let slot = index;
        if slot >= self.count {
            // A title slot past a heap the host caps. Shaders indexing the
            // heap itself read nothing there. That is only right while the
            // title keeps no view there, so those it writes are counted
            // (the title also writes null and stale records over its whole
            // heap, which say nothing).
            if view.is_some() {
                static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let writes = WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if writes.is_power_of_two() {
                    tracing::warn!(
                        index,
                        count = self.count,
                        for_draw,
                        writes,
                        "a title view without a host slot: shaders indexing the heap read nothing there"
                    );
                }
            }
            return false;
        }
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[slot as usize] != 0 {
                made[slot as usize] = 0;
                self.make_null(slot);
                self.note_address(slot, None);
            }
            return false;
        };
        if made[slot as usize] != key {
            let later = !for_draw && is_placed_texture(view);
            made[slot as usize] = if later { 0 } else { key };
            self.make_slot(slot, view, later);
        }
        true
    }

    /// [`HostHeap::set_slot`] in a heap that shares its views: title slot
    /// `index` leads to the host slot of the view it holds now, which is
    /// made if no other title slot holds it yet, and gives up the view it
    /// held before.
    fn share(&self, slots: &Slots, made: &mut [u64], index: u32, key: u64, view: Option<&View>, for_draw: bool) -> bool {
        let Some(&old) = made.get(index as usize) else {
            return false;
        };
        let key = if view.is_some() { key } else { 0 };
        let mut map = slots.map.lock().unwrap_or_else(|e| e.into_inner());
        if old == key {
            // The same view, by another write or for a draw.
            if let (Some(view), Some(shared)) = (view, map.views.get_mut(&key)) {
                self.freshen(shared, view, for_draw);
            }
            return key != 0;
        }
        // The new view first: one that only moves between title slots
        // keeps its host slot.
        let slot = match view {
            Some(view) => self.acquire(&mut map, key, view, for_draw),
            None => Some(slots.null),
        };
        let (slot, key) = match slot {
            Some(slot) => (slot, key),
            None => {
                // Every host slot holds a view some title slot holds.
                // Shaders indexing the heap read nothing here; the view is
                // asked for again when the title next writes its page.
                map.refused += 1;
                if map.refused.is_power_of_two() {
                    tracing::warn!(
                        index,
                        count = self.count,
                        for_draw,
                        writes = map.refused,
                        "a title view without a host slot: shaders indexing the heap read nothing there"
                    );
                }
                (slots.null, 0)
            }
        };
        made[index as usize] = key;
        if map.host[index as usize] != slot {
            map.host[index as usize] = slot;
            slots.write(index, slot);
        }
        match (old != 0, key != 0) {
            (false, true) => map.live += 1,
            (true, false) => map.live -= 1,
            _ => {}
        }
        if old != 0 {
            map.release(old);
        }
        key != 0
    }

    /// The host slot of the view of record `key`, for one more title slot
    /// that holds it: the one it has, or a free one it is made in. None
    /// with every slot taken.
    fn acquire(&self, map: &mut SlotMap, key: u64, view: &View, for_draw: bool) -> Option<u32> {
        if let Some(shared) = map.views.get_mut(&key) {
            shared.held += 1;
            self.freshen(shared, view, for_draw);
            return Some(shared.slot);
        }
        let slot = match map.take() {
            Some(slot) => slot,
            None => {
                // Those of views no title slot holds any more, first.
                self.settle_in(map);
                map.take()?
            }
        };
        let later = !for_draw && is_placed_texture(view);
        self.make_slot(slot, view, later);
        let shared = Shared {
            slot,
            held: 1,
            made: !later,
            buffer: view_stride(view) != 0,
            // After it is made: that may make its host copy.
            stamp: crate::shadow::generation(),
        };
        map.views.insert(key, shared);
        count!(CREATED);
        Some(slot)
    }

    /// A view some title slot holds already, as another takes it or a draw
    /// reads it: made now if it was left for a draw, and made again if its
    /// host resource is no longer the one `view` is made on. A view's
    /// record outlives what it was made on (a constant buffer's address
    /// names whichever title buffer lies there, a placed texture's memory
    /// gets another host texture), and the other title slots that hold it
    /// read the same host slot.
    fn freshen(&self, shared: &mut Shared, view: &View, for_draw: bool) {
        let stale = if !shared.made {
            for_draw
        } else if shared.buffer {
            // Host copies are made and released rarely: only then is the
            // view's looked up again.
            let now = crate::shadow::generation();
            let moved = shared.stamp != now && !self.reads(shared.slot, view, true);
            shared.stamp = now;
            moved
        } else {
            for_draw && is_placed_texture(view) && !self.reads(shared.slot, view, false)
        };
        if stale {
            self.make_slot(shared.slot, view, false);
            shared.made = true;
            count!(REMADE);
        }
    }

    /// Whether host slot `slot`, made from `view`, views the host resource
    /// `view` would be made on now. A buffer view the host could not make
    /// (`buffer`) is tried again when a copy may be there for it; a
    /// texture view is left as it is.
    fn reads(&self, slot: u32, view: &View, buffer: bool) -> bool {
        use windows::core::Interface;
        let now = match view {
            View::Constant(address, size) => {
                crate::shadow::find(*address, u64::from(*size)).map(|s| s.buffer.as_raw())
            }
            _ if buffer => view_copy(view).map(|(shadow, ..)| shadow.buffer.as_raw()),
            _ => match placed_texture(view) {
                Some(View::Shader(texture, _)) => {
                    crate::resource::host_resource(texture as crate::object::This).map(Interface::as_raw)
                }
                _ => None,
            },
        };
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let held = kept.get(slot as usize).and_then(Option::as_ref).map(|k| k.0.as_raw());
        held == now || (!buffer && held.is_none())
    }

    /// Makes title slots `first..last` from safely copied title bytes `bytes`
    /// (`for_draw = false`), locking `self.made` once and `VIEWS` at most once
    /// (cloning only the views of changed slots).
    /// Slots are keyed as `mirror` does, by title index, so an unchanged
    /// slot is skipped by what it was last made from.
    pub(crate) fn mirror_bytes(&self, first: u32, last: u32, step: u64, bytes: &[u8]) {
        let mut guard = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let end = last.min(guard.len() as u32);
        if first >= end {
            return;
        }
        // The changed slots are found under `VIEWS` (cloning only those)
        // and made after it is released: making a view calls back into
        // code that takes `VIEWS` itself (std mutexes do not re-enter).
        let mut changed = Vec::new();
        scan_bytes(
            &mut guard[..],
            first,
            end,
            step,
            bytes,
            |made, index| made[index as usize],
            |_, index, key, view| changed.push((index, key, view.cloned())),
        );
        for (index, key, view) in changed {
            self.set_slot(&mut guard, index, key, view.as_ref(), false);
        }
    }

    /// The view title slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let Some(slot) = self.slot(index) else {
            return (index < self.title).then(|| "null".into());
        };
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        // A view the host could not make leaves a null view, its record
        // noted all the same; it keeps no resource.
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(slot as usize).is_some_and(Option::is_some);
        if key == 0 || (!kept && !self.sampler) {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }

    /// Whether title slot `index` holds a view made on the host (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        let key = self.made.lock().unwrap_or_else(|e| e.into_inner())[index as usize];
        match &self.slots {
            Some(slots) => {
                let map = slots.map.lock().unwrap_or_else(|e| e.into_inner());
                map.views.get(&key).is_some_and(|shared| shared.made)
            }
            None => key != 0,
        }
    }
}

/// Reads the 12-byte `[tag, index, generation]` descriptor header safely from
/// copied title bytes.
fn read_header(bytes: &[u8]) -> [u32; 3] {
    [
        u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
    ]
}

/// Scans slots `first..end` from safely copied title bytes `bytes` (starting at
/// slot `first`, `step` bytes apart) against what `prior` says each was made
/// from, invoking `visit` only for changed or newly-stale slots. Acquires
/// `VIEWS` at most once, skips empty unchanged slots without consulting
/// `VIEWS`, and checks unchanged populated slots by generation only (no
/// `View::clone()`).
fn scan_bytes<M: ?Sized>(
    made: &mut M,
    first: u32,
    end: u32,
    step: u64,
    bytes: &[u8],
    prior: impl Fn(&M, u32) -> u64,
    mut visit: impl FnMut(&mut M, u32, u64, Option<&View>),
) {
    if first >= end {
        return;
    }
    let step = step as usize;
    let need = (end - 1 - first) as usize * step + 12;
    assert!(bytes.len() >= need);
    let read = |index: u32| -> [u32; 3] {
        let off = (index - first) as usize * step;
        read_header(&bytes[off..off + 12])
    };
    let mut index = first;
    while index < end && read(index)[0] != VIEW_TAG {
        if prior(made, index) != 0 {
            visit(made, index, 0, None);
        }
        index += 1;
    }
    if index < end {
        with_views(|views| {
            for index in index..end {
                let ([tag, i, generation], prev) = (read(index), prior(made, index));
                if tag != VIEW_TAG {
                    if prev != 0 {
                        visit(made, index, 0, None);
                    }
                    continue;
                }
                let key = u64::from(generation) << 32 | u64::from(i);
                if prev == key {
                    if prev != 0 && !views.matches(i, generation) {
                        visit(made, index, 0, None);
                    }
                } else if let Some(view) = views.get(i, generation) {
                    visit(made, index, key, Some(view));
                } else if prev != 0 {
                    visit(made, index, 0, None);
                }
            }
        });
    }
}

/// Pure safe wrapper around [`scan_bytes`] for unit tests.
#[cfg(test)]
pub(super) fn classify_bytes(
    made: &[u64],
    first: u32,
    end: u32,
    step: u64,
    bytes: &[u8],
) -> Vec<(u32, u64, Option<View>)> {
    let mut snapshot = made.to_vec();
    let mut updates = Vec::new();
    scan_bytes(
        &mut snapshot[..],
        first,
        end,
        step,
        bytes,
        |made, index| made[index as usize],
        |_, index, key, view| updates.push((index, key, view.cloned())),
    );
    updates
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The byte stride of a buffer view (its element size, at least 1), 0 for
/// other views: what the view-stride table holds for it.
fn view_stride(view: &View) -> u32 {
    let (unordered, words): (bool, &[u32; 10]) = match view {
        // Read as raw buffers (`make`).
        View::Constant(..) => return 4,
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            match desc[..10].try_into() {
                Ok(words) => (*unordered, words),
                Err(_) => return 0,
            }
        }
        _ => return 0,
    };
    // The view dimension: buffer (1) for shader and unordered views alike.
    if words[1] != 1 {
        return 0;
    }
    let element = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        if buffer.StructureByteStride > 0 {
            u64::from(buffer.StructureByteStride)
        } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        }
    } else {
        element_size(&srv_desc(words))
    };
    // Nonzero: shaders taking views at computed indices read the null buffer
    // view where the table holds 0 (`victory_shader::emit`).
    element.max(1) as u32
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// Writes slot `index`'s view address into the view addresses at `data`:
/// a buffer view's title address (dword 0 low, dword 1 bits 0-15 high);
/// zeros for anything else.
fn write_address(data: usize, index: usize, view: Option<&View>) {
    let address = view.and_then(buffer_range).map_or(0, |(a, _)| a);
    let words = [address as u32, (address >> 32) as u32 & 0xFFFF];
    // SAFETY: the mapped view addresses hold 8 bytes per title slot.
    unsafe { ((data + 8 * index) as *mut [u32; 2]).write_unaligned(words) };
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host copy a buffer view's bytes are in (the one its host view
/// reads, [`crate::shadow::for_view`]) and the view's title range.
pub(crate) fn view_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    let (shadow, _) = crate::shadow::for_view(address, len, view_align(view))?;
    Some((shadow, address, len))
}

/// Where the host copy a buffer view reads starts ([`copy_align`]), as
/// [`buffer_srv`] and [`buffer_uav`] ask for it, so that draws find and
/// refresh the copy the slot shows.
fn view_align(view: &View) -> u64 {
    let words: Option<(bool, &[u32; 10])> = match view {
        View::Shader(_, Some(desc)) => Some((false, desc)),
        View::Unordered(_, _, Some(desc)) => Some((true, desc)),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            desc[..10].try_into().ok().map(|w| (*unordered, w))
        }
        _ => None,
    };
    let element = u64::from(view_stride(view));
    match words {
        Some((false, w)) if w[1] == 1 => {
            let desc = srv_desc(w);
            // SAFETY: a buffer view's union member.
            copy_align(desc.Format, unsafe { desc.Anonymous.Buffer }.StructureByteStride, element)
        }
        Some((true, w)) if w[1] == 1 => {
            let desc = uav_desc(w);
            // SAFETY: a buffer view's union member.
            copy_align(desc.Format, unsafe { desc.Anonymous.Buffer }.StructureByteStride, element)
        }
        _ => element,
    }
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any, and the view's
/// own copy of title memory ([`crate::shadow::for_view`]): the slot holds
/// it, so that draws reading the view find and refresh that copy.
type Kept = Option<(ID3D12Resource, Option<std::sync::Arc<crate::shadow::Shadow>>)>;

/// Buffer views refused so far (diagnostics: the first are logged).
#[cfg(feature = "diagnostics")]
static REFUSED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Where a raw view may start in its buffer, in bytes.
const RAW_ALIGN: u64 = D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT as u64;

/// The stride of a structured buffer view the host makes raw: its
/// structure stride when it is a whole number of dwords (what translated
/// shaders read by index), else 0 (left as it is).
///
/// Translated shaders read untyped buffers as raw bytes (scalar loads at a
/// byte address, raw loads at `element * stride + offset`), and indexed
/// reads of structured buffers through raw handles too. NVIDIA addresses a
/// structured view by its own stride whatever the read (an RTX 5090 read
/// zeros at byte 20 of a 16-byte-stride view; a raw read at byte `b` of a
/// 12-byte view lands at `b / 4 * 12`), so only a raw view reads the same
/// everywhere. The view-stride table keeps the title's stride. Ported from
/// fxceless/victory-gta6 (GPL-3.0), commit 578d459c, `raw_window`.
fn raw_stride(format: DXGI_FORMAT, stride: u32) -> u32 {
    if format == DXGI_FORMAT_UNKNOWN && stride > 0 && stride.is_multiple_of(4) {
        stride
    } else {
        0
    }
}

/// `count` elements of `stride` bytes (a whole number of dwords) from byte
/// `start` (a dword) of a host copy as a raw view: its first dword, its
/// dwords, and the bytes it starts before `start`; none for too many
/// dwords for one view. A raw view starts on 16 bytes
/// (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`), a console view on any dword
/// (element 1 of the meshlet culling's 456-byte constants, of 12- and
/// 24-byte strides): the view starts on the 16 bytes below, in the same
/// copy, and translated shaders add the bytes skipped, which the
/// view-stride table keeps (`victory_shader::emit::VIEW_SKIP_SHIFT`). From
/// astrosteveo's review of #1762: a copy of the view's own bytes did not
/// see writes made through the buffer's view earlier in the same list, nor
/// that view its writes.
fn raw_window(start: u64, count: u32, stride: u32) -> Option<(u64, u32, u32)> {
    if stride == 0 || !start.is_multiple_of(4) {
        return None;
    }
    let skip = start % RAW_ALIGN;
    let dwords = (skip + u64::from(count) * u64::from(stride)) / 4;
    let dwords = u32::try_from(dwords).ok().filter(|&d| d <= 1 << 27)?;
    Some(((start - skip) / 4, dwords, skip as u32))
}

/// Where the host copy of a buffer view's bytes must start: on a dword if
/// the host makes the view raw ([`raw_stride`], [`raw_window`]), else on
/// one of its elements (`element`). [`crate::shadow::for_view`] makes a
/// copy of the view's own bytes when its buffer's copy does not start it
/// there.
fn copy_align(format: DXGI_FORMAT, stride: u32, element: u64) -> u64 {
    if raw_stride(format, stride) > 0 { 4 } else { element }
}

/// Logs a structured view the host leaves structured (a stride that is not
/// a whole number of dwords, or too many dwords for one raw view): raw
/// reads of it may read the wrong bytes on NVIDIA.
fn left_structured(format: DXGI_FORMAT, stride: u32, count: u32) {
    if format == DXGI_FORMAT_UNKNOWN && stride > 0 {
        once!(stride, count, "structured buffer view left structured on the host");
    }
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if element == 0 {
        once!(len, "buffer view of elements without a size");
        return None;
    }
    // The copy its own bytes are in, its first element counted from there
    // (a structured view made raw: on a dword of it).
    let start = address + buffer.FirstElement * element;
    let stride = buffer.StructureByteStride;
    let align = copy_align(desc.Format, stride, element);
    let Some((shadow, own)) =
        crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, align)
    else {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), len, element, count = buffer.NumElements, "refused buffer view: no host copy");
        }
        return None;
    };
    let offset = shadow.offset(start);
    let mut skip = 0;
    let element = match raw_window(offset, buffer.NumElements, raw_stride(desc.Format, stride)) {
        Some((first, count, early)) => {
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            buffer.FirstElement = first;
            buffer.NumElements = count;
            buffer.StructureByteStride = 0;
            buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            skip = early;
            4
        }
        None => {
            left_structured(desc.Format, stride, buffer.NumElements);
            buffer.FirstElement = offset / element;
            element
        }
    };
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), first, count, element, "refused buffer view: empty or past its copy");
        }
        once!(first, count, element, "buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    SKIP.with(|s| s.set(skip));
    Some(Some((shadow.buffer.clone(), own.then_some(shadow))))
}

/// Whether `count` elements of `element` bytes from element `first` lie in
/// `buffer`. The console takes a view that is empty or runs past its
/// memory; the host removes the device for one.
fn in_buffer(buffer: &ID3D12Resource, first: u64, count: u32, element: u64) -> bool {
    // SAFETY: a live buffer.
    within(unsafe { buffer.GetDesc() }.Width, first, count, element)
}

fn within(width: u64, first: u64, count: u32, element: u64) -> bool {
    count > 0 && first.saturating_add(u64::from(count)).saturating_mul(element) <= width
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    if element == 0 {
        once!(len, "unordered buffer view of elements without a size");
        return None;
    }
    let start = address + buffer.FirstElement * element;
    let stride = buffer.StructureByteStride;
    let align = copy_align(desc.Format, stride, element);
    let (shadow, own) = crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, align)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    let offset = shadow.offset(start);
    buffer.CounterOffsetInBytes = 0;
    let mut skip = 0;
    let element = match raw_window(offset, buffer.NumElements, raw_stride(desc.Format, stride)) {
        Some((first, count, early)) => {
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            buffer.FirstElement = first;
            buffer.NumElements = count;
            buffer.StructureByteStride = 0;
            buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            skip = early;
            4
        }
        None => {
            left_structured(desc.Format, stride, buffer.NumElements);
            buffer.FirstElement = offset / element;
            element
        }
    };
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        once!(first, count, element, "unordered buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    // Shaders mark bytes counted from the host view's first byte.
    MARK.with(|m| m.set(shadow.mark_entry(start - u64::from(skip))));
    SKIP.with(|s| s.set(skip));
    Some(Some((shadow.buffer.clone(), own.then_some(shadow))))
}

thread_local! {
    /// The view-mark entry of the unordered buffer view [`make`] last made
    /// on this thread ([`make_marked`]).
    static MARK: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
    /// The bytes the raw view [`make`] last made on this thread starts
    /// before the title's view ([`raw_window`]).
    static SKIP: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// [`make`], the view-mark entry of the view if it is an unordered buffer
/// view whose writes shaders mark (`victory_shader::emit`, "Write marks"),
/// and the bytes the host's view starts before the title's ([`raw_window`];
/// for its view-stride entry).
fn make_marked(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> (Option<Kept>, Option<(u32, u32)>, u32) {
    MARK.with(|m| m.set(None));
    SKIP.with(|s| s.set(0));
    let kept = make(view, at);
    let skip = if kept.is_some() { SKIP.with(std::cell::Cell::take) } else { 0 };
    (kept, MARK.with(std::cell::Cell::take), skip)
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in a format neither of the
/// resource's family nor among the formats it was made castable to
/// (`castable`, `resource::castable`). A view of a typed texture in another
/// format of its family takes the texture's format unless it is castable.
/// shortcut: such a view (sRGB of UNORM, say, on a texture the host made
/// without castable formats) reads without its conversion.
fn fits(texture: &ID3D12Resource, castable: &[u32], mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        if typeless {
            return false;
        }
        // An unordered view in a format the host cannot store removes the
        // device (B5G5R5A1_UNORM on NVIDIA, #1753).
        if unordered && !typed_store(own) {
            once!(own, "an unordered view in a format the host cannot store");
            return false;
        }
        return true;
    }
    let cast = castable.contains(&view);
    if family(view) != family(own) && !cast {
        once!(view, own, "view format of another family");
        return false;
    }
    // The format the view is made in: the texture's own, below, for a view
    // in another format of a typed texture's family it is not castable to.
    let made = if !typeless && !cast { own } else { view };
    if unordered && !typed_store(made) {
        once!(view, own, "an unordered view in a format the host cannot store");
        return false;
    }
    if let Some(format) = format
        && !typeless
        && !cast
    {
        *format = d.Format;
    }
    true
}

/// Whether a view without a description of `resource` (a title texture)
/// may take its host texture's defaults: not when the host texture has
/// another format than the title's (typeless, or shared with a title
/// texture of another format of the same memory).
/// shortcut: such views are null; spell the description out (dimension,
/// mips) if the title makes them.
fn default_view(resource: crate::object::This) -> bool {
    let other = crate::resource::view_format(resource);
    if other.is_some() {
        once!(format = other, "a view without a description of a texture whose host texture has another format");
    }
    other.is_none()
}

/// Whether the host stores to typed unordered views of `format`.
fn typed_store(format: u32) -> bool {
    static KNOWN: Mutex<Vec<(u32, bool)>> = Mutex::new(Vec::new());
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, ok)) = known.iter().find(|(f, _)| *f == format) {
        return ok;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    let ok = host().is_some_and(|h| {
        // SAFETY: a plain query into a struct of its size.
        unsafe {
            h.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
    });
    known.push((format, ok));
    ok
}

/// Makes the host descriptor for `view` at `at` (in any host view heap):
/// the host resource it views, if it can.
pub(crate) fn make_view(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    make(view, at).flatten().map(|k| k.0)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            // Constant buffers in the title's slots, through the heap itself
            // or a table, are read as raw buffers by translated shaders
            // (`victory_shader::emit`, "Constant buffers in the heap"): the
            // console needs 16-byte alignment, a host constant buffer view
            // 256, which most of the title's do not have on the host.
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let bytes = u64::from(size).min(shadow.left(address));
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: (bytes / 4) as u32,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            buffer_srv(address, bytes, desc, at)
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(copy) = crate::resource::host_copy(resource) {
                let texture = &copy.texture;
                let mut desc = desc.as_ref().map(srv_desc);
                if desc.is_none() && !default_view(resource) {
                    return None;
                }
                if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some((texture.clone(), None)))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(copy) = crate::resource::host_copy(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let texture = &copy.texture;
            // A depth texture's unordered views write its twin.
            // SAFETY: a live resource.
            if unsafe { texture.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
                return crate::twin::uav(resource, &copy, desc.as_ref(), at)
                    .map(|twin| Some((twin, None)));
            }
            let mut desc = desc.as_ref().map(uav_desc);
            if desc.is_none() && !default_view(resource) {
                return None;
            }
            if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some((texture.clone(), None)))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// Whether `view` is a view of a texture the title placed in memory itself.
fn is_placed_texture(view: &View) -> bool {
    matches!(view, View::Placed(_, resource, _) if resource[0] != 1)
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        // Read as raw buffers, as made (`make`).
        View::Constant(..) => return null_buffer_view(at),
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which structured views the host makes raw, and where their copies
    /// start: a stride of whole dwords on a dword of its copy becomes the
    /// same bytes in dwords, from the 16 bytes at or below (the bytes
    /// skipped go to the view-stride table); odd strides and typed views
    /// stay as they are (their copy on an element).
    #[test]
    fn structured_views_become_raw_views_of_the_same_bytes() {
        // 10 elements of 12 bytes at byte 48: dwords 12..42.
        assert_eq!(raw_window(48, 10, raw_stride(DXGI_FORMAT_UNKNOWN, 12)), Some((12, 30, 0)));
        // Element 1 of a 456-byte view is not on 16 bytes: from byte 448,
        // 8 bytes early.
        assert_eq!(raw_window(456, 4, raw_stride(DXGI_FORMAT_UNKNOWN, 456)), Some((112, 458, 8)));
        assert_eq!(raw_window(12, 4, raw_stride(DXGI_FORMAT_UNKNOWN, 12)), Some((0, 15, 12)));
        // Not on a dword: no raw view (its copy starts on one).
        assert_eq!(raw_window(6, 4, raw_stride(DXGI_FORMAT_UNKNOWN, 12)), None);
        assert_eq!(copy_align(DXGI_FORMAT_UNKNOWN, 456, 456), 4);
        // A stride of 6 bytes, a typed view: not made raw.
        assert_eq!(raw_stride(DXGI_FORMAT_UNKNOWN, 6), 0);
        assert_eq!(raw_window(48, 10, raw_stride(DXGI_FORMAT_UNKNOWN, 6)), None);
        assert_eq!(copy_align(DXGI_FORMAT_UNKNOWN, 6, 6), 6);
        assert_eq!(raw_stride(DXGI_FORMAT_R32G32B32A32_FLOAT, 0), 0);
        assert_eq!(copy_align(DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16), 16);
        // Too many dwords for one view.
        assert_eq!(raw_window(0, u32::MAX, 8), None);
    }

    /// The world's 16-byte view of 0 elements removed the device.
    #[test]
    fn empty_and_overlong_buffer_views_are_refused() {
        assert!(within(64, 0, 4, 16));
        assert!(within(64, 3, 1, 16));
        assert!(!within(64, 0, 0, 16));
        assert!(!within(64, 3, 2, 16));
        assert!(!within(64, u64::MAX, 1, 16));
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Every index a translated shader can reach exists (clamped below
        // `heap_views`, which is `HEAP_VIEWS` unless the host makes no view
        // heap that large, and `HEAP_SAMPLERS`).
        // SAFETY: plain queries of live heaps.
        unsafe {
            let views = victory_shader::emit::heap_views();
            let limit = victory_shader::emit::host::Host { device: host().unwrap().device.clone() }.view_heap_limit();
            assert_eq!(views, limit, "what the host makes");
            assert!(heap.heap.GetDesc().NumDescriptors >= views);
            let capped = views < victory_shader::emit::HEAP_VIEWS;
            assert_eq!(heap.dynamic, if capped { CAPPED_DYNAMIC } else { DYNAMIC });
            let samplers = HostHeap::new(true, 128).unwrap();
            assert!(samplers.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_SAMPLERS);
        }
        // A slot that stops viewing a resource leaves it alive until the GPU
        // work in flight has run (a read of a freed host texture otherwise).
        // SAFETY: a plain committed buffer; one reference taken and given back.
        unsafe {
            let host = host().unwrap();
            let mut buffer: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer = buffer.unwrap();
            let references = |b: &ID3D12Resource| {
                let raw = windows::core::Interface::as_raw(b);
                let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
                ((**unknown).AddRef)(raw);
                ((**unknown).Release)(raw)
            };
            let slot = heap.count + 2;
            heap.raw_view_at(slot, Some(&buffer), 0, 256, false);
            assert_eq!(references(&buffer), 2, "ours and the slot's");
            heap.make_null(slot);
            assert!(references(&buffer) >= 2, "the slot's, retired");
            crate::host::release_retired();
            assert_eq!(references(&buffer), 1, "ours");
        }
        // Dynamic slots follow those for the title's.
        let first = heap.count();
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (first, first + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(first));
        assert_eq!(heap.dynamic(40), Some(first + heap.dynamic - 40));
    }

    /// A heap on a host that makes none larger (as NVIDIA's 1,000,000
    /// slots, here 65,536) has every slot a shader translated for it can
    /// index: slots for the title's views, the dynamic range, slots for
    /// view pairs, then the null buffer views and the clamp's slot.
    #[test]
    fn a_capped_heap_keeps_dynamic_slots_pairs_and_null_views() {
        assert_eq!(
            capped_layout(PORTABLE_VIEWS),
            (852_541, CAPPED_DYNAMIC, CAPPED_PAIRS)
        );
        assert_eq!(capped_layout(1 << 16), (24_577, 32_766, 8190));
        assert_eq!(capped_layout(2), (0, 0, 0));
        let Some(host) = host() else {
            return; // no host GPU
        };
        let views = 1 << 16;
        let heap = HostHeap::sized(false, 100_000, views).unwrap();
        // SAFETY: a plain query of a live heap.
        assert_eq!(unsafe { heap.heap.GetDesc() }.NumDescriptors, views);
        // 24,577 slots for the views of the title's 100,000.
        assert_eq!((heap.count(), heap.dynamic), (24_577, 32_766));
        // The whole dynamic range in one run, then nothing until released.
        assert_eq!(heap.dynamic(32_766), Some(24_577));
        assert_eq!(heap.dynamic(1), None);
        // Pairs past it, below the null views; then from the dynamic slots.
        let first = 24_577 + 32_766;
        let pairs: Vec<u32> = (0..(views - 3 - first) / 2).map(|_| heap.pair().unwrap()).collect();
        assert_eq!(pairs[0], first);
        assert!(pairs[pairs.len() - 1] + 2 <= views - 3 && pairs.len() > 4000);
        assert_eq!(heap.pair(), None);
        heap.release(24_577, 32_766);
        assert_eq!(heap.pair(), Some(24_577));
        // The null buffer views (raw: 4-byte elements) are where shaders
        // translated for such a heap read them, the last slot left null.
        let (_, strides) = heap.strides.as_ref().unwrap();
        // SAFETY: dwords of the mapped table, which covers these indices.
        let stride = |index: u32| unsafe { (*strides as *const u32).add(index as usize).read_volatile() };
        assert_eq!([stride(views - 3), stride(views - 2), stride(views - 1)], [4, 4, 0]);
        // A smaller title heap keeps the title's order (its table says
        // so, for its slots and the host's own, which tables made afresh
        // lie in): its 4 slots, the dynamic slots, and the rest are pairs'.
        let small = HostHeap::sized(false, 4, views).unwrap();
        assert_eq!((small.count(), small.dynamic, small.pair()), (4, 32_766, Some(4 + 32_766)));
        assert_eq!((small.slot(3), small.slot(4), small.run(0, 3), small.run(2, 4)), (Some(3), None, Some(0), None));
        assert!(small.slots.is_none() && small.slots().is_some());
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// The title's records `n` of 32 bytes, zeroed, and their handles.
    fn records(n: usize) -> (Vec<[u32; 8]>, impl Fn(usize) -> u64) {
        let mut records = vec![[0u32; 8]; n];
        let start = records.as_mut_ptr() as u64;
        (records, move |k: usize| start + 32 * k as u64)
    }

    /// On a host that caps its view heaps, each distinct view of the
    /// title's has one host slot however many of the title's slots hold
    /// it: a shader finds it through the view-slot table (then the
    /// view-stride table, by host slot), a slot without a view leads to
    /// the null slot, and a table is read where the title has it. The host
    /// slot goes back, null and unmarked, when the last title slot gives
    /// the view up and the sync is over.
    #[test]
    fn a_capped_heap_holds_each_distinct_view_once() {
        if host().is_none() {
            return; // no host GPU
        }
        let views = 1 << 16;
        let heap = HostHeap::sized(false, 100_000, views).unwrap();
        let slots = heap.slots.as_ref().unwrap();
        // SAFETY: a dword of the mapped slot table, which covers these slots.
        let entry = |index: u32| unsafe { (slots.table.1 as *const u32).add(index as usize).read_volatile() };
        // SAFETY: a dword of the mapped stride table, which covers these slots.
        let stride = |slot: u32| unsafe { (heap.strides.as_ref().unwrap().1 as *const u32).add(slot as usize).read_volatile() };
        // SAFETY: two dwords of the mapped mark table, which covers these slots.
        let mark = |slot: u32| unsafe { (heap.marks.as_ref().unwrap().1 as *const u32).add(2 * slot as usize).read_volatile() };
        let null = views - 1;
        let last = victory_shader::emit::HEAP_VIEWS - 1;
        assert_eq!((entry(0), entry(99_999), entry(last)), (null, null, null));
        // The title's descriptors: constant buffer views (raw buffer views
        // on the host, 4-byte elements), records 0 and 2 of the same one,
        // and nothing in record 3.
        let (_records, handle) = records(4);
        for (k, address) in [(0, 0x1000), (1, 0x2000), (2, 0x1000)] {
            crate::descriptor::create_cbv(std::ptr::null(), &[address, 1, 256, 0], handle(k));
        }
        assert!(heap.mirror(90_000, handle(0), false).is_some());
        assert!(heap.mirror(90_001, handle(1), false).is_some());
        assert!(heap.mirror(7, handle(2), false).is_some());
        assert!(heap.mirror(90_002, handle(3), false).is_none());
        let (a, b) = (heap.slot(90_000).unwrap(), heap.slot(90_001).unwrap());
        assert_eq!((a, b), (0, 1), "host slots in the order the views came");
        assert_eq!((entry(90_000), entry(90_001), entry(7), entry(90_002)), (a, b, a, null));
        assert_eq!((stride(a), stride(b), stride(null)), (4, 4, 0));
        {
            let map = slots.map.lock().unwrap();
            assert_eq!((map.used, map.live, map.views.len()), (2, 3, 2));
        }
        // Through a table: at the title's index, wherever its views lie.
        assert_eq!((heap.run(90_000, 90_002), heap.run(6, 7)), (Some(90_000), Some(6)));
        assert_eq!(heap.run(99_999, 100_000), None, "past the title's heap");
        assert_eq!(heap.slot(100_000), None);
        // View marks are kept by host slot, as shaders read them.
        heap.set_mark(b, Some((7, 0)));
        // One of the two title slots holding a view gives it up: the other
        // still reads it.
        // SAFETY: a record of ours.
        unsafe { (handle(0) as *mut [u32; 3]).write([0; 3]) };
        assert!(heap.mirror(90_000, handle(0), false).is_none());
        heap.settle();
        assert_eq!((entry(90_000), entry(7), stride(a)), (null, a, 4));
        // The only title slot holding the other gives it up: the view
        // stays until the sync is over, then its slot is null and free.
        // SAFETY: a record of ours.
        unsafe { (handle(1) as *mut [u32; 3]).write([0; 3]) };
        assert!(heap.mirror(90_001, handle(1), false).is_none());
        assert_eq!((entry(90_001), stride(b)), (null, 4));
        heap.settle();
        assert_eq!((stride(b), mark(b)), (0, victory_shader::emit::VIEW_MARK_NONE));
        let map = slots.map.lock().unwrap();
        assert_eq!((map.used, map.live, map.views.len()), (1, 1, 1));
    }

    /// The bulk sync keys slots by title index: a view in a capped heap is
    /// made, found unchanged without a remake, keeps its host slot when it
    /// moves to another title slot within a sync, and gives it back when
    /// the title writes nothing there.
    #[test]
    fn bulk_mirror_follows_the_slot_table() {
        if host().is_none() {
            return; // no host GPU
        }
        let heap = HostHeap::sized(false, 100_000, 1 << 16).unwrap();
        let (_records, handle) = records(2);
        crate::descriptor::create_cbv(std::ptr::null(), &[0x1000, 1, 256, 0], handle(0));
        let bytes = |at: u64| {
            let mut b = [0u8; 12];
            // SAFETY: 12 bytes of a record of ours.
            unsafe { std::ptr::copy_nonoverlapping(at as *const u8, b.as_mut_ptr(), 12) };
            b
        };
        let index = 90_000;
        heap.mirror_bytes(index, index + 1, 32, &bytes(handle(0)));
        assert!(heap.made(index));
        let slot = heap.slot(index).unwrap();
        let key = heap.made.lock().unwrap()[index as usize];
        // Unchanged: skipped, still at the same host slot.
        heap.mirror_bytes(index, index + 1, 32, &bytes(handle(0)));
        assert_eq!((heap.slot(index), heap.made.lock().unwrap()[index as usize]), (Some(slot), key));
        // The title moves it to slot 5: the same host slot, not made again.
        heap.mirror_bytes(index, index + 1, 32, &bytes(handle(1)));
        heap.mirror_bytes(5, 6, 32, &bytes(handle(0)));
        heap.settle();
        assert_eq!((heap.slot(index), heap.slot(5)), (None, Some(slot)));
        assert!(!heap.made(index) && heap.made(5));
        // The title writes nothing there: the host slot is given back.
        heap.mirror_bytes(5, 6, 32, &bytes(handle(1)));
        heap.settle();
        assert!(!heap.made(5) && heap.slot(5).is_none());
        let map = heap.slots.as_ref().unwrap().map.lock().unwrap();
        assert_eq!((map.used, map.live, map.views.len()), (0, 0, 0));
    }

    /// A host texture of `format` that allows unordered access, made as
    /// `resource::host_texture` makes it (castable to its element size's
    /// formats where the host takes that), and those formats.
    fn unordered_texture(format: u32) -> Option<(ID3D12Resource, Vec<u32>)> {
        use windows::core::Interface;
        let device = &host()?.device;
        let desc = D3D12_RESOURCE_DESC1 {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT(format as i32),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let castable = crate::resource::castable(format);
        let formats: Vec<DXGI_FORMAT> = castable.iter().map(|&f| DXGI_FORMAT(f as i32)).collect();
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain object creation; the list lives through the call.
        unsafe {
            device
                .cast::<ID3D12Device10>()
                .ok()?
                .CreateCommittedResource3(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_BARRIER_LAYOUT_COMMON,
                    None,
                    None::<&ID3D12ProtectedResourceSession>,
                    Some(&formats),
                    &mut texture,
                )
                .ok()?;
        }
        Some((texture?, castable))
    }

    /// An unordered view is made only in a format the host can store, in
    /// the texture's own format too: B5G5R5A1_UNORM has none on NVIDIA, and
    /// making one removed the device at boot (#1753).
    #[test]
    fn unordered_views_only_in_formats_the_host_stores() {
        const B5G5R5A1_UNORM: u32 = 86;
        const R8G8B8A8_UNORM: u32 = 28;
        let Some((stored, castable)) = unordered_texture(R8G8B8A8_UNORM) else {
            return; // no host GPU
        };
        assert!(typed_store(R8G8B8A8_UNORM));
        assert!(fits(&stored, &castable, None, true));
        let mut own = DXGI_FORMAT(R8G8B8A8_UNORM as i32);
        assert!(fits(&stored, &castable, Some(&mut own), true));
        // The host may refuse such a texture outright; where it makes one,
        // its views follow the host's typed store support.
        let Some((texture, castable)) = unordered_texture(B5G5R5A1_UNORM) else {
            return;
        };
        let stores = typed_store(B5G5R5A1_UNORM);
        assert_eq!(fits(&texture, &castable, None, true), stores);
        let mut own = DXGI_FORMAT(B5G5R5A1_UNORM as i32);
        assert_eq!(fits(&texture, &castable, Some(&mut own), true), stores);
        // Shader views are not affected.
        assert!(fits(&texture, &castable, None, false));
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        // Dynamic slots follow those for the title's.
        let base = heap.count();
        assert_eq!((first, left, right, rest), (base, base + 1, base + 3, base + 5));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(left, 2);
        heap.release(right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    /// Batch classification across empty, unchanged, updated, placed-texture,
    /// and `forget()`-stale slots matches per-slot `mirror(..., false)` and
    /// emits zero updates for unchanged populated slots on repeat sync.
    #[test]
    fn classify_range_matches_per_slot_mirror_across_pages() {
        use super::super::{copy_simple, create_cbv, create_placed_srv, create_srv, create_uav, forget};
        let (per_page, null, no_desc) = ((crate::watch::PAGE / 32) as u32, std::ptr::null(), std::ptr::null());
        let (total, mut buf) = (per_page * 3, vec![0u8; per_page as usize * 3 * 32]);
        let (start, p2) = (buf.as_mut_ptr() as u64, per_page * 2);
        let slot = |i: u32| start + u64::from(i) * 32;
        let (res_a, res_b, res_c, res_d) = (0x5555_1000usize, 0x5555_2000, 0x5555_3000, 0x5555_4000);
        create_cbv(null, &[0x4000, 0, 512, 0], slot(1));
        create_srv(null, res_a as crate::object::This, no_desc, slot(2));
        create_srv(null, res_b as crate::object::This, no_desc, slot(3));
        create_uav(null, res_c as crate::object::This, null, no_desc, slot(4));
        create_placed_srv(null, &[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0u32; 16], slot(5));
        let write_raw = |i: u32, w: [u32; 3]| {
            // SAFETY: callers only pass indices `< total` inside `buf`.
            unsafe { (slot(i) as *mut [u32; 3]).write_unaligned(w) };
        };
        for (i, w) in [(6, [VIEW_TAG, 0, 0]), (7, [VIEW_TAG, u32::MAX, 1]), (8, [0xdead_beef, 1, 1])] {
            write_raw(i, w);
        }
        copy_simple(null, 2, slot(p2 + 1), slot(1), 0);
        create_srv(null, res_d as crate::object::This, no_desc, slot(p2 + 4));
        let sync_all = |batch: &mut [u64], reference: &mut [u64]| -> usize {
            (0..3).map(|p| {
                let (first, end) = (p * per_page, (p + 1) * per_page);
                let byte_from = first as usize * 32;
                let byte_to = (end - 1) as usize * 32 + 12;
                // SAFETY: `buf` stays pinned and no other reference mutates it during this read.
                let slice = unsafe { std::slice::from_raw_parts((start as *const u8).add(byte_from), byte_to - byte_from) };
                let got = classify_bytes(batch, first, end, 32, slice);
                for &(idx, key, ref view) in &got {
                    batch[idx as usize] = view.as_ref().filter(|v| !is_placed_texture(v)).map_or(0, |_| key);
                }
                let mut expected = Vec::new();
                for idx in first..end {
                    // SAFETY: `idx` lies inside `buf`.
                    let [tag, i, g] = unsafe { (slot(idx) as *const [u32; 3]).read_unaligned() };
                    let key = u64::from(g) << 32 | u64::from(i);
                    match (tag == VIEW_TAG).then(|| view_at(i, g)).flatten() {
                        None if reference[idx as usize] != 0 => { reference[idx as usize] = 0; expected.push((idx, 0, None)); }
                        Some(v) if reference[idx as usize] != key => {
                            reference[idx as usize] = if is_placed_texture(&v) { 0 } else { key };
                            expected.push((idx, key, Some(v)));
                        }
                        _ => {}
                    }
                }
                assert_eq!((&got, &batch[first as usize..end as usize]), (&expected, &reference[first as usize..end as usize]));
                got.len()
            }).sum()
        };
        let (mut batch, mut reference) = (vec![0u64; total as usize], vec![0u64; total as usize]);
        assert_eq!(sync_all(&mut batch, &mut reference), 8);
        // SAFETY: `slot(5)` lies inside `buf`; simulate a draw realizing the placed texture.
        let [_, i5, g5] = unsafe { (slot(5) as *const [u32; 3]).read_unaligned() };
        (batch[5], reference[5]) = (u64::from(g5) << 32 | u64::from(i5), u64::from(g5) << 32 | u64::from(i5));
        assert_eq!(sync_all(&mut batch, &mut reference), 0);
        create_cbv(null, &[0x8000, 0, 128, 0], slot(3));
        write_raw(p2 + 1, [0, 0, 0]);
        for r in [res_a, res_d] { forget(r); }
        create_srv(null, res_d as crate::object::This, no_desc, slot(p2 + 5));
        assert_eq!(sync_all(&mut batch, &mut reference), 6);
        assert_eq!((batch[2], batch[(p2 + 1) as usize], batch[(p2 + 2) as usize], batch[(p2 + 4) as usize]), (0, 0, 0, 0));
        assert_ne!(batch[(p2 + 5) as usize], 0);
        assert_eq!(sync_all(&mut batch, &mut reference), 0);
        for r in [res_b, res_c, res_d] { forget(r); }
    }
}
