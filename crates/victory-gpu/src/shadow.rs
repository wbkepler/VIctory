//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<std::collections::HashSet<(usize, usize)>>,
    /// The copy's bytes before the submission ran where its shaders write:
    /// what the GPU wrote differs. Disjoint segments (offset, bytes) in
    /// order, and the 64 KiB pieces (by title address) already compared.
    before: Mutex<Snapshot>,
    /// Its first bit in the buffer views' written-pages bitmap, one per
    /// 64 KiB title page it touches (tracked memory only): writes through
    /// views of it are marked there ([`Shadow::mark_entry`]).
    marks: Option<u32>,
    /// In a GPU upload heap ([`register_read_only`]): the CPU must not read
    /// it back.
    pub(crate) upload: std::sync::atomic::AtomicBool,
    /// A copy shaders may write, in a GPU upload heap (`buffer`, `gpu`), with
    /// `data` its mirror in system memory, where the CPU reads and writes:
    /// what the CPU writes goes on into video memory ([`Shadow::push`]), what
    /// the GPU wrote comes back before the CPU reads it ([`pull`]).
    vram: Option<Vram>,
}

/// The video memory side of a mirrored copy.
struct Vram {
    /// The mirror's buffer (system memory), where [`pull`] copies to.
    mirror: ID3D12Resource,
    /// The copy in video memory, mapped (written, never read).
    data: *mut u8,
    /// A bit per [`PAGE`] of the copy (by offset) whose mirror is behind
    /// video memory: refreshes write video memory only, and the mirror
    /// catches up when the CPU reads it ([`Shadow::freshen`]).
    stale: Mutex<Vec<u64>>,
    /// Offsets (whole pages, sorted, apart) a submission takes snapshots of
    /// ([`snapshots_ahead`]): refreshes there write the mirror too, so that
    /// the snapshots need no copy back from video memory.
    ahead: Mutex<Vec<(usize, usize)>>,
}

/// Snapshot segments (offset, bytes) and the pieces compared.
type Snapshot = (Vec<(usize, Vec<u8>)>, std::collections::HashSet<usize>);

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool, which keeps the largest up to
/// [`POOL_BYTES`] in all: a world submission snapshots several copies, and
/// fresh buffers cost a page fault per 4 KiB touched (millions per frame).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    let mut total = 0;
    pool.retain(|v| {
        total += v.capacity();
        total <= POOL_BYTES
    });
}

/// Snapshot memory kept for reuse.
const POOL_BYTES: usize = 2 << 30;

/// The runs `[start, end)` of bytes where the copy differs from title
/// memory (`host`, `title`: the same bytes of each) and `gpu(i)` says the
/// GPU wrote byte `i`: what a copy-back puts into title memory.
///
/// Equal stretches (most of a page) are skipped a block at a time, then a
/// line at a time inside a block that differs: one comparison per 8 bytes
/// cost ~30 us per 64 KiB page, ~70 ms per world frame of marked pages.
fn differing(host: &[u8], title: &[u8], gpu: impl Fn(usize) -> bool) -> Vec<(usize, usize)> {
    const BLOCK: usize = 1 << 10;
    const LINE: usize = 32;
    let n = host.len().min(title.len());
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut block = 0;
    while block < n {
        let block_end = (block + BLOCK).min(n);
        if host[block..block_end] != title[block..block_end] {
            let mut line = block;
            while line < block_end {
                let line_end = (line + LINE).min(block_end);
                if host[line..line_end] != title[line..line_end] {
                    for i in line..line_end {
                        if host[i] != title[i] && gpu(i) {
                            match runs.last_mut() {
                                Some((_, end)) if *end == i => *end = i + 1,
                                _ => runs.push((i, i + 1)),
                            }
                        }
                    }
                }
                line = line_end;
            }
        }
        block = block_end;
    }
    runs
}

/// Runs of a copy-back closer than this go into title memory as one span,
/// with title memory's bytes between them as the copy-back read them: a
/// kernel copy per run cost over 1 ms per page in the world, where shaders
/// change many scattered words of a page.
const SPAN_GAP: usize = 256;

/// Writes the runs `[start, end)` of `merged` (title memory as read, with
/// the shaders' bytes in the runs) into title memory at `at`, and nothing
/// far from them: the title's threads write the same pages while a
/// copy-back runs, and a byte they wrote after the copy-back read the page
/// must stay. (Putting back the whole page undid those writes.) True if a
/// run was written.
fn write_runs(at: usize, merged: &[u8], runs: &[(usize, usize)]) -> bool {
    if runs.is_empty() {
        return false;
    }
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for &(start, end) in runs {
        match spans.last_mut() {
            Some((_, last)) if start - *last < SPAN_GAP => *last = end,
            _ => spans.push((start, end)),
        }
    }
    let _writing = crate::watch::writing(at, merged.len());
    for (start, end) in spans {
        crate::host::write_mapped((at + start) as *mut u8, merged[start..].as_ptr(), end - start);
    }
    true
}

/// Outside `runs`, the copy takes title memory's bytes (`title`, read at the
/// copy-back).
fn take_others(host: &mut [u8], title: &[u8], runs: &[(usize, usize)]) {
    let mut at = 0;
    for &(start, end) in runs {
        host[at..start].copy_from_slice(&title[at..start]);
        at = end;
    }
    host[at..].copy_from_slice(&title[at..]);
}

/// After a copy-back: the page at `page` is taken, then the copy's bytes
/// `host` (from title address `from`) are read from title memory again. The
/// title writes the page while it goes back, so bytes read before the guard
/// is back may be old: a title write that lands before the guard is in this
/// read, and one after it faults and marks the page changed. (Taking the
/// page with the bytes read earlier lost such writes from the copy, and a
/// later copy-back of the page put the copy's older bytes over them.)
fn take_and_read(t: &crate::watch::Tracked, page: usize, from: usize, host: &mut [u8]) {
    t.taken(&[page / PAGE]);
    crate::host::read_into(from as *const u8, host);
}

thread_local! {
    /// Title memory as a copy-back reads it, a page at most: a fresh zeroed
    /// buffer cost a page's memset per page, thousands per world frame.
    static TITLE: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(vec![0; PAGE]);
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        self.freshen(at, at + 4 * n);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
        // As written on the GPU and pulled back.
        self.push(at, bytes.len());
    }

    /// Whether it is in video memory with a mirror ([`Shadow::push`]).
    #[cfg(test)]
    pub(crate) fn mirrored(&self) -> bool {
        self.vram.is_some()
    }

    /// The pages (by index in the copy) its mirror is behind on.
    #[cfg(test)]
    fn stale_pages(&self) -> Vec<usize> {
        self.stale_runs(0, self.len as usize)
            .into_iter()
            .flat_map(|(_, at, n)| at / PAGE..(at + n).div_ceil(PAGE))
            .collect()
    }

    /// The CPU wrote `[at, at + len)` (offsets) of the mirror: the same
    /// bytes into the copy in video memory.
    fn push(&self, at: usize, len: usize) {
        if let Some(v) = &self.vram {
            let len = len.min((self.len as usize).saturating_sub(at));
            // SAFETY: inside both mappings (cut to the copy); the mirror is
            // system memory, the video memory is only written.
            unsafe { std::ptr::copy_nonoverlapping(self.data.add(at), v.data.add(at), len) };
        }
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        self.freshen(at, at + 4 * n);
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        // A pending page first (lazy copy-back).
        lazy::flush_range(page, PAGE, lazy::Path::CopyBack);
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        // A title write while the page is read saves its bytes from before.
        if let Some(t) = &self.tracked {
            t.arm(from, to);
        }
        self.freshen(from - start, to - start);
        if !self.put_back(page, from, to, None) {
            return 0;
        }
        // Our write is not the title's: the copy takes the page as it is.
        if let Some(t) = &self.tracked {
            // SAFETY: inside the mapping; only the submitting thread writes it.
            let host =
                unsafe { std::slice::from_raw_parts_mut(self.data.add(from - start), to - from) };
            take_and_read(t, page, from, host);
            self.push(from - start, to - from);
        }
        to - from
    }

    /// The copy-back of `[from, to)` in the page at `page`: the bytes the
    /// shaders changed go into title memory, and no others (the title's
    /// threads write the page meanwhile: a whole page put back undid their
    /// writes since it was read), and the copy takes the title's others.
    /// What the shaders changed differs from the copy's bytes before the
    /// submission: `saved`, or what the page held when a title write landed
    /// while it was read (saved by the guard while the page is armed), else
    /// title memory. True if bytes were written.
    fn put_back(&self, page: usize, from: usize, to: usize, saved: Option<Box<[u8]>>) -> bool {
        TITLE.with(|title| self.put_back_with(page, from, to, saved, &mut title.borrow_mut()))
    }

    fn put_back_with(
        &self,
        page: usize,
        from: usize,
        to: usize,
        saved: Option<Box<[u8]>>,
        title: &mut [u8],
    ) -> bool {
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host =
            unsafe { std::slice::from_raw_parts_mut(self.data.add(from - self.start as usize), n) };
        let title = &mut title[..n];
        let read = crate::host::read_into(from as *const u8, title);
        #[cfg(test)]
        tests::after_read();
        // Our own write below must not save the page for this copy again.
        let saved = match &self.tracked {
            Some(t) => {
                t.unarm(page / PAGE);
                saved.or_else(|| t.saved(page / PAGE))
            }
            None => saved,
        };
        if !read {
            return false;
        }
        let runs = match &saved {
            None => differing(host, title, |_| true),
            Some(before) => {
                let before = &before[from - page..to - page];
                differing(host, title, |i| host[i] != before[i])
            }
        };
        // The copy takes the title's other bytes, and holds what goes back.
        // Without bytes saved, the runs are every byte that differs: the
        // others are the title's already (copying them cost a 64 KiB write
        // per page), and the copy is as `freshen` left it, equal to video
        // memory: nothing to push. Else what it takes goes there too.
        if saved.is_some() {
            take_others(host, title, &runs);
            self.push(from - self.start as usize, n);
        }
        write_runs(from, host, &runs)
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Arc<Shadow> {
        Self::with_vram(buffer, data, gpu, start, len, None)
    }

    fn with_vram(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        vram: Option<Vram>,
    ) -> Arc<Shadow> {
        let tracked = crate::watch::track(start as usize, len as usize, true);
        let marks = tracked.as_ref().and_then(|_| marks().take(pages(start, len)));
        let shadow = Arc::new(Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked,
            pending: Mutex::default(),
            before: Mutex::default(),
            marks,
            upload: std::sync::atomic::AtomicBool::new(false),
            vram,
        });
        if let Some(base) = marks {
            self::marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        shadow
    }

    /// The view-mark entry (`victory_shader::emit`, "Write marks") of a
    /// view of this copy whose first byte is at title address `start`: the
    /// bit of that byte's page and its offset in the page. None where
    /// writes through it are not marked (memory not tracked, or no bits
    /// left): such views declare what they write ([`Shadow::gpu_writes`]).
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through views of this copy are marked.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    /// Whether this copy tracks dirty 64 KiB pages via [`crate::watch`].
    pub(crate) fn is_tracked(&self) -> bool {
        self.tracked.is_some()
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    ///
    /// `marked`: the shaders mark the pages they write through this copy's
    /// views (`victory_shader::emit`, "Write marks"): no snapshot then, only
    /// the pages they marked are compared ([`Shadow::copy_back_marked`]).
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64, marked: bool) {
        if self.upload.load(std::sync::atomic::Ordering::Relaxed) {
            // A view of the memory that writes (through another resource,
            // or a placed view) found this read-only buffer's copy: right,
            // but its copy-back reads video memory at ~50 MB/s.
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    start = format_args!("{:#x}", self.start),
                    "shaders write the copy of a buffer made without unordered access, in upload memory: slow copy-back"
                );
            });
        }
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize, marked && self.marked());
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, segments covering the ranges the submission
    /// declares (the world declares overlapping views of its 174 MB heap
    /// dozens of times; one segment over all of them copied most of the
    /// heap), in reused memory (fresh snapshots cost ~2.3 million page
    /// faults per world frame).
    ///
    /// With `marked`, nothing is kept: the pages are armed instead, so that
    /// a title write there before the copy-back keeps the page's bytes from
    /// before it ([`crate::watch::Tracked::arm`]).
    pub(crate) fn writing(&self, address: usize, len: usize, marked: bool) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
        if marked && let Some(t) = &self.tracked {
            t.arm(address, address + len);
            return;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        self.freshen(from, to);
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let segments = &mut before.0;
        // The segments the range overlaps or touches.
        let first = segments.partition_point(|(at, bytes)| at + bytes.len() < from);
        let last = first + segments[first..].iter().take_while(|(at, _)| *at <= to).count();
        if let [(at, bytes)] = &segments[first..last]
            && *at <= from
            && to <= at + bytes.len()
        {
            return;
        }
        let start = segments.get(first).filter(|_| first < last).map_or(from, |s| s.0.min(from));
        let end = segments[first..last].last().map_or(to, |(at, bytes)| (at + bytes.len()).max(to));
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SnapshotBytes, (end - start) as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::snapshot(self.start, self.len, (end - start) as u64);
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        let mut at = start;
        for (taken, bytes) in segments.drain(first..last) {
            grown.extend_from_slice(host(at, taken));
            grown.extend_from_slice(&bytes);
            at = taken + bytes.len();
            give_back(bytes);
        }
        grown.extend_from_slice(host(at, end));
        segments.insert(first, (start, grown));
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_or_not(address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        self.copy_back_or_not(address, len, false);
    }

    fn copy_back_or_not(&self, address: usize, len: usize, back: bool) -> usize {
        if back && lazy::on() {
            // Pending pages under a snapshot first (lazy copy-back): then
            // title memory differs from the copy by this submission's bytes
            // alone. (Marked ranges have none: their pages stay pending.)
            let (from, to) = self.inside(address, len);
            let snapshotted = self
                .before
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .0
                .iter()
                .any(|(at, bytes)| *at <= from && to <= at + bytes.len());
            if snapshotted && from < to {
                lazy::flush_range(self.start as usize + from, to - from, lazy::Path::CopyBack);
            }
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (segments, compared) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = segments.iter().find(|(at, bytes)| *at <= from && to <= at + bytes.len())
            && back
            && from < to
        {
            let before = &before[from - taken..to - taken];
            // The mirror is read below (with snapshots ahead, its pages are
            // current already).
            self.freshen(from, to);
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let title = &mut title[..n];
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && !compared.insert(start + at) {
                    at += n;
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
                    // The GPU's bytes go into title memory, and no others:
                    // the title writes the piece meanwhile.
                    let runs = differing(host, title, |i| host[i] != old[i]);
                    // The copy takes the title's bytes in turn.
                    take_others(host, title, &runs);
                    if write_runs(start + at, host, &runs) {
                        written += n;
                    }
                    synced.push((start + at, n));
                    // The copy took the title's bytes: so does video memory.
                    self.push(at, n);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                for (address, n) in synced {
                    if (first..last).contains(&(address / PAGE)) {
                        // SAFETY: inside the mapping, as above.
                        let host = unsafe {
                            std::slice::from_raw_parts_mut(self.data.add(address - start), n)
                        };
                        take_and_read(t, address, address, host);
                    }
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
            if let Some(t) = &self.tracked {
                t.disarm();
            }
        }
        written
    }

    /// After a submission whose shaders marked the 64 KiB title page at
    /// `page` as written through views of this copy: the bytes they changed
    /// go into title memory, and the copy takes the title's others. What
    /// they changed is what differs from the copy's bytes before the
    /// submission: title memory where the title has not written the page
    /// since the copy took it, else what the page held when the title first
    /// wrote it ([`crate::watch::Tracked::saved`], if the page was armed).
    /// Returns the bytes written.
    // shortcut: a page the title wrote that was not armed (a write through
    // the heap's views no list declared) keeps the title's bytes, losing
    // the shaders' there (counted).
    pub(crate) fn copy_back_marked(&self, page: usize) -> usize {
        self.copy_back_marked_from(page, None)
    }

    /// [`Shadow::copy_back_marked`], with `base` (the whole page) as the
    /// copy's bytes before the submission if given: a lazy copy-back of a
    /// page several copies hold GPU bytes of, where title memory holds the
    /// earlier copies' bytes when a later one goes back (as after a
    /// submission, where the first copy-back's write saved the page for the
    /// others).
    fn copy_back_marked_from(&self, page: usize, base: Option<&[u8]>) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        self.freshen(from - start, to - start);
        let saved = match base {
            Some(b) => Some(Box::from(b)),
            None => t.saved(page / PAGE),
        };
        if saved.is_none() && t.changed(page / PAGE) {
            t.unarm(page / PAGE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarkedKept, 1);
            return 0;
        }
        // A title write while the page is read saves its bytes from before.
        t.arm(from, to);
        let written = self.put_back(page, from, to, saved);
        // Equal to title memory now: taken, unless it holds the page in part.
        // A page neither we nor the title wrote meanwhile (a title write
        // marks it changed) is still taken and guarded, and the copy has
        // title memory's bytes: nothing to read again. (A third of the
        // world's marked pages hold no changed byte.)
        if from == page && to == page + PAGE && (written || t.changed(page / PAGE)) {
            // SAFETY: inside the mapping; only the submitting thread writes it.
            let host =
                unsafe { std::slice::from_raw_parts_mut(self.data.add(from - start), to - from) };
            take_and_read(t, page, from, host);
            self.push(from - start, to - from);
        }
        if written { to - from } else { 0 }
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them, what the title
            // wrote since it was taken (`update_busy`).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    bytes += self.update_free(at, a);
                }
                if e > a.max(at) {
                    bytes += self.update_busy(a.max(at), e);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` where a list submitted earlier
    /// declared that shaders write. A submission waits for the one before it,
    /// so none of this submission's shaders has run yet: the pages the title
    /// wrote since they were taken are copied as anywhere else. Skipped, the
    /// title's constants there stayed stale for the whole submission (a
    /// dispatch read its sizes as zeros and looped until the device was
    /// removed). The same bytes go into the snapshot `writing` took, so
    /// copy-back neither takes them for the GPU's nor puts them back over a
    /// later title write.
    fn update_busy(&self, from: usize, to: usize) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        let runs = t.take(from, to);
        if runs.is_empty() {
            return 0;
        }
        let start = self.start as usize;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut copied = 0;
        for (a, b) in runs {
            copied += self.copy(a, b);
            let (a, b) = (a - start, b - start);
            for (at, bytes) in before.0.iter_mut() {
                let (s, e) = ((*at).max(a), (*at + bytes.len()).min(b));
                if s < e {
                    // The mirror too (with no snapshot ahead, a refresh
                    // wrote video memory only).
                    if self.vram.is_some() {
                        self.copy_both(s, e);
                    }
                    // SAFETY: inside the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    bytes[s - *at..e - *at].copy_from_slice(host);
                }
            }
        }
        copied
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes: the pages changed since taken, or all of it where the
    /// memory is not tracked.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if let Some(t) = &self.tracked {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            let runs = t.take(from, to);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
            // Cut to the range: next to it shaders may be writing the copy.
            let copied: usize = runs.into_iter().map(|(a, b)| self.copy(a, b)).sum();
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::Checked, (to - from) as u64);
                add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
            }
            return copied;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy:
    /// into video memory alone for a mirrored copy, whose mirror catches up
    /// when the CPU reads it ([`Shadow::freshen`]; writing it too cost a
    /// second copy of everything refreshed).
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        if let Some(v) = &self.vram {
            let (a, b) = (from - start, to - start);
            let ahead = v.ahead.lock().unwrap_or_else(|e| e.into_inner());
            let mut at = a;
            for &(s, e) in ahead.iter() {
                if e <= at {
                    continue;
                }
                if s >= b {
                    break;
                }
                if at < s {
                    self.copy_video(v, at, s);
                }
                let e = e.min(b);
                self.copy_both(at.max(s), e);
                at = e;
            }
            if at < b {
                self.copy_video(v, at, b);
            }
            return to - from;
        }
        self.copy_both(from - start, to - start);
        to - from
    }

    /// Title memory at offsets `[from, to)` into the copy in video memory
    /// only: the mirror is behind there until [`Shadow::freshen`].
    fn copy_video(&self, v: &Vram, from: usize, to: usize) {
        // SAFETY: inside the video memory mapping (offsets within the copy).
        let at = unsafe { v.data.add(from) };
        crate::host::copy_mapped(
            at,
            (self.start as usize + from) as *const u8,
            to - from,
            false,
        );
        self.stale(from, to);
    }

    /// Title memory at offsets `[from, to)` into the copy (the mirror, and
    /// video memory after it).
    fn copy_both(&self, from: usize, to: usize) {
        // SAFETY: inside the mapping (offsets within the copy).
        let at = unsafe { self.data.add(from) };
        crate::host::copy_mapped(
            at,
            (self.start as usize + from) as *const u8,
            to - from,
            false,
        );
        self.push(from, to - from);
    }

    /// The mirror is behind video memory over offsets `[from, to)`.
    fn stale(&self, from: usize, to: usize) {
        let Some(v) = &self.vram else { return };
        let mut bits = v.stale.lock().unwrap_or_else(|e| e.into_inner());
        for k in from / PAGE..to.div_ceil(PAGE) {
            bits[k / 64] |= 1 << (k % 64);
        }
    }

    /// Before the CPU reads offsets `[from, to)` of a mirrored copy: the
    /// pages there the mirror is behind on come from video memory (one GPU
    /// copy; nothing when none is).
    fn freshen(&self, from: usize, to: usize) {
        let mut runs = self.stale_runs(from, to);
        bring_back(&mut runs);
    }

    /// The pages over offsets `[from, to)` the mirror is behind on, as runs
    /// `(copy, offset, len)` (whole pages, or to the copy's end).
    fn stale_runs(&self, from: usize, to: usize) -> Vec<(&Shadow, usize, usize)> {
        let Some(v) = &self.vram else {
            return Vec::new();
        };
        let len = self.len as usize;
        let mut runs: Vec<(&Shadow, usize, usize)> = Vec::new();
        let bits = v.stale.lock().unwrap_or_else(|e| e.into_inner());
        for k in from / PAGE..to.div_ceil(PAGE).min(len.div_ceil(PAGE)) {
            if bits[k / 64] >> (k % 64) & 1 == 0 {
                continue;
            }
            let (a, b) = (k * PAGE, ((k + 1) * PAGE).min(len));
            match runs.last_mut() {
                Some(r) if r.1 + r.2 == a => r.2 = b - r.1,
                _ => runs.push((self, a, b - a)),
            }
        }
        runs
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if let Some(base) = self.marks {
            let mut m = marks();
            m.owners.remove(&base);
            m.give_back(base, pages(self.start, self.len));
        }
    }
}

/// The 64 KiB title pages `[start, start + len)` touches.
fn pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// The bits of the buffer views' written-pages bitmap (`victory_shader::
/// emit`, "Write marks"): a run per tracked copy, a bit per title page it
/// touches, for as long as it lives.
#[derive(Default)]
struct Marks {
    /// Free runs below `end`, by first bit: their length.
    free: BTreeMap<u32, u32>,
    /// Bits handed out at some time: those past it are clear.
    end: u32,
    /// The copy holding each run, by its first bit.
    owners: BTreeMap<u32, std::sync::Weak<Shadow>>,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
    owners: BTreeMap::new(),
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// A run of `n` bits: its first. None when the bitmap is full (the last
    /// bit takes the marks of offsets past it, so it is never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end + n < victory_shader::emit::VIEW_WRITE_BITS {
            self.end += n;
            return Some(self.end - n);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no bits left in the views' written-pages bitmap: copies made now declare their shaders' writes"));
        None
    }

    /// The run of `n` bits at `at` is free again (joined to its neighbours).
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// The buffer views' written-pages bitmap: the buffer, its mapping (bits
/// as dwords) and host GPU address.
struct Bitmap {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is read and cleared
// only between submissions, which run one at a time.
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

fn bitmap() -> Option<&'static Bitmap> {
    static BITMAP: std::sync::OnceLock<Option<Bitmap>> = std::sync::OnceLock::new();
    BITMAP
        .get_or_init(|| {
            let bytes = u64::from(victory_shader::emit::VIEW_WRITE_BITS / 8);
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages".into());
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Bitmap {
                _buffer: buffer,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes.
pub(crate) fn view_writes() -> Option<u64> {
    bitmap().map(|b| b.gpu)
}

/// After a submission: every page its shaders marked as written through
/// views goes back ([`Shadow::copy_back_marked`]), but for pages in
/// `withheld` (ranges of copies that lists whose writes stay out of title
/// memory declared); the bitmap is cleared. Before the copy-backs of
/// declared ranges, which disarm the copies.
// shortcut: marks do not tell lists apart: a page that a withheld list and
// another both wrote through views goes back unless the withheld list
// declared it.
pub(crate) fn copy_back_marks(withheld: &[(Arc<Shadow>, usize, usize)]) {
    let Some(b) = bitmap() else {
        return;
    };
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
    let words = (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize;
    // SAFETY: the mapped bitmap; the GPU is done with it (one submission
    // runs at a time).
    let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, words) };
    // Marks of offsets past the bitmap land in its last bit, no copy's.
    bits[words - 1] = 0;
    let mut marked = Vec::new();
    let used = (marks().end as usize).div_ceil(32).min(words);
    for (w, word) in bits[..used]
        .iter_mut()
        .enumerate()
        .filter(|(_, word)| **word != 0)
    {
        let mut x = std::mem::take(word);
        while x != 0 {
            marked.push((w * 32) as u32 + x.trailing_zeros());
            x &= x - 1;
        }
    }
    if marked.is_empty() {
        return;
    }
    let pages: Vec<(Arc<Shadow>, usize)> = {
        let m = marks();
        marked
            .iter()
            .filter_map(|&n| {
                let (&base, copy) = m.owners.range(..=n).next_back()?;
                let copy = copy.upgrade()?;
                let k = n - base;
                (k < pages(copy.start, copy.len)).then(|| {
                    let page = (((copy.start >> 16) + u64::from(k)) << 16) as usize;
                    (copy, page)
                })
            })
            .collect()
    };
    pull(
        &pages
            .iter()
            .map(|(copy, page)| (copy.clone(), *page, PAGE))
            .collect::<Vec<_>>(),
    );
    for (copy, page) in pages {
        if withheld
            .iter()
            .any(|(c, a, n)| Arc::ptr_eq(c, &copy) && *a < page + PAGE && page < a + n)
        {
            continue;
        }
        // Lazy copy-back: the page waits, hidden, for the CPU's first look.
        if lazy::on() && lazy::defer(&copy, page) {
            continue;
        }
        let written = copy.copy_back_marked(page);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::MarkedPages, 1);
            crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = written;
    }
}

/// Before the CPU reads what shaders may have written: the title address
/// ranges `(copy, address, len)` of copies in video memory are copied into
/// their mirrors, by the GPU, at once (one list for all; ranges of other
/// copies are skipped).
// shortcut: shaders' writes outside every declared range and marked page
// (to the copy, not to title memory) stay in video memory: the mirror keeps
// the bytes from before until a later range brings them.
pub(crate) fn pull(ranges: &[(Arc<Shadow>, usize, usize)]) {
    // Whole pages, so that the pages pulled are current in the mirror.
    let mut runs: Vec<(&Shadow, usize, usize)> = Vec::new();
    for (copy, address, len) in ranges {
        if copy.vram.is_none() {
            continue;
        }
        let (from, to) = copy.inside(*address, *len);
        if from >= to {
            continue;
        }
        let (from, to) = (
            from / PAGE * PAGE,
            (to.div_ceil(PAGE) * PAGE).min(copy.len as usize),
        );
        runs.push((copy, from, to - from));
    }
    bring_back(&mut runs);
}

/// Before a submission whose lists declared that shaders write title
/// address ranges `(copy, address, len)` (not marked): their snapshots
/// ([`Shadow::writing`]) read the copies' mirrors. The pages there the
/// mirrors are behind on come back now, all in one GPU copy, and until
/// [`snapshots_done`] refreshes there write the mirrors too: else each
/// snapshot waited for a GPU copy of its own.
pub(crate) fn snapshots_ahead(ranges: &[(Arc<Shadow>, usize, usize)]) {
    let mut runs: Vec<(&Shadow, usize, usize)> = Vec::new();
    for (copy, address, len) in ranges {
        let Some(v) = &copy.vram else { continue };
        let (from, to) = copy.inside(*address, *len);
        if from >= to {
            continue;
        }
        let (from, to) = (
            from / PAGE * PAGE,
            (to.div_ceil(PAGE) * PAGE).min(copy.len as usize),
        );
        {
            let mut ahead = v.ahead.lock().unwrap_or_else(|e| e.into_inner());
            ahead.push((from, to));
            ahead.sort_unstable();
            let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ahead.len());
            for &(a, b) in ahead.iter() {
                match merged.last_mut() {
                    Some(m) if a <= m.1 => m.1 = m.1.max(b),
                    _ => merged.push((a, b)),
                }
            }
            *ahead = merged;
        }
        runs.extend(copy.stale_runs(from, to));
    }
    bring_back(&mut runs);
}

/// After the submission's copy-backs: refreshes write video memory alone
/// again ([`snapshots_ahead`]).
pub(crate) fn snapshots_done(ranges: &[(Arc<Shadow>, usize, usize)]) {
    for (copy, _, _) in ranges {
        if let Some(v) = &copy.vram {
            v.ahead.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
    }
}

/// Copies runs `(copy, offset, len)` of mirrored copies (whole pages, or
/// to the copy's end) from video memory into their mirrors, by the GPU,
/// in one list; the pages are current in the mirrors after. Each copy's
/// stale pages stay locked over the GPU copy: a refresh meanwhile marks
/// its page again once the lock is released, so the bit is not lost.
fn bring_back(runs: &mut [(&Shadow, usize, usize)]) {
    if runs.is_empty() {
        return;
    }
    runs.sort_unstable_by_key(|r| (std::ptr::from_ref(r.0) as usize, r.1));
    // Overlapping or adjacent runs of a copy as one region.
    let mut merged: Vec<(&Shadow, usize, usize)> = Vec::with_capacity(runs.len());
    for &(copy, at, n) in runs.iter() {
        if let Some(last) = merged.last_mut()
            && std::ptr::eq(last.0, copy)
            && at <= last.1 + last.2
        {
            last.2 = last.2.max(at + n - last.1);
            continue;
        }
        merged.push((copy, at, n));
    }
    // Locked in address order (the runs are sorted by copy).
    let mut locked = Vec::new();
    let mut regions = Vec::with_capacity(merged.len());
    for &(copy, at, n) in &merged {
        let Some(v) = &copy.vram else { continue };
        if !locked.last().is_some_and(|(c, _)| std::ptr::eq(*c, copy)) {
            locked.push((copy, v.stale.lock().unwrap_or_else(|e| e.into_inner())));
        }
        let bits = &mut locked.last_mut().expect("just pushed").1;
        for k in at / PAGE..(at + n).div_ceil(PAGE) {
            bits[k / 64] &= !(1 << (k % 64));
        }
        regions.push((&v.mirror, at as u64, &copy.buffer, at as u64, n as u64));
    }
    #[cfg(test)]
    if !regions.is_empty() {
        BROUGHT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(host) = host() {
        host.copy_now(&regions);
    }
}

/// GPU copies [`bring_back`] made (tests).
#[cfg(test)]
static BROUGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
    /// The title made it without unordered access: shaders only read it,
    /// so its copy may be in a GPU upload heap ([`register_read_only`]).
    read_only: bool,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    register_as(resource, start, len, false);
}

/// [`register`] for a buffer the title made without unordered access:
/// shaders only read it, and the CPU only writes its copy (refreshes), so
/// the copy can be in video memory the CPU writes through the PCI BAR
/// ([`Host::gpu_upload_buffer`](crate::host::Host::gpu_upload_buffer)).
pub(crate) fn register_read_only(resource: usize, start: u64, len: u64) {
    register_as(resource, start, len, true);
}

fn register_as(resource: usize, start: u64, len: u64, read_only: bool) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
            read_only,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let removed = buffers().remove(&(start, resource));
    if let Some(Entry { shadow: Some(copy), .. }) = removed {
        // Its memory may go next: what the GPU wrote goes there first.
        lazy::flush_copy(&copy);
        changed();
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access): only the others'
        // copies may be in upload memory.
        let size = len.next_multiple_of(1 << 16);
        let host = host()?;
        let upload = entry
            .read_only
            .then(|| host.gpu_upload_buffer(size))
            .flatten();
        let in_upload = upload.is_some();
        // A buffer shaders may write: in video memory too, with a mirror.
        let mut vram = None;
        let (buffer, data) = match upload {
            Some(made) => made,
            None => {
                let mirror = host.shared_buffer(size)?;
                match (!entry.read_only && host.gpu_upload_writes())
                    .then(|| host.gpu_upload_buffer(size))
                    .flatten()
                {
                    Some((video, video_data)) => {
                        vram = Some(Vram {
                            mirror: mirror.0,
                            data: video_data,
                            stale: Mutex::new(vec![0; (size as usize).div_ceil(PAGE).div_ceil(64)]),
                            ahead: Mutex::new(Vec::new()),
                        });
                        (video, mirror.1)
                    }
                    None => mirror,
                }
            }
        };
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let shadow = Shadow::with_vram(buffer, data, gpu, key.0, len, vram);
        shadow
            .upload
            .store(in_upload, std::sync::atomic::Ordering::Relaxed);
        entry.shadow = Some(shadow);
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Shadow::new(buffer, data, gpu, start, len);
    map.insert((start, len), shadow.clone());
    changed();
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Shadow::new(buffer, data, gpu, address, len);
    map.insert((address, len), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// Every host copy of title memory made so far (buffers and windows).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

/// Lazy copy-back (`VICTORY_LAZY_COPY_BACK`, on unless `0`): a page shaders marked as
/// written through views of a copy does not go back into title memory
/// after the submission. Its title mappings (every address showing it) are
/// hidden instead (no access), and the copy keeps the GPU's bytes: the
/// page is pending. The first CPU access faults ([`fault`]) and runs the
/// page's copy-back ([`Shadow::copy_back_marked`]), then the access runs
/// again. Writers that do not fault (the kernel, our own reads and writes
/// of title memory, mappings, new trackers) bring pending pages in their
/// range back first ([`flush_range`]).
///
/// While a page is pending, title memory there is the copy as it was before
/// the first submission that wrote it (the title cannot write it), so the
/// copy-back's comparison finds every byte the GPU wrote since, however
/// many submissions marked it again. Other copies and textures holding the
/// page ([`crate::watch::Other`]) bring it back before they take title
/// memory there (the copy-back marks them changed where the GPU's bytes
/// differ, as right after the submission); a tracker made later brings it
/// back first. Pages a descriptor heap holds go back at once (its sync takes
/// pages under the heaps' lock). Another copy marking a pending page joins
/// it if its copy there is title memory's (not changed since it took it),
/// else the page goes back first; with several copies, each one's bytes
/// are told from the page as it was hidden. A page the copy holds in part
/// is hidden whole (the title's bytes beside the copy wait with it) and only
/// the copy's part goes back.
///
/// shortcut: a page pending when a later submission's writes to it are
/// withheld (shaders read memory with no copy) puts those back with the
/// earlier ones. The kernel reading a pending page for the title (a
/// `WriteFile` from it) fails instead of faulting.
pub(crate) mod lazy {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
    use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

    use super::Shadow;
    use crate::watch::PAGE;

    /// Whether copy-backs of marked pages wait for the CPU's first look.
    pub(crate) fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::switch::on_by_default("VICTORY_LAZY_COPY_BACK"))
    }

    /// Where pending pages were brought back from.
    #[derive(Clone, Copy)]
    pub(crate) enum Path {
        /// A CPU access faulted on a hidden page.
        Fault,
        /// The kernel or our own write into title memory ([`crate::watch::writing`]).
        KernelWrite,
        /// Memory mapped or unmapped there.
        Remap,
        /// The pages mapped at another address too.
        Alias,
        /// A new tracker (copy, texture, heap) of the memory.
        Track,
        /// Our own reads of title memory (`crate::host::read_into`).
        HostRead,
        /// Our own copies from title memory (`crate::host::copy_mapped`).
        HostCopy,
        /// DirectStorage writing the physical pages through its own views.
        Physical,
        /// The title released a buffer whose copy holds pending pages.
        Unregister,
        /// A copy-back of declared ranges or computed addresses there.
        CopyBack,
        /// Another tracker holding the page takes title memory there.
        Holder,
    }

    const PATHS: [&str; 11] = [
        "CPU faults",
        "kernel writes",
        "remaps",
        "new aliases",
        "new trackers",
        "host reads",
        "host copies",
        "physical-page writes",
        "released buffers",
        "other copy-backs",
        "other holders' refreshes",
    ];

    /// A pending page: the copies (and their page) holding its GPU bytes,
    /// the pages hidden (it and its other mappings, page numbers), the
    /// thread bringing it back if one is, and the frame it was deferred in.
    struct Group {
        copies: Vec<(Arc<Shadow>, usize)>,
        pages: Vec<usize>,
        busy: Option<usize>,
        frame: u64,
        /// The other trackers holding it, until it is back.
        _others: Vec<crate::watch::Other>,
    }

    #[derive(Default)]
    struct Lazy {
        /// By page (address).
        groups: HashMap<usize, Group>,
        /// Each hidden page (page number): its group. Ordered: the hooks ask
        /// for ranges (every refresh copy and read of title memory).
        at: std::collections::BTreeMap<usize, usize>,
    }

    static LAZY: LazyLock<std::sync::RwLock<Lazy>> = LazyLock::new(Default::default);

    fn lazy() -> std::sync::RwLockWriteGuard<'static, Lazy> {
        LAZY.write().unwrap_or_else(|e| e.into_inner())
    }

    fn lazy_read() -> std::sync::RwLockReadGuard<'static, Lazy> {
        LAZY.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Pending pages (groups): the hooks cost nothing when there are none.
    pub(super) static PENDING: AtomicUsize = AtomicUsize::new(0);

    /// Whether any page is pending.
    pub(crate) fn any() -> bool {
        PENDING.load(Relaxed) != 0
    }

    /// Counters since the last log line (some logged by diagnostics only).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    enum K {
        Deferred,
        Remarked,
        Joined,
        WithCopies,
        WithTextures,
        Partial,
        AliasPending,
        Aliases,
        Unguarded,
        Busy,
        Heap,
        Protect,
        TitleSaved,
        TitleChanged,
        StaleJoin,
        FlushUs,
        GateUs,
        FaultSubmit,
        Age0,
        Age1,
        AgeMore,
        AgeFrames,
        AtPresent,
        Presents,
    }
    const COUNTERS: usize = 24;
    static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];
    static FLUSHED: [AtomicU64; PATHS.len()] = [const { AtomicU64::new(0) }; PATHS.len()];

    fn add(k: K, n: u64) {
        COUNT[k as usize].fetch_add(n, Relaxed);
    }

    /// This thread, as a number.
    fn me() -> usize {
        thread_local! { static ME: u8 = const { 0 }; }
        ME.with(|m| std::ptr::from_ref(m) as usize)
    }

    thread_local! {
        /// This thread submits or holds [`GATE`]: its copy-backs need no
        /// wait for the GPU.
        static GATED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Held while the GPU runs a submission, and by lazy copy-backs on
    /// other threads than the submitting one: those write copies, which a
    /// running submission's shaders may be writing.
    static GATE: Mutex<()> = Mutex::new(());

    /// Frames presented so far (diagnostics; 0 otherwise).
    fn frame_now() -> u64 {
        #[cfg(feature = "diagnostics")]
        return crate::queue::presented();
        #[cfg(not(feature = "diagnostics"))]
        0
    }

    /// The submitting thread, while it holds the submission lock.
    pub(crate) struct Submitting(bool);

    impl Submitting {
        pub(crate) fn enter() -> Self {
            Submitting(GATED.replace(true))
        }
    }

    impl Drop for Submitting {
        fn drop(&mut self) {
            GATED.set(self.0);
        }
    }

    /// While the GPU runs a submission (lazy copy-back on).
    pub(crate) fn gpu_running() -> Option<MutexGuard<'static, ()>> {
        on().then(|| GATE.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// [`GATE`] for a copy-back on this thread, unless it has it already.
    struct Gate(Option<MutexGuard<'static, ()>>);

    impl Gate {
        fn take() -> Self {
            if GATED.get() {
                return Gate(None);
            }
            let waited = std::time::Instant::now();
            let guard = GATE.lock().unwrap_or_else(|e| e.into_inner());
            add(K::GateUs, waited.elapsed().as_micros() as u64);
            GATED.set(true);
            Gate(Some(guard))
        }
    }

    impl Drop for Gate {
        fn drop(&mut self) {
            if self.0.is_some() {
                GATED.set(false);
            }
        }
    }

    /// After a submission (its marked pages pulled into the mirrors): the
    /// marked page at `page` of `copy` waits, hidden, instead of going back.
    /// False where it must go back now: the title wrote it during the
    /// submission, a descriptor heap holds it, it cannot be hidden, or it
    /// is pending already for other copies and this one's copy of it is not
    /// title memory's (then it went back first).
    pub(crate) fn defer(copy: &Arc<Shadow>, page: usize) -> bool {
        let me = me();
        loop {
            let mut l = lazy();
            let Some(&c) = l.at.get(&(page / PAGE)) else {
                break;
            };
            let g = l.groups.get_mut(&c).expect("indexed pages have their group");
            if g.busy.is_none() {
                // Marked again while pending: still pending, title memory
                // still as before the first of them.
                if !g.copies.iter().any(|(x, p)| Arc::ptr_eq(x, copy) && *p == page) {
                    // Another copy (or mapping): its bytes before are title
                    // memory's only if it took the page since it changed.
                    let fresh = copy
                        .tracked
                        .as_ref()
                        .is_some_and(|t| !t.changed(page / PAGE) && !t.has_saved(page / PAGE));
                    if !fresh {
                        drop(l);
                        flush(c, Path::CopyBack);
                        add(K::StaleJoin, 1);
                        return false;
                    }
                    g.copies.push((copy.clone(), page));
                    add(K::Joined, 1);
                }
                add(K::Remarked, 1);
                return true;
            }
            drop(l);
            // Another thread brings it back: then it goes as any page.
            std::thread::yield_now();
        }
        let Some(t) = &copy.tracked else {
            return false;
        };
        let (start, end) = (copy.start as usize, (copy.start + copy.len) as usize);
        if page + PAGE <= start || page >= end {
            return false;
        }
        if t.has_saved(page / PAGE) {
            add(K::TitleSaved, 1);
            return false;
        }
        if t.changed(page / PAGE) {
            add(K::TitleChanged, 1);
            return false;
        }
        // Indexed (busy) before it is hidden: a fault meanwhile waits.
        let pages = crate::watch::alias_pages(page / PAGE);
        {
            let mut l = lazy();
            for &p in &pages {
                if l.at.contains_key(&p) {
                    // Another mapping of it is pending already (as another
                    // copy's): this one goes back now.
                    add(K::AliasPending, 1);
                    return false;
                }
            }
            for &p in &pages {
                l.at.insert(p, page);
            }
            l.groups.insert(
                page,
                Group {
                    copies: vec![(copy.clone(), page)],
                    pages: pages.clone(),
                    busy: Some(me),
                    frame: frame_now(),
                    _others: Vec::new(),
                },
            );
            PENDING.store(l.groups.len(), Relaxed);
        }
        let hidden = t.hide(page / PAGE, &pages);
        let mut l = lazy();
        match hidden {
            Ok(others) => {
                t.unarm(page / PAGE);
                if others.iter().any(|o| o.kind() == crate::watch::Kind::Copy) {
                    add(K::WithCopies, 1);
                }
                if others.iter().any(|o| o.kind() == crate::watch::Kind::Texture) {
                    add(K::WithTextures, 1);
                }
                if page < start || page + PAGE > end {
                    add(K::Partial, 1);
                }
                if let Some(g) = l.groups.get_mut(&page) {
                    g.busy = None;
                    g._others = others;
                }
                add(K::Deferred, 1);
                true
            }
            Err(why) => {
                l.groups.remove(&page);
                for p in &pages {
                    l.at.remove(p);
                }
                PENDING.store(l.groups.len(), Relaxed);
                use crate::watch::Refused;
                add(
                    match why {
                        Refused::Aliases => K::Aliases,
                        Refused::Unguarded => K::Unguarded,
                        Refused::Busy => K::Busy,
                        Refused::Heap => K::Heap,
                        Refused::Protect => K::Protect,
                    },
                    1,
                );
                false
            }
        }
    }

    /// A CPU access faulted at `at`: if a hidden page, it is brought back
    /// (or was, by another thread) and the access may run again.
    pub(crate) fn fault(at: usize) -> bool {
        let c = match lazy_read().at.get(&(at / PAGE)) {
            Some(&c) => c,
            None => return false,
        };
        if GATED.get() {
            add(K::FaultSubmit, 1);
        }
        flush(c, Path::Fault)
    }

    /// Brings back the pending pages in title memory `[at, at + len)` and
    /// at every other address showing them.
    pub(crate) fn flush_range(at: usize, len: usize, path: Path) {
        if len == 0 || PENDING.load(Relaxed) == 0 {
            return;
        }
        let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
        let mut groups: Vec<usize> = {
            let l = lazy_read();
            let mut hit = l.at.range(first..last).map(|(_, &c)| c).peekable();
            if hit.peek().is_none() {
                return;
            }
            hit.collect()
        };
        groups.sort_unstable();
        groups.dedup();
        for c in groups {
            flush(c, path);
        }
    }

    /// Brings back the pending pages `copy` holds (its buffer is released).
    pub(crate) fn flush_copy(copy: &Arc<Shadow>) {
        if PENDING.load(Relaxed) == 0 {
            return;
        }
        let groups: Vec<usize> = lazy_read()
            .groups
            .iter()
            .filter(|(_, g)| g.copies.iter().any(|(x, _)| Arc::ptr_eq(x, copy)))
            .map(|(&c, _)| c)
            .collect();
        for c in groups {
            flush(c, Path::Unregister);
        }
    }

    /// Brings pending page `c` back: shown read-only again, and its copies'
    /// copy-backs run as after a submission. Waits while another thread
    /// does it. False if this thread is doing it already.
    fn flush(c: usize, path: Path) -> bool {
        let me = me();
        let (copies, pages, frame) = loop {
            let mut l = lazy();
            let Some(g) = l.groups.get_mut(&c) else {
                return true;
            };
            match g.busy {
                None => {
                    g.busy = Some(me);
                    break (g.copies.clone(), g.pages.clone(), g.frame);
                }
                Some(o) if o == me => return false,
                Some(_) => {}
            }
            drop(l);
            std::thread::yield_now();
        };
        let started = std::time::Instant::now();
        {
            let _gate = Gate::take();
            crate::watch::unhide(&pages);
            // Several copies: each one's bytes before are the page as it was
            // hidden, not as the copies before it in turn leave it.
            let base = (copies.len() > 1)
                .then(|| {
                    let mut base = vec![0u8; PAGE].into_boxed_slice();
                    crate::host::read_unflushed(c as *const u8, &mut base).then_some(base)
                })
                .flatten();
            for (copy, page) in &copies {
                let armed = copy.tracked.as_ref().is_some_and(|t| t.armed(page / PAGE));
                copy.copy_back_marked_from(*page, base.as_deref());
                // Armed for a submission prepared meanwhile: still.
                if armed && let Some(t) = &copy.tracked {
                    t.arm(*page, page + PAGE);
                }
            }
        }
        {
            let mut l = lazy();
            l.groups.remove(&c);
            for p in &pages {
                l.at.remove(p);
            }
            PENDING.store(l.groups.len(), Relaxed);
        }
        add(K::FlushUs, started.elapsed().as_micros() as u64);
        FLUSHED[path as usize].fetch_add(1, Relaxed);
        let age = frame_now().saturating_sub(frame);
        add(
            match age {
                0 => K::Age0,
                1 => K::Age1,
                _ => K::AgeMore,
            },
            1,
        );
        add(K::AgeFrames, age);
        true
    }

    /// At each presented frame (diagnostics): pages pending then.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn frame_done() {
        add(K::AtPresent, PENDING.load(Relaxed) as u64);
        add(K::Presents, 1);
    }

    /// Logs the counters per frame over `frames` frames (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn log(frames: f64) {
        if !on() {
            return;
        }
        let c: Vec<f64> = COUNT.iter().map(|a| a.swap(0, Relaxed) as f64).collect();
        let f: Vec<f64> = FLUSHED.iter().map(|a| a.swap(0, Relaxed) as f64).collect();
        let flushed: f64 = f.iter().sum();
        let paths: Vec<String> = PATHS
            .iter()
            .zip(&f)
            .filter(|(_, n)| **n > 0.0)
            .map(|(name, n)| format!("{name} {:.1}", n / frames))
            .collect();
        let per = |k: K| c[k as usize] / frames;
        tracing::info!(
            "lazy copy-back per frame: {:.0} pages deferred ({:.0} with other copies holding them, {:.0} with textures, {:.0} held in part), {:.0} marked again while pending ({:.0} by another copy); {:.1} brought back ({}) in {:.2} ms (waiting for the GPU {:.2} ms; {:.1} faults on the submitting thread); not deferred: {:.0} title-written ({:.0} saved), {:.0} heap-held, {:.0} unguarded, {:.0} busy, {:.0} other mapping pending, {:.0} aliases moved, {:.0} not protectable, {:.0} stale copy joining; age when brought back: {:.1} same frame, {:.1} next, {:.1} later (mean {:.1} frames); {:.0} pages pending at present",
            per(K::Deferred),
            per(K::WithCopies),
            per(K::WithTextures),
            per(K::Partial),
            per(K::Remarked),
            per(K::Joined),
            flushed / frames,
            paths.join(", "),
            per(K::FlushUs) / 1e3,
            per(K::GateUs) / 1e3,
            per(K::FaultSubmit),
            per(K::TitleChanged) + per(K::TitleSaved),
            per(K::TitleSaved),
            per(K::Heap),
            per(K::Unguarded),
            per(K::Busy),
            per(K::AliasPending),
            per(K::Aliases),
            per(K::Protect),
            per(K::StaleJoin),
            per(K::Age0),
            per(K::Age1),
            per(K::AgeMore),
            c[K::AgeFrames as usize] / flushed.max(1.0),
            c[K::AtPresent as usize] / c[K::Presents as usize].max(1.0),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::*;

    /// Runs this binary's test `name` again in a child process with
    /// `VICTORY_GPU_UPLOAD=1`, and `VICTORY_GPU_UPLOAD_WRITES=1` if `writes`
    /// (the switches are read once per process): true
    /// in that child, where the test does its work; false in the parent,
    /// once the child passed.
    fn rerun_with_gpu_upload(name: &str, writes: bool) -> bool {
        const CHILD: &str = "VICTORY_TEST_GPU_UPLOAD_CHILD";
        if std::env::var_os(CHILD).is_some() {
            return true;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                name,
                "--exact",
                "--test-threads=1",
                "--nocapture",
                "--include-ignored",
            ])
            .env(CHILD, "1")
            .env("VICTORY_GPU_UPLOAD", "1")
            .env("VICTORY_GPU_UPLOAD_WRITES", if writes { "1" } else { "0" })
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "{name} with VICTORY_GPU_UPLOAD=1:\n{text}"
        );
        assert!(text.contains("1 passed"), "{name} did not run:\n{text}");
        eprintln!("{text}");
        false
    }

    /// Read-only buffers come and go, more bytes of them in all than the
    /// GPU upload budget, a few at a time: each one's copy still goes in a
    /// GPU upload heap, and its bytes leave the budget once the last
    /// reference to its buffer is gone (a clone, as descriptor tables hold,
    /// keeps them charged). Counted for the whole run, copies went to
    /// system memory for good after the budget's worth (#2120).
    #[test]
    fn upload_budget_is_given_back_when_copies_go() {
        use windows::core::Interface;
        if !rerun_with_gpu_upload(
            "shadow::tests::upload_budget_is_given_back_when_copies_go",
            false,
        ) {
            return;
        }
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !host.gpu_upload() {
            return; // no GPU upload heaps (resizable BAR) here
        }
        const LEN: usize = 128 << 20;
        let p = crate::watch::tests::pool(LEN / PAGE);
        let base = crate::host::upload_bytes();
        // 2.5 GiB in all, past the 2 GiB budget.
        for round in 0..20 {
            register_read_only(p, p as u64, LEN as u64);
            let copy = find(p as u64, LEN as u64).expect("a copy");
            assert_eq!(
                heap_type(&copy),
                D3D12_HEAP_TYPE_GPU_UPLOAD,
                "round {round}: in a GPU upload heap"
            );
            assert_eq!(
                crate::host::upload_bytes(),
                base + LEN as u64,
                "round {round}: charged"
            );
            // A clone of the buffer, as a descriptor table holds, outlives
            // the copy: still charged until it goes.
            let held = copy.buffer.clone();
            let weak = Arc::downgrade(&copy);
            unregister(p, p as u64);
            drop(copy);
            assert!(weak.upgrade().is_none(), "round {round}: the copy is gone");
            assert_eq!(
                crate::host::upload_bytes(),
                base + LEN as u64,
                "round {round}: held"
            );
            assert!(!held.as_raw().is_null());
            drop(held);
            assert_eq!(
                crate::host::upload_bytes(),
                base,
                "round {round}: given back"
            );
        }
        crate::watch::tests::free(p);
    }

    /// A buffer shaders may write, its copy made (mirrored in video memory
    /// when the switches are on), and title memory filled with a pattern
    /// the copy then took. Returns the pool, its length and the copy.
    fn mirrored_buffer(pages: usize) -> (usize, usize, Arc<Shadow>) {
        let p = crate::watch::tests::pool(pages);
        let len = pages * PAGE;
        register(p, p as u64, len as u64);
        let copy = find(p as u64, len as u64).expect("a copy");
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, len) };
        for (i, b) in title.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        copy.update(p, len);
        (p, len, copy)
    }

    /// Records into `list` the GPU writing `bytes` into the copy at title
    /// address `at` (a copy from a buffer of ours, as a shader would store).
    fn gpu_store(list: &HostList, copy: &Shadow, at: usize, bytes: &[u8]) {
        let host = host().unwrap();
        let (src, data) = host.shared_buffer(bytes.len() as u64).unwrap();
        // SAFETY: our mapped buffer of `bytes.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
        list.keep(&src);
        // SAFETY: an open list; both buffers live and common.
        unsafe {
            list.list.CopyBufferRegion(
                &copy.buffer,
                copy.offset(at as u64),
                &src,
                0,
                bytes.len() as u64,
            )
        };
    }

    /// The bytes of the copy as the GPU sees them (video memory, or the
    /// system memory copy), copied out by the GPU.
    fn gpu_bytes(copy: &Shadow, at: usize, n: usize) -> Vec<u8> {
        let host = host().unwrap();
        let (out, data) = host.shared_buffer(n as u64).unwrap();
        host.copy_now(&[(&out, 0, &copy.buffer, copy.offset(at as u64), n as u64)]);
        // SAFETY: our mapped buffer, the copy done.
        unsafe { std::slice::from_raw_parts(data, n) }.to_vec()
    }

    /// With `VICTORY_GPU_UPLOAD_WRITES=1` too, the copy of a buffer shaders
    /// may write is in a GPU upload heap, mirrored in system memory: the
    /// GPU sees what a refresh wrote, and the mirror holds the same.
    #[test]
    fn written_buffers_copies_are_in_video_memory_with_a_mirror() {
        if !rerun_with_gpu_upload(
            "shadow::tests::written_buffers_copies_are_in_video_memory_with_a_mirror",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return; // no GPU upload heaps here
        }
        let (p, len, copy) = mirrored_buffer(2);
        assert!(copy.mirrored());
        assert_eq!(heap_type(&copy), D3D12_HEAP_TYPE_GPU_UPLOAD);
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        assert!(
            gpu_bytes(&copy, p, len) == title,
            "the GPU sees the refreshed bytes"
        );
        // The refresh wrote video memory only; the mirror catches up when
        // the CPU reads it.
        // SAFETY: the mirror's bytes (read again after `freshen` below).
        let mirror = unsafe { std::slice::from_raw_parts(copy.data, len) };
        assert!(mirror != title, "the refresh did not write the mirror");
        copy.freshen(0, len);
        // SAFETY: as above.
        let mirror = unsafe { std::slice::from_raw_parts(copy.data, len) };
        assert!(mirror == title, "the mirror has caught up");
        // The title writes again; the next refresh reaches video memory.
        // SAFETY: our pool's bytes.
        unsafe { ((p + PAGE + 100) as *mut u8).write_volatile(0xA5) };
        copy.update(p, len);
        assert_eq!(gpu_bytes(&copy, p + PAGE + 100, 1), [0xA5]);
        unregister(p, p as u64);
    }

    /// End to end through a submission: the GPU writes the copy in video
    /// memory, in a range a list declared (`marked`: and in a page shaders
    /// marked as written), while the title writes another byte of the same
    /// page; afterwards title memory holds both writes. Without the mirror
    /// being brought up to date (`pull`) the GPU's bytes would never reach
    /// it, nor title memory.
    fn gpu_writes_reach_title_memory(marked: bool) {
        let host = host().unwrap();
        let (p, len, copy) = mirrored_buffer(2);
        assert!(copy.mirrored());
        let list = host.new_list().unwrap();
        let (at, gpu) = (p + PAGE + 64, [0xC3u8; 16]);
        copy.refresh(&list, p as u64, len as u64);
        copy.gpu_writes(&list, p as u64, len as u64, marked);
        gpu_store(&list, &copy, at, &gpu);
        list.close();
        // The title writes the same page meanwhile, past the GPU's bytes.
        // SAFETY: our pool's bytes.
        unsafe { ((p + PAGE + 200) as *mut u8).write_volatile(0x3C) };
        if marked {
            // The shaders' mark for the page (the GPU copy sets none).
            let (bit, _) = copy.mark_entry(at as u64).expect("marked memory");
            let words = bitmap().unwrap().data as *mut u32;
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *words.add(bit as usize / 32) |= 1 << (bit % 32) };
        }
        host.execute(&[&list]);
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        assert_eq!(
            title[PAGE + 64..PAGE + 80],
            gpu,
            "the GPU's bytes (marked {marked})"
        );
        assert_eq!(
            title[PAGE + 200],
            0x3C,
            "the title's byte (marked {marked})"
        );
        assert_eq!(
            title[PAGE + 63],
            ((PAGE + 63) % 251) as u8,
            "the rest as it was"
        );
        assert_eq!(
            gpu_bytes(&copy, p + PAGE + 64, 16),
            gpu,
            "video memory keeps them"
        );
        unregister(p, p as u64);
    }

    #[test]
    fn gpu_writes_to_mirrored_copies_reach_title_memory() {
        if !rerun_with_gpu_upload(
            "shadow::tests::gpu_writes_to_mirrored_copies_reach_title_memory",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return;
        }
        gpu_writes_reach_title_memory(false);
        if bitmap().is_some() {
            gpu_writes_reach_title_memory(true);
        }
    }

    /// A list whose writes stay out of title memory (it read memory with no
    /// copy): the copy keeps the GPU's bytes, and so does its mirror (the
    /// CPU's next look at the range sees them), while title memory keeps
    /// its own.
    #[test]
    fn withheld_writes_stay_in_the_copy_and_its_mirror() {
        if !rerun_with_gpu_upload(
            "shadow::tests::withheld_writes_stay_in_the_copy_and_its_mirror",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return;
        }
        let (p, len, copy) = mirrored_buffer(1);
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, len as u64);
        copy.gpu_writes(&list, p as u64, len as u64, false);
        gpu_store(&list, &copy, p + 8, &[0x99; 8]);
        list.defer_missed(|| true);
        list.close();
        host.execute(&[&list]);
        // SAFETY: our pool's bytes, and the mirror's.
        let (title, mirror) = unsafe {
            (
                std::slice::from_raw_parts(p as *const u8, len),
                std::slice::from_raw_parts(copy.data, len),
            )
        };
        assert_eq!(title[8], 8, "title memory keeps its byte");
        assert_eq!(mirror[8..16], [0x99; 8], "the mirror has the GPU's");
        unregister(p, p as u64);
    }

    /// The mirror falls behind where a refresh wrote video memory only, and
    /// catches up on what the CPU reads, and only there: a read of a page
    /// brings that page back; pages GPU writes are pulled for come back
    /// whole; a refresh where a snapshot is ahead writes both.
    #[test]
    fn mirrors_catch_up_on_the_pages_read() {
        if !rerun_with_gpu_upload("shadow::tests::mirrors_catch_up_on_the_pages_read", true) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return;
        }
        let (p, len, copy) = mirrored_buffer(4);
        assert_eq!(
            copy.stale_pages(),
            [0, 1, 2, 3],
            "the refresh wrote video memory only"
        );
        // SAFETY: our pool's bytes, and the mirror's.
        let (title, mirror) = unsafe {
            (
                std::slice::from_raw_parts(p as *const u8, len),
                std::slice::from_raw_parts(copy.data, len),
            )
        };
        // A read of a few bytes of page 1.
        assert_eq!(
            copy.read_host((p + PAGE + 8) as u64, 1)[0].to_le_bytes(),
            title[PAGE + 8..PAGE + 12]
        );
        assert_eq!(copy.stale_pages(), [0, 2, 3]);
        assert!(
            mirror[PAGE..2 * PAGE] == title[PAGE..2 * PAGE],
            "page 1 came back whole"
        );
        assert!(
            mirror[2 * PAGE..3 * PAGE] != title[2 * PAGE..3 * PAGE],
            "page 2 did not"
        );
        // Bytes the GPU wrote in page 3 are pulled: the whole page comes.
        pull(&[(copy.clone(), p + 3 * PAGE + 100, 4)]);
        assert_eq!(copy.stale_pages(), [0, 2]);
        assert!(mirror[3 * PAGE..] == title[3 * PAGE..]);
        // A snapshot ahead over page 2: the refresh there writes both.
        let ahead = [(copy.clone(), p + 2 * PAGE + 10, 10)];
        snapshots_ahead(&ahead);
        assert_eq!(copy.stale_pages(), [0], "page 2 brought back up front");
        // SAFETY: our pool's bytes.
        unsafe {
            ((p + 2 * PAGE + 500) as *mut u8).write_volatile(0x11);
            ((p + 40) as *mut u8).write_volatile(0x22);
        }
        copy.update(p, len);
        assert_eq!(copy.stale_pages(), [0], "page 2 stays current");
        assert_eq!(mirror[2 * PAGE + 500], 0x11);
        snapshots_done(&ahead);
        // SAFETY: our pool's bytes.
        unsafe { ((p + 2 * PAGE + 600) as *mut u8).write_volatile(0x33) };
        copy.update(p, len);
        assert_eq!(copy.stale_pages(), [0, 2], "video memory only again");
        assert_eq!(gpu_bytes(&copy, p + 2 * PAGE + 600, 1), [0x33]);
        copy.freshen(0, len);
        assert!(copy.stale_pages().is_empty());
        assert!(mirror == title, "the mirror has all of it");
        unregister(p, p as u64);
    }

    /// A submission whose list declares shaders writing several ranges,
    /// apart, of a copy the refresh left behind in its mirror: their
    /// snapshots come back in one GPU copy, and what shaders write in
    /// another in one more (not one per range); title memory then has the
    /// GPU's bytes and the title's.
    #[test]
    fn snapshots_of_mirrored_copies_come_back_at_once() {
        if !rerun_with_gpu_upload(
            "shadow::tests::snapshots_of_mirrored_copies_come_back_at_once",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return;
        }
        let (p, len, copy) = mirrored_buffer(4);
        assert_eq!(copy.stale_pages().len(), 4);
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, len as u64);
        for k in 0..4 {
            copy.gpu_writes(&list, (p + k * PAGE + 64) as u64, 256, false);
        }
        gpu_store(&list, &copy, p + 2 * PAGE + 100, &[0xE7; 8]);
        list.close();
        // The title writes page 3, inside its range, before the submission.
        // SAFETY: our pool's bytes.
        unsafe { ((p + 3 * PAGE + 70) as *mut u8).write_volatile(0x5A) };
        let before = BROUGHT.load(std::sync::atomic::Ordering::Relaxed);
        host.execute(&[&list]);
        let copies = BROUGHT.load(std::sync::atomic::Ordering::Relaxed) - before;
        assert_eq!(copies, 2, "the snapshots' pages, then what the GPU wrote");
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        assert_eq!(
            title[2 * PAGE + 100..2 * PAGE + 108],
            [0xE7; 8],
            "the GPU's bytes"
        );
        assert_eq!(title[3 * PAGE + 70], 0x5A, "the title's byte");
        assert_eq!(title[64], 64, "the rest as it was");
        unregister(p, p as u64);
    }

    /// A range declared with marks, over a copy the refresh left behind in
    /// its mirror: only the page shaders marked comes back (one GPU copy),
    /// not the range, and its bytes reach title memory.
    #[test]
    fn marked_ranges_of_mirrored_copies_bring_back_their_marked_pages_only() {
        if !rerun_with_gpu_upload(
            "shadow::tests::marked_ranges_of_mirrored_copies_bring_back_their_marked_pages_only",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() || bitmap().is_none() {
            return;
        }
        let (p, len, copy) = mirrored_buffer(4);
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, len as u64);
        copy.gpu_writes(&list, p as u64, len as u64, true);
        let at = p + 2 * PAGE + 100;
        gpu_store(&list, &copy, at, &[0x4D; 8]);
        list.close();
        let (bit, _) = copy.mark_entry(at as u64).expect("marked memory");
        let words = bitmap().unwrap().data as *mut u32;
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *words.add(bit as usize / 32) |= 1 << (bit % 32) };
        let before = BROUGHT.load(std::sync::atomic::Ordering::Relaxed);
        host.execute(&[&list]);
        let copies = BROUGHT.load(std::sync::atomic::Ordering::Relaxed) - before;
        assert_eq!(copies, 1, "the marked page alone");
        assert_eq!(copy.stale_pages(), [0, 1, 3], "the other pages stay behind");
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        assert_eq!(title[2 * PAGE + 100..2 * PAGE + 108], [0x4D; 8]);
        unregister(p, p as u64);
    }

    /// The copy's tests against system memory copies, again with copies in
    /// video memory and their mirrors behind (snapshots, refreshes under a
    /// busy range, copy-back by bytes and by marked pages).
    #[test]
    fn copy_tests_pass_with_mirrored_copies() {
        if std::env::var_os("VICTORY_TEST_GPU_UPLOAD_CHILD").is_some() {
            return;
        }
        for name in [
            "shadow::tests::title_writes_under_a_busy_range_are_copied",
            "shadow::tests::snapshots_cover_the_declared_ranges_and_writes_go_back_once",
            "shadow::tests::marked_pages_go_back_by_the_bytes_the_gpu_changed",
        ] {
            rerun_with_gpu_upload(name, true);
        }
    }

    /// A shader reading and writing 256 MiB of a written buffer's copy, in
    /// system memory and (`VICTORY_GPU_UPLOAD_WRITES=1`) in video memory, and
    /// what bringing 64 marked pages back to the mirror costs.
    /// `cargo test --release -p victory-gpu --lib written_copies -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark: run in release with --ignored --nocapture"]
    fn written_copies_in_video_and_system_memory() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        if !rerun_with_gpu_upload(
            "shadow::tests::written_copies_in_video_and_system_memory",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            eprintln!("no GPU upload heaps here");
            return;
        }
        const SRC: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer b : register(u0);
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // Read 16 dwords, write them back incremented: 64 bytes read and
    // written per thread.
    for (uint k = 0; k < 16; k++) {
        uint at = (id.x * 16 + k) * 4;
        b.Store(at, b.Load(at) + 1);
    }
}"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source outlives the call.
        unsafe {
            D3DCompile(
                SRC.as_ptr().cast(),
                SRC.len(),
                s!("b"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: the blob's bytes; plain creation.
        let (root, pso) = unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        const BYTES: usize = 256 << 20;
        // The system memory copy: a mirrored copy's mirror, used by the GPU
        // as the copies are without the switch.
        let p = crate::watch::tests::pool(BYTES / PAGE);
        register(p, p as u64, BYTES as u64);
        let copy = find(p as u64, BYTES as u64).unwrap();
        assert!(copy.mirrored());
        copy.update(p, BYTES);
        let mirror = &copy.vram.as_ref().unwrap().mirror;
        for (label, buffer) in [
            ("system memory (as without the switch)", mirror),
            ("video memory (GPU upload heap)", &copy.buffer),
        ] {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let list = host.new_list().unwrap();
                // SAFETY: an open list, live objects.
                unsafe {
                    list.list.SetComputeRootSignature(&root);
                    list.list.SetPipelineState(&pso);
                    list.list
                        .SetComputeRootUnorderedAccessView(0, buffer.GetGPUVirtualAddress());
                    list.list.Dispatch((BYTES / 4 / 16 / 256) as u32, 1, 1);
                }
                list.close();
                let t = std::time::Instant::now();
                host.execute(&[&list]);
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "a shader reading and writing 256 MiB of the copy in {label}: {:.2} ms ({:.1} GB/s each way)",
                best * 1e3,
                BYTES as f64 / best / 1e9
            );
        }
        // Bringing written pages back: 64 scattered 64 KiB pages (4 MiB).
        let pages: Vec<_> = (0..64)
            .map(|k| (copy.clone(), p + k * 61 * PAGE % BYTES, PAGE))
            .collect();
        let mut best = f64::MAX;
        for _ in 0..10 {
            let t = std::time::Instant::now();
            pull(&pages);
            best = best.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "bringing 64 written 64 KiB pages back to the mirror: {:.3} ms",
            best * 1e3
        );
        let one = [(copy.clone(), p, PAGE)];
        let mut best = f64::MAX;
        for _ in 0..10 {
            let t = std::time::Instant::now();
            pull(&one);
            best = best.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "bringing 1 written page back (the fixed cost of a pull): {:.3} ms",
            best * 1e3
        );
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// The heap type a copy's buffer is in.
    fn heap_type(copy: &Shadow) -> D3D12_HEAP_TYPE {
        let mut props = D3D12_HEAP_PROPERTIES::default();
        // SAFETY: a live committed buffer.
        unsafe {
            copy.buffer
                .GetHeapProperties(Some(&mut props), None)
                .unwrap()
        };
        props.Type
    }

    /// With `VICTORY_GPU_UPLOAD=1`, the copy of a buffer made without
    /// unordered access is in a GPU upload heap: a refresh writes the
    /// title's bytes there, and the GPU reads them (copied out into system
    /// memory here). A buffer shaders may write keeps its copy in system
    /// memory.
    #[test]
    fn read_only_buffers_get_their_copies_in_video_memory() {
        if !rerun_with_gpu_upload(
            "shadow::tests::read_only_buffers_get_their_copies_in_video_memory",
            false,
        ) {
            return;
        }
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !host.gpu_upload() {
            return; // no GPU upload heaps (resizable BAR) here
        }
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register_read_only(p, p as u64, len as u64);
        let copy = find(p as u64, len as u64).expect("a copy");
        assert!(copy.upload.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(heap_type(&copy), D3D12_HEAP_TYPE_GPU_UPLOAD);
        // The title writes; the copy is refreshed; the GPU reads it.
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, len) };
        for (i, b) in title.iter_mut().enumerate() {
            *b = (i % 253) as u8;
        }
        copy.update(p, len);
        let (out, data) = host.shared_buffer(len as u64).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: an open list; both buffers are live and start common.
        unsafe {
            list.list
                .CopyBufferRegion(&out, 0, &copy.buffer, 0, len as u64)
        };
        list.close();
        host.execute(&[&list]);
        // SAFETY: the mapped output, `len` bytes, the copy done.
        let read = unsafe { std::slice::from_raw_parts(data, len) };
        assert!(
            read == &title[..],
            "what the GPU read is what the title wrote"
        );
        unregister(p, p as u64);
        // A buffer shaders may write.
        let q = crate::watch::tests::pool(1);
        register(q, q as u64, PAGE as u64);
        let writable = find(q as u64, PAGE as u64).expect("a copy");
        assert!(!writable.upload.load(std::sync::atomic::Ordering::Relaxed));
        assert_ne!(heap_type(&writable), D3D12_HEAP_TYPE_GPU_UPLOAD);
        unregister(q, q as u64);
    }

    /// A shader reading 256 MiB of a read-only buffer's copy (GPU upload
    /// heap) and of a writable one's (system memory), with
    /// `VICTORY_GPU_UPLOAD=1`.
    /// `cargo test --release -p victory-gpu --lib reading_copies -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark: run in release with --ignored --nocapture"]
    fn reading_copies_in_video_and_system_memory() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::{Interface, s};
        if !rerun_with_gpu_upload(
            "shadow::tests::reading_copies_in_video_and_system_memory",
            false,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload() {
            eprintln!("no GPU upload heaps here");
            return;
        }
        const SRC: &str = r#"
#define RS "SRV(t0), UAV(u0)"
ByteAddressBuffer b : register(t0);
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint s = 0;
    for (uint k = 0; k < 16; k++) s += b.Load((id.x * 16 + k) * 4);
    if (s == 0x12345678) o.Store(0, s);
}"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source outlives the call.
        unsafe {
            D3DCompile(
                SRC.as_ptr().cast(),
                SRC.len(),
                s!("b"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: the blob's bytes; plain creation.
        let (root, pso) = unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        const BYTES: usize = 256 << 20;
        let (out, _) = host.shared_buffer(256).unwrap();
        for read_only in [false, true] {
            let p = crate::watch::tests::pool(BYTES / PAGE);
            if read_only {
                register_read_only(p, p as u64, BYTES as u64);
            } else {
                register(p, p as u64, BYTES as u64);
            }
            let copy = find(p as u64, BYTES as u64).unwrap();
            // The first refresh copies all of it (every page is new).
            let t = std::time::Instant::now();
            copy.update(p, BYTES);
            let refresh = t.elapsed().as_secs_f64();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let list = host.new_list().unwrap();
                // SAFETY: an open list, live objects.
                unsafe {
                    list.list.SetComputeRootSignature(&root);
                    list.list.SetPipelineState(&pso);
                    list.list.SetComputeRootShaderResourceView(0, copy.gpu);
                    list.list
                        .SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
                    list.list.Dispatch((BYTES / 4 / 16 / 256) as u32, 1, 1);
                }
                list.close();
                let t = std::time::Instant::now();
                host.execute(&[&list]);
                best = best.min(t.elapsed().as_secs_f64());
            }
            let _ = copy.buffer.as_raw();
            eprintln!(
                "{}: shader reads 256 MiB in {:.2} ms ({:.1} GB/s); refresh writes it in {:.1} ms ({:.1} GB/s)",
                if read_only {
                    "read-only buffer's copy (GPU upload heap)"
                } else {
                    "writable buffer's copy (system memory)"
                },
                best * 1e3,
                BYTES as f64 / best / 1e9,
                refresh * 1e3,
                BYTES as f64 / refresh / 1e9
            );
            unregister(p, p as u64);
            crate::watch::tests::free(p);
        }
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders: the refresh before the
    /// submission runs still brings it to the copy, a shader's write still
    /// goes back, and a title write made after the refresh is not undone.
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(p, len);
        // An earlier list declares all of it as written by shaders.
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(p, len);
        // SAFETY: bytes of our pool.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
        }
        unregister(p, p as u64);
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        copy.writing(p + 16, 64, false);
        copy.writing(p + 3 * PAGE, 100, false);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(p + PAGE, 2 * PAGE, false);
        copy.writing(p + PAGE, PAGE, false);
        copy.writing(p + 80, 8, false);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "merged where they overlap or touch");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE), PAGE, "the piece written");
        assert_eq!(copy.copy_back(p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }

    /// Pages shaders marked (`copy_back_marks`, `victory_shader::emit`
    /// "Write marks") go back by the bytes the GPU changed. The title writes
    /// a marked page after the copy took it (the guard saves its bytes from
    /// before) and a page nobody marked: the shader's byte and the title's
    /// both stay in the first; the second keeps the title's, and the copy's
    /// byte there (no mark) is not put back. The bits are cleared.
    #[test]
    fn marked_pages_go_back_by_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap()) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        copy.update(p, len);
        copy.writing(p, len, true);
        assert!(copy.before.lock().unwrap().0.is_empty(), "no snapshot");
        // The shaders write a byte in each page; the title writes pages 1
        // and 2 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        copy.write_host((p + 2 * PAGE + 30) as u64, &[0xcc]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe {
            title(PAGE + 40).write_volatile(0x11);
            title(2 * PAGE + 50).write_volatile(0x22);
        }
        // The shaders marked pages 0 and 1.
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[]);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // SAFETY: dwords of the mapped bitmap.
        let left: u32 = (base / 32..=(base + 1) / 32).map(|w| unsafe { *bits.add(w) }).fold(0, |a, w| a | w);
        assert_eq!(left, 0, "bits cleared");
        assert_eq!(copy.copy_back(p, len), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    /// A marked page whose bytes the shaders left as they were (as for a
    /// third of the world's marked pages) goes back as nothing: title memory
    /// is not written, and the page stays taken (not changed, so the next
    /// refresh does not copy it again).
    #[test]
    fn a_marked_page_the_shaders_left_alone_stays_taken() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let (Some(copy), Some(b)) = (find(p as u64, PAGE as u64), bitmap()) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, PAGE);
        copy.writing(p, PAGE, true);
        // The shaders write the byte the copy already has.
        copy.write_host((p + 10) as u64, &[0]);
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *(b.data as *mut u32).add(base / 32) |= 1 << (base % 32) };
        copy_back_marks(&[]);
        let t = copy.tracked.as_ref().expect("tracked memory");
        assert!(!t.changed(p / PAGE), "still taken");
        assert_eq!(copy.copy_back(p, PAGE), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    /// Lazy copy-back (`VICTORY_LAZY_COPY_BACK=1`, in a child process: the
    /// switch is read once): marked pages stay pending, hidden, and title
    /// memory keeps its bytes; a page marked again stays pending and brings
    /// both submissions' bytes; the CPU's first read faults one page back,
    /// our own read of title memory brings the other.
    #[test]
    fn lazy_copy_back_waits_for_the_first_cpu_access() {
        const CHILD: &str = "VICTORY_TEST_LAZY_CHILD";
        let name = "shadow::tests::lazy_copy_back_waits_for_the_first_cpu_access";
        if std::env::var_os(CHILD).is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([name, "--exact", "--test-threads=1", "--nocapture"])
                .env(CHILD, "1")
                .env("VICTORY_LAZY_COPY_BACK", "1")
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
            assert!(out.status.success() && text.contains("1 passed"), "{name} with the switch on:\n{text}");
            return;
        }
        assert!(lazy::on());
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap()) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        let mark = |n: usize| {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        };
        let submission = |bytes: &[(usize, u8)]| {
            copy.update(p, len);
            copy.writing(p, len, true);
            for &(at, v) in bytes {
                copy.write_host((p + at) as u64, &[v]);
                mark(base + at / PAGE);
            }
            copy_back_marks(&[]);
            assert_eq!(copy.copy_back(p, len), 0);
        };
        submission(&[(10, 0xaa), (PAGE + 20, 0xbb)]);
        let mut probe = [0u8; 1];
        assert!(
            !crate::host::read_unflushed((p + 10) as *const u8, &mut probe),
            "pending pages are hidden"
        );
        // Marked again while pending: the second submission's byte too.
        submission(&[(11, 0xcc)]);
        // SAFETY: bytes of our pool (the fault brings the page back).
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(11)), (0xaa, 0xcc), "the first access brings both bytes");
        assert!(crate::host::read_into((p + PAGE + 20) as *const u8, &mut probe));
        assert_eq!(probe[0], 0xbb, "our own read brings the other page");
        let t = copy.tracked.as_ref().expect("tracked memory");
        assert!(!t.changed(p / PAGE) && !t.changed(p / PAGE + 1), "both taken again");
        assert_eq!(lazy::PENDING.load(std::sync::atomic::Ordering::Relaxed), 0);
        unregister(p, p as u64);
    }

    /// Lazy copy-back (as above, in a child process) of a page two copies
    /// hold, the inner one in part: both copies' bytes wait, hidden, and the
    /// first access brings both (neither puts its old bytes over the
    /// other's); a page only the outer copy wrote comes back when the inner
    /// copy refreshes its part, which then has the outer copy's byte.
    #[test]
    fn lazy_copy_back_of_a_page_two_copies_hold() {
        const CHILD: &str = "VICTORY_TEST_LAZY_CHILD";
        let name = "shadow::tests::lazy_copy_back_of_a_page_two_copies_hold";
        if std::env::var_os(CHILD).is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([name, "--exact", "--test-threads=1", "--nocapture"])
                .env(CHILD, "1")
                .env("VICTORY_LAZY_COPY_BACK", "1")
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
            assert!(out.status.success() && text.contains("1 passed"), "{name} with the switch on:\n{text}");
            return;
        }
        assert!(lazy::on());
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        let (inner_at, inner_len) = (p + 256, 4096);
        register(p, p as u64, len as u64);
        register(p + 1, inner_at as u64, inner_len as u64);
        let (Some(outer), Some(inner), Some(b)) = (
            find(p as u64, len as u64),
            find(inner_at as u64, inner_len as u64),
            bitmap(),
        ) else {
            unregister(p, p as u64);
            unregister(p + 1, inner_at as u64);
            return; // no host GPU here
        };
        assert!(!Arc::ptr_eq(&outer, &inner));
        let write = |copy: &Shadow, at: usize, v: u8| {
            copy.write_host((p + at) as u64, &[v]);
            let n = copy.marks.expect("tracked memory gets mark bits") as usize + (p + at) / PAGE
                - copy.start as usize / PAGE;
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        };
        let pending = || lazy::PENDING.load(std::sync::atomic::Ordering::Relaxed);
        let mut probe = [0u8; 1];
        // Both copies write page 0 in one submission.
        outer.update(p, len);
        inner.update(inner_at, inner_len);
        outer.writing(p, len, true);
        inner.writing(inner_at, inner_len, true);
        write(&outer, 300, 0xaa);
        write(&inner, 400, 0xbb);
        copy_back_marks(&[]);
        assert_eq!(outer.copy_back(p, len), 0);
        assert_eq!(inner.copy_back(inner_at, inner_len), 0);
        assert_eq!(pending(), 1, "one page, both copies");
        assert!(!crate::host::read_unflushed((p + 300) as *const u8, &mut probe), "hidden");
        // SAFETY: bytes of our pool (the fault brings the page back).
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!((read(300), read(400)), (0xaa, 0xbb), "both copies' bytes");
        assert_eq!(pending(), 0);
        // Only the outer copy writes page 0, in the inner copy's part.
        outer.update(p, len);
        inner.update(inner_at, inner_len);
        outer.writing(p, len, true);
        write(&outer, 500, 0xcc);
        copy_back_marks(&[]);
        assert_eq!(outer.copy_back(p, len), 0);
        assert_eq!(pending(), 1, "deferred though the inner copy holds it");
        inner.update(inner_at, inner_len);
        assert_eq!(pending(), 0, "the inner copy's refresh brought it back");
        assert_eq!(inner.read_host((p + 500) as u64, 1)[0] & 0xff, 0xcc, "and has the outer copy's byte");
        assert_eq!(read(500), 0xcc);
        unregister(p + 1, inner_at as u64);
        unregister(p, p as u64);
    }

    thread_local! {
        /// Title writes [`after_read`] makes, last first.
        static AFTER_READ: std::cell::RefCell<Vec<(usize, u8)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// A title thread's write landing just after a copy-back read the page
    /// (tests): the next of [`AFTER_READ`]'s.
    pub(super) fn after_read() {
        if let Some((at, value)) = AFTER_READ.with(|w| w.borrow_mut().pop()) {
            // SAFETY: a byte of the test's title memory (guarded or not: a
            // guard's fault opens it).
            unsafe { (at as *mut u8).write_volatile(value) };
        }
    }

    /// The title writes a page while it goes back, just after the copy-back
    /// read it. Only the bytes the shaders changed go into title memory, for
    /// a page marked through views and a page shaders wrote at computed
    /// addresses alike: the title's byte stays (a whole page put back undid
    /// it), and the copy has it too.
    #[test]
    fn a_copy_back_puts_back_only_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap()) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        // The shaders write a byte in each page.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        let title = |at: usize| (p + at) as *mut u8;
        AFTER_READ.with(|w| *w.borrow_mut() = vec![(p + PAGE + 40, 0x88), (p + 40, 0x77)]);
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        // SAFETY: (reads below) bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        // Page 0 was marked through a view.
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *(b.data as *mut u32).add(base / 32) |= 1 << (base % 32) };
        copy_back_marks(&[]);
        assert_eq!(
            (read(10), read(40)),
            (0xaa, 0x77),
            "the shader's byte and the title's, page 0"
        );
        assert_eq!(host(40), 0x77, "the copy reads the title's byte");
        // Page 1 was written at addresses shaders computed.
        assert_eq!(copy.copy_back_page(p + PAGE), PAGE);
        assert_eq!(
            (read(PAGE + 10), read(PAGE + 40)),
            (0xbb, 0x88),
            "both bytes, page 1"
        );
        assert_eq!(host(PAGE + 40), 0x88, "the copy reads the title's byte");
        unregister(p, p as u64);
    }

    /// The block-wise comparison finds the same runs as a byte-wise one:
    /// across block and line edges, at the ends, with the GPU's bytes told
    /// apart, and for lengths that are not whole blocks.
    #[test]
    fn differing_matches_a_byte_wise_comparison() {
        let naive = |host: &[u8], title: &[u8], gpu: &dyn Fn(usize) -> bool| {
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for i in 0..host.len() {
                if host[i] != title[i] && gpu(i) {
                    match runs.last_mut() {
                        Some((_, end)) if *end == i => *end = i + 1,
                        _ => runs.push((i, i + 1)),
                    }
                }
            }
            runs
        };
        for len in [0, 1, 31, 33, 1023, 1025, PAGE - 7, PAGE] {
            let title = vec![0x5au8; len];
            let mut host = title.clone();
            // Single bytes, runs over line and block edges, the last byte.
            let mut at = 3;
            while at < len {
                host[at] ^= 0xff;
                at += 997;
            }
            for i in (1020..1060).filter(|&i| i < len) {
                host[i] = 0;
            }
            if let Some(last) = host.last_mut() {
                *last = 1;
            }
            let gpu = |i: usize| !i.is_multiple_of(5);
            assert_eq!(
                differing(&host, &title, gpu),
                naive(&host, &title, &gpu),
                "length {len}"
            );
            assert_eq!(
                differing(&host, &title, |_| true),
                naive(&host, &title, &|_| true),
                "length {len}"
            );
        }
    }

    /// Mark bits are handed out in runs and come back joined.
    #[test]
    fn mark_runs_are_reused_and_joined() {
        let mut m = Marks::default();
        assert_eq!((m.take(3), m.take(2)), (Some(0), Some(3)));
        m.give_back(0, 3);
        assert_eq!(m.take(1), Some(0), "first fit");
        m.give_back(3, 2);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&1, &4)], "3-4 joins the free 1-2");
        m.give_back(0, 1);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&0, &5)], "and 0 joins them");
        assert_eq!(m.take(5), Some(0));
        assert_eq!(pages(0x1_fff0, 0x20), 2, "a range across a page boundary");
        assert_eq!(pages(0x2_0000, 0x1_0000), 1);
    }
}
