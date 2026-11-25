//! A DXIL vertex, pixel, compute or mesh shader under construction: the module,
//! the `dx.op` declarations it calls, its signatures and bound resources,
//! and the metadata and container that wrap it up (SM 6.6, validator 1.6).

use super::bitcode::{Attr, Md, Module, Ty, V, pred};
use super::container::{self, Binding, Psv, SigElement};

/// Where the shader reads its user data: `b0` in this space.
pub const USER_DATA_SPACE: u32 = 0;
/// Where immediate samplers are bound: `s<n>` in this space.
pub const SAMPLER_SPACE: u32 = 0;
/// Where root CBVs and SRVs are bound: `b<root parameter index>` and
/// `t<root parameter index>` in this space.
pub const ROOT_CBV_SPACE: u32 = 1;
/// The root SRV slot (`t255` in [`ROOT_CBV_SPACE`], past every title root
/// parameter) of the view-stride table: dword `i` is the byte stride of
/// the buffer view at view-heap index `i`, and in its top bits
/// ([`VIEW_SKIP_SHIFT`]) the bytes the host's view starts before it.
pub const VIEW_STRIDES: u8 = 255;
/// The root SRV slot of the host's view-block table, in [`ROOT_CBV_SPACE`]:
/// read only by shaders translated for a host that holds the title's view
/// slots by blocks ([`set_view_heap`]). Its dword `b` is the heap index of
/// the first of the title's slots `b << VIEW_BLOCK_SHIFT ..`, for every `b`
/// below [`HEAP_VIEWS`]` >> VIEW_BLOCK_SHIFT`.
pub const VIEW_BLOCKS: u8 = 254;
/// Title view slots come in blocks of `1 << VIEW_BLOCK_SHIFT` on such a
/// host: a block's slots follow each other in the host's heap, in the
/// title's order, wherever the block lies.
pub const VIEW_BLOCK_SHIFT: u32 = 10;
/// Where a view-stride table dword keeps the bytes (0, 4, 8 or 12) the
/// host's raw view of a structured buffer view starts before the view's
/// first byte; the stride is in the bits below. A raw view starts on 16
/// bytes (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`), a console view on any
/// dword: element 1 of a 12-, 24- or 456-byte stride is not on 16. The
/// host makes such a view raw from the 16 bytes below, in its buffer's own
/// copy, and translated programs add these bytes to every byte they
/// address through it (`emit`, "Views off 16 bytes"). From astrosteveo's
/// review of #1762: a copy of the view's own bytes instead missed writes
/// made through the buffer's view in the same list, and the reverse.
pub const VIEW_SKIP_SHIFT: u32 = 24;
/// Where the title-memory page table is bound: `t0` (a raw root SRV) in
/// this space (see `emit` "Title memory").
pub const MEMORY_SPACE: u32 = 2;
/// Heap indices are clamped below these (views, samplers): the host's heaps
/// are at least this large, the slots past their own null (`emit`, "Heap
/// indices"). Views: on a host that makes view heaps this large; else below
/// [`heap_views`].
pub const HEAP_VIEWS: u32 = 1 << 21;
pub const HEAP_SAMPLERS: u32 = 2048;

/// What translations assume of the host's view heaps: their slots, and
/// whether they hold the title's slots by blocks ([`set_view_heap`]).
static VIEW_HEAP: [std::sync::atomic::AtomicU32; 2] = [
    std::sync::atomic::AtomicU32::new(HEAP_VIEWS),
    std::sync::atomic::AtomicU32::new(0),
];

#[cfg(test)]
thread_local! {
    /// [`VIEW_HEAP`] for the translations of one test's thread, so that a
    /// test translating for another kind of host does not change what the
    /// tests beside it get.
    pub(crate) static TEST_VIEW_HEAP: std::cell::Cell<Option<(u32, bool)>> = const { std::cell::Cell::new(None) };
}

fn view_heap() -> (u32, bool) {
    #[cfg(test)]
    if let Some(heap) = TEST_VIEW_HEAP.get() {
        return heap;
    }
    let relaxed = std::sync::atomic::Ordering::Relaxed;
    (VIEW_HEAP[0].load(relaxed), VIEW_HEAP[1].load(relaxed) != 0)
}

/// The slots of the host's view heaps, which view heap indices are clamped
/// below: [`HEAP_VIEWS`] unless the host said otherwise ([`set_view_heap`]).
pub fn heap_views() -> u32 {
    view_heap().0
}

/// Whether the host's view heaps hold the title's slots by blocks: an
/// index a shader takes from the resource heap itself is the title's, and
/// goes through the view-block table ([`VIEW_BLOCKS`]) to the host's.
/// False unless the host said otherwise ([`set_view_heap`]).
pub fn view_blocks() -> bool {
    view_heap().1
}

/// For a host whose shader-visible view heaps are smaller than
/// [`HEAP_VIEWS`] (1,000,000 slots on NVIDIA; an index past a heap is
/// undefined there, a removed device in practice): every view heap has
/// `views` slots, the last three null as below. Such a heap has no room for
/// every slot of a title heap that large beside the host's own, so with
/// `blocks` it holds the title's slots by blocks, each wherever it fits
/// ([`VIEW_BLOCKS`]). Set before the first translation. The values are part
/// of the translation, so a shader cache must key on them (`victory-gpu`'s
/// does).
pub fn set_view_heap(views: u32, blocks: bool) {
    let relaxed = std::sync::atomic::Ordering::Relaxed;
    VIEW_HEAP[0].store(views.clamp(4, HEAP_VIEWS), relaxed);
    VIEW_HEAP[1].store(u32::from(blocks), relaxed);
}

/// The heap index of the host's null raw buffer view (the null raw
/// unordered access view at the index below), read instead of a slot that
/// holds no buffer view where a shader reads one at an index it computes
/// (`emit`, "Buffer views at computed heap indices"): two below the end of
/// the host's view heaps, whose last slot stays null.
pub fn null_buffer_view() -> u32 {
    heap_views() - 2
}
/// The view-mark table's register: `t2` (a raw root SRV) in
/// [`MEMORY_SPACE`] (`emit`, "Write marks").
pub const VIEW_MARKS: u32 = 2;
/// The written-pages bitmap of buffer views: `u1` (a raw root UAV) in
/// [`MEMORY_SPACE`].
pub const VIEW_WRITES: u32 = 1;
/// A view-mark table entry's base for a slot whose writes are not marked.
pub const VIEW_MARK_NONE: u32 = u32::MAX;
/// Bits in the written-pages bitmap of buffer views; marks past it go to
/// the last bit.
pub const VIEW_WRITE_BITS: u32 = 1 << 22;

/// `ShaderFlags` bits (metadata) and their `SFI0` feature bits.
pub(crate) mod flags {
    pub const RAW_BUFFERS: u64 = 1 << 4;
    pub const STENCIL_REF: u64 = 1 << 11;
    pub const TYPED_UAV_LOADS: u64 = 1 << 13;
    pub const ROVS: u64 = 1 << 18;
    pub const WAVE_OPS: u64 = 1 << 19;
    pub const INT64: u64 = 1 << 20;
    pub const ATOMIC64_TYPED: u64 = 1 << 27;
    pub const RESOURCE_HEAP: u64 = 1 << 30;
    pub const SAMPLER_HEAP: u64 = 1 << 31;
    pub const ATOMIC64_HEAP: u64 = 1 << 32;
}

pub(crate) struct Types {
    pub void: Ty,
    pub i1: Ty,
    pub i32: Ty,
    pub f32: Ty,
    pub handle: Ty,
    pub props: Ty,
    pub bind: Ty,
    pub cbuf_ret: Ty,
    pub res_ret: Ty,
}

/// What a mesh shader declares beyond its thread-group size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mesh {
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
}

/// What a geometry shader declares (`GSState`, `PSV0`): its one output
/// stream is stream 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Geometry {
    /// DXIL input primitive: 1 points, 2 lines, 3 triangles, 6 lines with
    /// adjacency, 7 triangles with adjacency.
    pub input: u32,
    /// Output topology: 1 points, 3 line strips, 5 triangle strips.
    pub topology: u32,
    /// The most vertices one run emits.
    pub max_vertices: u32,
    /// Runs per input primitive (`[instance(n)]`).
    pub instances: u32,
}

impl Geometry {
    /// Triangles in, triangle strips out, one run per primitive.
    pub fn triangles(max_vertices: u32) -> Self {
        Geometry {
            input: 3,
            topology: 5,
            max_vertices,
            instances: 1,
        }
    }

    /// The vertices of one input primitive.
    pub fn input_vertices(&self) -> u32 {
        match self.input {
            1 => 1,
            2 => 2,
            6 => 4,
            7 => 6,
            _ => 3,
        }
    }
}

pub(crate) struct Shader {
    pub m: Module,
    pub t: Types,
    pub pixel: bool,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// A mesh shader's per-primitive outputs.
    pub prims: Vec<SigElement>,
    /// A mesh shader (`threads` is its thread-group size).
    pub mesh: Option<Mesh>,
    /// A geometry shader: the generated one of rectangle lists (`rect`) or
    /// a title's (the console's merged geometry stage).
    pub gs: Option<Geometry>,
    /// Dwords of user data the shader reads (0: no user-data buffer).
    pub user_data: u32,
    /// Immediate samplers, bound as `s0..` in [`SAMPLER_SPACE`].
    pub samplers: u32,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    /// Immediate samplers uniform control flow chooses between: handles
    /// `a` and `b` and when `a` is taken. An operation on `a` runs on both,
    /// and the result is chosen.
    pub sampler_choices: Vec<(V, V, V)>,
    /// Root CBVs read, by root parameter index, with their handles.
    pub root_cbvs: Vec<(u8, V)>,
    /// Root SRVs (raw buffers) read, likewise.
    pub root_srvs: Vec<(u8, V)>,
    /// Root UAVs (raw buffers) read or written, likewise.
    pub root_uavs: Vec<(u8, V)>,
    /// The title-memory page table's handle, once read.
    pub memory: Option<V>,
    /// The written-pages bitmap's handle, once title memory is written.
    pub memory_dirty: Option<V>,
    /// The view addresses' handle, once a descriptor's address is read.
    pub view_addresses: Option<V>,
    /// The view-mark table's and the buffer views' written-pages bitmap's
    /// handles, once a write through a view is marked.
    pub view_marks: Option<V>,
    pub view_writes: Option<V>,
    /// The heap index of each handle of an unordered buffer view (writes
    /// through it are marked by the index's view-mark entry).
    pub view_index: std::collections::HashMap<V, V>,
    /// The heap index of each handle of a raw buffer view, shader resource
    /// or unordered access (its view-stride entry gives the bytes the
    /// host's view starts early, [`VIEW_SKIP_SHIFT`]): as asked for, and
    /// clamped below [`HEAP_VIEWS`].
    pub buffer_index: std::collections::HashMap<V, (V, V)>,
    pub flags: u64,
    /// Heap handles made with a non-uniform index.
    pub nonuniform_handles: u32,
    /// LDS accesses lowered with their bound (for tests).
    pub lds_bounds: u32,
    /// A compute shader with this thread-group size.
    pub threads: Option<[u32; 3]>,
    /// The wave size a compute shader requires (`WaveSize`).
    pub wave_size: Option<u32>,
    /// Thread-group shared memory, in dwords.
    pub lds_dwords: u32,
    lds: Option<V>,
    main: V,
    pub entry: usize,
    /// The `dx.op` operations called (names without overload).
    ops: std::collections::BTreeSet<String>,
}

thread_local! {
    /// See [`last_summary`].
    static LAST: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// What the module this thread finished last declares and calls: its
/// thread-group size and wave size, and its `dx.op` operations. A
/// diagnostic for tools that report a module the validator refuses, since
/// the validator's message for an operation the stage or the entry's
/// properties do not allow names neither.
pub(crate) fn last_summary() -> String {
    LAST.with(|l| l.borrow().clone())
}

impl Shader {
    pub fn new(pixel: bool) -> Self {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i8 = m.int(8);
        let i32 = m.int(32);
        let f32 = m.float();
        let i8p = m.ptr(i8);
        let handle = m.named_struct("dx.types.Handle", vec![i8p]);
        let props = m.named_struct("dx.types.ResourceProperties", vec![i32, i32]);
        let bind = m.named_struct("dx.types.ResBind", vec![i32, i32, i32, i8]);
        let cbuf_ret = m.named_struct("dx.types.CBufRet.i32", vec![i32, i32, i32, i32]);
        let res_ret = m.named_struct("dx.types.ResRet.f32", vec![f32, f32, f32, f32, i32]);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        let main = m.function("main", main_ty, None, true);
        let entry = m.new_block();
        Shader {
            m,
            t: Types {
                void,
                i1,
                i32,
                f32,
                handle,
                props,
                bind,
                cbuf_ret,
                res_ret,
            },
            pixel,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh: None,
            gs: None,
            user_data: 0,
            samplers: 0,
            cmp_samplers: 0,
            sampler_choices: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            memory: None,
            memory_dirty: None,
            view_addresses: None,
            view_marks: None,
            view_writes: None,
            view_index: std::collections::HashMap::new(),
            buffer_index: std::collections::HashMap::new(),
            flags: 0,
            nonuniform_handles: 0,
            lds_bounds: 0,
            threads: None,
            wave_size: None,
            lds_dwords: 0,
            lds: None,
            main,
            entry,
            ops: std::collections::BTreeSet::new(),
        }
    }

    /// The thread-group shared memory: `[lds_dwords x i32]` in address
    /// space 3.
    pub fn lds(&mut self) -> V {
        if let Some(g) = self.lds {
            return g;
        }
        let arr = self.m.array(self.lds_dwords, self.t.i32);
        let g = self.m.global("lds", arr, 3);
        self.lds = Some(g);
        g
    }

    /// Calls `dx.op.<name>` (class and overload) with `opcode` and `args`.
    pub fn op(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        let mut params = vec![self.t.i32];
        params.extend(args.iter().map(|&a| self.m.type_of(a)));
        let fty = self.m.func_ty(ret, params);
        self.m.ptr(fty);
        self.ops
            .insert(name.split('.').next().unwrap_or(name).to_owned());
        let f = self
            .m
            .function(&format!("dx.op.{name}"), fty, Some(attr), false);
        let mut all = vec![self.m.i32(opcode)];
        all.extend(args);
        let choice = self
            .sampler_choices
            .iter()
            .copied()
            .find(|&(a, _, _)| args.contains(&a));
        let r = self.m.call(block, f, all.clone());
        let Some((a, b, cond)) = choice else {
            return r;
        };
        for x in &mut all {
            if *x == a {
                *x = b;
            }
        }
        let other = self.m.call(block, f, all);
        if ret == self.t.res_ret {
            // Drivers miscompile a select of the result structure (AMD
            // read zeros): its fields are chosen where they are read.
            self.m.chosen.insert(r, (cond, other));
            return r;
        }
        self.m.select(block, cond, r, other)
    }

    /// The handle of the user-data constant buffer, made in the entry block.
    pub fn user_data_handle(&mut self) -> V {
        let bind = [
            self.m.i32(0),
            self.m.i32(0),
            self.m.i32(USER_DATA_SPACE),
            self.m.i8(2), // CBV
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let (i, f) = (self.m.i32(0), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        // The size is known in `finish`, which fixes this placeholder.
        self.annotate(block, h, 13, u32::MAX)
    }

    /// The handle of the root CBV of root parameter `slot`, made in the
    /// entry block.
    pub fn root_cbv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_cbvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(2)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 13, 65536);
        self.root_cbvs.push((slot, h));
        h
    }

    /// The handle of the root SRV (a raw buffer) of root parameter `slot`.
    pub fn root_srv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_srvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_srvs.push((slot, h));
        h
    }

    /// The handle of the root UAV (a raw buffer) of root parameter `slot`.
    pub fn root_uav_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_uavs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(1)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11 | 0x1000, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_uavs.push((slot, h));
        h
    }

    /// The handle of the title-memory page table (`t0` in [`MEMORY_SPACE`],
    /// a raw buffer), made in the entry block.
    pub fn memory_handle(&mut self) -> V {
        if let Some(h) = self.memory {
            return h;
        }
        let r = self.m.i32(0);
        let bind = [r, r, self.m.i32(MEMORY_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.memory = Some(h);
        h
    }

    /// The handle of the view addresses (`t1` in [`MEMORY_SPACE`], a raw
    /// buffer: per resource-heap index, a buffer view's title address as a
    /// buffer descriptor's dwords 0 and 1), made in the entry block.
    pub fn view_addresses_handle(&mut self) -> V {
        if let Some(h) = self.view_addresses {
            return h;
        }
        let r = self.m.i32(1);
        let bind = [r, r, self.m.i32(MEMORY_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.view_addresses = Some(h);
        h
    }

    /// The handle of the bitmap of title-memory pages written (`u0` in
    /// [`MEMORY_SPACE`], a raw buffer), made in the entry block.
    pub fn memory_dirty_handle(&mut self) -> V {
        if let Some(h) = self.memory_dirty {
            return h;
        }
        let r = self.m.i32(0);
        let bind = [r, r, self.m.i32(MEMORY_SPACE), self.m.i8(1)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11 | 0x1000, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.memory_dirty = Some(h);
        h
    }

    /// The handle of the view-mark table (`t2` in [`MEMORY_SPACE`], a raw
    /// buffer: per resource-heap index, two dwords), made in the entry block.
    pub fn view_marks_handle(&mut self) -> V {
        if let Some(h) = self.view_marks {
            return h;
        }
        let h = self.memory_space_handle(VIEW_MARKS, false);
        self.view_marks = Some(h);
        h
    }

    /// The handle of the buffer views' written-pages bitmap (`u1` in
    /// [`MEMORY_SPACE`], a raw buffer), made in the entry block.
    pub fn view_writes_handle(&mut self) -> V {
        if let Some(h) = self.view_writes {
            return h;
        }
        let h = self.memory_space_handle(VIEW_WRITES, true);
        self.view_writes = Some(h);
        h
    }

    /// A raw buffer bound at `register` in [`MEMORY_SPACE`] (`uav`: `u`,
    /// else `t`), made in the entry block.
    fn memory_space_handle(&mut self, register: u32, uav: bool) -> V {
        let r = self.m.i32(register);
        let bind = [r, r, self.m.i32(MEMORY_SPACE), self.m.i8(u8::from(uav))];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, if uav { 11 | 0x1000 } else { 11 }, 0);
        self.flags |= flags::RAW_BUFFERS;
        h
    }

    /// The handle of immediate sampler `n` (a comparison sampler when its
    /// bit in `cmp_samplers` is set).
    pub fn sampler_handle(&mut self, block: usize, n: u32) -> V {
        let bind = [
            self.m.i32(n),
            self.m.i32(n),
            self.m.i32(SAMPLER_SPACE),
            self.m.i8(3), // sampler
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let (i, f) = (self.m.i32(n), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        let cmp = if self.cmp_samplers >> n & 1 != 0 {
            0x8000
        } else {
            0
        };
        self.annotate(block, h, 14 | cmp, 0)
    }

    pub fn annotate(&mut self, block: usize, h: V, props0: u32, props1: u32) -> V {
        let p = [self.m.i32(props0), self.m.i32(props1)];
        let p = self.m.aggregate(self.t.props, &p);
        self.op(
            block,
            "annotateHandle",
            Attr::ReadNone,
            self.t.handle,
            216,
            &[h, p],
        )
    }

    /// A descriptor heap handle, annotated with `props`; `nonuniform` when
    /// the index may differ between lanes (`NonUniformResourceIndex`).
    pub fn heap_handle(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
    ) -> V {
        self.flags |= if sampler {
            flags::SAMPLER_HEAP
        } else {
            flags::RESOURCE_HEAP
        };
        self.nonuniform_handles += u32::from(nonuniform);
        let asked = index;
        // An index past the host's heap faults the device (a console shader
        // reads whatever memory follows its heap): the last slot instead.
        let last = self.m.i32(if sampler { HEAP_SAMPLERS } else { heap_views() } - 1);
        let over = self.m.cmp(block, pred::IUGT, index, last);
        let index = self.m.select(block, over, last, index);
        let (s, n) = (self.m.bool(sampler), self.m.bool(nonuniform));
        let h = self.op(
            block,
            "createHandleFromHeap",
            Attr::ReadNone,
            self.t.handle,
            218,
            &[index, s, n],
        );
        let h = self.annotate(block, h, props.0, props.1);
        // Unordered typed, raw and structured buffers (kinds 10-12).
        if !sampler && props.0 & 0x1000 != 0 && (10..=12).contains(&(props.0 & 0xFF)) {
            self.view_index.insert(h, index);
        }
        if !sampler && props.0 & 0xFF == 11 {
            self.buffer_index.insert(h, (asked, index));
        }
        h
    }

    /// The finished container.
    pub fn finish(mut self) -> Vec<u8> {
        let threads = self
            .threads
            .map_or(String::new(), |[x, y, z]| format!("threads {x}x{y}x{z}, "));
        let wave = self.wave_size.map_or("any".to_owned(), |n| n.to_string());
        let ops: Vec<&str> = self.ops.iter().map(String::as_str).collect();
        let summary = format!("{threads}wave size {wave}; ops: {}", ops.join(", "));
        LAST.with(|l| *l.borrow_mut() = summary);
        let ud_size = (self.user_data * 4).next_multiple_of(16);
        self.fix_user_data_size(ud_size);
        let mut resources = None;
        let mut bindings = Vec::new();
        let i32c = |s: &mut Self, v: u32| {
            let c = s.m.i32(v);
            s.m.md_value(c)
        };
        if self.user_data > 0
            || self.samplers > 0
            || !self.root_cbvs.is_empty()
            || !self.root_srvs.is_empty()
            || !self.root_uavs.is_empty()
            || self.memory.is_some()
            || self.memory_dirty.is_some()
            || self.view_addresses.is_some()
            || self.view_marks.is_some()
            || self.view_writes.is_some()
        {
            let empty = self.m.md_str("");
            let mut cbvs = None;
            let mut cbv_list = Vec::new();
            if self.user_data > 0 {
                let arr = self.m.array(self.user_data.next_multiple_of(4), self.t.i32);
                let ty = self.m.named_struct("UserData", vec![arr]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let ops = vec![
                    Some(i32c(&mut self, 0)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, USER_DATA_SPACE)),
                    Some(i32c(&mut self, 0)),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, ud_size)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: USER_DATA_SPACE,
                    register: 0,
                    dxil_kind: 13,
                });
            }
            let blob = self.m.array(16384, self.t.i32);
            let ty = self.m.named_struct("RootCbv", vec![blob]);
            let p = self.m.ptr(ty);
            let u = self.m.undef(p);
            for (slot, _) in self.root_cbvs.clone() {
                let ops = vec![
                    Some(i32c(&mut self, cbv_list.len() as u32)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_CBV_SPACE)),
                    Some(i32c(&mut self, u32::from(slot))),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 65536)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: ROOT_CBV_SPACE,
                    register: u32::from(slot),
                    dxil_kind: 13,
                });
            }
            if !cbv_list.is_empty() {
                cbvs = Some(self.m.md_node(cbv_list));
            }
            let mut srvs = None;
            if !self.root_srvs.is_empty()
                || self.memory.is_some()
                || self.view_addresses.is_some()
                || self.view_marks.is_some()
            {
                let ty = self
                    .m
                    .named_struct("struct.ByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                let all: Vec<(u32, u32)> = self
                    .root_srvs
                    .iter()
                    .map(|&(slot, _)| (ROOT_CBV_SPACE, u32::from(slot)))
                    .chain(self.memory.map(|_| (MEMORY_SPACE, 0)))
                    .chain(self.view_addresses.map(|_| (MEMORY_SPACE, 1)))
                    .chain(self.view_marks.map(|_| (MEMORY_SPACE, VIEW_MARKS)))
                    .collect();
                for (n, (space, slot)) in all.into_iter().enumerate() {
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, space)),
                        Some(i32c(&mut self, slot)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(i32c(&mut self, 0)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 4,
                        space,
                        register: slot,
                        dxil_kind: 11,
                    });
                }
                srvs = Some(self.m.md_node(list));
            }
            let mut uavs = None;
            if !self.root_uavs.is_empty() || self.memory_dirty.is_some() || self.view_writes.is_some() {
                let ty = self
                    .m
                    .named_struct("struct.RWByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let no = self.m.bool(false);
                let no = self.m.md_value(no);
                let mut list = Vec::new();
                let all: Vec<(u32, u32)> = self
                    .root_uavs
                    .iter()
                    .map(|&(slot, _)| (ROOT_CBV_SPACE, u32::from(slot)))
                    .chain(self.memory_dirty.map(|_| (MEMORY_SPACE, 0)))
                    .chain(self.view_writes.map(|_| (MEMORY_SPACE, VIEW_WRITES)))
                    .collect();
                for (n, (space, slot)) in all.into_iter().enumerate() {
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, space)),
                        Some(i32c(&mut self, slot)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(no),
                        Some(no),
                        Some(no),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 7,
                        space,
                        register: slot,
                        dxil_kind: 11,
                    });
                }
                uavs = Some(self.m.md_node(list));
            }
            let mut samplers = None;
            if self.samplers > 0 {
                let ty = self.m.named_struct("struct.SamplerState", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for n in 0..self.samplers {
                    let kind = (self.cmp_samplers >> n & 1) as u32;
                    let ops = vec![
                        Some(i32c(&mut self, n)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, SAMPLER_SPACE)),
                        Some(i32c(&mut self, n)),
                        Some(i32c(&mut self, 1)),
                        // Sampler kind: 1 comparison.
                        Some(i32c(&mut self, kind)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 1,
                        space: SAMPLER_SPACE,
                        register: n,
                        dxil_kind: 14,
                    });
                }
                samplers = Some(self.m.md_node(list));
            }
            resources = Some(self.m.md_node(vec![srvs, uavs, cbvs, samplers]));
            // PSV0 lists them as the module does: CBVs, samplers, SRVs.
            bindings.sort_by_key(|b| match b.kind {
                2 => 0,
                1 => 1,
                _ => 2,
            });
        }
        if let Some(r) = resources {
            self.m.named_md("dx.resources", vec![r]);
        }

        let graphics = self.threads.is_none() || self.mesh.is_some();
        if graphics {
            // shortcut: every output is declared to depend on every input; the
            // exact sets need a dependence pass. Drivers read this state.
            let scalars = |els: &[SigElement]| {
                els.iter()
                    .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
                    .max()
                    .unwrap_or(0)
            };
            let (ni, no) = (scalars(&self.inputs), scalars(&self.outputs));
            let mut view = vec![ni, no];
            // Mesh shaders have no inputs: then the primitive outputs' count.
            if self.mesh.is_some() {
                view.push(scalars(&self.prims));
            }
            for _ in 0..ni {
                for k in 0..no.div_ceil(32) {
                    let bits = (no - 32 * k).min(32);
                    view.push(if bits == 32 {
                        u32::MAX
                    } else {
                        (1 << bits) - 1
                    });
                }
            }
            // A geometry shader has four streams, each its output count and
            // table; the other three empty.
            if self.gs.is_some() {
                view.extend([0, 0, 0]);
            }
            let arr = self.m.array(view.len() as u32, self.t.i32);
            let elems: Vec<V> = view.iter().map(|&v| self.m.i32(v)).collect();
            let arr = self.m.aggregate(arr, &elems);
            let arr = self.m.md_value(arr);
            let view_node = self.m.md_node(vec![Some(arr)]);
            self.m.named_md("dx.viewIdState", vec![view_node]);
        }

        if self.pixel {
            container::same_mode_per_row(&mut self.inputs, true);
        } else {
            container::same_mode_per_row(&mut self.outputs, false);
        }
        let sigs = if graphics {
            let ins = self.signature_md(self.inputs.clone());
            let outs = self.signature_md(self.outputs.clone());
            let prims = self.signature_md(self.prims.clone());
            Some(self.m.md_node(vec![ins, outs, prims]))
        } else {
            None
        };
        // A pixel shader with an SV_StencilRef output (as DXC collects it).
        if self.pixel && self.outputs.iter().any(|o| o.kind == 20) {
            self.flags |= flags::STENCIL_REF;
        }
        // Entry properties: shader flags (tag 0), thread-group size (4),
        // wave size (11).
        let mut props = Vec::new();
        if self.flags != 0 {
            props.push(i32c(&mut self, 0));
            let i64t = self.m.int(64);
            let f = self.m.int_const(i64t, self.flags as i64);
            props.push(self.m.md_value(f));
        }
        if let Some(threads) = self.threads {
            let t = threads.map(|n| Some(i32c(&mut self, n)));
            let t = self.m.md_node(t.to_vec());
            match self.mesh {
                // Mesh state (tag 9): threads, maximum vertices and
                // primitives, topology, payload bytes.
                Some(mesh) => {
                    props.push(i32c(&mut self, 9));
                    let state = [
                        mesh.max_vertices,
                        mesh.max_primitives,
                        mesh.topology.into(),
                        0,
                    ]
                    .map(|n| Some(i32c(&mut self, n)));
                    let mut node = vec![Some(t)];
                    node.extend(state);
                    props.push(self.m.md_node(node));
                }
                None => {
                    props.push(i32c(&mut self, 4));
                    props.push(t);
                }
            }
        }
        // Geometry state (tag 1): input primitive, maximum vertices, stream
        // mask (stream 0), output topology, instances.
        if let Some(gs) = self.gs {
            props.push(i32c(&mut self, 1));
            let state = [gs.input, gs.max_vertices, 1, gs.topology, gs.instances]
                .map(|n| Some(i32c(&mut self, n)));
            props.push(self.m.md_node(state.to_vec()));
        }
        if let Some(n) = self.wave_size {
            props.push(i32c(&mut self, 11));
            let n = Some(i32c(&mut self, n));
            props.push(self.m.md_node(vec![n]));
        }
        let props =
            (!props.is_empty()).then(|| self.m.md_node(props.into_iter().map(Some).collect()));
        let main = self.m.md_value(self.main);
        let name = self.m.md_str("main");
        let entry = self
            .m
            .md_node(vec![Some(main), Some(name), sigs, resources, props]);

        let ident = self.m.md_str("victory");
        let ident = self.m.md_node(vec![Some(ident)]);
        self.m.named_md("llvm.ident", vec![ident]);
        let one = i32c(&mut self, 1);
        let six = i32c(&mut self, 6);
        let version = self.m.md_node(vec![Some(one), Some(six)]);
        self.m.named_md("dx.version", vec![version]);
        self.m.named_md("dx.valver", vec![version]);
        // Program kinds: 0 pixel, 1 vertex, 2 geometry, 5 compute, 13 mesh.
        let (name, kind) = match (self.threads, self.pixel) {
            _ if self.mesh.is_some() => ("ms", 13u32),
            _ if self.gs.is_some() => ("gs", 2),
            (Some(_), _) => ("cs", 5u32),
            (None, true) => ("ps", 0),
            (None, false) => ("vs", 1),
        };
        let name = self.m.md_str(name);
        let sm = self.m.md_node(vec![Some(name), Some(six), Some(six)]);
        self.m.named_md("dx.shaderModel", vec![sm]);
        self.m.named_md("dx.entryPoints", vec![entry]);

        let bitcode = self.m.write();
        let mut dxil = Vec::with_capacity(24 + bitcode.len());
        dxil.extend((kind << 16 | 0x66).to_le_bytes());
        dxil.extend((((24 + bitcode.len()) / 4) as u32).to_le_bytes());
        dxil.extend(b"DXIL");
        dxil.extend(0x106u32.to_le_bytes());
        dxil.extend(16u32.to_le_bytes());
        dxil.extend((bitcode.len() as u32).to_le_bytes());
        dxil.extend(&bitcode);

        let mut features = 0u64;
        if self.flags & flags::STENCIL_REF != 0 {
            features |= 0x200;
        }
        if self.flags & flags::TYPED_UAV_LOADS != 0 {
            features |= 0x800;
        }
        if self.flags & flags::ROVS != 0 {
            features |= 0x1000;
        }
        if self.flags & flags::WAVE_OPS != 0 {
            features |= 0x4000;
        }
        if self.flags & flags::INT64 != 0 {
            features |= 0x8000;
        }
        if self.flags & flags::RESOURCE_HEAP != 0 {
            features |= 0x200_0000;
        }
        if self.flags & flags::SAMPLER_HEAP != 0 {
            features |= 0x400_0000;
        }
        if self.flags & flags::ATOMIC64_TYPED != 0 {
            features |= 0x40_0000;
        }
        if self.flags & flags::ATOMIC64_HEAP != 0 {
            features |= 0x1000_0000;
        }
        let sfi0 = features.to_le_bytes();
        let isg1 = container::signature(&self.inputs, true);
        let osg1 = container::signature(&self.outputs, false);
        let flag = if self.pixel {
            self.outputs.iter().any(|o| (17..=19).contains(&o.kind))
        } else {
            self.outputs.iter().any(|o| o.kind == 3)
        };
        let psv0 = container::psv(&Psv {
            stage: kind as u8,
            flag: flag && self.threads.is_none(),
            inputs: &self.inputs,
            outputs: &self.outputs,
            prims: &self.prims,
            mesh: self.mesh.map(|m| (m, self.lds_dwords * 4)),
            gs: self.gs,
            bindings: &bindings,
            threads: self.threads.unwrap_or_default(),
            wave_size: self.wave_size,
        });
        let hash = [0u8; 20];
        let psg1 = container::signature(&self.prims, false);
        let mut parts: Vec<(&[u8; 4], &[u8])> =
            vec![(b"SFI0", &sfi0), (b"ISG1", &isg1), (b"OSG1", &osg1)];
        if self.mesh.is_some() {
            parts.push((b"PSG1", &psg1));
        }
        parts.extend([(b"PSV0", &psv0[..]), (b"HASH", &hash), (b"DXIL", &dxil)]);
        container::build(&parts)
    }

    /// The user-data buffer's annotation carries its size, known only now.
    fn fix_user_data_size(&mut self, size: u32) {
        let old = [self.m.i32(13), self.m.i32(u32::MAX)];
        let old = self.m.aggregate(self.t.props, &old);
        let new = [self.m.i32(13), self.m.i32(size)];
        let new = self.m.aggregate(self.t.props, &new);
        self.m.replace_const(old, new);
    }

    fn signature_md(&mut self, elements: Vec<SigElement>) -> Option<Md> {
        if elements.is_empty() {
            return None;
        }
        let mut list = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let c32 = |s: &mut Self, v: u32| {
                let c = s.m.i32(v);
                Some(s.m.md_value(c))
            };
            let c8 = |s: &mut Self, v: u8| {
                let c = s.m.i8(v);
                Some(s.m.md_value(c))
            };
            let idx = c32(self, e.index);
            let indices = Some(self.m.md_node(vec![idx]));
            let usage = if e.usage != 0 {
                let (tag, mask) = (c32(self, 3), c32(self, u32::from(e.usage)));
                Some(self.m.md_node(vec![tag, mask]))
            } else {
                None
            };
            let ops = vec![
                c32(self, id as u32),
                Some(self.m.md_str(&e.name)),
                c8(self, e.comp),
                c8(self, e.kind),
                indices,
                c8(self, e.interp),
                c32(self, 1),
                c8(self, e.cols),
                c32(self, e.row.unwrap_or(u32::MAX)),
                c8(self, if e.row.is_some() { e.col } else { u8::MAX }),
                usage,
            ];
            list.push(Some(self.m.md_node(ops)));
        }
        Some(self.m.md_node(list))
    }
}
