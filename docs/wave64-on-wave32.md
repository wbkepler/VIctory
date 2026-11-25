# Console wave64 compute on 32-lane hosts

Status: steps 1, 2, 4 and 5 are in. On NVIDIA, wave64 programs whose lanes
read only their own half run as two 32-lane waves. Groups of one console
wave also run in the console flow: masks read whole, lanes of the other
half, divergent branches and loops, LDS shared in the console's
instruction order, and (in groups of exactly 64 threads) whole-wave
reductions and prefix sums, of integers and floats. Still open: groups of
several console waves, whole-wave sections in other groups, and
`ds_permute`; each stops with its own reason. (`ds_bpermute` gathers within each 32-lane half
and is lowered as an in-half read, like `ds_swizzle`.)

## The problem

The console runs most of the title's compute programs in waves of 64
lanes. Per-lane code does not care how many lanes a host wave has, so most
of them translate without a wave size. A program that depends on the lane
layout declares the console's (`WaveSize(64)`, see `emit::lower`, where
`wave_size` is set):

- **console lanes**: it reads other lanes (`v_readlane`, `v_writelane`,
  DPP, `ds_swizzle`, `ds_(b)permute`, `v_permlane*`), counts with
  `v_mbcnt`, reads a lane mask as a number, or tests its own bit of a mask
  held as a number;
- **one wave sharing LDS**: the group is a single console wave and uses
  LDS, where the console compiler leaves out barriers because one wave runs
  in order.

D3D12 reports NVIDIA's wave range as 32..32, so these programs cannot run
there as they are. Since step 1 they stop at translation with the reason
(`emit::Target`) instead of being refused at pipeline creation.
`victory translate --host-lanes 32-32` counts them.

## The form

One console wave of 64 runs as two host waves of 32 lanes ("halves") of the
same thread group. The shader declares `WaveSize(32)`; console lane
= console thread % 64, half = bit 5 of it.

**Lane assignment.** The relabelling in `Lower::thread_id` assumes each
host wave holds a wave-aligned run of flat thread ids: it takes the first
lane's flat id, rounded down to the declared wave size, plus the host lane.
XCL's notes report NVIDIA forming waves from non-consecutive thread indices
in some 3-D groups. On an RTX 5070 with this relabelling, every group
`emit::split_tests` tries is exact: 32, 64 and 128 threads in 1-D, and
4x4x4, 8x8x2, 2x4x8, 16x2x4, 8x8x4 and 4x8x8. If another driver breaks it,
the fallback is to take console threads from a groupshared counter: one lane
of each host wave adds 32 to it when the wave starts and shares the result,
and the wave's lanes take the next 32 console threads in lane order.

**What stays inside a half** needs nothing: DPP16 and DPP8 (rows of 16,
groups of 8), `ds_swizzle` (groups of 32 or 4), and `v_permlane16` /
`v_permlanex16` (the two rows of a 32-lane half; not lowered at all yet).
Their lowering computes the source lane from the host lane's place in its
row or group, so it is unchanged in the split form.

Step 4 (`split` in `Lower::new`) runs a program as two 32-lane waves when
the host has no 64-lane waves and the program:
- reads other lanes only through DPP and `ds_swizzle`, and has no
  `v_readfirstlane` (the console wave's first lane is not in the upper
  half; such programs take the console flow);
- reads no mask whole and tests no bit of a mask held as a number;
- has no whole-wave sections (their reductions would cover a half);
- uses no LDS (the console compiler may have left out barriers between
  lanes of one wave).

Anything else still stops with the step 1 reason. `mask_dword`, the wave
reduction lowering and `lane_bit` refuse the split form as a safety net.
(A wave64 program that declares no wave size at all still runs in whatever
waves the host forms, as on `main`: its `v_readfirstlane` reads the host
wave's first lane.)

**What crosses halves** goes through an exchange in groupshared memory:

| operation | exchanged |
|---|---|
| a lane mask read whole (ballots: scalar compares, `s_bcnt1`, `s_ff1`, mask tests) | each half's 32-bit ballot |
| `v_mbcnt_lo` in the upper half | the lower half's ballot |
| `v_readlane`, `v_readfirstlane` (uniform lane) | the value of lane `l & 31` from the half that owns `l` |
| `v_writelane` to the other half | the value and lane |
| `ds_bpermute` | nothing: on GFX10 a wave64's backward permute gathers within each 32-lane half (AMD's RDNA 2 ISA reference, as cited by the lowering's contributor; not checked on console hardware), so it is an in-half read |
| `ds_permute` | open: check whether GFX10 wave64 crosses halves for it, then 32 words per half if it does |

A mask read only lane by lane (a `v_cndmask` selector, a carry-in) needs
just the lane's own half. Exchanging those anyway is correct but slow.

## Synchronisation: the console flow

Both halves must reach every exchange together. On the console they do,
because a wave runs every instruction whatever EXEC holds. Our per-lane
structurisation turns EXEC into per-lane branches, so a half with no lane
on a path would skip that path and never reach an exchange there.

An earlier version of this plan made only the regions with an exchange
uniform. It turned out cleaner to run the whole program as the console
does (step 5, `lower::CONSOLE_FLOW`):

- `structure::structure_console` builds the per-lane booleans without
  resolving mask joins from what arriving lanes know, since every lane
  arrives. Each join that differs between edges becomes a phi.
  `Structured::execs` gives the lane's EXEC bit at each vector
  instruction.
- The lowering guards every vector instruction with that EXEC bit.
  `v_readlane`, `v_writelane` and `v_readfirstlane` ignore EXEC, as on the
  console.
- Branches are the console's `s_cbranch_*`. EXEC and VCC tests, and the
  SCC of a mask operation, ask whether some lane of the console wave has
  the bit: each half's ballot, exchanged. Both halves then take every
  branch the same way, and loops run while any lane of the console wave
  does.
- Exchange (`Lower::exchange`): each half stores its dword to group-shared
  slot `half`, then a group barrier, both read both slots, and a second
  barrier. Every thread of the group reaches it because the group is one
  console wave whose halves take the same branches. A group of 32 threads
  or fewer has no upper half: its word is 0 and no barrier is needed.
- Masks read whole (`mask_dword`) are the exchanged ballots. Lane bits of
  masks held as numbers use the console lane. `v_readlane` and
  `v_readfirstlane` read the owning half's value. `v_writelane` and
  `v_mbcnt` count with the console lane.

`lower` tries the per-lane form first. A program that needs the console
flow stops with `CONSOLE_FLOW`, and `emit` lowers it again from the
console structure. Eligible: wave64 compute on a host without 64-lane
waves, at most 64 threads per group, no `ds_permute`, and whole-wave
sections only in a group of exactly 64 threads.

**LDS.** The console runs an instruction in all 64 lanes before the next,
so one wave shares LDS without `s_barrier`: a lane reads what a lane of the
other half wrote one instruction earlier. In the console flow both halves
reach every LDS access together, so a group barrier before each access puts
it after every earlier one. This also takes the programs that needed 64-lane
waves only for that (one console wave per group, LDS, no cross-lane read).
Two testers' counts on the title put those first: 120 of the 223 programs
#233 still stopped on a 32-lane host. On an RTX 5070 most LDS tests pass
without the barrier too (that card keeps the two waves in step, and a
branch on EXEC is an exchange, which is a barrier itself). One does not:
with a write under one EXEC mask and a read under another and no branch
between them, the lower half reads before the upper half writes every
time unless the barrier is there
(`lds_written_and_read_in_different_masked_blocks`).

**Whole-wave sections** (contributed by Raxstreams). A section stays
recognised as it is (`emit::wave`), and only what it leaves crosses the
halves, where the section is lowered: each half's reduction is exchanged
and the two are combined with the section's own operation (sum, and, or,
xor, minimum, maximum of integers); a prefix sum in the upper half adds the
lower half's total. Only in a group of exactly 64 threads, so both halves
are whole. Floats cross as their bits: minimum and maximum stay exact, and
a sum is each half's sum added last, one more order of additions next to
the console's and the host wave's own (the last bits may differ, as they
already may in one host wave). A title run on #439 put sections next: 24
of the 45 programs the stack still stopped.

What still stops says why: `console wave64 lanes in a group of more than 64
threads`, `ds_permute across the halves`, or `a whole-wave section` (in
other groups), each `on a host without 64-lane waves`.

### Groups of several console waves (planned)

A group barrier is reached by every thread of the group only where all its
console waves branch alike, and they need not: each console wave takes its
`s_cbranch_*` for itself. The earlier plan was a meeting point per console
wave (arrival counters, double-buffered slots, a bounded wait). It is
dropped: a wait is met only if the partner half runs meanwhile, and D3D12
promises no such thing. On an RTX 5070 the upper half of a group does not
write LDS before the lower half reads it unless a barrier orders them, in
20 of 20 runs (`lds_written_and_read_in_different_masked_blocks`), which is
what waves run one after the other would give. A wait would then never be
met, and a bounded one gives a wrong result instead of a hang.

What the programs are (an offline translate over a local copy of the
title's shaders, 43 programs stopped for their group size): groups of
16x16x1 (15), 1024x1x1 (7), 8x16x1 (6), 256x1x1 (4), 512x1x1 (2), 128x1x1
(2) and 32x32x1 (1). Nearly all count mask bits (`v_mbcnt`) and read a mask
as a number; about two thirds have no LDS and no `s_barrier`.

**Without LDS: one console wave per host group.** The console waves of
such a group share nothing but the resources every group shares, so they
can be separate host groups. The program is emitted with 64 threads per
group, and a dispatch of (x, y, z) groups of k console waves each becomes
(x * k, y, z) host groups: host group id x gives the console group
(x / k) and the wave in it (x % k), the flat thread id is 64 * wave +
the id in the host group, and the console's thread and group ids follow
from those. Each host group is then one console wave in two host waves,
which is the console flow as it is. Needed outside the translator: the
emitted shader says its k, direct dispatches multiply x (within D3D12's
65535 groups per dimension; refuse beyond), and indirect dispatches
multiply the record's x where the indirect records are already rewritten
(`victory-gpu`'s clamp pass).

**With LDS and `s_barrier`:** the waves of a group do meet there, so they
must stay one host group. Open; candidates are group barriers where every
console wave provably arrives (as #189 places them) and, for the rest, a
group-wide form in which every thread runs every block any console wave of
its group enters, with each wave's scalars and effects guarded by whether
that wave is really there.

### The console flow on every host

The per-lane form is wrong for a program that reads or writes a named
lane (`v_readlane`, `v_writelane`) where its lanes branch differently
(#225). Those instructions ignore EXEC on the console and a wave runs a loop
while any lane does. Per lane, a lane that left a loop or skipped a branch
is not there to be read or written, and never sees scalars set inside.
Measured before the change: a loop lane t runs t times, reading lane L in
each iteration, had 12 of 32 lanes wrong (wave32, RTX 5070 and Radeon) and
24 of 64 (wave64, Radeon); a `v_writelane` to lane 40 inside a branch only
lanes below 20 take never wrote lane 40.

Such compute programs now ask for the console flow on every host (the
second case of `CONSOLE_FLOW`): a program with `v_readlane` or
`v_writelane`, at least one branch that is not uniform, no `ds_permute`
and no whole-wave sections. On a host that runs the console's wave size,
one host wave is the console wave, so ballots are the wave's own and nothing
is exchanged; any group size and LDS are fine. If the console flow cannot
lower the program, it keeps the per-lane form (`Parts::per_lane_only`), so
nothing that translated before is refused.

## Testing

The differential tests need the console SDK's compiler, which we do not
assume. These tests are hand-assembled GFX10 programs instead (as in
`emit::tests`), run twice:

- in the native form on an adapter with 64-lane waves (AMD RDNA, such as
  the integrated GPU of Ryzen 7000 desktop CPUs), as the reference;
- in the split form on a 32-lane adapter (NVIDIA),

then compared buffer for buffer. `emit::host::Host::on(name)` and
`VICTORY_ADAPTER=<part of the adapter's name>` pick the adapter;
`emit::tests::each_adapter_gets_what_it_runs` checks every adapter in the
machine (on a Ryzen 7800X3D with an RTX 5070: the Radeon reports 32-64 and
takes `WaveSize(64)`, the NVIDIA reports 32-32).

## Steps

1. ~~Translate for the host's wave range; cache by it; `--host-lanes`.~~
2. ~~Adapter selection for tests and tools.~~
3. ~~Measure the title's programs by reason (`--host-lanes 32-32`), to
   order the later steps.~~ Testers did, and from `965ce30` on a local
   offline run does.
4. ~~In-half programs: console-lane programs whose cross-lane operations
   all stay inside a half, with no mask read whole.~~ No new lane assignment
   was needed (see above). A later widening is LDS in multi-wave groups,
   whose waves already meet at barriers.
5. ~~The console flow and the one-wave exchange (ballots, readlane,
   readfirstlane, writelane, mbcnt, divergent branches and loops).~~
6. ~~One-wave LDS sharing (a group barrier before each LDS access in the
   console flow).~~
7. Multi-wave groups without LDS: one console wave per host group (the
   translator's ids, then direct and indirect dispatch counts).
8. Multi-wave groups with LDS.

## Credit

The approach of pairing two 32-lane waves through groupshared memory, and
the finding that NVIDIA forms waves from non-consecutive thread ids in some
3-D groups, come from XCL's design notes (`docs/wave64-on-wave32.md` in
github.com/ivall/xcl and its fork Bathyal). XCL is GPL-2.0-only. Nothing
here is copied from its code.
