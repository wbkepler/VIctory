//! Frame pacing: the title sets a frame interval and waits for each frame
//! to begin. Ours is a clock started by the first interval set, ticking
//! once per interval; the token is the frame number. Each event kind
//! (origin, vblank, flip) happens once per tick, so two waits for the same
//! kind never end on the same tick: Grand Theft Auto VI polls the origin
//! with "token on time-out" (starting its next frame at once) and is paced
//! by a blocking wait for the flip before each present.
//!
//! shortcut: event offsets are ignored (every event is at the tick); tie
//! the clock to presentation if a title paces by them.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::object::{HRESULT, S_FALSE, S_OK, This};

/// What a wait returns when its time is up first.
const TIMEOUT: HRESULT = S_FALSE;
/// The frame-origin event: the one that starts frames and makes tokens.
const ORIGIN: u32 = 0;
/// The flip event (present pacing).
const FLIP: u32 = 2;
/// Make the frame's token even if the wait times out (the title starts the
/// frame early).
const TOKEN_ON_TIMEOUT: u32 = 2;
const SIXTIETH: Duration = Duration::from_micros(16_667);
/// Sentinel stored in `FPS_LIMIT` when no user override is active.
const FPS_UNSET: u64 = u64::MAX;
/// Sub-millisecond spin tail after the waitable timer wakes so high-refresh
/// pacing (60/120/144/240 Hz) lands accurately without burning CPU.
const SPIN_TAIL: Duration = Duration::from_micros(500);

struct Clock {
    start: Instant,
    /// Interval requested by the title via `set_interval`.
    title_interval: Duration,
    /// Effective interval in use (`None` when uncapped).
    interval: Option<Duration>,
    /// The tick number at `start` (ticks go on across interval changes).
    first: u64,
}

static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);
static FRAME: AtomicU64 = AtomicU64::new(0);
static FLIPS: AtomicU64 = AtomicU64::new(0);
static FPS_LIMIT: AtomicU64 = AtomicU64::new(FPS_UNSET);
static EPOCH: OnceLock<Instant> = OnceLock::new();
static LAST_ORIGIN_US: AtomicU64 = AtomicU64::new(0);
static LAST_EVENT_US: AtomicU64 = AtomicU64::new(0);
static LAST_FRAME_US: AtomicU64 = AtomicU64::new(0);

/// Monotonic frame-pacing statistics recorded by the frame clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Total frames started (origin tokens handed out).
    pub frames: u64,
    /// Total flip waits completed.
    pub flips: u64,
    /// Monotonic timestamp (microseconds since the clock started) of the most
    /// recent completed frame event.
    pub last_timestamp_us: u64,
    /// Measured duration (microseconds) between the last two frame origins.
    pub last_frame_us: u64,
    /// Effective pacing interval in microseconds (`0` when uncapped).
    pub target_interval_us: u64,
}

fn clock() -> std::sync::MutexGuard<'static, Option<Clock>> {
    CLOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn epoch_us() -> u64 {
    let start = EPOCH.get_or_init(Instant::now);
    (start.elapsed().as_micros() as u64).max(1)
}

/// Computes the effective pacing interval given the title's requested
/// interval and the optional user FPS limit (`None` = title default,
/// `Some(0)` = uncapped, `Some(n)` = cap at `n` FPS).
fn effective_interval(title_interval: Duration, limit: Option<u32>) -> Option<Duration> {
    match limit {
        None => Some(title_interval),
        Some(0) => None,
        Some(fps) => {
            let us = (1_000_000.0 / f64::from(fps)).round().max(1.0) as u64;
            Some(Duration::from_micros(us))
        }
    }
}

fn fps_limit() -> Option<u32> {
    let raw = FPS_LIMIT.load(Ordering::Relaxed);
    (raw != FPS_UNSET).then_some(raw as u32)
}

/// Sets a process-wide frame-rate override (`None` = title default,
/// `Some(0)` = uncapped, `Some(n)` = cap at `n` FPS).
pub fn set_fps_limit(limit: Option<u32>) {
    FPS_LIMIT.store(limit.map_or(FPS_UNSET, u64::from), Ordering::Relaxed);
    let mut guard = clock();
    if let Some(c) = guard.as_mut() {
        let first = LAST
            .iter()
            .map(|t| t.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0);
        c.start = Instant::now();
        c.interval = effective_interval(c.title_interval, limit);
        c.first = first;
    }
}

/// Returns a monotonic snapshot of frame-pacing statistics.
pub fn stats() -> FrameStats {
    let target_interval_us = {
        let guard = clock();
        let title_interval = guard.as_ref().map_or(SIXTIETH, |c| c.title_interval);
        effective_interval(title_interval, fps_limit()).map_or(0, |d| d.as_micros() as u64)
    };
    FrameStats {
        frames: FRAME.load(Ordering::Relaxed),
        flips: FLIPS.load(Ordering::Relaxed),
        last_timestamp_us: LAST_EVENT_US.load(Ordering::Relaxed),
        last_frame_us: LAST_FRAME_US.load(Ordering::Relaxed),
        target_interval_us,
    }
}

/// Device method: a frame every `length` microseconds. The period is how
/// many intervals a frame may take to reach the display (pipeline depth),
/// not part of the rate.
pub(crate) extern "system" fn set_interval(
    _: This,
    _output: This,
    length: u32,
    _period: u32,
    _flags: u32,
) -> HRESULT {
    let title_interval = match length {
        0 => SIXTIETH,
        us => Duration::from_micros(u64::from(us)),
    };
    let limit = fps_limit();
    let interval = effective_interval(title_interval, limit);
    match (limit, interval) {
        (Some(0), None) => {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            let mut first_log = false;
            LOGGED.call_once(|| {
                first_log = true;
                tracing::info!(
                    ?title_interval,
                    "frame interval overridden: uncapped (fps_limit = 0)"
                );
            });
            if !first_log {
                tracing::debug!(
                    ?title_interval,
                    period = _period,
                    flags = _flags,
                    "frame interval re-set; keeping uncapped override (fps_limit = 0)"
                );
            }
        }
        (Some(fps), Some(eff)) => {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            let mut first_log = false;
            LOGGED.call_once(|| {
                first_log = true;
                tracing::info!(
                    ?title_interval,
                    effective = ?eff,
                    fps,
                    "frame interval overridden by fps_limit"
                );
            });
            if !first_log {
                tracing::debug!(
                    ?title_interval,
                    effective = ?eff,
                    fps,
                    period = _period,
                    flags = _flags,
                    "frame interval re-set; keeping fps_limit override"
                );
            }
        }
        _ => {
            tracing::info!(interval = ?title_interval, "frame interval");
        }
    }
    // A new interval starts from now, after the last tick handed out.
    let first = LAST
        .iter()
        .map(|t| t.load(Ordering::Relaxed))
        .max()
        .unwrap_or(0);
    *clock() = Some(Clock {
        start: Instant::now(),
        title_interval,
        interval,
        first,
    });
    S_OK
}

/// Device method: events are scheduled at every tick.
pub(crate) extern "system" fn schedule(
    _: This,
    _kind: u32,
    _offset: u32,
    _list: *const u8,
    _flags: u32,
) -> HRESULT {
    S_OK
}

/// Frames begun so far (frame waits that ended).
pub(crate) fn current() -> u64 {
    FRAME.load(Ordering::Relaxed)
}

/// Per event kind (origin, vblank, flip and later kinds): the last tick a
/// wait for it ended on.
static LAST: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn last(kind: u32) -> &'static AtomicU64 {
    &LAST[(kind as usize).min(LAST.len() - 1)]
}

/// The next tick after now and after the last one a wait for `kind` took
/// (an event happens once per tick: a wait woken a little early must not
/// let the next one end on the same tick), and how long until it. A wait
/// that comes within `limit` takes the tick here, under the clock's lock,
/// so two threads waiting for one kind never take the same tick.
fn next_tick(kind: u32, limit: Duration) -> (u64, Duration) {
    let mut clock = clock();
    let fps = fps_limit();
    let c = clock.get_or_insert_with(|| Clock {
        start: Instant::now(),
        title_interval: SIXTIETH,
        interval: effective_interval(SIXTIETH, fps),
        first: 0,
    });
    let Some(interval) = c.interval else {
        // Uncapped: no wait, so the tick is always taken here.
        let tick = last(kind).load(Ordering::Relaxed) + 1;
        last(kind).fetch_max(tick, Ordering::Relaxed);
        return (tick, Duration::ZERO);
    };
    let elapsed = c.start.elapsed().as_nanos();
    let interval = interval.as_nanos().max(1);
    let tick =
        (c.first + (elapsed / interval + 1) as u64).max(last(kind).load(Ordering::Relaxed) + 1);
    let at = u128::from(tick - c.first) * interval;
    let wait = Duration::from_nanos(at.saturating_sub(elapsed) as u64);
    if wait <= limit {
        last(kind).fetch_max(tick, Ordering::Relaxed);
    }
    (tick, wait)
}

/// Sleeps for `duration` using a high-resolution Windows waitable timer plus a
/// short spin tail for accurate high-refresh pacing.
fn sleep_precise(duration: Duration) {
    if duration.is_zero() {
        return;
    }
    let deadline = Instant::now() + duration;
    if let Some(coarse) = duration.checked_sub(SPIN_TAIL)
        && !coarse.is_zero()
    {
        sleep_waitable(coarse);
    }
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

#[cfg(windows)]
fn sleep_waitable(duration: Duration) {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::{
        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateWaitableTimerExW, INFINITE, SetWaitableTimer,
        WaitForSingleObject,
    };
    const TIMER_ALL_ACCESS: u32 = 0x001F_0003;

    struct Timer(HANDLE);
    impl Drop for Timer {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: handle owned exclusively by this thread-local wrapper.
                unsafe { CloseHandle(self.0) };
            }
        }
    }

    thread_local! {
        static TIMER: Timer = {
            // SAFETY: creating an unnamed high-resolution synchronization waitable timer.
            let handle = unsafe {
                CreateWaitableTimerExW(
                    std::ptr::null(),
                    std::ptr::null(),
                    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                    TIMER_ALL_ACCESS,
                )
            };
            Timer(handle)
        };
    }

    let waited = TIMER.with(|t| {
        if t.0.is_null() {
            return false;
        }
        // Negative relative due time in 100-nanosecond intervals.
        let ticks = (duration.as_nanos() / 100).clamp(1, i64::MAX as u128) as i64;
        let due = -ticks;
        // SAFETY: `t.0` is a live waitable timer handle owned by this thread.
        let armed = unsafe { SetWaitableTimer(t.0, &due, 0, None, std::ptr::null(), 0) != 0 };
        if !armed {
            return false;
        }
        // SAFETY: waiting on the armed timer handle.
        unsafe { WaitForSingleObject(t.0, INFINITE) };
        true
    });
    if !waited {
        std::thread::sleep(duration);
    }
}

#[cfg(not(windows))]
fn sleep_waitable(duration: Duration) {
    std::thread::sleep(duration);
}

/// Device method: blocks until the next tick (or the time-out); an origin
/// wait then hands out the new frame's token. Other events keep the token
/// the title passes. Ticks are absolute (start plus whole intervals), so
/// lateness never accumulates.
pub(crate) extern "system" fn wait(
    _: This,
    kind: u32,
    timeout_ms: u32,
    _list: *const u8,
    flags: u32,
    token: *mut u64,
) -> HRESULT {
    let origin = kind == ORIGIN;
    let limit = Duration::from_millis(u64::from(timeout_ms));
    // The tick is only logged (diagnostics); taking it is what counts.
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    let (tick, wait) = next_tick(kind, limit);
    let hr = if wait > limit {
        sleep_precise(limit);
        TIMEOUT
    } else {
        sleep_precise(wait);
        S_OK
    };
    let now_us = epoch_us();
    LAST_EVENT_US.fetch_max(now_us, Ordering::Relaxed);
    if kind == FLIP && hr == S_OK {
        FLIPS.fetch_add(1, Ordering::Relaxed);
    }
    // A frame begun on a time-out takes no tick: its flip still does.
    if origin && (hr == S_OK || flags & TOKEN_ON_TIMEOUT != 0) {
        // Tokens count frames (consecutive), not ticks.
        let frame = FRAME.fetch_add(1, Ordering::Relaxed) + 1;
        let prev = LAST_ORIGIN_US.swap(now_us, Ordering::Relaxed);
        if prev != 0 {
            LAST_FRAME_US.store(now_us.saturating_sub(prev), Ordering::Relaxed);
        }
        if !token.is_null() {
            // SAFETY: checked non-null; the title passes its token.
            unsafe { token.write(frame) };
        }
    }
    #[cfg(feature = "diagnostics")]
    tracing::debug!(
        kind,
        timeout_ms,
        flags,
        hr,
        tick,
        wait_us = wait.as_micros() as u64,
        frame = current(),
        "frame wait"
    );
    hr
}

/// Device method: refreshes monotonic frame statistics and reports 0 console
/// records until the console's frame-statistics output struct layout is
/// verified (avoiding unverified writes into caller-owned `out`).
pub(crate) extern "system" fn statistics(
    _: This,
    _token: u64,
    _kinds: u32,
    count: *mut u32,
    _out: *mut u8,
) -> HRESULT {
    LAST_EVENT_US.fetch_max(epoch_us(), Ordering::Relaxed);
    if !count.is_null() {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
    }
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_interval_honors_override_modes() {
        let thirty = Duration::from_micros(33_333);
        assert_eq!(effective_interval(thirty, None), Some(thirty));
        assert_eq!(effective_interval(thirty, Some(0)), None);
        assert_eq!(
            effective_interval(thirty, Some(60)),
            Some(Duration::from_micros(16_667))
        );
        assert_eq!(
            effective_interval(thirty, Some(120)),
            Some(Duration::from_micros(8_333))
        );
        assert_eq!(
            effective_interval(thirty, Some(144)),
            Some(Duration::from_micros(6_944))
        );
        assert_eq!(
            effective_interval(thirty, Some(240)),
            Some(Duration::from_micros(4_167))
        );
    }

    #[test]
    fn waits_end_on_ticks_and_count_frames() {
        let null = std::ptr::null();
        set_fps_limit(None);
        assert_eq!(set_interval(null, null, 2_000, 1, 0), S_OK);
        let mut first = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut first), S_OK);
        let before = Instant::now();
        let mut second = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut second), S_OK);
        assert!(second > first);
        assert!(before.elapsed() <= Duration::from_millis(50));
        assert_eq!(wait(null, 0, 0, std::ptr::null(), 0, &mut second), TIMEOUT);
        // Tokens count frames, a frame started early on a time-out too.
        let mut early = 0;
        let flags = TOKEN_ON_TIMEOUT;
        assert_eq!(
            wait(null, 0, 0, std::ptr::null(), flags, &mut early),
            TIMEOUT
        );
        let mut next = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut next), S_OK);
        assert!(early == second + 1 && next == early + 1);
        assert!(LAST[0].load(Ordering::Relaxed) >= 2);

        // The title's loop (the clock is global, so in the same test): an
        // origin poll that times out (and starts the frame), then a
        // blocking flip wait. Each flip takes its own tick.
        // 50 ms ticks: a thread woken 15-30 ms late on a busy host (seen with
        // 5 ms ticks) still makes its tick.
        assert_eq!(set_interval(null, null, 50_000, 3, 0), S_OK);
        let mut token = 0;
        let flips: Vec<u64> = (0..4)
            .map(|_| {
                let _ = wait(null, 0, 0, std::ptr::null(), TOKEN_ON_TIMEOUT, &mut token);
                assert_eq!(
                    wait(null, FLIP, u32::MAX, std::ptr::null(), 0, &mut token),
                    S_OK
                );
                LAST[2].load(Ordering::Relaxed)
            })
            .collect();
        assert!(flips.windows(2).all(|w| w[1] == w[0] + 1), "{flips:?}");
        // Even when a flip wait begins on the tick another just ended on.
        let (tick, _) = next_tick(FLIP, Duration::ZERO);
        assert!(tick > flips[3]);

        // Uncapped override (Some(0)): even with a 30 FPS title interval,
        // waits return immediately and still advance frames and flips.
        set_fps_limit(Some(0));
        assert_eq!(set_interval(null, null, 33_333, 3, 0), S_OK);
        let start = Instant::now();
        let frames_before = current();
        for _ in 0..8 {
            let mut tok = 0;
            assert_eq!(wait(null, ORIGIN, 0, std::ptr::null(), 0, &mut tok), S_OK);
            assert_eq!(
                wait(null, FLIP, u32::MAX, std::ptr::null(), 0, &mut tok),
                S_OK
            );
        }
        assert!(
            start.elapsed() < Duration::from_millis(50),
            "uncapped mode must not sleep on the title's 30 FPS interval"
        );
        assert_eq!(current(), frames_before + 8);
        let uncapped_stats = stats();
        assert_eq!(uncapped_stats.target_interval_us, 0);
        assert!(uncapped_stats.frames >= frames_before + 8);
        assert!(uncapped_stats.flips >= 12);
        assert!(uncapped_stats.last_timestamp_us > 0);

        // Repeated `set_interval` calls under uncapped mode keep zero wait
        // across every event kind (origin=0, vblank=1, flip=2).
        assert_eq!(set_interval(null, null, 33_333, 2, 1), S_OK);
        assert_eq!(stats().target_interval_us, 0);
        let recall_start = Instant::now();
        for kind in [ORIGIN, 1, FLIP] {
            let mut tok = 0;
            assert_eq!(
                wait(null, kind, u32::MAX, std::ptr::null(), 0, &mut tok),
                S_OK
            );
        }
        assert!(
            recall_start.elapsed() < Duration::from_millis(20),
            "repeated set_interval under uncapped mode must not re-enable 30 FPS waits"
        );

        // Fixed FPS override (e.g. 120 FPS) overrides the 30 FPS interval.
        set_fps_limit(Some(120));
        assert_eq!(set_interval(null, null, 33_333, 3, 0), S_OK);
        assert_eq!(stats().target_interval_us, 8_333);

        // COM statistics entry point safely reports 0 records without writing `out`.
        let mut count = 99u32;
        assert_eq!(
            statistics(null, token, 0, &mut count, std::ptr::null_mut()),
            S_OK
        );
        assert_eq!(count, 0);

        // Restore default override state.
        set_fps_limit(None);
    }

    /// Two waits for one kind that begin together (two threads) take two
    /// ticks: the first takes its tick before it sleeps. A time-out takes
    /// none. The old code marked the tick after the sleep, so both got it.
    #[test]
    fn waits_begun_together_take_different_ticks() {
        const VBLANK: u32 = 1;
        let long = Duration::from_secs(1);
        let (first, _) = next_tick(VBLANK, long);
        let (second, _) = next_tick(VBLANK, long);
        assert!(second > first, "{second} after {first}");
        next_tick(VBLANK, Duration::ZERO);
        assert_eq!(last(VBLANK).load(Ordering::Relaxed), second);
    }
}
