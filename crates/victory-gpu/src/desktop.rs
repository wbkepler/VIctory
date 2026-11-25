//! The interactive desktop. Started from another desktop (a sandbox's own,
//! say), our windows would land where nobody sees them. [`attach`] moves the
//! process to the interactive window station (`WinSta0`) and the calling
//! thread to its `Default` desktop. A thread's desktop is its own, so every
//! thread that makes windows calls it before its first one; on a thread that
//! already has windows, or when either object can't be opened, nothing
//! changes.

use std::cell::Cell;
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::GENERIC_ALL;
use windows_sys::Win32::System::StationsAndDesktops::{
    HDESK, OpenDesktopW, OpenWindowStationW, SetProcessWindowStation, SetThreadDesktop,
};
use windows_sys::core::w;

/// `WINSTA_ALL_ACCESS`.
const WINSTA_ALL_ACCESS: u32 = 0x37f;

/// The `Default` desktop, opened once (0 if it couldn't be). Kept open for
/// the life of the process: threads keep using it.
static DESKTOP: OnceLock<usize> = OnceLock::new();

thread_local! {
    /// Whether this thread has been attached.
    static ATTACHED: Cell<bool> = const { Cell::new(false) };
}

fn desktop() -> HDESK {
    *DESKTOP.get_or_init(|| {
        // SAFETY: plain calls with literal names; the handles stay open.
        unsafe {
            let station = OpenWindowStationW(w!("WinSta0"), 0, WINSTA_ALL_ACCESS);
            if !station.is_null() && SetProcessWindowStation(station) == 0 {
                tracing::debug!("could not switch to the interactive window station");
            }
            let desktop = OpenDesktopW(w!("Default"), 0, 0, GENERIC_ALL);
            if desktop.is_null() {
                tracing::debug!("could not open the default desktop");
            }
            desktop as usize
        }
    }) as HDESK
}

/// Attaches the process to `WinSta0` and the calling thread to its
/// `Default` desktop (once per thread).
pub fn attach() {
    if ATTACHED.replace(true) {
        return;
    }
    let desktop = desktop();
    if desktop.is_null() {
        return;
    }
    // SAFETY: a desktop handle that stays open for the process's life.
    if unsafe { SetThreadDesktop(desktop) } == 0 {
        tracing::debug!("this thread stays on its own desktop");
    }
}
