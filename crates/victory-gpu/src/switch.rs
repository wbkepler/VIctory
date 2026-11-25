//! The `VICTORY_*` on/off switches (`VICTORY_D3D12_DEBUG=1` and its siblings).

use std::ffi::OsStr;

/// Whether a switch's value turns it on: it is set, and is not `0`.
///
/// A switch used to be on whenever its variable existed, so `VICTORY_D3D12_WARP=0` (typed to switch it off, or left
/// over from comparing two runs) still selected the software adapter. Any other value keeps the old meaning.
fn value_is_on(value: Option<&OsStr>) -> bool {
    value.is_some_and(|v| v != "0")
}

/// Whether the environment variable `name` switches something on.
pub(crate) fn on(name: &str) -> bool {
    value_is_on(std::env::var_os(name).as_deref())
}

/// Whether a switch that is on unless set to `0` is on (an unset variable
/// is on). Unit tests keep the old default (off when unset): they check the
/// paths these switches replace, and a test that wants one sets it.
fn default_on_value(value: Option<&OsStr>) -> bool {
    match value {
        None => !cfg!(test),
        Some(v) => v != "0",
    }
}

/// Whether the environment variable `name`, a switch that is on by default,
/// is on: unset or any value but `0` (the proven performance switches,
/// `VICTORY_GPU_UPLOAD`, `VICTORY_GPU_UPLOAD_WRITES`, `VICTORY_LAZY_COPY_BACK`).
pub(crate) fn on_by_default(name: &str) -> bool {
    default_on_value(std::env::var_os(name).as_deref())
}

#[cfg(test)]
mod tests {
    use super::value_is_on;
    use std::ffi::OsStr;

    #[test]
    fn unset_is_off() {
        assert!(!value_is_on(None));
    }

    #[test]
    fn zero_is_off() {
        assert!(!value_is_on(Some(OsStr::new("0"))));
    }

    #[test]
    fn one_is_on() {
        assert!(value_is_on(Some(OsStr::new("1"))));
    }

    #[test]
    fn a_default_on_switch_is_off_only_at_zero() {
        use super::default_on_value;
        assert!(!default_on_value(Some(OsStr::new("0"))));
        assert!(default_on_value(Some(OsStr::new("1"))));
        // Unit tests keep the old default; the game gets the new one.
        assert!(!default_on_value(None));
    }

    #[test]
    fn other_values_keep_their_old_meaning() {
        // Documented as `=1`; anything else that was on before stays on: only `0` is new.
        for value in ["", "true", "yes", "2", "00", " 0", "0 "] {
            assert!(value_is_on(Some(OsStr::new(value))), "{value:?}");
        }
    }
}
