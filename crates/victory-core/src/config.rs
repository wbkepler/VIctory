//! Where the user's files live.
//!
//! Resolution order for each value: environment variable, then the config
//! file (`VICTORY_CONFIG`, else `./victory.toml`), then the default.
//! See `victory.example.toml` at the repository root.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::policy::ensure_outside_repo;

#[derive(Debug, Default, Clone)]
pub struct Config {
    /// The user's dump of Grand Theft Auto VI.
    pub game_dir: Option<PathBuf>,
    /// Root for all derived data. Always outside the repository.
    pub cache_dir: PathBuf,
    /// What titles write and must keep (persistent local storage). Never
    /// inside the cache, which may be deleted at any time.
    pub data_dir: PathBuf,
    /// Other titles used to exercise platform-level code.
    pub test_titles: Vec<PathBuf>,
    /// The user's copy of AMD FidelityFX for PC (the folder with its
    /// upscaler DLL), which replaces a title's console build.
    pub fidelityfx_dir: Option<PathBuf>,
    /// Graphics pacing and presentation settings.
    pub graphics: GraphicsConfig,
    /// The config file that was read, if any.
    pub source: Option<PathBuf>,
}

/// Graphics pacing and presentation settings (`[graphics]` in `victory.toml`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphicsConfig {
    /// Target frame-rate cap in FPS (`None` = title default, `Some(0)` =
    /// uncapped, `Some(n)` = cap at `n` FPS).
    pub fps_limit: Option<u32>,
    /// Allow DXGI flip-model tearing when supported by the display and adapter.
    pub allow_tearing: bool,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            fps_limit: None,
            allow_tearing: true,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    paths: FilePaths,
    #[serde(default)]
    graphics: FileGraphics,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePaths {
    game_dir: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    #[serde(default)]
    test_titles: Vec<PathBuf>,
    fidelityfx_dir: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileGraphics {
    fps_limit: Option<u32>,
    allow_tearing: Option<bool>,
}

impl Config {
    /// Loads configuration from the environment and the config file.
    pub fn load() -> Result<Self> {
        let explicit = explicit_config_path(|k| std::env::var_os(k));
        let path = explicit
            .clone()
            .unwrap_or_else(|| PathBuf::from("victory.toml"));
        let file = if path.is_file() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            Some((path, text))
        } else if let Some(p) = explicit {
            anyhow::bail!(
                "VICTORY_CONFIG points at {}, which does not exist",
                p.display()
            );
        } else {
            None
        };
        let (source, parsed) = match file {
            Some((path, text)) => {
                let parsed: FileConfig =
                    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
                (Some(path), parsed)
            }
            None => (None, FileConfig::default()),
        };
        Self::resolve(parsed, source, |k| std::env::var_os(k))
    }

    fn resolve(
        file: FileConfig,
        source: Option<PathBuf>,
        env: impl Fn(&str) -> Option<std::ffi::OsString>,
    ) -> Result<Self> {
        let env_path = |k: &str| env(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        let test_titles = match env("VICTORY_TEST_TITLES") {
            Some(v) if !v.is_empty() => std::env::split_paths(&v)
                .filter(|p| !p.as_os_str().is_empty())
                .collect(),
            _ => file.paths.test_titles,
        };
        let cache_dir = env_path("VICTORY_CACHE_DIR")
            .or(file.paths.cache_dir)
            .map_or_else(|| default_dir("cache"), Ok)?;
        let data_dir = env_path("VICTORY_DATA_DIR")
            .or(file.paths.data_dir)
            .map_or_else(|| default_dir("data"), Ok)?;
        let fps_limit = match env("VICTORY_FPS_LIMIT")
            .map(|v| v.to_string_lossy().trim().to_owned())
            .filter(|s| !s.is_empty())
        {
            Some(raw) => {
                let parsed = raw.parse::<u32>().with_context(|| {
                    format!(
                        "parsing VICTORY_FPS_LIMIT={raw:?} as a non-negative integer FPS limit (0 = uncapped)"
                    )
                })?;
                Some(parsed)
            }
            None => file.graphics.fps_limit,
        };
        let allow_tearing = match env("VICTORY_ALLOW_TEARING")
            .map(|v| v.to_string_lossy().trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
        {
            Some(raw) => match raw.as_str() {
                "1" | "true" | "yes" | "on" => true,
                "0" | "false" | "no" | "off" => false,
                _ => anyhow::bail!(
                    "parsing VICTORY_ALLOW_TEARING={raw:?} as a boolean (expected true/false or 1/0)"
                ),
            },
            None => file.graphics.allow_tearing.unwrap_or(true),
        };
        Ok(Config {
            game_dir: env_path("VICTORY_GAME_DIR").or(file.paths.game_dir),
            cache_dir,
            data_dir,
            test_titles,
            fidelityfx_dir: env_path("VICTORY_FIDELITYFX_DIR").or(file.paths.fidelityfx_dir),
            graphics: GraphicsConfig {
                fps_limit,
                allow_tearing,
            },
            source,
        })
    }

    /// The game directory, or a helpful error if it is not configured.
    pub fn require_game_dir(&self) -> Result<&Path> {
        self.game_dir.as_deref().context(
            "no game directory configured: set VICTORY_GAME_DIR or [paths].game_dir in \
             victory.toml (see victory.example.toml)",
        )
    }

    /// Returns `<cache>/<sub>`, creating it, after checking the asset policy.
    pub fn cache_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.cache_dir, sub)
    }

    /// Returns `<data>/<sub>`, creating it, after checking the asset policy.
    pub fn data_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.data_dir, sub)
    }
}

/// The config file explicitly named by the environment, if any. An empty
/// `VICTORY_CONFIG` means unset, like the other path variables.
fn explicit_config_path(env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
    env("VICTORY_CONFIG")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn subdir(root: &Path, sub: &str) -> Result<PathBuf> {
    let dir = root.join(sub);
    ensure_outside_repo(&dir)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// `%LOCALAPPDATA%/victory/<name>` (elsewhere the XDG or home cache).
fn default_dir(name: &str) -> Result<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .context("cannot determine a user directory; set VICTORY_CACHE_DIR and VICTORY_DATA_DIR")?;
    Ok(base.join("victory").join(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn environment_overrides_file() {
        let file = FileConfig {
            paths: FilePaths {
                game_dir: Some("from-file".into()),
                cache_dir: Some("cache-from-file".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_GAME_DIR").then(|| OsString::from("from-env"));
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-env"));
        assert_eq!(cfg.cache_dir, PathBuf::from("cache-from-file"));
    }

    #[test]
    fn empty_env_values_are_ignored() {
        let file = FileConfig {
            paths: FilePaths {
                game_dir: Some("from-file".into()),
                ..Default::default()
            },
            graphics: FileGraphics {
                fps_limit: Some(120),
                allow_tearing: Some(false),
            },
        };
        let env = |k: &str| {
            matches!(
                k,
                "VICTORY_GAME_DIR" | "VICTORY_FPS_LIMIT" | "VICTORY_ALLOW_TEARING"
            )
            .then(OsString::new)
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-file"));
        assert_eq!(cfg.graphics.fps_limit, Some(120));
        assert!(!cfg.graphics.allow_tearing);
    }

    #[test]
    fn graphics_section_is_optional_and_configurable() {
        let legacy: FileConfig = toml::from_str(
            r#"
[paths]
game_dir = 'C:\games\test'
"#,
        )
        .unwrap();
        let cfg = Config::resolve(legacy, None, |_| None).unwrap();
        assert_eq!(cfg.graphics.fps_limit, None);
        assert!(cfg.graphics.allow_tearing);

        let with_graphics: FileConfig = toml::from_str(
            r#"
[paths]
game_dir = 'C:\games\test'

[graphics]
fps_limit = 0
allow_tearing = false
"#,
        )
        .unwrap();
        let cfg = Config::resolve(with_graphics, None, |_| None).unwrap();
        assert_eq!(cfg.graphics.fps_limit, Some(0));
        assert!(!cfg.graphics.allow_tearing);
    }

    #[test]
    fn fps_limit_and_tearing_env_override_file() {
        let file = FileConfig {
            graphics: FileGraphics {
                fps_limit: Some(60),
                allow_tearing: Some(false),
            },
            ..Default::default()
        };
        let env = |k: &str| match k {
            "VICTORY_FPS_LIMIT" => Some(OsString::from("144")),
            "VICTORY_ALLOW_TEARING" => Some(OsString::from("true")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.graphics.fps_limit, Some(144));
        assert!(cfg.graphics.allow_tearing);

        let bad_fps = Config::resolve(FileConfig::default(), None, |k| {
            (k == "VICTORY_FPS_LIMIT").then(|| OsString::from("fast"))
        });
        assert!(bad_fps.is_err());

        let bad_tearing = Config::resolve(FileConfig::default(), None, |k| {
            (k == "VICTORY_ALLOW_TEARING").then(|| OsString::from("maybe"))
        });
        assert!(bad_tearing.is_err());
    }

    #[test]
    fn empty_test_titles_env_is_ignored() {
        let file = FileConfig {
            paths: FilePaths {
                test_titles: vec!["from-file".into()],
                ..Default::default()
            },
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_TEST_TITLES").then(OsString::new);
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.test_titles, vec![PathBuf::from("from-file")]);
    }

    #[test]
    fn test_titles_env_splits_paths() {
        let value = std::env::join_paths(["first", "second"]).unwrap();
        let file = FileConfig::default();
        let env = |k: &str| (k == "VICTORY_TEST_TITLES").then(|| value.clone());
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(
            cfg.test_titles,
            vec![PathBuf::from("first"), PathBuf::from("second")]
        );
    }

    #[test]
    fn empty_config_env_is_ignored() {
        let empty = |k: &str| (k == "VICTORY_CONFIG").then(OsString::new);
        assert_eq!(explicit_config_path(empty), None);
        let set = |k: &str| (k == "VICTORY_CONFIG").then(|| OsString::from("custom.toml"));
        assert_eq!(
            explicit_config_path(set),
            Some(PathBuf::from("custom.toml"))
        );
    }

    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../victory.example.toml");
        let parsed: FileConfig = toml::from_str(text).unwrap();
        assert!(parsed.paths.game_dir.is_some());
    }

    #[test]
    fn cache_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        let cfg = Config {
            cache_dir: inside,
            ..Default::default()
        };
        assert!(cfg.cache_subdir("x").is_err());
    }

    #[test]
    fn data_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let cfg = Config {
            data_dir: inside,
            ..Default::default()
        };
        assert!(cfg.data_subdir("x").is_err());
    }
}
