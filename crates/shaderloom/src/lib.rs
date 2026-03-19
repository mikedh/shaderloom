pub mod globutils;
pub mod gpu_exec;
pub mod luaexec;
pub mod naga_parse;

use anyhow::Result;
use luaexec::LuaExecutor;
use std::path::Path;

/// Main interface for the Shaderloom shader preprocessor.
///
/// This struct provides access to the shader preprocessing functionality
/// that can be used from Rust code, including in build scripts.
pub struct Shaderloom {
    executor: LuaExecutor,
}

impl Shaderloom {
    /// Create a new Shaderloom instance.
    ///
    /// This initializes the Lua runtime with all the embedded shader processing scripts.
    pub fn new() -> Self {
        Self {
            executor: LuaExecutor::new(),
        }
    }

    /// Build/bundle shaders from a loom.lua configuration file.
    ///
    /// This is equivalent to running `shaderloom build <path>` from the command line.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the loom.lua configuration file
    pub fn build_from_file(&self, path: impl AsRef<Path>) -> Result<()> {
        self.executor.run_script(path)
    }

    /// Run a specific Lua module with an optional argument.
    ///
    /// This is equivalent to running `shaderloom run <module> [arg]` from the command line.
    ///
    /// # Arguments
    ///
    /// * `module` - Name of the Lua module to run
    /// * `arg` - Optional string argument to pass to the module
    pub fn run_module(&self, module: &str, arg: Option<String>) -> Result<()> {
        self.executor.run_module(module, arg)
    }

    /// Get access to the underlying Lua executor for advanced usage.
    ///
    /// This provides direct access to the Lua runtime if you need to perform
    /// more complex operations not covered by the high-level API.
    pub fn executor(&self) -> &LuaExecutor {
        &self.executor
    }
}

impl Default for Shaderloom {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export related types for advanced users
pub use globutils::{GlobItem, glob_items};
pub use gpu_exec::{Binding, create_gpu, run_compute};
pub use naga_parse::{LuaWGSLModule, parse_and_validate_wgsl, parse_wgsl};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shaderloom_creation() {
        let _shaderloom = Shaderloom::new();
    }

    #[test]
    fn test_lua_modules() {
        let shaderloom = Shaderloom::new();

        // Test various Lua modules
        shaderloom
            .executor()
            .run_tests("utils.stringmanip")
            .unwrap();
        shaderloom.executor().run_tests("utils.common").unwrap();
        shaderloom
            .executor()
            .run_tests("preprocess.chunker")
            .unwrap();
        shaderloom
            .executor()
            .run_tests("preprocess.preprocessor")
            .unwrap();
        shaderloom.executor().run_tests("analysis.naga").unwrap();
        shaderloom.executor().run_tests("analysis.unify").unwrap();
        shaderloom
            .executor()
            .run_tests("targets.python.xgpu")
            .unwrap();
        shaderloom.executor().run_tests("tests.dev").unwrap();
        shaderloom
            .executor()
            .run_tests("tests.shader_test")
            .unwrap();
    }

    #[test]
    fn lua_string_utils() {
        LuaExecutor::new().run_tests("utils.stringmanip").unwrap();
    }

    #[test]
    fn lua_common_utils() {
        LuaExecutor::new().run_tests("utils.common").unwrap();
    }

    #[test]
    fn lua_preprocess() {
        LuaExecutor::new().run_tests("preprocess.chunker").unwrap();
        LuaExecutor::new()
            .run_tests("preprocess.preprocessor")
            .unwrap();
    }

    #[test]
    fn lua_naga() {
        LuaExecutor::new().run_tests("analysis.naga").unwrap();
    }

    #[test]
    fn lua_unify() {
        LuaExecutor::new().run_tests("analysis.unify").unwrap();
    }

    #[test]
    fn lua_python_target() {
        LuaExecutor::new().run_tests("targets.python.xgpu").unwrap();
    }

    #[test]
    fn lua_dev() {
        LuaExecutor::new().run_tests("tests.dev").unwrap();
    }

    #[test]
    fn lua_shader_test() {
        LuaExecutor::new()
            .run_tests("tests.shader_test")
            .unwrap();
    }
}
