//! Integration tests for the shaderloom library

use shaderloom::Shaderloom;
use std::path::Path;

#[test]
fn test_build_wgpu_example() {
    let shaderloom = Shaderloom::new();

    // Resolve relative to the crate manifest so the test runs regardless of the
    // process CWD (the examples live at the workspace root, not under the crate).
    let example_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/wgpu_bundle");
    let example_path = format!("{example_dir}/loom.lua");

    if Path::new(&example_path).exists() {
        shaderloom
            .build_from_file(&example_path)
            .expect("Failed to build example shader bundle");

        // triangle.wgsl marks `MAX_VERTS` with `# export()`; the rust.wgpu target
        // must emit it as a `pub const` in the generated struct file (exercises the
        // full preprocess -> naga -> emit -> default-template path).
        let structs = format!("{example_dir}/shader_structs.rs");
        let generated =
            std::fs::read_to_string(&structs).expect("struct definitions file not generated");
        assert!(
            generated.contains("pub const MAX_VERTS: u32 = 3;"),
            "exported const missing from generated structs:\n{generated}"
        );
        // Don't leave the generated artifact in the source tree.
        let _ = std::fs::remove_file(&structs);
    }
}

#[test]
fn test_run_module() {
    let shaderloom = Shaderloom::new();

    // Test running a specific module (this should work with the embedded Lua modules)
    shaderloom
        .run_module("utils.common", None)
        .expect("Failed to run utils.common module");
}
