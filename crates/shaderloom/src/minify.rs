//! WGSL minification via a naga round-trip.
//!
//! Kept in its own module so the rest of shaderloom is untouched. `minify_wgsl`
//! parses WGSL, re-emits it through naga's WGSL backend (which discards comments
//! and normalizes whitespace), and — when asked — shortens identifiers. The
//! result is always re-validated; if a transformed form fails to round-trip we
//! fall back to a form that is known to parse, so we never emit broken WGSL.

use anyhow::{Result, anyhow};

use naga::Module;
use naga::back::wgsl as wgsl_out;
use naga::front::wgsl;
use naga::valid::Capabilities as Caps;
use naga::valid::{ModuleInfo, ValidationFlags, Validator};

/// Capabilities Naga's WGSL front end can validate (mirrors the Naga CLI:
/// CLIP/CULL distance aren't expressible in WGSL).
fn wgsl_caps() -> Caps {
    Caps::all() & !(Caps::CLIP_DISTANCE | Caps::CULL_DISTANCE)
}

fn validate(module: &Module) -> Result<ModuleInfo> {
    Validator::new(ValidationFlags::all(), wgsl_caps())
        .validate(module)
        .map_err(|e| anyhow!("{}", e))
}

/// Minify a WGSL source string by round-tripping it through naga.
///
/// Always strips comments and normalizes whitespace (naga discards comments at
/// parse time and the WGSL backend emits canonical formatting). With `rename`,
/// additionally shortens identifiers (function/global/const names, parameters,
/// locals, `let` bindings) — **never** entry-point names (the host selects
/// pipelines by those strings) and not struct type/member names (they live in a
/// content-addressed arena). The result is re-parsed + re-validated; the renamed
/// form is used only if it round-trips, otherwise the comment-stripped form,
/// which always does.
pub fn minify_wgsl(src: &str, rename: bool) -> Result<String> {
    let mut module = wgsl::parse_str(src).map_err(|e| anyhow!("{}", e.emit_to_string(src)))?;
    validate(&module)?; // ensure the input is valid before compacting

    // Dead-code elimination: drop functions, globals, consts, and types that are
    // not reachable from an entry point. A real win for shaders that `#include`
    // shared helper files but use only a subset. `KeepUnused::No` keeps exactly
    // what the entry points reach.
    naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
    let info = validate(&module)?;

    // Comment-free, whitespace-normalized. Always valid; used as the fallback.
    let base = wgsl_out::write_string(&module, &info, wgsl_out::WriterFlags::empty())
        .map_err(|e| anyhow!("WGSL backend failed: {}", e))?;

    if !rename {
        return Ok(roundtrip_or(base.clone(), || base));
    }

    rename_identifiers(&mut module);
    // Names don't affect type/expression analysis, but revalidate to hand the
    // writer a matching ModuleInfo and to catch anything unexpected.
    let info = validate(&module)?;
    let renamed = wgsl_out::write_string(&module, &info, wgsl_out::WriterFlags::empty())
        .map_err(|e| anyhow!("WGSL backend failed after rename: {}", e))?;

    Ok(roundtrip_or(renamed, || base))
}

/// Return `candidate` if it re-parses + re-validates, else the `fallback`.
fn roundtrip_or(candidate: String, fallback: impl FnOnce() -> String) -> String {
    let ok = wgsl::parse_str(&candidate)
        .map_err(|_| ())
        .and_then(|m| validate(&m).map_err(|_| ()))
        .is_ok();
    if ok {
        candidate
    } else {
        eprintln!("shaderloom: minified WGSL failed to round-trip; falling back");
        fallback()
    }
}

/// Shorten identifiers in the mutable arenas to compact unique names.
///
/// Skipped on purpose: **entry-point names** (host references them by string)
/// and **struct type/member names** (`UniqueArena` rebuild/dedup hazard, small
/// win). naga's `Namer` reserves WGSL keywords and uniquifies collisions, so a
/// generated name that happens to be a keyword stays valid.
fn rename_identifiers(module: &mut Module) {
    let mut counter = 0usize;
    let mut next = || {
        let n = short_name(counter);
        counter += 1;
        n
    };

    for (_, c) in module.constants.iter_mut() {
        c.name = Some(next());
    }
    for (_, g) in module.global_variables.iter_mut() {
        g.name = Some(next());
    }
    for (_, f) in module.functions.iter_mut() {
        f.name = Some(next());
        for arg in f.arguments.iter_mut() {
            arg.name = Some(next());
        }
        for (_, local) in f.local_variables.iter_mut() {
            local.name = Some(next());
        }
        for name in f.named_expressions.values_mut() {
            *name = next();
        }
    }
    for ep in module.entry_points.iter_mut() {
        // ep.name is deliberately left untouched.
        for arg in ep.function.arguments.iter_mut() {
            arg.name = Some(next());
        }
        for (_, local) in ep.function.local_variables.iter_mut() {
            local.name = Some(next());
        }
        for name in ep.function.named_expressions.values_mut() {
            *name = next();
        }
    }
}

/// Generate a short identifier from a counter: `a..z`, then `aa..az`, `ba..`, …
fn short_name(mut n: usize) -> String {
    const ALPHA: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    let mut s = Vec::new();
    loop {
        s.push(ALPHA[n % 26]);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }
    s.reverse();
    s.into_iter().map(|b| b as char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHADER: &str = r#"
// A header comment that should be stripped.
const GROUPSIZE: u32 = 8u;
const COUNT: u32 = GROUPSIZE + 2u;

struct Uniforms {
    @align(16) scale: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;

var<workgroup> scratch: array<f32, COUNT>;

// helper with a descriptive_long_name we expect to be shortened
fn descriptive_long_name(an_argument: f32) -> f32 {
    let an_intermediate_value = an_argument * uniforms.scale;
    return an_intermediate_value + 1.0;
}

@compute @workgroup_size(GROUPSIZE, GROUPSIZE, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    scratch[0] = 0.0;
    out_buf[gid.x] = descriptive_long_name(f32(gid.x));
}
"#;

    #[test]
    fn strips_comments_and_preserves_entry_point() {
        let out = minify_wgsl(SHADER, /*rename=*/ true).unwrap();
        assert!(
            !out.contains("header comment"),
            "comments not stripped:\n{out}"
        );
        assert!(!out.contains("//"), "line comments remain:\n{out}");
        // Host selects the pipeline by this exact name — must survive.
        assert!(out.contains("fn cs_main"), "entry point renamed:\n{out}");
        assert!(
            !out.contains("descriptive_long_name") && !out.contains("an_intermediate_value"),
            "identifiers not shortened:\n{out}"
        );
        assert!(wgsl::parse_str(&out).is_ok());
    }

    #[test]
    fn round_trip_without_rename_keeps_names() {
        let out = minify_wgsl(SHADER, /*rename=*/ false).unwrap();
        assert!(
            !out.contains("//"),
            "comments should still be stripped:\n{out}"
        );
        assert!(out.contains("cs_main"));
        assert!(
            out.contains("descriptive_long_name"),
            "names unexpectedly changed:\n{out}"
        );
    }

    #[test]
    fn dead_code_is_stripped() {
        let src = r#"
fn used_helper(x: f32) -> f32 { return x * 2.0; }
fn unused_dead_helper(x: f32) -> f32 { return x + 999.0; }
@group(0) @binding(0) var<storage, read_write> out_buf: array<f32>;
@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out_buf[gid.x] = used_helper(f32(gid.x));
}
"#;
        // rename=false so names are stable for the assertions.
        let out = minify_wgsl(src, false).unwrap();
        assert!(out.contains("used_helper"), "used helper dropped:\n{out}");
        assert!(
            !out.contains("unused_dead_helper") && !out.contains("999.0"),
            "dead code not stripped:\n{out}"
        );
    }

    #[test]
    fn workgroup_size_const_survives_round_trip() {
        // `const` used in @workgroup_size / array size must round-trip cleanly.
        let out = minify_wgsl(SHADER, true).unwrap();
        assert!(
            wgsl::parse_str(&out).is_ok(),
            "minified shader invalid:\n{out}"
        );
    }
}
