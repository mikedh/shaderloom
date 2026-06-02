//! WGSL minification via a naga round-trip — with a hard ABI-preservation guard.
//!
//! `minify_wgsl` parses WGSL, re-emits it through naga's WGSL backend (which
//! discards comments and normalizes whitespace) and — when asked — shortens
//! identifiers. naga's WGSL backend is a code *generator*, not a guaranteed
//! lossless round-trip: in particular it is known to drop explicit
//! `@align`/`@size` struct-member attributes, which silently changes uniform/
//! storage buffer offsets. The host uploads bytes in the *source* layout, so a
//! layout-shifted shader reads every field past the first scalar from the wrong
//! offset — producing garbage with no validation error.
//!
//! To make minification safe to ship, every candidate output is checked against
//! the source's **interface**: struct memory layouts (member byte offsets +
//! struct span), the `@group/@binding` resource interface, and entry-point
//! names + workgroup sizes. If any candidate diverges, we fall back — ultimately
//! to the original source — so the emitted WGSL is always interchangeable with
//! what the host was compiled against. Correctness beats size.

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

/// The host-observable ABI of a module. Two modules with equal `Interface` are
/// byte-for-byte interchangeable from the host's point of view: same buffer
/// layouts, same bind groups, same pipeline entry points. Identifier names
/// (other than entry points, which the host selects by string) are deliberately
/// excluded so renaming doesn't trip the guard.
#[derive(PartialEq, Eq, Debug)]
struct Interface {
    /// Per struct type: `(span, member_byte_offsets)`. This is exactly what an
    /// `@align`/`@size` drop changes. Sorted (a multiset) so type reordering is
    /// not a false positive.
    structs: Vec<(u32, Vec<u32>)>,
    /// Resource bindings: `(group, binding, address_space)`. Sorted.
    bindings: Vec<(u32, u32, String)>,
    /// Entry points: `(name, stage, workgroup_size)`. Names are the host ABI.
    entry_points: Vec<(String, String, [u32; 3])>,
}

fn interface(module: &Module) -> Interface {
    let mut structs: Vec<(u32, Vec<u32>)> = module
        .types
        .iter()
        .filter_map(|(_, ty)| match &ty.inner {
            naga::TypeInner::Struct { members, span } => {
                Some((*span, members.iter().map(|m| m.offset).collect()))
            }
            _ => None,
        })
        .collect();
    structs.sort();

    let mut bindings: Vec<(u32, u32, String)> = module
        .global_variables
        .iter()
        .filter_map(|(_, gv)| {
            gv.binding
                .as_ref()
                .map(|b| (b.group, b.binding, format!("{:?}", gv.space)))
        })
        .collect();
    bindings.sort();

    let mut entry_points: Vec<(String, String, [u32; 3])> = module
        .entry_points
        .iter()
        .map(|ep| (ep.name.clone(), format!("{:?}", ep.stage), ep.workgroup_size))
        .collect();
    entry_points.sort();

    Interface {
        structs,
        bindings,
        entry_points,
    }
}

/// True iff `candidate` is valid WGSL whose interface is identical to `want`.
fn preserves_interface(candidate: &str, want: &Interface) -> bool {
    match wgsl::parse_str(candidate) {
        Ok(m) => validate(&m).is_ok() && interface(&m) == *want,
        Err(_) => false,
    }
}

/// Strip WGSL comments (line `//` and *nesting* block `/* */`) and blank lines,
/// touching nothing else. The token stream is unchanged, so the result parses
/// to an identical module — it can NEVER change a struct layout or binding ABI.
/// This is the layout-safe fallback for shaders the naga round-trip would
/// corrupt (its WGSL backend dropping `@align`/`@size`). WGSL has no string
/// literals, so a plain scanner is sufficient; block comments nest per spec.
fn strip_comments(src: &str) -> String {
    let mut uncommented = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    let mut block_depth: u32 = 0;
    while let Some(c) = chars.next() {
        if block_depth > 0 {
            match c {
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    block_depth += 1;
                }
                '*' if chars.peek() == Some(&'/') => {
                    chars.next();
                    block_depth -= 1;
                }
                '\n' => uncommented.push('\n'), // keep line breaks inside comments
                _ => {}
            }
        } else if c == '/' && chars.peek() == Some(&'/') {
            // Line comment: consume through the newline (which we re-emit).
            for n in chars.by_ref() {
                if n == '\n' {
                    uncommented.push('\n');
                    break;
                }
            }
        } else if c == '/' && chars.peek() == Some(&'*') {
            chars.next();
            block_depth = 1;
        } else {
            uncommented.push(c);
        }
    }
    // Drop blank lines and trailing whitespace.
    let mut out = String::with_capacity(uncommented.len());
    for line in uncommented.lines() {
        let trimmed = line.trim_end();
        if !trimmed.trim_start().is_empty() {
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    out
}

/// Minify a WGSL source string by round-tripping it through naga.
///
/// Always strips comments and normalizes whitespace (naga discards comments at
/// parse time and the WGSL backend emits canonical formatting). With `rename`,
/// additionally shortens identifiers (function/global/const names, parameters,
/// locals, `let` bindings) — **never** entry-point names (the host selects
/// pipelines by those strings) and not struct type/member names.
///
/// Every candidate output is verified to preserve the source's struct layouts,
/// binding interface, and entry points (see [`Interface`]). The first candidate
/// that preserves the interface is returned, most-minified first; if none do
/// (e.g. naga dropped an `@align`), the **original source** is returned
/// unchanged so the result is always ABI-compatible with the host.
pub fn minify_wgsl(src: &str, rename: bool) -> Result<String> {
    let module = wgsl::parse_str(src).map_err(|e| anyhow!("{}", e.emit_to_string(src)))?;
    let info = validate(&module)?; // ensure the input is valid before minifying

    // The contract the output must honor, captured from the source module.
    let want = interface(&module);

    // NOTE: we intentionally do NOT run `naga::compact` dead-code elimination
    // here. DCE drops globals not reached from an entry point, which can remove
    // a binding the host still binds (an ABI change), and its size win is
    // marginal next to comment/whitespace stripping. Correctness first.

    // Candidate outputs. The naga round-trip (with/without rename) is the most
    // aggressive but can corrupt layout; `strip_comments` is a token-preserving
    // textual pass that is *always* layout-safe — so a shader whose `@align`
    // structs the round-trip would break still gets its comments/whitespace
    // stripped; raw `src` is the last resort. We keep every candidate that
    // provably preserves the source interface and emit the SMALLEST one, so a
    // layout hazard costs only the rename/canonicalization win on that shader,
    // not all minification of it.
    let mut candidates: Vec<String> = Vec::new();
    if rename {
        let mut renamed = module.clone();
        rename_identifiers(&mut renamed);
        if let Ok(rinfo) = validate(&renamed) {
            if let Ok(s) = wgsl_out::write_string(&renamed, &rinfo, wgsl_out::WriterFlags::empty()) {
                candidates.push(s);
            }
        }
    }
    if let Ok(s) = wgsl_out::write_string(&module, &info, wgsl_out::WriterFlags::empty()) {
        candidates.push(s); // comment-free, whitespace-normalized, original names
    }
    candidates.push(strip_comments(src)); // token-preserving → always layout-safe
    candidates.push(src.to_string()); // guaranteed valid + interface-preserving

    candidates
        .into_iter()
        .filter(|c| preserves_interface(c, &want))
        .min_by_key(|c| c.len())
        .ok_or_else(|| anyhow!("no interface-preserving minification (unreachable)"))
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

    /// Align-free shader: minification (incl. rename) is safe, so the output is
    /// comment-free and renamed.
    const SHADER: &str = r#"
// A header comment that should be stripped.
const GROUPSIZE: u32 = 8u;
const COUNT: u32 = GROUPSIZE + 2u;

struct Uniforms {
    scale: f32,
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

    /// `struct_layouts` of a parsed module: `(span, member_offsets)` per struct.
    fn layouts(wgsl_src: &str) -> Vec<(u32, Vec<u32>)> {
        interface(&wgsl::parse_str(wgsl_src).unwrap()).structs
    }

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

    /// Shaders exercising every layout-affecting WGSL construct. Each declares a
    /// struct whose member byte offsets would shift if the minifier dropped a
    /// layout attribute — which is exactly the bug that silently zeroed the
    /// carve (naga's WGSL backend dropping `@align`). Storage space keeps the
    /// fixtures valid for arbitrary offsets; the layout math is space-agnostic.
    const LAYOUT_HAZARDS: &[(&str, &str)] = &[
        // The real bug: `@align(16)` on scalars after two mat4x4 — offsets
        // 0,64,128,144,160,176... If `@align(16)` is dropped they pack to
        // 0,64,128,132,136,140 and every uniform field past the first is read
        // from the wrong byte.
        (
            "align16_scalars",
            r#"
struct DepthViewUniforms {
  @align(16) proj_mat: mat4x4<f32>,
  @align(16) metric_proj_mat: mat4x4<f32>,
  @align(16) tool_rad: f32,
  @align(16) tool_rad_px: f32,
  @align(16) footprint: i32,
  @align(16) axial_finish: f32,
}
@group(0) @binding(0) var<storage, read> u: DepthViewUniforms;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn cs_main() { o[0] = u.tool_rad + u.tool_rad_px + f32(u.footprint) + u.axial_finish + u.proj_mat[0][0] + u.metric_proj_mat[3][3]; }
"#,
        ),
        // vec3 leaves a 4-byte hole a following scalar packs into (offset 12).
        (
            "vec3_scalar_packing",
            r#"
struct S { v: vec3<f32>, s: f32, w: vec3<f32>, t: f32 }
@group(0) @binding(0) var<storage, read> u: S;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn cs_main() { o[0] = u.s + u.t + u.v.x + u.w.y; }
"#,
        ),
        // Explicit @size override changes the following member's offset.
        (
            "size_override",
            r#"
struct S { @size(32) a: f32, b: f32, c: f32 }
@group(0) @binding(0) var<storage, read> u: S;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn cs_main() { o[0] = u.a + u.b + u.c; }
"#,
        ),
        // Nested struct + array stride + mat3x3 (each column 16-aligned).
        (
            "nested_array_mat3",
            r#"
struct Inner { @align(16) a: f32, b: f32 }
struct S { m: mat3x3<f32>, arr: array<vec4<f32>, 3>, x: Inner, tail: f32 }
@group(0) @binding(0) var<storage, read> u: S;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn cs_main() { o[0] = u.m[0][0] + u.arr[2].w + u.x.a + u.x.b + u.tail; }
"#,
        ),
    ];

    /// The byte-level regression that would have crashed the original minifier
    /// hard: minify MUST NOT change any struct's `(size, member byte offsets)`,
    /// for any layout attribute, with or without renaming. The original
    /// `@align`-dropping minifier fails this on `align16_scalars` immediately.
    #[test]
    fn minify_never_alters_struct_byte_layout() {
        for (name, src) in LAYOUT_HAZARDS {
            let want = layouts(src);
            assert!(!want.is_empty(), "{name}: no structs parsed");
            for rename in [false, true] {
                let out = minify_wgsl(src, rename).unwrap();
                assert_eq!(
                    layouts(&out),
                    want,
                    "{name}: minify changed struct byte layout (rename={rename})\n--- minified ---\n{out}"
                );
            }
        }
    }

    /// The layout guard is load-bearing, proven directly: `preserves_interface`
    /// must REJECT a struct whose member offsets differ from the source (the
    /// packed form naga's backend emitted) and ACCEPT the faithful one. If this
    /// regresses, `minify_never_alters_struct_byte_layout` would silently start
    /// trusting broken output.
    #[test]
    fn guard_rejects_layout_shift() {
        let aligned = r#"
struct S { @align(16) a: f32, @align(16) b: f32 }
@group(0) @binding(0) var<storage, read> u: S;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1) fn cs_main() { o[0] = u.a + u.b; }
"#;
        // Same fields, `@align` dropped — `b` moves from byte 16 to byte 4.
        let packed = r#"
struct S { a: f32, b: f32 }
@group(0) @binding(0) var<storage, read> u: S;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1) fn cs_main() { o[0] = u.a + u.b; }
"#;
        assert_ne!(layouts(aligned), layouts(packed), "premise: layouts differ");
        let want = interface(&wgsl::parse_str(aligned).unwrap());
        assert!(
            preserves_interface(aligned, &want),
            "faithful form wrongly rejected"
        );
        assert!(
            !preserves_interface(packed, &want),
            "layout-shifted form accepted — the guard is broken"
        );
    }

    /// Entry-point names and workgroup sizes survive (host ABI).
    #[test]
    fn interface_is_preserved_for_minifiable_shader() {
        let before = interface(&wgsl::parse_str(SHADER).unwrap());
        let out = minify_wgsl(SHADER, true).unwrap();
        let after = interface(&wgsl::parse_str(&out).unwrap());
        assert_eq!(before, after, "interface drifted:\n{out}");
    }

    /// A shader the naga round-trip would corrupt (`@align` struct) must STILL be
    /// minified — just layout-safely. The output must drop comments and shrink,
    /// keep the exact byte layout, and not be the raw source.
    #[test]
    fn align_shader_is_still_comment_stripped() {
        const ALIGNED_WITH_COMMENTS: &str = r#"
// AUTOGENERATED header that is pure bloat and must not ship.
struct DepthViewUniforms {
    @align(16) proj_mat: mat4x4<f32>,   // projection
    @align(16) tool_rad: f32,           // tool radius (m)
    @align(16) tool_rad_px: f32,        /* in pixels */
    @align(16) axial_finish: f32,       // finish stock
}
@group(0) @binding(0) var<storage, read> u: DepthViewUniforms;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
// the entry point
@compute @workgroup_size(1)
fn cs_main() {
    // touch every field
    o[0] = u.tool_rad + u.tool_rad_px + u.axial_finish + u.proj_mat[0][0];
}
"#;
        let want = layouts(ALIGNED_WITH_COMMENTS);
        for rename in [false, true] {
            let out = minify_wgsl(ALIGNED_WITH_COMMENTS, rename).unwrap();
            // Layout byte-for-byte identical (the whole point).
            assert_eq!(layouts(&out), want, "layout changed (rename={rename}):\n{out}");
            // Still minified: comments gone, smaller, and NOT the raw source.
            assert!(!out.contains("//") && !out.contains("/*"), "comments survived:\n{out}");
            assert!(
                out.len() < ALIGNED_WITH_COMMENTS.len(),
                "not actually minified (rename={rename})"
            );
            assert_ne!(out, ALIGNED_WITH_COMMENTS, "fell back to raw source");
            // And the @align attributes are still present (proves layout-safe path).
            assert!(out.contains("@align(16)"), "align dropped:\n{out}");
        }
    }

    /// Nested and line comments (incl. `/* */`) are removed without disturbing
    /// tokens, and the result still parses.
    #[test]
    fn strip_comments_handles_nested_blocks() {
        let src = "const A: u32 = 1u; /* a /* nested */ block */ const B: u32 = 2u; // line\nconst C: u32 = 3u;";
        let out = strip_comments(src);
        assert!(!out.contains("/*") && !out.contains("//"), "comments remain: {out}");
        assert!(out.contains("const A") && out.contains("const B") && out.contains("const C"));
        assert!(wgsl::parse_str(&out).is_ok(), "stripped output invalid: {out}");
    }
}
