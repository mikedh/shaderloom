//! WGSL minification via a naga round-trip, made **layout-faithful**.
//!
//! `minify_wgsl` runs the full naga pipeline — `compact` (dead-code elimination,
//! incl. dead locals/sub-expressions) + identifier renaming + the WGSL backend.
//! That backend is faithful for everything *except* struct layout: its
//! `write_struct` ignores `StructMember.offset` and never emits `@align`/`@size`,
//! so it silently repacks `var<uniform>`/`storage` structs to *natural* layout.
//! The host (`mod.rs`, generated from the source layout, with `_pad` fields)
//! still expects the source layout, so the bytes desync — every field past the
//! first scalar is read from the wrong offset, with no validation error.
//!
//! It is **not** the minifier's place to second-guess the source's `@align`. So
//! after the round-trip we **re-inject** the exact byte layout: each data-struct
//! member gets an `@size(stride)` attribute computed from the IR, which pins its
//! offset to the source value. naga handles literally everything else.
//!
//! Every candidate output is checked against the source ABI (struct byte
//! layouts, `@group/@binding` set, entry points). Any candidate that diverges is
//! discarded; the smallest surviving candidate is emitted. So a re-inject bug or
//! a `compact` that would drop a host-bound binding degrades to a correct
//! fallback — never to wrong bytes.

use std::collections::BTreeMap;

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

// ---------------------------------------------------------------------------
// ABI: the host-observable contract every minified output must preserve.
// ---------------------------------------------------------------------------

/// Struct byte layouts keyed by type name (`name -> (span, member_offsets)`).
/// Names are stable through the pipeline (`rename` skips struct type + member
/// names; `compact` doesn't rename), so name-keying lets `compact` drop *unused*
/// structs while still catching any layout shift on the ones that remain.
type StructLayouts = BTreeMap<String, (u32, Vec<u32>)>;

fn struct_layouts(module: &Module) -> StructLayouts {
    module
        .types
        .iter()
        .filter_map(|(_, ty)| match &ty.inner {
            naga::TypeInner::Struct { members, span } => ty
                .name
                .clone()
                .map(|n| (n, (*span, members.iter().map(|m| m.offset).collect()))),
            _ => None,
        })
        .collect()
}

fn bindings(module: &Module) -> Vec<(u32, u32, String)> {
    let mut b: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(_, gv)| {
            gv.binding
                .as_ref()
                .map(|r| (r.group, r.binding, format!("{:?}", gv.space)))
        })
        .collect();
    b.sort();
    b
}

fn entry_points(module: &Module) -> Vec<(String, String, [u32; 3])> {
    let mut e: Vec<_> = module
        .entry_points
        .iter()
        .map(|ep| (ep.name.clone(), format!("{:?}", ep.stage), ep.workgroup_size))
        .collect();
    e.sort();
    e
}

/// True iff `candidate` is valid WGSL whose ABI matches the source: identical
/// entry points and resource bindings (no host binding dropped/added/moved), and
/// every struct it still declares has the **same byte layout** as the
/// same-named source struct. `compact` removing an *unused* struct is allowed; a
/// shifted offset or a dropped binding is not.
fn preserves_abi(
    candidate: &str,
    want_structs: &StructLayouts,
    want_bindings: &[(u32, u32, String)],
    want_entries: &[(String, String, [u32; 3])],
) -> bool {
    let module = match wgsl::parse_str(candidate) {
        Ok(m) => m,
        Err(_) => return false,
    };
    if validate(&module).is_err() {
        return false;
    }
    if entry_points(&module) != want_entries || bindings(&module) != *want_bindings {
        return false;
    }
    struct_layouts(&module)
        .iter()
        .all(|(name, layout)| want_structs.get(name) == Some(layout))
}

// ---------------------------------------------------------------------------
// Layout re-injection.
// ---------------------------------------------------------------------------

/// For each **data struct** (every member has `binding == None`; I/O structs
/// with `@location`/`@builtin` are emitted correctly by naga and skipped), the
/// list of `(member_name, stride)` where `stride` is the byte distance to the
/// next member (`span - offset` for the last). `@size(stride)` on each member
/// pins its offset to the source value: member *i* lands at `Σ stride[<i]`, and
/// every source offset is already naturally aligned, so nothing realigns.
fn data_struct_strides(module: &Module) -> Vec<(String, Vec<(String, u32)>)> {
    let mut out = Vec::new();
    for (_, ty) in module.types.iter() {
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            continue;
        };
        let Some(name) = ty.name.clone() else { continue };
        // Skip I/O structs and any struct whose first member isn't at offset 0
        // (none here; the guard backstops if that assumption is ever false).
        if members.iter().any(|m| m.binding.is_some())
            || members.first().is_none_or(|m| m.offset != 0)
        {
            continue;
        }
        let strides = members
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                let next = members.get(i + 1).map_or(*span, |n| n.offset);
                m.name.clone().map(|name| (name, next - m.offset))
            })
            .collect();
        out.push((name, strides));
    }
    out
}

/// Rewrite each data struct in `text` (naga's emission) to carry its
/// `@size(stride)` attributes, reproducing the source byte layout exactly.
fn reinject_layout(module: &Module, text: String) -> String {
    let mut text = text;
    for (struct_name, members) in data_struct_strides(module) {
        // naga emits `struct NAME {` (one space) with no nested braces in the
        // body, so the next `}` closes it.
        let header = format!("struct {struct_name} {{");
        let Some(start) = text.find(&header) else {
            continue;
        };
        let body_start = start + header.len();
        let Some(rel_end) = text[body_start..].find('}') else {
            continue;
        };
        let block_end = body_start + rel_end;
        let mut block = text[start..block_end].to_string();
        for (member, stride) in &members {
            // `name: ` is unique within a struct body (member types never
            // contain it), so this anchors the one member declaration.
            let anchor = format!("{member}: ");
            let with_size = format!("@size({stride}) {member}: ");
            block = block.replacen(&anchor, &with_size, 1);
        }
        text.replace_range(start..block_end, &block);
    }
    text
}

// ---------------------------------------------------------------------------
// Textual comment strip (the always-safe fallback / last resort).
// ---------------------------------------------------------------------------

/// Strip WGSL comments (line `//` and *nesting* block `/* */`) and blank lines,
/// touching nothing else — token stream unchanged, so layout/ABI is preserved by
/// construction. WGSL has no string literals, so a plain scanner suffices.
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
                '\n' => uncommented.push('\n'),
                _ => {}
            }
        } else if c == '/' && chars.peek() == Some(&'/') {
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

/// Collapse formatting whitespace — naga emits one indented statement per line,
/// but WGSL is delimiter-based, so the newlines + indentation are pure fat.
///
/// Conservative by design: runs of whitespace collapse to a single space, and a
/// space is dropped only when it sits next to a separator (`{}()[];,:`) that
/// cannot fuse with a neighbour into a different token. Spaces around operators
/// (`= + - < > & | .` …) are **left alone** so we can never silently change
/// tokenization/semantics (e.g. `- -` → `--`). The ABI guard re-parses the
/// result regardless, so a surprise degrades to a fallback rather than wrong WGSL.
fn collapse_whitespace(text: &str) -> String {
    // 1. Collapse every whitespace run to a single space.
    let mut single = String::with_capacity(text.len());
    let mut in_ws = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !in_ws {
                single.push(' ');
                in_ws = true;
            }
        } else {
            single.push(c);
            in_ws = false;
        }
    }
    // 2. Drop spaces adjacent to token separators.
    let is_sep = |c: char| matches!(c, '{' | '}' | '(' | ')' | '[' | ']' | ';' | ',' | ':');
    let chars: Vec<char> = single.chars().collect();
    let mut out = String::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == ' ' {
            let prev_sep = out.chars().last().is_some_and(is_sep);
            let next_sep = chars.get(i + 1).copied().is_some_and(is_sep);
            if prev_sep || next_sep {
                continue;
            }
        }
        out.push(c);
    }
    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

/// Minify a WGSL source string for shipping in a binary.
///
/// The goal is **obfuscation**, not byte count: the shipped shader should be
/// tree-shaken (dead code — functions, globals, *and* dead locals — gone via
/// `compact`) and renamed to opaque identifiers, then run through naga's WGSL
/// backend. naga's writer drops struct `@align`/`@size`, so we **re-inject** the
/// exact byte layout afterwards (see [`reinject_layout`]). This single path is
/// what ships whenever it preserves the source ABI — even when a plain
/// comment-strip would be *smaller*, because readable identifiers in the binary
/// are the thing we're trying to avoid.
///
/// The fallbacks exist purely for correctness, in obfuscation-first order: if
/// `compact` would drop a host-bound binding we keep the rename but skip the
/// tree-shake; only if the naga path can't round-trip at all do we fall back to
/// a (readable but correct) comment-strip, then the raw source. The raw source
/// always preserves its own ABI, so this never fails or ships wrong bytes.
pub fn minify_wgsl(src: &str, rename: bool) -> Result<String> {
    let module = wgsl::parse_str(src).map_err(|e| anyhow!("{}", e.emit_to_string(src)))?;
    validate(&module)?;

    let want_structs = struct_layouts(&module);
    let want_bindings = bindings(&module);
    let want_entries = entry_points(&module);

    // compact (tree-shake) + optional rename → naga write → re-inject layout.
    let build = |do_compact: bool| -> Option<String> {
        let mut m = module.clone();
        if do_compact {
            naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
        }
        if rename {
            rename_identifiers(&mut m);
        }
        let info = validate(&m).ok()?;
        let text = wgsl_out::write_string(&m, &info, wgsl_out::WriterFlags::empty()).ok()?;
        Some(collapse_whitespace(&reinject_layout(&m, text)))
    };

    // Obfuscation-first preference; the first ABI-preserving form wins.
    let ordered = [
        build(true),                              // tree-shake + rename + re-inject + ws
        build(false),                             // rename + re-inject (compact dropped a binding)
        Some(collapse_whitespace(&strip_comments(src))), // correctness fallback (readable-ish)
        Some(src.to_string()),                    // last resort
    ];
    for cand in ordered.into_iter().flatten() {
        if preserves_abi(&cand, &want_structs, &want_bindings, &want_entries) {
            return Ok(cand);
        }
    }
    Err(anyhow!("unreachable: raw source preserves its own ABI"))
}

/// Shorten identifiers in the mutable arenas to compact unique names.
///
/// Skipped on purpose: **entry-point names** (host references them by string)
/// and **struct type/member names** (kept stable so layout re-injection and the
/// ABI guard can match by name). naga's `Namer` reserves WGSL keywords and
/// uniquifies collisions, so a generated keyword-like name stays valid.
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

    fn layouts(src: &str) -> StructLayouts {
        struct_layouts(&wgsl::parse_str(src).unwrap())
    }

    /// The exact carve bug, end to end: a `var<uniform>` with `@align(16)` on
    /// scalars must round-trip byte-identical. The naive naga write drops the
    /// `@align`; re-injection must restore the offsets `0,64,128,144,160,…`.
    const ALIGN_SHADER: &str = r#"
// header comment, should be stripped
struct DepthViewUniforms {
    @align(16) proj_mat: mat4x4<f32>,   // projection
    @align(16) metric_proj_mat: mat4x4<f32>,
    @align(16) tool_rad: f32,           // radius
    @align(16) tool_rad_px: f32,        /* px */
    @align(16) footprint: i32,
    @align(16) axial_finish: f32,
}
@group(0) @binding(0) var<uniform> u: DepthViewUniforms;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(8)
fn cs_main() {
    o[0] = u.tool_rad + u.tool_rad_px + f32(u.footprint) + u.axial_finish
         + u.proj_mat[0][0] + u.metric_proj_mat[3][3];
}
"#;

    #[test]
    fn naive_naga_write_drops_align_then_reinject_restores_it() {
        let module = wgsl::parse_str(ALIGN_SHADER).unwrap();
        let info = validate(&module).unwrap();
        let want = struct_layouts(&module);

        // Prove the hazard is real: naga's writer alone shifts the layout.
        let naive = wgsl_out::write_string(&module, &info, wgsl_out::WriterFlags::empty()).unwrap();
        assert_ne!(
            layouts(&naive),
            want,
            "naga write unexpectedly preserved @align (hazard premise)"
        );

        // Re-injection restores it byte-for-byte.
        let fixed = reinject_layout(&module, naive);
        assert_eq!(layouts(&fixed), want, "re-inject did not restore layout:\n{fixed}");
        assert!(fixed.contains("@size("), "no @size attribute injected:\n{fixed}");
    }

    #[test]
    fn minify_ships_the_obfuscating_path_with_correct_layout() {
        let want = layouts(ALIGN_SHADER);
        for rename in [false, true] {
            let out = minify_wgsl(ALIGN_SHADER, rename).unwrap();
            // Layout byte-identical (the carve correctness invariant).
            assert_eq!(layouts(&out), want, "layout changed (rename={rename}):\n{out}");
            // The naga path shipped (re-injected `@size`), NOT the readable
            // comment-strip fallback — even though the latter may be smaller.
            assert!(out.contains("@size("), "did not take the obfuscating path:\n{out}");
            assert!(!out.contains("//") && !out.contains("/*"), "comments survived:\n{out}");
            // Formatting whitespace collapsed.
            assert!(!out.contains('\n'), "newlines survived:\n{out}");
        }
        // With rename, descriptive global identifiers are obfuscated away.
        let renamed = minify_wgsl(ALIGN_SHADER, true).unwrap();
        assert!(!renamed.contains("var<uniform> u"), "globals not renamed:\n{renamed}");
    }

    /// Byte-layout corpus across every layout-affecting construct. Fails hard if
    /// any of them ever drifts through the pipeline.
    #[test]
    fn minify_never_alters_struct_byte_layout() {
        const HAZARDS: &[&str] = &[
            // @align(16) scalars (the bug).
            r#"struct S { @align(16) a: vec4<f32>, @align(16) b: f32, @align(16) c: f32 }
               @group(0) @binding(0) var<storage, read> u: S;
               @group(0) @binding(1) var<storage, read_write> o: array<f32>;
               @compute @workgroup_size(1) fn m() { o[0] = u.b + u.c + u.a.x; }"#,
            // vec3 + scalar packing (scalar fills the vec3 tail).
            r#"struct S { v: vec3<f32>, s: f32, w: vec3<f32>, t: f32 }
               @group(0) @binding(0) var<storage, read> u: S;
               @group(0) @binding(1) var<storage, read_write> o: array<f32>;
               @compute @workgroup_size(1) fn m() { o[0] = u.s + u.t + u.v.x + u.w.y; }"#,
            // @size override.
            r#"struct S { @size(32) a: f32, b: f32, c: f32 }
               @group(0) @binding(0) var<storage, read> u: S;
               @group(0) @binding(1) var<storage, read_write> o: array<f32>;
               @compute @workgroup_size(1) fn m() { o[0] = u.a + u.b + u.c; }"#,
            // nested struct + array stride + mat3.
            r#"struct Inner { @align(16) a: f32, b: f32 }
               struct S { m: mat3x3<f32>, arr: array<vec4<f32>, 3>, x: Inner, tail: f32 }
               @group(0) @binding(0) var<storage, read> u: S;
               @group(0) @binding(1) var<storage, read_write> o: array<f32>;
               @compute @workgroup_size(1) fn m() { o[0] = u.m[0][0] + u.arr[2].w + u.x.a + u.tail; }"#,
        ];
        for (i, src) in HAZARDS.iter().enumerate() {
            let want = layouts(src);
            for rename in [false, true] {
                let out = minify_wgsl(src, rename).unwrap();
                assert_eq!(layouts(&out), want, "hazard {i} drifted (rename={rename}):\n{out}");
            }
        }
    }

    /// `compact` (DCE) actually removes dead code — the user's reason for keeping
    /// the IR pipeline. A large dead helper makes the compact candidate the
    /// smallest, so it is the one emitted.
    #[test]
    fn compact_removes_dead_code() {
        let src = r#"
fn used(x: f32) -> f32 { return x * 2.0; }
fn dead_helper_with_a_recognizable_body(x: f32) -> f32 {
    var acc = 0.0;
    for (var i = 0; i < 1000; i = i + 1) { acc = acc + x * 3.14159 - 2.71828; }
    return acc + 123456.0;
}
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn m() { o[0] = used(1.0); }
"#;
        let out = minify_wgsl(src, true).unwrap();
        assert!(!out.contains("123456.0"), "dead code survived:\n{out}");
        assert!(wgsl::parse_str(&out).is_ok());
    }

    /// If `compact` would drop a host-bound (declared-but-unused) binding, that
    /// candidate fails the ABI guard and we fall back to one that keeps it.
    #[test]
    fn unused_binding_is_never_dropped() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@group(0) @binding(1) var<storage, read> unused_but_host_binds_it: array<f32>;
@compute @workgroup_size(1)
fn m() { o[0] = 1.0; }
"#;
        let out = minify_wgsl(src, true).unwrap();
        let got = bindings(&wgsl::parse_str(&out).unwrap());
        assert_eq!(got, bindings(&wgsl::parse_str(src).unwrap()), "a binding was dropped:\n{out}");
    }

    /// `compact` tree-shakes dead `#include`-style helpers (incl. transitively
    /// dead chains a→b) while re-injection preserves the `@align` layout — the
    /// obfuscation goal and the correctness invariant together on one shader.
    #[test]
    fn tree_shake_removes_dead_chain_keeps_layout() {
        let src = r#"
struct U { @align(16) a: f32, @align(16) b: f32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
fn used(x: f32) -> f32 { return x * 2.0; }
fn dead_a(x: f32) -> f32 { return dead_b(x) + 111.0; }
fn dead_b(x: f32) -> f32 { return x + 222.0; }
@compute @workgroup_size(1)
fn cs_main() { o[0] = used(u.a) + u.b; }
"#;
        let want = layouts(src);
        let out = minify_wgsl(src, true).unwrap();
        assert_eq!(layouts(&out), want, "layout changed:\n{out}");
        // The whole dead chain (a→b) is gone; the used helper + entry stay.
        assert!(!out.contains("111.0") && !out.contains("222.0"), "dead code survived:\n{out}");
        assert!(out.contains("cs_main"), "entry point removed:\n{out}");
        assert!(wgsl::parse_str(&out).is_ok());
    }

    /// Whitespace collapse tightens separators but never touches operator
    /// spacing (so it can't fuse `- -` into `--`, `> >` into `>>`, etc.).
    #[test]
    fn collapse_whitespace_tightens_separators_keeps_operators() {
        assert_eq!(collapse_whitespace("a = b ;\n  c ( d , e ) ;"), "a = b;c(d,e);");
        // operator-adjacent spaces preserved:
        assert_eq!(collapse_whitespace("x  -  -y"), "x - -y");
        assert_eq!(collapse_whitespace("a < < b"), "a < < b");
    }

    #[test]
    fn strip_comments_handles_nested_blocks() {
        let src = "const A: u32 = 1u; /* a /* nested */ block */ const B: u32 = 2u; // line\nconst C: u32 = 3u;";
        let out = strip_comments(src);
        assert!(!out.contains("/*") && !out.contains("//"), "comments remain: {out}");
        assert!(out.contains("const A") && out.contains("const B") && out.contains("const C"));
        assert!(wgsl::parse_str(&out).is_ok());
    }
}
