//! WGSL minification via a naga round-trip, made **layout-faithful**.
//!
//! `minify_wgsl` tree-shakes (`compact`), optionally renames identifiers, and
//! re-emits through naga's WGSL backend. That backend drops struct `@align`/
//! `@size` and repacks `var<uniform>`/`storage` structs to natural layout — but
//! the generated host still expects the source layout, so the bytes would
//! silently desync. We therefore **re-inject** the exact byte layout afterwards
//! (see [`reinject_layout`]).
//!
//! Every candidate is re-checked against the source ABI (struct byte layouts,
//! `@group/@binding` set, entry points); any divergence is discarded and we fall
//! back, ending at the raw source. So a re-inject bug or a `compact` that drops a
//! host-bound binding degrades to a correct output, never to wrong bytes.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};

use naga::Module;
use naga::back::wgsl as wgsl_out;
use naga::front::wgsl;
use naga::valid::Capabilities as Caps;
use naga::valid::{ModuleInfo, ValidationFlags, Validator};

/// Capabilities Naga's WGSL front end can validate (per the Naga CLI, CLIP/CULL
/// distance aren't expressible in WGSL). Kept local so this module is
/// self-contained.
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

fn bindings(module: &Module) -> Vec<(u32, u32, naga::AddressSpace)> {
    let mut b: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(_, gv)| gv.binding.as_ref().map(|r| (r.group, r.binding, gv.space)))
        .collect();
    b.sort();
    b
}

fn entry_points(module: &Module) -> Vec<(String, naga::ShaderStage, [u32; 3])> {
    let mut e: Vec<_> = module
        .entry_points
        .iter()
        .map(|ep| (ep.name.clone(), ep.stage, ep.workgroup_size))
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
    want_bindings: &[(u32, u32, naga::AddressSpace)],
    want_entries: &[(String, naga::ShaderStage, [u32; 3])],
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

/// Rewrite the data structs in `text` (naga's emission) to carry `@size(stride)`
/// attributes that reproduce the source byte layout — but **only** the structs
/// naga didn't already lay out correctly.
///
/// naga's natural layout matches the source for most structs; those are left
/// untouched, both to keep the output clean and to shrink the footprint of this
/// string surgery to exactly the structs with non-natural (`@align`/`@size`)
/// layout. `want` is the source layout; `natural` is what naga emitted here.
fn reinject_layout(module: &Module, text: String, want: &StructLayouts) -> String {
    // What naga laid out on its own (no `@size` yet). If this somehow doesn't
    // parse, an empty map means "differs from source" → fall back to stamping.
    let natural = wgsl::parse_str(&text)
        .map(|m| struct_layouts(&m))
        .unwrap_or_default();

    let mut text = text;
    for (struct_name, members) in data_struct_strides(module) {
        // naga already got this struct right — nothing to pin.
        if natural.get(&struct_name) == want.get(&struct_name) {
            continue;
        }
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
// Type aliases.
// ---------------------------------------------------------------------------

/// Replace naga's fully-qualified vector/matrix types with the equivalent WGSL
/// predeclared aliases (`vec4<f32>` → `vec4f`, `mat3x3<f32>` → `mat3x3f`). Same
/// types — lossless — it just drops the `<…>` boilerplate. Each pattern is a
/// complete type token (identifiers can't contain `<>`, and naga emits no
/// comments/strings), so a plain substring replace is safe.
fn shorten_type_aliases(text: &str) -> String {
    const ALIASES: &[(&str, &str)] = &[
        ("vec2<f32>", "vec2f"), ("vec3<f32>", "vec3f"), ("vec4<f32>", "vec4f"),
        ("vec2<i32>", "vec2i"), ("vec3<i32>", "vec3i"), ("vec4<i32>", "vec4i"),
        ("vec2<u32>", "vec2u"), ("vec3<u32>", "vec3u"), ("vec4<u32>", "vec4u"),
        ("vec2<f16>", "vec2h"), ("vec3<f16>", "vec3h"), ("vec4<f16>", "vec4h"),
        ("mat2x2<f32>", "mat2x2f"), ("mat2x3<f32>", "mat2x3f"), ("mat2x4<f32>", "mat2x4f"),
        ("mat3x2<f32>", "mat3x2f"), ("mat3x3<f32>", "mat3x3f"), ("mat3x4<f32>", "mat3x4f"),
        ("mat4x2<f32>", "mat4x2f"), ("mat4x3<f32>", "mat4x3f"), ("mat4x4<f32>", "mat4x4f"),
    ];
    let mut text = text.to_string();
    for (long, short) in ALIASES {
        if text.contains(long) {
            text = text.replace(long, short);
        }
    }
    text
}

// ---------------------------------------------------------------------------
// Whitespace collapse.
// ---------------------------------------------------------------------------

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
/// The goal is **obfuscation**, not byte count: tree-shake dead code via
/// `compact`, optionally rename identifiers to opaque names, re-emit through
/// naga's WGSL backend, then re-inject the source byte layout (see
/// [`reinject_layout`]).
///
/// Candidates are tried obfuscation-first and the first ABI-preserving one wins:
/// tree-shake + rename, then rename only (in case `compact` would drop a
/// host-bound binding), then the raw source. The raw source preserves its own
/// ABI, so this never fails or ships wrong bytes.
pub fn minify_wgsl(src: &str, rename: bool) -> Result<String> {
    let module = wgsl::parse_str(src).map_err(|e| anyhow!("{}", e.emit_to_string(src)))?;
    validate(&module)?;

    let want_structs = struct_layouts(&module);
    let want_bindings = bindings(&module);
    let want_entries = entry_points(&module);

    // compact (tree-shake) + optional rename → naga write → re-inject layout → ws.
    let build = |do_compact: bool| -> Option<String> {
        let mut m = module.clone();
        if do_compact {
            naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
        }
        let counter = if rename { rename_identifiers(&mut m) } else { 0 };
        let info = validate(&m).ok()?;
        let text = wgsl_out::write_string(&m, &info, wgsl_out::WriterFlags::empty()).ok()?;
        let text = reinject_layout(&m, text, &want_structs);
        // The IR rename can't reach naga's write-time baked temporaries (`_e12`),
        // so shorten them in the emitted text, continuing the same counter.
        let text = if rename { rename_baked_temps(&text, counter) } else { text };
        Some(collapse_whitespace(&shorten_type_aliases(&text)))
    };

    let preserved = |c: &str| preserves_abi(c, &want_structs, &want_bindings, &want_entries);

    // Obfuscation-first; the first ABI-preserving candidate wins. Each `build`
    // runs only if the previous candidate was rejected, and the raw source
    // preserves its own ABI, so this always returns.
    if let Some(c) = build(true).filter(|c| preserved(c)) {
        return Ok(c);
    }
    if let Some(c) = build(false).filter(|c| preserved(c)) {
        return Ok(c);
    }
    if preserved(src) {
        return Ok(src.to_string());
    }
    Err(anyhow!("unreachable: raw source preserves its own ABI"))
}

/// Shorten identifiers in the mutable arenas to compact unique names, returning
/// the next free counter value (so [`rename_baked_temps`] can continue the same
/// sequence without colliding).
///
/// Skipped on purpose: **entry-point names** (host references them by string)
/// and **struct type/member names** (kept stable so layout re-injection and the
/// ABI guard can match by name). naga's `Namer` reserves WGSL keywords and
/// uniquifies collisions, so a generated keyword-like name stays valid.
fn rename_identifiers(module: &mut Module) -> usize {
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
    counter
}

/// Rename naga's *baked* temporaries (`_e12`, …): the WGSL backend invents these
/// for reused sub-expressions at write time, so they never exist in the IR and
/// [`rename_identifiers`] can't reach them. Each distinct `_e<n>` gets a fresh
/// short name continuing from `counter`, so every name in the module stays
/// unique. Matched token-wise (a `_e1` embedded in a longer identifier is left
/// alone) and emitted by copying verbatim slices, so it's UTF-8 safe.
///
/// Unlike [`rename_identifiers`], this runs *after* naga's `Namer`, so it must
/// avoid WGSL reserved words itself — a minted `as`/`let`/`if` would be invalid.
fn rename_baked_temps(text: &str, mut counter: usize) -> String {
    use naga::keywords::wgsl::RESERVED;
    let mut mint = move || loop {
        let n = short_name(counter);
        counter += 1;
        if !RESERVED.contains(&n.as_str()) {
            return n;
        }
    };

    let bytes = text.as_bytes();
    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut names: BTreeMap<&str, String> = BTreeMap::new();
    let mut out = String::with_capacity(text.len());
    let mut last = 0; // start of the not-yet-flushed verbatim run
    let mut i = 0;
    while i < bytes.len() {
        let at_boundary = i == 0 || !is_word(bytes[i - 1]);
        let is_baked = bytes[i] == b'_'
            && bytes.get(i + 1) == Some(&b'e')
            && bytes.get(i + 2).is_some_and(u8::is_ascii_digit);
        if at_boundary && is_baked {
            let mut j = i + 3;
            while bytes.get(j).is_some_and(u8::is_ascii_digit) {
                j += 1;
            }
            out.push_str(&text[last..i]);
            let short = names.entry(&text[i..j]).or_insert_with(&mut mint);
            out.push_str(short);
            last = j;
            i = j;
        } else {
            i += 1;
        }
    }
    out.push_str(&text[last..]);
    out
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
        let fixed = reinject_layout(&module, naive, &want);
        assert_eq!(layouts(&fixed), want, "re-inject did not restore layout:\n{fixed}");
        assert!(fixed.contains("@size("), "no @size attribute injected:\n{fixed}");
    }

    /// Pin the **absolute** byte layout of the `@align(16)` uniform through a
    /// full `minify_wgsl` round-trip. The other layout tests assert
    /// source-equals-output, which would still pass if naga ever changed how it
    /// lays out `@align` (both sides drifting together); this hard-codes the ABI
    /// the host actually depends on, so such a drift fails loudly here.
    #[test]
    fn align_byte_layout_is_pinned_through_minify() {
        // mat4x4 (64) , mat4x4 (64) , then four @align(16) scalars at 16-byte
        // steps; struct rounds up to 192.
        const EXPECT_SPAN: u32 = 192;
        const EXPECT_OFFSETS: &[u32] = &[0, 64, 128, 144, 160, 176];

        let (span, offsets) = layouts(ALIGN_SHADER)["DepthViewUniforms"].clone();
        assert_eq!((span, offsets.as_slice()), (EXPECT_SPAN, EXPECT_OFFSETS),
            "source @align layout is not what the host ABI expects");

        for rename in [false, true] {
            let out = minify_wgsl(ALIGN_SHADER, rename).unwrap();
            let (span, offsets) = layouts(&out)["DepthViewUniforms"].clone();
            assert_eq!((span, offsets.as_slice()), (EXPECT_SPAN, EXPECT_OFFSETS),
                "minified @align layout drifted (rename={rename}):\n{out}");
        }
    }

    /// The ABI guard is the entire safety argument for the re-inject hack
    /// ("degrades to a correct fallback, never wrong bytes"), so prove it
    /// actually *rejects* divergence — not just that the happy path is correct.
    #[test]
    fn abi_guard_rejects_divergence() {
        let module = wgsl::parse_str(ALIGN_SHADER).unwrap();
        let want_s = struct_layouts(&module);
        let want_b = bindings(&module);
        let want_e = entry_points(&module);

        // Positive control: the source preserves its own ABI.
        assert!(preserves_abi(ALIGN_SHADER, &want_s, &want_b, &want_e));

        // Layout shift: naga's naive write drops `@align`, moving offsets. The
        // guard must catch it (this is exactly the hazard re-injection fixes).
        let info = validate(&module).unwrap();
        let naive = wgsl_out::write_string(&module, &info, wgsl_out::WriterFlags::empty()).unwrap();
        assert!(
            !preserves_abi(&naive, &want_s, &want_b, &want_e),
            "guard accepted a layout-shifted candidate:\n{naive}"
        );

        // Dropped binding: valid WGSL with the same struct/entry but binding(1)
        // gone. A host that binds it would desync, so the guard must reject it.
        let dropped = r#"
struct DepthViewUniforms {
    @align(16) proj_mat: mat4x4<f32>,
    @align(16) metric_proj_mat: mat4x4<f32>,
    @align(16) tool_rad: f32,
    @align(16) tool_rad_px: f32,
    @align(16) footprint: i32,
    @align(16) axial_finish: f32,
}
@group(0) @binding(0) var<uniform> u: DepthViewUniforms;
@compute @workgroup_size(8)
fn cs_main() { _ = u.tool_rad; }
"#;
        // Sanity: the candidate is valid WGSL (so we're testing the ABI check,
        // not a parse failure).
        assert!(wgsl::parse_str(dropped).is_ok());
        assert!(
            !preserves_abi(dropped, &want_s, &want_b, &want_e),
            "guard accepted a candidate missing a binding:\n{dropped}"
        );
    }

    #[test]
    fn minify_ships_the_obfuscating_path_with_correct_layout() {
        let want = layouts(ALIGN_SHADER);
        for rename in [false, true] {
            let out = minify_wgsl(ALIGN_SHADER, rename).unwrap();
            // Layout byte-identical (the carve correctness invariant).
            assert_eq!(layouts(&out), want, "layout changed (rename={rename}):\n{out}");
            // The naga path shipped (re-injected `@size`), not the raw fallback.
            assert!(out.contains("@size("), "did not take the obfuscating path:\n{out}");
            assert!(!out.contains("//") && !out.contains("/*"), "comments survived:\n{out}");
            // Formatting whitespace collapsed.
            assert!(!out.contains('\n'), "newlines survived:\n{out}");
        }
        // With rename, descriptive global identifiers are obfuscated away.
        let renamed = minify_wgsl(ALIGN_SHADER, true).unwrap();
        assert!(!renamed.contains("var<uniform> u"), "globals not renamed:\n{renamed}");
    }

    /// A naturally-packed struct (no `@align`/`@size` overrides) is laid out
    /// correctly by naga on its own, so re-injection must leave it untouched: no
    /// `@size` noise, and the string surgery's footprint stays minimal.
    #[test]
    fn naturally_packed_struct_gets_no_size() {
        let src = r#"
struct P { width: i32, height: i32, count: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn m() { o[0] = (f32(p.width + p.height) * p.scale) + f32(p.count); }
"#;
        let want = layouts(src);
        for rename in [false, true] {
            let out = minify_wgsl(src, rename).unwrap();
            assert_eq!(layouts(&out), want, "layout changed (rename={rename}):\n{out}");
            assert!(!out.contains("@size("), "stamped @size on a natural struct:\n{out}");
        }
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

    /// Render pipeline (vertex + fragment) with both a `var<uniform>` **data
    /// struct** and an **I/O struct** (`@builtin`/`@location` members). Proves
    /// minify is stage-agnostic and that the I/O struct is left alone: its
    /// members are interpolated varyings, not memory-mapped, so they must *not*
    /// get `@size` — only the uniform data struct is re-injected.
    #[test]
    fn minify_render_pipeline_preserves_io_struct_and_stages() {
        let src = r#"
struct Uniforms {
    @align(16) color: vec3<f32>,
    @align(16) scale: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) in_pos: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(in_pos * u.scale, 0.0, 1.0);
    out.uv = in_pos;
    return out;
}

@fragment
fn fs_main(frag: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(u.color * frag.uv.x, 1.0);
}
"#;
        let want = layouts(src);
        let want_entries = entry_points(&wgsl::parse_str(src).unwrap());
        for rename in [false, true] {
            let out = minify_wgsl(src, rename).unwrap();
            // Uniform data struct layout preserved byte-for-byte.
            assert_eq!(layouts(&out), want, "layout changed (rename={rename}):\n{out}");
            // Both pipeline stages survive with their stage kinds intact.
            assert_eq!(
                entry_points(&wgsl::parse_str(&out).unwrap()),
                want_entries,
                "entry points/stages changed (rename={rename}):\n{out}"
            );
            // Exactly the two `Uniforms` members get `@size`; the I/O struct
            // (`VsOut`) is skipped, so the count is 2, not 4.
            assert_eq!(
                out.matches("@size(").count(),
                2,
                "I/O struct must not be re-injected (rename={rename}):\n{out}"
            );
        }
    }

    /// Any `_e<digit>` token (naga's baked-temporary naming scheme) present.
    fn has_baked_temp(s: &str) -> bool {
        s.as_bytes()
            .windows(3)
            .any(|w| w == b"_e0" || (w[0] == b'_' && w[1] == b'e' && w[2].is_ascii_digit()))
    }

    /// The post-pass is token-wise, gives each distinct temp a fresh name from
    /// the counter, and never touches a `_eN` embedded in a longer identifier.
    #[test]
    fn rename_baked_temps_is_tokenwise() {
        assert_eq!(
            rename_baked_temps("let _e1 = x; let _e2 = _e1; var my_e1: i32;", 0),
            "let a = x; let b = a; var my_e1: i32;"
        );
    }

    /// Running after naga's `Namer`, the pass must skip WGSL reserved words it
    /// would otherwise mint. `short_name(44) == "as"` (a keyword), so a temp
    /// minted at counter 44 must skip to the next legal name.
    #[test]
    fn rename_baked_temps_skips_reserved_words() {
        assert_eq!(short_name(44), "as", "premise: short_name(44) is a keyword");
        let out = rename_baked_temps("x _e9 y", 44);
        assert_ne!(out, "x as y", "minted a reserved word");
        assert!(wgsl::parse_str(&format!("fn f() {{ let {} = 1; }}",
            out.trim_start_matches("x ").trim_end_matches(" y"))).is_ok());
    }

    /// End to end: naga's writer invents baked temporaries the IR rename can't
    /// see, so renamed output must contain none. (The sanity check proves the
    /// test shader actually produces some, so this can't pass vacuously.)
    #[test]
    fn rename_eliminates_baked_temporaries() {
        let module = wgsl::parse_str(ALIGN_SHADER).unwrap();
        let info = validate(&module).unwrap();
        let naive = wgsl_out::write_string(&module, &info, wgsl_out::WriterFlags::empty()).unwrap();
        assert!(has_baked_temp(&naive), "test shader has no baked temps to exercise");

        let out = minify_wgsl(ALIGN_SHADER, true).unwrap();
        assert!(!has_baked_temp(&out), "baked temporaries survived rename:\n{out}");
        assert!(wgsl::parse_str(&out).is_ok(), "renamed output is not valid WGSL:\n{out}");
    }

    /// Normalize a shader to a name/layout/whitespace-independent canonical form
    /// that captures only its **compute logic** (expressions + control flow):
    /// parse → compact → deterministic rename → naga's own writer (which drops
    /// `@size`, normalizes whitespace, and re-bakes temporaries).
    fn canonical(src: &str) -> String {
        let mut m = wgsl::parse_str(src).unwrap();
        naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
        rename_identifiers(&mut m);
        let info = validate(&m).unwrap();
        wgsl_out::write_string(&m, &info, wgsl_out::WriterFlags::empty()).unwrap()
    }

    /// Semantic round-trip: the minified output must compute the *same function*
    /// as the source, not merely preserve layout/ABI. We compare canonical forms,
    /// but the reference is naga's own faithful emission of the (compacted)
    /// source rather than the source itself — so both sides have been through
    /// exactly one naga write and baked-temporary materialization is symmetric
    /// (otherwise inline-vs-`let` differences read as false mismatches). What
    /// survives canonicalization is purely the expression/control-flow shape, so
    /// this catches our text passes (re-inject, alias, whitespace, baked rename)
    /// silently altering logic — the one failure mode the layout/ABI tests miss.
    #[test]
    fn minify_preserves_compute_logic() {
        // arithmetic, matrix indexing, function calls (ALIGN_SHADER) +
        // control flow (for/if/else-if/else/break/continue/loop) + a render
        // pipeline (vertex+fragment, I/O struct) — the constructs our text
        // munging could plausibly corrupt.
        const CONTROL_FLOW: &str = r#"
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = i32(gid.x);
    var acc = 0.0;
    for (var i = 0; i < n; i = i + 1) {
        if (i % 2 == 0) { acc = acc + f32(i) - 1.0; }
        else if (i > 100) { break; }
        else { continue; }
    }
    var k = 0u;
    loop {
        k = k + 1u;
        if (k >= 10u) { break; }
    }
    o[gid.x] = (acc * 2.0) - f32(k);
}
"#;
        const RENDER: &str = r#"
struct U { @align(16) tint: vec3<f32>, @align(16) gain: f32 }
@group(0) @binding(0) var<uniform> u: U;
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex
fn vs(@location(0) p: vec2<f32>) -> VsOut {
    var o: VsOut;
    o.pos = vec4<f32>(p * u.gain, 0.0, 1.0);
    o.uv = p;
    return o;
}
@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(u.tint * i.uv.x, 1.0);
}
"#;
        for src in [ALIGN_SHADER, CONTROL_FLOW, RENDER] {
            // reference = naga's faithful write of the compacted source.
            let mut m = wgsl::parse_str(src).unwrap();
            naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
            let info = validate(&m).unwrap();
            let reference = wgsl_out::write_string(&m, &info, wgsl_out::WriterFlags::empty()).unwrap();

            let minified = minify_wgsl(src, true).unwrap();
            assert_eq!(
                canonical(&reference),
                canonical(&minified),
                "minified output diverged from source compute logic:\n{minified}"
            );
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
        // The binding-preserving fallback still obfuscates: the descriptive name
        // is renamed away, so we kept the rename and only skipped the tree-shake.
        assert!(!out.contains("unused_but_host_binds_it"), "fallback skipped rename:\n{out}");
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

    /// Vector/matrix types collapse to their predeclared WGSL aliases (lossless),
    /// including inside `array<…>`; `array<f32>` (no alias) is left alone.
    #[test]
    fn type_aliases_are_shortened() {
        assert_eq!(shorten_type_aliases("vec4<f32>(1f)"), "vec4f(1f)");
        assert_eq!(shorten_type_aliases("array<vec2<i32>,3>"), "array<vec2i,3>");
        assert_eq!(shorten_type_aliases("mat4x4<f32>"), "mat4x4f");
        assert_eq!(shorten_type_aliases("array<f32>"), "array<f32>");

        // End to end: ALIGN_SHADER's mat4x4<f32> members come out aliased, still
        // valid and layout-preserving.
        let out = minify_wgsl(ALIGN_SHADER, true).unwrap();
        assert!(out.contains("mat4x4f") && !out.contains("mat4x4<f32>"),
            "matrix type not aliased:\n{out}");
        assert!(wgsl::parse_str(&out).is_ok());
    }

    /// Whitespace collapse tightens separators but never touches operator
    /// spacing (so it can't fuse `- -` into `--`, `> >` into `>>`, etc.).
    #[test]
    fn collapse_whitespace_tightens_separators_keeps_operators() {
        assert_eq!(collapse_whitespace("a = b ;\n  c ( d , e ) ;"), "a = b;c(d,e);");
        // operator-adjacent spaces preserved (can't fuse `- -`/`< <`/`> >`):
        assert_eq!(collapse_whitespace("x  -  -y"), "x - -y");
        assert_eq!(collapse_whitespace("a < < b"), "a < < b");
        assert_eq!(collapse_whitespace("a > > b"), "a > > b");
    }

    /// `short_name` is a bijective base-26 counter (`a..z`, `aa..az`, `ba..`, …).
    /// Pin the rollover boundaries; the multi-letter path only fires past 26
    /// identifiers, which no shader test reaches.
    #[test]
    fn short_name_base26_rollover() {
        for (n, want) in [
            (0, "a"),
            (25, "z"),
            (26, "aa"),
            (51, "az"),
            (52, "ba"),
            (701, "zz"),
            (702, "aaa"),
        ] {
            assert_eq!(short_name(n), want, "short_name({n})");
        }
    }

    /// Invalid WGSL is rejected (the parse-failure branch), not silently passed
    /// through as a "minified" string.
    #[test]
    fn invalid_wgsl_is_an_error() {
        assert!(minify_wgsl("this is not wgsl {{{", true).is_err());
        assert!(minify_wgsl("fn m( { ", false).is_err());
    }

    /// Minifying an already-minified shader is a fixed point: it stays valid and
    /// byte-layout-identical (guards against re-inject double-application or any
    /// drift through a second round-trip).
    #[test]
    fn minify_is_idempotent() {
        let want = layouts(ALIGN_SHADER);
        for rename in [false, true] {
            let once = minify_wgsl(ALIGN_SHADER, rename).unwrap();
            let twice = minify_wgsl(&once, rename).unwrap();
            assert_eq!(layouts(&twice), want, "layout drifted on re-minify:\n{twice}");
            assert!(wgsl::parse_str(&twice).is_ok(), "re-minify not valid WGSL:\n{twice}");
        }
    }
}
