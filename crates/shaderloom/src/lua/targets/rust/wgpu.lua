-- targets.rust.wgpu
--
--

local utils = require "utils.common"

local m = {}

local RUST_SCALARS = {
    f32 = "f32",
    f16 = "f16",
    u32 = "u32",
    i32 = "i32",
    bool = "u8", -- ??
}

function m.rust_typename(ty)
    local kind = ty.kind
    if kind == "scalar" then
        return RUST_SCALARS[ty.name]
    elseif kind == "vector" or kind == "array" then
        local count = assert(ty.count, "array dtypes must have fixed size!")
        return ("[%s; %d]"):format(m.rust_typename(ty.inner), count)
    elseif kind == "matrix" then
        return ("[%s; %d]"):format(
            m.rust_typename(ty.inner),
            ty.rows * ty.cols
        )
    elseif kind == "atomic" then
        return m.rust_typename(ty.inner)
    elseif kind == "struct" then
        return ty.name
    else
        error("Could not create dtype for " .. kind)
    end
end

local STRUCT_TEMPLATE = [[
${HEADER}
${VISIBILITY} struct ${NAME} {
${FIELDS}
}
]]

local STRUCT_FILE_TEMPLATE = [[
use bytemuck::{Pod, Zeroable};

${STRUCTS}
${CONSTS}]]

---@class RustStructMember
---@field name string
---@field visibility string?
---@field comment string?
---@field derive string?
---@field ty TypeDef?
---@field tyname string
---@field is_padding boolean

---@class RustStruct
---@field name string
---@field derive string?
---@field comment string?
---@field repr string?
---@field visibility string?
---@field header string[]?
---@field fields RustStructMember[]

---@param rstruct RustStruct
---@param template string
---@return string
function m.format_rust_struct(rstruct, template)
    local field_frags = {}
    for _, field in ipairs(rstruct.fields) do
        if field.comment then
            table.insert(field_frags, "    " .. field.comment)
        end
        if field.derive then
            table.insert(field_frags, "    " .. field.derive)
        end
        local vis = field.visibility or "pub"
        table.insert(field_frags, ("    %s %s: %s,"):format(vis, field.name, field.tyname))
    end
    local header = rstruct.header or {}
    if rstruct.comment then
        table.insert(header, rstruct.comment)
    end
    if rstruct.derive then
        table.insert(header, rstruct.derive)
    end
    if rstruct.repr then
        table.insert(header, rstruct.repr)
    end
    return template:with{
        HEADER = table.concat(header, "\n"),
        VISIBILITY = rstruct.visibility or "pub",
        NAME = rstruct.name,
        FIELDS = table.concat(field_frags, "\n"),
    }
end

function m.prepare_struct(options, ty)
    ---@type RustStructMember[]
    local fields = {}
    local function add_field(field)
        if options.field_decorator then
            field = options.field_decorator(field, ty) or field
        end
        table.insert(fields, field)
    end

    local pad_id = 0
    local cur_offset = 0
    local function pad_to(target)
        local npad = target - cur_offset
        if npad <= 0 then return end
        cur_offset = target
        add_field{
            name=("_pad_%s"):format(pad_id),
            tyname=("[u8; %d]"):format(npad),
            is_padding=true
        }
        pad_id = pad_id + 1
    end

    for _, member in ipairs(ty.members) do
        local tysize = member.ty.size
        local offset = member.offset
        pad_to(offset)
        add_field{
            name=member.name,
            ty=member.ty,
            tyname=m.rust_typename(member.ty)
        }
        cur_offset = offset + tysize
    end
    pad_to(ty.size)
    local rstruct = {
        name=ty.name,
        ty=ty,
        derive="#[derive(Copy, Clone, Pod, Zeroable)]",
        repr="#[repr(C)]",
        fields=fields
    }
    if options.struct_decorator then
        rstruct = options.struct_decorator(rstruct) or rstruct
    end
    return rstruct
end

-- Render a single module-scope `const` (identified by handle) as a Rust literal.
-- Only literal initializers are supported (naga const-folds derived consts to a
-- literal, so e.g. `MAXOPS + 2u` arrives already folded); anything else is skipped.
function m.const_literal(data, init_handle, ty)
    -- naga handles are 0-based; Lua arrays are 1-based.
    local expr = data.global_expressions and data.global_expressions[init_handle + 1]
    local lit = expr and expr.Literal
    if not lit then return nil end
    local _, value = next(lit)
    if ty.name == "f32" or ty.name == "f16" then
        local s = tostring(value)
        if not s:find("[.eE]") then s = s .. ".0" end -- force a float literal
        return s
    elseif ty.name == "bool" then
        return tostring(value)
    else
        return ("%d"):format(value)
    end
end

-- Collect the `# export()`-annotated module-scope consts across all shaders,
-- deduped by name (conflicting definitions of the same name are an error).
function m.gather_consts(shaders, parsed)
    local log = require "log"
    local by_name, order = {}, {}
    for idx, shader in ipairs(shaders) do
        local exports = shader.annotations and shader.annotations.exports
        local pdef = parsed[idx]
        if exports and next(exports) and pdef and pdef.raw.constants then
            for _, c in ipairs(pdef.raw.constants) do
                if exports[c.name] then
                    local ty = assert(pdef.types[c.ty],
                        ("exported const '%s' has unknown type"):format(c.name))
                    local tyname = m.rust_typename(ty)
                    -- A `const` is a value, not a memory-mapped struct field, so the
                    -- Pod `bool -> u8` mapping doesn't apply; emit a real Rust bool
                    -- to match the `true`/`false` literal from `const_literal`.
                    if ty.kind == "scalar" and ty.name == "bool" then tyname = "bool" end
                    local value = m.const_literal(pdef.raw, c.init, ty)
                    if not value then
                        log.warn(("skipping export const '%s' (non-literal initializer)")
                            :format(c.name))
                    elseif by_name[c.name] then
                        assert(by_name[c.name].value == value and by_name[c.name].tyname == tyname,
                            ("exported const '%s' has conflicting definitions"):format(c.name))
                    else
                        by_name[c.name] = {name = c.name, tyname = tyname, value = value}
                        table.insert(order, c.name)
                    end
                end
            end
        end
    end
    local out = {}
    for _, name in ipairs(order) do table.insert(out, by_name[name]) end
    return out
end

function m.write_struct_defs(options, structs, consts, env)
    if type(options) == 'string' then
        options = {output = options}
    end

    local fileio = require "utils.fileio"
    local frags = {}
    for _, struct in ipairs(structs.structs) do
        local rstruct = m.prepare_struct(options, struct)
        local formatted = m.format_rust_struct(rstruct, options.struct_template or STRUCT_TEMPLATE)
        table.insert(frags, formatted)
        if options.struct_impl then
            local impl = assert(options.struct_impl(rstruct, struct), "struct_impl must return a string!")
            table.insert(frags, impl)
        end
    end
    table.sort(frags)
    local struct_str = table.concat(frags, "\n")

    -- `pub const` block for the `# export()`-annotated shader consts, sorted for
    -- determinism. Rendered into the `${CONSTS}` slot of the default template
    -- (after the structs); custom templates may include `${CONSTS}` to place it.
    -- When there are no exported consts the slot is empty so no blank lines leak.
    local const_frags = {}
    for _, c in ipairs(consts or {}) do
        table.insert(const_frags, ("pub const %s: %s = %s;"):format(c.name, c.tyname, c.value))
    end
    local const_str = ""
    if #const_frags > 0 then
        table.sort(const_frags)
        const_str = "\n" .. table.concat(const_frags, "\n") .. "\n"
    end

    local body = (options.file_template or STRUCT_FILE_TEMPLATE)
        :with{STRUCTS = struct_str, CONSTS = const_str}

    fileio.write(options.output:with(env), body)
end

function m.build(options)
    local raw = require "targets.raw"
    local unify = require "analysis.unify"
    local shaders = raw.preprocess(options.shaders, options.include_dirs)
    local config = options.config
    local parsed
    if config.validate or config.struct_definitions then
        parsed = raw.validate(shaders)
    end
    if config.struct_definitions then
        local structs = unify.unify_host_shared_structs(parsed)
        local consts = m.gather_consts(shaders, parsed)
        m.write_struct_defs(config.struct_definitions, structs, consts, options.env)
    end
    if config.bundle then
        raw.emit_bundle(config.bundle, shaders, options.env)
    end
    if config.loose_files then
        raw.emit_loose_shaders(config.loose_files, shaders, options.env)
    end
end

local tests = {}
m._tests = tests

-- Build a {name -> const} index from a gather_consts result for easy assertions.
local function index_consts(consts)
    local by = {}
    for _, c in ipairs(consts) do by[c.name] = c end
    return by
end

-- Exercises the `# export()` const path end-to-end at the analysis layer:
-- gather_consts -> const_literal -> rust_typename, across literal kinds.
function tests.export_consts()
    local naga = require "analysis.naga"
    local src = [[
    const MAXOPS: i32 = 254;
    const NEG: i32 = -5;
    const GROUPSIZE: u32 = 8u;
    const COUNT: u32 = GROUPSIZE + 2u;
    const SCALE: f32 = 1.5;
    const WHOLE: f32 = 2.0;
    const FLAG: bool = true;
    const NOT_EXPORTED: u32 = 3u;

    @compute @workgroup_size(GROUPSIZE)
    fn cs_main() {}
    ]]
    local parsed, errs = naga.parse(src, true)
    assert(not errs, errs)
    local exports = {MAXOPS = true, NEG = true, COUNT = true, SCALE = true, WHOLE = true, FLAG = true}
    local by = index_consts(m.gather_consts(
        {{annotations = {exports = exports}}}, {parsed}))

    assert(by.MAXOPS and by.MAXOPS.tyname == "i32" and by.MAXOPS.value == "254",
        "integer const")
    assert(by.NEG and by.NEG.tyname == "i32" and by.NEG.value == "-5",
        "negative integer const")
    assert(by.COUNT and by.COUNT.tyname == "u32" and by.COUNT.value == "10",
        "derived const folded to a literal")
    assert(by.SCALE and by.SCALE.tyname == "f32" and by.SCALE.value == "1.5",
        "f32 const")
    assert(by.WHOLE and by.WHOLE.tyname == "f32" and by.WHOLE.value == "2.0",
        "whole-number f32 forced to a float literal")
    -- regression: a bool const must emit Rust `bool`/`true`, not the Pod `u8`.
    assert(by.FLAG and by.FLAG.tyname == "bool" and by.FLAG.value == "true",
        "bool const emits a Rust bool")
    assert(by.NOT_EXPORTED == nil, "unmarked const is not gathered")
end

-- Identical same-named consts across shaders dedup to one; conflicting ones error.
function tests.export_consts_dedup_and_conflict()
    local naga = require "analysis.naga"
    local function parse(src)
        local parsed, errs = naga.parse(src, true)
        assert(not errs, errs)
        return parsed
    end
    local eight_a = parse("const N: u32 = 8u;")
    local eight_b = parse("const N: u32 = 8u;")
    local nine = parse("const N: u32 = 9u;")
    local exp = {{annotations = {exports = {N = true}}}, {annotations = {exports = {N = true}}}}

    local consts = m.gather_consts(exp, {eight_a, eight_b})
    assert(#consts == 1 and consts[1].value == "8", "identical consts dedup to one")

    assert(not pcall(m.gather_consts, exp, {eight_a, nine}),
        "conflicting definitions of an exported const must error")
end

-- regression: the default struct-file template must actually render exported
-- consts (the `${CONSTS}` slot), and leave nothing behind when there are none.
function tests.default_template_renders_consts()
    local with_const = STRUCT_FILE_TEMPLATE:with{
        STRUCTS = "// structs", CONSTS = "\npub const A: u32 = 1u;\n"}
    assert(with_const:find("pub const A: u32 = 1u;", 1, true),
        "CONSTS slot is rendered into the default template")
    local empty = STRUCT_FILE_TEMPLATE:with{STRUCTS = "// structs", CONSTS = ""}
    assert(not empty:find("${CONSTS}", 1, true),
        "no leftover ${CONSTS} placeholder when there are no consts")
end

-- A const whose initializer isn't a literal (e.g. a composite) is skipped, not fatal.
function tests.export_const_non_literal_skipped()
    local naga = require "analysis.naga"
    local src = [[
    const PAIR: vec2<u32> = vec2<u32>(1u, 2u);
    @compute @workgroup_size(1) fn cs() {}
    ]]
    local parsed, errs = naga.parse(src, true)
    assert(not errs, errs)
    local consts = m.gather_consts({{annotations = {exports = {PAIR = true}}}}, {parsed})
    assert(#consts == 0, "non-literal (composite) const is skipped")
end

return m