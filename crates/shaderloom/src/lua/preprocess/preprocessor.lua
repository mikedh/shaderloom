local class = require "miniclass"
local utils = require "utils.common"
local chunker = require "preprocess.chunker"

local Preprocessor = class "Preprocessor"

function Preprocessor:init(resolver)
    self.resolver = resolver
    self:clear()
end

function Preprocessor:emit_raw(src)
    src = tostring(src)
    table.insert(self.frags, src)
    self.annotation_cursor = self.annotation_cursor + #src
end

function Preprocessor:emit(src)
    self:emit_raw(src)
    --self:process_source(src) -- not sure if recursing here is a good idea
end

function Preprocessor:annotate(annotator, args)
    table.insert(self.annotations, {
        eval=annotator,
        pos=self.annotation_cursor,
        args=args
    })
end

local function identity_annotation(tab)
    return tab.category, tab.name, tab.payload
end
function Preprocessor:_pre_annotate(category, name, payload)
    table.insert(self.annotations, {
        category = category,
        name = name,
        payload = payload,
        eval = identity_annotation
    })
end

---match one or more patterns agains the next semicolon-delimited
---statement in the source
---@param source string
---@param pos number
---@param patterns string | string[]
---@return string | nil
local function match_in_next_statement(source, pos, patterns)
    if type(patterns) == 'string' then
        patterns = {patterns}
    end
    local end_pos = source:find(";", pos)
    local statement = source:sub(pos, end_pos)
    for _, patt in ipairs(patterns) do
        local match = statement:match(patt)
        if match then return match end
    end
end

-- captures name from e.g., "var tex_whatever: texture_multisampled_2d<f32>;"
local VISIBILITY_PATTERNS = {
    "var%s+([^%s:]*)%s*:",
    "var%s*%b<>%s*([^%s:]*)%s*:"
}

local function _annotate_visibility(call_info, source)
    local var_name = assert(
        match_in_next_statement(source, call_info.pos, VISIBILITY_PATTERNS),
        "Unmatched visibility annotation"
    )
    return "visibility", var_name, call_info.args
end

function Preprocessor:annotate_visibility(...)
    -- handle calling both as 
    -- visibility("fragment", "vertex") and
    -- visibility{"fragment", "vertex"}
    local args = {...}
    if #args == 1 and type(args[1]) == 'table' then
        args = args[1]
    end
    self:annotate(_annotate_visibility, utils.set(args))
end

local BINDGROUP_PATTERN = "@group%(([^%(]*)%)"
local function _annotate_bindgroup(call_info, source)
    local group_id_str = assert(
        match_in_next_statement(source, call_info.pos, BINDGROUP_PATTERN),
        "Unmatched bindgroup @group() annotation"
    )
    local group_id = assert(
        tonumber(group_id_str),
        ("group id '%s' is not a number!"):format(group_id_str)
    )
    return "bindgroups", group_id, call_info.args
end

function Preprocessor:annotate_bindgroup(args)
    if type(args) == 'string' then
        args = {name = args}
    end
    self:annotate(_annotate_bindgroup, args or {})
end

function Preprocessor:annotate_name(name)
    self:_pre_annotate("name", nil, name)
end

function Preprocessor:include(name)
    self:process_source(self.resolver(name), name)
end

function Preprocessor:_bind(name)
    local func = assert(self[name], "Missing bind! " .. name)
    return function(...)
        return func(self, ...)
    end
end

function Preprocessor:clear()
    self.frags = {}
    self.annotation_cursor = 1
    self.annotations = {}
    self.env = {
        emit = self:_bind("emit"),
        emit_raw = self:_bind("emit_raw"),
        include = self:_bind("include"),
        visibility = self:_bind("annotate_visibility"),
        bindgroup = self:_bind("annotate_bindgroup"),
        name = self:_bind("annotate_name"),
    }
    setmetatable(self.env, {
        __index = _G
    })
end

function Preprocessor:process_source(source, name)
    local translated = chunker.translate_source(source)
    local chunk, err = loadstring_env(translated, name, self.env)
    if not chunk then
        print("BAD PREPROC:")
        print(translated)
        error("Preprocessor error in " .. name .. ": " .. tostring(err))
    end
    chunk()
end

---@class Visibility
---@field vertex boolean?
---@field fragment boolean?
---@field compute boolean?

---@class BindgroupInfo
---@field id number
---@field name string?
---@field shared boolean?

---@class Annotations
---@field visibility table<string, Visibility>
---@field name string?
---@field bindgroups table<number, BindgroupInfo>

---@class PreprocessorOutput
---@field source string
---@field annotations Annotations

---Get the preprocessed output
---@return PreprocessorOutput
function Preprocessor:get_output()
    local output = table.concat(self.frags, "")
    local annotations = {visibility={}, bindgroups={}}
    for _, annotator in ipairs(self.annotations) do
        local category, name, annotation = annotator:eval(output)
        if category and name then
            annotations[category][name] = annotation
        elseif category then
            annotations[category] = annotation
        end
    end
    return {source=output, annotations=annotations}
end

local tests = {}

local function test_proc(files)
    local resolver = function(name)
        return assert(files[name], "Missing " .. name)
    end
    local pp = Preprocessor(resolver)
    pp:include("MAIN")
    return pp:get_output()
end

function tests.chunker_braces()
    local dedent = require("utils.stringmanip").dedent
    local eq = require("utils.deepeq").string_equal
    local source = "return a[b]]\n"
    local translated = chunker.translate_source(source)
    assert(eq(translated, "emit_raw[=[return a[b]]\n]=]"))
end

function tests.identity_translation()
    local dedent = require("utils.stringmanip").dedent
    local eq = require("utils.deepeq").string_equal
    local files = {
        MAIN=dedent[[
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            v_indices.data[global_id.x] = collatz_iterations(v_indices.data[global_id.x]);
        }
        ]]
    }
    local translated = test_proc(files).source
    assert(eq(files.MAIN, translated))
end

function tests.only_preprocessor()
    local dedent = require("utils.stringmanip").dedent
    local eq = require("utils.deepeq").string_equal
    local files = {
        MAIN=dedent[[
        # 
        # 
        # emit_raw "asdf"
        # thing = 12
        # ]]
    }
    local expected = "asdf"
    local translated = test_proc(files).source
    assert(eq(expected, translated))
end


function tests.double_emit_regression()
    local dedent = require("utils.stringmanip").dedent
    local eq = require("utils.deepeq").string_equal
    local files = {
        MAIN=dedent[[
        // Foo

        # emit_raw "bar"
        # ]]
    }
    local expected = "// Foo\nbar"
    local translated = test_proc(files).source
    assert(eq(expected, translated))
end

function tests.inline_translation()
    local dedent = require("utils.stringmanip").dedent
    local eq = require("utils.deepeq").string_equal
    local files = {
        MAIN=dedent[[
        # function one() 
        #   return 1
        # end
        @compute @workgroup_size(${one()})
        #-- a preprocessor comment
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            v_indices.data[global_id.x] = collatz_iterations(v_indices.data[global_id.x]);
        }
        ]]
    }
    local expected = dedent[[
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        v_indices.data[global_id.x] = collatz_iterations(v_indices.data[global_id.x]);
    }
    ]]
    local translated = test_proc(files).source
    assert(eq(expected, translated))
end

function tests.includes()
    local dedent = require("utils.stringmanip").dedent
    local eq = require("utils.deepeq").string_equal
    local files = {
        MAIN=dedent[[
        #include "other"
        fn eh() {
        }
        ]],
        other=dedent[[
        @compute @workgroup_size(1)
        ]]
    }
    local expected = dedent[[
    @compute @workgroup_size(1)
    fn eh() {
    }
    ]]
    local translated = test_proc(files).source
    assert(eq(expected, translated))
end

function tests.visibility_annotation()
    local deq = require("utils.deepeq").dict_exact_equal
    local seq = require("utils.deepeq").string_equal
    assert(seq(
        ("var < workgroup > foo : u32"):match(VISIBILITY_PATTERNS[2], 1),
        "foo"
    ))
    assert(seq(
        ("var tex_whatever: texture_2d<f32>;"):match(VISIBILITY_PATTERNS[1], 1),
        "tex_whatever"
    ))

    local dedent = require("utils.stringmanip").dedent
    local files = {
        MAIN=dedent[[
        # visibility "fragment"
        var < workgroup > foo : u32;
        # visibility("fragment", "vertex")
        var tex_whatever: texture_2d<f32>;
        # visibility{"vertex"}
        @binding(0) @group(12)
        var<storage, read_write > v_ehhhh_32 : array<f32>;
        ]],
    }
    local annotations = test_proc(files).annotations
    assert(deq(annotations.visibility.foo, {fragment=true}))
    assert(deq(annotations.visibility.tex_whatever, {fragment=true, vertex=true}))
    assert(deq(annotations.visibility.v_ehhhh_32, {vertex=true}))
end

function tests.bindgroup_annotation()
    local deq = require("utils.deepeq").dict_exact_equal
    local seq = require("utils.deepeq").string_equal

    local dedent = require("utils.stringmanip").dedent
    local files = {
        MAIN=dedent[[
        # bindgroup "foobar"
        @group(0) @binding(0) 
        var<storage, read_write > v_ehhhh_32 : array<f32>;
        # bindgroup "uniforms"
        # FOOBAR = 7;
        @group(${FOOBAR}) @binding(3) 
        var<storage, read_write > v_ehhhh_32 : array<f32>;
        ]],
    }
    local proc = test_proc(files)
    local translated, annotations = proc.source, proc.annotations
    assert(deq(annotations.bindgroups[7], {name="uniforms"}))
    assert(deq(annotations.bindgroups[0], {name="foobar"}))
end

function tests.name_annotation()
    local seq = require("utils.deepeq").string_equal

    local dedent = require("utils.stringmanip").dedent
    local files = {
        MAIN=dedent[[
        # name "some_shader"
        fn main() {}
        ]],
    }
    local annotations = test_proc(files).annotations
    assert(seq(annotations.name, "some_shader"))
end

return {
    Preprocessor = Preprocessor,
    _tests = tests
}