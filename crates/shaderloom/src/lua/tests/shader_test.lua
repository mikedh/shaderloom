local M = {}

--- Parse "// @test name(args) op expected" lines, associate with following fn
function M.scan_file(filepath, source)
    local tests = {}
    local pending = {}
    local include_file = filepath:match("[^/]+$")

    for line in source:gmatch("[^\n]+") do
        local name, args, op, expected = line:match(
            "^%s*// @test%s+([%w_]+)%((.-)%)%s*([<>=~]+)%s*([%S]+)"
        )
        if name then
            table.insert(pending, {
                name = name,
                args = args,
                op = op,
                expected = tonumber(expected),
                include_file = include_file,
            })
        else
            local fn_name = line:match("^%s*fn%s+([%w_]+)")
            if fn_name and #pending > 0 then
                for _, test in ipairs(pending) do
                    test.fn_name = fn_name
                    table.insert(tests, test)
                end
                pending = {}
            elseif not line:match("^%s*$") and not line:match("^%s*//") then
                pending = {}
            end
        end
    end

    return tests
end

--- Generate a test compute shader source (before preprocessing)
function M.generate_shader(test)
    return ('#include "%s"\n\n'
        .. '@group(0) @binding(0) var<storage, read_write> _result: array<f32>;\n\n'
        .. '@compute @workgroup_size(1)\n'
        .. 'fn cs_main() {\n'
        .. '    _result[0] = %s(%s);\n'
        .. '}\n'):format(test.include_file, test.fn_name, test.args)
end

--- Check assertion
function M.check(result, op, expected)
    if op == "<" then return result < expected
    elseif op == ">" then return result > expected
    elseif op == "<=" then return result <= expected
    elseif op == ">=" then return result >= expected
    elseif op == "==" then return result == expected
    elseif op == "~=" then return math.abs(result - expected) <= 1e-5
    else error("Unknown op: " .. op) end
end

--- Run all tests found in a set of source files
function M.run(filepaths, include_dirs)
    local pp = require("preprocess.preprocessor")
    local fileio = require("utils.fileio")
    local resolver = fileio.create_resolver(include_dirs)

    local tests = {}
    for _, fp in ipairs(filepaths) do
        local src = fileio.read(fp)
        for _, t in ipairs(M.scan_file(fp, src)) do
            table.insert(tests, t)
        end
    end

    local passed, failed = 0, 0
    for _, test in ipairs(tests) do
        local shader_src = M.generate_shader(test)
        local proc = pp.Preprocessor(resolver)
        proc:process_source(shader_src, "test:" .. test.name)
        local wgsl = proc:get_output().source

        local output = string.pack("f", 0.0)
        local results = loom:run_compute(wgsl, "cs_main", {
            {kind = "read_write", data = output},
        }, {1, 1, 1})
        local val = string.unpack("f", results[1])

        if M.check(val, test.op, test.expected) then
            passed = passed + 1
        else
            loom:print(("FAIL %s: %s(%s) = %g, expected %s %g")
                :format(test.name, test.fn_name, test.args, val, test.op, test.expected))
            failed = failed + 1
        end
    end

    loom:print(("%d passed, %d failed"):format(passed, failed))
    assert(failed == 0, failed .. " shader test(s) failed")
end

M._tests = {}

function M._tests.scan_basic()
    local tests = M.scan_file("test.wgsl", table.concat({
        "// @test foo(1.0, 2.0) < 3.0",
        "fn bar(a: f32, b: f32) -> f32 {",
    }, "\n"))
    assert(#tests == 1)
    assert(tests[1].name == "foo")
    assert(tests[1].fn_name == "bar")
    assert(tests[1].args == "1.0, 2.0")
    assert(tests[1].op == "<")
    assert(tests[1].expected == 3.0)
end

function M._tests.scan_multiple_tests_one_fn()
    local tests = M.scan_file("test.wgsl", table.concat({
        "// @test a(1.0) < 2.0",
        "// @test b(3.0) > 1.0",
        "fn baz(x: f32) -> f32 {",
    }, "\n"))
    assert(#tests == 2)
    assert(tests[1].name == "a")
    assert(tests[2].name == "b")
    assert(tests[1].fn_name == "baz")
    assert(tests[2].fn_name == "baz")
end

function M._tests.scan_comment_between()
    local tests = M.scan_file("test.wgsl", table.concat({
        "// @test x(1.0) ~= 1.0",
        "// helper function",
        "fn helper(v: f32) -> f32 {",
    }, "\n"))
    assert(#tests == 1)
    assert(tests[1].name == "x")
    assert(tests[1].fn_name == "helper")
end

function M._tests.scan_clears_on_non_comment()
    local tests = M.scan_file("test.wgsl", table.concat({
        "// @test x(1.0) == 1.0",
        "var<private> foo: f32;",
        "fn other(v: f32) -> f32 {",
    }, "\n"))
    assert(#tests == 0)
end

function M._tests.scan_multiple_fns()
    local tests = M.scan_file("test.wgsl", table.concat({
        "// @test a(1.0) < 2.0",
        "fn first(x: f32) -> f32 {",
        "// @test b(3.0) > 1.0",
        "fn second(x: f32) -> f32 {",
    }, "\n"))
    assert(#tests == 2)
    assert(tests[1].fn_name == "first")
    assert(tests[2].fn_name == "second")
end

function M._tests.generate_basic()
    local test = {
        include_file = "foo.inc.wgsl",
        fn_name = "bar",
        args = "1.0, 2.0",
    }
    local src = M.generate_shader(test)
    assert(src:find('#include "foo.inc.wgsl"'))
    assert(src:find("_result%[0%] = bar%(1.0, 2.0%)"))
    assert(src:find("fn cs_main"))
end

function M._tests.check_ops()
    assert(M.check(1.0, "<", 2.0))
    assert(not M.check(2.0, "<", 1.0))
    assert(M.check(2.0, ">", 1.0))
    assert(M.check(1.0, "<=", 1.0))
    assert(M.check(1.0, ">=", 1.0))
    assert(M.check(1.0, "==", 1.0))
    assert(M.check(1.000005, "~=", 1.0))
    assert(not M.check(1.1, "~=", 1.0))
end

function M._tests.scan_include_file_from_path()
    local tests = M.scan_file("/some/path/to/trimath.inc.wgsl", table.concat({
        "// @test t(1.0) < 2.0",
        "fn f(x: f32) -> f32 {",
    }, "\n"))
    assert(tests[1].include_file == "trimath.inc.wgsl")
end

return M
