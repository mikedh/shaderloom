-- Splits shader source with embedded preprocessor parts
-- into chunks that are either verbatim source or Lua preprocessor code.

local chunker = {}

--- Embeds inner string into a Lua raw/multiline literal.
--- 
--- E.g., if inner is `foo`, produces `[[foo]]`. 
--- Handles the case where inner includes multiline brackets:
--- E.g., `foo[[bar]]` is embedded as `[=[foo[[bar]]]=]`
--- 
--- @param inner string
--- @return string
local function embed_lua_multiline(inner)
    -- count the maximum number of sequential `]`s seen
    -- in the inner string
    local max_closing = 0
    for run in inner:gmatch("%]+") do
        max_closing = math.max(max_closing, #run)
    end
    local open, close = "[[", "]]"
    if max_closing >= 2 then
        local req_nest = math.max(0, max_closing - 1)
        open = "[" .. string.rep("=", req_nest) .. "["
        close = "]" .. string.rep("=", req_nest) .. "]"
    end
    return open .. inner .. close
end

local function emit_raw(frags, src)
    if src:sub(1,1) == "\n" then
        -- quirk with first newline of multiline strings being ignored
        src = "\n" .. src
    end
    table.insert(frags, ("emit_raw%s"):format(embed_lua_multiline(src)))
end

local function emit(frags, src)
    table.insert(frags, ("emit(%s)"):format(src))
end

local function handle_source(src, frags)
    local cursor = 1
    local src_len = #src
    while cursor <= src_len do
        local start = cursor
        local macro_start, macro_end, macro_expr = src:find("$(%b{})", cursor)
        if macro_start then
            if macro_start > start then
                -- emit the source up to the start of the macro
                emit_raw(frags, src:sub(start, macro_start-1))
            end
            -- strip {} surrounding macro expression
            emit(frags, macro_expr:sub(2, #macro_expr-1))
            cursor = macro_end + 1
        else -- no interpolations, emit remainder of source    
            local subssrc = src:sub(start, src_len)
            emit_raw(frags, subssrc)
            break
        end
    end
end

---
--- Translate source into a runnable Lua script
---
---@param src string
---@return string
function chunker.translate_source(src)
    local cursor = 1
    -- because we're operating on a line-by-line basis we
    -- need the source to end in a newline.
    if src:sub(#src, #src) ~= "\n" then src = src .. "\n" end
    local src_len = #src

    local frags = {}

    while cursor <= src_len do
        local match_start, match_end, pre_line = src:find("^%s*#([^\n]*)\n", cursor)
        if match_start then
            -- this is a preprocessor line `# ...`
            -- preprocessor lines are inserted verbatim
            table.insert(frags, pre_line)
            cursor = match_end + 1
        else
            -- not a preprocessor line, so find the next preprocessor line
            local start = cursor
            cursor = (src:find("\n%s*#", cursor) or src_len)
            local shader_src = src:sub(start, cursor)
            handle_source(shader_src, frags)
            cursor = cursor + 1
        end
    end

    return table.concat(frags, "\n")
end

return chunker
