use std::path::Path;

use crate::globutils::glob_items;
use crate::naga_parse::{LuaWGSLModule, parse_and_validate_wgsl, parse_wgsl};
use anyhow::Result;
use mlua::{Function, Lua, LuaSerdeExt, Table, UserData};

static LUA_EMBEDS: &str = include_str!(concat!(env!("OUT_DIR"), "/embedded_lua_bundle.lua"));

pub struct LuaLoomInterface {}

impl LuaLoomInterface {
    pub fn new() -> Self {
        Self {}
    }
}

impl UserData for LuaLoomInterface {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("glob", |_, _this: &Self, pattern: String| {
            Ok(glob_items(&pattern)?)
        });

        methods.add_method("parse_wgsl", |_, _this: &Self, src: String| {
            Ok(parse_wgsl(&src)?)
        });

        methods.add_method(
            "minify_wgsl",
            |_, _this: &Self, (src, rename): (String, Option<bool>)| {
                crate::minify::minify_wgsl(&src, rename.unwrap_or(false))
                    .map_err(|e| mlua::Error::RuntimeError(e.to_string()))
            },
        );

        methods.add_method(
            "parse_and_validate_wgsl",
            |_, _this: &Self, (src, flags): (String, Option<u8>)| {
                let (module, info) = parse_and_validate_wgsl(&src, flags);
                let lua_module = module.map(|module| LuaWGSLModule { module });
                Ok((lua_module, info))
            },
        );

        methods.add_method("print", |_, _this: &Self, msg: String| {
            println!("{}", msg);
            Ok(())
        });
    }
}

pub struct LuaExecutor {
    lua: Lua,
}

impl LuaExecutor {
    pub fn new() -> Self {
        // need to create "unsafe" Lua state to have 'debug' Lua library
        let lua = unsafe { Lua::unsafe_new() };
        let globals = lua.globals();

        globals.set("null", lua.null()).unwrap();
        globals.set("loom", LuaLoomInterface::new()).unwrap();
        globals.set("__raw_embed", LUA_EMBEDS).unwrap();

        if let Err(e) = lua.load(LUA_EMBEDS).set_name("=<BUNDLE>").exec() {
            //println!("{}", LUA_EMBEDS);
            let msg = e.to_string();
            panic!(
                "Lua error in '{}': {}",
                concat!(env!("OUT_DIR"), "/embedded_lua_bundle.lua"),
                msg
            );
        };
        Self { lua }
    }

    pub fn run_module(&self, module_name: &str, arg: Option<String>) -> Result<()> {
        let run_module: Function = self.lua.globals().get("_run_module")?;
        run_module.call::<()>((module_name, arg))?;
        Ok(())
    }

    #[cfg(test)]
    pub fn run_tests(&self, module_name: &str) -> Result<()> {
        let run_tests: Function = self.lua.globals().get("_run_tests")?;
        run_tests.call::<()>(module_name)?;
        Ok(())
    }

    pub fn update_config(&self, config: Table) -> Result<()> {
        let update_config: Function = self.lua.globals().get("_update_config")?;
        update_config.call::<()>(config)?;
        Ok(())
    }

    pub fn run_script(&self, infile: impl AsRef<Path>) -> Result<()> {
        let args = self.lua.create_table()?;
        let infile = infile.as_ref();

        if let Some(p) = infile.parent() {
            let mut p = p.to_string_lossy().into_owned();
            if p.is_empty() {
                p = ".".into();
            }
            args.set("SCRIPTDIR", p)?;
        }
        args.set("SCRIPTPATH", infile)?;

        if let Ok(p) = std::path::absolute(infile) {
            if let Some(p) = p.parent() {
                args.set("ABSSCRIPTDIR", p.to_string_lossy())?;
            }
            args.set("ABSSCRIPTPATH", p.to_string_lossy())?;
        };
        self.update_config(args)?;

        self.run_module("cli.exec_script", None)
    }
}
