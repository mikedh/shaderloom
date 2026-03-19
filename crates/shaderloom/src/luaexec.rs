use std::path::Path;

use crate::globutils::glob_items;
use crate::gpu_exec;
use crate::naga_parse::{LuaWGSLModule, parse_and_validate_wgsl, parse_wgsl};
use anyhow::Result;
use mlua::{Function, Lua, LuaSerdeExt, Table, UserData};

static LUA_EMBEDS: &str = include_str!(concat!(env!("OUT_DIR"), "/embedded_lua_bundle.lua"));

#[derive(Default)]
pub struct LuaLoomInterface {}

impl UserData for LuaLoomInterface {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("glob", |_, _this: &Self, pattern: String| {
            Ok(glob_items(&pattern)?)
        });

        methods.add_method("parse_wgsl", |_, _this: &Self, src: String| {
            Ok(parse_wgsl(&src)?)
        });

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

        methods.add_method(
            "run_compute",
            |lua,
             _this,
             (source, entry_point, bindings_table, workgroups_table): (
                String,
                String,
                Table,
                Table,
            )| {
                let (device, queue) = gpu_exec::create_gpu()?;

                let wg: [u32; 3] = [
                    workgroups_table.get(1)?,
                    workgroups_table.get(2)?,
                    workgroups_table.get(3)?,
                ];

                let mut all_data: Vec<Vec<u8>> = Vec::new();
                let mut kinds: Vec<String> = Vec::new();
                for entry in bindings_table.sequence_values::<Table>() {
                    let entry = entry?;
                    let kind: String = entry.get("kind")?;
                    let data: mlua::String = entry.get("data")?;
                    all_data.push(data.as_bytes().to_vec());
                    kinds.push(kind);
                }

                let mut bindings = Vec::new();
                for (data, kind) in all_data.iter().zip(kinds.iter()) {
                    bindings.push(match kind.as_str() {
                        "uniform" => gpu_exec::Binding::Uniform(data),
                        "read" => gpu_exec::Binding::StorageRead(data),
                        "read_write" => gpu_exec::Binding::StorageReadWrite(data),
                        other => {
                            return Err(mlua::Error::RuntimeError(format!(
                                "Unknown binding kind: {other}"
                            )))
                        }
                    });
                }

                let results =
                    gpu_exec::run_compute(&device, &queue, &source, &entry_point, &bindings, wg)?;

                let result_table = lua.create_table()?;
                for (i, data) in results.iter().enumerate() {
                    result_table.set(i + 1, lua.create_string(data)?)?;
                }
                Ok(result_table)
            },
        );
    }
}

pub struct LuaExecutor {
    lua: Lua,
}

impl Default for LuaExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl LuaExecutor {
    pub fn new() -> Self {
        // need to create "unsafe" Lua state to have 'debug' Lua library
        let lua = unsafe { Lua::unsafe_new() };
        let globals = lua.globals();

        globals.set("null", lua.null()).unwrap();
        globals.set("loom", LuaLoomInterface::default()).unwrap();
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
