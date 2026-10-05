//! Sandboxed Lua execution. The core standard library is loaded, while file,
//! package, and process APIs are absent. Scripts are stepped with a finite
//! instruction budget so a runaway loop yields control back to the engine.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use glam::{Quat, Vec3};
use piccolo::{
    Callback, CallbackReturn, Closure, Context, Executor, Fuel, IntoValue, Lua, StaticError, Table,
    Value,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    AlphaMode3d, Camera3d, CameraProjection3d, Collider3d, ColliderShape, Engine, EntityId,
    GltfAssetData, InputState, Lighting3d, Material3d, MeshData, MeshKind, PointLight3d,
    RigidBody3d, RigidBodyType, TextureAssetId, Transform,
};

const MAX_LUA_OBJ_BYTES: usize = 256 * 1024;
const MAX_LUA_GLTF_BASE64_BYTES: usize = 1_500_000;
const MAX_LUA_OBJ_LOADS_PER_RUN: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct ScriptLimits {
    pub max_fuel_per_run: i32,
    pub max_memory_bytes: usize,
    pub fuel_per_step: i32,
    pub max_source_bytes: usize,
}

impl Default for ScriptLimits {
    fn default() -> Self {
        Self {
            max_fuel_per_run: 250_000,
            max_memory_bytes: 16 * 1024 * 1024,
            fuel_per_step: 4_096,
            max_source_bytes: 1024 * 1024,
        }
    }
}

#[derive(Debug, Error)]
pub enum ScriptError {
    #[error("Lua execution failed: {0}")]
    Lua(#[from] StaticError),
    #[error("script exceeded its instruction budget of {0}")]
    FuelExhausted(i32),
    #[error("script memory limit exceeded: {used} bytes used, limit is {limit}")]
    MemoryLimit { used: usize, limit: usize },
    #[error("script source is {size} bytes, limit is {limit}")]
    SourceTooLarge { size: usize, limit: usize },
    #[error("script limits must be positive")]
    InvalidLimits,
    #[error("engine API operation failed: {0}")]
    Engine(String),
}

pub struct ScriptRuntime {
    lua: Lua,
    limits: ScriptLimits,
}

impl ScriptRuntime {
    pub fn new(limits: ScriptLimits) -> Result<Self, ScriptError> {
        if limits.max_fuel_per_run <= 0
            || limits.fuel_per_step <= 0
            || limits.max_memory_bytes == 0
            || limits.max_source_bytes == 0
        {
            return Err(ScriptError::InvalidLimits);
        }
        Ok(Self {
            lua: Lua::core(),
            limits,
        })
    }

    pub fn default_sandbox() -> Self {
        Self::new(ScriptLimits::default()).expect("default Lua limits are valid")
    }

    /// Execute a source chunk using the existing Lua state. Each call has its
    /// own fuel budget, while globals persist across calls.
    pub fn run(&mut self, name: &str, source: &str) -> Result<(), ScriptError> {
        self.run_inner(name, source, None)
    }

    /// Run a bounded Lua chunk with the `engine` table bound to this scene.
    /// Lua entity mutations are committed through the normal engine methods
    /// after the chunk yields, including when the chunk raises a Lua error.
    pub fn run_with_engine(
        &mut self,
        engine: &mut Engine,
        name: &str,
        source: &str,
    ) -> Result<(), ScriptError> {
        self.run_with_engine_context(engine, None, 0.0, name, source)
    }

    /// Run a chunk with the current entity and fixed-step delta available as
    /// `engine.self()` and `engine.delta_time()`.
    pub fn run_with_engine_context(
        &mut self,
        engine: &mut Engine,
        owner: Option<EntityId>,
        delta_seconds: f64,
        name: &str,
        source: &str,
    ) -> Result<(), ScriptError> {
        let bridge = Rc::new(RefCell::new(EngineBridge::from_engine(
            engine,
            owner,
            delta_seconds,
        )));
        let result = self.run_inner(name, source, Some(Rc::clone(&bridge)));
        apply_engine_commands(engine, bridge)?;
        result
    }

    fn run_inner(
        &mut self,
        name: &str,
        source: &str,
        bridge: Option<Rc<RefCell<EngineBridge>>>,
    ) -> Result<(), ScriptError> {
        if source.len() > self.limits.max_source_bytes {
            return Err(ScriptError::SourceTooLarge {
                size: source.len(),
                limit: self.limits.max_source_bytes,
            });
        }
        let executor = self.lua.try_enter(|ctx| {
            if let Some(bridge) = bridge {
                install_engine_api(ctx, bridge);
            } else {
                ctx.set_global("engine", Value::Nil)?;
            }
            let closure = Closure::load(ctx, Some(name), source.as_bytes())?;
            Ok(ctx.stash(Executor::start(ctx, closure.into(), ())))
        })?;

        let mut remaining = self.limits.max_fuel_per_run;
        while remaining > 0 {
            let step_budget = remaining.min(self.limits.fuel_per_step);
            let mut fuel = Fuel::with(step_budget);
            let complete = self
                .lua
                .enter(|ctx| ctx.fetch(&executor).step(ctx, &mut fuel));
            remaining -= step_budget.saturating_sub(fuel.remaining().max(0)).max(1);

            let memory_used = self.lua.total_memory();
            if memory_used > self.limits.max_memory_bytes {
                return Err(ScriptError::MemoryLimit {
                    used: memory_used,
                    limit: self.limits.max_memory_bytes,
                });
            }

            if complete {
                self.lua.execute::<()>(&executor)?;
                return Ok(());
            }
        }

        Err(ScriptError::FuelExhausted(self.limits.max_fuel_per_run))
    }

    pub fn memory_used(&self) -> usize {
        self.lua.total_memory()
    }

    pub fn collect_garbage(&mut self) {
        self.lua.gc_collect();
    }
}

#[derive(Clone)]
struct ShadowEntity {
    id: i64,
    name: String,
    transform: Transform,
    material: Option<Material3d>,
    rigid_body: Option<RigidBody3d>,
    collider: Option<Collider3d>,
    parent_id: Option<i64>,
    base_color_texture: Option<u64>,
    visible: bool,
    point_light: Option<PointLight3d>,
}

enum EngineCommand {
    Spawn {
        temporary_id: i64,
        name: String,
        position: [f32; 3],
        mesh: MeshKind,
    },
    SpawnObj {
        temporary_id: i64,
        name: String,
        position: [f32; 3],
        mesh: MeshData,
    },
    SpawnGltf {
        temporary_id: i64,
        name: String,
        position: [f32; 3],
        asset: GltfAssetData,
    },
    PlayGltfAnimation {
        id: i64,
        clip_index: usize,
        looping: bool,
    },
    StopGltfAnimation {
        id: i64,
    },
    SetGltfAnimationSpeed {
        id: i64,
        speed: f32,
    },
    SetPosition {
        id: i64,
        position: [f32; 3],
    },
    SetRotation {
        id: i64,
        rotation: [f32; 4],
    },
    SetScale {
        id: i64,
        scale: [f32; 3],
    },
    SetMaterial {
        id: i64,
        material: Material3d,
    },
    SetBaseColorTexture {
        id: i64,
        texture_id: Option<u64>,
    },
    SetVisibility {
        id: i64,
        visible: bool,
    },
    SetPointLight {
        id: i64,
        light: Option<PointLight3d>,
    },
    SetParent {
        id: i64,
        parent_id: Option<i64>,
    },
    AddRigidBody {
        id: i64,
        body: RigidBody3d,
    },
    RemoveRigidBody {
        id: i64,
    },
    SetVelocity {
        id: i64,
        velocity: Vec3,
    },
    ApplyForce {
        id: i64,
        force: Vec3,
    },
    SetCollider {
        id: i64,
        collider: Collider3d,
    },
    RemoveCollider {
        id: i64,
    },
    SetCamera(Camera3d),
    SetLighting(Lighting3d),
    SetGravity(Vec3),
    Delete {
        id: i64,
    },
}

struct EngineBridge {
    entities: Vec<ShadowEntity>,
    commands: Vec<EngineCommand>,
    next_temporary_id: i64,
    pending_obj_loads: usize,
    owner_id: Option<i64>,
    delta_seconds: f64,
    camera: Camera3d,
    gravity: Vec3,
    input: InputState,
    lighting: Lighting3d,
}

impl EngineBridge {
    fn from_engine(engine: &Engine, owner: Option<EntityId>, delta_seconds: f64) -> Self {
        let entities = engine
            .entities()
            .filter_map(|(id, name, transform)| {
                let material = engine.material(id).ok();
                Some(ShadowEntity {
                    id: i64::try_from(id.get()).ok()?,
                    name,
                    transform,
                    material,
                    rigid_body: engine.rigid_body(id).ok().flatten(),
                    collider: engine.collider(id).ok().flatten(),
                    parent_id: engine
                        .parent(id)
                        .ok()
                        .flatten()
                        .and_then(|parent| i64::try_from(parent.get()).ok()),
                    base_color_texture: engine
                        .base_color_texture(id)
                        .ok()
                        .flatten()
                        .map(|texture| texture.get()),
                    visible: engine.visibility(id).unwrap_or(true),
                    point_light: engine.point_light(id).ok().flatten(),
                })
            })
            .collect();
        Self {
            entities,
            commands: Vec::new(),
            next_temporary_id: -1,
            pending_obj_loads: 0,
            owner_id: owner.and_then(|id| i64::try_from(id.get()).ok()),
            delta_seconds,
            camera: engine.camera(),
            gravity: engine.gravity(),
            input: engine.input().clone(),
            lighting: engine.lighting(),
        }
    }
}

fn install_engine_api(ctx: Context<'_>, bridge: Rc<RefCell<EngineBridge>>) {
    let engine_api = Table::new(&ctx);

    let lighting_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_lighting",
            Callback::from_fn_with(&ctx, lighting_bridge, |bridge, ctx, _, mut stack| {
                let lighting = bridge.borrow().lighting;
                let result = Table::new(&ctx);
                result
                    .set(ctx, "direction_x", lighting.direction.x)
                    .unwrap();
                result
                    .set(ctx, "direction_y", lighting.direction.y)
                    .unwrap();
                result
                    .set(ctx, "direction_z", lighting.direction.z)
                    .unwrap();
                result.set(ctx, "color_r", lighting.color[0]).unwrap();
                result.set(ctx, "color_g", lighting.color[1]).unwrap();
                result.set(ctx, "color_b", lighting.color[2]).unwrap();
                result.set(ctx, "intensity", lighting.intensity).unwrap();
                result.set(ctx, "ambient_r", lighting.ambient[0]).unwrap();
                result.set(ctx, "ambient_g", lighting.ambient[1]).unwrap();
                result.set(ctx, "ambient_b", lighting.ambient[2]).unwrap();
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let lighting_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_lighting",
            Callback::from_fn_with(&ctx, lighting_bridge, |bridge, ctx, _, mut stack| {
                let (
                    dx, dy, dz, red, green, blue, intensity, ambient_r, ambient_g, ambient_b,
                ): (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) = stack.consume(ctx)?;
                let mut lighting = Lighting3d {
                    direction: Vec3::new(dx as f32, dy as f32, dz as f32),
                    color: [red as f32, green as f32, blue as f32],
                    intensity: intensity as f32,
                    ambient: [ambient_r as f32, ambient_g as f32, ambient_b as f32],
                };
                if !lighting.direction.is_finite()
                    || lighting.direction.length_squared() <= f32::EPSILON
                    || !lighting.intensity.is_finite()
                    || !(0.0..=1000.0).contains(&lighting.intensity)
                    || lighting
                        .color
                        .iter()
                        .any(|channel| !channel.is_finite() || !(0.0..=16.0).contains(channel))
                    || lighting
                        .ambient
                        .iter()
                        .any(|channel| !channel.is_finite() || !(0.0..=1.0).contains(channel))
                {
                    return Err("engine.set_lighting received invalid direction, color, intensity, or ambient values".into_value(ctx).into());
                }
                lighting.direction = lighting.direction.normalize();
                let mut bridge = bridge.borrow_mut();
                bridge.lighting = lighting;
                bridge
                    .commands
                    .push(EngineCommand::SetLighting(lighting));
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let point_light_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_point_light",
            Callback::from_fn_with(&ctx, point_light_bridge, |bridge, ctx, _, mut stack| {
                let (id, red, green, blue, intensity, radius):
                    (i64, f64, f64, f64, f64, f64) = stack.consume(ctx)?;
                let light = PointLight3d {
                    color: [red as f32, green as f32, blue as f32],
                    intensity: intensity as f32,
                    radius: radius as f32,
                };
                if light.color.iter().any(|value| {
                    !value.is_finite() || !(0.0..=16.0).contains(value)
                }) || !light.intensity.is_finite()
                    || !(0.0..=1000.0).contains(&light.intensity)
                    || !light.radius.is_finite()
                    || light.radius <= 0.0
                    || light.radius > 10_000.0
                {
                    return Err("engine.set_point_light requires RGB values from 0 through 16, intensity from 0 through 1000, and a radius from 0 through 10000".into_value(ctx).into());
                }
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                entity.point_light = Some(light);
                bridge
                    .commands
                    .push(EngineCommand::SetPointLight { id, light: Some(light) });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let point_light_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_point_light",
            Callback::from_fn_with(&ctx, point_light_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let bridge = bridge.borrow();
                let Some(entity) = bridge.entities.iter().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let Some(light) = entity.point_light else {
                    stack.replace(ctx, Value::Nil);
                    return Ok(CallbackReturn::Return);
                };
                let result = Table::new(&ctx);
                result.set(ctx, "color_r", light.color[0]).unwrap();
                result.set(ctx, "color_g", light.color[1]).unwrap();
                result.set(ctx, "color_b", light.color[2]).unwrap();
                result.set(ctx, "intensity", light.intensity).unwrap();
                result.set(ctx, "radius", light.radius).unwrap();
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let point_light_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "remove_point_light",
            Callback::from_fn_with(&ctx, point_light_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let existed = entity.point_light.take().is_some();
                bridge
                    .commands
                    .push(EngineCommand::SetPointLight { id, light: None });
                stack.replace(ctx, existed);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let texture_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_base_color_texture",
            Callback::from_fn_with(&ctx, texture_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let bridge = bridge.borrow();
                let Some(entity) = bridge.entities.iter().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                if entity.material.is_none() {
                    return Err(format!("engine entity {id} has no 3D material")
                        .into_value(ctx)
                        .into());
                }
                if let Some(texture_id) = entity.base_color_texture {
                    stack.replace(ctx, texture_id as i64);
                } else {
                    stack.replace(ctx, Value::Nil);
                }
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let visibility_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_visibility",
            Callback::from_fn_with(&ctx, visibility_bridge, |bridge, ctx, _, mut stack| {
                let (id, visible): (i64, bool) = stack.consume(ctx)?;
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                entity.visible = visible;
                bridge
                    .commands
                    .push(EngineCommand::SetVisibility { id, visible });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let visibility_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_visibility",
            Callback::from_fn_with(&ctx, visibility_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let bridge = bridge.borrow();
                let Some(entity) = bridge.entities.iter().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                stack.replace(ctx, entity.visible);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let visibility_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "is_visible",
            Callback::from_fn_with(&ctx, visibility_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let bridge = bridge.borrow();
                if !bridge.entities.iter().any(|entity| entity.id == id) {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                }
                stack.replace(ctx, shadow_is_visible(&bridge.entities, id));
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let input_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "is_key_down",
            Callback::from_fn_with(&ctx, input_bridge, |bridge, ctx, _, mut stack| {
                let key: String = stack.consume(ctx)?;
                let down = bridge
                    .borrow()
                    .input
                    .keys_down
                    .contains(&key.to_ascii_lowercase());
                stack.replace(ctx, down);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();
    let input_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "is_mouse_button_down",
            Callback::from_fn_with(&ctx, input_bridge, |bridge, ctx, _, mut stack| {
                let button: String = stack.consume(ctx)?;
                let down = bridge
                    .borrow()
                    .input
                    .mouse_buttons_down
                    .contains(&button.to_ascii_lowercase());
                stack.replace(ctx, down);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();
    let input_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "mouse_delta",
            Callback::from_fn_with(&ctx, input_bridge, |bridge, ctx, _, mut stack| {
                let delta = bridge.borrow().input.mouse_delta;
                let result = Table::new(&ctx);
                result.set(ctx, 1, delta[0]).unwrap();
                result.set(ctx, 2, delta[1]).unwrap();
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let list_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "list_entities",
            Callback::from_fn_with(&ctx, list_bridge, |bridge, ctx, _, mut stack| {
                let result = Table::new(&ctx);
                let entities = bridge.borrow().entities.clone();
                for (index, entity) in entities.iter().enumerate() {
                    let row = Table::new(&ctx);
                    row.set(ctx, "id", entity.id).unwrap();
                    row.set(ctx, "name", entity.name.clone()).unwrap();
                    row.set(ctx, "x", entity.transform.translation.x).unwrap();
                    row.set(ctx, "y", entity.transform.translation.y).unwrap();
                    row.set(ctx, "z", entity.transform.translation.z).unwrap();
                    row.set(ctx, "visible", entity.visible).unwrap();
                    row.set(
                        ctx,
                        "effectively_visible",
                        shadow_is_visible(&entities, entity.id),
                    )
                    .unwrap();
                    if let Some(parent_id) = entity.parent_id {
                        row.set(ctx, "parent_id", parent_id).unwrap();
                    }
                    result.set(ctx, (index + 1) as i64, row).unwrap();
                }
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let world_transform_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_world_transform",
            Callback::from_fn_with(&ctx, world_transform_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let bridge = bridge.borrow();
                let Some(entity) = bridge.entities.iter().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let mut chain = vec![entity];
                let mut current = entity;
                while let Some(parent_id) = current.parent_id {
                    let Some(parent) = bridge.entities.iter().find(|entity| entity.id == parent_id)
                    else {
                        return Err(format!("unknown parent entity {parent_id}")
                            .into_value(ctx)
                            .into());
                    };
                    chain.push(parent);
                    current = parent;
                }
                let matrix = chain
                    .iter()
                    .rev()
                    .fold(glam::Mat4::IDENTITY, |matrix, entity| {
                        matrix
                            * glam::Mat4::from_scale_rotation_translation(
                                entity.transform.scale,
                                entity.transform.rotation,
                                entity.transform.translation,
                            )
                    });
                let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
                let result = Table::new(&ctx);
                for (field, vector) in [
                    ("position", translation.to_array().to_vec()),
                    ("scale", scale.to_array().to_vec()),
                    ("rotation", rotation.to_array().to_vec()),
                ] {
                    let values = Table::new(&ctx);
                    for (index, value) in vector.into_iter().enumerate() {
                        values.set(ctx, (index + 1) as i64, value).unwrap();
                    }
                    result.set(ctx, field, values).unwrap();
                }
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let set_parent_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_parent",
            Callback::from_fn_with(&ctx, set_parent_bridge, |bridge, ctx, _, mut stack| {
                let (id, parent_id): (i64, Option<i64>) = stack.consume(ctx)?;
                let mut bridge = bridge.borrow_mut();
                let Some(index) = bridge.entities.iter().position(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                if let Some(parent_id) = parent_id
                    && !bridge.entities.iter().any(|entity| entity.id == parent_id)
                {
                    return Err(format!("unknown parent entity {parent_id}")
                        .into_value(ctx)
                        .into());
                }
                bridge.entities[index].parent_id = parent_id;
                bridge
                    .commands
                    .push(EngineCommand::SetParent { id, parent_id });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let gravity_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_gravity",
            Callback::from_fn_with(&ctx, gravity_bridge, |bridge, ctx, _, mut stack| {
                let gravity = bridge.borrow().gravity;
                let result = Table::new(&ctx);
                for (index, value) in gravity.to_array().into_iter().enumerate() {
                    result.set(ctx, (index + 1) as i64, value).unwrap();
                }
                result.set(ctx, "x", gravity.x).unwrap();
                result.set(ctx, "y", gravity.y).unwrap();
                result.set(ctx, "z", gravity.z).unwrap();
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let set_gravity_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_gravity",
            Callback::from_fn_with(&ctx, set_gravity_bridge, |bridge, ctx, _, mut stack| {
                let (x, y, z): (f64, f64, f64) = stack.consume(ctx)?;
                let Some(gravity) = finite_position(x, y, z) else {
                    return Err("engine.set_gravity requires finite coordinates"
                        .into_value(ctx)
                        .into());
                };
                let gravity = Vec3::from_array(gravity);
                let mut bridge = bridge.borrow_mut();
                bridge.gravity = gravity;
                bridge.commands.push(EngineCommand::SetGravity(gravity));
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let rigid_body_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_rigid_body",
            Callback::from_fn_with(&ctx, rigid_body_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let bridge = bridge.borrow();
                let Some(entity) = bridge.entities.iter().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let Some(body) = entity.rigid_body else {
                    stack.replace(ctx, Value::Nil);
                    return Ok(CallbackReturn::Return);
                };
                let result = Table::new(&ctx);
                result
                    .set(
                        ctx,
                        "body_type",
                        match body.body_type {
                            RigidBodyType::Dynamic => "dynamic",
                            RigidBodyType::Kinematic => "kinematic",
                            RigidBodyType::Static => "static",
                        },
                    )
                    .unwrap();
                result.set(ctx, "mass", body.mass).unwrap();
                result
                    .set(ctx, "gravity_scale", body.gravity_scale)
                    .unwrap();
                result
                    .set(ctx, "linear_damping", body.linear_damping)
                    .unwrap();
                let velocity = Table::new(&ctx);
                for (index, value) in body.velocity.to_array().into_iter().enumerate() {
                    velocity.set(ctx, (index + 1) as i64, value).unwrap();
                }
                result.set(ctx, "velocity", velocity).unwrap();
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let add_body_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "add_rigid_body",
            Callback::from_fn_with(&ctx, add_body_bridge, |bridge, ctx, _, mut stack| {
                let (id, body_type, mass, gravity_scale, linear_damping): (
                    i64,
                    Option<String>,
                    Option<f64>,
                    Option<f64>,
                    Option<f64>,
                ) = stack.consume(ctx)?;
                let defaults = RigidBody3d::default();
                let body_type = match body_type.as_deref().unwrap_or("dynamic") {
                    "dynamic" => RigidBodyType::Dynamic,
                    "kinematic" => RigidBodyType::Kinematic,
                    "static" => RigidBodyType::Static,
                    _ => {
                        return Err("body_type must be dynamic, kinematic, or static"
                            .into_value(ctx)
                            .into());
                    }
                };
                let body = RigidBody3d {
                    body_type,
                    mass: mass.map(|value| value as f32).unwrap_or(defaults.mass),
                    gravity_scale: gravity_scale
                        .map(|value| value as f32)
                        .unwrap_or(defaults.gravity_scale),
                    linear_damping: linear_damping
                        .map(|value| value as f32)
                        .unwrap_or(defaults.linear_damping),
                    ..defaults
                };
                if !body.mass.is_finite()
                    || body.mass <= 0.0
                    || !body.gravity_scale.is_finite()
                    || !body.linear_damping.is_finite()
                    || body.linear_damping < 0.0
                {
                    return Err("rigid body mass must be positive and damping nonnegative"
                        .into_value(ctx)
                        .into());
                }
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                entity.rigid_body = Some(body);
                bridge
                    .commands
                    .push(EngineCommand::AddRigidBody { id, body });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let remove_body_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "remove_rigid_body",
            Callback::from_fn_with(&ctx, remove_body_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let removed = entity.rigid_body.take().is_some();
                if removed {
                    bridge.commands.push(EngineCommand::RemoveRigidBody { id });
                }
                stack.replace(ctx, removed);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let velocity_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_velocity",
            Callback::from_fn_with(&ctx, velocity_bridge, |bridge, ctx, _, mut stack| {
                let (id, x, y, z): (i64, f64, f64, f64) = stack.consume(ctx)?;
                let Some(velocity) = finite_position(x, y, z) else {
                    return Err("engine.set_velocity requires finite coordinates"
                        .into_value(ctx)
                        .into());
                };
                let velocity = Vec3::from_array(velocity);
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let Some(body) = entity.rigid_body.as_mut() else {
                    return Err(format!("engine entity {id} has no rigid body")
                        .into_value(ctx)
                        .into());
                };
                body.velocity = velocity;
                bridge
                    .commands
                    .push(EngineCommand::SetVelocity { id, velocity });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let force_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "apply_force",
            Callback::from_fn_with(&ctx, force_bridge, |bridge, ctx, _, mut stack| {
                let (id, x, y, z): (i64, f64, f64, f64) = stack.consume(ctx)?;
                let Some(force) = finite_position(x, y, z) else {
                    return Err("engine.apply_force requires finite coordinates"
                        .into_value(ctx)
                        .into());
                };
                let force = Vec3::from_array(force);
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let Some(body) = entity.rigid_body.as_mut() else {
                    return Err(format!("engine entity {id} has no rigid body")
                        .into_value(ctx)
                        .into());
                };
                let accumulated_force = body.accumulated_force + force;
                if !accumulated_force.is_finite() {
                    return Err("engine.apply_force exceeds the finite physics range"
                        .into_value(ctx)
                        .into());
                }
                body.accumulated_force = accumulated_force;
                bridge
                    .commands
                    .push(EngineCommand::ApplyForce { id, force });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let get_collider_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_collider",
            Callback::from_fn_with(&ctx, get_collider_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let bridge = bridge.borrow();
                let Some(entity) = bridge.entities.iter().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let Some(collider) = entity.collider else {
                    stack.replace(ctx, Value::Nil);
                    return Ok(CallbackReturn::Return);
                };
                let result = Table::new(&ctx);
                result
                    .set(ctx, "restitution", collider.restitution)
                    .unwrap();
                result.set(ctx, "friction", collider.friction).unwrap();
                match collider.shape {
                    ColliderShape::Sphere { radius } => {
                        result.set(ctx, "shape", "sphere").unwrap();
                        result.set(ctx, "radius", radius).unwrap();
                    }
                    ColliderShape::Box { half_extents } => {
                        result.set(ctx, "shape", "box").unwrap();
                        let values = Table::new(&ctx);
                        for (index, value) in half_extents.to_array().into_iter().enumerate() {
                            values.set(ctx, (index + 1) as i64, value).unwrap();
                        }
                        result.set(ctx, "half_extents", values).unwrap();
                    }
                }
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let sphere_collider_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_sphere_collider",
            Callback::from_fn_with(&ctx, sphere_collider_bridge, |bridge, ctx, _, mut stack| {
                let (id, radius, restitution, friction): (
                    i64,
                    f64,
                    Option<f64>,
                    Option<f64>,
                ) = stack.consume(ctx)?;
                let collider = Collider3d {
                    shape: ColliderShape::Sphere {
                        radius: radius as f32,
                    },
                    restitution: restitution.unwrap_or(0.0) as f32,
                    friction: friction.unwrap_or(0.5) as f32,
                };
                if !radius.is_finite()
                    || radius <= 0.0
                    || !collider.restitution.is_finite()
                    || !(0.0..=1.0).contains(&collider.restitution)
                    || !collider.friction.is_finite()
                    || !(0.0..=1.0).contains(&collider.friction)
                {
                    return Err("sphere collider radius must be positive; friction and restitution must be between 0 and 1"
                        .into_value(ctx).into());
                }
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                entity.collider = Some(collider);
                bridge
                    .commands
                    .push(EngineCommand::SetCollider { id, collider });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let box_collider_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_box_collider",
            Callback::from_fn_with(&ctx, box_collider_bridge, |bridge, ctx, _, mut stack| {
                let (id, x, y, z, restitution, friction): (
                    i64,
                    f64,
                    f64,
                    f64,
                    Option<f64>,
                    Option<f64>,
                ) = stack.consume(ctx)?;
                let Some(half_extents) = finite_position(x, y, z) else {
                    return Err("box collider half extents must be finite and positive"
                        .into_value(ctx)
                        .into());
                };
                let collider = Collider3d {
                    shape: ColliderShape::Box {
                        half_extents: Vec3::from_array(half_extents),
                    },
                    restitution: restitution.unwrap_or(0.0) as f32,
                    friction: friction.unwrap_or(0.5) as f32,
                };
                if half_extents.iter().any(|value| *value <= 0.0)
                    || !collider.restitution.is_finite()
                    || !(0.0..=1.0).contains(&collider.restitution)
                    || !collider.friction.is_finite()
                    || !(0.0..=1.0).contains(&collider.friction)
                {
                    return Err("box collider half extents must be positive; friction and restitution must be between 0 and 1"
                        .into_value(ctx).into());
                }
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                entity.collider = Some(collider);
                bridge
                    .commands
                    .push(EngineCommand::SetCollider { id, collider });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let remove_collider_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "remove_collider",
            Callback::from_fn_with(&ctx, remove_collider_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let removed = entity.collider.take().is_some();
                if removed {
                    bridge.commands.push(EngineCommand::RemoveCollider { id });
                }
                stack.replace(ctx, removed);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let transform_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_transform",
            Callback::from_fn_with(&ctx, transform_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let bridge = bridge.borrow();
                let Some(entity) = bridge.entities.iter().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let result = Table::new(&ctx);
                let translation = entity.transform.translation.to_array();
                result.set(ctx, "x", translation[0]).unwrap();
                result.set(ctx, "y", translation[1]).unwrap();
                result.set(ctx, "z", translation[2]).unwrap();
                let rotation = Table::new(&ctx);
                for (index, value) in entity.transform.rotation.to_array().into_iter().enumerate() {
                    rotation.set(ctx, (index + 1) as i64, value).unwrap();
                }
                result.set(ctx, "rotation", rotation).unwrap();
                let scale = Table::new(&ctx);
                for (index, value) in entity.transform.scale.to_array().into_iter().enumerate() {
                    scale.set(ctx, (index + 1) as i64, value).unwrap();
                }
                result.set(ctx, "scale", scale).unwrap();
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let material_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_material",
            Callback::from_fn_with(&ctx, material_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let bridge = bridge.borrow();
                let Some(entity) = bridge.entities.iter().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let Some(material) = entity.material else {
                    stack.replace(ctx, Value::Nil);
                    return Ok(CallbackReturn::Return);
                };
                let result = Table::new(&ctx);
                for (index, value) in material.base_color.into_iter().enumerate() {
                    result.set(ctx, (index + 1) as i64, value).unwrap();
                }
                result.set(ctx, "r", material.base_color[0]).unwrap();
                result.set(ctx, "g", material.base_color[1]).unwrap();
                result.set(ctx, "b", material.base_color[2]).unwrap();
                result.set(ctx, "a", material.base_color[3]).unwrap();
                result.set(ctx, "roughness", material.roughness).unwrap();
                result.set(ctx, "metallic", material.metallic).unwrap();
                result
                    .set(
                        ctx,
                        "alpha_mode",
                        format!("{:?}", material.alpha_mode).to_lowercase(),
                    )
                    .unwrap();
                result
                    .set(ctx, "alpha_cutoff", material.alpha_cutoff)
                    .unwrap();
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let get_camera_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "get_camera",
            Callback::from_fn_with(&ctx, get_camera_bridge, |bridge, ctx, _, mut stack| {
                let camera = bridge.borrow().camera;
                let result = Table::new(&ctx);
                for (field, vector) in [
                    ("position", camera.position),
                    ("target", camera.target),
                    ("up", camera.up),
                ] {
                    let values = Table::new(&ctx);
                    for (index, value) in vector.to_array().into_iter().enumerate() {
                        values.set(ctx, (index + 1) as i64, value).unwrap();
                    }
                    result.set(ctx, field, values).unwrap();
                }
                result
                    .set(
                        ctx,
                        "vertical_fov_degrees",
                        camera.vertical_fov_radians.to_degrees(),
                    )
                    .unwrap();
                result.set(ctx, "near", camera.near).unwrap();
                result.set(ctx, "far", camera.far).unwrap();
                let clear_color = Table::new(&ctx);
                for (index, value) in camera.clear_color.into_iter().enumerate() {
                    clear_color.set(ctx, (index + 1) as i64, value).unwrap();
                }
                result.set(ctx, "clear_color", clear_color).unwrap();
                let (projection, orthographic_size) = match camera.projection {
                    CameraProjection3d::Perspective => ("perspective", 0.0),
                    CameraProjection3d::Orthographic { vertical_size } => {
                        ("orthographic", vertical_size)
                    }
                };
                result.set(ctx, "projection", projection).unwrap();
                result
                    .set(ctx, "orthographic_vertical_size", orthographic_size)
                    .unwrap();
                stack.replace(ctx, result);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let projection_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_camera_projection",
            Callback::from_fn_with(&ctx, projection_bridge, |bridge, ctx, _, mut stack| {
                let (kind, vertical_size): (String, f64) = stack.consume(ctx)?;
                let projection = match kind.as_str() {
                    "perspective" => CameraProjection3d::Perspective,
                    "orthographic" if vertical_size.is_finite() => {
                        CameraProjection3d::Orthographic {
                            vertical_size: vertical_size as f32,
                        }
                    }
                    _ => {
                        return Err("projection must be perspective or orthographic with a valid vertical size"
                            .into_value(ctx)
                            .into());
                    }
                };
                let mut bridge = bridge.borrow_mut();
                let camera = Camera3d {
                    projection,
                    ..bridge.camera
                }
                .validated()
                .map_err(|error| error.to_string().into_value(ctx))?;
                bridge.camera = camera;
                bridge.commands.push(EngineCommand::SetCamera(camera));
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let camera_position_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_camera_position",
            Callback::from_fn_with(&ctx, camera_position_bridge, |bridge, ctx, _, mut stack| {
                let (x, y, z): (f64, f64, f64) = stack.consume(ctx)?;
                let Some(position) = finite_position(x, y, z) else {
                    return Err("engine.set_camera_position requires finite coordinates"
                        .into_value(ctx)
                        .into());
                };
                let mut bridge = bridge.borrow_mut();
                let camera = Camera3d {
                    position: Vec3::from_array(position),
                    ..bridge.camera
                }
                .validated()
                .map_err(|error| error.to_string().into_value(ctx))?;
                bridge.camera = camera;
                bridge.commands.push(EngineCommand::SetCamera(camera));
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let camera_target_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_camera_target",
            Callback::from_fn_with(&ctx, camera_target_bridge, |bridge, ctx, _, mut stack| {
                let (x, y, z): (f64, f64, f64) = stack.consume(ctx)?;
                let Some(target) = finite_position(x, y, z) else {
                    return Err("engine.set_camera_target requires finite coordinates"
                        .into_value(ctx)
                        .into());
                };
                let mut bridge = bridge.borrow_mut();
                let camera = Camera3d {
                    target: Vec3::from_array(target),
                    ..bridge.camera
                }
                .validated()
                .map_err(|error| error.to_string().into_value(ctx))?;
                bridge.camera = camera;
                bridge.commands.push(EngineCommand::SetCamera(camera));
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let owner_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "self",
            Callback::from_fn_with(&ctx, owner_bridge, |bridge, ctx, _, mut stack| {
                let owner_id = bridge
                    .borrow()
                    .owner_id
                    .map(Value::from)
                    .unwrap_or(Value::Nil);
                stack.replace(ctx, owner_id);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let delta_seconds = bridge.borrow().delta_seconds;
    engine_api
        .set(
            ctx,
            "delta_time",
            Callback::from_fn(&ctx, move |ctx, _, mut stack| {
                stack.replace(ctx, delta_seconds);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let spawn_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "spawn",
            Callback::from_fn_with(&ctx, spawn_bridge, |bridge, ctx, _, mut stack| {
                let (name, x, y, z, primitive): (String, f64, f64, f64, Option<String>) =
                    stack.consume(ctx)?;
                let Some(position) = finite_position(x, y, z) else {
                    return Err("engine.spawn requires finite x, y, and z coordinates"
                        .into_value(ctx)
                        .into());
                };
                let mesh = match primitive.as_deref().unwrap_or("cube") {
                    "cube" => MeshKind::Cube,
                    "sphere" => MeshKind::Sphere,
                    "plane" => MeshKind::Plane,
                    _ => {
                        return Err("engine.spawn primitive must be cube, sphere, or plane"
                            .into_value(ctx)
                            .into());
                    }
                };
                let mut bridge = bridge.borrow_mut();
                let temporary_id = bridge.next_temporary_id;
                bridge.next_temporary_id -= 1;
                bridge.entities.push(ShadowEntity {
                    id: temporary_id,
                    name: name.clone(),
                    transform: Transform {
                        translation: Vec3::from_array(position),
                        ..Transform::default()
                    },
                    material: Some(Material3d::default()),
                    rigid_body: None,
                    collider: None,
                    parent_id: None,
                    base_color_texture: None,
                    visible: true,
                    point_light: None,
                });
                bridge.commands.push(EngineCommand::Spawn {
                    temporary_id,
                    name,
                    position,
                    mesh,
                });
                stack.replace(ctx, temporary_id);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let obj_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "load_obj_mesh",
            Callback::from_fn_with(&ctx, obj_bridge, |bridge, ctx, _, mut stack| {
                let (name, source, x, y, z): (String, String, f64, f64, f64) =
                    stack.consume(ctx)?;
                let Some(position) = finite_position(x, y, z) else {
                    return Err(
                        "engine.load_obj_mesh requires finite x, y, and z coordinates"
                            .into_value(ctx)
                            .into(),
                    );
                };
                if source.len() > MAX_LUA_OBJ_BYTES {
                    return Err(format!(
                        "engine.load_obj_mesh source exceeds the {MAX_LUA_OBJ_BYTES}-byte Lua limit"
                    )
                    .into_value(ctx)
                    .into());
                }
                if bridge.borrow().pending_obj_loads >= MAX_LUA_OBJ_LOADS_PER_RUN {
                    return Err(format!(
                        "engine.load_obj_mesh permits {MAX_LUA_OBJ_LOADS_PER_RUN} loads per script run"
                    )
                    .into_value(ctx)
                    .into());
                }
                let mesh = MeshData::from_obj(&source)
                    .map_err(|error| error.to_string().into_value(ctx))?;
                let mut bridge = bridge.borrow_mut();
                bridge.pending_obj_loads += 1;
                let temporary_id = bridge.next_temporary_id;
                bridge.next_temporary_id -= 1;
                bridge.entities.push(ShadowEntity {
                    id: temporary_id,
                    name: name.clone(),
                    transform: Transform {
                        translation: Vec3::from_array(position),
                        ..Transform::default()
                    },
                    material: Some(Material3d::default()),
                    rigid_body: None,
                    collider: None,
                    parent_id: None,
                    base_color_texture: None,
                    visible: true,
                    point_light: None,
                });
                bridge.commands.push(EngineCommand::SpawnObj {
                    temporary_id,
                    name,
                    position,
                    mesh,
                });
                stack.replace(ctx, temporary_id);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let gltf_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "load_gltf_mesh",
            Callback::from_fn_with(&ctx, gltf_bridge, |bridge, ctx, _, mut stack| {
                let (name, source_base64, x, y, z): (String, String, f64, f64, f64) =
                    stack.consume(ctx)?;
                let Some(position) = finite_position(x, y, z) else {
                    return Err("engine.load_gltf_mesh requires finite x, y, and z coordinates"
                        .into_value(ctx)
                        .into());
                };
                if source_base64.len() > MAX_LUA_GLTF_BASE64_BYTES {
                    return Err(format!(
                        "engine.load_gltf_mesh base64 source exceeds {MAX_LUA_GLTF_BASE64_BYTES} bytes"
                    )
                    .into_value(ctx)
                    .into());
                }
                if bridge.borrow().pending_obj_loads >= MAX_LUA_OBJ_LOADS_PER_RUN {
                    return Err(format!(
                        "engine.load_gltf_mesh permits {MAX_LUA_OBJ_LOADS_PER_RUN} mesh loads per script run"
                    )
                    .into_value(ctx)
                    .into());
                }
                let source = STANDARD
                    .decode(source_base64)
                    .map_err(|error| error.to_string().into_value(ctx))?;
                let asset = GltfAssetData::from_gltf(&source)
                    .map_err(|error| error.to_string().into_value(ctx))?;
                let mut bridge = bridge.borrow_mut();
                bridge.pending_obj_loads += 1;
                let temporary_id = bridge.next_temporary_id;
                bridge.next_temporary_id -= 1;
                bridge.entities.push(ShadowEntity {
                    id: temporary_id,
                    name: name.clone(),
                    transform: Transform {
                        translation: Vec3::from_array(position),
                        ..Transform::default()
                    },
                    material: None,
                    rigid_body: None,
                    collider: None,
                    parent_id: None,
                    base_color_texture: None,
                    visible: true,
                    point_light: None,
                });
                bridge.commands.push(EngineCommand::SpawnGltf {
                    temporary_id,
                    name,
                    position,
                    asset,
                });
                stack.replace(ctx, temporary_id);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let animation_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "play_gltf_animation",
            Callback::from_fn_with(&ctx, animation_bridge, |bridge, ctx, _, mut stack| {
                let (id, clip_index, looping): (i64, i64, bool) = stack.consume(ctx)?;
                let Ok(clip_index) = usize::try_from(clip_index) else {
                    return Err("clip_index must be a nonnegative integer"
                        .into_value(ctx)
                        .into());
                };
                bridge
                    .borrow_mut()
                    .commands
                    .push(EngineCommand::PlayGltfAnimation {
                        id,
                        clip_index,
                        looping,
                    });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let animation_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "stop_gltf_animation",
            Callback::from_fn_with(&ctx, animation_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                bridge
                    .borrow_mut()
                    .commands
                    .push(EngineCommand::StopGltfAnimation { id });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let animation_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_gltf_animation_speed",
            Callback::from_fn_with(&ctx, animation_bridge, |bridge, ctx, _, mut stack| {
                let (id, speed): (i64, f64) = stack.consume(ctx)?;
                if !speed.is_finite() || speed.abs() > 100.0 {
                    return Err("animation speed must be finite and between -100 and 100"
                        .into_value(ctx)
                        .into());
                }
                bridge
                    .borrow_mut()
                    .commands
                    .push(EngineCommand::SetGltfAnimationSpeed {
                        id,
                        speed: speed as f32,
                    });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let position_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_position",
            Callback::from_fn_with(&ctx, position_bridge, |bridge, ctx, _, mut stack| {
                let (id, x, y, z): (i64, f64, f64, f64) = stack.consume(ctx)?;
                let Some(position) = finite_position(x, y, z) else {
                    return Err(
                        "engine.set_position requires finite x, y, and z coordinates"
                            .into_value(ctx)
                            .into(),
                    );
                };
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                entity.transform.translation = Vec3::from_array(position);
                bridge
                    .commands
                    .push(EngineCommand::SetPosition { id, position });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let rotation_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_rotation",
            Callback::from_fn_with(&ctx, rotation_bridge, |bridge, ctx, _, mut stack| {
                let (id, x, y, z, w): (i64, f64, f64, f64, f64) = stack.consume(ctx)?;
                let Some(rotation) = finite_rotation(x, y, z, w) else {
                    return Err("engine.set_rotation requires a finite, nonzero quaternion"
                        .into_value(ctx)
                        .into());
                };
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                entity.transform.rotation = Quat::from_array(rotation);
                bridge
                    .commands
                    .push(EngineCommand::SetRotation { id, rotation });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let scale_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_scale",
            Callback::from_fn_with(&ctx, scale_bridge, |bridge, ctx, _, mut stack| {
                let (id, x, y, z): (i64, f64, f64, f64) = stack.consume(ctx)?;
                let Some(scale) = finite_position(x, y, z) else {
                    return Err("engine.set_scale requires finite x, y, and z values"
                        .into_value(ctx)
                        .into());
                };
                let mut bridge = bridge.borrow_mut();
                let Some(entity) = bridge.entities.iter_mut().find(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                entity.transform.scale = Vec3::from_array(scale);
                bridge.commands.push(EngineCommand::SetScale { id, scale });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let set_material_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_material",
            Callback::from_fn_with(&ctx, set_material_bridge, |bridge, ctx, _, mut stack| {
                let (id, r, g, b, a, roughness, metallic): (
                    i64,
                    f64,
                    f64,
                    f64,
                    f64,
                    Option<f64>,
                    Option<f64>,
                ) = stack.consume(ctx)?;
                let base_color = [r as f32, g as f32, b as f32, a as f32];
                if base_color
                    .iter()
                    .any(|channel| !channel.is_finite() || !(0.0..=1.0).contains(channel))
                    || roughness.is_some_and(|value| {
                        !value.is_finite() || !(0.0..=1.0).contains(&value)
                    })
                    || metallic.is_some_and(|value| {
                        !value.is_finite() || !(0.0..=1.0).contains(&value)
                    })
                {
                    return Err(
                        "engine.set_material requires RGBA, roughness, and metallic values from 0 through 1"
                            .into_value(ctx)
                            .into(),
                    );
                }
                let mut bridge = bridge.borrow_mut();
                let Some(index) = bridge.entities.iter().position(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let Some(current) = bridge.entities[index].material else {
                    return Err(format!("engine entity {id} has no 3D material")
                        .into_value(ctx)
                        .into());
                };
                let material = Material3d {
                    base_color,
                    roughness: roughness.map(|value| value as f32).unwrap_or(current.roughness),
                    metallic: metallic.map(|value| value as f32).unwrap_or(current.metallic),
                    ..current
                };
                bridge.entities[index].material = Some(material);
                bridge.commands.push(EngineCommand::SetMaterial { id, material });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let alpha_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_alpha_mode",
            Callback::from_fn_with(&ctx, alpha_bridge, |bridge, ctx, _, mut stack| {
                let (id, mode, cutoff): (i64, String, f64) = stack.consume(ctx)?;
                let alpha_mode = match mode.as_str() {
                    "auto" => AlphaMode3d::Auto,
                    "opaque" => AlphaMode3d::Opaque,
                    "mask" => AlphaMode3d::Mask,
                    "blend" => AlphaMode3d::Blend,
                    _ => {
                        return Err("alpha mode must be auto, opaque, mask, or blend"
                            .into_value(ctx)
                            .into());
                    }
                };
                let alpha_cutoff = cutoff as f32;
                if !alpha_cutoff.is_finite() || !(0.0..=1.0).contains(&alpha_cutoff) {
                    return Err("alpha cutoff must be between 0 and 1"
                        .into_value(ctx)
                        .into());
                }
                let mut bridge = bridge.borrow_mut();
                let Some(index) = bridge.entities.iter().position(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                let Some(current) = bridge.entities[index].material else {
                    return Err(format!("engine entity {id} has no 3D material")
                        .into_value(ctx)
                        .into());
                };
                let material = Material3d {
                    alpha_mode,
                    alpha_cutoff,
                    ..current
                };
                bridge.entities[index].material = Some(material);
                bridge
                    .commands
                    .push(EngineCommand::SetMaterial { id, material });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    engine_api
        .set(
            ctx,
            "delete",
            Callback::from_fn_with(&ctx, Rc::clone(&bridge), |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let mut bridge = bridge.borrow_mut();
                let Some(index) = bridge.entities.iter().position(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                bridge.entities.swap_remove(index);
                bridge.commands.push(EngineCommand::Delete { id });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    let texture_bridge = Rc::clone(&bridge);
    engine_api
        .set(
            ctx,
            "set_base_color_texture",
            Callback::from_fn_with(&ctx, texture_bridge, |bridge, ctx, _, mut stack| {
                let id: i64 = stack.consume(ctx)?;
                let texture_id: Option<i64> = stack.consume(ctx)?;
                let mut bridge = bridge.borrow_mut();
                let Some(index) = bridge.entities.iter().position(|entity| entity.id == id) else {
                    return Err(format!("unknown engine entity {id}").into_value(ctx).into());
                };
                if bridge.entities[index].material.is_none() {
                    return Err(format!("engine entity {id} has no 3D material")
                        .into_value(ctx)
                        .into());
                }
                let texture_id = texture_id.map(|id| u64::try_from(id).unwrap_or(0));
                if texture_id == Some(0) {
                    return Err("texture asset ID must be positive".into_value(ctx).into());
                }
                bridge.entities[index].base_color_texture = texture_id;
                bridge
                    .commands
                    .push(EngineCommand::SetBaseColorTexture { id, texture_id });
                stack.replace(ctx, true);
                Ok(CallbackReturn::Return)
            }),
        )
        .unwrap();

    ctx.set_global("engine", engine_api).unwrap();
}

fn finite_position(x: f64, y: f64, z: f64) -> Option<[f32; 3]> {
    let position = [x as f32, y as f32, z as f32];
    position
        .iter()
        .all(|value| value.is_finite())
        .then_some(position)
}

fn finite_rotation(x: f64, y: f64, z: f64, w: f64) -> Option<[f32; 4]> {
    let rotation = Quat::from_xyzw(x as f32, y as f32, z as f32, w as f32);
    let length_squared = rotation.length_squared();
    (rotation.is_finite() && length_squared.is_finite() && length_squared > f32::EPSILON)
        .then(|| rotation.normalize().to_array())
}

fn apply_engine_commands(
    engine: &mut Engine,
    bridge: Rc<RefCell<EngineBridge>>,
) -> Result<(), ScriptError> {
    let commands = std::mem::take(&mut bridge.borrow_mut().commands);
    let mut spawned_ids = HashMap::<i64, EntityId>::new();
    for command in commands {
        match command {
            EngineCommand::Spawn {
                temporary_id,
                name,
                position,
                mesh,
            } => {
                let transform = Transform {
                    translation: Vec3::from_array(position),
                    ..Transform::default()
                };
                spawned_ids.insert(
                    temporary_id,
                    engine
                        .spawn_mesh(name, transform, mesh, Material3d::default())
                        .map_err(|error| ScriptError::Engine(error.to_string()))?,
                );
            }
            EngineCommand::SpawnObj {
                temporary_id,
                name,
                position,
                mesh,
            } => {
                let transform = Transform {
                    translation: Vec3::from_array(position),
                    ..Transform::default()
                };
                let mesh_asset_id = engine.add_mesh_asset(mesh);
                let entity_id = engine
                    .spawn_mesh_asset(name, transform, mesh_asset_id, Material3d::default())
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
                spawned_ids.insert(temporary_id, entity_id);
            }
            EngineCommand::SpawnGltf {
                temporary_id,
                name,
                position,
                asset,
            } => {
                let instance = engine
                    .spawn_gltf_asset(name, Vec3::from_array(position), asset)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
                spawned_ids.insert(temporary_id, instance.root);
            }
            EngineCommand::PlayGltfAnimation {
                id,
                clip_index,
                looping,
            } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .play_gltf_animation(entity_id, clip_index, looping)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::StopGltfAnimation { id } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .stop_gltf_animation(entity_id)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetGltfAnimationSpeed { id, speed } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .set_gltf_animation_speed(entity_id, speed)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetPosition { id, position } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                let mut transform = engine
                    .transform(entity_id)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
                transform.translation = Vec3::from_array(position);
                engine
                    .set_transform(entity_id, transform)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetRotation { id, rotation } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                let mut transform = engine
                    .transform(entity_id)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
                transform.rotation = Quat::from_array(rotation);
                engine
                    .set_transform(entity_id, transform)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetScale { id, scale } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                let mut transform = engine
                    .transform(entity_id)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
                transform.scale = Vec3::from_array(scale);
                engine
                    .set_transform(entity_id, transform)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetMaterial { id, material } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .set_material(entity_id, material)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetBaseColorTexture { id, texture_id } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .set_base_color_texture(entity_id, texture_id.map(TextureAssetId))
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetVisibility { id, visible } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .set_visibility(entity_id, visible)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetPointLight { id, light } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .set_point_light(entity_id, light)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetParent { id, parent_id } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                let parent_id = parent_id
                    .map(|parent_id| resolve_script_id(parent_id, &spawned_ids))
                    .transpose()?;
                engine
                    .set_parent(entity_id, parent_id)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::AddRigidBody { id, body } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .add_rigid_body(entity_id, body)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::RemoveRigidBody { id } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .remove_rigid_body(entity_id)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetVelocity { id, velocity } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .set_velocity(entity_id, velocity)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::ApplyForce { id, force } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .apply_force(entity_id, force)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetCollider { id, collider } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .set_collider(entity_id, collider)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::RemoveCollider { id } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .remove_collider(entity_id)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
            EngineCommand::SetCamera(camera) => engine
                .set_camera(camera)
                .map_err(|error| ScriptError::Engine(error.to_string()))?,
            EngineCommand::SetLighting(lighting) => engine
                .set_lighting(lighting)
                .map_err(|error| ScriptError::Engine(error.to_string()))?,
            EngineCommand::SetGravity(gravity) => engine
                .set_gravity(gravity)
                .map_err(|error| ScriptError::Engine(error.to_string()))?,
            EngineCommand::Delete { id } => {
                let entity_id = resolve_script_id(id, &spawned_ids)?;
                engine
                    .despawn(entity_id)
                    .map_err(|error| ScriptError::Engine(error.to_string()))?;
            }
        }
    }
    Ok(())
}

fn shadow_is_visible(entities: &[ShadowEntity], id: i64) -> bool {
    let mut current = Some(id);
    for _ in 0..=entities.len() {
        let Some(id) = current else {
            return true;
        };
        let Some(entity) = entities.iter().find(|entity| entity.id == id) else {
            return false;
        };
        if !entity.visible {
            return false;
        }
        current = entity.parent_id;
    }
    false
}

fn resolve_script_id(
    id: i64,
    spawned_ids: &HashMap<i64, EntityId>,
) -> Result<EntityId, ScriptError> {
    if id < 0 {
        spawned_ids
            .get(&id)
            .copied()
            .ok_or_else(|| ScriptError::Engine(format!("unknown temporary entity {id}")))
    } else {
        u64::try_from(id)
            .map(EntityId::from_control)
            .map_err(|_| ScriptError::Engine(format!("invalid entity id {id}")))
    }
}

/// Owns persistent Lua states attached to scene entities. Setup chunks run
/// once at attach time; an optional global `update(dt)` runs on each fixed step.
#[derive(Default)]
pub struct ScriptHost {
    scripts: Vec<EntityScript>,
}

struct EntityScript {
    entity_id: EntityId,
    source: String,
    runtime: ScriptRuntime,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScriptFailure {
    pub entity_id: EntityId,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttachedScriptInfo {
    pub entity_id: EntityId,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScriptWorkerSnapshot {
    pub scripts: Vec<AttachedScriptInfo>,
    pub recent_failures: Vec<ScriptFailure>,
    pub engine: crate::EngineSnapshot,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SimulationAdvance {
    pub ticks_advanced: u32,
    pub engine: crate::EngineSnapshot,
}

impl ScriptHost {
    /// Attach and initialize a persistent script for a live scene entity.
    pub fn attach(
        &mut self,
        engine: &mut Engine,
        entity_id: EntityId,
        source: &str,
    ) -> Result<(), ScriptError> {
        engine
            .name(entity_id)
            .map_err(|error| ScriptError::Engine(error.to_string()))?;
        let mut runtime = ScriptRuntime::default_sandbox();
        runtime.run_with_engine_context(
            engine,
            Some(entity_id),
            0.0,
            &format!("entity://{}/setup", entity_id.get()),
            source,
        )?;

        if let Some(existing) = self
            .scripts
            .iter_mut()
            .find(|script| script.entity_id == entity_id)
        {
            existing.source = source.to_owned();
            existing.runtime = runtime;
        } else {
            self.scripts.push(EntityScript {
                entity_id,
                source: source.to_owned(),
                runtime,
            });
        }
        Ok(())
    }

    /// Detach a script, returning whether one was present.
    pub fn detach(&mut self, entity_id: EntityId) -> bool {
        let Some(index) = self
            .scripts
            .iter()
            .position(|script| script.entity_id == entity_id)
        else {
            return false;
        };
        self.scripts.remove(index);
        true
    }

    fn clear(&mut self) {
        self.scripts.clear();
    }

    /// Run attached update callbacks in attachment order. Scripts that fail or
    /// whose entities have been removed are detached and reported once.
    pub fn update(&mut self, engine: &mut Engine, delta_seconds: f64) -> Vec<ScriptFailure> {
        let mut failures = Vec::new();
        let mut index = 0;
        while index < self.scripts.len() {
            let entity_id = self.scripts[index].entity_id;
            let result = if engine.name(entity_id).is_err() {
                Err(ScriptError::Engine(format!(
                    "script owner entity {} no longer exists",
                    entity_id.get()
                )))
            } else {
                self.scripts[index].runtime.run_with_engine_context(
                    engine,
                    Some(entity_id),
                    delta_seconds,
                    &format!("entity://{}/update", entity_id.get()),
                    "if update ~= nil then update(engine.delta_time()) end",
                )
            };

            if let Err(error) = result {
                failures.push(ScriptFailure {
                    entity_id,
                    message: error.to_string(),
                });
                self.scripts.remove(index);
            } else {
                index += 1;
            }
        }
        // Every script in this host tick observes the same accumulated motion.
        engine.clear_mouse_motion();
        failures
    }

    pub fn script_count(&self) -> usize {
        self.scripts.len()
    }

    /// Return serializable script definitions without exposing Lua VM state.
    pub fn scene_scripts(&self) -> Vec<crate::SceneScript> {
        self.scripts
            .iter()
            .map(|script| crate::SceneScript {
                entity_id: script.entity_id.get(),
                source: script.source.clone(),
            })
            .collect()
    }

    /// Recreate fresh Lua runtimes from stored source in attachment order.
    /// Callers should use a candidate engine and script host so failed setup
    /// leaves their active scene unchanged.
    pub fn restore_scene_scripts(
        &mut self,
        engine: &mut Engine,
        definitions: &[crate::SceneScript],
    ) -> Result<(), String> {
        self.clear();
        for definition in definitions {
            self.attach(
                engine,
                EntityId::from_control(definition.entity_id),
                &definition.source,
            )
            .map_err(|error| {
                format!(
                    "script setup failed for entity {}: {error}",
                    definition.entity_id
                )
            })?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn list(&self) -> Vec<AttachedScriptInfo> {
        self.scripts
            .iter()
            .map(|script| AttachedScriptInfo {
                entity_id: script.entity_id,
                source: script.source.clone(),
            })
            .collect()
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod worker {
    use std::{
        collections::VecDeque,
        sync::{
            Arc, Mutex, MutexGuard,
            mpsc::{self, Receiver, RecvTimeoutError, SyncSender},
        },
        thread,
        time::{Duration, Instant},
    };

    use super::{
        ScriptFailure, ScriptHost, ScriptRuntime, ScriptWorkerSnapshot, SimulationAdvance,
    };
    use crate::{Engine, EntityId};

    enum Request {
        Attach {
            id: u64,
            source: String,
            reply: SyncSender<Result<(), String>>,
        },
        Detach {
            id: u64,
            reply: SyncSender<bool>,
        },
        List {
            reply: SyncSender<ScriptWorkerSnapshot>,
        },
        RunLua {
            source: String,
            reply: SyncSender<Result<usize, String>>,
        },
        Advance {
            delta_seconds: f64,
            max_steps: u32,
            reply: SyncSender<Result<SimulationAdvance, String>>,
        },
        LoadScene {
            scene_json: String,
            reply: SyncSender<Result<crate::EngineSnapshot, String>>,
        },
        ExportScene {
            reply: SyncSender<Result<String, String>>,
        },
    }

    /// Thread-safe handle to the script worker. Piccolo stays on its owning
    /// thread; callers exchange only source strings, IDs, and snapshots.
    #[derive(Clone)]
    pub struct ScriptWorker {
        requests: SyncSender<Request>,
    }

    impl ScriptWorker {
        pub fn new(engine: Arc<Mutex<Engine>>) -> Self {
            let (requests, receiver) = mpsc::sync_channel(64);
            thread::Builder::new()
                .name("gpteng-script-worker".into())
                .spawn(move || worker_loop(engine, receiver))
                .expect("failed to start the Lua script worker");
            Self { requests }
        }

        pub fn attach(&self, id: u64, source: impl Into<String>) -> Result<(), String> {
            let (reply, result) = mpsc::sync_channel(1);
            self.requests
                .send(Request::Attach {
                    id,
                    source: source.into(),
                    reply,
                })
                .map_err(|_| "Lua script worker is unavailable".to_owned())?;
            result
                .recv()
                .map_err(|_| "Lua script worker stopped before replying".to_owned())?
        }

        pub fn detach(&self, id: u64) -> Result<bool, String> {
            let (reply, result) = mpsc::sync_channel(1);
            self.requests
                .send(Request::Detach { id, reply })
                .map_err(|_| "Lua script worker is unavailable".to_owned())?;
            result
                .recv()
                .map_err(|_| "Lua script worker stopped before replying".to_owned())
        }

        pub fn list(&self) -> Result<ScriptWorkerSnapshot, String> {
            let (reply, result) = mpsc::sync_channel(1);
            self.requests
                .send(Request::List { reply })
                .map_err(|_| "Lua script worker is unavailable".to_owned())?;
            result
                .recv()
                .map_err(|_| "Lua script worker stopped before replying".to_owned())
        }

        pub fn run_lua(&self, source: impl Into<String>) -> Result<usize, String> {
            let (reply, result) = mpsc::sync_channel(1);
            self.requests
                .send(Request::RunLua {
                    source: source.into(),
                    reply,
                })
                .map_err(|_| "Lua script worker is unavailable".to_owned())?;
            result
                .recv()
                .map_err(|_| "Lua script worker stopped before replying".to_owned())?
        }

        pub fn advance_simulation(
            &self,
            delta_seconds: f64,
            max_steps: u32,
        ) -> Result<SimulationAdvance, String> {
            if !delta_seconds.is_finite()
                || delta_seconds <= 0.0
                || delta_seconds > 60.0
                || !(1..=1000).contains(&max_steps)
            {
                return Err(
                    "delta_seconds must be in (0, 60] and max_steps must be 1 through 1000".into(),
                );
            }
            let (reply, result) = mpsc::sync_channel(1);
            self.requests
                .send(Request::Advance {
                    delta_seconds,
                    max_steps,
                    reply,
                })
                .map_err(|_| "Lua script worker is unavailable".to_owned())?;
            result
                .recv()
                .map_err(|_| "Lua script worker stopped before replying".to_owned())?
        }

        /// Replace the engine scene on the worker thread and clear script VM
        /// state whose owners or captured scene handles may no longer exist.
        pub fn load_scene_json(
            &self,
            scene_json: impl Into<String>,
        ) -> Result<crate::EngineSnapshot, String> {
            let (reply, result) = mpsc::sync_channel(1);
            self.requests
                .send(Request::LoadScene {
                    scene_json: scene_json.into(),
                    reply,
                })
                .map_err(|_| "Lua script worker is unavailable".to_owned())?;
            result
                .recv()
                .map_err(|_| "Lua script worker stopped before replying".to_owned())?
        }

        /// Export engine and attached script source from one worker-thread
        /// snapshot, avoiding races with fixed-step updates or script calls.
        pub fn export_scene_json(&self) -> Result<String, String> {
            let (reply, result) = mpsc::sync_channel(1);
            self.requests
                .send(Request::ExportScene { reply })
                .map_err(|_| "Lua script worker is unavailable".to_owned())?;
            result
                .recv()
                .map_err(|_| "Lua script worker stopped before replying".to_owned())?
        }
    }

    fn worker_loop(engine: Arc<Mutex<Engine>>, requests: Receiver<Request>) {
        let mut scripts = ScriptHost::default();
        let mut lua_tool = ScriptRuntime::default_sandbox();
        let mut recent_failures = VecDeque::<ScriptFailure>::new();
        let mut last_tick = Instant::now();

        loop {
            match requests.recv_timeout(Duration::from_millis(16)) {
                Ok(request) => {
                    let now = Instant::now();
                    let elapsed = now.duration_since(last_tick).as_secs_f64();
                    tick_engine(&engine, &mut scripts, elapsed, &mut recent_failures);
                    last_tick = now;
                    process_request(
                        request,
                        &engine,
                        &mut scripts,
                        &mut lua_tool,
                        &mut recent_failures,
                    );
                }
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    let now = Instant::now();
                    let elapsed = now.duration_since(last_tick).as_secs_f64();
                    last_tick = now;
                    tick_engine(&engine, &mut scripts, elapsed, &mut recent_failures);
                }
            }
        }
    }

    fn process_request(
        request: Request,
        engine: &Arc<Mutex<Engine>>,
        scripts: &mut ScriptHost,
        lua_tool: &mut ScriptRuntime,
        recent_failures: &mut VecDeque<ScriptFailure>,
    ) {
        match request {
            Request::Attach { id, source, reply } => {
                let result = scripts
                    .attach(
                        &mut lock_engine(engine),
                        EntityId::from_control(id),
                        &source,
                    )
                    .map_err(|error| error.to_string());
                let _ = reply.send(result);
            }
            Request::Detach { id, reply } => {
                let detached = scripts.detach(EntityId::from_control(id));
                let _ = reply.send(detached);
            }
            Request::List { reply } => {
                let snapshot = ScriptWorkerSnapshot {
                    scripts: scripts.list(),
                    recent_failures: recent_failures.iter().cloned().collect(),
                    engine: lock_engine(engine).snapshot(),
                };
                let _ = reply.send(snapshot);
            }
            Request::RunLua { source, reply } => {
                let result = lua_tool
                    .run_with_engine(&mut lock_engine(engine), "mcp://run_lua", &source)
                    .map(|()| lua_tool.memory_used())
                    .map_err(|error| error.to_string());
                let _ = reply.send(result);
            }
            Request::Advance {
                delta_seconds,
                max_steps,
                reply,
            } => {
                let mut world = lock_engine(engine);
                let ticks = world.advance(delta_seconds, max_steps);
                let fixed_delta = world.fixed_delta_seconds();
                let mut failures = Vec::new();
                for _ in 0..ticks {
                    failures.extend(scripts.update(&mut world, fixed_delta));
                }
                record_failures(recent_failures, failures);
                let result = SimulationAdvance {
                    ticks_advanced: ticks,
                    engine: world.snapshot(),
                };
                let _ = reply.send(Ok(result));
            }
            Request::LoadScene { scene_json, reply } => {
                let mut world = lock_engine(engine);
                let result = (|| {
                    let document =
                        Engine::parse_scene_json(&scene_json).map_err(|error| error.to_string())?;
                    let definitions = document.scripts.clone();
                    let mut replacement =
                        Engine::from_scene(document).map_err(|error| error.to_string())?;
                    let mut replacement_scripts = ScriptHost::default();
                    replacement_scripts.restore_scene_scripts(&mut replacement, &definitions)?;
                    world.replace_scene(replacement);
                    *scripts = replacement_scripts;
                    *lua_tool = ScriptRuntime::default_sandbox();
                    recent_failures.clear();
                    Ok(world.snapshot())
                })();
                let _ = reply.send(result);
            }
            Request::ExportScene { reply } => {
                let result = lock_engine(engine)
                    .to_scene_json_with_scripts(scripts.scene_scripts())
                    .map_err(|error| error.to_string());
                let _ = reply.send(result);
            }
        }
    }

    fn tick_engine(
        engine: &Arc<Mutex<Engine>>,
        scripts: &mut ScriptHost,
        elapsed: f64,
        recent_failures: &mut VecDeque<ScriptFailure>,
    ) {
        let mut world = lock_engine(engine);
        let ticks = world.advance(elapsed, 4);
        let fixed_delta = world.fixed_delta_seconds();
        let mut failures = Vec::new();
        for _ in 0..ticks {
            failures.extend(scripts.update(&mut world, fixed_delta));
        }
        record_failures(recent_failures, failures);
    }

    fn record_failures(recent: &mut VecDeque<ScriptFailure>, failures: Vec<ScriptFailure>) {
        for failure in failures {
            recent.push_back(failure);
            while recent.len() > 128 {
                recent.pop_front();
            }
        }
    }

    fn lock_engine(engine: &Arc<Mutex<Engine>>) -> MutexGuard<'_, Engine> {
        engine
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use worker::ScriptWorker;
