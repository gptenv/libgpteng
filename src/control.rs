//! Transport-neutral commands shared by MCP, editor, game code, and hosts.
//! MCP tools will be thin, validated wrappers around this API.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

use crate::{
    engine::{
        AlphaMode3d, Camera3d, CameraProjection3d, Collider3d, ColliderShape, Engine, EngineError,
        EntityId, Lighting3d, Material3d, MeshKind, PointLight3d, RigidBody3d, RigidBodyType,
        TextureAssetId, Transform,
    },
    mesh::{GltfAssetData, MeshData, MeshLoadError},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntitySnapshot {
    pub id: u64,
    pub name: String,
    pub transform: Transform,
    pub parent_id: Option<u64>,
    pub visible: bool,
    pub effectively_visible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpawnEntity {
    pub name: String,
    pub position: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct AnimationRoot {
    id: u64,
}

#[derive(Debug, Deserialize)]
struct PlayAnimation {
    id: u64,
    clip_index: usize,
    looping: bool,
}

#[derive(Debug, Deserialize)]
struct SetAnimationSpeed {
    id: u64,
    speed: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolDefinition {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
    pub read_only: bool,
}

#[derive(Debug, Error)]
pub enum ControlError {
    #[error("unknown engine tool: {0}")]
    UnknownTool(String),
    #[error("invalid tool arguments: {0}")]
    InvalidArguments(#[from] serde_json::Error),
    #[error("invalid value: {0}")]
    InvalidValue(String),
    #[error(transparent)]
    InvalidMesh(#[from] MeshLoadError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[cfg(feature = "scripting-lua")]
    #[error("Lua script failed: {0}")]
    Script(String),
}

/// Canonical tool catalog for MCP and WebMCP. Both protocol adapters publish
/// these definitions and dispatch calls through [`call_tool`].
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_entities",
            description: "List scene entities with their names and transforms",
            input_schema: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "get_input_state",
            description: "Read the current transient keyboard and mouse state supplied by the host",
            input_schema: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "get_lighting",
            description: "Read the scene-wide directional and ambient lighting settings",
            input_schema: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "set_lighting",
            description: "Set the scene-wide directional light, linear RGB color and strength, and linear RGB ambient contribution",
            input_schema: json!({ "type": "object", "properties": { "direction": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }, "color": { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 16 }, "minItems": 3, "maxItems": 3 }, "intensity": { "type": "number", "minimum": 0, "maximum": 1000 }, "ambient": { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 1 }, "minItems": 3, "maxItems": 3 } }, "required": ["direction", "color", "intensity", "ambient"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "set_point_light",
            description: "Attach or replace a point light on a scene entity",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "color": { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 16 }, "minItems": 3, "maxItems": 3 }, "intensity": { "type": "number", "minimum": 0, "maximum": 1000 }, "radius": { "type": "number", "exclusiveMinimum": 0, "maximum": 10000 } }, "required": ["id", "color", "intensity", "radius"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "get_point_light",
            description: "Read the point light attached to an entity, if present",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "remove_point_light",
            description: "Remove an entity's point light component",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "set_visibility",
            description: "Set an entity's local render visibility; hidden ancestors hide their descendants too",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "visible": { "type": "boolean" } }, "required": ["id", "visible"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "get_visibility",
            description: "Read an entity's local and inherited render visibility",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "create_texture_asset",
            description: "Create a bounded sRGBA8 image asset from tightly packed RGBA bytes",
            input_schema: json!({ "type": "object", "properties": { "width": { "type": "integer", "minimum": 1, "maximum": 4096 }, "height": { "type": "integer", "minimum": 1, "maximum": 4096 }, "rgba8": { "type": "array", "items": { "type": "integer", "minimum": 0, "maximum": 255 }, "maxItems": 16777216, "description": "Row-major RGBA bytes; exactly width × height × 4 values" } }, "required": ["width", "height", "rgba8"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "load_texture_asset",
            description: "Decode a bounded PNG, JPEG, WebP, BMP, GIF, TIFF, or PNM byte stream into an sRGBA8 texture asset",
            input_schema: json!({ "type": "object", "properties": { "encoded_bytes": { "type": "array", "items": { "type": "integer", "minimum": 0, "maximum": 255 }, "maxItems": 16777216, "description": "Encoded image file as byte values; maximum 16 MiB. Animated formats use the first frame." } }, "required": ["encoded_bytes"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "set_base_color_texture",
            description: "Assign a texture asset to a mesh entity, or clear it with null",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "texture_asset_id": { "anyOf": [{ "type": "integer", "minimum": 1 }, { "type": "null" }] } }, "required": ["id", "texture_asset_id"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "get_base_color_texture",
            description: "Read the assigned base-color texture asset ID from a mesh entity",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "export_scene",
            description: "Export scene state and attached Lua source as a versioned JSON document",
            input_schema: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "load_scene",
            description: "Validate a versioned scene and initialize its saved Lua scripts before replacing the active scene",
            input_schema: json!({ "type": "object", "properties": { "scene_json": { "type": "string", "description": "Scene JSON returned by export_scene" } }, "required": ["scene_json"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "spawn_entity",
            description: "Create a named scene entity at a world-space position",
            input_schema: json!({ "type": "object", "properties": { "name": { "type": "string" }, "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 } }, "required": ["name", "position"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "spawn_primitive",
            description: "Create a cube, sphere, or plane mesh in the scene",
            input_schema: json!({ "type": "object", "properties": { "name": { "type": "string" }, "primitive": { "type": "string", "enum": ["cube", "sphere", "plane"] }, "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }, "base_color": { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 1 }, "minItems": 4, "maxItems": 4 }, "roughness": { "type": "number", "minimum": 0, "maximum": 1 }, "metallic": { "type": "number", "minimum": 0, "maximum": 1 } }, "required": ["name", "primitive", "position"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "load_obj_mesh",
            description: "Load a bounded Wavefront OBJ mesh and spawn it into the scene",
            input_schema: json!({ "type": "object", "properties": { "name": { "type": "string" }, "source": { "type": "string", "description": "OBJ text with positions, optional normals, and polygon faces" }, "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }, "base_color": { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 1 }, "minItems": 4, "maxItems": 4 }, "roughness": { "type": "number", "minimum": 0, "maximum": 1 }, "metallic": { "type": "number", "minimum": 0, "maximum": 1 } }, "required": ["name", "source", "position"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "load_gltf_mesh",
            description: "Load a bounded glTF 2.0 or GLB scene with node hierarchy and animation clips",
            input_schema: json!({ "type": "object", "properties": { "name": { "type": "string" }, "source_base64": { "type": "string", "maxLength": 44739244, "description": "Base64-encoded glTF JSON or binary GLB bytes. Embed buffers as data URIs; external resources are rejected." }, "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }, "base_color": { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 1 }, "minItems": 4, "maxItems": 4 }, "roughness": { "type": "number", "minimum": 0, "maximum": 1 }, "metallic": { "type": "number", "minimum": 0, "maximum": 1 } }, "required": ["name", "source_base64", "position"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "list_gltf_animations",
            description: "List animation clips and playback state for an imported glTF scene root",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "play_gltf_animation",
            description: "Play an imported glTF animation clip by index, optionally looping it",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "clip_index": { "type": "integer", "minimum": 0 }, "looping": { "type": "boolean" } }, "required": ["id", "clip_index", "looping"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "stop_gltf_animation",
            description: "Stop an imported glTF animation player",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "set_gltf_animation_speed",
            description: "Set glTF animation playback speed from -100 to 100; negative values play backward",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "speed": { "type": "number", "minimum": -100, "maximum": 100 } }, "required": ["id", "speed"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "set_position",
            description: "Set a scene entity's world-space position",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 } }, "required": ["id", "position"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "set_parent",
            description: "Set or clear an entity's parent while preserving its local transform",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "parent_id": { "anyOf": [{ "type": "integer", "minimum": 1 }, { "type": "null" }] } }, "required": ["id", "parent_id"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "get_world_transform",
            description: "Get an entity's composed world-space transform",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "set_transform",
            description: "Set an entity's position, quaternion rotation, and scale",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }, "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4, "description": "Quaternion [x, y, z, w]" }, "scale": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 } }, "required": ["id", "position", "rotation", "scale"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "get_material",
            description: "Get an entity's base color, PBR response, and alpha mode",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "set_material",
            description: "Set an entity's base color, PBR response, and alpha mode",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "base_color": { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 1 }, "minItems": 4, "maxItems": 4 }, "roughness": { "type": "number", "minimum": 0, "maximum": 1 }, "metallic": { "type": "number", "minimum": 0, "maximum": 1 }, "alpha_mode": { "type": "string", "enum": ["auto", "opaque", "mask", "blend"] }, "alpha_cutoff": { "type": "number", "minimum": 0, "maximum": 1 } }, "required": ["id", "base_color"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "delete_entity",
            description: "Delete a scene entity",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "advance_simulation",
            description: "Advance the engine's fixed-step simulation by elapsed seconds",
            input_schema: json!({ "type": "object", "properties": { "delta_seconds": { "type": "number", "exclusiveMinimum": 0, "maximum": 60 }, "max_steps": { "type": "integer", "minimum": 1, "maximum": 1000 } }, "required": ["delta_seconds"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "add_rigid_body",
            description: "Add or replace an entity's dynamic, kinematic, or static rigid body",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "body_type": { "type": "string", "enum": ["dynamic", "kinematic", "static"] }, "mass": { "type": "number", "exclusiveMinimum": 0 }, "gravity_scale": { "type": "number" }, "linear_damping": { "type": "number", "minimum": 0 } }, "required": ["id"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "get_rigid_body",
            description: "Get an entity's rigid body state or null if it has no body",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "remove_rigid_body",
            description: "Remove an entity's rigid body component",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "set_velocity",
            description: "Set the linear velocity of an entity with a rigid body",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "velocity": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 } }, "required": ["id", "velocity"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "apply_force",
            description: "Apply a force to an entity's dynamic rigid body for the next fixed step",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "force": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 } }, "required": ["id", "force"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "get_gravity",
            description: "Get global gravity acceleration in world units per second squared",
            input_schema: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "set_gravity",
            description: "Set global gravity acceleration in world units per second squared",
            input_schema: json!({ "type": "object", "properties": { "gravity": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 } }, "required": ["gravity"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "set_collider",
            description: "Add or replace a sphere or box collider on an entity",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "shape": { "type": "string", "enum": ["sphere", "box"] }, "radius": { "type": "number", "exclusiveMinimum": 0 }, "half_extents": { "type": "array", "items": { "type": "number", "exclusiveMinimum": 0 }, "minItems": 3, "maxItems": 3 }, "restitution": { "type": "number", "minimum": 0, "maximum": 1 }, "friction": { "type": "number", "minimum": 0, "maximum": 1 } }, "required": ["id", "shape"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "get_collider",
            description: "Get an entity's collider or null if it has none",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "remove_collider",
            description: "Remove an entity's collider component",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: false,
        },
        ToolDefinition {
            name: "get_engine_info",
            description: "Get the entity count and fixed-step simulation counters",
            input_schema: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "get_camera",
            description: "Get the active scene camera and perspective settings",
            input_schema: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
            read_only: true,
        },
        ToolDefinition {
            name: "set_camera",
            description: "Set the active scene camera position, target, up vector, projection, and clear color",
            input_schema: json!({ "type": "object", "properties": { "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }, "target": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }, "up": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }, "vertical_fov_degrees": { "type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 180 }, "projection": { "type": "string", "enum": ["perspective", "orthographic"] }, "orthographic_vertical_size": { "type": "number", "exclusiveMinimum": 0 }, "near": { "type": "number", "exclusiveMinimum": 0 }, "far": { "type": "number", "exclusiveMinimum": 0 }, "clear_color": { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 1 }, "minItems": 4, "maxItems": 4 } }, "required": ["position", "target", "up", "vertical_fov_degrees", "near", "far", "clear_color"], "additionalProperties": false }),
            read_only: false,
        },
        #[cfg(feature = "scripting-lua")]
        ToolDefinition {
            name: "attach_script",
            description: "Attach and initialize a persistent bounded Lua script on an entity",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 }, "source": { "type": "string" } }, "required": ["id", "source"], "additionalProperties": false }),
            read_only: false,
        },
        #[cfg(feature = "scripting-lua")]
        ToolDefinition {
            name: "detach_script",
            description: "Stop and remove an entity's attached Lua script",
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer", "minimum": 1 } }, "required": ["id"], "additionalProperties": false }),
            read_only: false,
        },
        #[cfg(feature = "scripting-lua")]
        ToolDefinition {
            name: "list_scripts",
            description: "List persistent Lua scripts and recent script failures",
            input_schema: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
            read_only: true,
        },
        #[cfg(feature = "scripting-lua")]
        ToolDefinition {
            name: "run_lua",
            description: "Run a Lua chunk in the persistent bounded MCP sandbox with access to the engine scene",
            input_schema: json!({ "type": "object", "properties": { "source": { "type": "string", "description": "Lua source code to execute" } }, "required": ["source"], "additionalProperties": false }),
            read_only: false,
        },
    ]
}

pub fn call_tool(engine: &mut Engine, name: &str, arguments: Value) -> Result<Value, ControlError> {
    let mut control = EngineControl::new(engine);
    match name {
        "list_entities" => Ok(serde_json::to_value(control.list_entities())?),
        "get_input_state" => Ok(serde_json::to_value(control.engine.input())?),
        "get_lighting" => Ok(serde_json::to_value(control.engine.lighting())?),
        "set_lighting" => {
            let request: SetLighting = serde_json::from_value(arguments)?;
            let lighting = Lighting3d {
                direction: Vec3::from_array(request.direction),
                color: request.color,
                intensity: request.intensity,
                ambient: request.ambient,
            };
            control.engine.set_lighting(lighting)?;
            Ok(serde_json::to_value(control.engine.lighting())?)
        }
        "set_point_light" => {
            let request: SetPointLight = serde_json::from_value(arguments)?;
            control.engine.set_point_light(
                EntityId::from_control(request.id),
                Some(PointLight3d {
                    color: request.color,
                    intensity: request.intensity,
                    radius: request.radius,
                }),
            )?;
            Ok(json!({ "ok": true, "id": request.id }))
        }
        "get_point_light" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            Ok(serde_json::to_value(
                control
                    .engine
                    .point_light(EntityId::from_control(request.id))?,
            )?)
        }
        "remove_point_light" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            let existed = control
                .engine
                .point_light(EntityId::from_control(request.id))?
                .is_some();
            control
                .engine
                .set_point_light(EntityId::from_control(request.id), None)?;
            Ok(json!({ "ok": true, "id": request.id, "removed": existed }))
        }
        "set_visibility" => {
            let request: SetVisibility = serde_json::from_value(arguments)?;
            control
                .engine
                .set_visibility(EntityId::from_control(request.id), request.visible)?;
            Ok(json!({ "ok": true, "id": request.id, "visible": request.visible }))
        }
        "get_visibility" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            let id = EntityId::from_control(request.id);
            Ok(json!({
                "id": request.id,
                "visible": control.engine.visibility(id)?,
                "effectively_visible": control.engine.is_visible(id)?
            }))
        }
        "create_texture_asset" => {
            let request: CreateTextureAsset = serde_json::from_value(arguments)?;
            let id =
                control
                    .engine
                    .add_texture_asset(request.width, request.height, request.rgba8)?;
            Ok(
                json!({ "texture_asset_id": id.get(), "width": request.width, "height": request.height }),
            )
        }
        "load_texture_asset" => {
            let request: LoadTextureAsset = serde_json::from_value(arguments)?;
            let texture_id = control
                .engine
                .add_texture_asset_from_encoded(&request.encoded_bytes)?;
            let texture = control
                .engine
                .texture_asset(texture_id)
                .expect("new texture exists");
            Ok(
                json!({ "texture_asset_id": texture_id.get(), "width": texture.width, "height": texture.height }),
            )
        }
        "set_base_color_texture" => {
            let request: SetBaseColorTexture = serde_json::from_value(arguments)?;
            let texture = request.texture_asset_id.map(TextureAssetId::from_control);
            control
                .engine
                .set_base_color_texture(EntityId::from_control(request.id), texture)?;
            Ok(
                json!({ "ok": true, "id": request.id, "texture_asset_id": request.texture_asset_id }),
            )
        }
        "get_base_color_texture" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            let texture_asset_id = control
                .engine
                .base_color_texture(EntityId::from_control(request.id))?
                .map(TextureAssetId::get);
            Ok(json!({ "id": request.id, "texture_asset_id": texture_asset_id }))
        }
        "export_scene" => Ok(json!({ "scene_json": control.engine.to_scene_json()? })),
        "load_scene" => {
            let request: LoadScene = serde_json::from_value(arguments)?;
            control.engine.load_scene_json(&request.scene_json)?;
            Ok(json!({
                "ok": true,
                "engine": control.engine.snapshot(),
                "mesh_asset_count": control.engine.mesh_asset_count()
            }))
        }
        "spawn_entity" => {
            let request = serde_json::from_value(arguments)?;
            Ok(serde_json::to_value(control.spawn_entity(request)?)?)
        }
        "spawn_primitive" => {
            let request: SpawnPrimitive = serde_json::from_value(arguments)?;
            Ok(serde_json::to_value(control.spawn_primitive(request)?)?)
        }
        "load_obj_mesh" => {
            let request: LoadObjMesh = serde_json::from_value(arguments)?;
            let (entity, asset_id) = control.load_obj_mesh(request)?;
            Ok(json!({ "entity": entity, "mesh_asset_id": asset_id.get() }))
        }
        "load_gltf_mesh" => {
            let request: LoadGltfMesh = serde_json::from_value(arguments)?;
            let (entity, node_entity_ids, mesh_asset_ids, primitive_entity_ids, animations) =
                control.load_gltf_mesh(request)?;
            Ok(json!({
                "entity": entity,
                "node_entity_ids": node_entity_ids,
                "mesh_asset_ids": mesh_asset_ids,
                "primitive_entity_ids": primitive_entity_ids,
                "animations": animations
            }))
        }
        "list_gltf_animations" => {
            let request: AnimationRoot = serde_json::from_value(arguments)?;
            let root = EntityId::from_control(request.id);
            Ok(json!({
                "id": request.id,
                "animations": control.engine.gltf_animation_names(root)?,
                "state": control.engine.gltf_animation_state(root)?
            }))
        }
        "play_gltf_animation" => {
            let request: PlayAnimation = serde_json::from_value(arguments)?;
            let state = control.engine.play_gltf_animation(
                EntityId::from_control(request.id),
                request.clip_index,
                request.looping,
            )?;
            Ok(serde_json::to_value(state)?)
        }
        "stop_gltf_animation" => {
            let request: AnimationRoot = serde_json::from_value(arguments)?;
            let state = control
                .engine
                .stop_gltf_animation(EntityId::from_control(request.id))?;
            Ok(serde_json::to_value(state)?)
        }
        "set_gltf_animation_speed" => {
            let request: SetAnimationSpeed = serde_json::from_value(arguments)?;
            let state = control
                .engine
                .set_gltf_animation_speed(EntityId::from_control(request.id), request.speed)?;
            Ok(serde_json::to_value(state)?)
        }
        "set_position" => {
            let request: SetPosition = serde_json::from_value(arguments)?;
            control.set_position(request.id, request.position)?;
            Ok(json!({ "ok": true }))
        }
        "set_parent" => {
            let request: SetParent = serde_json::from_value(arguments)?;
            control.set_parent(request.id, request.parent_id)?;
            Ok(json!({ "ok": true }))
        }
        "get_world_transform" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            Ok(serde_json::to_value(
                control
                    .engine
                    .world_transform(EntityId::from_control(request.id))?,
            )?)
        }
        "set_transform" => {
            let request: SetTransform = serde_json::from_value(arguments)?;
            control.set_transform(request)?;
            Ok(json!({ "ok": true }))
        }
        "set_material" => {
            let request: SetMaterial = serde_json::from_value(arguments)?;
            control.set_material(
                request.id,
                request.base_color,
                request.roughness,
                request.metallic,
                request.alpha_mode,
                request.alpha_cutoff,
            )?;
            Ok(json!({ "ok": true }))
        }
        "get_material" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            Ok(serde_json::to_value(
                control
                    .engine
                    .material(EntityId::from_control(request.id))?,
            )?)
        }
        "delete_entity" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            control.delete_entity(request.id)?;
            Ok(json!({ "ok": true }))
        }
        "advance_simulation" => {
            let request: AdvanceSimulation = serde_json::from_value(arguments)?;
            let ticks = control.advance_simulation(request.delta_seconds, request.max_steps)?;
            Ok(json!({ "ticks_advanced": ticks, "engine": control.engine.snapshot() }))
        }
        "add_rigid_body" => {
            let request: AddRigidBody = serde_json::from_value(arguments)?;
            let body_type = match request.body_type.as_deref().unwrap_or("dynamic") {
                "dynamic" => RigidBodyType::Dynamic,
                "kinematic" => RigidBodyType::Kinematic,
                "static" => RigidBodyType::Static,
                other => {
                    return Err(ControlError::InvalidValue(format!(
                        "unknown rigid body type {other}"
                    )));
                }
            };
            let defaults = RigidBody3d::default();
            let body = RigidBody3d {
                body_type,
                mass: request.mass.unwrap_or(defaults.mass),
                gravity_scale: request.gravity_scale.unwrap_or(defaults.gravity_scale),
                linear_damping: request.linear_damping.unwrap_or(defaults.linear_damping),
                ..defaults
            };
            control
                .engine
                .add_rigid_body(EntityId::from_control(request.id), body)?;
            Ok(json!({ "ok": true }))
        }
        "get_rigid_body" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            Ok(serde_json::to_value(
                control
                    .engine
                    .rigid_body(EntityId::from_control(request.id))?,
            )?)
        }
        "remove_rigid_body" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            Ok(
                json!({ "removed": control.engine.remove_rigid_body(EntityId::from_control(request.id))? }),
            )
        }
        "set_velocity" => {
            let request: VectorRequest = serde_json::from_value(arguments)?;
            control.engine.set_velocity(
                EntityId::from_control(request.id),
                Vec3::from_array(request.vector),
            )?;
            Ok(json!({ "ok": true }))
        }
        "apply_force" => {
            let request: VectorRequest = serde_json::from_value(arguments)?;
            control.engine.apply_force(
                EntityId::from_control(request.id),
                Vec3::from_array(request.vector),
            )?;
            Ok(json!({ "ok": true }))
        }
        "get_gravity" => Ok(serde_json::to_value(control.engine.gravity())?),
        "set_gravity" => {
            let request: SetGravity = serde_json::from_value(arguments)?;
            control
                .engine
                .set_gravity(Vec3::from_array(request.gravity))?;
            Ok(serde_json::to_value(control.engine.gravity())?)
        }
        "set_collider" => {
            let request: SetCollider = serde_json::from_value(arguments)?;
            let shape = match request.shape.as_str() {
                "sphere" => ColliderShape::Sphere {
                    radius: request.radius.unwrap_or(0.5),
                },
                "box" => ColliderShape::Box {
                    half_extents: Vec3::from_array(request.half_extents.unwrap_or([0.5; 3])),
                },
                other => {
                    return Err(ControlError::InvalidValue(format!(
                        "unknown collider shape {other}"
                    )));
                }
            };
            control.engine.set_collider(
                EntityId::from_control(request.id),
                Collider3d {
                    shape,
                    restitution: request.restitution.unwrap_or(0.0),
                    friction: request.friction.unwrap_or(0.5),
                },
            )?;
            Ok(json!({ "ok": true }))
        }
        "get_collider" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            Ok(serde_json::to_value(
                control
                    .engine
                    .collider(EntityId::from_control(request.id))?,
            )?)
        }
        "remove_collider" => {
            let request: EntityArgument = serde_json::from_value(arguments)?;
            Ok(
                json!({ "removed": control.engine.remove_collider(EntityId::from_control(request.id))? }),
            )
        }
        "get_engine_info" => Ok(serde_json::to_value(control.engine.snapshot())?),
        "get_camera" => Ok(serde_json::to_value(control.get_camera())?),
        "set_camera" => {
            let request: SetCamera = serde_json::from_value(arguments)?;
            control.set_camera(request)?;
            Ok(serde_json::to_value(control.get_camera())?)
        }
        #[cfg(feature = "scripting-lua")]
        "run_lua" => {
            let request: RunLua = serde_json::from_value(arguments)?;
            let mut runtime = crate::scripting::ScriptRuntime::default_sandbox();
            runtime
                .run_with_engine(engine, "mcp://run_lua", &request.source)
                .map_err(|error| ControlError::Script(error.to_string()))?;
            Ok(json!({ "ok": true, "memory_bytes": runtime.memory_used() }))
        }
        _ => Err(ControlError::UnknownTool(name.to_owned())),
    }
}

#[derive(Deserialize)]
struct LoadTextureAsset {
    encoded_bytes: Vec<u8>,
}

#[derive(Deserialize)]
struct LoadScene {
    scene_json: String,
}

#[cfg(feature = "scripting-lua")]
#[derive(Deserialize)]
struct RunLua {
    source: String,
}

#[derive(Deserialize)]
struct SetPosition {
    id: u64,
    position: [f32; 3],
}

#[derive(Deserialize)]
struct SetVisibility {
    id: u64,
    visible: bool,
}

#[derive(Deserialize)]
struct SetLighting {
    direction: [f32; 3],
    color: [f32; 3],
    intensity: f32,
    ambient: [f32; 3],
}

#[derive(Deserialize)]
struct SetPointLight {
    id: u64,
    color: [f32; 3],
    intensity: f32,
    radius: f32,
}

#[derive(Deserialize)]
struct CreateTextureAsset {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

#[derive(Deserialize)]
struct SetBaseColorTexture {
    id: u64,
    texture_asset_id: Option<u64>,
}

#[derive(Deserialize)]
struct SetParent {
    id: u64,
    parent_id: Option<u64>,
}

#[derive(Deserialize)]
struct SpawnPrimitive {
    name: String,
    primitive: String,
    position: [f32; 3],
    base_color: Option<[f32; 4]>,
    roughness: Option<f32>,
    metallic: Option<f32>,
}

#[derive(Deserialize)]
struct LoadObjMesh {
    name: String,
    source: String,
    position: [f32; 3],
    base_color: Option<[f32; 4]>,
    roughness: Option<f32>,
    metallic: Option<f32>,
}

#[derive(Deserialize)]
struct LoadGltfMesh {
    name: String,
    source_base64: String,
    position: [f32; 3],
    base_color: Option<[f32; 4]>,
    roughness: Option<f32>,
    metallic: Option<f32>,
}

#[derive(Deserialize)]
struct SetTransform {
    id: u64,
    position: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

#[derive(Deserialize)]
struct SetMaterial {
    id: u64,
    base_color: [f32; 4],
    roughness: Option<f32>,
    metallic: Option<f32>,
    alpha_mode: Option<String>,
    alpha_cutoff: Option<f32>,
}

#[derive(Deserialize)]
struct AdvanceSimulation {
    delta_seconds: f64,
    max_steps: Option<u32>,
}

#[derive(Deserialize)]
struct AddRigidBody {
    id: u64,
    body_type: Option<String>,
    mass: Option<f32>,
    gravity_scale: Option<f32>,
    linear_damping: Option<f32>,
}

#[derive(Deserialize)]
struct VectorRequest {
    id: u64,
    #[serde(alias = "velocity", alias = "force")]
    vector: [f32; 3],
}

#[derive(Deserialize)]
struct SetGravity {
    gravity: [f32; 3],
}

#[derive(Deserialize)]
struct SetCollider {
    id: u64,
    shape: String,
    radius: Option<f32>,
    half_extents: Option<[f32; 3]>,
    restitution: Option<f32>,
    friction: Option<f32>,
}

#[derive(Deserialize)]
struct SetCamera {
    position: [f32; 3],
    target: [f32; 3],
    up: [f32; 3],
    vertical_fov_degrees: f32,
    #[serde(default)]
    projection: Option<String>,
    #[serde(default)]
    orthographic_vertical_size: Option<f32>,
    near: f32,
    far: f32,
    clear_color: [f32; 4],
}

#[derive(Deserialize)]
struct EntityArgument {
    id: u64,
}

/// Engine operations exposed to remote control clients. Keeping this interface
/// transport-neutral lets native MCP and browser-hosted MCP adapters drive the
/// same runtime without placing protocol code in the simulation core.
pub struct EngineControl<'a> {
    engine: &'a mut Engine,
}

impl<'a> EngineControl<'a> {
    pub fn new(engine: &'a mut Engine) -> Self {
        Self { engine }
    }

    pub fn list_entities(&self) -> Vec<EntitySnapshot> {
        self.engine
            .entities()
            .map(|(id, name, transform)| EntitySnapshot {
                id: id.get(),
                name,
                transform,
                parent_id: self.engine.parent(id).ok().flatten().map(EntityId::get),
                visible: self.engine.visibility(id).unwrap_or(true),
                effectively_visible: self.engine.is_visible(id).unwrap_or(true),
            })
            .collect()
    }

    pub fn get_camera(&self) -> Camera3d {
        self.engine.camera()
    }

    pub fn get_material(&self, id: u64) -> Result<Material3d, EngineError> {
        self.engine.material(EntityId::from_control(id))
    }

    fn set_camera(&mut self, request: SetCamera) -> Result<(), EngineError> {
        let projection = match request.projection.as_deref().unwrap_or("perspective") {
            "perspective" => CameraProjection3d::Perspective,
            "orthographic" => CameraProjection3d::Orthographic {
                vertical_size: request.orthographic_vertical_size.unwrap_or(10.0),
            },
            _ => return Err(EngineError::InvalidCamera),
        };
        self.engine.set_camera(Camera3d {
            position: Vec3::from_array(request.position),
            target: Vec3::from_array(request.target),
            up: Vec3::from_array(request.up),
            projection,
            vertical_fov_radians: request.vertical_fov_degrees.to_radians(),
            near: request.near,
            far: request.far,
            clear_color: request.clear_color,
        })
    }

    pub fn spawn_entity(&mut self, request: SpawnEntity) -> Result<EntitySnapshot, ControlError> {
        let transform = Transform {
            translation: Vec3::from_array(request.position),
            ..Transform::default()
        };
        let id = self.engine.spawn(request.name.clone(), transform)?;
        Ok(EntitySnapshot {
            id: id.get(),
            name: request.name,
            transform,
            parent_id: None,
            visible: true,
            effectively_visible: true,
        })
    }

    fn spawn_primitive(&mut self, request: SpawnPrimitive) -> Result<EntitySnapshot, ControlError> {
        if request.position.iter().any(|value| !value.is_finite()) {
            return Err(ControlError::InvalidValue(
                "position values must be finite".into(),
            ));
        }
        let mesh = match request.primitive.as_str() {
            "cube" => MeshKind::Cube,
            "sphere" => MeshKind::Sphere,
            "plane" => MeshKind::Plane,
            _ => {
                return Err(ControlError::InvalidValue(format!(
                    "unknown primitive {}; expected cube, sphere, or plane",
                    request.primitive
                )));
            }
        };
        let material = make_material(request.base_color, request.roughness, request.metallic)?;
        let transform = Transform {
            translation: Vec3::from_array(request.position),
            ..Transform::default()
        };
        let id = self
            .engine
            .spawn_mesh(request.name.clone(), transform, mesh, material)?;
        Ok(EntitySnapshot {
            id: id.get(),
            name: request.name,
            transform,
            parent_id: None,
            visible: true,
            effectively_visible: true,
        })
    }

    fn load_obj_mesh(
        &mut self,
        request: LoadObjMesh,
    ) -> Result<(EntitySnapshot, crate::mesh::MeshAssetId), ControlError> {
        if request.position.iter().any(|value| !value.is_finite()) {
            return Err(ControlError::InvalidValue(
                "position values must be finite".into(),
            ));
        }
        let material = make_material(request.base_color, request.roughness, request.metallic)?;
        let mesh = MeshData::from_obj(&request.source)?;
        let transform = Transform {
            translation: Vec3::from_array(request.position),
            ..Transform::default()
        };
        let asset_id = self.engine.add_mesh_asset(mesh);
        let id =
            self.engine
                .spawn_mesh_asset(request.name.clone(), transform, asset_id, material)?;
        Ok((
            EntitySnapshot {
                id: id.get(),
                name: request.name,
                transform,
                parent_id: None,
                visible: true,
                effectively_visible: true,
            },
            asset_id,
        ))
    }

    fn load_gltf_mesh(
        &mut self,
        request: LoadGltfMesh,
    ) -> Result<(EntitySnapshot, Vec<u64>, Vec<u64>, Vec<u64>, Vec<String>), ControlError> {
        if request.position.iter().any(|value| !value.is_finite()) {
            return Err(ControlError::InvalidValue(
                "position values must be finite".into(),
            ));
        }
        if request.source_base64.len() > 44_739_244 {
            return Err(ControlError::InvalidValue(
                "base64 glTF source exceeds the input limit".into(),
            ));
        }
        let source = STANDARD
            .decode(&request.source_base64)
            .map_err(|error| ControlError::InvalidValue(format!("invalid base64: {error}")))?;
        let asset = GltfAssetData::from_gltf(&source)?;
        make_material(request.base_color, request.roughness, request.metallic)?;
        let instance = self.engine.spawn_gltf_asset(
            request.name.clone(),
            Vec3::from_array(request.position),
            asset,
        )?;
        for id in &instance.primitives {
            let mut material = self.engine.material(*id)?;
            if let Some(base_color) = request.base_color {
                material.base_color = base_color;
            }
            if let Some(roughness) = request.roughness {
                material.roughness = roughness;
            }
            if let Some(metallic) = request.metallic {
                material.metallic = metallic;
            }
            self.engine.set_material(*id, material)?;
        }
        let transform = self.engine.transform(instance.root)?;
        Ok((
            EntitySnapshot {
                id: instance.root.get(),
                name: request.name,
                transform,
                parent_id: None,
                visible: true,
                effectively_visible: true,
            },
            instance.nodes.iter().map(|id| id.get()).collect(),
            instance.mesh_assets.iter().map(|id| id.get()).collect(),
            instance.primitives.iter().map(|id| id.get()).collect(),
            instance.animations,
        ))
    }

    pub fn set_position(&mut self, id: u64, position: [f32; 3]) -> Result<(), EngineError> {
        let id = EntityId::from_control(id);
        let mut transform = self.engine.transform(id)?;
        transform.translation = Vec3::from_array(position);
        self.engine.set_transform(id, transform)
    }

    pub fn set_parent(&mut self, id: u64, parent_id: Option<u64>) -> Result<(), EngineError> {
        self.engine.set_parent(
            EntityId::from_control(id),
            parent_id.map(EntityId::from_control),
        )
    }

    fn set_transform(&mut self, request: SetTransform) -> Result<(), ControlError> {
        let rotation = Quat::from_array(request.rotation);
        if request
            .position
            .iter()
            .chain(request.scale.iter())
            .any(|value| !value.is_finite())
            || request.rotation.iter().any(|value| !value.is_finite())
            || !rotation.length_squared().is_finite()
            || rotation.length_squared() <= f32::EPSILON
        {
            return Err(ControlError::InvalidValue(
                "transform values must be finite and rotation must be nonzero".into(),
            ));
        }
        self.engine.set_transform(
            EntityId::from_control(request.id),
            Transform {
                translation: Vec3::from_array(request.position),
                rotation: rotation.normalize(),
                scale: Vec3::from_array(request.scale),
            },
        )?;
        Ok(())
    }

    fn set_material(
        &mut self,
        id: u64,
        base_color: [f32; 4],
        roughness: Option<f32>,
        metallic: Option<f32>,
        alpha_mode: Option<String>,
        alpha_cutoff: Option<f32>,
    ) -> Result<(), EngineError> {
        let id = EntityId::from_control(id);
        let current = self.engine.material(id)?;
        let alpha_mode = match alpha_mode.as_deref() {
            None => current.alpha_mode,
            Some("auto") => AlphaMode3d::Auto,
            Some("opaque") => AlphaMode3d::Opaque,
            Some("mask") => AlphaMode3d::Mask,
            Some("blend") => AlphaMode3d::Blend,
            Some(_) => return Err(EngineError::InvalidMaterial),
        };
        self.engine.set_material(
            id,
            Material3d {
                base_color,
                roughness: roughness.unwrap_or(current.roughness),
                metallic: metallic.unwrap_or(current.metallic),
                alpha_mode,
                alpha_cutoff: alpha_cutoff.unwrap_or(current.alpha_cutoff),
            },
        )
    }

    fn advance_simulation(
        &mut self,
        delta_seconds: f64,
        max_steps: Option<u32>,
    ) -> Result<u32, ControlError> {
        if !delta_seconds.is_finite() || delta_seconds <= 0.0 || delta_seconds > 60.0 {
            return Err(ControlError::InvalidValue(
                "delta_seconds must be finite and between 0 and 60".into(),
            ));
        }
        let max_steps = max_steps.unwrap_or(4);
        if !(1..=1000).contains(&max_steps) {
            return Err(ControlError::InvalidValue(
                "max_steps must be between 1 and 1000".into(),
            ));
        }
        Ok(self.engine.advance(delta_seconds, max_steps))
    }

    pub fn delete_entity(&mut self, id: u64) -> Result<(), EngineError> {
        self.engine.despawn(EntityId::from_control(id))
    }
}

fn make_material(
    base_color: Option<[f32; 4]>,
    roughness: Option<f32>,
    metallic: Option<f32>,
) -> Result<Material3d, ControlError> {
    let defaults = Material3d::default();
    let material = Material3d {
        base_color: base_color.unwrap_or(defaults.base_color),
        roughness: roughness.unwrap_or(defaults.roughness),
        metallic: metallic.unwrap_or(defaults.metallic),
        ..defaults
    };
    if material
        .base_color
        .iter()
        .chain([&material.roughness, &material.metallic])
        .chain([&material.alpha_cutoff])
        .any(|channel| !channel.is_finite() || !(0.0..=1.0).contains(channel))
    {
        return Err(ControlError::InvalidValue(
            "base_color, roughness, and metallic values must be between 0 and 1".into(),
        ));
    }
    Ok(material)
}

impl EntityId {
    pub(crate) const fn from_control(id: u64) -> Self {
        Self(id)
    }
}

impl TextureAssetId {
    pub(crate) const fn from_control(id: u64) -> Self {
        Self(id)
    }
}
