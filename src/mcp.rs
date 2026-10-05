//! Native MCP server exposing engine inspection and scene mutation tools.
//! The same `EngineControl` API is available to non-MCP callers and WASM hosts.

use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
};
use rmcp::{
    ServiceExt,
    handler::server::wrapper::Parameters,
    schemars, tool, tool_router,
    transport::{
        stdio,
        streamable_http_server::{
            StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
        },
    },
};
use serde_json::Value;

#[cfg(feature = "scripting-lua")]
use crate::{ScriptWorker, control::ControlError};
use crate::{
    control::{call_tool, tool_definitions},
    engine::{Engine, EngineConfig},
};

#[derive(Clone)]
pub struct EngineMcpServer {
    engine: Arc<Mutex<Engine>>,
    #[cfg(feature = "scripting-lua")]
    scripts: ScriptWorker,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SpawnEntityParams {
    /// Human-readable name for the new scene entity.
    pub name: String,
    /// World-space position as [x, y, z].
    pub position: [f32; 3],
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SpawnPrimitiveParams {
    /// Human-readable name for the new scene entity.
    pub name: String,
    /// Built-in mesh kind: cube, sphere, or plane.
    pub primitive: String,
    /// World-space position [x, y, z].
    pub position: [f32; 3],
    /// Optional linear RGBA base color. Defaults to the engine blue material.
    pub base_color: Option<[f32; 4]>,
    /// Optional roughness from 0 (smooth) through 1 (rough).
    pub roughness: Option<f32>,
    /// Optional metallic response from 0 (dielectric) through 1 (metal).
    pub metallic: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct LoadObjMeshParams {
    /// Human-readable name for the new scene entity.
    pub name: String,
    /// Wavefront OBJ text containing positions and polygon faces.
    pub source: String,
    /// World-space position [x, y, z].
    pub position: [f32; 3],
    /// Optional linear RGBA base color.
    pub base_color: Option<[f32; 4]>,
    /// Optional roughness from 0 (smooth) through 1 (rough).
    pub roughness: Option<f32>,
    /// Optional metallic response from 0 (dielectric) through 1 (metal).
    pub metallic: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct LoadGltfMeshParams {
    /// Human-readable name for the new scene entity.
    pub name: String,
    /// Base64-encoded glTF JSON or binary GLB bytes. Buffers must be embedded.
    pub source_base64: String,
    /// World-space position [x, y, z].
    pub position: [f32; 3],
    /// Optional linear RGBA base color.
    pub base_color: Option<[f32; 4]>,
    /// Optional roughness from 0 (smooth) through 1 (rough).
    pub roughness: Option<f32>,
    /// Optional metallic response from 0 (dielectric) through 1 (metal).
    pub metallic: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct PlayGltfAnimationParams {
    /// ID of the imported glTF scene root.
    pub id: u64,
    /// Zero-based animation clip index returned by list_gltf_animations.
    pub clip_index: usize,
    /// Whether playback should wrap at the clip end.
    pub looping: bool,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetGltfAnimationSpeedParams {
    /// ID of the imported glTF scene root.
    pub id: u64,
    /// Playback speed from -100 to 100; negative values play backward.
    pub speed: f32,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CreateTextureAssetParams {
    /// Image width in pixels (maximum 4096).
    pub width: u32,
    /// Image height in pixels (maximum 4096).
    pub height: u32,
    /// Row-major sRGBA8 byte values; the length must equal width × height × 4.
    pub rgba8: Vec<u8>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct LoadTextureAssetParams {
    /// Encoded image bytes, maximum 16 MiB.
    pub encoded_bytes: Vec<u8>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetVisibilityParams {
    /// Numeric engine entity identifier.
    pub id: u64,
    /// Whether this entity is locally visible.
    pub visible: bool,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetVisibilityParams {
    /// Numeric engine entity identifier.
    pub id: u64,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetLightingParams {
    /// Direction from a shaded point toward the directional light.
    pub direction: [f32; 3],
    /// Linear RGB multiplier; channels are limited to 16.
    pub color: [f32; 3],
    /// Directional light strength, from 0 through 1000.
    pub intensity: f32,
    /// Linear ambient RGB contribution, each channel from 0 through 1.
    pub ambient: [f32; 3],
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetPointLightParams {
    /// Numeric engine entity identifier.
    pub id: u64,
    /// Linear RGB point light color, channels from 0 through 16.
    pub color: [f32; 3],
    /// Point light intensity from 0 through 1000.
    pub intensity: f32,
    /// Maximum light influence distance in world units.
    pub radius: f32,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetBaseColorTextureParams {
    /// Numeric engine entity identifier.
    pub id: u64,
    /// Texture asset ID, or null to clear the texture.
    pub texture_asset_id: Option<u64>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetPositionParams {
    /// Numeric engine entity identifier.
    pub id: u64,
    /// New world-space position as [x, y, z].
    pub position: [f32; 3],
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetParentParams {
    /// Numeric engine entity identifier to reparent.
    pub id: u64,
    /// New parent identifier, or null to make the entity a root.
    pub parent_id: Option<u64>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetTransformParams {
    /// Numeric engine entity identifier.
    pub id: u64,
    /// New world-space position [x, y, z].
    pub position: [f32; 3],
    /// World-space rotation quaternion [x, y, z, w].
    pub rotation: [f32; 4],
    /// Non-uniform world-space scale [x, y, z].
    pub scale: [f32; 3],
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetMaterialParams {
    /// Numeric engine entity identifier.
    pub id: u64,
    /// Linear RGBA base color with channels from 0 through 1.
    pub base_color: [f32; 4],
    /// Optional roughness from 0 (smooth) through 1 (rough).
    pub roughness: Option<f32>,
    /// Optional metallic response from 0 (dielectric) through 1 (metal).
    pub metallic: Option<f32>,
    /// Optional alpha mode: auto, opaque, mask, or blend.
    pub alpha_mode: Option<String>,
    /// Optional alpha threshold used by mask mode.
    pub alpha_cutoff: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetCameraParams {
    /// Camera position in world space [x, y, z].
    pub position: [f32; 3],
    /// World-space point the camera looks at.
    pub target: [f32; 3],
    /// Camera up direction [x, y, z].
    pub up: [f32; 3],
    /// Vertical field of view in degrees, between 0 and 180.
    pub vertical_fov_degrees: f32,
    /// Perspective or orthographic projection (defaults to perspective).
    pub projection: Option<String>,
    /// Orthographic vertical span in world units (defaults to 10).
    pub orthographic_vertical_size: Option<f32>,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
    /// Linear RGBA clear color with channels from 0 through 1.
    pub clear_color: [f32; 4],
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AdvanceSimulationParams {
    /// Elapsed seconds to add to the simulation accumulator, from above 0 through 60.
    pub delta_seconds: f64,
    /// Maximum fixed steps to process in this call, default 4 and range 1 through 1000.
    pub max_steps: Option<u32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AddRigidBodyParams {
    /// Numeric engine entity identifier.
    pub id: u64,
    /// Body type: dynamic, kinematic, or static. Defaults to dynamic.
    pub body_type: Option<String>,
    /// Positive mass used for force integration. Defaults to 1.
    pub mass: Option<f32>,
    /// Multiplier applied to world gravity. Defaults to 1.
    pub gravity_scale: Option<f32>,
    /// Exponential linear velocity damping per second. Defaults to 0.05.
    pub linear_damping: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct Vector3Params {
    /// Numeric engine entity identifier.
    pub id: u64,
    /// Velocity in world units per second.
    pub velocity: [f32; 3],
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ForceParams {
    /// Numeric engine entity identifier.
    pub id: u64,
    /// Force vector applied through the next fixed step.
    pub force: [f32; 3],
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetGravityParams {
    /// World gravity acceleration [x, y, z], in world units per second squared.
    pub gravity: [f32; 3],
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SetColliderParams {
    /// Numeric engine entity identifier.
    pub id: u64,
    /// Collider shape: sphere or box.
    pub shape: String,
    /// Sphere radius. Defaults to 0.5.
    pub radius: Option<f32>,
    /// Box half extents [x, y, z]. Defaults to [0.5, 0.5, 0.5].
    pub half_extents: Option<[f32; 3]>,
    /// Bounciness from 0 through 1. Defaults to 0.
    pub restitution: Option<f32>,
    /// Surface friction from 0 through 1. Defaults to 0.5.
    pub friction: Option<f32>,
}

#[cfg(feature = "scripting-lua")]
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AttachScriptParams {
    /// Numeric engine entity identifier to own the script.
    pub id: u64,
    /// Lua source chunk. Define `update(dt)` to run on each fixed simulation step.
    pub source: String,
}

#[cfg(feature = "scripting-lua")]
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ScriptEntityParams {
    /// Numeric engine entity identifier.
    pub id: u64,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeleteEntityParams {
    /// Numeric engine entity identifier.
    pub id: u64,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct LoadSceneParams {
    /// Versioned JSON string returned by the export_scene tool.
    pub scene_json: String,
}

#[cfg(feature = "scripting-lua")]
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RunLuaParams {
    /// Bounded Lua source code with access to the current engine scene.
    pub source: String,
}

impl EngineMcpServer {
    pub fn new(engine: Engine) -> Self {
        Self::new_shared(Arc::new(Mutex::new(engine)))
    }

    pub fn new_shared(engine: Arc<Mutex<Engine>>) -> Self {
        #[cfg(feature = "scripting-lua")]
        let scripts = ScriptWorker::new(Arc::clone(&engine));
        Self {
            engine,
            #[cfg(feature = "scripting-lua")]
            scripts,
        }
    }

    #[cfg(feature = "scripting-lua")]
    /// Reuse an existing worker tied to the same engine to share persistent
    /// scripts and simulation ticks with another host surface such as desktop.
    pub fn new_shared_with_script_worker(
        engine: Arc<Mutex<Engine>>,
        scripts: ScriptWorker,
    ) -> Self {
        Self { engine, scripts }
    }

    pub fn shared_engine(&self) -> Arc<Mutex<Engine>> {
        Arc::clone(&self.engine)
    }

    pub fn with_default_engine() -> Self {
        let engine = Engine::new(EngineConfig::default())
            .expect("the default engine configuration is valid");
        Self::new(engine)
    }

    fn lock_engine(&self) -> std::sync::MutexGuard<'_, Engine> {
        self.engine
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn dispatch_tool(&self, name: &str, arguments: Value) -> Result<Value, ControlError> {
        #[cfg(feature = "scripting-lua")]
        match name {
            "export_scene" => {
                let scene_json = self
                    .scripts
                    .export_scene_json()
                    .map_err(ControlError::Script)?;
                return Ok(serde_json::json!({ "scene_json": scene_json }));
            }
            "load_scene" => {
                let params: LoadSceneParams = serde_json::from_value(arguments)?;
                let engine = self
                    .scripts
                    .load_scene_json(params.scene_json)
                    .map_err(ControlError::Script)?;
                return Ok(serde_json::json!({ "ok": true, "engine": engine }));
            }
            "attach_script" => {
                let params: AttachScriptParams = serde_json::from_value(arguments)?;
                self.scripts
                    .attach(params.id, params.source)
                    .map_err(ControlError::Script)?;
                return Ok(serde_json::json!({ "ok": true }));
            }
            "detach_script" => {
                let params: ScriptEntityParams = serde_json::from_value(arguments)?;
                let detached = self
                    .scripts
                    .detach(params.id)
                    .map_err(ControlError::Script)?;
                return Ok(serde_json::json!({ "detached": detached }));
            }
            "list_scripts" => {
                return serde_json::to_value(self.scripts.list().map_err(ControlError::Script)?)
                    .map_err(ControlError::from);
            }
            "run_lua" => {
                let params: RunLuaParams = serde_json::from_value(arguments)?;
                let memory_bytes = self
                    .scripts
                    .run_lua(params.source)
                    .map_err(ControlError::Script)?;
                return Ok(serde_json::json!({ "ok": true, "memory_bytes": memory_bytes }));
            }
            "advance_simulation" => {
                let params: AdvanceSimulationParams = serde_json::from_value(arguments)?;
                let max_steps = params.max_steps.unwrap_or(4);
                if !params.delta_seconds.is_finite()
                    || params.delta_seconds <= 0.0
                    || params.delta_seconds > 60.0
                    || !(1..=1000).contains(&max_steps)
                {
                    return Err(ControlError::InvalidValue(
                        "delta_seconds must be in (0, 60] and max_steps must be 1 through 1000"
                            .into(),
                    ));
                }
                let result = self
                    .scripts
                    .advance_simulation(params.delta_seconds, max_steps)
                    .map_err(ControlError::Script)?;
                return serde_json::to_value(result).map_err(ControlError::from);
            }
            _ => {}
        }

        let mut engine = self.lock_engine();
        call_tool(&mut engine, name, arguments)
    }
}

/// Run this engine control surface on MCP stdio. The process should reserve
/// stdout for the protocol stream and send diagnostic output to stderr.
pub async fn serve_stdio(
    server: EngineMcpServer,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// Run Streamable HTTP MCP, with an optional WebMCP tool page and JSON bridge.
/// Both browser and MCP calls use the same catalog and Rust dispatcher.
pub async fn serve_http(
    server: EngineMcpServer,
    bind: SocketAddr,
    enable_webmcp: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let service = StreamableHttpService::new(
        {
            let server = server.clone();
            move || Ok(server.clone())
        },
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default().with_max_request_body_bytes(64 * 1024 * 1024),
    );

    let router = if enable_webmcp {
        Router::new()
            .nest_service("/mcp", service)
            .route("/webmcp", get(webmcp_page))
            .route("/api/tools", get(list_webmcp_tools))
            .route("/api/tools/{name}", post(call_webmcp_tool))
            .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
            .with_state(server)
    } else {
        Router::new().nest_service("/mcp", service)
    };

    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

async fn webmcp_page() -> Html<&'static str> {
    Html(include_str!("../assets/webmcp.html"))
}

async fn list_webmcp_tools() -> Json<Vec<crate::control::ToolDefinition>> {
    Json(tool_definitions())
}

async fn call_webmcp_tool(
    State(server): State<EngineMcpServer>,
    Path(name): Path<String>,
    Json(arguments): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    server
        .dispatch_tool(&name, arguments)
        .map(Json)
        .map_err(|error| {
            let status = match error {
                crate::control::ControlError::UnknownTool(_)
                | crate::control::ControlError::InvalidArguments(_)
                | crate::control::ControlError::InvalidValue(_)
                | crate::control::ControlError::InvalidMesh(_) => StatusCode::BAD_REQUEST,
                crate::control::ControlError::Engine(_) => StatusCode::UNPROCESSABLE_ENTITY,
                #[cfg(feature = "scripting-lua")]
                crate::control::ControlError::Script(_) => StatusCode::UNPROCESSABLE_ENTITY,
            };
            (
                status,
                Json(serde_json::json!({ "error": error.to_string() })),
            )
        })
}

#[tool_router(server_handler)]
impl EngineMcpServer {
    #[tool(description = "Export scene state and attached Lua source as versioned JSON")]
    fn export_scene(&self) -> String {
        format_tool_result(self.dispatch_tool("export_scene", serde_json::json!({})))
    }

    #[tool(
        description = "Validate a versioned scene and initialize its saved Lua scripts before replacing the active scene"
    )]
    fn load_scene(&self, Parameters(params): Parameters<LoadSceneParams>) -> String {
        format_tool_result(self.dispatch_tool(
            "load_scene",
            serde_json::json!({ "scene_json": params.scene_json }),
        ))
    }

    #[tool(description = "List scene entities with their names and transforms")]
    fn list_entities(&self) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "list_entities",
            serde_json::json!({}),
        ))
    }

    #[tool(
        description = "Read the current transient keyboard and mouse state supplied by the host"
    )]
    fn get_input_state(&self) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "get_input_state",
            serde_json::json!({}),
        ))
    }

    #[tool(description = "Set local scene visibility; an invisible parent hides its descendants")]
    fn set_visibility(&self, Parameters(params): Parameters<SetVisibilityParams>) -> String {
        format_tool_result(self.dispatch_tool(
            "set_visibility",
            serde_json::json!({ "id": params.id, "visible": params.visible }),
        ))
    }

    #[tool(description = "Read local and inherited scene visibility for an entity")]
    fn get_visibility(&self, Parameters(params): Parameters<GetVisibilityParams>) -> String {
        format_tool_result(
            self.dispatch_tool("get_visibility", serde_json::json!({ "id": params.id })),
        )
    }

    #[tool(description = "Read the scene-wide directional and ambient lighting")]
    fn get_lighting(&self) -> String {
        format_tool_result(self.dispatch_tool("get_lighting", serde_json::json!({})))
    }

    #[tool(description = "Set the scene-wide directional and ambient lighting")]
    fn set_lighting(&self, Parameters(params): Parameters<SetLightingParams>) -> String {
        format_tool_result(self.dispatch_tool(
            "set_lighting",
            serde_json::json!({
                "direction": params.direction,
                "color": params.color,
                "intensity": params.intensity,
                "ambient": params.ambient
            }),
        ))
    }

    #[tool(description = "Attach or replace a point light on an entity")]
    fn set_point_light(&self, Parameters(params): Parameters<SetPointLightParams>) -> String {
        format_tool_result(self.dispatch_tool(
            "set_point_light",
            serde_json::json!({
                "id": params.id,
                "color": params.color,
                "intensity": params.intensity,
                "radius": params.radius
            }),
        ))
    }

    #[tool(description = "Read an entity's point light, if present")]
    fn get_point_light(&self, Parameters(params): Parameters<DeleteEntityParams>) -> String {
        format_tool_result(
            self.dispatch_tool("get_point_light", serde_json::json!({ "id": params.id })),
        )
    }

    #[tool(description = "Remove an entity's point light")]
    fn remove_point_light(&self, Parameters(params): Parameters<DeleteEntityParams>) -> String {
        format_tool_result(
            self.dispatch_tool("remove_point_light", serde_json::json!({ "id": params.id })),
        )
    }

    #[tool(description = "Create a bounded sRGBA8 texture asset from packed RGBA byte values")]
    fn create_texture_asset(
        &self,
        Parameters(params): Parameters<CreateTextureAssetParams>,
    ) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "create_texture_asset",
            serde_json::json!({ "width": params.width, "height": params.height, "rgba8": params.rgba8 }),
        ))
    }

    #[tool(description = "Decode supported image bytes into a bounded sRGBA8 texture asset")]
    fn load_texture_asset(&self, Parameters(params): Parameters<LoadTextureAssetParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "load_texture_asset",
            serde_json::json!({ "encoded_bytes": params.encoded_bytes }),
        ))
    }

    #[tool(description = "Assign or clear the base-color texture on a mesh entity")]
    fn set_base_color_texture(
        &self,
        Parameters(params): Parameters<SetBaseColorTextureParams>,
    ) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "set_base_color_texture",
            serde_json::json!({ "id": params.id, "texture_asset_id": params.texture_asset_id }),
        ))
    }

    #[tool(description = "Read the base-color texture asset assigned to a mesh entity")]
    fn get_base_color_texture(&self, Parameters(params): Parameters<DeleteEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "get_base_color_texture",
            serde_json::json!({ "id": params.id }),
        ))
    }

    #[tool(description = "Create a named scene entity at a world-space position")]
    fn spawn_entity(&self, Parameters(params): Parameters<SpawnEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "spawn_entity",
            serde_json::json!({ "name": params.name, "position": params.position }),
        ))
    }

    #[tool(description = "Create a cube, sphere, or plane mesh in the scene")]
    fn spawn_primitive(&self, Parameters(params): Parameters<SpawnPrimitiveParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "spawn_primitive",
            serde_json::json!({
                "name": params.name,
                "primitive": params.primitive,
                "position": params.position,
                "base_color": params.base_color,
                "roughness": params.roughness,
                "metallic": params.metallic
            }),
        ))
    }

    #[tool(description = "Load a bounded Wavefront OBJ mesh and spawn it into the scene")]
    fn load_obj_mesh(&self, Parameters(params): Parameters<LoadObjMeshParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "load_obj_mesh",
            serde_json::json!({
                "name": params.name,
                "source": params.source,
                "position": params.position,
                "base_color": params.base_color,
                "roughness": params.roughness,
                "metallic": params.metallic
            }),
        ))
    }

    #[tool(
        description = "Load a bounded glTF 2.0 or GLB static scene and spawn its editable node hierarchy into the scene"
    )]
    fn load_gltf_mesh(&self, Parameters(params): Parameters<LoadGltfMeshParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "load_gltf_mesh",
            serde_json::json!({
                "name": params.name,
                "source_base64": params.source_base64,
                "position": params.position,
                "base_color": params.base_color,
                "roughness": params.roughness,
                "metallic": params.metallic
            }),
        ))
    }

    #[tool(description = "List animation clips and the playback state of an imported glTF scene")]
    fn list_gltf_animations(&self, Parameters(params): Parameters<ScriptEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "list_gltf_animations",
            serde_json::json!({ "id": params.id }),
        ))
    }

    #[tool(description = "Play a glTF animation clip by index")]
    fn play_gltf_animation(
        &self,
        Parameters(params): Parameters<PlayGltfAnimationParams>,
    ) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "play_gltf_animation",
            serde_json::json!({
                "id": params.id,
                "clip_index": params.clip_index,
                "looping": params.looping
            }),
        ))
    }

    #[tool(description = "Stop playback on an imported glTF animation player")]
    fn stop_gltf_animation(&self, Parameters(params): Parameters<ScriptEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "stop_gltf_animation",
            serde_json::json!({ "id": params.id }),
        ))
    }

    #[tool(description = "Set the speed of an imported glTF animation player")]
    fn set_gltf_animation_speed(
        &self,
        Parameters(params): Parameters<SetGltfAnimationSpeedParams>,
    ) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "set_gltf_animation_speed",
            serde_json::json!({ "id": params.id, "speed": params.speed }),
        ))
    }

    #[tool(description = "Set a scene entity's world-space position")]
    fn set_position(&self, Parameters(params): Parameters<SetPositionParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "set_position",
            serde_json::json!({ "id": params.id, "position": params.position }),
        ))
    }

    #[tool(description = "Set or clear an entity's parent while preserving its local transform")]
    fn set_parent(&self, Parameters(params): Parameters<SetParentParams>) -> String {
        format_tool_result(self.dispatch_tool(
            "set_parent",
            serde_json::json!({ "id": params.id, "parent_id": params.parent_id }),
        ))
    }

    #[tool(description = "Get an entity's composed world-space transform")]
    fn get_world_transform(&self, Parameters(params): Parameters<DeleteEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "get_world_transform",
            serde_json::json!({ "id": params.id }),
        ))
    }

    #[tool(description = "Set an entity's position, quaternion rotation, and scale")]
    fn set_transform(&self, Parameters(params): Parameters<SetTransformParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "set_transform",
            serde_json::json!({
                "id": params.id,
                "position": params.position,
                "rotation": params.rotation,
                "scale": params.scale
            }),
        ))
    }

    #[tool(description = "Set an entity's base color, PBR response, and alpha mode")]
    fn set_material(&self, Parameters(params): Parameters<SetMaterialParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "set_material",
            serde_json::json!({
                "id": params.id,
                "base_color": params.base_color,
                "roughness": params.roughness,
                "metallic": params.metallic,
                "alpha_mode": params.alpha_mode,
                "alpha_cutoff": params.alpha_cutoff
            }),
        ))
    }

    #[tool(description = "Get an entity's base color, roughness, and metallic response")]
    fn get_material(&self, Parameters(params): Parameters<DeleteEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "get_material",
            serde_json::json!({ "id": params.id }),
        ))
    }

    #[tool(description = "Delete a scene entity")]
    fn delete_entity(&self, Parameters(params): Parameters<DeleteEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "delete_entity",
            serde_json::json!({ "id": params.id }),
        ))
    }

    #[tool(description = "Advance fixed-step simulation by elapsed seconds")]
    fn advance_simulation(
        &self,
        Parameters(params): Parameters<AdvanceSimulationParams>,
    ) -> String {
        format_tool_result(self.dispatch_tool(
            "advance_simulation",
            serde_json::json!({
                "delta_seconds": params.delta_seconds,
                "max_steps": params.max_steps
            }),
        ))
    }

    #[tool(description = "Add or replace an entity's dynamic, kinematic, or static rigid body")]
    fn add_rigid_body(&self, Parameters(params): Parameters<AddRigidBodyParams>) -> String {
        format_tool_result(self.dispatch_tool(
            "add_rigid_body",
            serde_json::json!({
                "id": params.id,
                "body_type": params.body_type,
                "mass": params.mass,
                "gravity_scale": params.gravity_scale,
                "linear_damping": params.linear_damping
            }),
        ))
    }

    #[tool(description = "Get an entity's rigid body state or null if it has no body")]
    fn get_rigid_body(&self, Parameters(params): Parameters<DeleteEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "get_rigid_body",
            serde_json::json!({ "id": params.id }),
        ))
    }

    #[tool(description = "Remove an entity's rigid body component")]
    fn remove_rigid_body(&self, Parameters(params): Parameters<DeleteEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "remove_rigid_body",
            serde_json::json!({ "id": params.id }),
        ))
    }

    #[tool(description = "Set the linear velocity of an entity with a rigid body")]
    fn set_velocity(&self, Parameters(params): Parameters<Vector3Params>) -> String {
        format_tool_result(self.dispatch_tool(
            "set_velocity",
            serde_json::json!({ "id": params.id, "velocity": params.velocity }),
        ))
    }

    #[tool(description = "Apply a force to an entity's dynamic rigid body")]
    fn apply_force(&self, Parameters(params): Parameters<ForceParams>) -> String {
        format_tool_result(self.dispatch_tool(
            "apply_force",
            serde_json::json!({ "id": params.id, "force": params.force }),
        ))
    }

    #[tool(description = "Get global gravity acceleration")]
    fn get_gravity(&self) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(&mut engine, "get_gravity", serde_json::json!({})))
    }

    #[tool(description = "Set global gravity acceleration in world units per second squared")]
    fn set_gravity(&self, Parameters(params): Parameters<SetGravityParams>) -> String {
        format_tool_result(self.dispatch_tool(
            "set_gravity",
            serde_json::json!({ "gravity": params.gravity }),
        ))
    }

    #[tool(description = "Add or replace a sphere or box collider on an entity")]
    fn set_collider(&self, Parameters(params): Parameters<SetColliderParams>) -> String {
        format_tool_result(self.dispatch_tool(
            "set_collider",
            serde_json::json!({
                "id": params.id,
                "shape": params.shape,
                "radius": params.radius,
                "half_extents": params.half_extents,
                "restitution": params.restitution,
                "friction": params.friction
            }),
        ))
    }

    #[tool(description = "Get an entity's collider or null if it has none")]
    fn get_collider(&self, Parameters(params): Parameters<DeleteEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "get_collider",
            serde_json::json!({ "id": params.id }),
        ))
    }

    #[tool(description = "Remove an entity's collider component")]
    fn remove_collider(&self, Parameters(params): Parameters<DeleteEntityParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "remove_collider",
            serde_json::json!({ "id": params.id }),
        ))
    }

    #[tool(description = "Get entity count and fixed-step simulation counters")]
    fn get_engine_info(&self) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "get_engine_info",
            serde_json::json!({}),
        ))
    }

    #[tool(description = "Get the active scene camera and perspective settings")]
    fn get_camera(&self) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(&mut engine, "get_camera", serde_json::json!({})))
    }

    #[tool(
        description = "Set the active scene camera position, target, projection, and clear color"
    )]
    fn set_camera(&self, Parameters(params): Parameters<SetCameraParams>) -> String {
        let mut engine = self.lock_engine();
        format_tool_result(call_tool(
            &mut engine,
            "set_camera",
            serde_json::json!({
                "position": params.position,
                "target": params.target,
                "up": params.up,
                "vertical_fov_degrees": params.vertical_fov_degrees,
                "near": params.near,
                "far": params.far,
                "clear_color": params.clear_color
            }),
        ))
    }

    #[cfg(feature = "scripting-lua")]
    #[tool(description = "Run a bounded Lua chunk with access to the current engine scene")]
    fn run_lua(&self, Parameters(params): Parameters<RunLuaParams>) -> String {
        format_tool_result(
            self.dispatch_tool("run_lua", serde_json::json!({ "source": params.source })),
        )
    }

    #[cfg(feature = "scripting-lua")]
    #[tool(description = "Attach and initialize a persistent Lua script on an entity")]
    fn attach_script(&self, Parameters(params): Parameters<AttachScriptParams>) -> String {
        format_tool_result(self.dispatch_tool(
            "attach_script",
            serde_json::json!({ "id": params.id, "source": params.source }),
        ))
    }

    #[cfg(feature = "scripting-lua")]
    #[tool(description = "Stop and remove an entity's attached Lua script")]
    fn detach_script(&self, Parameters(params): Parameters<ScriptEntityParams>) -> String {
        format_tool_result(
            self.dispatch_tool("detach_script", serde_json::json!({ "id": params.id })),
        )
    }

    #[cfg(feature = "scripting-lua")]
    #[tool(description = "List attached Lua scripts and recent script failures")]
    fn list_scripts(&self) -> String {
        format_tool_result(self.dispatch_tool("list_scripts", serde_json::json!({})))
    }
}

fn format_tool_result(result: Result<serde_json::Value, crate::control::ControlError>) -> String {
    match result {
        Ok(value) => value.to_string(),
        Err(error) => serde_json::json!({ "error": error.to_string() }).to_string(),
    }
}
