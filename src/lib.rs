//! A platform-neutral game engine core with optional rendering, Lua scripting,
//! and MCP control adapters.
//!
//! The simulation model has no dependency on a window system, GPU, or network
//! transport. Native and browser hosts can therefore drive the same [`Engine`]
//! state and command API.

pub mod control;
pub mod engine;
#[cfg(all(feature = "mcp", not(target_arch = "wasm32")))]
pub mod mcp;
pub mod mesh;
#[cfg(feature = "render-wgpu")]
pub mod render;
#[cfg(feature = "scripting-lua")]
pub mod scripting;
#[cfg(all(
    target_arch = "wasm32",
    feature = "render-wgpu",
    feature = "scripting-lua"
))]
pub mod wasm;

pub use engine::{
    AlphaMode3d, BaseColorTexture, Camera3d, CameraProjection3d, CaricatureHead, Collider3d,
    ColliderShape, Engine, EngineConfig, EngineError, EngineSnapshot, EntityId, GltfAnimationState,
    GltfSceneInstance, InputState, Lighting3d, Material3d, Mesh3d, MeshKind, MeshSource,
    PointLight3d, RigidBody3d, RigidBodyType, SceneAnimationPlayer, SceneDocument, SceneEntity,
    SceneMeshAsset, SceneMeshSource, SceneScript, SceneTextureAsset, TextureAssetId, TextureData,
    Transform, Visibility3d,
};
pub use mesh::{
    GltfAnimationChannelData, GltfAnimationData, GltfAnimationKeyframe, GltfAnimationPath,
    GltfAssetData, GltfInterpolation, GltfNodeData, GltfPrimitiveData, MeshAssetId, MeshData,
    MeshLoadError, MeshVertex,
};
#[cfg(feature = "render-wgpu")]
pub use render::{RenderCamera, RenderStats, SceneRenderer};

#[cfg(all(
    target_arch = "wasm32",
    feature = "render-wgpu",
    feature = "scripting-lua"
))]
pub use wasm::BrowserEngine;

#[cfg(feature = "scripting-lua")]
pub use scripting::{
    AttachedScriptInfo, ScriptFailure, ScriptHost, ScriptLimits, ScriptRuntime,
    ScriptWorkerSnapshot, SimulationAdvance,
};

#[cfg(all(feature = "scripting-lua", not(target_arch = "wasm32")))]
pub use scripting::ScriptWorker;

/// Crate version, also used by the native host for its combined version output.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
