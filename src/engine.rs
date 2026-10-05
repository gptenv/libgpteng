use glam::{Mat4, Quat, Vec3};
use hecs::{Entity, World};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, io::Cursor, sync::Arc};
use thiserror::Error;

use crate::mesh::{
    GltfAnimationData, GltfAnimationPath, GltfAssetData, GltfInterpolation, MeshAssetId, MeshData,
};

const SCENE_FORMAT_VERSION: u32 = 1;
const MAX_SCENE_JSON_BYTES: usize = 64 * 1024 * 1024;
const MAX_SCENE_ENTITIES: usize = 100_000;
const MAX_SCENE_MESH_ASSETS: usize = 4_096;
const MAX_SCENE_MESH_ELEMENTS: usize = 4_000_000;
const MAX_SCENE_TEXTURE_ASSETS: usize = 256;
const MAX_SCENE_TEXTURE_BYTES: usize = 16 * 1024 * 1024;
const MAX_ENCODED_TEXTURE_BYTES: usize = 16 * 1024 * 1024;
const MAX_TEXTURE_DIMENSION: u32 = 4096;
const MAX_SCENE_SCRIPT_BYTES: usize = 1024 * 1024;
const MAX_SCENE_ALL_SCRIPT_BYTES: usize = 8 * 1024 * 1024;
const MAX_SCENE_ANIMATION_KEYFRAMES: usize = 1_000_000;

fn default_visible() -> bool {
    true
}

/// Portable scene document for files, editor integrations, and engine tools.
/// Script source is serializable; VM globals and timing counters stay
/// host-owned and are recreated when a host attaches the saved definitions.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneDocument {
    pub format_version: u32,
    pub fixed_delta_seconds: f64,
    pub gravity: Vec3,
    pub camera: Camera3d,
    #[serde(default)]
    pub lighting: Lighting3d,
    pub mesh_assets: Vec<SceneMeshAsset>,
    #[serde(default)]
    pub texture_assets: Vec<SceneTextureAsset>,
    pub entities: Vec<SceneEntity>,
    #[serde(default)]
    pub scripts: Vec<SceneScript>,
    #[serde(default)]
    pub animation_players: Vec<SceneAnimationPlayer>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneAnimationPlayer {
    pub root_id: u64,
    pub node_ids: Vec<u64>,
    pub base_transforms: Vec<Transform>,
    pub clips: Vec<GltfAnimationData>,
    pub state: GltfAnimationState,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneScript {
    pub entity_id: u64,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneMeshAsset {
    pub id: u64,
    pub mesh: MeshData,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneTextureAsset {
    pub id: u64,
    pub texture: TextureData,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum SceneMeshSource {
    Primitive(MeshKind),
    Asset(u64),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneEntity {
    pub id: u64,
    pub name: String,
    pub transform: Transform,
    pub parent_id: Option<u64>,
    pub mesh: Option<SceneMeshSource>,
    pub material: Option<Material3d>,
    pub rigid_body: Option<RigidBody3d>,
    pub collider: Option<Collider3d>,
    #[serde(default)]
    pub point_light: Option<PointLight3d>,
    #[serde(default = "default_visible")]
    pub visible: bool,
    #[serde(default)]
    pub base_color_texture: Option<u64>,
}

/// A CPU-side, tightly packed, non-premultiplied sRGBA8 image.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct TextureAssetId(pub(crate) u64);

impl TextureAssetId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl TextureData {
    fn from_encoded(encoded: &[u8]) -> Result<Self, EngineError> {
        if encoded.is_empty() || encoded.len() > MAX_ENCODED_TEXTURE_BYTES {
            return Err(EngineError::InvalidEncodedTexture(
                "encoded input must be 1 byte to 16 MiB".to_owned(),
            ));
        }
        let mut reader = image::ImageReader::new(Cursor::new(encoded))
            .with_guessed_format()
            .map_err(|error| EngineError::InvalidEncodedTexture(error.to_string()))?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(MAX_TEXTURE_DIMENSION);
        limits.max_image_height = Some(MAX_TEXTURE_DIMENSION);
        limits.max_alloc = Some((MAX_SCENE_TEXTURE_BYTES * 2) as u64);
        reader.limits(limits);
        let rgba = reader
            .decode()
            .map_err(|error| EngineError::InvalidEncodedTexture(error.to_string()))?
            .into_rgba8();
        Self::new(rgba.width(), rgba.height(), rgba.into_raw())
    }

    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, EngineError> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4));
        if width == 0
            || height == 0
            || width > MAX_TEXTURE_DIMENSION
            || height > MAX_TEXTURE_DIMENSION
            || expected != Some(rgba8.len())
            || rgba8.len() > MAX_SCENE_TEXTURE_BYTES
        {
            return Err(EngineError::InvalidTexture);
        }
        Ok(Self {
            width,
            height,
            rgba8,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BaseColorTexture(pub TextureAssetId);

/// Whether an entity participates in scene rendering. Ancestors' visibility
/// is inherited by descendants when calculating effective visibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Visibility3d(pub bool);

impl Default for Visibility3d {
    fn default() -> Self {
        Self(true)
    }
}

/// Entity handles for the reusable, animated-demo caricature scene component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CaricatureHead {
    pub root: EntityId,
    pub face: EntityId,
}

/// Stable, serializable identifier for a live entity in this engine instance.
/// The raw ECS handle is intentionally kept inside the engine boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct EntityId(pub(crate) u64);

impl EntityId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    fn validated(self) -> Result<Self, EngineError> {
        let rotation_length_squared = self.rotation.length_squared();
        if !self.translation.is_finite()
            || !self.scale.is_finite()
            || !self.rotation.is_finite()
            || !rotation_length_squared.is_finite()
            || rotation_length_squared <= f32::EPSILON
        {
            return Err(EngineError::InvalidTransform);
        }
        Ok(Self {
            rotation: self.rotation.normalize(),
            ..self
        })
    }
}

/// Projection mode for the active camera.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CameraProjection3d {
    #[default]
    Perspective,
    Orthographic {
        vertical_size: f32,
    },
}

/// Active camera shared by desktop, browser, gameplay, and tools.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera3d {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    #[serde(default)]
    pub projection: CameraProjection3d,
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
    pub clear_color: [f32; 4],
}

/// Scene-wide directional and ambient lighting configuration.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lighting3d {
    /// Direction from the shaded point toward the light source.
    pub direction: Vec3,
    /// Linear RGB radiance multiplier. Channels are limited to 16.
    pub color: [f32; 3],
    /// Nonnegative directional-light strength, limited to 1000.
    pub intensity: f32,
    /// Linear RGB ambient contribution in the 0..1 range.
    pub ambient: [f32; 3],
}

impl Default for Lighting3d {
    fn default() -> Self {
        Self {
            direction: Vec3::new(-0.4, 0.7, 0.6),
            color: [1.0; 3],
            intensity: 1.0,
            ambient: [0.12; 3],
        }
    }
}

impl Lighting3d {
    fn validated(self) -> Result<Self, EngineError> {
        if !self.direction.is_finite()
            || self.direction.length_squared() <= f32::EPSILON
            || !self.intensity.is_finite()
            || !(0.0..=1000.0).contains(&self.intensity)
            || self
                .color
                .iter()
                .any(|channel| !channel.is_finite() || !(0.0..=16.0).contains(channel))
            || self
                .ambient
                .iter()
                .any(|channel| !channel.is_finite() || !(0.0..=1.0).contains(channel))
        {
            return Err(EngineError::InvalidLighting);
        }
        Ok(Self {
            direction: self.direction.normalize(),
            ..self
        })
    }
}

impl Camera3d {
    pub(crate) fn validated(self) -> Result<Self, EngineError> {
        let view_direction = self.target - self.position;
        if !self.position.is_finite()
            || !self.target.is_finite()
            || !self.up.is_finite()
            || !self.vertical_fov_radians.is_finite()
            || !self.near.is_finite()
            || !self.far.is_finite()
            || !view_direction.length_squared().is_finite()
            || view_direction.length_squared() <= f32::EPSILON
            || !self.up.length_squared().is_finite()
            || self.up.length_squared() <= f32::EPSILON
            || !view_direction.cross(self.up).length_squared().is_finite()
            || view_direction.cross(self.up).length_squared() <= f32::EPSILON
            || match self.projection {
                CameraProjection3d::Perspective => {
                    !(0.01..std::f32::consts::PI - 0.01).contains(&self.vertical_fov_radians)
                }
                CameraProjection3d::Orthographic { vertical_size } => {
                    !vertical_size.is_finite() || !(0.001..=100_000.0).contains(&vertical_size)
                }
            }
            || self.near <= 0.0
            || self.far <= self.near
            || self
                .clear_color
                .iter()
                .any(|channel| !channel.is_finite() || !(0.0..=1.0).contains(channel))
        {
            return Err(EngineError::InvalidCamera);
        }
        Ok(Self {
            up: self.up.normalize(),
            ..self
        })
    }
}

impl Default for Camera3d {
    fn default() -> Self {
        Self {
            position: Vec3::new(2.8, 2.2, 4.2),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: CameraProjection3d::Perspective,
            vertical_fov_radians: 55.0_f32.to_radians(),
            near: 0.1,
            far: 100.0,
            clear_color: [0.035, 0.055, 0.085, 1.0],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshKind {
    Cube,
    Sphere,
    Plane,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RigidBodyType {
    Dynamic,
    Kinematic,
    Static,
}

/// Linear rigid-body state integrated at the engine's fixed timestep.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidBody3d {
    pub body_type: RigidBodyType,
    pub velocity: Vec3,
    pub accumulated_force: Vec3,
    pub mass: f32,
    pub gravity_scale: f32,
    pub linear_damping: f32,
}

impl Default for RigidBody3d {
    fn default() -> Self {
        Self {
            body_type: RigidBodyType::Dynamic,
            velocity: Vec3::ZERO,
            accumulated_force: Vec3::ZERO,
            mass: 1.0,
            gravity_scale: 1.0,
            linear_damping: 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "shape")]
pub enum ColliderShape {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Collider3d {
    pub shape: ColliderShape,
    pub restitution: f32,
    pub friction: f32,
}

/// A point light attached to an entity and positioned by its world transform.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointLight3d {
    /// Linear RGB intensity multiplier; each channel is limited to 16.
    pub color: [f32; 3],
    /// Nonnegative radiance strength, limited to 1000.
    pub intensity: f32,
    /// Maximum influence distance in world units, in (0, 10000].
    pub radius: f32,
}

impl Default for PointLight3d {
    fn default() -> Self {
        Self {
            color: [1.0; 3],
            intensity: 1.0,
            radius: 10.0,
        }
    }
}

impl PointLight3d {
    fn validated(self) -> Result<Self, EngineError> {
        if self
            .color
            .iter()
            .any(|channel| !channel.is_finite() || !(0.0..=16.0).contains(channel))
            || !self.intensity.is_finite()
            || !(0.0..=1000.0).contains(&self.intensity)
            || !self.radius.is_finite()
            || self.radius <= 0.0
            || self.radius > 10_000.0
        {
            return Err(EngineError::InvalidPointLight);
        }
        Ok(self)
    }
}

impl Default for Collider3d {
    fn default() -> Self {
        Self {
            shape: ColliderShape::Box {
                half_extents: Vec3::splat(0.5),
            },
            restitution: 0.0,
            friction: 0.5,
        }
    }
}

impl MeshKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cube => "cube",
            Self::Sphere => "sphere",
            Self::Plane => "plane",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum MeshSource {
    Primitive(MeshKind),
    Asset(MeshAssetId),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Mesh3d(pub MeshSource);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaMode3d {
    Opaque,
    Mask,
    Blend,
    #[default]
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material3d {
    /// Linear RGBA base color.
    pub base_color: [f32; 4],
    /// Microfacet surface roughness. 0 is smooth and 1 is fully rough.
    pub roughness: f32,
    /// Metallic response from 0 (dielectric) to 1 (metal).
    pub metallic: f32,
    /// Alpha compositing mode. `auto` keeps legacy alpha inference behavior.
    #[serde(default)]
    pub alpha_mode: AlphaMode3d,
    /// Alpha threshold used by `mask` mode.
    #[serde(default = "default_alpha_cutoff")]
    pub alpha_cutoff: f32,
}

fn default_alpha_cutoff() -> f32 {
    0.5
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EngineSnapshot {
    pub entity_count: usize,
    pub fixed_steps: u64,
    pub fixed_delta_seconds: f64,
}

#[derive(Clone, Debug)]
pub struct GltfSceneInstance {
    pub root: EntityId,
    pub nodes: Vec<EntityId>,
    pub primitives: Vec<EntityId>,
    pub mesh_assets: Vec<MeshAssetId>,
    pub animations: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GltfAnimationState {
    pub clip_index: usize,
    pub clip_name: String,
    pub time_seconds: f32,
    pub speed: f32,
    pub looping: bool,
    pub playing: bool,
}

#[derive(Clone, Debug)]
struct GltfAnimationPlayer {
    root: EntityId,
    nodes: Vec<EntityId>,
    base_transforms: Vec<Transform>,
    clips: Vec<GltfAnimationData>,
    state: GltfAnimationState,
}

impl Default for Material3d {
    fn default() -> Self {
        Self {
            base_color: [0.18, 0.52, 0.88, 1.0],
            roughness: 0.55,
            metallic: 0.0,
            alpha_mode: AlphaMode3d::Auto,
            alpha_cutoff: default_alpha_cutoff(),
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Fixed simulation step in seconds. A value of 1/60 is a useful default.
    pub fixed_delta_seconds: f64,
    /// Constant acceleration applied to dynamic bodies, in world units/s².
    pub gravity: Vec3,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            fixed_delta_seconds: 1.0 / 60.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("input names must contain 1 to 32 ASCII characters and mouse motion must be finite")]
    InvalidInput,
    #[error(
        "texture dimensions must be between 1 and 4096 and RGBA data must have exactly width × height × 4 bytes (maximum 16 MiB)"
    )]
    InvalidTexture,
    #[error("encoded image is invalid, unsupported, or exceeds the image limits: {0}")]
    InvalidEncodedTexture(String),
    #[error("texture asset {0} does not exist")]
    UnknownTextureAsset(u64),
    #[error("fixed_delta_seconds must be finite and greater than zero")]
    InvalidFixedDelta,
    #[error("gravity must be finite")]
    InvalidGravity,
    #[error("entity {0} does not exist")]
    UnknownEntity(u64),
    #[error("entity {0} has no 3D material")]
    MissingMaterial(u64),
    #[error("entity {0} has no rigid body")]
    MissingRigidBody(u64),
    #[error(
        "collider parameters must be finite and positive; friction and restitution must be from 0 through 1"
    )]
    InvalidCollider,
    #[error(
        "rigid body mass must be positive; velocity, force, gravity scale, and damping must be finite; damping cannot be negative"
    )]
    InvalidRigidBody,
    #[error("material color, roughness, and metallic values must be finite and from 0 through 1")]
    InvalidMaterial,
    #[error(
        "transform translation and scale must be finite and rotation must be finite and nonzero"
    )]
    InvalidTransform,
    #[error("camera vectors, projection planes, field of view, and clear color are invalid")]
    InvalidCamera,
    #[error("lighting direction, color, intensity, or ambient values are invalid")]
    InvalidLighting,
    #[error("point light color, intensity, or radius is invalid")]
    InvalidPointLight,
    #[error("mesh asset {0} does not exist")]
    UnknownMeshAsset(u64),
    #[error("entity {0} has no glTF animation player")]
    UnknownAnimationPlayer(u64),
    #[error("glTF animation clip index {0} is out of range")]
    UnknownAnimationClip(usize),
    #[error("animation speed must be finite and have magnitude at most 100")]
    InvalidAnimationSpeed,
    #[error("an entity cannot be parented to itself or one of its descendants")]
    ParentCycle,
    #[error("scene document is invalid: {0}")]
    InvalidScene(String),
}

/// The authoritative simulation world. Rendering and host transports observe
/// or mutate it through explicit engine APIs rather than owning duplicate state.
pub struct Engine {
    config: EngineConfig,
    world: World,
    entity_ids: Vec<(EntityId, Entity)>,
    parents: Vec<(EntityId, EntityId)>,
    next_entity_id: u64,
    mesh_assets: Vec<(MeshAssetId, Arc<MeshData>)>,
    next_mesh_asset_id: u64,
    texture_assets: Vec<(TextureAssetId, Arc<TextureData>)>,
    next_texture_asset_id: u64,
    texture_asset_generation: u64,
    mesh_asset_generation: u64,
    gltf_animation_players: Vec<GltfAnimationPlayer>,
    active_camera: Camera3d,
    lighting: Lighting3d,
    elapsed_seconds: f64,
    fixed_steps: u64,
    input: InputState,
}

/// Host input visible to gameplay code. Key and button names are normalized
/// strings (for example `W`, `Space`, `left`, and `right`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InputState {
    pub keys_down: HashSet<String>,
    pub mouse_buttons_down: HashSet<String>,
    pub mouse_delta: [f64; 2],
}

impl Engine {
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        if !config.fixed_delta_seconds.is_finite() || config.fixed_delta_seconds <= 0.0 {
            return Err(EngineError::InvalidFixedDelta);
        }
        if !config.gravity.is_finite() {
            return Err(EngineError::InvalidGravity);
        }

        Ok(Self {
            config,
            world: World::new(),
            entity_ids: Vec::new(),
            parents: Vec::new(),
            next_entity_id: 1,
            mesh_assets: Vec::new(),
            next_mesh_asset_id: 1,
            texture_assets: Vec::new(),
            next_texture_asset_id: 1,
            texture_asset_generation: 0,
            mesh_asset_generation: 0,
            gltf_animation_players: Vec::new(),
            active_camera: Camera3d::default(),
            lighting: Lighting3d::default(),
            elapsed_seconds: 0.0,
            fixed_steps: 0,
            input: InputState::default(),
        })
    }

    /// Current host input snapshot. Input is transient and is not part of scene files.
    pub fn input(&self) -> &InputState {
        &self.input
    }

    /// Update a normalized key's held state. Names are case-insensitive and bounded.
    pub fn set_key_down(&mut self, key: &str, down: bool) -> Result<(), EngineError> {
        let key = normalize_input_name(key)?;
        if down {
            self.input.keys_down.insert(key);
        } else {
            self.input.keys_down.remove(&key);
        }
        Ok(())
    }

    /// Update a normalized mouse button's held state.
    pub fn set_mouse_button_down(&mut self, button: &str, down: bool) -> Result<(), EngineError> {
        let button = normalize_input_name(button)?;
        if down {
            self.input.mouse_buttons_down.insert(button);
        } else {
            self.input.mouse_buttons_down.remove(&button);
        }
        Ok(())
    }

    /// Accumulate relative pointer motion until the host clears it.
    pub fn add_mouse_motion(&mut self, dx: f64, dy: f64) -> Result<(), EngineError> {
        if !dx.is_finite() || !dy.is_finite() {
            return Err(EngineError::InvalidInput);
        }
        self.input.mouse_delta[0] += dx;
        self.input.mouse_delta[1] += dy;
        Ok(())
    }

    /// Clear accumulated relative pointer motion after the host frame is consumed.
    pub fn clear_mouse_motion(&mut self) {
        self.input.mouse_delta = [0.0, 0.0];
    }

    /// Capture engine-owned scene state in a versioned, serializable document.
    pub fn to_scene(&self) -> Result<SceneDocument, EngineError> {
        let mut entities = Vec::with_capacity(self.entity_ids.len());
        for (id, entity) in &self.entity_ids {
            let name = self
                .world
                .get::<&Name>(*entity)
                .map_err(|error| EngineError::InvalidScene(error.to_string()))?
                .0
                .clone();
            let transform = *self
                .world
                .get::<&Transform>(*entity)
                .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
            let mesh = self
                .world
                .get::<&Mesh3d>(*entity)
                .ok()
                .map(|mesh| match mesh.0 {
                    MeshSource::Primitive(kind) => SceneMeshSource::Primitive(kind),
                    MeshSource::Asset(asset_id) => SceneMeshSource::Asset(asset_id.get()),
                });
            let material = self.world.get::<&Material3d>(*entity).ok().map(|v| *v);
            let rigid_body = self.world.get::<&RigidBody3d>(*entity).ok().map(|v| *v);
            let collider = self.world.get::<&Collider3d>(*entity).ok().map(|v| *v);
            let point_light = self
                .world
                .get::<&PointLight3d>(*entity)
                .ok()
                .map(|light| *light);
            let visible = self
                .world
                .get::<&Visibility3d>(*entity)
                .map(|visibility| visibility.0)
                .unwrap_or(true);
            let base_color_texture = self
                .world
                .get::<&BaseColorTexture>(*entity)
                .ok()
                .map(|texture| texture.0.get());
            let parent_id = self
                .parents
                .iter()
                .find_map(|(child, parent)| (*child == *id).then_some(parent.get()));
            entities.push(SceneEntity {
                id: id.get(),
                name,
                transform,
                parent_id,
                mesh,
                material,
                rigid_body,
                collider,
                point_light,
                visible,
                base_color_texture,
            });
        }

        let mesh_assets = self
            .mesh_assets
            .iter()
            .map(|(id, mesh)| SceneMeshAsset {
                id: id.get(),
                mesh: mesh.as_ref().clone(),
            })
            .collect();
        let texture_assets = self
            .texture_assets
            .iter()
            .map(|(id, texture)| SceneTextureAsset {
                id: id.get(),
                texture: texture.as_ref().clone(),
            })
            .collect();
        let animation_players = self
            .gltf_animation_players
            .iter()
            .map(|player| SceneAnimationPlayer {
                root_id: player.root.get(),
                node_ids: player.nodes.iter().map(|id| id.get()).collect(),
                base_transforms: player.base_transforms.clone(),
                clips: player.clips.clone(),
                state: player.state.clone(),
            })
            .collect();
        Ok(SceneDocument {
            format_version: SCENE_FORMAT_VERSION,
            fixed_delta_seconds: self.config.fixed_delta_seconds,
            gravity: self.config.gravity,
            camera: self.active_camera,
            lighting: self.lighting,
            mesh_assets,
            texture_assets,
            entities,
            scripts: Vec::new(),
            animation_players,
        })
    }

    /// Serialize this engine's scene to bounded JSON suitable for persistence
    /// or transport through MCP and WebMCP.
    pub fn to_scene_json(&self) -> Result<String, EngineError> {
        self.to_scene_json_with_scripts(Vec::new())
    }

    /// Serialize the scene together with source code for host-owned entity
    /// scripts. Lua globals and VM state are not serialized.
    pub fn to_scene_json_with_scripts(
        &self,
        scripts: Vec<SceneScript>,
    ) -> Result<String, EngineError> {
        let mut scene = self.to_scene()?;
        scene.scripts = scripts;
        validate_scene_limits(&scene)?;
        validate_scene_scripts(&scene)?;
        let json = serde_json::to_string(&scene)
            .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
        if json.len() > MAX_SCENE_JSON_BYTES {
            return Err(EngineError::InvalidScene(format!(
                "encoded scene exceeds the {MAX_SCENE_JSON_BYTES}-byte limit"
            )));
        }
        Ok(json)
    }

    /// Construct an engine from a parsed scene. Import validates all component
    /// data, references, and hierarchy. Script definitions remain available in
    /// the document for the hosting script runtime to attach.
    pub fn from_scene(scene: SceneDocument) -> Result<Self, EngineError> {
        validate_scene_limits(&scene)?;
        validate_scene_scripts(&scene)?;

        let mut engine = Self::new(EngineConfig {
            fixed_delta_seconds: scene.fixed_delta_seconds,
            gravity: scene.gravity,
        })?;
        engine.set_camera(scene.camera)?;
        engine.set_lighting(scene.lighting)?;

        let mut asset_ids = HashSet::with_capacity(scene.mesh_assets.len());
        let mut max_asset_id = 0;
        for asset in scene.mesh_assets {
            if asset.id == 0 || asset.id >= u64::MAX - 1 || !asset_ids.insert(asset.id) {
                return Err(EngineError::InvalidScene(format!(
                    "mesh asset ID {} is invalid or duplicated",
                    asset.id
                )));
            }
            let mesh = MeshData::new(
                asset.mesh.vertices().to_vec(),
                asset.mesh.indices().to_vec(),
            )
            .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
            engine
                .mesh_assets
                .push((MeshAssetId(asset.id), Arc::new(mesh)));
            max_asset_id = max_asset_id.max(asset.id);
        }
        engine.next_mesh_asset_id = max_asset_id + 1;

        let mut texture_ids = HashSet::with_capacity(scene.texture_assets.len());
        let mut max_texture_id = 0;
        for asset in scene.texture_assets {
            if asset.id == 0 || asset.id >= u64::MAX - 1 || !texture_ids.insert(asset.id) {
                return Err(EngineError::InvalidScene(format!(
                    "texture asset ID {} is invalid or duplicated",
                    asset.id
                )));
            }
            let texture = TextureData::new(
                asset.texture.width,
                asset.texture.height,
                asset.texture.rgba8,
            )?;
            engine
                .texture_assets
                .push((TextureAssetId(asset.id), Arc::new(texture)));
            max_texture_id = max_texture_id.max(asset.id);
        }
        engine.next_texture_asset_id = max_texture_id + 1;

        let mut entity_ids = HashSet::with_capacity(scene.entities.len());
        let mut max_entity_id = 0;
        for record in &scene.entities {
            if record.id == 0 || record.id >= u64::MAX - 1 || !entity_ids.insert(record.id) {
                return Err(EngineError::InvalidScene(format!(
                    "entity ID {} is invalid or duplicated",
                    record.id
                )));
            }
            let transform = record.transform.validated()?;
            if let Some(material) = record.material {
                validate_material(material)?;
            }
            if let Some(body) = record.rigid_body {
                validate_rigid_body(body)?;
            }
            if let Some(collider) = record.collider {
                validate_collider(collider)?;
            }
            if let Some(light) = record.point_light {
                PointLight3d::validated(light)?;
            }
            let source = match record.mesh {
                Some(SceneMeshSource::Primitive(kind)) => Some(MeshSource::Primitive(kind)),
                Some(SceneMeshSource::Asset(asset_id)) => {
                    if !asset_ids.contains(&asset_id) {
                        return Err(EngineError::InvalidScene(format!(
                            "entity {} references missing mesh asset {asset_id}",
                            record.id
                        )));
                    }
                    Some(MeshSource::Asset(MeshAssetId(asset_id)))
                }
                None => None,
            };
            if source.is_some() != record.material.is_some() {
                return Err(EngineError::InvalidScene(format!(
                    "entity {} must have both a mesh and material, or neither",
                    record.id
                )));
            }
            if let Some(texture_id) = record.base_color_texture {
                if source.is_none() || !texture_ids.contains(&texture_id) {
                    return Err(EngineError::InvalidScene(format!(
                        "entity {} references a missing texture asset or has no mesh",
                        record.id
                    )));
                }
            }

            let entity = engine.world.spawn((Name(record.name.clone()), transform));
            engine
                .world
                .insert(entity, (Visibility3d(record.visible),))
                .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
            if let (Some(source), Some(material)) = (source, record.material) {
                engine
                    .world
                    .insert(entity, (Mesh3d(source), material))
                    .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
            }
            if let Some(texture_id) = record.base_color_texture {
                engine
                    .world
                    .insert(entity, (BaseColorTexture(TextureAssetId(texture_id)),))
                    .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
            }
            if let Some(body) = record.rigid_body {
                engine
                    .world
                    .insert(entity, (body,))
                    .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
            }
            if let Some(collider) = record.collider {
                engine
                    .world
                    .insert(entity, (collider,))
                    .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
            }
            if let Some(light) = record.point_light {
                engine
                    .world
                    .insert(entity, (light,))
                    .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
            }
            engine.entity_ids.push((EntityId(record.id), entity));
            max_entity_id = max_entity_id.max(record.id);
        }
        engine.next_entity_id = max_entity_id + 1;

        for record in &scene.entities {
            if let Some(parent_id) = record.parent_id {
                engine
                    .set_parent(EntityId(record.id), Some(EntityId(parent_id)))
                    .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
            }
        }
        let mut animation_roots = HashSet::new();
        for saved in scene.animation_players {
            if saved.root_id == 0
                || !animation_roots.insert(saved.root_id)
                || saved.clips.is_empty()
                || saved.state.clip_index >= saved.clips.len()
                || saved.node_ids.len() != saved.base_transforms.len()
                || !saved.state.speed.is_finite()
                || saved.state.speed.abs() > 100.0
            {
                return Err(EngineError::InvalidScene(
                    "glTF animation player state is invalid".into(),
                ));
            }
            let root = EntityId(saved.root_id);
            engine.resolve(root)?;
            let nodes = saved
                .node_ids
                .into_iter()
                .map(|id| {
                    let entity = EntityId(id);
                    engine.resolve(entity)?;
                    Ok(entity)
                })
                .collect::<Result<Vec<_>, EngineError>>()?;
            let base_transforms = saved
                .base_transforms
                .into_iter()
                .map(Transform::validated)
                .collect::<Result<Vec<_>, _>>()?;
            let clip = &saved.clips[saved.state.clip_index];
            if !clip.duration_seconds.is_finite()
                || clip.duration_seconds < 0.0
                || !saved.state.time_seconds.is_finite()
                || saved.state.time_seconds < 0.0
                || saved.state.time_seconds > clip.duration_seconds
                || clip.channels.iter().any(|channel| {
                    channel.node_index >= nodes.len()
                        || channel.keyframes.is_empty()
                        || channel.keyframes.iter().any(|keyframe| {
                            !keyframe.time.is_finite()
                                || !keyframe.value.is_finite()
                                || !keyframe.in_tangent.is_finite()
                                || !keyframe.out_tangent.is_finite()
                        })
                        || channel
                            .keyframes
                            .windows(2)
                            .any(|pair| pair[0].time >= pair[1].time)
                })
            {
                return Err(EngineError::InvalidScene(
                    "glTF animation clip data is invalid".into(),
                ));
            }
            let player = GltfAnimationPlayer {
                root,
                nodes,
                base_transforms,
                clips: saved.clips,
                state: saved.state,
            };
            apply_gltf_animation(
                &mut engine.world,
                &engine.entity_ids,
                &player,
                player.state.time_seconds,
            );
            engine.gltf_animation_players.push(player);
        }
        Ok(engine)
    }

    /// Parse and validate a bounded JSON scene document.
    pub fn from_scene_json(json: &str) -> Result<Self, EngineError> {
        let scene = Self::parse_scene_json(json)?;
        Self::from_scene(scene)
    }

    /// Parse the bounded JSON document. Use [`Engine::from_scene`] to validate
    /// component values and references before importing it.
    pub fn parse_scene_json(json: &str) -> Result<SceneDocument, EngineError> {
        if json.len() > MAX_SCENE_JSON_BYTES {
            return Err(EngineError::InvalidScene(format!(
                "input scene exceeds the {MAX_SCENE_JSON_BYTES}-byte limit"
            )));
        }
        let scene: SceneDocument = serde_json::from_str(json)
            .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
        Ok(scene)
    }

    /// Replace all engine-owned scene state atomically after complete import
    /// validation. Existing state remains available if parsing fails. This
    /// low-level engine operation does not execute the document's saved scripts;
    /// script-aware hosts should restore them on a candidate engine first.
    pub fn load_scene_json(&mut self, json: &str) -> Result<(), EngineError> {
        let scene = Self::parse_scene_json(json)?;
        let replacement = Self::from_scene(scene)?;
        self.replace_scene(replacement);
        Ok(())
    }

    pub(crate) fn replace_scene(&mut self, mut replacement: Engine) {
        replacement.mesh_asset_generation = self.mesh_asset_generation.wrapping_add(1);
        replacement.texture_asset_generation = self.texture_asset_generation.wrapping_add(1);
        *self = replacement;
    }

    pub fn spawn(
        &mut self,
        name: impl Into<String>,
        transform: Transform,
    ) -> Result<EntityId, EngineError> {
        self.spawn_mesh(name, transform, MeshKind::Cube, Material3d::default())
    }

    pub fn spawn_mesh(
        &mut self,
        name: impl Into<String>,
        transform: Transform,
        mesh: MeshKind,
        material: Material3d,
    ) -> Result<EntityId, EngineError> {
        let transform = transform.validated()?;
        validate_material(material)?;
        let id = self.allocate_entity_id();
        let entity = self.world.spawn((
            Name(name.into()),
            transform,
            Mesh3d(MeshSource::Primitive(mesh)),
            material,
            Visibility3d::default(),
        ));
        self.entity_ids.push((id, entity));
        Ok(id)
    }

    pub fn add_mesh_asset(&mut self, mesh: MeshData) -> MeshAssetId {
        let id = MeshAssetId(self.next_mesh_asset_id);
        self.next_mesh_asset_id = self.next_mesh_asset_id.wrapping_add(1).max(1);
        self.mesh_assets.push((id, Arc::new(mesh)));
        self.mesh_asset_generation = self.mesh_asset_generation.wrapping_add(1);
        id
    }

    /// Add a bounded sRGBA8 texture asset and return its stable scene ID.
    pub fn add_texture_asset(
        &mut self,
        width: u32,
        height: u32,
        rgba8: Vec<u8>,
    ) -> Result<TextureAssetId, EngineError> {
        let texture = TextureData::new(width, height, rgba8)?;
        let current_bytes = self
            .texture_assets
            .iter()
            .map(|(_, texture)| texture.rgba8.len())
            .sum::<usize>();
        if self.texture_assets.len() >= MAX_SCENE_TEXTURE_ASSETS
            || current_bytes.saturating_add(texture.rgba8.len()) > MAX_SCENE_TEXTURE_BYTES
        {
            return Err(EngineError::InvalidTexture);
        }
        let id = TextureAssetId(self.next_texture_asset_id);
        self.next_texture_asset_id = self.next_texture_asset_id.wrapping_add(1).max(1);
        self.texture_assets.push((id, Arc::new(texture)));
        self.texture_asset_generation = self.texture_asset_generation.wrapping_add(1);
        Ok(id)
    }

    /// Decode a bounded PNG, JPEG, WebP, BMP, GIF, TIFF, or PNM image into an
    /// sRGBA8 scene texture asset. Animated formats use their first frame.
    pub fn add_texture_asset_from_encoded(
        &mut self,
        encoded: &[u8],
    ) -> Result<TextureAssetId, EngineError> {
        let texture = TextureData::from_encoded(encoded)?;
        self.add_texture_asset(texture.width, texture.height, texture.rgba8)
    }

    /// Read a CPU texture asset by stable ID.
    pub fn texture_asset(&self, id: TextureAssetId) -> Option<&TextureData> {
        self.texture_assets
            .iter()
            .find_map(|(asset_id, texture)| (*asset_id == id).then_some(texture.as_ref()))
    }

    pub fn texture_assets(&self) -> impl Iterator<Item = (TextureAssetId, &TextureData)> {
        self.texture_assets
            .iter()
            .map(|(id, texture)| (*id, texture.as_ref()))
    }

    pub fn texture_asset_count(&self) -> usize {
        self.texture_assets.len()
    }

    /// Assign or clear an entity's base-color texture. Textures are sampled
    /// using the mesh's UV coordinates and repeat in both axes.
    pub fn set_base_color_texture(
        &mut self,
        id: EntityId,
        texture: Option<TextureAssetId>,
    ) -> Result<(), EngineError> {
        let entity = self.resolve(id)?;
        if self.world.get::<&Material3d>(entity).is_err() {
            return Err(EngineError::MissingMaterial(id.get()));
        }
        if let Some(texture) = texture {
            if self.texture_asset(texture).is_none() {
                return Err(EngineError::UnknownTextureAsset(texture.get()));
            }
            self.world
                .insert(entity, (BaseColorTexture(texture),))
                .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
        } else {
            let _ = self.world.remove_one::<BaseColorTexture>(entity);
        }
        Ok(())
    }

    pub fn base_color_texture(&self, id: EntityId) -> Result<Option<TextureAssetId>, EngineError> {
        let entity = self.resolve(id)?;
        if self.world.get::<&Material3d>(entity).is_err() {
            return Err(EngineError::MissingMaterial(id.get()));
        }
        Ok(self
            .world
            .get::<&BaseColorTexture>(entity)
            .ok()
            .map(|value| value.0))
    }

    pub fn spawn_mesh_asset(
        &mut self,
        name: impl Into<String>,
        transform: Transform,
        mesh: MeshAssetId,
        material: Material3d,
    ) -> Result<EntityId, EngineError> {
        if self.mesh_asset(mesh).is_none() {
            return Err(EngineError::UnknownMeshAsset(mesh.get()));
        }
        let transform = transform.validated()?;
        validate_material(material)?;
        let id = self.allocate_entity_id();
        let entity = self.world.spawn((
            Name(name.into()),
            transform,
            Mesh3d(MeshSource::Asset(mesh)),
            material,
            Visibility3d::default(),
        ));
        self.entity_ids.push((id, entity));
        Ok(id)
    }

    /// Spawn an imported glTF scene as a transform root with one renderable
    /// child per primitive, preserving base-color PBR values and textures.
    pub fn spawn_gltf_asset(
        &mut self,
        name: impl Into<String>,
        position: Vec3,
        asset: GltfAssetData,
    ) -> Result<GltfSceneInstance, EngineError> {
        let name = name.into();
        let root_transform = Transform {
            translation: position,
            ..Transform::default()
        }
        .validated()?;
        if self
            .mesh_assets
            .len()
            .saturating_add(asset.primitives.len())
            > MAX_SCENE_MESH_ASSETS
            || self
                .entity_ids
                .len()
                .saturating_add(asset.primitives.len() + asset.nodes.len() + 1)
                > MAX_SCENE_ENTITIES
        {
            return Err(EngineError::InvalidScene(
                "glTF import would exceed the scene mesh or entity limit".into(),
            ));
        }
        let mut used_images = HashSet::new();
        let materials = asset
            .primitives
            .iter()
            .map(|primitive| {
                if let Some(image) = primitive.base_color_image {
                    if image >= asset.images.len() {
                        return Err(EngineError::InvalidScene(
                            "glTF base-color image index is out of bounds".into(),
                        ));
                    }
                    used_images.insert(image);
                }
                let material = Material3d {
                    base_color: primitive.base_color,
                    roughness: primitive.roughness,
                    metallic: primitive.metallic,
                    alpha_mode: match primitive.alpha_mode {
                        gltf::material::AlphaMode::Opaque => AlphaMode3d::Opaque,
                        gltf::material::AlphaMode::Mask => AlphaMode3d::Mask,
                        gltf::material::AlphaMode::Blend => AlphaMode3d::Blend,
                    },
                    alpha_cutoff: primitive.alpha_cutoff,
                };
                validate_material(material)?;
                Ok(material)
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        let mut decoded_images = Vec::with_capacity(used_images.len());
        for image_index in used_images {
            decoded_images.push((
                image_index,
                TextureData::from_encoded(&asset.images[image_index])?,
            ));
        }
        let existing_texture_bytes = self
            .texture_assets
            .iter()
            .map(|(_, texture)| texture.rgba8.len())
            .sum::<usize>();
        let imported_texture_bytes = decoded_images
            .iter()
            .map(|(_, texture)| texture.rgba8.len())
            .sum::<usize>();
        if self
            .texture_assets
            .len()
            .saturating_add(decoded_images.len())
            > MAX_SCENE_TEXTURE_ASSETS
            || existing_texture_bytes.saturating_add(imported_texture_bytes)
                > MAX_SCENE_TEXTURE_BYTES
        {
            return Err(EngineError::InvalidTexture);
        }

        let root = self.spawn_empty(name.clone(), root_transform)?;
        let mut node_entities = Vec::with_capacity(asset.nodes.len());
        for node in &asset.nodes {
            let matrix = Mat4::from_cols_array(&node.transform);
            let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
            let transform = Transform {
                translation,
                rotation,
                scale,
            }
            .validated()?;
            let entity = self.spawn_empty(format!("{name} / {}", node.name), transform)?;
            let parent = match node.parent {
                Some(index) => node_entities.get(index).copied().ok_or_else(|| {
                    EngineError::InvalidScene("glTF node parent is out of range".into())
                })?,
                None => root,
            };
            self.set_parent(entity, Some(parent))?;
            node_entities.push(entity);
        }
        let mut texture_ids = vec![None; asset.images.len()];
        for (image_index, texture) in decoded_images {
            let id = self.add_texture_asset(texture.width, texture.height, texture.rgba8)?;
            texture_ids[image_index] = Some(id);
        }
        let mut primitive_ids = Vec::with_capacity(asset.primitives.len());
        let mut mesh_assets = Vec::with_capacity(asset.primitives.len());
        for (primitive, material) in asset.primitives.into_iter().zip(materials) {
            let mesh_id = self.add_mesh_asset(primitive.mesh);
            let child = self.spawn_mesh_asset(
                format!("{name} / {}", primitive.name),
                Transform::default(),
                mesh_id,
                material,
            )?;
            let parent = node_entities
                .get(primitive.node_index)
                .copied()
                .ok_or_else(|| {
                    EngineError::InvalidScene("glTF primitive node is out of range".into())
                })?;
            self.set_parent(child, Some(parent))?;
            if let Some(image_index) = primitive.base_color_image {
                self.set_base_color_texture(child, texture_ids[image_index])?;
            }
            primitive_ids.push(child);
            mesh_assets.push(mesh_id);
        }
        let animation_names = asset
            .animations
            .iter()
            .map(|animation| animation.name.clone())
            .collect::<Vec<_>>();
        if !asset.animations.is_empty() {
            let base_transforms = node_entities
                .iter()
                .map(|id| self.transform(*id))
                .collect::<Result<Vec<_>, _>>()?;
            let first_animation_name = asset.animations[0].name.clone();
            let player = GltfAnimationPlayer {
                root,
                nodes: node_entities.clone(),
                base_transforms,
                clips: asset.animations,
                state: GltfAnimationState {
                    clip_index: 0,
                    clip_name: first_animation_name,
                    time_seconds: 0.0,
                    speed: 1.0,
                    looping: true,
                    playing: true,
                },
            };
            apply_gltf_animation(&mut self.world, &self.entity_ids, &player, 0.0);
            self.gltf_animation_players.push(player);
        }
        Ok(GltfSceneInstance {
            root,
            nodes: node_entities,
            primitives: primitive_ids,
            mesh_assets,
            animations: animation_names,
        })
    }

    pub fn spawn_empty(
        &mut self,
        name: impl Into<String>,
        transform: Transform,
    ) -> Result<EntityId, EngineError> {
        let transform = transform.validated()?;
        let id = self.allocate_entity_id();
        let entity = self
            .world
            .spawn((Name(name.into()), transform, Visibility3d::default()));
        self.entity_ids.push((id, entity));
        Ok(id)
    }

    /// Build a recognizable, stylized Donald Trump head from engine primitives.
    /// The returned face handle can be textured or recolored like any other mesh.
    pub fn spawn_donald_trump_caricature(
        &mut self,
        position: Vec3,
    ) -> Result<CaricatureHead, EngineError> {
        let root = self.spawn_empty(
            "Donald Trump caricature head",
            Transform {
                translation: position,
                ..Transform::default()
            },
        )?;
        let face = self.spawn_caricature_part(
            root,
            "Face",
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.73, 0.93, 0.59),
            [0.91, 0.48, 0.22, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Neck",
            Vec3::new(0.0, -0.91, -0.01),
            Vec3::new(0.31, 0.36, 0.34),
            [0.78, 0.37, 0.17, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Left ear",
            Vec3::new(-0.72, -0.04, 0.0),
            Vec3::new(0.17, 0.27, 0.16),
            [0.84, 0.41, 0.19, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Right ear",
            Vec3::new(0.72, -0.04, 0.0),
            Vec3::new(0.17, 0.27, 0.16),
            [0.84, 0.41, 0.19, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Nose bridge",
            Vec3::new(0.0, -0.04, 0.53),
            Vec3::new(0.18, 0.39, 0.22),
            [0.96, 0.56, 0.28, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Nose tip",
            Vec3::new(0.0, -0.30, 0.67),
            Vec3::new(0.24, 0.17, 0.21),
            [0.96, 0.56, 0.28, 1.0],
        )?;

        for (side, x) in [("Left", -0.29), ("Right", 0.29)] {
            self.spawn_caricature_part(
                root,
                format!("{side} eye white"),
                Vec3::new(x, 0.19, 0.49),
                Vec3::new(0.20, 0.13, 0.10),
                [0.97, 0.91, 0.77, 1.0],
            )?;
            self.spawn_caricature_part(
                root,
                format!("{side} blue iris"),
                Vec3::new(x, 0.18, 0.579),
                Vec3::new(0.071, 0.087, 0.035),
                [0.12, 0.42, 0.72, 1.0],
            )?;
            self.spawn_caricature_part(
                root,
                format!("{side} pupil"),
                Vec3::new(x, 0.18, 0.607),
                Vec3::new(0.031, 0.045, 0.018),
                [0.035, 0.025, 0.02, 1.0],
            )?;
            let brow_x = x * 1.03;
            self.spawn_caricature_part(
                root,
                format!("{side} heavy brow"),
                Vec3::new(brow_x, 0.39, 0.51),
                Vec3::new(0.25, 0.075, 0.095),
                [0.48, 0.24, 0.10, 1.0],
            )?;
        }
        // A raised, swept golden forelock plus the side hairline.
        self.spawn_caricature_part(
            root,
            "Golden hair cap",
            Vec3::new(0.03, 0.78, -0.02),
            Vec3::new(0.77, 0.33, 0.61),
            [0.97, 0.66, 0.08, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Swept blond forelock",
            Vec3::new(0.22, 0.87, 0.25),
            Vec3::new(0.62, 0.17, 0.43),
            [1.0, 0.76, 0.12, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Hair sweep tip",
            Vec3::new(0.65, 0.77, 0.20),
            Vec3::new(0.28, 0.14, 0.29),
            [0.91, 0.54, 0.035, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Left side hair",
            Vec3::new(-0.64, 0.49, 0.04),
            Vec3::new(0.15, 0.39, 0.52),
            [0.91, 0.56, 0.04, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Right side hair",
            Vec3::new(0.64, 0.49, 0.04),
            Vec3::new(0.15, 0.39, 0.52),
            [0.91, 0.56, 0.04, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Smiling mouth",
            Vec3::new(0.0, -0.54, 0.535),
            Vec3::new(0.31, 0.055, 0.05),
            [0.29, 0.055, 0.035, 1.0],
        )?;
        self.spawn_caricature_part(
            root,
            "Lower lip",
            Vec3::new(0.0, -0.62, 0.51),
            Vec3::new(0.23, 0.055, 0.045),
            [0.67, 0.22, 0.13, 1.0],
        )?;
        Ok(CaricatureHead { root, face })
    }

    fn spawn_caricature_part(
        &mut self,
        root: EntityId,
        name: impl Into<String>,
        translation: Vec3,
        scale: Vec3,
        color: [f32; 4],
    ) -> Result<EntityId, EngineError> {
        let part = self.spawn_mesh(
            format!("Trump caricature {}", name.into()),
            Transform {
                translation,
                scale,
                ..Transform::default()
            },
            MeshKind::Sphere,
            Material3d {
                base_color: color,
                roughness: 0.8,
                metallic: 0.0,
                ..Material3d::default()
            },
        )?;
        self.set_parent(part, Some(root))?;
        Ok(part)
    }

    pub fn despawn(&mut self, id: EntityId) -> Result<(), EngineError> {
        let index = self
            .entity_ids
            .iter()
            .position(|(candidate, _)| *candidate == id)
            .ok_or(EngineError::UnknownEntity(id.0))?;
        let detached_children = self
            .parents
            .iter()
            .filter_map(|(child, parent)| {
                (*parent == id).then(|| {
                    self.world_transform(*child)
                        .map(|transform| (*child, transform))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (_, entity) = self.entity_ids.swap_remove(index);
        self.gltf_animation_players
            .retain(|player| player.root != id && !player.nodes.contains(&id));
        self.world
            .despawn(entity)
            .map_err(|_| EngineError::UnknownEntity(id.0))?;
        // Keep children alive as roots at their previous world-space poses.
        self.parents
            .retain(|(child, parent)| *child != id && *parent != id);
        for (child, world_transform) in detached_children {
            self.set_transform(child, world_transform)?;
        }
        Ok(())
    }

    /// Set an entity's local visibility flag.
    pub fn set_visibility(&mut self, id: EntityId, visible: bool) -> Result<(), EngineError> {
        let entity = self.resolve(id)?;
        self.world
            .insert(entity, (Visibility3d(visible),))
            .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
        Ok(())
    }

    /// Return the entity's local visibility flag. Missing visibility
    /// components are treated as visible for compatibility.
    pub fn visibility(&self, id: EntityId) -> Result<bool, EngineError> {
        let entity = self.resolve(id)?;
        Ok(self
            .world
            .get::<&Visibility3d>(entity)
            .map(|visibility| visibility.0)
            .unwrap_or(true))
    }

    /// Return whether this entity and all of its ancestors are locally visible.
    pub fn is_visible(&self, id: EntityId) -> Result<bool, EngineError> {
        let mut current = Some(id);
        while let Some(entity_id) = current {
            if !self.visibility(entity_id)? {
                return Ok(false);
            }
            current = self
                .parents
                .iter()
                .find_map(|(child, parent)| (*child == entity_id).then_some(*parent));
        }
        Ok(true)
    }

    /// Add, replace, or remove the point light attached to an entity.
    pub fn set_point_light(
        &mut self,
        id: EntityId,
        light: Option<PointLight3d>,
    ) -> Result<(), EngineError> {
        let entity = self.resolve(id)?;
        if let Some(light) = light {
            self.world
                .insert(entity, (light.validated()?,))
                .map_err(|error| EngineError::InvalidScene(error.to_string()))?;
        } else {
            let _ = self.world.remove_one::<PointLight3d>(entity);
        }
        Ok(())
    }

    pub fn point_light(&self, id: EntityId) -> Result<Option<PointLight3d>, EngineError> {
        let entity = self.resolve(id)?;
        Ok(self
            .world
            .get::<&PointLight3d>(entity)
            .ok()
            .map(|light| *light))
    }

    /// Iterate effectively visible point lights in world space.
    pub fn point_lights(&self) -> impl Iterator<Item = (EntityId, Vec3, PointLight3d)> + '_ {
        self.entity_ids.iter().filter_map(|(id, entity)| {
            if !self.is_visible(*id).ok()? {
                return None;
            }
            let light = *self.world.get::<&PointLight3d>(*entity).ok()?;
            let position = self.world_matrix(*id).ok()?.w_axis.truncate();
            Some((*id, position, light))
        })
    }

    /// Return an entity's direct parent, if it has one.
    pub fn parent(&self, id: EntityId) -> Result<Option<EntityId>, EngineError> {
        self.resolve(id)?;
        Ok(self
            .parents
            .iter()
            .find_map(|(child, parent)| (*child == id).then_some(*parent)))
    }

    /// Reparent an entity while preserving its local transform. Passing `None`
    /// makes it a root. Parenting is rejected if it would create a cycle.
    pub fn set_parent(
        &mut self,
        child: EntityId,
        parent: Option<EntityId>,
    ) -> Result<(), EngineError> {
        self.resolve(child)?;
        if let Some(parent) = parent {
            self.resolve(parent)?;
            let mut ancestor = Some(parent);
            while let Some(current) = ancestor {
                if current == child {
                    return Err(EngineError::ParentCycle);
                }
                ancestor = self
                    .parents
                    .iter()
                    .find_map(|(candidate, parent)| (*candidate == current).then_some(*parent));
            }
        }

        self.parents.retain(|(candidate, _)| *candidate != child);
        if let Some(parent) = parent {
            self.parents.push((child, parent));
        }
        Ok(())
    }

    /// Return the entity's world matrix after composing all ancestor transforms.
    pub fn world_matrix(&self, id: EntityId) -> Result<Mat4, EngineError> {
        self.resolve(id)?;
        let mut chain = vec![id];
        let mut current = id;
        while let Some(parent) = self
            .parents
            .iter()
            .find_map(|(child, parent)| (*child == current).then_some(*parent))
        {
            chain.push(parent);
            current = parent;
        }

        let mut world = Mat4::IDENTITY;
        for entity_id in chain.into_iter().rev() {
            let transform = self.transform(entity_id)?;
            world *= Mat4::from_scale_rotation_translation(
                transform.scale,
                transform.rotation,
                transform.translation,
            );
        }
        Ok(world)
    }

    /// Return the composed world-space transform. Use [`Engine::world_matrix`]
    /// when exact affine composition (including shear) must be preserved.
    pub fn world_transform(&self, id: EntityId) -> Result<Transform, EngineError> {
        let (scale, rotation, translation) = self.world_matrix(id)?.to_scale_rotation_translation();
        Ok(Transform {
            translation,
            rotation,
            scale,
        })
    }

    pub fn set_transform(&mut self, id: EntityId, transform: Transform) -> Result<(), EngineError> {
        let transform = transform.validated()?;
        let entity = self.resolve(id)?;
        *self
            .world
            .get::<&mut Transform>(entity)
            .map_err(|_| EngineError::UnknownEntity(id.0))? = transform;
        Ok(())
    }

    pub fn set_material(&mut self, id: EntityId, material: Material3d) -> Result<(), EngineError> {
        validate_material(material)?;
        let entity = self.resolve(id)?;
        *self
            .world
            .get::<&mut Material3d>(entity)
            .map_err(|_| EngineError::MissingMaterial(id.0))? = material;
        Ok(())
    }

    pub fn add_rigid_body(&mut self, id: EntityId, body: RigidBody3d) -> Result<(), EngineError> {
        validate_rigid_body(body)?;
        let entity = self.resolve(id)?;
        self.world
            .insert(entity, (body,))
            .map_err(|_| EngineError::UnknownEntity(id.0))?;
        Ok(())
    }

    pub fn rigid_body(&self, id: EntityId) -> Result<Option<RigidBody3d>, EngineError> {
        let entity = self.resolve(id)?;
        Ok(self
            .world
            .get::<&RigidBody3d>(entity)
            .ok()
            .map(|body| *body))
    }

    pub fn set_collider(&mut self, id: EntityId, collider: Collider3d) -> Result<(), EngineError> {
        validate_collider(collider)?;
        let entity = self.resolve(id)?;
        self.world
            .insert(entity, (collider,))
            .map_err(|_| EngineError::UnknownEntity(id.0))?;
        Ok(())
    }

    pub fn collider(&self, id: EntityId) -> Result<Option<Collider3d>, EngineError> {
        let entity = self.resolve(id)?;
        Ok(self
            .world
            .get::<&Collider3d>(entity)
            .ok()
            .map(|collider| *collider))
    }

    pub fn remove_collider(&mut self, id: EntityId) -> Result<bool, EngineError> {
        let entity = self.resolve(id)?;
        Ok(self.world.remove_one::<Collider3d>(entity).is_ok())
    }

    pub fn remove_rigid_body(&mut self, id: EntityId) -> Result<bool, EngineError> {
        let entity = self.resolve(id)?;
        Ok(self.world.remove_one::<RigidBody3d>(entity).is_ok())
    }

    pub fn set_velocity(&mut self, id: EntityId, velocity: Vec3) -> Result<(), EngineError> {
        if !velocity.is_finite() {
            return Err(EngineError::InvalidRigidBody);
        }
        let entity = self.resolve(id)?;
        let mut body = self
            .world
            .get::<&mut RigidBody3d>(entity)
            .map_err(|_| EngineError::MissingRigidBody(id.0))?;
        body.velocity = velocity;
        Ok(())
    }

    pub fn apply_force(&mut self, id: EntityId, force: Vec3) -> Result<(), EngineError> {
        if !force.is_finite() {
            return Err(EngineError::InvalidRigidBody);
        }
        let entity = self.resolve(id)?;
        let mut body = self
            .world
            .get::<&mut RigidBody3d>(entity)
            .map_err(|_| EngineError::MissingRigidBody(id.0))?;
        let accumulated_force = body.accumulated_force + force;
        if !accumulated_force.is_finite() {
            return Err(EngineError::InvalidRigidBody);
        }
        body.accumulated_force = accumulated_force;
        Ok(())
    }

    pub fn camera(&self) -> Camera3d {
        self.active_camera
    }

    pub fn set_camera(&mut self, camera: Camera3d) -> Result<(), EngineError> {
        self.active_camera = camera.validated()?;
        Ok(())
    }

    pub fn lighting(&self) -> Lighting3d {
        self.lighting
    }

    pub fn set_lighting(&mut self, lighting: Lighting3d) -> Result<(), EngineError> {
        self.lighting = lighting.validated()?;
        Ok(())
    }

    pub fn gravity(&self) -> Vec3 {
        self.config.gravity
    }

    pub fn set_gravity(&mut self, gravity: Vec3) -> Result<(), EngineError> {
        if !gravity.is_finite() {
            return Err(EngineError::InvalidGravity);
        }
        self.config.gravity = gravity;
        Ok(())
    }

    pub fn transform(&self, id: EntityId) -> Result<Transform, EngineError> {
        let entity = self.resolve(id)?;
        self.world
            .get::<&Transform>(entity)
            .map(|transform| *transform)
            .map_err(|_| EngineError::UnknownEntity(id.0))
    }

    pub fn material(&self, id: EntityId) -> Result<Material3d, EngineError> {
        let entity = self.resolve(id)?;
        self.world
            .get::<&Material3d>(entity)
            .map(|material| *material)
            .map_err(|_| EngineError::MissingMaterial(id.0))
    }

    pub fn name(&self, id: EntityId) -> Result<String, EngineError> {
        let entity = self.resolve(id)?;
        self.world
            .get::<&Name>(entity)
            .map(|name| name.0.clone())
            .map_err(|_| EngineError::UnknownEntity(id.0))
    }

    pub fn entities(&self) -> impl Iterator<Item = (EntityId, String, Transform)> + '_ {
        self.entity_ids.iter().filter_map(|(id, entity)| {
            let name = self.world.get::<&Name>(*entity).ok()?;
            let transform = self.world.get::<&Transform>(*entity).ok()?;
            Some((*id, name.0.clone(), *transform))
        })
    }

    pub fn renderables(
        &self,
    ) -> impl Iterator<
        Item = (
            EntityId,
            Mat4,
            MeshSource,
            Material3d,
            Option<TextureAssetId>,
        ),
    > + '_ {
        self.entity_ids.iter().filter_map(|(id, entity)| {
            if !self.is_visible(*id).ok()? {
                return None;
            }
            let _transform = self.world.get::<&Transform>(*entity).ok()?;
            let mesh = self.world.get::<&Mesh3d>(*entity).ok()?;
            let material = self.world.get::<&Material3d>(*entity).ok()?;
            let texture = self
                .world
                .get::<&BaseColorTexture>(*entity)
                .ok()
                .map(|v| v.0);
            Some((
                *id,
                self.world_matrix(*id).ok()?,
                mesh.0,
                *material,
                texture,
            ))
        })
    }

    pub fn mesh_asset(&self, id: MeshAssetId) -> Option<&MeshData> {
        self.mesh_assets
            .iter()
            .find_map(|(asset_id, data)| (*asset_id == id).then_some(data.as_ref()))
    }

    pub fn mesh_asset_count(&self) -> usize {
        self.mesh_assets.len()
    }

    /// Changes whenever mesh or texture asset identities may no longer match
    /// cached renderer resources, including after a scene replacement.
    pub fn mesh_asset_generation(&self) -> u64 {
        self.mesh_asset_generation
    }

    pub fn texture_asset_generation(&self) -> u64 {
        self.texture_asset_generation
    }

    pub fn gltf_animation_names(&self, root: EntityId) -> Result<Vec<String>, EngineError> {
        self.gltf_animation_players
            .iter()
            .find(|player| player.root == root)
            .map(|player| player.clips.iter().map(|clip| clip.name.clone()).collect())
            .ok_or(EngineError::UnknownAnimationPlayer(root.get()))
    }

    pub fn gltf_animation_state(&self, root: EntityId) -> Result<GltfAnimationState, EngineError> {
        self.gltf_animation_players
            .iter()
            .find(|player| player.root == root)
            .map(|player| player.state.clone())
            .ok_or(EngineError::UnknownAnimationPlayer(root.get()))
    }

    pub fn play_gltf_animation(
        &mut self,
        root: EntityId,
        clip_index: usize,
        looping: bool,
    ) -> Result<GltfAnimationState, EngineError> {
        let player = self
            .gltf_animation_players
            .iter_mut()
            .find(|player| player.root == root)
            .ok_or(EngineError::UnknownAnimationPlayer(root.get()))?;
        let clip = player
            .clips
            .get(clip_index)
            .ok_or(EngineError::UnknownAnimationClip(clip_index))?;
        player.state.clip_index = clip_index;
        player.state.clip_name = clip.name.clone();
        player.state.time_seconds = if player.state.speed < 0.0 {
            clip.duration_seconds
        } else {
            0.0
        };
        player.state.looping = looping;
        player.state.playing = true;
        let state = player.state.clone();
        apply_gltf_animation(
            &mut self.world,
            &self.entity_ids,
            player,
            player.state.time_seconds,
        );
        Ok(state)
    }

    pub fn stop_gltf_animation(
        &mut self,
        root: EntityId,
    ) -> Result<GltfAnimationState, EngineError> {
        let player = self
            .gltf_animation_players
            .iter_mut()
            .find(|player| player.root == root)
            .ok_or(EngineError::UnknownAnimationPlayer(root.get()))?;
        player.state.playing = false;
        Ok(player.state.clone())
    }

    pub fn set_gltf_animation_speed(
        &mut self,
        root: EntityId,
        speed: f32,
    ) -> Result<GltfAnimationState, EngineError> {
        if !speed.is_finite() || speed.abs() > 100.0 {
            return Err(EngineError::InvalidAnimationSpeed);
        }
        let player = self
            .gltf_animation_players
            .iter_mut()
            .find(|player| player.root == root)
            .ok_or(EngineError::UnknownAnimationPlayer(root.get()))?;
        if speed < 0.0 && player.state.time_seconds <= 0.0 {
            player.state.time_seconds = player.clips[player.state.clip_index].duration_seconds;
        } else if speed > 0.0
            && player.state.time_seconds >= player.clips[player.state.clip_index].duration_seconds
        {
            player.state.time_seconds = 0.0;
        }
        player.state.speed = speed;
        Ok(player.state.clone())
    }

    /// Advance by elapsed wall time, returning the number of fixed ticks run.
    /// The caller controls frame pacing, so this method is usable without an OS loop.
    pub fn advance(&mut self, delta_seconds: f64, max_steps: u32) -> u32 {
        if !delta_seconds.is_finite() || delta_seconds <= 0.0 {
            return 0;
        }
        self.elapsed_seconds += delta_seconds;
        let step = self.config.fixed_delta_seconds;
        let mut count = 0;
        while self.elapsed_seconds >= step && count < max_steps {
            self.elapsed_seconds -= step;
            self.integrate_physics(step as f32);
            self.advance_gltf_animations(step as f32);
            self.fixed_steps += 1;
            count += 1;
        }
        count
    }

    fn advance_gltf_animations(&mut self, delta_seconds: f32) {
        let mut players = std::mem::take(&mut self.gltf_animation_players);
        for player in &mut players {
            if !player.state.playing {
                continue;
            }
            let duration = player.clips[player.state.clip_index].duration_seconds;
            let time = player.state.time_seconds + delta_seconds * player.state.speed;
            if duration <= 0.0 {
                player.state.time_seconds = 0.0;
                if !player.state.looping {
                    player.state.playing = false;
                }
            } else if player.state.looping {
                player.state.time_seconds = time.rem_euclid(duration);
            } else if time >= duration {
                player.state.time_seconds = duration;
                player.state.playing = false;
            } else if time <= 0.0 {
                player.state.time_seconds = 0.0;
                if player.state.speed < 0.0 {
                    player.state.playing = false;
                }
            } else {
                player.state.time_seconds = time;
            }
            apply_gltf_animation(
                &mut self.world,
                &self.entity_ids,
                player,
                player.state.time_seconds,
            );
        }
        self.gltf_animation_players = players;
    }

    fn integrate_physics(&mut self, delta_seconds: f32) {
        let gravity = self.config.gravity;
        for (id, entity) in &self.entity_ids {
            let parent_world = self
                .parents
                .iter()
                .find_map(|(child, parent)| (*child == *id).then_some(*parent))
                .and_then(|parent| self.world_matrix(parent).ok());
            let parent_inverse = parent_world.map(|matrix| matrix.inverse());
            let Ok(mut body) = self.world.get::<&mut RigidBody3d>(*entity) else {
                continue;
            };
            let body_type = body.body_type;
            let damping = (-body.linear_damping * delta_seconds).exp();
            let velocity = match body_type {
                RigidBodyType::Static => {
                    body.accumulated_force = Vec3::ZERO;
                    continue;
                }
                RigidBodyType::Kinematic => {
                    body.accumulated_force = Vec3::ZERO;
                    body.velocity *= damping;
                    body.velocity
                }
                RigidBodyType::Dynamic => {
                    let acceleration =
                        body.accumulated_force / body.mass + gravity * body.gravity_scale;
                    body.velocity += acceleration * delta_seconds;
                    body.velocity *= damping;
                    body.accumulated_force = Vec3::ZERO;
                    body.velocity
                }
            };
            if !velocity.is_finite() {
                body.velocity = Vec3::ZERO;
                body.accumulated_force = Vec3::ZERO;
                continue;
            }
            drop(body);
            if let Ok(mut transform) = self.world.get::<&mut Transform>(*entity) {
                if let (Some(parent_world), Some(parent_inverse)) = (parent_world, parent_inverse) {
                    if parent_inverse.is_finite() {
                        let world_position = parent_world.transform_point3(transform.translation)
                            + velocity * delta_seconds;
                        let local_position = parent_inverse.transform_point3(world_position);
                        if local_position.is_finite() {
                            transform.translation = local_position;
                        }
                    }
                } else {
                    let position = transform.translation + velocity * delta_seconds;
                    if position.is_finite() {
                        transform.translation = position;
                    }
                }
            }
        }
        self.resolve_collisions();
    }

    fn resolve_collisions(&mut self) {
        let mut colliders = self
            .entity_ids
            .iter()
            .filter_map(|(id, entity)| {
                let collider = *self.world.get::<&Collider3d>(*entity).ok()?;
                let world = world_collider(self.world_matrix(*id).ok()?, collider.shape)?;
                Some((*id, collider, world))
            })
            .collect::<Vec<_>>();

        // Sort on the x axis and stop each scan as soon as the next AABB starts
        // beyond the current one. This keeps sparse scenes near O(n log n)
        // instead of sending every pair through narrow-phase collision tests.
        colliders.sort_unstable_by(|(a_id, _, a), (b_id, _, b)| {
            a.min
                .x
                .total_cmp(&b.min.x)
                .then_with(|| a_id.get().cmp(&b_id.get()))
        });

        for a_index in 0..colliders.len() {
            for b_index in (a_index + 1)..colliders.len() {
                let (a_id, a_material, a) = colliders[a_index];
                let (b_id, b_material, b) = colliders[b_index];
                if b.min.x > a.max.x {
                    break;
                }
                if b.min.y > a.max.y || b.max.y < a.min.y || b.min.z > a.max.z || b.max.z < a.min.z
                {
                    continue;
                }
                if inverse_mass(self.rigid_body(a_id).ok().flatten()) == 0.0
                    && inverse_mass(self.rigid_body(b_id).ok().flatten()) == 0.0
                {
                    continue;
                }
                let Some(contact) = collision_contact(a, b) else {
                    continue;
                };
                self.resolve_contact(a_id, b_id, contact, a_material, b_material);
            }
        }
    }

    fn resolve_contact(
        &mut self,
        a_id: EntityId,
        b_id: EntityId,
        contact: CollisionContact,
        a_collider: Collider3d,
        b_collider: Collider3d,
    ) {
        let a_body = self.rigid_body(a_id).ok().flatten();
        let b_body = self.rigid_body(b_id).ok().flatten();
        let a_inverse_mass = inverse_mass(a_body);
        let b_inverse_mass = inverse_mass(b_body);
        let inverse_mass_sum = a_inverse_mass + b_inverse_mass;
        if inverse_mass_sum <= 0.0 {
            return;
        }

        let correction =
            contact.normal * ((contact.penetration - 0.001).max(0.0) * 0.8 / inverse_mass_sum);
        if a_inverse_mass > 0.0 {
            self.translate_world(a_id, -correction * a_inverse_mass);
        }
        if b_inverse_mass > 0.0 {
            self.translate_world(b_id, correction * b_inverse_mass);
        }

        let a_velocity = a_body.map_or(Vec3::ZERO, |body| body.velocity);
        let b_velocity = b_body.map_or(Vec3::ZERO, |body| body.velocity);
        let relative_velocity = b_velocity - a_velocity;
        let normal_speed = relative_velocity.dot(contact.normal);
        if normal_speed >= 0.0 {
            return;
        }
        let restitution = a_collider.restitution.min(b_collider.restitution);
        let normal_impulse = -(1.0 + restitution) * normal_speed / inverse_mass_sum;
        let impulse = contact.normal * normal_impulse;
        self.apply_velocity_impulse(a_id, -impulse, a_inverse_mass);
        self.apply_velocity_impulse(b_id, impulse, b_inverse_mass);

        let tangent_velocity = relative_velocity - contact.normal * normal_speed;
        let tangent_length_squared = tangent_velocity.length_squared();
        if tangent_length_squared > f32::EPSILON {
            let tangent = tangent_velocity / tangent_length_squared.sqrt();
            let tangent_impulse = -relative_velocity.dot(tangent) / inverse_mass_sum;
            let friction = (a_collider.friction * b_collider.friction).sqrt();
            let friction_impulse = tangent
                * tangent_impulse.clamp(-normal_impulse * friction, normal_impulse * friction);
            self.apply_velocity_impulse(a_id, -friction_impulse, a_inverse_mass);
            self.apply_velocity_impulse(b_id, friction_impulse, b_inverse_mass);
        }
    }

    fn apply_velocity_impulse(&mut self, id: EntityId, impulse: Vec3, inverse_mass: f32) {
        if inverse_mass <= 0.0 {
            return;
        }
        if let Ok(entity) = self.resolve(id)
            && let Ok(mut body) = self.world.get::<&mut RigidBody3d>(entity)
            && body.body_type == RigidBodyType::Dynamic
        {
            let velocity = body.velocity + impulse * inverse_mass;
            if velocity.is_finite() {
                body.velocity = velocity;
            }
        }
    }

    fn translate_world(&mut self, id: EntityId, delta: Vec3) {
        let parent = self
            .parents
            .iter()
            .find_map(|(child, parent)| (*child == id).then_some(*parent));
        let local_delta = if let Some(parent) = parent {
            let Ok(parent_world) = self.world_matrix(parent) else {
                return;
            };
            let inverse = parent_world.inverse();
            if !inverse.is_finite() {
                return;
            }
            inverse.transform_vector3(delta)
        } else {
            delta
        };
        let Ok(entity) = self.resolve(id) else {
            return;
        };
        if let Ok(mut transform) = self.world.get::<&mut Transform>(entity) {
            let translation = transform.translation + local_delta;
            if translation.is_finite() {
                transform.translation = translation;
            }
        }
    }

    pub fn fixed_steps(&self) -> u64 {
        self.fixed_steps
    }

    pub fn fixed_delta_seconds(&self) -> f64 {
        self.config.fixed_delta_seconds
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot {
            entity_count: self.entity_ids.len(),
            fixed_steps: self.fixed_steps,
            fixed_delta_seconds: self.config.fixed_delta_seconds,
        }
    }

    fn allocate_entity_id(&mut self) -> EntityId {
        let id = EntityId(self.next_entity_id);
        self.next_entity_id = self.next_entity_id.wrapping_add(1).max(1);
        id
    }

    fn resolve(&self, id: EntityId) -> Result<Entity, EngineError> {
        self.entity_ids
            .iter()
            .find_map(|(candidate, entity)| (*candidate == id).then_some(*entity))
            .ok_or(EngineError::UnknownEntity(id.0))
    }
}

fn apply_gltf_animation(
    world: &mut World,
    entities: &[(EntityId, Entity)],
    player: &GltfAnimationPlayer,
    time_seconds: f32,
) {
    for (id, base_transform) in player.nodes.iter().zip(&player.base_transforms) {
        if let Some((_, entity)) = entities.iter().find(|(entity_id, _)| entity_id == id)
            && let Ok(mut transform) = world.get::<&mut Transform>(*entity)
        {
            *transform = *base_transform;
        }
    }
    let Some(clip) = player.clips.get(player.state.clip_index) else {
        return;
    };
    for channel in &clip.channels {
        let Some(node_id) = player.nodes.get(channel.node_index) else {
            continue;
        };
        let Some((_, entity)) = entities.iter().find(|(entity_id, _)| entity_id == node_id) else {
            continue;
        };
        let value = sample_gltf_channel(channel, time_seconds);
        let Ok(mut transform) = world.get::<&mut Transform>(*entity) else {
            continue;
        };
        match channel.path {
            GltfAnimationPath::Translation => transform.translation = value.truncate(),
            GltfAnimationPath::Scale => transform.scale = value.truncate(),
            GltfAnimationPath::Rotation => {
                transform.rotation = normalized_quaternion(value);
            }
        }
    }
}

fn sample_gltf_channel(
    channel: &crate::mesh::GltfAnimationChannelData,
    time_seconds: f32,
) -> glam::Vec4 {
    let Some(first) = channel.keyframes.first() else {
        return glam::Vec4::ZERO;
    };
    if time_seconds <= first.time {
        return first.value;
    }
    let Some(last) = channel.keyframes.last() else {
        return first.value;
    };
    if time_seconds >= last.time {
        return last.value;
    }
    let right_index = channel
        .keyframes
        .partition_point(|keyframe| keyframe.time <= time_seconds);
    let left = &channel.keyframes[right_index - 1];
    let right = &channel.keyframes[right_index];
    if channel.interpolation == GltfInterpolation::Step {
        return left.value;
    }
    let duration = right.time - left.time;
    let amount = ((time_seconds - left.time) / duration).clamp(0.0, 1.0);
    if channel.interpolation == GltfInterpolation::CubicSpline {
        let t2 = amount * amount;
        let t3 = t2 * amount;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + amount;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        let value = left.value * h00
            + left.out_tangent * (h10 * duration)
            + right.value * h01
            + right.in_tangent * (h11 * duration);
        if channel.path == GltfAnimationPath::Rotation {
            return glam::Vec4::from_array(normalized_quaternion(value).to_array());
        }
        value
    } else if channel.path == GltfAnimationPath::Rotation {
        let a = normalized_quaternion(left.value);
        let b = normalized_quaternion(right.value);
        let rotation = normalized_quaternion(glam::Vec4::from_array(a.slerp(b, amount).to_array()));
        glam::Vec4::from_array(rotation.to_array())
    } else {
        left.value.lerp(right.value, amount)
    }
}

fn normalized_quaternion(value: glam::Vec4) -> Quat {
    let quaternion = Quat::from_array(value.to_array());
    if quaternion.length_squared() > f32::EPSILON {
        quaternion.normalize()
    } else {
        Quat::IDENTITY
    }
}

fn normalize_input_name(name: &str) -> Result<String, EngineError> {
    let name = name.trim();
    if name.is_empty() || name.len() > 32 || !name.is_ascii() || name.chars().any(char::is_control)
    {
        return Err(EngineError::InvalidInput);
    }
    Ok(name.to_ascii_lowercase())
}

fn validate_scene_limits(scene: &SceneDocument) -> Result<(), EngineError> {
    if scene.format_version != SCENE_FORMAT_VERSION {
        return Err(EngineError::InvalidScene(format!(
            "unsupported format version {}; expected {SCENE_FORMAT_VERSION}",
            scene.format_version
        )));
    }
    if scene.entities.len() > MAX_SCENE_ENTITIES {
        return Err(EngineError::InvalidScene(format!(
            "scene exceeds the {MAX_SCENE_ENTITIES}-entity limit"
        )));
    }
    if scene.mesh_assets.len() > MAX_SCENE_MESH_ASSETS {
        return Err(EngineError::InvalidScene(format!(
            "scene exceeds the {MAX_SCENE_MESH_ASSETS}-mesh-asset limit"
        )));
    }
    if scene.texture_assets.len() > MAX_SCENE_TEXTURE_ASSETS {
        return Err(EngineError::InvalidScene(format!(
            "scene exceeds the {MAX_SCENE_TEXTURE_ASSETS}-texture-asset limit"
        )));
    }
    if scene.animation_players.len() > MAX_SCENE_ENTITIES {
        return Err(EngineError::InvalidScene(format!(
            "scene exceeds the {MAX_SCENE_ENTITIES}-animation-player limit"
        )));
    }
    let animation_keyframes = scene
        .animation_players
        .iter()
        .flat_map(|player| &player.clips)
        .flat_map(|clip| &clip.channels)
        .fold(0usize, |total, channel| {
            total.saturating_add(channel.keyframes.len())
        });
    if animation_keyframes > MAX_SCENE_ANIMATION_KEYFRAMES {
        return Err(EngineError::InvalidScene(format!(
            "scene exceeds the {MAX_SCENE_ANIMATION_KEYFRAMES}-animation-keyframe limit"
        )));
    }
    let texture_byte_count = scene
        .texture_assets
        .iter()
        .try_fold(0usize, |total, asset| {
            let expected = (asset.texture.width as usize)
                .checked_mul(asset.texture.height as usize)?
                .checked_mul(4)?;
            (expected == asset.texture.rgba8.len())
                .then(|| total.checked_add(expected))
                .flatten()
        });
    if texture_byte_count.is_none_or(|bytes| bytes > MAX_SCENE_TEXTURE_BYTES) {
        return Err(EngineError::InvalidScene(format!(
            "scene texture data is invalid or exceeds the {MAX_SCENE_TEXTURE_BYTES}-byte limit"
        )));
    }
    let mesh_element_count = scene.mesh_assets.iter().fold(0usize, |total, asset| {
        total
            .saturating_add(asset.mesh.vertices().len())
            .saturating_add(asset.mesh.indices().len())
    });
    if mesh_element_count > MAX_SCENE_MESH_ELEMENTS {
        return Err(EngineError::InvalidScene(format!(
            "scene exceeds the {MAX_SCENE_MESH_ELEMENTS}-mesh-element limit"
        )));
    }
    Ok(())
}

fn validate_scene_scripts(scene: &SceneDocument) -> Result<(), EngineError> {
    let entity_ids = scene
        .entities
        .iter()
        .map(|entity| entity.id)
        .collect::<HashSet<_>>();
    let mut scripted_entities = HashSet::with_capacity(scene.scripts.len());
    let mut script_bytes = 0usize;
    for script in &scene.scripts {
        if !entity_ids.contains(&script.entity_id) || !scripted_entities.insert(script.entity_id) {
            return Err(EngineError::InvalidScene(format!(
                "script owner {} is missing or has more than one attached script",
                script.entity_id
            )));
        }
        if script.source.len() > MAX_SCENE_SCRIPT_BYTES {
            return Err(EngineError::InvalidScene(format!(
                "script for entity {} exceeds the {MAX_SCENE_SCRIPT_BYTES}-byte limit",
                script.entity_id
            )));
        }
        script_bytes = script_bytes.saturating_add(script.source.len());
    }
    if script_bytes > MAX_SCENE_ALL_SCRIPT_BYTES {
        return Err(EngineError::InvalidScene(format!(
            "scene scripts exceed the {MAX_SCENE_ALL_SCRIPT_BYTES}-byte total limit"
        )));
    }
    Ok(())
}

fn validate_material(material: Material3d) -> Result<(), EngineError> {
    if material
        .base_color
        .iter()
        .any(|channel| !channel.is_finite() || !(0.0..=1.0).contains(channel))
        || !material.roughness.is_finite()
        || !(0.0..=1.0).contains(&material.roughness)
        || !material.metallic.is_finite()
        || !(0.0..=1.0).contains(&material.metallic)
        || !material.alpha_cutoff.is_finite()
        || !(0.0..=1.0).contains(&material.alpha_cutoff)
    {
        return Err(EngineError::InvalidMaterial);
    }
    Ok(())
}

fn validate_rigid_body(body: RigidBody3d) -> Result<(), EngineError> {
    if !body.velocity.is_finite()
        || !body.accumulated_force.is_finite()
        || !body.mass.is_finite()
        || body.mass <= 0.0
        || !body.gravity_scale.is_finite()
        || !body.linear_damping.is_finite()
        || body.linear_damping < 0.0
    {
        return Err(EngineError::InvalidRigidBody);
    }
    Ok(())
}

fn validate_collider(collider: Collider3d) -> Result<(), EngineError> {
    let shape_is_valid = match collider.shape {
        ColliderShape::Sphere { radius } => radius.is_finite() && radius > 0.0,
        ColliderShape::Box { half_extents } => {
            half_extents.is_finite() && half_extents.min_element() > 0.0
        }
    };
    if !shape_is_valid
        || !collider.restitution.is_finite()
        || !(0.0..=1.0).contains(&collider.restitution)
        || !collider.friction.is_finite()
        || !(0.0..=1.0).contains(&collider.friction)
    {
        return Err(EngineError::InvalidCollider);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct WorldCollider {
    center: Vec3,
    min: Vec3,
    max: Vec3,
    shape: WorldColliderShape,
}

#[derive(Clone, Copy)]
enum WorldColliderShape {
    Sphere { radius: f32 },
    Box,
}

#[derive(Clone, Copy)]
struct CollisionContact {
    /// Contact normal pointing from the first collider toward the second.
    normal: Vec3,
    penetration: f32,
}

fn world_collider(model: Mat4, shape: ColliderShape) -> Option<WorldCollider> {
    let center = model.transform_point3(Vec3::ZERO);
    let x_axis = model.x_axis.truncate();
    let y_axis = model.y_axis.truncate();
    let z_axis = model.z_axis.truncate();
    let (shape, extents) = match shape {
        ColliderShape::Sphere { radius } => {
            let scale = x_axis.length().max(y_axis.length()).max(z_axis.length());
            let radius = radius * scale;
            (WorldColliderShape::Sphere { radius }, Vec3::splat(radius))
        }
        ColliderShape::Box { half_extents } => {
            let extents = x_axis.abs() * half_extents.x
                + y_axis.abs() * half_extents.y
                + z_axis.abs() * half_extents.z;
            (WorldColliderShape::Box, extents)
        }
    };
    let min = center - extents;
    let max = center + extents;
    if !center.is_finite() || !extents.is_finite() || !min.is_finite() || !max.is_finite() {
        return None;
    }
    Some(WorldCollider {
        center,
        min,
        max,
        shape,
    })
}

fn collision_contact(a: WorldCollider, b: WorldCollider) -> Option<CollisionContact> {
    match (a.shape, b.shape) {
        (
            WorldColliderShape::Sphere { radius: a_radius },
            WorldColliderShape::Sphere { radius: b_radius },
        ) => {
            let delta = b.center - a.center;
            let distance_squared = delta.length_squared();
            let combined_radius = a_radius + b_radius;
            if distance_squared >= combined_radius * combined_radius {
                return None;
            }
            let distance = distance_squared.sqrt();
            Some(CollisionContact {
                normal: if distance > f32::EPSILON {
                    delta / distance
                } else {
                    Vec3::X
                },
                penetration: combined_radius - distance,
            })
        }
        (WorldColliderShape::Sphere { radius }, WorldColliderShape::Box) => {
            sphere_box_contact(a.center, radius, b)
        }
        (WorldColliderShape::Box, WorldColliderShape::Sphere { radius }) => {
            sphere_box_contact(b.center, radius, a).map(|contact| CollisionContact {
                normal: -contact.normal,
                penetration: contact.penetration,
            })
        }
        (WorldColliderShape::Box, WorldColliderShape::Box) => {
            let overlap = a.max.min(b.max) - a.min.max(b.min);
            if overlap.min_element() <= 0.0 {
                return None;
            }
            let axis = if overlap.x <= overlap.y && overlap.x <= overlap.z {
                0
            } else if overlap.y <= overlap.z {
                1
            } else {
                2
            };
            let mut normal = Vec3::ZERO;
            normal[axis] = if b.center[axis] >= a.center[axis] {
                1.0
            } else {
                -1.0
            };
            Some(CollisionContact {
                normal,
                penetration: overlap[axis],
            })
        }
    }
}

fn sphere_box_contact(
    sphere_center: Vec3,
    radius: f32,
    box_collider: WorldCollider,
) -> Option<CollisionContact> {
    let closest = sphere_center.clamp(box_collider.min, box_collider.max);
    let delta = closest - sphere_center;
    let distance_squared = delta.length_squared();
    if distance_squared >= radius * radius {
        return None;
    }
    if distance_squared > f32::EPSILON {
        let distance = distance_squared.sqrt();
        return Some(CollisionContact {
            normal: delta / distance,
            penetration: radius - distance,
        });
    }

    let face_distances = [
        sphere_center.x - box_collider.min.x,
        box_collider.max.x - sphere_center.x,
        sphere_center.y - box_collider.min.y,
        box_collider.max.y - sphere_center.y,
        sphere_center.z - box_collider.min.z,
        box_collider.max.z - sphere_center.z,
    ];
    let (face, distance) = face_distances
        .into_iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))?;
    let mut normal = Vec3::ZERO;
    normal[face / 2] = if face.is_multiple_of(2) { 1.0 } else { -1.0 };
    Some(CollisionContact {
        normal,
        penetration: radius + distance,
    })
}

fn inverse_mass(body: Option<RigidBody3d>) -> f32 {
    body.filter(|body| body.body_type == RigidBodyType::Dynamic)
        .map_or(0.0, |body| 1.0 / body.mass)
}

#[derive(Clone, Debug)]
pub struct Name(pub String);
