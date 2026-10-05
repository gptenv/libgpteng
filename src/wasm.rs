//! Browser host for the wgpu renderer. It owns a canvas surface, advances the
//! same simulation and Lua script APIs as native hosts, and renders on RAF.

use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};

use glam::Vec3;
use wasm_bindgen::{JsCast, prelude::*};
use web_sys::HtmlCanvasElement;

use crate::{
    AlphaMode3d, Camera3d, CameraProjection3d, Collider3d, ColliderShape, Engine, EngineConfig,
    EntityId, GltfAssetData, Lighting3d, Material3d, MeshKind, PointLight3d, RenderCamera,
    RigidBody3d, RigidBodyType, SceneRenderer, ScriptHost, ScriptRuntime, TextureAssetId,
    Transform,
};

struct BrowserState {
    engine: Engine,
    scripts: ScriptHost,
    lua: ScriptRuntime,
    canvas: HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: SceneRenderer,
    config: wgpu::SurfaceConfiguration,
    last_frame_ms: Option<f64>,
}

impl BrowserState {
    fn frame(&mut self, timestamp_ms: f64) {
        let delta_seconds = self
            .last_frame_ms
            .replace(timestamp_ms)
            .map(|last| ((timestamp_ms - last) / 1000.0).clamp(0.0, 0.1))
            .unwrap_or(0.0);
        let fixed_delta = self.engine.fixed_delta_seconds();
        let steps = self.engine.advance(delta_seconds, 4);
        for _ in 0..steps {
            let failures = self.scripts.update(&mut self.engine, fixed_delta);
            for failure in failures {
                web_sys::console::error_1(&JsValue::from_str(&failure.message));
            }
        }

        if self.resize_canvas() {
            self.surface.configure(&self.device, &self.config);
            self.renderer
                .resize(&self.device, self.config.width, self.config.height);
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render(
            &self.engine,
            RenderCamera::from(self.engine.camera()),
            &self.device,
            &self.queue,
            &view,
        );
        self.queue.present(frame);
    }

    fn resize_canvas(&mut self) -> bool {
        let Some(window) = web_sys::window() else {
            return false;
        };
        let scale = window.device_pixel_ratio().max(1.0);
        let width = ((self.canvas.client_width().max(1) as f64 * scale).round() as u32).max(1);
        let height = ((self.canvas.client_height().max(1) as f64 * scale).round() as u32).max(1);
        if self.canvas.width() == width && self.canvas.height() == height {
            return false;
        }
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.config.width = width;
        self.config.height = height;
        true
    }
}

/// A live browser engine instance. Dropping it stops its animation loop.
#[wasm_bindgen]
pub struct BrowserEngine {
    state: Rc<RefCell<BrowserState>>,
    callback: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>,
    frame_id: Rc<Cell<Option<i32>>>,
}

#[wasm_bindgen]
impl BrowserEngine {
    /// Initialize WebGPU on a canvas, create a demo scene, and start rendering.
    pub async fn start(canvas: HtmlCanvasElement) -> Result<BrowserEngine, JsValue> {
        let instance = wgpu::util::new_instance_with_webgpu_detection(
            wgpu::InstanceDescriptor::new_without_display_handle(),
        )
        .await;
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(js_error)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(js_error)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(js_error)?;

        let canvas_width = canvas.client_width().max(1) as u32;
        let canvas_height = canvas.client_height().max(1) as u32;
        canvas.set_width(canvas_width);
        canvas.set_height(canvas_height);
        let config = surface
            .get_default_config(&adapter, canvas_width, canvas_height)
            .ok_or_else(|| JsValue::from_str("the browser GPU cannot present to this canvas"))?;
        surface.configure(&device, &config);
        let renderer =
            SceneRenderer::new(&device, &queue, config.format, config.width, config.height);

        let mut engine = Engine::new(EngineConfig::default()).map_err(js_error)?;
        let face_texture = engine
            .add_texture_asset(
                2,
                2,
                vec![
                    235, 133, 72, 255, 224, 117, 63, 255, 242, 143, 79, 255, 231, 126, 68, 255,
                ],
            )
            .map_err(js_error)?;
        let demo = engine
            .spawn_donald_trump_caricature(Vec3::ZERO)
            .map_err(js_error)?;
        engine
            .set_base_color_texture(demo.face, Some(face_texture))
            .map_err(js_error)?;
        let translucent_orb = engine
            .spawn_mesh(
                "Translucent demo orb",
                Transform {
                    translation: Vec3::new(-1.05, 0.48, 0.18),
                    scale: Vec3::splat(0.42),
                    ..Transform::default()
                },
                MeshKind::Sphere,
                Material3d {
                    base_color: [0.12, 0.72, 0.98, 0.38],
                    roughness: 0.12,
                    metallic: 0.15,
                    ..Material3d::default()
                },
            )
            .map_err(js_error)?;
        engine
            .set_parent(translucent_orb, Some(demo.root))
            .map_err(js_error)?;
        engine
            .set_point_light(
                translucent_orb,
                Some(PointLight3d {
                    color: [0.3, 0.68, 1.0],
                    intensity: 1.4,
                    radius: 3.0,
                }),
            )
            .map_err(js_error)?;
        let mut scripts = ScriptHost::default();
        scripts
            .attach(
                &mut engine,
                demo.root,
                "local id = engine.self()\nlocal elapsed = 0\nfunction update(dt)\n  elapsed = elapsed + dt\n  local yaw = math.sin(elapsed * 0.7) * 0.16\n  local pitch = math.sin(elapsed * 1.1) * 0.035\n  local sx = math.sin(pitch * 0.5)\n  local cy = math.cos(yaw * 0.5)\n  local cx = math.cos(pitch * 0.5)\n  local sy = math.sin(yaw * 0.5)\n  engine.set_rotation(id, sx * cy, cx * sy, -sx * sy, cx * cy)\nend",
            )
            .map_err(js_error)?;

        let state = Rc::new(RefCell::new(BrowserState {
            engine,
            scripts,
            lua: ScriptRuntime::default_sandbox(),
            canvas,
            surface,
            device,
            queue,
            renderer,
            config,
            last_frame_ms: None,
        }));
        let callback = Rc::new(RefCell::new(None));
        let weak_callback: Weak<RefCell<Option<Closure<dyn FnMut(f64)>>>> =
            Rc::downgrade(&callback);
        let frame_id = Rc::new(Cell::new(None));
        let callback_frame_id = Rc::clone(&frame_id);
        let frame_state = Rc::clone(&state);
        let frame_callback = Closure::wrap(Box::new(move |timestamp_ms: f64| {
            frame_state.borrow_mut().frame(timestamp_ms);
            if let (Some(window), Some(holder)) = (web_sys::window(), weak_callback.upgrade()) {
                if let Some(callback) = holder.borrow().as_ref() {
                    if let Ok(id) =
                        window.request_animation_frame(callback.as_ref().unchecked_ref())
                    {
                        callback_frame_id.set(Some(id));
                    }
                }
            }
        }) as Box<dyn FnMut(f64)>);
        *callback.borrow_mut() = Some(frame_callback);
        if let (Some(window), Some(callback)) = (web_sys::window(), callback.borrow().as_ref()) {
            let id = window.request_animation_frame(callback.as_ref().unchecked_ref())?;
            frame_id.set(Some(id));
        } else {
            return Err(JsValue::from_str("browser window is unavailable"));
        }

        Ok(BrowserEngine {
            state,
            callback,
            frame_id,
        })
    }

    /// Create a built-in primitive in the running browser scene.
    pub fn spawn_primitive(
        &self,
        name: String,
        primitive: String,
        x: f32,
        y: f32,
        z: f32,
    ) -> Result<u64, JsValue> {
        let kind = match primitive.as_str() {
            "cube" => MeshKind::Cube,
            "sphere" => MeshKind::Sphere,
            "plane" => MeshKind::Plane,
            _ => {
                return Err(JsValue::from_str(
                    "primitive must be cube, sphere, or plane",
                ));
            }
        };
        let transform = Transform {
            translation: Vec3::new(x, y, z),
            ..Transform::default()
        };
        let mut state = self.state.borrow_mut();
        state
            .engine
            .spawn_mesh(name, transform, kind, Material3d::default())
            .map(EntityId::get)
            .map_err(js_error)
    }

    /// Import an embedded glTF/GLB mesh and spawn it at a world-space position.
    pub fn load_gltf_mesh(
        &self,
        name: String,
        source: Vec<u8>,
        x: f32,
        y: f32,
        z: f32,
    ) -> Result<u64, JsValue> {
        let asset = GltfAssetData::from_gltf(&source).map_err(js_error)?;
        let mut state = self.state.borrow_mut();
        state
            .engine
            .spawn_gltf_asset(name, Vec3::new(x, y, z), asset)
            .map(|instance| instance.root.get())
            .map_err(js_error)
    }

    /// Return the imported scene's clip names and current playback state as JSON.
    pub fn list_gltf_animations(&self, root_id: u64) -> Result<String, JsValue> {
        let state = self.state.borrow();
        let root = EntityId::from_control(root_id);
        let animations = state.engine.gltf_animation_names(root).map_err(js_error)?;
        let playback = state.engine.gltf_animation_state(root).map_err(js_error)?;
        serde_json::to_string(&serde_json::json!({
            "animations": animations,
            "state": playback
        }))
        .map_err(js_error)
    }

    /// Start an imported glTF animation clip by its zero-based index.
    pub fn play_gltf_animation(
        &self,
        root_id: u64,
        clip_index: usize,
        looping: bool,
    ) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .play_gltf_animation(EntityId::from_control(root_id), clip_index, looping)
            .map(|_| ())
            .map_err(js_error)
    }

    /// Stop an imported glTF animation player at its current pose.
    pub fn stop_gltf_animation(&self, root_id: u64) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .stop_gltf_animation(EntityId::from_control(root_id))
            .map(|_| ())
            .map_err(js_error)
    }

    /// Set glTF animation speed from -100 to 100; negative values play backward.
    pub fn set_gltf_animation_speed(&self, root_id: u64, speed: f32) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .set_gltf_animation_speed(EntityId::from_control(root_id), speed)
            .map(|_| ())
            .map_err(js_error)
    }

    /// Create a bounded, row-major sRGBA8 texture asset from browser-provided bytes.
    pub fn create_texture_asset(
        &self,
        width: u32,
        height: u32,
        rgba8: Vec<u8>,
    ) -> Result<u64, JsValue> {
        self.state
            .borrow_mut()
            .engine
            .add_texture_asset(width, height, rgba8)
            .map(TextureAssetId::get)
            .map_err(js_error)
    }

    /// Decode supported image bytes and create a bounded texture asset.
    pub fn load_texture_asset(&self, encoded: Vec<u8>) -> Result<u64, JsValue> {
        self.state
            .borrow_mut()
            .engine
            .add_texture_asset_from_encoded(&encoded)
            .map(TextureAssetId::get)
            .map_err(js_error)
    }

    /// Assign a texture asset to a mesh entity, or pass `None` to clear it.
    pub fn set_base_color_texture(
        &self,
        entity_id: u64,
        texture_asset_id: Option<u64>,
    ) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .set_base_color_texture(
                EntityId::from_control(entity_id),
                texture_asset_id.map(TextureAssetId),
            )
            .map_err(js_error)
    }

    /// Return an entity's texture asset ID, or `None` when it uses its flat color.
    pub fn base_color_texture(&self, entity_id: u64) -> Result<Option<u64>, JsValue> {
        self.state
            .borrow()
            .engine
            .base_color_texture(EntityId::from_control(entity_id))
            .map(|texture| texture.map(TextureAssetId::get))
            .map_err(js_error)
    }

    /// Set the local render visibility of an entity.
    pub fn set_visibility(&self, entity_id: u64, visible: bool) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .set_visibility(EntityId::from_control(entity_id), visible)
            .map_err(js_error)
    }

    /// Read the entity's local render visibility flag.
    pub fn visibility(&self, entity_id: u64) -> Result<bool, JsValue> {
        self.state
            .borrow()
            .engine
            .visibility(EntityId::from_control(entity_id))
            .map_err(js_error)
    }

    /// Read whether this entity and all ancestors are visible.
    pub fn is_visible(&self, entity_id: u64) -> Result<bool, JsValue> {
        self.state
            .borrow()
            .engine
            .is_visible(EntityId::from_control(entity_id))
            .map_err(js_error)
    }

    /// Read lighting as `[direction_x, direction_y, direction_z, color_r,
    /// color_g, color_b, intensity, ambient_r, ambient_g, ambient_b]`.
    pub fn lighting(&self) -> Vec<f32> {
        let lighting = self.state.borrow().engine.lighting();
        vec![
            lighting.direction.x,
            lighting.direction.y,
            lighting.direction.z,
            lighting.color[0],
            lighting.color[1],
            lighting.color[2],
            lighting.intensity,
            lighting.ambient[0],
            lighting.ambient[1],
            lighting.ambient[2],
        ]
    }

    /// Update scene-wide directional and ambient lighting.
    pub fn set_lighting(
        &self,
        direction: Vec<f32>,
        color: Vec<f32>,
        intensity: f32,
        ambient: Vec<f32>,
    ) -> Result<(), JsValue> {
        if direction.len() != 3 || color.len() != 3 || ambient.len() != 3 {
            return Err(JsValue::from_str(
                "lighting direction, color, and ambient values must each have three channels",
            ));
        }
        self.state
            .borrow_mut()
            .engine
            .set_lighting(Lighting3d {
                direction: Vec3::from_array(direction.try_into().expect("length checked")),
                color: color.try_into().expect("length checked"),
                intensity,
                ambient: ambient.try_into().expect("length checked"),
            })
            .map_err(js_error)
    }

    /// Attach or replace a point light on an entity.
    pub fn set_point_light(
        &self,
        entity_id: u64,
        color: Vec<f32>,
        intensity: f32,
        radius: f32,
    ) -> Result<(), JsValue> {
        if color.len() != 3 {
            return Err(JsValue::from_str(
                "point light color must have three channels",
            ));
        }
        self.state
            .borrow_mut()
            .engine
            .set_point_light(
                EntityId::from_control(entity_id),
                Some(PointLight3d {
                    color: color.try_into().expect("length checked"),
                    intensity,
                    radius,
                }),
            )
            .map_err(js_error)
    }

    /// Return `[red, green, blue, intensity, radius]`, or an empty array if absent.
    pub fn point_light(&self, entity_id: u64) -> Result<Vec<f32>, JsValue> {
        self.state
            .borrow()
            .engine
            .point_light(EntityId::from_control(entity_id))
            .map(|light| {
                light
                    .map(|light| {
                        vec![
                            light.color[0],
                            light.color[1],
                            light.color[2],
                            light.intensity,
                            light.radius,
                        ]
                    })
                    .unwrap_or_default()
            })
            .map_err(js_error)
    }

    /// Remove an entity's point light and report whether one existed.
    pub fn remove_point_light(&self, entity_id: u64) -> Result<bool, JsValue> {
        let mut state = self.state.borrow_mut();
        let engine = &mut state.engine;
        let id = EntityId::from_control(entity_id);
        let existed = engine.point_light(id).map_err(js_error)?.is_some();
        engine.set_point_light(id, None).map_err(js_error)?;
        Ok(existed)
    }

    /// Run a chunk in the persistent sandbox with access to the current scene.
    pub fn run_lua(&self, source: String) -> Result<(), JsValue> {
        let mut state = self.state.borrow_mut();
        let BrowserState { engine, lua, .. } = &mut *state;
        lua.run_with_engine(engine, "wasm://run_lua", &source)
            .map_err(js_error)
    }

    /// Attach a persistent sandboxed Lua script to an entity. The script's
    /// optional `update(dt)` callback runs on fixed simulation steps.
    pub fn attach_script(&self, entity_id: u64, source: String) -> Result<(), JsValue> {
        let mut state = self.state.borrow_mut();
        let BrowserState {
            engine, scripts, ..
        } = &mut *state;
        scripts
            .attach(engine, EntityId::from_control(entity_id), &source)
            .map_err(js_error)
    }

    /// Detach an entity's persistent Lua script, if one is attached.
    pub fn detach_script(&self, entity_id: u64) -> bool {
        self.state
            .borrow_mut()
            .scripts
            .detach(EntityId::from_control(entity_id))
    }

    /// Feed a normalized keyboard state from browser event handlers.
    pub fn set_key_down(&self, key: String, down: bool) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .set_key_down(&key, down)
            .map_err(js_error)
    }

    /// Feed a normalized mouse-button state from browser event handlers.
    pub fn set_mouse_button_down(&self, button: String, down: bool) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .set_mouse_button_down(&button, down)
            .map_err(js_error)
    }

    /// Feed relative pointer movement from browser event handlers.
    pub fn add_mouse_motion(&self, dx: f64, dy: f64) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .add_mouse_motion(dx, dy)
            .map_err(js_error)
    }

    /// Clear accumulated pointer motion after consuming a browser input frame.
    pub fn clear_mouse_motion(&self) {
        self.state.borrow_mut().engine.clear_mouse_motion();
    }

    /// Read the transient host input snapshot as JSON.
    pub fn input_state_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self.state.borrow().engine.input()).map_err(js_error)
    }

    /// Set the world position of a live entity.
    pub fn set_position(&self, id: u64, x: f32, y: f32, z: f32) -> Result<(), JsValue> {
        let mut state = self.state.borrow_mut();
        let entity_id = EntityId::from_control(id);
        let mut transform = state.engine.transform(entity_id).map_err(js_error)?;
        transform.translation = Vec3::new(x, y, z);
        state
            .engine
            .set_transform(entity_id, transform)
            .map_err(js_error)
    }

    /// Set base color, roughness, and metallic response for a 3D entity.
    pub fn set_material(
        &self,
        id: u64,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        roughness: f32,
        metallic: f32,
    ) -> Result<(), JsValue> {
        let mut state = self.state.borrow_mut();
        let entity = EntityId::from_control(id);
        let current = state.engine.material(entity).map_err(js_error)?;
        state
            .engine
            .set_material(
                entity,
                Material3d {
                    base_color: [r, g, b, a],
                    roughness,
                    metallic,
                    ..current
                },
            )
            .map_err(js_error)
    }

    /// Set opaque, alpha-mask, alpha-blend, or automatic alpha handling.
    pub fn set_alpha_mode(&self, id: u64, mode: String, cutoff: f32) -> Result<(), JsValue> {
        let alpha_mode = match mode.as_str() {
            "auto" => AlphaMode3d::Auto,
            "opaque" => AlphaMode3d::Opaque,
            "mask" => AlphaMode3d::Mask,
            "blend" => AlphaMode3d::Blend,
            _ => {
                return Err(JsValue::from_str(
                    "alpha mode must be auto, opaque, mask, or blend",
                ));
            }
        };
        let mut state = self.state.borrow_mut();
        let entity = EntityId::from_control(id);
        let current = state.engine.material(entity).map_err(js_error)?;
        state
            .engine
            .set_material(
                entity,
                Material3d {
                    alpha_mode,
                    alpha_cutoff: cutoff,
                    ..current
                },
            )
            .map_err(js_error)
    }

    /// Return base RGBA, roughness, metallic, alpha-mode code, and alpha cutoff.
    pub fn material_properties(&self, id: u64) -> Result<Vec<f32>, JsValue> {
        self.state
            .borrow()
            .engine
            .material(EntityId::from_control(id))
            .map(|material| {
                let alpha_mode = match material.alpha_mode {
                    AlphaMode3d::Opaque => 0.0,
                    AlphaMode3d::Mask => 1.0,
                    AlphaMode3d::Blend => 2.0,
                    AlphaMode3d::Auto => 3.0,
                };
                vec![
                    material.base_color[0],
                    material.base_color[1],
                    material.base_color[2],
                    material.base_color[3],
                    material.roughness,
                    material.metallic,
                    alpha_mode,
                    material.alpha_cutoff,
                ]
            })
            .map_err(js_error)
    }

    /// Attach a rigid body using the engine's fixed-step physics simulation.
    pub fn add_rigid_body(
        &self,
        id: u64,
        body_type: String,
        mass: f32,
        gravity_scale: f32,
        linear_damping: f32,
    ) -> Result<(), JsValue> {
        let body_type = match body_type.as_str() {
            "dynamic" => RigidBodyType::Dynamic,
            "kinematic" => RigidBodyType::Kinematic,
            "static" => RigidBodyType::Static,
            _ => {
                return Err(JsValue::from_str(
                    "body_type must be dynamic, kinematic, or static",
                ));
            }
        };
        self.state
            .borrow_mut()
            .engine
            .add_rigid_body(
                EntityId::from_control(id),
                RigidBody3d {
                    body_type,
                    mass,
                    gravity_scale,
                    linear_damping,
                    ..RigidBody3d::default()
                },
            )
            .map_err(js_error)
    }

    /// Return body type code (dynamic=0, kinematic=1, static=2), velocity,
    /// mass, gravity scale, damping, and accumulated force.
    pub fn rigid_body_properties(&self, id: u64) -> Result<Vec<f32>, JsValue> {
        let body = self
            .state
            .borrow()
            .engine
            .rigid_body(EntityId::from_control(id))
            .map_err(js_error)?
            .ok_or_else(|| JsValue::from_str("entity has no rigid body"))?;
        let body_type = match body.body_type {
            RigidBodyType::Dynamic => 0.0,
            RigidBodyType::Kinematic => 1.0,
            RigidBodyType::Static => 2.0,
        };
        Ok(vec![
            body_type,
            body.velocity.x,
            body.velocity.y,
            body.velocity.z,
            body.mass,
            body.gravity_scale,
            body.linear_damping,
            body.accumulated_force.x,
            body.accumulated_force.y,
            body.accumulated_force.z,
        ])
    }

    pub fn remove_rigid_body(&self, id: u64) -> Result<bool, JsValue> {
        self.state
            .borrow_mut()
            .engine
            .remove_rigid_body(EntityId::from_control(id))
            .map_err(js_error)
    }

    /// Set a sphere collider with optional bounce and friction values.
    pub fn set_sphere_collider(
        &self,
        id: u64,
        radius: f32,
        restitution: f32,
        friction: f32,
    ) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .set_collider(
                EntityId::from_control(id),
                Collider3d {
                    shape: ColliderShape::Sphere { radius },
                    restitution,
                    friction,
                },
            )
            .map_err(js_error)
    }

    /// Set an axis-aligned box collider from local-space half extents.
    pub fn set_box_collider(
        &self,
        id: u64,
        half_x: f32,
        half_y: f32,
        half_z: f32,
        restitution: f32,
        friction: f32,
    ) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .set_collider(
                EntityId::from_control(id),
                Collider3d {
                    shape: ColliderShape::Box {
                        half_extents: glam::Vec3::new(half_x, half_y, half_z),
                    },
                    restitution,
                    friction,
                },
            )
            .map_err(js_error)
    }

    pub fn collider_properties(&self, id: u64) -> Result<Vec<f32>, JsValue> {
        let collider = self
            .state
            .borrow()
            .engine
            .collider(EntityId::from_control(id))
            .map_err(js_error)?
            .ok_or_else(|| JsValue::from_str("entity has no collider"))?;
        let mut result = match collider.shape {
            ColliderShape::Sphere { radius } => vec![0.0, radius, 0.0, 0.0],
            ColliderShape::Box { half_extents } => {
                vec![1.0, half_extents.x, half_extents.y, half_extents.z]
            }
        };
        result.extend([collider.restitution, collider.friction]);
        Ok(result)
    }

    pub fn remove_collider(&self, id: u64) -> Result<bool, JsValue> {
        self.state
            .borrow_mut()
            .engine
            .remove_collider(EntityId::from_control(id))
            .map_err(js_error)
    }

    /// Set an entity's linear velocity in world units per second.
    pub fn set_velocity(&self, id: u64, x: f32, y: f32, z: f32) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .set_velocity(EntityId::from_control(id), glam::Vec3::new(x, y, z))
            .map_err(js_error)
    }

    /// Apply a force through the next fixed physics step.
    pub fn apply_force(&self, id: u64, x: f32, y: f32, z: f32) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .apply_force(EntityId::from_control(id), glam::Vec3::new(x, y, z))
            .map_err(js_error)
    }

    /// Get or set world gravity as a three-element vector.
    pub fn gravity(&self) -> Vec<f32> {
        self.state.borrow().engine.gravity().to_array().to_vec()
    }

    pub fn set_gravity(&self, x: f32, y: f32, z: f32) -> Result<(), JsValue> {
        self.state
            .borrow_mut()
            .engine
            .set_gravity(glam::Vec3::new(x, y, z))
            .map_err(js_error)
    }

    /// Set an entity's parent, or pass `None` to make it a root entity.
    pub fn set_parent(&self, id: u64, parent_id: Option<u64>) -> Result<(), JsValue> {
        let mut state = self.state.borrow_mut();
        state
            .engine
            .set_parent(
                EntityId::from_control(id),
                parent_id.map(EntityId::from_control),
            )
            .map_err(js_error)
    }

    /// Return the entity's composed world transform as a column-major matrix.
    pub fn world_matrix(&self, id: u64) -> Result<Vec<f32>, JsValue> {
        self.state
            .borrow()
            .engine
            .world_matrix(EntityId::from_control(id))
            .map(|matrix| matrix.to_cols_array().to_vec())
            .map_err(js_error)
    }

    /// Set the active camera's world-space position.
    pub fn set_camera_position(&self, x: f32, y: f32, z: f32) -> Result<(), JsValue> {
        let mut state = self.state.borrow_mut();
        let camera = Camera3d {
            position: Vec3::new(x, y, z),
            ..state.engine.camera()
        };
        state.engine.set_camera(camera).map_err(js_error)
    }

    /// Set the world-space point the active camera looks at.
    pub fn set_camera_target(&self, x: f32, y: f32, z: f32) -> Result<(), JsValue> {
        let mut state = self.state.borrow_mut();
        let camera = Camera3d {
            target: Vec3::new(x, y, z),
            ..state.engine.camera()
        };
        state.engine.set_camera(camera).map_err(js_error)
    }

    /// Set camera projection to `perspective` or `orthographic`.
    pub fn set_camera_projection(
        &self,
        projection: String,
        orthographic_vertical_size: f32,
    ) -> Result<(), JsValue> {
        let mut state = self.state.borrow_mut();
        let projection = match projection.as_str() {
            "perspective" => CameraProjection3d::Perspective,
            "orthographic" => CameraProjection3d::Orthographic {
                vertical_size: orthographic_vertical_size,
            },
            _ => {
                return Err(JsValue::from_str(
                    "projection must be perspective or orthographic",
                ));
            }
        };
        let camera = Camera3d {
            projection,
            ..state.engine.camera()
        };
        state.engine.set_camera(camera).map_err(js_error)
    }

    /// Export engine-owned scene data as a versioned JSON string.
    pub fn export_scene_json(&self) -> Result<String, JsValue> {
        let state = self.state.borrow();
        state
            .engine
            .to_scene_json_with_scripts(state.scripts.scene_scripts())
            .map_err(js_error)
    }

    /// Replace the current scene from validated JSON and recreate fresh Lua
    /// runtimes from the saved entity script sources.
    pub fn load_scene_json(&self, scene_json: String) -> Result<(), JsValue> {
        let mut state = self.state.borrow_mut();
        let document = Engine::parse_scene_json(&scene_json).map_err(js_error)?;
        let definitions = document.scripts.clone();
        let mut replacement = Engine::from_scene(document).map_err(js_error)?;
        let mut scripts = ScriptHost::default();
        scripts
            .restore_scene_scripts(&mut replacement, &definitions)
            .map_err(js_error)?;
        state.engine.replace_scene(replacement);
        state.scripts = scripts;
        state.lua = ScriptRuntime::default_sandbox();
        Ok(())
    }

    /// Stop the animation loop. The instance can still be queried and mutated.
    pub fn stop(&self) {
        if let (Some(window), Some(frame_id)) = (web_sys::window(), self.frame_id.take()) {
            let _ = window.cancel_animation_frame(frame_id);
        }
        self.callback.borrow_mut().take();
    }
}

impl Drop for BrowserEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
