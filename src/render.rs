//! wgpu device setup and a cached, instanced forward renderer. Window and canvas
//! ownership stays in the host; this module renders engine mesh components.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::engine::{
    AlphaMode3d, Camera3d, CameraProjection3d, Engine, MeshKind, MeshSource, TextureAssetId,
    TextureData,
};

#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub prefer_low_power: bool,
    pub force_fallback_adapter: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            prefer_low_power: false,
            force_fallback_adapter: false,
        }
    }
}

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("no compatible graphics adapter was found: {0}")]
    Adapter(#[from] wgpu::RequestAdapterError),
    #[error("failed to create a graphics device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
}

/// Per-frame visibility and submission counts from the scene renderer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderStats {
    pub candidate_meshes: usize,
    pub visible_meshes: usize,
    pub frustum_culled_meshes: usize,
    pub transparent_meshes: usize,
    pub instance_count: usize,
    pub draw_calls: usize,
}

/// Shared wgpu device state. A surface is created separately from a host-owned
/// window or browser canvas and configured by the platform layer.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn request(config: RenderConfig) -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if config.prefer_low_power {
                    wgpu::PowerPreference::LowPower
                } else {
                    wgpu::PowerPreference::HighPerformance
                },
                force_fallback_adapter: config.force_fallback_adapter,
                ..Default::default()
            })
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

/// Camera settings used by the forward-rendering path.
#[derive(Clone, Copy, Debug)]
pub struct RenderCamera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub projection: CameraProjection3d,
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
    pub clear_color: wgpu::Color,
}

impl Default for RenderCamera {
    fn default() -> Self {
        Self {
            position: Vec3::new(2.8, 2.2, 4.2),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: CameraProjection3d::Perspective,
            vertical_fov_radians: 55.0_f32.to_radians(),
            near: 0.1,
            far: 100.0,
            clear_color: wgpu::Color {
                r: 0.035,
                g: 0.055,
                b: 0.085,
                a: 1.0,
            },
        }
    }
}

impl RenderCamera {
    fn view_projection(self, aspect: f32) -> Mat4 {
        let direction = (self.target - self.position).normalize_or_zero();
        let view = Mat4::look_to_rh(self.position, direction, self.up);
        let aspect = aspect.max(0.001);
        let projection = match self.projection {
            CameraProjection3d::Perspective => {
                Mat4::perspective_rh(self.vertical_fov_radians, aspect, self.near, self.far)
            }
            CameraProjection3d::Orthographic { vertical_size } => {
                let half_height = vertical_size * 0.5;
                let half_width = half_height * aspect;
                Mat4::orthographic_rh(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    self.near,
                    self.far,
                )
            }
        };
        projection * view
    }

    fn frustum(self, aspect: f32) -> Option<Frustum> {
        let direction = (self.target - self.position).normalize_or_zero();
        if !direction.is_finite()
            || direction.length_squared() < 0.99
            || !self.up.is_finite()
            || match self.projection {
                CameraProjection3d::Perspective => {
                    !self.vertical_fov_radians.is_finite()
                        || !(0.01..std::f32::consts::PI - 0.01).contains(&self.vertical_fov_radians)
                }
                CameraProjection3d::Orthographic { vertical_size } => {
                    !vertical_size.is_finite() || vertical_size <= 0.0
                }
            }
            || !self.near.is_finite()
            || !self.far.is_finite()
            || self.near <= 0.0
            || self.far <= self.near
            || !aspect.is_finite()
            || aspect <= 0.0
        {
            return None;
        }
        let right = direction.cross(self.up).normalize_or_zero();
        if right.length_squared() < 0.99 {
            return None;
        }
        let up = right.cross(direction).normalize_or_zero();
        let at_origin = |normal: Vec3| Plane {
            normal,
            distance: -normal.dot(self.position),
        };
        let side_planes = match self.projection {
            CameraProjection3d::Perspective => {
                let tan_vertical = (self.vertical_fov_radians * 0.5).tan();
                let tan_horizontal = tan_vertical * aspect;
                [
                    at_origin((right + direction * tan_horizontal).normalize()),
                    at_origin((-right + direction * tan_horizontal).normalize()),
                    at_origin((up + direction * tan_vertical).normalize()),
                    at_origin((-up + direction * tan_vertical).normalize()),
                ]
            }
            CameraProjection3d::Orthographic { vertical_size } => {
                let half_height = vertical_size * 0.5;
                let half_width = half_height * aspect;
                [
                    Plane {
                        normal: right,
                        distance: half_width - right.dot(self.position),
                    },
                    Plane {
                        normal: -right,
                        distance: half_width + right.dot(self.position),
                    },
                    Plane {
                        normal: up,
                        distance: half_height - up.dot(self.position),
                    },
                    Plane {
                        normal: -up,
                        distance: half_height + up.dot(self.position),
                    },
                ]
            }
        };
        Some(Frustum {
            planes: [
                Plane {
                    normal: direction,
                    distance: -direction.dot(self.position) - self.near,
                },
                Plane {
                    normal: -direction,
                    distance: direction.dot(self.position) + self.far,
                },
                side_planes[0],
                side_planes[1],
                side_planes[2],
                side_planes[3],
            ],
        })
    }
}

#[derive(Clone, Copy)]
struct Plane {
    normal: Vec3,
    distance: f32,
}

#[derive(Clone, Copy)]
struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    fn intersects(self, bounds: WorldBounds) -> bool {
        self.planes.iter().all(|plane| {
            let projected_radius = plane.normal.abs().dot(bounds.extents);
            plane.normal.dot(bounds.center) + plane.distance + projected_radius >= 0.0
        })
    }
}

#[derive(Clone, Copy)]
struct MeshBounds {
    center: Vec3,
    extents: Vec3,
}

impl MeshBounds {
    fn from_vertices(vertices: &[MeshVertex]) -> Self {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for vertex in vertices {
            let position = Vec3::from_array(vertex.position);
            min = min.min(position);
            max = max.max(position);
        }
        if vertices.is_empty() {
            return Self {
                center: Vec3::ZERO,
                extents: Vec3::ZERO,
            };
        }
        Self {
            center: (min + max) * 0.5,
            extents: (max - min) * 0.5,
        }
    }

    fn transformed(self, model: Mat4) -> WorldBounds {
        let linear = Mat3::from_mat4(model);
        let extents = linear.x_axis.abs() * self.extents.x
            + linear.y_axis.abs() * self.extents.y
            + linear.z_axis.abs() * self.extents.z;
        WorldBounds {
            center: model.transform_point3(self.center),
            extents,
        }
    }
}

#[derive(Clone, Copy)]
struct WorldBounds {
    center: Vec3,
    extents: Vec3,
}

impl From<Camera3d> for RenderCamera {
    fn from(camera: Camera3d) -> Self {
        Self {
            position: camera.position,
            target: camera.target,
            up: camera.up,
            projection: camera.projection,
            vertical_fov_radians: camera.vertical_fov_radians,
            near: camera.near,
            far: camera.far,
            clear_color: wgpu::Color {
                r: camera.clear_color[0] as f64,
                g: camera.clear_color[1] as f64,
                b: camera.clear_color[2] as f64,
                a: camera.clear_color[3] as f64,
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_projection: [[f32; 4]; 4],
    position: [f32; 4],
    light_direction: [f32; 4],
    light_color_intensity: [f32; 4],
    ambient: [f32; 4],
    point_lights: [PointLightUniform; MAX_POINT_LIGHTS],
    point_light_count: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointLightUniform {
    position_radius: [f32; 4],
    color_intensity: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
    tex_coords: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 3],
    color: [f32; 4],
    surface: [f32; 4],
}

const MESH_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 9] = wgpu::vertex_attr_array![
    3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4,
    7 => Float32x4, 8 => Float32x4, 9 => Float32x4, 10 => Float32x4,
    11 => Float32x4
];
const MAX_POINT_LIGHTS: usize = 8;

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    instances: Vec<InstanceRaw>,
    draws: Vec<(Option<TextureAssetId>, std::ops::Range<u32>)>,
    bounds: MeshBounds,
}

struct GpuTexture {
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    has_alpha: bool,
}

struct TransparentInstance {
    depth: f32,
    mesh: MeshSource,
    texture: Option<TextureAssetId>,
    data: InstanceRaw,
}

/// Forward renderer with cached built-in GPU meshes and per-mesh instancing.
pub struct SceneRenderer {
    pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    white_texture: GpuTexture,
    textures: std::collections::HashMap<TextureAssetId, GpuTexture>,
    meshes: std::collections::HashMap<MeshSource, GpuMesh>,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    width: u32,
    height: u32,
    mesh_asset_generation: Option<u64>,
    texture_asset_generation: Option<u64>,
    frustum_culling_enabled: bool,
    stats: RenderStats,
}

impl SceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpteng camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpteng camera layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpteng camera bind group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpteng material texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gpteng repeating linear material sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let white_texture = create_gpu_texture(
            device,
            queue,
            &texture_layout,
            &sampler,
            &TextureData {
                width: 1,
                height: 1,
                rgba8: vec![255; 4],
            },
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpteng scene pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&texture_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpteng basic 3d shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../assets/basic3d.wgsl").into()),
        });
        let pipeline = create_scene_pipeline(
            device,
            &pipeline_layout,
            &shader,
            format,
            "gpteng opaque scene pipeline",
            wgpu::BlendState::REPLACE,
            true,
        );
        let transparent_pipeline = create_scene_pipeline(
            device,
            &pipeline_layout,
            &shader,
            format,
            "gpteng transparent scene pipeline",
            wgpu::BlendState::ALPHA_BLENDING,
            false,
        );
        let meshes = [MeshKind::Cube, MeshKind::Sphere, MeshKind::Plane]
            .into_iter()
            .map(|kind| {
                (
                    MeshSource::Primitive(kind),
                    create_gpu_mesh(device, primitive_mesh_data(kind)),
                )
            })
            .collect();
        let (depth_texture, depth_view) = create_depth_target(device, width, height);

        Self {
            pipeline,
            transparent_pipeline,
            camera_buffer,
            camera_bind_group,
            texture_layout,
            sampler,
            white_texture,
            textures: std::collections::HashMap::new(),
            meshes,
            depth_texture,
            depth_view,
            width: width.max(1),
            height: height.max(1),
            mesh_asset_generation: None,
            texture_asset_generation: None,
            frustum_culling_enabled: true,
            stats: RenderStats::default(),
        }
    }

    /// Enable or disable conservative camera-frustum culling. It is enabled by
    /// default; disabling it is useful for visibility debugging.
    pub fn set_frustum_culling_enabled(&mut self, enabled: bool) {
        self.frustum_culling_enabled = enabled;
    }

    pub fn frustum_culling_enabled(&self) -> bool {
        self.frustum_culling_enabled
    }

    pub fn stats(&self) -> RenderStats {
        self.stats
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.width == width && self.height == height {
            return;
        }
        let (texture, view) = create_depth_target(device, width, height);
        self.depth_texture = texture;
        self.depth_view = view;
        self.width = width;
        self.height = height;
    }

    pub fn render(
        &mut self,
        engine: &Engine,
        camera: RenderCamera,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
    ) {
        let generation = engine.mesh_asset_generation();
        if self
            .mesh_asset_generation
            .is_some_and(|cached| cached != generation)
        {
            self.meshes
                .retain(|source, _| matches!(source, MeshSource::Primitive(_)));
        }
        self.mesh_asset_generation = Some(generation);
        let texture_generation = engine.texture_asset_generation();
        if self
            .texture_asset_generation
            .is_some_and(|cached| cached != texture_generation)
        {
            self.textures.clear();
        }
        self.texture_asset_generation = Some(texture_generation);

        for gpu_mesh in self.meshes.values_mut() {
            gpu_mesh.instances.clear();
            gpu_mesh.draws.clear();
        }
        for (texture_id, texture_data) in engine.texture_assets() {
            if !self.textures.contains_key(&texture_id) {
                self.textures.insert(
                    texture_id,
                    create_gpu_texture(
                        device,
                        queue,
                        &self.texture_layout,
                        &self.sampler,
                        texture_data,
                    ),
                );
            }
        }

        let mut batches = std::collections::HashMap::<
            (MeshSource, Option<TextureAssetId>),
            Vec<InstanceRaw>,
        >::new();
        let mut transparent_instances = Vec::new();
        let forward = (camera.target - camera.position).normalize_or_zero();
        let frustum = self
            .frustum_culling_enabled
            .then(|| camera.frustum(self.width as f32 / self.height as f32))
            .flatten();
        let mut stats = RenderStats::default();
        for (_, model, mesh, material, texture) in engine.renderables() {
            stats.candidate_meshes += 1;
            if !self.meshes.contains_key(&mesh) {
                if let MeshSource::Asset(asset_id) = mesh {
                    if let Some(asset) = engine.mesh_asset(asset_id) {
                        let vertices = asset
                            .vertices()
                            .iter()
                            .map(|v| MeshVertex {
                                position: v.position,
                                normal: v.normal,
                                tex_coords: v.tex_coords,
                            })
                            .collect();
                        let indices = asset.indices().to_vec();
                        self.meshes
                            .insert(mesh, create_gpu_mesh(device, (vertices, indices)));
                    }
                }
            }
            let Some(gpu_mesh) = self.meshes.get(&mesh) else {
                continue;
            };
            if frustum
                .is_some_and(|frustum| !frustum.intersects(gpu_mesh.bounds.transformed(model)))
            {
                stats.frustum_culled_meshes += 1;
                continue;
            }
            stats.visible_meshes += 1;
            let texture_has_alpha = texture
                .and_then(|id| self.textures.get(&id))
                .is_some_and(|texture| texture.has_alpha);
            let instance = instance_data(model, material);
            let is_transparent = match material.alpha_mode {
                AlphaMode3d::Opaque | AlphaMode3d::Mask => false,
                AlphaMode3d::Blend => true,
                AlphaMode3d::Auto => material.base_color[3] < 1.0 || texture_has_alpha,
            };
            if is_transparent {
                stats.transparent_meshes += 1;
                let center = gpu_mesh.bounds.transformed(model).center;
                let depth = forward.dot(center - camera.position);
                transparent_instances.push(TransparentInstance {
                    depth: if depth.is_finite() { depth } else { 0.0 },
                    mesh,
                    texture,
                    data: instance,
                });
            } else {
                batches.entry((mesh, texture)).or_default().push(instance);
            }
        }
        stats.instance_count =
            batches.values().map(Vec::len).sum::<usize>() + transparent_instances.len();
        for ((mesh, texture), instances) in batches {
            if let Some(gpu_mesh) = self.meshes.get_mut(&mesh) {
                let start = gpu_mesh.instances.len() as u32;
                gpu_mesh.instances.extend(instances);
                let end = gpu_mesh.instances.len() as u32;
                gpu_mesh.draws.push((texture, start..end));
                stats.draw_calls += 1;
            }
        }
        transparent_instances.sort_by(|left, right| right.depth.total_cmp(&left.depth));
        let mut transparent_draws = Vec::with_capacity(transparent_instances.len());
        for instance in transparent_instances {
            if let Some(gpu_mesh) = self.meshes.get_mut(&instance.mesh) {
                let start = gpu_mesh.instances.len() as u32;
                gpu_mesh.instances.push(instance.data);
                let end = gpu_mesh.instances.len() as u32;
                transparent_draws.push((instance.mesh, instance.texture, start..end));
                stats.draw_calls += 1;
            }
        }
        self.stats = stats;
        for gpu_mesh in self.meshes.values_mut() {
            gpu_mesh.ensure_instance_capacity(device, gpu_mesh.instances.len());
            if !gpu_mesh.instances.is_empty() {
                queue.write_buffer(
                    &gpu_mesh.instance_buffer,
                    0,
                    bytemuck::cast_slice(&gpu_mesh.instances),
                );
            }
        }

        let lighting = engine.lighting();
        let mut point_lights = engine
            .point_lights()
            .map(|(_, position, light)| {
                let distance_squared = position.distance_squared(camera.position);
                let strongest_channel = light.color.iter().copied().fold(0.0_f32, f32::max);
                let score = light.intensity * strongest_channel * light.radius * light.radius
                    / (distance_squared + 1.0);
                (
                    score,
                    PointLightUniform {
                        position_radius: [position.x, position.y, position.z, light.radius],
                        color_intensity: [
                            light.color[0],
                            light.color[1],
                            light.color[2],
                            light.intensity,
                        ],
                    },
                )
            })
            .collect::<Vec<_>>();
        point_lights.sort_by(|left, right| right.0.total_cmp(&left.0));
        point_lights.truncate(MAX_POINT_LIGHTS);
        let mut point_light_uniforms = [PointLightUniform::zeroed(); MAX_POINT_LIGHTS];
        for (index, (_, light)) in point_lights.iter().enumerate() {
            point_light_uniforms[index] = *light;
        }
        let uniform = CameraUniform {
            view_projection: camera
                .view_projection(self.width as f32 / self.height as f32)
                .to_cols_array_2d(),
            position: [camera.position.x, camera.position.y, camera.position.z, 1.0],
            light_direction: lighting.direction.extend(0.0).to_array(),
            light_color_intensity: [
                lighting.color[0],
                lighting.color[1],
                lighting.color[2],
                lighting.intensity,
            ],
            ambient: [
                lighting.ambient[0],
                lighting.ambient[1],
                lighting.ambient[2],
                0.0,
            ],
            point_lights: point_light_uniforms,
            point_light_count: [point_lights.len() as u32, 0, 0, 0],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpteng scene encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gpteng 3d scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(camera.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            for gpu_mesh in self.meshes.values() {
                if gpu_mesh.instances.is_empty() {
                    continue;
                }
                pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu_mesh.instance_buffer.slice(..));
                pass.set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                for (texture_id, instances) in &gpu_mesh.draws {
                    let texture = texture_id
                        .and_then(|id| self.textures.get(&id))
                        .unwrap_or(&self.white_texture);
                    pass.set_bind_group(1, &texture.bind_group, &[]);
                    pass.draw_indexed(0..gpu_mesh.index_count, 0, instances.clone());
                }
            }
        }
        if !transparent_draws.is_empty() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gpteng transparent scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.transparent_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            for (mesh, texture_id, instances) in &transparent_draws {
                let Some(gpu_mesh) = self.meshes.get(mesh) else {
                    continue;
                };
                pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu_mesh.instance_buffer.slice(..));
                pass.set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                let texture = texture_id
                    .and_then(|id| self.textures.get(&id))
                    .unwrap_or(&self.white_texture);
                pass.set_bind_group(1, &texture.bind_group, &[]);
                pass.draw_indexed(0..gpu_mesh.index_count, 0, instances.clone());
            }
        }
        queue.submit(Some(encoder.finish()));
    }
}

impl GpuMesh {
    fn ensure_instance_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required <= self.instance_capacity {
            return;
        }
        self.instance_capacity = required.next_power_of_two();
        self.instance_buffer = create_instance_buffer(device, self.instance_capacity);
    }
}

fn create_scene_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    label: &'static str,
    blend: wgpu::BlendState,
    depth_write_enabled: bool,
) -> wgpu::RenderPipeline {
    let mesh_vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<MeshVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &MESH_VERTEX_ATTRIBUTES,
    };
    let instance_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &INSTANCE_ATTRIBUTES,
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(mesh_vertex_layout), Some(instance_layout)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(depth_write_enabled),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_gpu_mesh(
    device: &wgpu::Device,
    (vertices, indices): (Vec<MeshVertex>, Vec<u32>),
) -> GpuMesh {
    let bounds = MeshBounds::from_vertices(&vertices);
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpteng cached primitive vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpteng cached primitive indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let instance_capacity = 1;
    GpuMesh {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        instance_buffer: create_instance_buffer(device, instance_capacity),
        instance_capacity,
        instances: Vec::new(),
        draws: Vec::new(),
        bounds,
    }
}

fn create_gpu_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    data: &TextureData,
) -> GpuTexture {
    let has_alpha = data.rgba8.chunks_exact(4).any(|pixel| pixel[3] < 255);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gpteng sRGBA material texture"),
        size: wgpu::Extent3d {
            width: data.width,
            height: data.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data.rgba8,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(data.width * 4),
            rows_per_image: Some(data.height),
        },
        wgpu::Extent3d {
            width: data.width,
            height: data.height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpteng material texture bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    GpuTexture {
        _texture: texture,
        _view: view,
        bind_group,
        has_alpha,
    }
}

fn create_instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpteng primitive instance data"),
        size: (capacity.max(1) * std::mem::size_of::<InstanceRaw>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn instance_data(model: Mat4, material: crate::Material3d) -> InstanceRaw {
    let linear = Mat3::from_mat4(model);
    let normal_matrix = if linear.determinant().abs() > f32::EPSILON {
        linear.inverse().transpose()
    } else {
        let (_, rotation, _) = model.to_scale_rotation_translation();
        Mat3::from_quat(rotation)
    };
    let normal_columns = normal_matrix.to_cols_array_2d();
    InstanceRaw {
        model: model.to_cols_array_2d(),
        normal: normal_columns.map(|column| [column[0], column[1], column[2], 0.0]),
        color: material.base_color,
        surface: [
            material.roughness,
            material.metallic,
            match material.alpha_mode {
                AlphaMode3d::Opaque => 0.0,
                AlphaMode3d::Mask => 1.0,
                AlphaMode3d::Blend => 2.0,
                AlphaMode3d::Auto => 3.0,
            },
            material.alpha_cutoff,
        ],
    }
}

fn create_depth_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gpteng scene depth buffer"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn primitive_mesh_data(kind: MeshKind) -> (Vec<MeshVertex>, Vec<u32>) {
    match kind {
        MeshKind::Cube => {
            let vertices = CUBE_TRIANGLES
                .iter()
                .enumerate()
                .map(|(index, position)| {
                    let normal = CUBE_FACE_NORMALS[index / 6];
                    let tex_coords = if normal.x.abs() > 0.5 {
                        [position[2] + 0.5, position[1] + 0.5]
                    } else if normal.y.abs() > 0.5 {
                        [position[0] + 0.5, position[2] + 0.5]
                    } else {
                        [position[0] + 0.5, position[1] + 0.5]
                    };
                    MeshVertex {
                        position: *position,
                        normal: normal.to_array(),
                        tex_coords,
                    }
                })
                .collect::<Vec<_>>();
            let indices = (0..vertices.len() as u32).collect();
            (vertices, indices)
        }
        MeshKind::Plane => {
            let vertices = [
                Vec3::new(-0.5, 0.0, -0.5),
                Vec3::new(0.5, 0.0, -0.5),
                Vec3::new(0.5, 0.0, 0.5),
                Vec3::new(-0.5, 0.0, 0.5),
            ]
            .into_iter()
            .enumerate()
            .map(|(index, position)| MeshVertex {
                position: position.to_array(),
                normal: Vec3::Y.to_array(),
                tex_coords: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]][index],
            })
            .collect();
            (vertices, vec![0, 2, 1, 0, 3, 2])
        }
        MeshKind::Sphere => {
            const LONGITUDE_SEGMENTS: usize = 24;
            const LATITUDE_SEGMENTS: usize = 16;
            let row_length = LONGITUDE_SEGMENTS + 1;
            let mut vertices = Vec::with_capacity((LATITUDE_SEGMENTS + 1) * row_length);
            for latitude in 0..=LATITUDE_SEGMENTS {
                let v = latitude as f32 / LATITUDE_SEGMENTS as f32;
                for longitude in 0..=LONGITUDE_SEGMENTS {
                    let u = longitude as f32 / LONGITUDE_SEGMENTS as f32;
                    let position = sphere_point(u, v);
                    vertices.push(MeshVertex {
                        position: position.to_array(),
                        normal: position.normalize_or_zero().to_array(),
                        tex_coords: [u, v],
                    });
                }
            }

            let mut indices = Vec::with_capacity(LATITUDE_SEGMENTS * LONGITUDE_SEGMENTS * 6);
            for latitude in 0..LATITUDE_SEGMENTS {
                for longitude in 0..LONGITUDE_SEGMENTS {
                    let top_left = (latitude * row_length + longitude) as u32;
                    let bottom_left = top_left + row_length as u32;
                    let top_right = top_left + 1;
                    let bottom_right = bottom_left + 1;
                    indices.extend_from_slice(&[
                        top_left,
                        top_right,
                        bottom_right,
                        top_left,
                        bottom_right,
                        bottom_left,
                    ]);
                }
            }
            (vertices, indices)
        }
    }
}

fn sphere_point(longitude: f32, latitude: f32) -> Vec3 {
    let azimuth = longitude * std::f32::consts::TAU;
    let polar = latitude * std::f32::consts::PI;
    let radius = 0.5;
    Vec3::new(
        radius * polar.sin() * azimuth.cos(),
        radius * polar.cos(),
        radius * polar.sin() * azimuth.sin(),
    )
}

const CUBE_TRIANGLES: [[f32; 3]; 36] = [
    // Front (+Z)
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    // Back (-Z)
    [0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    // Right (+X)
    [0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
    // Left (-X)
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5],
    // Top (+Y)
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    // Bottom (-Y)
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, 0.5],
];

const CUBE_FACE_NORMALS: [Vec3; 6] = [
    Vec3::Z,
    Vec3::NEG_Z,
    Vec3::X,
    Vec3::NEG_X,
    Vec3::Y,
    Vec3::NEG_Y,
];
