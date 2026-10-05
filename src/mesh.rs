//! CPU mesh assets and bounded Wavefront OBJ and glTF importers.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_OBJ_SOURCE_BYTES: usize = 8 * 1024 * 1024;
const MAX_GLTF_SOURCE_BYTES: usize = 32 * 1024 * 1024;
const MAX_TRIANGLE_VERTICES: usize = 1_000_000;
const MAX_GLTF_PRIMITIVES: usize = 4_096;
const MAX_GLTF_NODES: usize = 100_000;
const MAX_GLTF_ANIMATIONS: usize = 1_024;
const MAX_GLTF_ANIMATION_KEYFRAMES: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct MeshAssetId(pub(crate) u64);

impl MeshAssetId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Texture coordinates in the 0..1 image domain. Values outside that range
    /// are allowed and repeat through the sampler's address mode.
    #[serde(default)]
    pub tex_coords: [f32; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshData {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct GltfAssetData {
    pub nodes: Vec<GltfNodeData>,
    pub primitives: Vec<GltfPrimitiveData>,
    pub animations: Vec<GltfAnimationData>,
    pub images: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GltfAnimationPath {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GltfInterpolation {
    Step,
    Linear,
    CubicSpline,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GltfAnimationKeyframe {
    pub time: f32,
    pub value: Vec4,
    pub in_tangent: Vec4,
    pub out_tangent: Vec4,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GltfAnimationChannelData {
    pub node_index: usize,
    pub path: GltfAnimationPath,
    pub interpolation: GltfInterpolation,
    pub keyframes: Vec<GltfAnimationKeyframe>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GltfAnimationData {
    pub name: String,
    pub duration_seconds: f32,
    pub channels: Vec<GltfAnimationChannelData>,
}

/// A node in the selected glTF scene. Parent indices refer to this compact,
/// scene-local node array, which keeps unrelated document scenes out of an
/// imported instance.
#[derive(Clone, Debug)]
pub struct GltfNodeData {
    pub name: String,
    pub transform: [f32; 16],
    pub parent: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct GltfPrimitiveData {
    pub name: String,
    pub node_index: usize,
    pub mesh: MeshData,
    pub base_color: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub alpha_mode: gltf::material::AlphaMode,
    pub alpha_cutoff: f32,
    pub base_color_image: Option<usize>,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum MeshLoadError {
    #[error("mesh has no indexed triangles")]
    Empty,
    #[error("mesh contains a non-finite vertex or normal")]
    NonFinite,
    #[error("mesh index {index} is outside the {vertex_count} vertices")]
    IndexOutOfBounds { index: u32, vertex_count: usize },
    #[error("OBJ source is {size} bytes, maximum is {limit}")]
    SourceTooLarge { size: usize, limit: usize },
    #[error("OBJ line {line}: {message}")]
    ObjLine { line: usize, message: String },
    #[error("OBJ mesh exceeds the {0} triangle-vertex limit")]
    TooManyVertices(usize),
    #[error("glTF source is {size} bytes, maximum is {limit}")]
    GltfSourceTooLarge { size: usize, limit: usize },
    #[error("glTF import failed: {0}")]
    Gltf(String),
    #[error("glTF references an external resource; embed buffers as data URIs or use GLB")]
    GltfExternalResource,
}

impl MeshData {
    pub fn new(vertices: Vec<MeshVertex>, indices: Vec<u32>) -> Result<Self, MeshLoadError> {
        if vertices.is_empty() || indices.is_empty() || indices.len() % 3 != 0 {
            return Err(MeshLoadError::Empty);
        }
        if vertices.iter().any(|vertex| {
            vertex
                .position
                .iter()
                .chain(vertex.normal.iter())
                .chain(vertex.tex_coords.iter())
                .any(|value| !value.is_finite())
        }) {
            return Err(MeshLoadError::NonFinite);
        }
        for &index in &indices {
            if index as usize >= vertices.len() {
                return Err(MeshLoadError::IndexOutOfBounds {
                    index,
                    vertex_count: vertices.len(),
                });
            }
        }
        Ok(Self { vertices, indices })
    }

    /// Parse OBJ positions, texture coordinates, normals, and polygon faces.
    /// Faces are triangulated and missing normals are generated per face.
    pub fn from_obj(source: &str) -> Result<Self, MeshLoadError> {
        if source.len() > MAX_OBJ_SOURCE_BYTES {
            return Err(MeshLoadError::SourceTooLarge {
                size: source.len(),
                limit: MAX_OBJ_SOURCE_BYTES,
            });
        }

        let mut positions = Vec::<Vec3>::new();
        let mut normals = Vec::<Vec3>::new();
        let mut tex_coords = Vec::<[f32; 2]>::new();
        let mut vertices = Vec::<MeshVertex>::new();
        let mut indices = Vec::<u32>::new();

        for (line_index, raw_line) in source.lines().enumerate() {
            let line_number = line_index + 1;
            let content = raw_line.split('#').next().unwrap_or_default().trim();
            if content.is_empty() {
                continue;
            }
            let mut words = content.split_whitespace();
            let Some(tag) = words.next() else {
                continue;
            };
            match tag {
                "v" => positions.push(parse_vec3(words, line_number, "position")?),
                "vn" => normals.push(parse_vec3(words, line_number, "normal")?),
                "vt" => tex_coords.push(parse_vec2(words, line_number, "texture coordinate")?),
                "f" => {
                    let face = words
                        .map(|word| {
                            parse_face_index(
                                word,
                                positions.len(),
                                tex_coords.len(),
                                normals.len(),
                                line_number,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if face.len() < 3 {
                        return Err(obj_error(line_number, "face needs at least three vertices"));
                    }
                    for corner in 1..face.len() - 1 {
                        let triangle = [face[0], face[corner], face[corner + 1]];
                        let p0 = positions[triangle[0].0];
                        let p1 = positions[triangle[1].0];
                        let p2 = positions[triangle[2].0];
                        let face_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                        if face_normal == Vec3::ZERO {
                            continue;
                        }
                        for (position_index, tex_coord_index, normal_index) in triangle {
                            if vertices.len() >= MAX_TRIANGLE_VERTICES {
                                return Err(MeshLoadError::TooManyVertices(MAX_TRIANGLE_VERTICES));
                            }
                            let normal = normal_index
                                .map(|index| normals[index].normalize_or_zero())
                                .filter(|normal| *normal != Vec3::ZERO)
                                .unwrap_or(face_normal);
                            let vertex_index = vertices.len() as u32;
                            vertices.push(MeshVertex {
                                position: positions[position_index].to_array(),
                                normal: normal.to_array(),
                                tex_coords: tex_coord_index
                                    .map(|index| tex_coords[index])
                                    .unwrap_or([0.0, 0.0]),
                            });
                            indices.push(vertex_index);
                        }
                    }
                }
                _ => {}
            }
        }

        Self::new(vertices, indices)
    }

    /// Import static triangle geometry from a glTF 2.0 JSON or binary GLB
    /// document. GLB buffers and embedded data URIs are supported; external
    /// file and network references are rejected so callers can safely pass
    /// untrusted asset bytes from MCP or a browser.
    pub fn from_gltf(source: &[u8]) -> Result<Self, MeshLoadError> {
        GltfAssetData::from_gltf(source)?.flatten()
    }

    pub fn vertices(&self) -> &[MeshVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
}

impl GltfAssetData {
    /// Parse embedded glTF/GLB geometry, material factors, and images.
    /// External resource paths are rejected.
    pub fn from_gltf(source: &[u8]) -> Result<Self, MeshLoadError> {
        if source.len() > MAX_GLTF_SOURCE_BYTES {
            return Err(MeshLoadError::GltfSourceTooLarge {
                size: source.len(),
                limit: MAX_GLTF_SOURCE_BYTES,
            });
        }
        let gltf = gltf::Gltf::from_slice(source)
            .map_err(|error| MeshLoadError::Gltf(error.to_string()))?;
        for image in gltf.document.images() {
            if let gltf::image::Source::Uri { uri, .. } = image.source()
                && !uri.starts_with("data:")
            {
                return Err(MeshLoadError::GltfExternalResource);
            }
        }
        let mut buffers = Vec::new();
        for buffer in gltf.document.buffers() {
            let data = match buffer.source() {
                gltf::buffer::Source::Bin => gltf
                    .blob
                    .as_deref()
                    .ok_or_else(|| MeshLoadError::Gltf("GLB buffer payload is missing".into()))?
                    .to_vec(),
                gltf::buffer::Source::Uri(uri) => decode_data_uri(uri)?,
            };
            if data.len() < buffer.length() {
                return Err(MeshLoadError::Gltf(format!(
                    "buffer {} is shorter than its declared length",
                    buffer.index()
                )));
            }
            buffers.push(data);
        }

        let mut meshes = Vec::new();
        let mut total_mesh_vertices = 0usize;
        for mesh in gltf.document.meshes() {
            let primitives = read_gltf_mesh(mesh, &buffers)?;
            total_mesh_vertices = total_mesh_vertices.saturating_add(
                primitives
                    .iter()
                    .map(|primitive| primitive.mesh.vertices.len())
                    .sum::<usize>(),
            );
            if total_mesh_vertices > MAX_TRIANGLE_VERTICES {
                return Err(MeshLoadError::TooManyVertices(MAX_TRIANGLE_VERTICES));
            }
            meshes.push(primitives);
        }
        let source_nodes = gltf
            .document
            .nodes()
            .map(|node| GltfNode {
                transform: Mat4::from_cols_array_2d(&node.transform().matrix()),
                mesh: node.mesh().map(|mesh| mesh.index()),
                children: node.children().map(|child| child.index()).collect(),
                name: format!("Node {}", node.index()),
            })
            .collect::<Vec<_>>();
        let scene = gltf
            .document
            .default_scene()
            .or_else(|| gltf.document.scenes().next())
            .ok_or_else(|| MeshLoadError::Gltf("document has no scene".into()))?;
        let mut pending = scene
            .nodes()
            .map(|node| (node.index(), None))
            .collect::<Vec<_>>();
        let mut nodes = Vec::new();
        let mut source_to_compact = vec![None; source_nodes.len()];
        let mut primitives = Vec::new();
        let mut total_instances = 0usize;
        while let Some((source_index, parent_index)) = pending.pop() {
            if nodes.len() >= MAX_GLTF_NODES {
                return Err(MeshLoadError::Gltf(format!(
                    "scene exceeds the {MAX_GLTF_NODES}-node limit"
                )));
            }
            let node = source_nodes.get(source_index).ok_or_else(|| {
                MeshLoadError::Gltf(format!("node {source_index} is out of range"))
            })?;
            let compact_index = nodes.len();
            source_to_compact[source_index] = Some(compact_index);
            nodes.push(GltfNodeData {
                name: node.name.clone(),
                transform: node.transform.to_cols_array(),
                parent: parent_index,
            });
            if let Some(mesh_index) = node.mesh {
                let mesh_primitives = meshes.get(mesh_index).ok_or_else(|| {
                    MeshLoadError::Gltf(format!("mesh {mesh_index} is out of range"))
                })?;
                for primitive in mesh_primitives {
                    if primitives.len() >= MAX_GLTF_PRIMITIVES {
                        return Err(MeshLoadError::Gltf(format!(
                            "scene exceeds the {MAX_GLTF_PRIMITIVES}-primitive limit"
                        )));
                    }
                    total_instances = total_instances.saturating_add(primitive.mesh.vertices.len());
                    if total_instances > MAX_TRIANGLE_VERTICES {
                        return Err(MeshLoadError::TooManyVertices(MAX_TRIANGLE_VERTICES));
                    }
                    primitives.push(GltfPrimitiveData {
                        name: primitive.name.clone(),
                        node_index: compact_index,
                        mesh: primitive.mesh.clone(),
                        base_color: primitive.base_color,
                        roughness: primitive.roughness,
                        metallic: primitive.metallic,
                        alpha_mode: primitive.alpha_mode,
                        alpha_cutoff: primitive.alpha_cutoff,
                        base_color_image: primitive.base_color_image,
                    });
                }
            }
            pending.extend(
                node.children
                    .iter()
                    .rev()
                    .copied()
                    .map(|child| (child, Some(compact_index))),
            );
        }
        let mut images = Vec::new();
        for image in gltf.document.images() {
            let encoded = match image.source() {
                gltf::image::Source::Uri { uri, .. } => decode_data_uri(uri)?,
                gltf::image::Source::View { view, .. } => {
                    let buffer = buffers.get(view.buffer().index()).ok_or_else(|| {
                        MeshLoadError::Gltf("image buffer view references a missing buffer".into())
                    })?;
                    let start = view.offset();
                    let end = start.checked_add(view.length()).ok_or_else(|| {
                        MeshLoadError::Gltf("image buffer view range overflowed".into())
                    })?;
                    buffer
                        .get(start..end)
                        .ok_or_else(|| {
                            MeshLoadError::Gltf("image buffer view is outside its buffer".into())
                        })?
                        .to_vec()
                }
            };
            images.push(encoded);
        }
        if primitives.is_empty() {
            return Err(MeshLoadError::Empty);
        }
        for primitive in &primitives {
            if primitive
                .base_color_image
                .is_some_and(|image| image >= images.len())
            {
                return Err(MeshLoadError::Gltf(
                    "base color texture references a missing image".into(),
                ));
            }
        }
        let animations =
            read_gltf_animations(gltf.document.animations(), &buffers, &source_to_compact)?;
        Ok(Self {
            nodes,
            primitives,
            animations,
            images,
        })
    }

    fn flatten(&self) -> Result<MeshData, MeshLoadError> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for primitive in &self.primitives {
            let mut chain = Vec::new();
            let mut node_index = Some(primitive.node_index);
            while let Some(index) = node_index {
                let node = self.nodes.get(index).ok_or_else(|| {
                    MeshLoadError::Gltf("primitive node index is out of range".into())
                })?;
                chain.push(Mat4::from_cols_array(&node.transform));
                node_index = node.parent;
            }
            let transform = chain
                .into_iter()
                .rev()
                .fold(Mat4::IDENTITY, |parent, local| parent * local);
            append_transformed_mesh(&primitive.mesh, transform, &mut vertices, &mut indices)?;
        }
        MeshData::new(vertices, indices)
    }
}

#[derive(Debug)]
struct GltfNode {
    transform: Mat4,
    mesh: Option<usize>,
    children: Vec<usize>,
    name: String,
}

fn decode_data_uri(uri: &str) -> Result<Vec<u8>, MeshLoadError> {
    let Some((metadata, payload)) = uri
        .strip_prefix("data:")
        .and_then(|value| value.split_once(','))
    else {
        return Err(MeshLoadError::GltfExternalResource);
    };
    if metadata.split(';').any(|part| part == "base64") {
        STANDARD
            .decode(payload)
            .map_err(|error| MeshLoadError::Gltf(format!("invalid base64 buffer URI: {error}")))
    } else {
        let bytes = payload.as_bytes();
        let mut decoded = Vec::with_capacity(bytes.len());
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] == b'%' {
                let Some(hex) = bytes.get(index + 1..index + 3) else {
                    return Err(MeshLoadError::Gltf("invalid escaped data URI".into()));
                };
                let hex = std::str::from_utf8(hex)
                    .ok()
                    .and_then(|hex| u8::from_str_radix(hex, 16).ok())
                    .ok_or_else(|| MeshLoadError::Gltf("invalid escaped data URI".into()))?;
                decoded.push(hex);
                index += 3;
            } else {
                decoded.push(bytes[index]);
                index += 1;
            }
        }
        Ok(decoded)
    }
}

fn read_gltf_mesh(
    mesh: gltf::Mesh<'_>,
    buffers: &[Vec<u8>],
) -> Result<Vec<GltfPrimitiveData>, MeshLoadError> {
    let mesh_index = mesh.index();
    let mut output = Vec::new();
    for primitive in mesh.primitives() {
        if !matches!(
            primitive.mode(),
            gltf::mesh::Mode::Triangles
                | gltf::mesh::Mode::TriangleStrip
                | gltf::mesh::Mode::TriangleFan
        ) {
            continue;
        }
        let gltf_material = primitive.material();
        let alpha_mode = gltf_material.alpha_mode();
        let alpha_cutoff = gltf_material.alpha_cutoff().unwrap_or(0.5);
        let pbr = gltf_material.pbr_metallic_roughness();
        let base_color = pbr.base_color_factor();
        let roughness = pbr.roughness_factor();
        let metallic = pbr.metallic_factor();
        let base_color_image = pbr
            .base_color_texture()
            .map(|info| info.texture().source().index());
        let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
        let positions = reader
            .read_positions()
            .ok_or_else(|| MeshLoadError::Gltf("triangle primitive has no positions".into()))?
            .map(Vec3::from_array)
            .collect::<Vec<_>>();
        if positions.is_empty() {
            continue;
        }
        if positions.len() > MAX_TRIANGLE_VERTICES {
            return Err(MeshLoadError::TooManyVertices(MAX_TRIANGLE_VERTICES));
        }
        let normals = reader
            .read_normals()
            .map(|values| values.map(Vec3::from_array).collect::<Vec<_>>());
        let tex_coords = reader
            .read_tex_coords(0)
            .map(|values| values.into_f32().map(Vec2::from_array).collect::<Vec<_>>());
        if normals
            .as_ref()
            .is_some_and(|values| values.len() != positions.len())
            || tex_coords
                .as_ref()
                .is_some_and(|values| values.len() != positions.len())
        {
            return Err(MeshLoadError::Gltf(
                "vertex attributes have inconsistent lengths".into(),
            ));
        }
        let source_indices = reader
            .read_indices()
            .map(|values| values.into_u32().collect::<Vec<_>>())
            .unwrap_or_else(|| (0..positions.len() as u32).collect());
        let triangle_index_count = match primitive.mode() {
            gltf::mesh::Mode::Triangles => source_indices.len() / 3 * 3,
            gltf::mesh::Mode::TriangleStrip | gltf::mesh::Mode::TriangleFan => {
                source_indices.len().saturating_sub(2).saturating_mul(3)
            }
            _ => 0,
        };
        if triangle_index_count > MAX_TRIANGLE_VERTICES {
            return Err(MeshLoadError::TooManyVertices(MAX_TRIANGLE_VERTICES));
        }
        let mut triangle_indices = Vec::with_capacity(triangle_index_count);
        match primitive.mode() {
            gltf::mesh::Mode::Triangles => {
                triangle_indices.extend_from_slice(&source_indices[..source_indices.len() / 3 * 3]);
            }
            gltf::mesh::Mode::TriangleStrip => {
                for index in 0..source_indices.len().saturating_sub(2) {
                    let triangle = if index % 2 == 0 {
                        [
                            source_indices[index],
                            source_indices[index + 1],
                            source_indices[index + 2],
                        ]
                    } else {
                        [
                            source_indices[index + 1],
                            source_indices[index],
                            source_indices[index + 2],
                        ]
                    };
                    triangle_indices.extend(triangle);
                }
            }
            gltf::mesh::Mode::TriangleFan => {
                for index in 1..source_indices.len().saturating_sub(1) {
                    triangle_indices.extend([
                        source_indices[0],
                        source_indices[index],
                        source_indices[index + 1],
                    ]);
                }
            }
            _ => unreachable!("non-triangle modes were skipped"),
        }
        if triangle_indices.is_empty() {
            continue;
        }
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        if triangle_indices
            .iter()
            .any(|index| *index as usize >= positions.len())
        {
            return Err(MeshLoadError::Gltf(
                "primitive index is outside its vertex attributes".into(),
            ));
        }
        let base = u32::try_from(vertices.len())
            .map_err(|_| MeshLoadError::TooManyVertices(MAX_TRIANGLE_VERTICES))?;
        if vertices.len().saturating_add(positions.len()) > MAX_TRIANGLE_VERTICES
            || indices.len().saturating_add(triangle_indices.len()) > MAX_TRIANGLE_VERTICES
        {
            return Err(MeshLoadError::TooManyVertices(MAX_TRIANGLE_VERTICES));
        }
        let mut vertex_normals = normals.unwrap_or_else(|| vec![Vec3::ZERO; positions.len()]);
        if vertex_normals.iter().any(|normal| *normal == Vec3::ZERO) {
            let mut generated = vec![Vec3::ZERO; positions.len()];
            for triangle in triangle_indices.chunks_exact(3) {
                let [a, b, c] = [
                    triangle[0] as usize,
                    triangle[1] as usize,
                    triangle[2] as usize,
                ];
                let normal = (positions[b] - positions[a])
                    .cross(positions[c] - positions[a])
                    .normalize_or_zero();
                generated[a] += normal;
                generated[b] += normal;
                generated[c] += normal;
            }
            for (normal, generated) in vertex_normals.iter_mut().zip(generated) {
                if *normal == Vec3::ZERO {
                    *normal = generated.normalize_or_zero();
                }
            }
        }
        for (index, position) in positions.into_iter().enumerate() {
            vertices.push(MeshVertex {
                position: position.to_array(),
                normal: vertex_normals[index].normalize_or_zero().to_array(),
                tex_coords: tex_coords
                    .as_ref()
                    .map(|values| values[index].to_array())
                    .unwrap_or([0.0; 2]),
            });
        }
        indices.extend(triangle_indices.into_iter().map(|index| base + index));
        output.push(GltfPrimitiveData {
            name: format!("Mesh {mesh_index} primitive {}", primitive.index()),
            node_index: 0,
            mesh: MeshData::new(vertices, indices)?,
            base_color,
            roughness,
            metallic,
            alpha_mode,
            alpha_cutoff,
            base_color_image,
        });
    }
    Ok(output)
}

fn read_gltf_animations<'a>(
    animations: impl Iterator<Item = gltf::Animation<'a>>,
    buffers: &'a [Vec<u8>],
    source_to_compact: &[Option<usize>],
) -> Result<Vec<GltfAnimationData>, MeshLoadError> {
    let mut result = Vec::new();
    let mut total_keyframes = 0usize;
    for animation in animations {
        if result.len() >= MAX_GLTF_ANIMATIONS {
            return Err(MeshLoadError::Gltf(format!(
                "document exceeds the {MAX_GLTF_ANIMATIONS}-animation limit"
            )));
        }
        let animation_index = animation.index();
        let mut channels = Vec::new();
        let mut duration_seconds: f32 = 0.0;
        for channel in animation.channels() {
            let Some(node_index) = source_to_compact
                .get(channel.target().node().index())
                .copied()
                .flatten()
            else {
                // The selected scene may omit nodes referenced by another scene's clip.
                continue;
            };
            let path = match channel.target().property() {
                gltf::animation::Property::Translation => GltfAnimationPath::Translation,
                gltf::animation::Property::Rotation => GltfAnimationPath::Rotation,
                gltf::animation::Property::Scale => GltfAnimationPath::Scale,
                // Morph targets are not imported yet; ignoring their channels leaves
                // the base mesh intact without affecting node transform tracks.
                gltf::animation::Property::MorphTargetWeights => continue,
            };
            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Step => GltfInterpolation::Step,
                gltf::animation::Interpolation::Linear => GltfInterpolation::Linear,
                gltf::animation::Interpolation::CubicSpline => GltfInterpolation::CubicSpline,
            };
            let reader = channel.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
            let times = reader
                .read_inputs()
                .ok_or_else(|| {
                    MeshLoadError::Gltf("animation input accessor is unreadable".into())
                })?
                .collect::<Vec<_>>();
            if times.is_empty()
                || times.iter().any(|time| !time.is_finite() || *time < 0.0)
                || times.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(MeshLoadError::Gltf(
                    "animation keyframe times must be finite, nonnegative, and increasing".into(),
                ));
            }
            total_keyframes = total_keyframes.saturating_add(times.len());
            if total_keyframes > MAX_GLTF_ANIMATION_KEYFRAMES {
                return Err(MeshLoadError::Gltf(format!(
                    "document exceeds the {MAX_GLTF_ANIMATION_KEYFRAMES}-keyframe limit"
                )));
            }
            duration_seconds = duration_seconds.max(*times.last().unwrap_or(&0.0));
            let values = match reader.read_outputs().ok_or_else(|| {
                MeshLoadError::Gltf("animation output accessor is unreadable".into())
            })? {
                gltf::animation::util::ReadOutputs::Translations(values) => values
                    .map(|value| Vec4::new(value[0], value[1], value[2], 0.0))
                    .collect::<Vec<_>>(),
                gltf::animation::util::ReadOutputs::Scales(values) => values
                    .map(|value| Vec4::new(value[0], value[1], value[2], 0.0))
                    .collect::<Vec<_>>(),
                gltf::animation::util::ReadOutputs::Rotations(values) => {
                    values.into_f32().map(Vec4::from_array).collect::<Vec<_>>()
                }
                gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => continue,
            };
            if values.iter().any(|value| !value.is_finite()) {
                return Err(MeshLoadError::Gltf(
                    "animation contains a non-finite keyframe value".into(),
                ));
            }
            let cubic = interpolation == GltfInterpolation::CubicSpline;
            let expected_values = times.len() * if cubic { 3 } else { 1 };
            if values.len() != expected_values {
                return Err(MeshLoadError::Gltf(
                    "animation input and output accessor lengths do not match".into(),
                ));
            }
            let keyframes: Vec<GltfAnimationKeyframe> = times
                .into_iter()
                .enumerate()
                .map(|(index, time)| {
                    if cubic {
                        let base = index * 3;
                        GltfAnimationKeyframe {
                            time,
                            in_tangent: values[base],
                            value: values[base + 1],
                            out_tangent: values[base + 2],
                        }
                    } else {
                        GltfAnimationKeyframe {
                            time,
                            value: values[index],
                            in_tangent: Vec4::ZERO,
                            out_tangent: Vec4::ZERO,
                        }
                    }
                })
                .collect();
            if path == GltfAnimationPath::Rotation
                && keyframes
                    .iter()
                    .any(|keyframe| keyframe.value.length_squared() <= f32::EPSILON)
            {
                return Err(MeshLoadError::Gltf(
                    "rotation animation keyframes must contain nonzero quaternions".into(),
                ));
            }
            channels.push(GltfAnimationChannelData {
                node_index,
                path,
                interpolation,
                keyframes,
            });
        }
        if !channels.is_empty() {
            result.push(GltfAnimationData {
                name: format!("Animation {animation_index}"),
                duration_seconds,
                channels,
            });
        }
    }
    Ok(result)
}

fn append_transformed_mesh(
    mesh: &MeshData,
    transform: Mat4,
    vertices: &mut Vec<MeshVertex>,
    indices: &mut Vec<u32>,
) -> Result<(), MeshLoadError> {
    let base = u32::try_from(vertices.len())
        .map_err(|_| MeshLoadError::TooManyVertices(MAX_TRIANGLE_VERTICES))?;
    if vertices.len().saturating_add(mesh.vertices.len()) > MAX_TRIANGLE_VERTICES
        || indices.len().saturating_add(mesh.indices.len()) > MAX_TRIANGLE_VERTICES
    {
        return Err(MeshLoadError::TooManyVertices(MAX_TRIANGLE_VERTICES));
    }
    let determinant = Mat3::from_mat4(transform).determinant();
    if !transform.is_finite() || !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
        return Err(MeshLoadError::Gltf(
            "node has a non-finite or singular transform".into(),
        ));
    }
    let normal_matrix = Mat3::from_mat4(transform).inverse().transpose();
    vertices.extend(mesh.vertices.iter().map(|vertex| {
        let position = transform.transform_point3(Vec3::from_array(vertex.position));
        let normal = normal_matrix
            .mul_vec3(Vec3::from_array(vertex.normal))
            .normalize_or_zero();
        MeshVertex {
            position: position.to_array(),
            normal: normal.to_array(),
            tex_coords: vertex.tex_coords,
        }
    }));
    if determinant < 0.0 {
        indices.extend(
            mesh.indices
                .chunks_exact(3)
                .flat_map(|triangle| [base + triangle[0], base + triangle[2], base + triangle[1]]),
        );
    } else {
        indices.extend(mesh.indices.iter().map(|index| base + index));
    }
    Ok(())
}

fn parse_vec3<'a>(
    mut words: impl Iterator<Item = &'a str>,
    line: usize,
    label: &str,
) -> Result<Vec3, MeshLoadError> {
    let values = (|| {
        Some(Vec3::new(
            words.next()?.parse().ok()?,
            words.next()?.parse().ok()?,
            words.next()?.parse().ok()?,
        ))
    })()
    .ok_or_else(|| obj_error(line, &format!("invalid {label}; expected three numbers")))?;
    if !values.is_finite() {
        return Err(obj_error(line, &format!("{label} must be finite")));
    }
    Ok(values)
}

fn parse_vec2<'a>(
    mut words: impl Iterator<Item = &'a str>,
    line: usize,
    label: &str,
) -> Result<[f32; 2], MeshLoadError> {
    let values = [
        words.next().and_then(|value| value.parse::<f32>().ok()),
        words.next().and_then(|value| value.parse::<f32>().ok()),
    ];
    let result = [
        values[0]
            .ok_or_else(|| obj_error(line, &format!("invalid {label}; expected two numbers")))?,
        values[1]
            .ok_or_else(|| obj_error(line, &format!("invalid {label}; expected two numbers")))?,
    ];
    if result.iter().any(|value| !value.is_finite()) {
        return Err(obj_error(line, &format!("{label} must be finite")));
    }
    Ok(result)
}

fn parse_face_index(
    token: &str,
    position_count: usize,
    tex_coord_count: usize,
    normal_count: usize,
    line: usize,
) -> Result<(usize, Option<usize>, Option<usize>), MeshLoadError> {
    let mut parts = token.split('/');
    let position = parse_obj_index(parts.next().unwrap_or_default(), position_count, line)?;
    let tex_coord = match parts.next() {
        Some(value) if !value.is_empty() => Some(parse_obj_index(value, tex_coord_count, line)?),
        _ => None,
    };
    let normal = match parts.next() {
        Some(value) if !value.is_empty() => Some(parse_obj_index(value, normal_count, line)?),
        _ => None,
    };
    if parts.next().is_some() {
        return Err(obj_error(
            line,
            "face vertex has too many slash-separated indices",
        ));
    }
    Ok((position, tex_coord, normal))
}

fn parse_obj_index(value: &str, count: usize, line: usize) -> Result<usize, MeshLoadError> {
    let parsed = value
        .parse::<i64>()
        .map_err(|_| obj_error(line, "face contains an invalid index"))?;
    if parsed == 0 {
        return Err(obj_error(
            line,
            "OBJ indices are one-based and cannot be zero",
        ));
    }
    let index = if parsed > 0 {
        usize::try_from(parsed - 1).ok()
    } else {
        isize::try_from(count)
            .ok()
            .and_then(|count| {
                isize::try_from(parsed)
                    .ok()
                    .and_then(|parsed| count.checked_add(parsed))
            })
            .and_then(|index| usize::try_from(index).ok())
    }
    .filter(|index| *index < count)
    .ok_or_else(|| obj_error(line, "face index is outside the declared data"))?;
    Ok(index)
}

fn obj_error(line: usize, message: &str) -> MeshLoadError {
    MeshLoadError::ObjLine {
        line,
        message: message.to_owned(),
    }
}
