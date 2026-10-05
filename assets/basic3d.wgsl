struct PointLightUniform {
    position_radius: vec4<f32>,
    color_intensity: vec4<f32>,
};

struct CameraUniform {
    view_projection: mat4x4<f32>,
    position: vec4<f32>,
    light_direction: vec4<f32>,
    light_color_intensity: vec4<f32>,
    ambient: vec4<f32>,
    point_lights: array<PointLightUniform, 8>,
    point_light_count: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var base_color_texture: texture_2d<f32>;
@group(1) @binding(1)
var base_color_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) normal_0: vec4<f32>,
    @location(8) normal_1: vec4<f32>,
    @location(9) normal_2: vec4<f32>,
    @location(10) color: vec4<f32>,
    @location(11) surface: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) surface: vec4<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    let normal_matrix = mat3x3<f32>(input.normal_0.xyz, input.normal_1.xyz, input.normal_2.xyz);
    let world_position = model * vec4<f32>(input.position, 1.0);
    output.position = camera.view_projection * world_position;
    output.color = input.color;
    output.normal = normal_matrix * input.normal;
    output.surface = input.surface;
    output.world_position = world_position.xyz;
    output.tex_coords = input.tex_coords;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = input.color * textureSample(base_color_texture, base_color_sampler, input.tex_coords);
    let alpha_mode = input.surface.z;
    if alpha_mode > 0.5 && alpha_mode < 1.5 && base_color.a < input.surface.w {
        discard;
    }
    let light_direction = normalize(camera.light_direction.xyz);
    let normal = normalize(input.normal);
    let view_direction = normalize(camera.position.xyz - input.world_position);
    let half_direction = normalize(light_direction + view_direction);
    let roughness = clamp(input.surface.x, 0.04, 1.0);
    let metallic = clamp(input.surface.y, 0.0, 1.0);
    let n_dot_l = max(dot(normal, light_direction), 0.0);
    let n_dot_h = max(dot(normal, half_direction), 0.0);
    let exponent = mix(128.0, 2.0, roughness * roughness);
    let specular_strength = pow(n_dot_h, exponent);
    let f0 = mix(vec3<f32>(0.04), base_color.rgb, metallic);
    let light_radiance = camera.light_color_intensity.rgb * camera.light_color_intensity.w;
    var diffuse = base_color.rgb * (1.0 - metallic) * n_dot_l * light_radiance;
    var specular = f0 * specular_strength * (1.0 - roughness * 0.55) * light_radiance;
    var light_index = 0u;
    loop {
        if light_index >= camera.point_light_count.x {
            break;
        }
        let point_light = camera.point_lights[light_index];
        let offset = point_light.position_radius.xyz - input.world_position;
        let distance = length(offset);
        let radius = point_light.position_radius.w;
        if distance > 0.0001 && distance < radius {
            let point_direction = offset / distance;
            let point_half_direction = normalize(point_direction + view_direction);
            let point_attenuation = pow(1.0 - distance / radius, 2.0);
            let point_radiance = point_light.color_intensity.rgb
                * point_light.color_intensity.w * point_attenuation;
            let point_n_dot_l = max(dot(normal, point_direction), 0.0);
            let point_n_dot_h = max(dot(normal, point_half_direction), 0.0);
            let point_specular = pow(point_n_dot_h, exponent)
                * (1.0 - roughness * 0.55);
            diffuse += base_color.rgb * (1.0 - metallic) * point_n_dot_l * point_radiance;
            specular += f0 * point_specular * point_radiance;
        }
        light_index += 1u;
    }
    let ambient = base_color.rgb * camera.ambient.rgb * (1.0 - metallic);
    var alpha = base_color.a;
    if alpha_mode < 0.5 || (alpha_mode > 0.5 && alpha_mode < 1.5) {
        alpha = 1.0;
    }
    return vec4<f32>(ambient + diffuse * 0.76 + specular, alpha);
}
