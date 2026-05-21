pub const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

struct Particle {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
}

var<private> CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
);

@vertex
fn vs_main(particle: Particle, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let corner = CORNERS[vertex_index % 6u];
    let billboard_size = 0.035;

    let to_camera = normalize(camera.camera_pos.xyz - particle.position.xyz);

    var ref_up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(dot(to_camera, ref_up)) > 0.9 {
        ref_up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent   = normalize(cross(ref_up, to_camera));
    let bitangent = normalize(cross(to_camera, tangent));

    let world_offset = (tangent * corner.x + bitangent * corner.y) * billboard_size;
    let world_pos = particle.position.xyz + world_offset;
    let clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);

    var out: VertexOutput;
    out.clip_position = clip_pos;
    out.uv = corner;
    out.world_normal = normalize(particle.normal.xyz);
    out.world_pos = particle.position.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist_sq = dot(in.uv, in.uv);
    let alpha = exp(-dist_sq * 2.0);
    if alpha < 0.02 {
        discard;
    }

    let light_dir = normalize(vec3<f32>(1.0, 1.5, 2.0));
    let view_dir  = normalize(camera.camera_pos.xyz - in.world_pos);
    var normal    = normalize(in.world_normal);
    if dot(normal, view_dir) < 0.0 { normal = -normal; }

    let diffuse = max(dot(normal, light_dir), 0.0);

    let half_dir  = normalize(light_dir + view_dir);
    let spec_power = 64.0;
    let specular  = pow(max(dot(normal, half_dir), 0.0), spec_power);

    let ambient    = 0.15;
    let brightness = ambient + (1.0 - ambient) * diffuse;

    let base_color = vec3<f32>(0.2, 0.8, 1.0);
    let color = base_color * brightness + vec3<f32>(0.8, 0.9, 1.0) * specular * 0.8;

    return vec4<f32>(color, alpha);
}
"#;

pub const COMPUTE_SHADER: &str = r#"
struct Bump {
    direction_amplitude: vec4<f32>,   // xyz = direction, w = amplitude
    concentration_pad:   vec4<f32>,   // x = concentration, yzw unused
    lean_axis_amount:    vec4<f32>,   // xyz = lean axis, w = lean_amount
    bend_amount_pad:     vec4<f32>,   // x = bend_amount, yzw unused
}

struct BlobUniforms {
    center_rmin: vec4<f32>,           // xyz = center, w = r_min
    counts: vec4<u32>,                // x = bump_count, y = particle_count, zw unused
    blob_bumps: array<Bump, 8>,
}

struct Particle {
    position: vec4<f32>,
    normal: vec4<f32>,
}

@group(0) @binding(0) var<uniform> blob: BlobUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

fn rotate_axis_angle(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

// von-Mises weight: 1 at peak, 0 away from bump_dir.
fn von_mises(dir: vec3<f32>, bump_dir: vec3<f32>, concentration: f32) -> f32 {
    return exp(concentration * clamp(dot(dir, bump_dir), -1.0, 1.0)) / exp(concentration);
}

// Joint-based bump contribution.
// j = dir * rmin                        — joint at particle foot
// lever = dir * amplitude * f           — arm along dir, scaled by bump intensity
// angle = lean_amount + bend_amount * f — lean: constant; bend: weighted by height
// Rotate lever around lean_axis by angle, add to joint.
fn bump_contribution(
    dir: vec3<f32>,
    bump_dir: vec3<f32>,
    lean_axis: vec3<f32>,
    lean_amount: f32,
    bend_amount: f32,
    concentration: f32,
    amplitude: f32,
) -> vec3<f32> {
    let f = von_mises(dir, bump_dir, concentration);
    let lever = dir * amplitude * f;
    let angle = lean_amount + bend_amount * f;
    if abs(angle) < 0.0001 { return lever; }
    return rotate_axis_angle(lever, lean_axis, angle);
}

fn eval_pos(dir: vec3<f32>) -> vec3<f32> {
    let rmin = blob.center_rmin.w;
    var pos = dir * rmin;
    for (var i = 0u; i < blob.counts.x; i++) {
        let bump_dir      = blob.blob_bumps[i].direction_amplitude.xyz;
        let amplitude     = blob.blob_bumps[i].direction_amplitude.w;
        let concentration = blob.blob_bumps[i].concentration_pad.x;
        let lean_axis     = blob.blob_bumps[i].lean_axis_amount.xyz;
        let lean_amount   = blob.blob_bumps[i].lean_axis_amount.w;
        let bend_amount   = blob.blob_bumps[i].bend_amount_pad.x;

        pos += bump_contribution(dir, bump_dir, lean_axis, lean_amount, bend_amount, concentration, amplitude);
    }
    return pos;
}

fn fd_normal(dir: vec3<f32>, dtheta: vec3<f32>, dphi: vec3<f32>) -> vec3<f32> {
    let eps = 0.001;
    let dp_dtheta = (eval_pos(normalize(dir + dtheta * eps)) - eval_pos(normalize(dir - dtheta * eps))) / (2.0 * eps);
    if length(dphi) < 0.01 {
        return dp_dtheta;
    }
    let dp_dphi = (eval_pos(normalize(dir + dphi * eps)) - eval_pos(normalize(dir - dphi * eps))) / (2.0 * eps);
    return cross(dp_dtheta, dp_dphi);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= blob.counts.y { return; }

    let golden = 2.399963229;
    let i = f32(idx);
    let n = f32(blob.counts.y);

    let theta = acos(1.0 - 2.0 * (i + 0.5) / n);
    let phi   = golden * i;

    let sin_t = sin(theta);
    let cos_t = cos(theta);
    let sin_p = sin(phi);
    let cos_p = cos(phi);

    let dir = vec3<f32>(sin_t * cos_p, cos_t, sin_t * sin_p);
    let ddir_dtheta = vec3<f32>(cos_t * cos_p, -sin_t, cos_t * sin_p);
    let ddir_dphi   = vec3<f32>(-sin_t * sin_p, 0.0, sin_t * cos_p);

    let pos = blob.center_rmin.xyz + eval_pos(dir);

    var normal = fd_normal(dir, ddir_dtheta, ddir_dphi);
    if length(normal) < 0.001 {
        normal = dir;
    } else if dot(normal, dir) < 0.0 {
        normal = -normal;
    }
    normal = normalize(normal);

    particles[idx].position = vec4<f32>(pos, 1.0);
    particles[idx].normal   = vec4<f32>(normal, 0.0);
}
"#;
