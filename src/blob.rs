use bytemuck::{Pod, Zeroable};

// ─── GPU structs ─────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuBump {
    pub direction_amplitude: [f32; 4],  // xyz = bump direction (normalized), w = amplitude
    pub concentration_pad: [f32; 4],    // x = concentration, yzw unused
    pub lean_axis_amount: [f32; 4],     // xyz = lean rotation axis, w = lean_amount
    pub bend_amount_pad: [f32; 4],      // x = bend_amount, yzw unused
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BlobUniforms {
    pub center_rmin: [f32; 4],          // xyz = center, w = r_min
    pub counts: [u32; 4],               // x = bump_count, y = particle_count, z = bend_mode, w unused
    pub bumps: [GpuBump; 8],
}

// ─── CPU blob parameters ─────────────────────────────────────────────────────

pub struct BumpParams {
    pub theta: f32,
    pub phi: f32,
    pub amplitude: f32,
    pub concentration: f32,
    pub lean_direction_phi: f32,
    pub lean_amount: f32,
    pub bend_amount: f32,
}

impl BumpParams {
    pub fn new(theta: f32, phi: f32, amplitude: f32, concentration: f32) -> Self {
        Self {
            theta,
            phi,
            amplitude,
            concentration,
            lean_direction_phi: 0.0,
            lean_amount: 0.0,
            bend_amount: 0.0,
        }
    }

    pub fn direction(&self) -> [f32; 3] {
        [
            self.theta.sin() * self.phi.cos(),
            self.theta.cos(),
            self.theta.sin() * self.phi.sin(),
        ]
    }

    /// Lean axis in world space — same computation as to_gpu().
    pub fn lean_axis(&self) -> [f32; 3] {
        use crate::math::{normalize, cross};
        let d = self.direction();
        let world_up = if d[1].abs() < 0.99 { [0.0f32, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let tangent   = normalize(cross(world_up, d));
        let bitangent = cross(d, tangent);
        let lean_dir = [
            tangent[0] * self.lean_direction_phi.cos() + bitangent[0] * self.lean_direction_phi.sin(),
            tangent[1] * self.lean_direction_phi.cos() + bitangent[1] * self.lean_direction_phi.sin(),
            tangent[2] * self.lean_direction_phi.cos() + bitangent[2] * self.lean_direction_phi.sin(),
        ];
        let raw = cross(d, lean_dir);
        let len = (raw[0].powi(2) + raw[1].powi(2) + raw[2].powi(2)).sqrt();
        if len > 1e-6 { [raw[0]/len, raw[1]/len, raw[2]/len] } else { [0.0, 0.0, 1.0] }
    }

    /// Ghost point position given a total tip angle (lean + bend at peak).
    /// ghost = bump_dir * rmin - normalize(lever_peak) * rmin
    fn ghost_magnitude(bump_dir: [f32; 3], amplitude: f32, lean_axis: [f32; 3], total_angle: f32, r_min: f32) -> f32 {
        use crate::math::{rotate_around_axis, scale, normalize, sub};
        let lever = rotate_around_axis(scale(bump_dir, amplitude), lean_axis, total_angle);
        let ghost = sub(scale(bump_dir, r_min), scale(normalize(lever), r_min));
        (ghost[0].powi(2) + ghost[1].powi(2) + ghost[2].powi(2)).sqrt()
    }

    /// Maximum safe lean — fixed 60 degree ceiling for now.
    pub fn max_lean(&self) -> f32 {
        std::f32::consts::PI / 3.0
    }

    /// Maximum safe bend — bisect on tangent condition in lean plane.
    /// Worst case: all angle is bend, lean = 0.
    pub fn max_bend(&self, r_min: f32) -> f32 {
        let c = self.concentration;
        let a = self.amplitude;

        let ghost_2d = |bend_angle: f32| -> [f32; 2] {
            let lx = bend_angle.sin() * a;
            let ly = bend_angle.cos() * a;
            let len = (lx*lx + ly*ly).sqrt();
            [-lx / len * r_min, r_min - ly / len * r_min]
        };

        let tangent_cross = |theta: f32, gx: f32, gy: f32| -> f32 {
            let exp_ratio = (c * theta.cos()).exp() / c.exp();
            let r  = r_min + a * exp_ratio;
            let dr = -a * c * theta.sin() * exp_ratio;
            let px = theta.sin() * r;
            let py = theta.cos() * r;
            let dpx = theta.cos() * r + theta.sin() * dr;
            let dpy = -theta.sin() * r + theta.cos() * dr;
            (px - gx) * dpy - (py - gy) * dpx
        };

        // Original tangent-from-ghost check — disabled: ghost near origin for small amplitudes
        // means tangent condition never triggers. Left for reference.
        // let tangent_exists = |bend_angle: f32| -> bool {
        //     let [gx, gy] = ghost_2d(bend_angle);
        //     let n = 64usize;
        //     let mut prev = tangent_cross(0.01, gx, gy);
        //     for i in 1..=n {
        //         let theta = 0.01 + (std::f32::consts::PI - 0.01) * i as f32 / n as f32;
        //         let curr = tangent_cross(theta, gx, gy);
        //         if prev * curr < 0.0 { return true; }
        //         prev = curr;
        //     }
        //     false
        // };

        // Full deformation formula in 2D lean plane.
        // In lean plane: bump_dir = (0,1), lean dir = (1,0)
        // f(theta) = exp(c*cos(theta)) / exp(c)  — Von-Mises weight
        // lever(theta) = (sin(bend*f)*a, cos(bend*f)*a)
        // pos(theta) = (sin(theta)*rmin + sin(bend*f)*a, cos(theta)*rmin + cos(bend*f)*a)
        // ghost = (-sin(bend)*rmin, rmin*(1-cos(bend)))
        // Tangent condition: (pos - ghost) x pos'(theta) = 0

        let pos_and_deriv = |theta: f32, bend_angle: f32| -> ([f32;2], [f32;2]) {
            let exp_ratio = (c * theta.cos()).exp() / c.exp();
            let f  = exp_ratio;
            let df = -c * theta.sin() * exp_ratio; // df/dtheta

            let bf  = bend_angle * f;
            let dbf = bend_angle * df; // d(bend*f)/dtheta

            let px = theta.sin() * r_min + bf.sin() * a;
            let py = theta.cos() * r_min + bf.cos() * a;
            let dpx = theta.cos() * r_min + bf.cos() * a * dbf;
            let dpy = -theta.sin() * r_min - bf.sin() * a * dbf;
            ([px, py], [dpx, dpy])
        };

        let ghost_pos = |bend_angle: f32| -> [f32; 2] {
            [-bend_angle.sin() * r_min, r_min * (1.0 - bend_angle.cos())]
        };

        let tangent_cross_full = |theta: f32, bend_angle: f32| -> f32 {
            let ([px, py], [dpx, dpy]) = pos_and_deriv(theta, bend_angle);
            let [gx, gy] = ghost_pos(bend_angle);
            (px - gx) * dpy - (py - gy) * dpx
        };

        let tangent_exists_full = |bend_angle: f32| -> bool {
            let n = 128usize;
            let mut prev = tangent_cross_full(0.001, bend_angle);
            for i in 1..=n {
                let theta = 0.001 + (std::f32::consts::PI - 0.001) * i as f32 / n as f32;
                let curr = tangent_cross_full(theta, bend_angle);
                if prev * curr < 0.0 { return true; }
                prev = curr;
            }
            false
        };

        if tangent_exists_full(0.001) { return 0.0; }
        if !tangent_exists_full(std::f32::consts::FRAC_PI_2) {
            return std::f32::consts::FRAC_PI_2;
        }

        let mut lo = 0.0f32;
        let mut hi = std::f32::consts::FRAC_PI_2;
        for _ in 0..24 {
            let mid = (lo + hi) * 0.5;
            if tangent_exists_full(mid) { hi = mid; } else { lo = mid; }
        }
        lo
    }


    pub fn to_gpu(&self) -> GpuBump {
        use crate::math::{normalize, cross};
        let d = self.direction();

        let world_up = if d[1].abs() < 0.99 { [0.0f32, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let tangent = normalize(cross(world_up, d));
        let bitangent = cross(d, tangent);

        let lean_dir = [
            tangent[0] * self.lean_direction_phi.cos() + bitangent[0] * self.lean_direction_phi.sin(),
            tangent[1] * self.lean_direction_phi.cos() + bitangent[1] * self.lean_direction_phi.sin(),
            tangent[2] * self.lean_direction_phi.cos() + bitangent[2] * self.lean_direction_phi.sin(),
        ];

        let lean_axis_raw = cross(d, lean_dir);
        let len = (lean_axis_raw[0].powi(2) + lean_axis_raw[1].powi(2) + lean_axis_raw[2].powi(2)).sqrt();
        let lean_axis = if len > 1e-6 {
            [lean_axis_raw[0]/len, lean_axis_raw[1]/len, lean_axis_raw[2]/len]
        } else {
            [0.0, 0.0, 1.0]
        };

        GpuBump {
            direction_amplitude: [d[0], d[1], d[2], self.amplitude],
            concentration_pad: [self.concentration, 0.0, 0.0, 0.0],
            lean_axis_amount: [lean_axis[0], lean_axis[1], lean_axis[2], self.lean_amount],
            bend_amount_pad: [self.bend_amount, 0.0, 0.0, 0.0],
        }
    }
}

pub struct BlobParams {
    pub r_min: f32,
    pub bumps: Vec<BumpParams>,
    pub target_volume: f32,
    /// 0 = Ghost Sphere, 1 = Joint Rotation, 2 = Simple Translation
    pub bend_mode: u32,
}

impl BlobParams {
    pub fn new(r_min: f32, bumps: Vec<BumpParams>) -> Self {
        let mut b = Self { r_min, bumps, target_volume: 0.0, bend_mode: 0 };
        b.target_volume = compute_volume(b.r_min, &b.bumps);
        b
    }

    pub fn to_uniforms(&self, num_particles: u32) -> BlobUniforms {
        let mut gpu_bumps = [GpuBump::zeroed(); 8];
        for (i, b) in self.bumps.iter().enumerate().take(8) {
            gpu_bumps[i] = b.to_gpu();
        }
        BlobUniforms {
            center_rmin: [0.0, 0.0, 0.0, self.r_min],
            counts: [self.bumps.len() as u32, num_particles, self.bend_mode, 0],
            bumps: gpu_bumps,
        }
    }
}

// ─── Volume conservation ─────────────────────────────────────────────────────

pub fn eval_r_cpu(r_min: f32, bumps: &[BumpParams], theta: f32, phi: f32) -> f32 {
    let dir = [
        theta.sin() * phi.cos(),
        theta.cos(),
        theta.sin() * phi.sin(),
    ];
    let mut r = r_min;
    for bump in bumps {
        let d = bump.direction();
        let dot = (dir[0]*d[0] + dir[1]*d[1] + dir[2]*d[2]).clamp(-1.0, 1.0);
        let normalized = (bump.concentration * dot).exp() / bump.concentration.exp();
        r += bump.amplitude * normalized;
    }
    r
}

pub fn compute_volume(r_min: f32, bumps: &[BumpParams]) -> f32 {
    let n_theta = 64usize;
    let n_phi = 64usize;
    let mut sum = 0.0f32;
    for i in 0..n_theta {
        let theta = std::f32::consts::PI * (i as f32 + 0.5) / n_theta as f32;
        for j in 0..n_phi {
            let phi = std::f32::consts::TAU * j as f32 / n_phi as f32;
            let r = eval_r_cpu(r_min, bumps, theta, phi);
            sum += r * r * r * theta.sin();
        }
    }
    let d_theta = std::f32::consts::PI / n_theta as f32;
    let d_phi = std::f32::consts::TAU / n_phi as f32;
    sum * d_theta * d_phi / 3.0
}

pub fn conserve_volume(params: &mut BlobParams) {
    let current = compute_volume(params.r_min, &params.bumps);
    if current <= 0.0 { return; }
    let scale = (params.target_volume / current).cbrt();
    params.r_min *= scale;
    for bump in &mut params.bumps {
        bump.amplitude *= scale;
    }
}
