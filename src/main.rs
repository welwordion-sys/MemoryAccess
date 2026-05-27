mod blob;
mod environment;
mod gpu;
mod math;
mod shaders;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use blob::{BlobParams, BumpParams, conserve_volume};
use gpu::GpuState;

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    blob_params: BlobParams,
}

impl Default for App {
    fn default() -> Self {
        let params = BlobParams::new(
            0.3,
            vec![
                BumpParams::new(std::f32::consts::FRAC_PI_2, 0.0, 0.3, 8.0),
                BumpParams::new(std::f32::consts::FRAC_PI_2, std::f32::consts::PI, 0.3, 8.0),
            ],
        );
        Self {
            window: None,
            gpu: None,
            blob_params: params,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Blob World")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .unwrap(),
        );
        let gpu = pollster::block_on(GpuState::new(Arc::clone(&window)));
        self.window = Some(window);
        self.gpu = Some(gpu);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let (Some(gpu), Some(window)) = (&mut self.gpu, &self.window) {
            let _ = gpu.egui_state.on_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::RedrawRequested => {
                if let (Some(gpu), Some(window)) = (&mut self.gpu, &self.window) {
                    let raw_input = gpu.egui_state.take_egui_input(window);
                    let full_output = gpu.egui_ctx.run(raw_input, |ctx| {
                        egui::Window::new("Blob Parameters").show(ctx, |ui| {
                            ui.add(egui::Slider::new(&mut self.blob_params.target_volume, 0.01..=10.0).text("volume"));

                            // ── Bend Mode Toggle ──────────────────────────────
                            ui.separator();
                            ui.label("Bend Mode");
                            ui.horizontal(|ui| {
                                ui.selectable_value(&mut self.blob_params.bend_mode, 0, "Ghost Sphere");
                                ui.selectable_value(&mut self.blob_params.bend_mode, 1, "Joint Rotation");
                                ui.selectable_value(&mut self.blob_params.bend_mode, 2, "Translation");
                            });
                            let mode_desc = match self.blob_params.bend_mode {
                                0 => "Shifted evaluation origin — soft coherent curvature",
                                1 => "Surface point pivots around bump foot — true geometric bend",
                                2 => "Direction offset — cheap baseline, tends to compress",
                                _ => "",
                            };
                            ui.label(egui::RichText::new(mode_desc).small().weak());
                            // ─────────────────────────────────────────────────

                            for (i, bump) in self.blob_params.bumps.iter_mut().enumerate() {
                                ui.separator();
                                ui.label(format!("Bump {}", i + 1));
                                ui.add(egui::Slider::new(&mut bump.amplitude, 0.0..=1.0).text("amplitude"));
                                ui.add(egui::Slider::new(&mut bump.concentration, 0.5..=20.0).text("concentration"));
                                ui.add(egui::Slider::new(&mut bump.theta, 0.0..=std::f32::consts::PI).text("theta"));
                                ui.add(egui::Slider::new(&mut bump.phi, 0.0..=std::f32::consts::TAU).text("phi"));
                                let max_lean = bump.max_lean();
                                let max_bend = bump.max_bend(self.blob_params.r_min);
                                bump.lean_amount = bump.lean_amount.min(max_lean);
                                bump.bend_amount = bump.bend_amount.min(max_bend);
                                ui.add(egui::Slider::new(&mut bump.lean_amount, 0.0..=max_lean).text("lean"));
                                ui.add(egui::Slider::new(&mut bump.lean_direction_phi, 0.0..=std::f32::consts::TAU).text("lean direction"));
                                ui.add(egui::Slider::new(&mut bump.bend_amount, 0.0..=max_bend).text("bend"));
                            }
                        });
                    });

                    gpu.egui_state.handle_platform_output(window, full_output.platform_output);

                    conserve_volume(&mut self.blob_params);
                    let uniforms = self.blob_params.to_uniforms(gpu.num_particles);
                    gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

                    let output = gpu.surface.get_current_texture().unwrap();
                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

                    let mut encoder = gpu.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor { label: None }
                    );

                    // Compute pass
                    {
                        let mut compute_pass = encoder.begin_compute_pass(
                            &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }
                        );
                        compute_pass.set_pipeline(&gpu.compute_pipeline);
                        compute_pass.set_bind_group(0, &gpu.compute_bind_group, &[]);
                        compute_pass.dispatch_workgroups((gpu.num_particles + 63) / 64, 1, 1);
                    }

                    // Blob render pass
                    {
                        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.1, a: 1.0 }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        render_pass.set_pipeline(&gpu.render_pipeline);
                        render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
                        render_pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                        render_pass.draw(0..6, 0..gpu.num_particles);
                    }

                    gpu.queue.submit(std::iter::once(encoder.finish()));

                    // egui in separate encoder
                    let mut egui_encoder = gpu.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor { label: None }
                    );

                    let tris = gpu.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                    for delta in &full_output.textures_delta.set {
                        gpu.egui_renderer.update_texture(&gpu.device, &gpu.queue, delta.0, &delta.1);
                    }
                    let screen_desc = egui_wgpu::ScreenDescriptor {
                        size_in_pixels: [window.inner_size().width, window.inner_size().height],
                        pixels_per_point: full_output.pixels_per_point,
                    };
                    gpu.egui_renderer.update_buffers(&gpu.device, &gpu.queue, &mut egui_encoder, &tris, &screen_desc);
                    {
                        let mut egui_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        gpu.egui_renderer.render(&mut egui_pass.forget_lifetime(), &tris, &screen_desc);
                    }
                    for id in &full_output.textures_delta.free {
                        gpu.egui_renderer.free_texture(id);
                    }

                    gpu.queue.submit(std::iter::once(egui_encoder.finish()));
                    output.present();
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
