use bytemuck;
use qip;
use qip::measurement_ops::MeasuredCondition;
use qip::pipeline::{RegisterInitialState, Representation};
use qip::state_ops::UnitaryOp;
use qip::Complex;
use std::borrow::Cow;
use std::cmp::max;
use wgpu;
use wgpu::util::DeviceExt;

pub struct GPUBackend<const N: usize> {
    n: u64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    state_buffer: Option<wgpu::Buffer>,
    arena_buffer: Option<wgpu::Buffer>,
    index_buffer: wgpu::Buffer,
    matmul: MatmulPipeline,
    measure: MeasurementPipeline,
}

struct MatmulPipeline {
    data_buffer: wgpu::Buffer,
    matmul_pipeline: wgpu::ComputePipeline,
    active_bindgroup: Option<wgpu::BindGroup>,
    swap_bindgroup: Option<wgpu::BindGroup>,
}

struct MeasurementPipeline {
    measure_pipeline: wgpu::ComputePipeline,
    active_bindgroup: Option<wgpu::BindGroup>,
    swap_bindgroup: Option<wgpu::BindGroup>,
}

impl MatmulPipeline {
    fn swap(&mut self) {
        swap_options(&mut self.active_bindgroup, &mut self.swap_bindgroup)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CComplex {
    real: f32,
    imag: f32,
}

impl<const N: usize> GPUBackend<N> {
    fn swap(&mut self) {
        swap_options(&mut self.state_buffer, &mut self.arena_buffer);
        self.matmul.swap();
    }

    pub async fn new_async_fullstate(n: u64, state: Vec<Complex<f32>>) -> Self {
        let nn = 1 << n;
        let cvec: Vec<_> = state
            .into_iter()
            .map(|c| CComplex {
                real: c.re,
                imag: c.im,
            })
            .collect();

        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let matmul_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../matmul.wgsl"))),
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (nn * 2 * std::mem::size_of::<f32>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (nn * 2 * std::mem::size_of::<f32>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            // ty: wgpu::BufferBindingType::Uniform,
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((N + 1) * std::mem::size_of::<u32>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            // ty: wgpu::BufferBindingType::Uniform,
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((2 << N) * 2 * std::mem::size_of::<f32>()) as _,
                            ),
                        },
                        count: None,
                    },
                ],
                label: None,
            });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matmul"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &matmul_shader,
            entry_point: "main",
        });

        let measure_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../measure.wgsl"))),
        });

        let measure_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (nn * 2 * std::mem::size_of::<f32>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (nn * 2 * std::mem::size_of::<f32>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            // ty: wgpu::BufferBindingType::Uniform,
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((N + 1) * std::mem::size_of::<u32>()) as _,
                            ),
                        },
                        count: None,
                    },
                ],
                label: None,
            });
        let measure_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("measure"),
                bind_group_layouts: &[&measure_bind_group_layout],
                push_constant_ranges: &[],
            });
        let measure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("measure pipeline"),
            layout: Some(&measure_pipeline_layout),
            module: &measure_shader,
            entry_point: "main",
        });

        let mut state_buffers = Vec::default();
        for i in 0..2 {
            state_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Buffer {}", i)),
                    contents: bytemuck::cast_slice(&cvec),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                }),
            );
        }

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&vec![0_u32; N + 1]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Data Buffer"),
            contents: bytemuck::cast_slice(&vec![0_f32; 2 * 2 << N]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // create two bind groups, one for each buffer as the src
        // where the alternate buffer is used as the dst

        let mut mm_bind_groups = Vec::default();
        for i in 0..2 {
            mm_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: state_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: state_buffers[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: data_buffer.as_entire_binding(),
                    },
                ],
                label: None,
            }));
        }

        let mut meas_bind_groups = Vec::default();
        for i in 0..2 {
            meas_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &measure_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: state_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: state_buffers[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: index_buffer.as_entire_binding(),
                    },
                ],
                label: None,
            }));
        }

        let arena_buffer = state_buffers.pop();
        let state_buffer = state_buffers.pop();
        let swap_mm_bindgroup = mm_bind_groups.pop();
        let active_mm_bindgroup = mm_bind_groups.pop();
        let swap_meas_bindgroup = meas_bind_groups.pop();
        let active_meas_bindgroup = meas_bind_groups.pop();

        Self {
            n,
            device,
            queue,
            state_buffer,
            arena_buffer,
            index_buffer,
            matmul: MatmulPipeline {
                data_buffer,
                matmul_pipeline,
                active_bindgroup: active_mm_bindgroup,
                swap_bindgroup: swap_mm_bindgroup,
            },
            measure: MeasurementPipeline {
                measure_pipeline,
                active_bindgroup: active_meas_bindgroup,
                swap_bindgroup: swap_meas_bindgroup,
            },
        }
    }

    pub async fn new_async(n: u64, states: &[RegisterInitialState<f32>]) -> Self {
        let max_init_n = states
            .iter()
            .map(|(indices, _)| indices)
            .cloned()
            .flatten()
            .max()
            .map(|m| m + 1);

        let n = max_init_n.map_or(n, |m| max(n, m));
        let nn = 1 << n;

        let mut cvec: Vec<Complex<f32>> = vec![Complex::default(); nn];
        qip::pipeline::fill_buffer_initial_state(n, states, &mut cvec);
        Self::new_async_fullstate(n, cvec).await
    }
}

impl<const N: usize> qip::pipeline::QuantumState<f32> for GPUBackend<N> {
    fn new(n: u64) -> Self {
        Self::new_from_initial_states(n, &[])
    }

    fn new_from_initial_states(n: u64, states: &[RegisterInitialState<f32>]) -> Self {
        pollster::block_on(GPUBackend::new_async(n, states))
    }

    fn n(&self) -> u64 {
        self.n
    }

    fn apply_op_with_name(&mut self, name: Option<&str>, op: &UnitaryOp) {
        match op {
            UnitaryOp::Matrix(indices, data) => {
                let cdata: Vec<_> = data
                    .iter()
                    .map(|c| CComplex {
                        real: c.re as f32,
                        imag: c.im as f32,
                    })
                    .collect();
                let indices: Vec<_> = indices.iter().map(|i| *i as u32).collect();
                self.queue.write_buffer(
                    &self.index_buffer,
                    0 as _,
                    bytemuck::cast_slice(&[indices.len()]),
                );
                self.queue.write_buffer(
                    &self.index_buffer,
                    std::mem::size_of::<u32>() as _,
                    bytemuck::cast_slice(&indices),
                );
                self.queue.write_buffer(
                    &self.matmul.data_buffer,
                    0 as _,
                    bytemuck::cast_slice(&cdata),
                );

                // get command encoder
                let mut command_encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                // compute pass
                command_encoder.push_debug_group(&format!("compute: {}", name.unwrap_or("None")));

                {
                    let mut cpass = command_encoder
                        .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                    cpass.set_pipeline(&self.matmul.matmul_pipeline);
                    cpass.set_bind_group(0, self.matmul.active_bindgroup.as_ref().unwrap(), &[]);
                    cpass.dispatch(1 << self.n, 1, 1);
                }
                // Swap all buffers and bindings
                self.swap();

                command_encoder.pop_debug_group();

                self.queue.submit(Some(command_encoder.finish()));
            }
            UnitaryOp::SparseMatrix(_, _) => {
                unimplemented!()
            }
            UnitaryOp::Swap(_, _) => {
                unimplemented!()
            }
            UnitaryOp::Control(_, _, _) => {
                unimplemented!()
            }
            UnitaryOp::Function(_, _, _) => {
                unimplemented!()
            }
        }
    }

    fn measure(
        &mut self,
        indices: &[u64],
        measured: Option<MeasuredCondition<f32>>,
        angle: f64,
    ) -> (u64, f32) {
        todo!()
    }

    fn soft_measure(&mut self, indices: &[u64], measured: Option<u64>, angle: f64) -> (u64, f32) {
        todo!()
    }

    fn state_magnitude(&self) -> f32 {
        todo!()
    }

    fn stochastic_measure(&mut self, indices: &[u64], angle: f64) -> Vec<f32> {
        let indices: Vec<_> = indices.iter().map(|i| *i as u32).collect();
        self.queue.write_buffer(
            &self.index_buffer,
            0 as _,
            bytemuck::cast_slice(&[indices.len()]),
        );
        self.queue.write_buffer(
            &self.index_buffer,
            std::mem::size_of::<u32>() as _,
            bytemuck::cast_slice(&indices),
        );

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((1 << indices.len()) * 2 * std::mem::size_of::<f32>()) as _,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // compute pass
        command_encoder.push_debug_group(&format!("stochastic measure"));

        {
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.measure.measure_pipeline);
            cpass.set_bind_group(0, self.measure.active_bindgroup.as_ref().unwrap(), &[]);
            cpass.dispatch(1 << indices.len(), 1, 1);
        }

        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));

        // ======= Now read from buffer =======

        let buf = self.arena_buffer.as_ref().unwrap();

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        command_encoder.copy_buffer_to_buffer(
            &buf,
            0,
            &staging_buffer,
            0,
            ((1 << indices.len()) * 2 * std::mem::size_of::<f32>()) as _,
        );
        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        // Gets the future representing when `staging_buffer` can be read from
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.device.poll(wgpu::Maintain::Wait);

        // Awaits until `buffer_future` can be read from
        if let Ok(()) = pollster::block_on(buffer_future) {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result: Vec<f32> = bytemuck::cast_slice(&data)
                .into_iter()
                .map(|c: &CComplex| c.real)
                .collect();

            // Returns data from buffer
            result
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    fn into_state(self, order: Representation) -> Vec<Complex<f32>> {
        let buf = self.state_buffer.as_ref().unwrap();

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((1 << self.n) * 2 * std::mem::size_of::<f32>()) as _,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // get command encoder
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        command_encoder.copy_buffer_to_buffer(
            &buf,
            0,
            &staging_buffer,
            0,
            ((1 << self.n) * 2 * std::mem::size_of::<f32>()) as _,
        );
        command_encoder.pop_debug_group();

        self.queue.submit(Some(command_encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        // Gets the future representing when `staging_buffer` can be read from
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.device.poll(wgpu::Maintain::Wait);

        // Awaits until `buffer_future` can be read from
        if let Ok(()) = pollster::block_on(buffer_future) {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result: Vec<Complex<f32>> = bytemuck::cast_slice(&data)
                .into_iter()
                .map(|c: &CComplex| Complex {
                    re: c.real,
                    im: c.imag,
                })
                .collect();

            // // // With the current interface, we have to make sure all mapped views are
            // // // dropped before we unmap the buffer.
            // drop(data);
            // buf.unmap(); // Unmaps buffer from memory

            // Returns data from buffer
            result
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}

fn swap_options<T>(a: &mut Option<T>, b: &mut Option<T>) {
    let ta = a.take();
    let tb = b.take();
    *a = tb;
    *b = ta;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
