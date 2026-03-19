use anyhow::Result;
use wgpu::util::DeviceExt;

/// Create a GPU device and queue.
pub fn create_gpu() -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
    )?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    Ok((device, queue))
}

/// A binding slot for a compute shader dispatch.
pub enum Binding<'a> {
    Uniform(&'a [u8]),
    StorageRead(&'a [u8]),
    StorageReadWrite(&'a [u8]),
}

/// Run a compute shader. Returns data from writable bindings in binding order.
pub fn run_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &str,
    entry_point: &str,
    bindings: &[Binding],
    workgroups: [u32; 3],
) -> Result<Vec<Vec<u8>>> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    let mut buffers = Vec::new();
    let mut writable_indices = Vec::new();

    for (i, binding) in bindings.iter().enumerate() {
        let (data, usage) = match binding {
            Binding::Uniform(d) => {
                (*d, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)
            }
            Binding::StorageRead(d) => {
                (*d, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST)
            }
            Binding::StorageReadWrite(d) => {
                writable_indices.push(i);
                (
                    *d,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                )
            }
        };
        buffers.push(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage,
        }));
    }

    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &entries,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }

    let mut staging_buffers = Vec::new();
    for &idx in &writable_indices {
        let size = buffers[idx].size();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffers[idx], 0, &staging, 0, size);
        staging_buffers.push(staging);
    }

    queue.submit([encoder.finish()]);

    let mut results = Vec::new();
    for staging in &staging_buffers {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        rx.recv().unwrap()?;
        results.push(slice.get_mapped_range().to_vec());
        staging.unmap();
    }

    Ok(results)
}
