struct Complex {
    real: f32;
    imag: f32;
};

[[block]]
struct State {
  state : [[stride(8)]] array<Complex>;
};

[[block]]
struct MatrixIndices {
    indices : [[stride(4)]] array<u32>;
};


[[group(0), binding(0)]]
var<storage, read> src : State;
[[group(0), binding(1)]]
var<storage, read_write> dst : State;
[[group(0), binding(2)]]
var<storage, read> mat_indices : MatrixIndices;


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let total = arrayLength(&src.state);
    let matn = mat_indices.indices[0];
    let index = global_id.x;
    if (index >= total) {
        return;
    }

    var real_acc: f32 = 0.0;
    var imag_acc: f32 = 0.0;

    dst.state[index].real = real_acc;
    dst.state[index].imag = imag_acc;
}