use log;
use qip::pipeline::Representation;
use qip::state_ops::{from_tuples, UnitaryOp};
use qip::{Complex, QuantumState};
use qip_gpu;

fn main() {
    env_logger::init();
    let state = pollster::block_on(qip_gpu::GPUBackend::<1>::new_async_fullstate(
        2,
        vec![
            Complex { re: 2.0, im: 3.0 },
            Complex { re: 5.0, im: 7.0 },
            Complex { re: 11.0, im: 13.0 },
            Complex { re: 17.0, im: 19.0 },
        ],
    ));
    println!("{:?}", state.into_state(Representation::BigEndian));

    let mut state = pollster::block_on(qip_gpu::GPUBackend::<1>::new_async_fullstate(
        2,
        vec![
            Complex { re: 2.0, im: 3.0 },
            Complex { re: 5.0, im: 7.0 },
            Complex { re: 11.0, im: 13.0 },
            Complex { re: 17.0, im: 19.0 },
        ],
    ));
    state.apply_op(&UnitaryOp::Matrix(
        vec![0],
        from_tuples(&[(0., 0.), (1., 0.), (1., 0.), (0., 0.)]),
    ));
    println!("{:?}", state.into_state(Representation::BigEndian));
}
