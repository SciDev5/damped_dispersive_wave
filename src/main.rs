use std::{
    array,
    collections::VecDeque,
    sync::{Arc, Mutex},
    thread::sleep,
    time::Duration,
    usize,
};

use cpal::{
    traits::{DeviceTrait, HostTrait},
    SampleRate,
};
use num_complex::{Complex, Complex32, ComplexFloat};

struct OscRaw {
    medium: Box<[Complex32]>,
    state: Box<[Complex32]>,
}

struct NormalMode {
    omega: f32,
    k: f32,
    damp: f32,
}
impl OscRaw {
    fn gen(sample_rate: f32, medium: &[NormalMode], state: &[Complex32]) -> Self {
        assert_eq!(medium.len(), state.len());
        Self {
            state: state.iter().copied().collect(),
            medium: medium
                .iter()
                .map(|NormalMode { omega, damp, .. }| {
                    (Complex32::new(-*damp, *omega) / sample_rate).exp()
                })
                .collect(),
        }
    }
    fn tick(&mut self) {
        for i in 0..self.medium.len() {
            self.state[i] *= self.medium[i];
        }
    }
    fn sample_x0(&self) -> (f32, f32) {
        let s: Complex32 = self.state.iter().sum();
        (s.re() * 0.05, s.im() * 0.05)
        // self.state.iter().map(|v| v.re()).sum()
        // self.state.iter().sum::<Complex32>().norm_sqr()
    }
}

// struct Bufs {
//     sample_buf: VecDeque<f32>,
//     n_ready: usize,
// }
// impl Bufs {
//     fn alloc_zeros_leading(&mut self, n: usize) {
//         self.sample_buf.extend(std::iter::repeat(0.0).take(n));
//     }
//     fn take_n<'a>(&'a mut self, n: usize) -> impl Iterator<Item = f32> + 'a {
//         self.sample_buf.drain(..n)
//     }
// }

fn gen_normal_modes<const N: usize>(
    k_fundamental: f32,
    dispersion_relation: impl Fn(f32) -> f32,
    damping_relation: impl Fn(f32) -> f32,
) -> [NormalMode; N] {
    array::from_fn(|i| {
        let k = k_fundamental * (i as i32 - (N as i32 / 2)) as f32;
        NormalMode {
            k,
            omega: dispersion_relation(k),
            damp: damping_relation(k),
        }
    })
}
fn gen_initial_state<const N: usize>(mut buf: [Complex32; N]) -> [Complex32; N] {
    let fft = rustfft::FftPlanner::new().plan_fft_forward(N);
    fft.process(&mut buf);
    let integral = buf.iter().map(|v| v.norm_sqr()).sum::<f32>();
    buf.iter_mut().for_each(|v| *v /= integral.sqrt());
    return buf;
}

fn main() {
    let device = cpal::default_host()
        .default_output_device()
        .expect("no output device available");
    let config = device
        .supported_output_configs()
        .unwrap()
        .filter(|v| v.channels() == 2)
        .next()
        .expect("no mono configs")
        .with_sample_rate(SampleRate(384000 / 64))
        // .with_max_sample_rate()
        .config();

    dbg!(config.sample_rate.0);

    const N: usize = 2048;
    // const K: usize = 128;
    const K: usize = 2;
    let o = Arc::new(Mutex::new(OscRaw::gen(1.0, &[], &[])));

    let mut c = 0;

    let stream = device.build_output_stream(
        &config,
        {
            let o = o.clone();
            let mut true_buf = VecDeque::new();
            let mut prev = { o.lock().unwrap().sample_x0() };
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut o = o.lock().unwrap();
                // react to stream events and read or write stream data here.

                for sample in data.iter_mut() {
                    if true_buf.is_empty() {
                        o.tick();
                        let v = o.sample_x0();
                        for j in 0..K / 2 {
                            let k = j as f32 / ((K / 2) as f32);
                            true_buf.push_front(prev.0 * k + (1.0 - k) * v.0);
                            true_buf.push_front(prev.1 * k + (1.0 - k) * v.1);
                        }
                        prev = v;
                    }

                    *sample = true_buf.pop_back().unwrap();
                }

                if c % (1000000000 * config.sample_rate.0 as usize) < data.len() {
                    // let p = ((c / (config.sample_rate.0 as usize)) as f32 * -0.01).exp();
                    let m = (c / (config.sample_rate.0 as usize)) as f32 * 100.0 + 100.0;
                    let medium = gen_normal_modes::<N>(
                        1.0,
                        |k| {
                            // 1.0 / (k / 20.0).cos() * 200000.0
                            // k.abs().powf(2.1) / 1.0
                            // k.powf(p) / 1.0
                            (k * k + m * m).sqrt()
                            // k / (k * k + m).sqrt()
                        },
                        |k| 0.0, // 0.00001 * (k * k + m * m).sqrt(), //0.1 * k.ln(),
                    );
                    let state_init: [Complex32; N] = array::from_fn(|i| {
                        // if i == N * 7 / 8 {
                        //     0.01
                        // } else {
                        //     0.0
                        // }
                        let x = i as f32 / N as f32;
                        // (x*20.0)
                        let x0 = ((x - 0.1) + 1.0) % 1.0 - 0.5;
                        let x1 = ((x - 0.30) + 1.0) % 1.0 - 0.5;
                        Complex::new((-(100.0 * x0).powi(2)).exp(), 0.0) * 0.1
                            + Complex::new((-(200.0 * x1).powi(2)).exp(), 0.0) * 0.1
                    });
                    // let state_init = gen_initial_state::<N>(array::from_fn(|i| {
                    //     // if i == N * 7 / 8 {
                    //     //     0.01
                    //     // } else {
                    //     //     0.0
                    //     // }
                    //     let x = i as f32 / N as f32;
                    //     // (x*20.0)
                    //     let x0 = ((x + 0.3) + 0.5) % 1.0 - 0.5;
                    //     Complex::from_polar((-(100.0 * x0).powi(2)).exp(), x * -1000.0)
                    // }));
                    *o = OscRaw::gen(
                        (config.sample_rate.0 * 64) as f32 / (1.0e+2 * std::f32::consts::TAU),
                        &medium,
                        &state_init,
                    );
                }
                c += data.len();
            }
        },
        move |err| {
            // react to errors here.
        },
        None, // None=blocking, Some(Duration)=timeout
    );

    loop {
        // sleep(Duration::from_millis(40));
        sleep(Duration::from_millis(20));

        let mut buf = { o.lock().unwrap().state.clone() };

        let fft = rustfft::FftPlanner::new().plan_fft_forward(N);
        fft.process(&mut buf);

        // dbg!(buf.iter().map(|v| v.norm_sqr().ln()).collect::<Vec<_>>());

        let mut s = String::new();
        for y in 0..40 {
            let y = 35 - y;
            const W: usize = 256;
            for x in 0..W {
                // if buf[x * 2].norm_sqr().ln() > y as f32 * (1.0 / 20.0) - 2.1 {
                let r = buf[x * (N / W)].re() > y as f32 * (6.0 / 40.0);
                let a = buf[x * (N / W)].abs() > y as f32 * (6.0 / 40.0);
                if r {
                    // s += ".";
                    s += "#";
                } else if a {
                    s += ":";
                } else {
                    s += " ";
                }
            }
            s += "\n";
        }
        println!("WAVE:\n{}", s);
        // break;
    }
}
