use core::hint::black_box;

use ops::Rng;
use pasta_curves::group::{ff::Field, ff::PrimeField, Curve, Group};
use pasta_curves::{pallas, vesta};
use risc0_zkvm::guest::env;

fn per_op(n: u32, mut f: impl FnMut()) -> u64 {
    let start = env::cycle_count();
    for _ in 0..n {
        f();
    }
    (env::cycle_count() - start) / u64::from(n)
}

fn main() {
    let (n, seed): (u32, u64) = env::read();
    if n > 0 {
        env::commit_slice(&ops::run(n as usize, seed));
        return;
    }
    let mut rng = Rng(seed);
    let (a, b) = (rng.fp(), rng.fp());
    let (c, d) = (rng.fq(), rng.fq());
    let (p, q) = (
        pallas::Point::generator() * c,
        pallas::Point::generator() * d,
    );
    let qa = q.to_affine();
    let (v, w) = (vesta::Point::generator() * a, vesta::Point::generator() * b);
    let wa = w.to_affine();
    let counts: [u64; 14] = [
        per_op(256, || {
            black_box(black_box(a) * black_box(b));
        }),
        per_op(256, || {
            black_box(black_box(a).square());
        }),
        per_op(256, || {
            black_box(black_box(a).invert());
        }),
        per_op(256, || {
            black_box(black_box(a).to_repr());
        }),
        per_op(256, || {
            black_box(black_box(c) * black_box(d));
        }),
        per_op(256, || {
            black_box(black_box(c).invert());
        }),
        per_op(256, || {
            black_box(black_box(p) + black_box(q));
        }),
        per_op(256, || {
            black_box(black_box(p) + black_box(qa));
        }),
        per_op(256, || {
            black_box(black_box(p).double());
        }),
        per_op(256, || {
            black_box(black_box(p).to_affine());
        }),
        per_op(8, || {
            black_box(black_box(p) * black_box(d));
        }),
        per_op(256, || {
            black_box(black_box(v) + black_box(w));
        }),
        per_op(256, || {
            black_box(black_box(v) + black_box(wa));
        }),
        per_op(256, || {
            black_box(black_box(v).double());
        }),
    ];
    env::commit(&counts);
}
