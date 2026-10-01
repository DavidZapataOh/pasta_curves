//! Field and curve operations whose results are compared between the zkVM
//! and the portable implementation.

use pasta_curves::{
    arithmetic::CurveAffine,
    group::{
        ff::{FromUniformBytes, PrimeField},
        prime::PrimeCurveAffine,
        Curve, Group, GroupEncoding,
    },
    pallas, vesta, Fp, Fq,
};

/// A xorshift generator, so both sides draw the same inputs.
pub struct Rng(pub u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn bytes(&mut self) -> [u8; 64] {
        let mut out = [0; 64];
        for chunk in out.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes());
        }
        out
    }

    pub fn fp(&mut self) -> Fp {
        Fp::from_uniform_bytes(&self.bytes())
    }

    pub fn fq(&mut self) -> Fq {
        Fq::from_uniform_bytes(&self.bytes())
    }
}

fn field_ops<F: PrimeField>(out: &mut Vec<u8>, a: F, b: F) {
    for x in [
        a * b,
        a.square(),
        a.invert().unwrap_or(F::ZERO),
        a.sqrt().unwrap_or(F::ZERO),
        a + b,
        a - b,
        -a,
        a.pow_vartime([0x1234_5678_9abc_def0, 7]),
    ] {
        out.extend_from_slice(x.to_repr().as_ref());
    }
    out.push(u8::from(bool::from(F::ZERO.invert().is_none())));
}

#[allow(clippy::eq_op)]
fn curve_ops<G>(out: &mut Vec<u8>, p: G, q: G, s: G::Scalar)
where
    G: Group + Curve + GroupEncoding,
    G::AffineRepr: GroupEncoding + Into<G> + Copy,
{
    let id = G::identity();
    for x in [
        p + q,
        p + q.to_affine().into(),
        p + p,
        p.double(),
        p - p,
        p + (-p),
        id + p,
        p + id,
        id.double(),
        p * s,
        (p + q) - q,
    ] {
        out.extend_from_slice(x.to_bytes().as_ref());
    }
    out.extend_from_slice(p.to_affine().to_bytes().as_ref());
    out.extend_from_slice(q.to_affine().to_bytes().as_ref());
}

/// Runs `n` rounds of every operation on inputs drawn from `seed` and returns
/// the encoded results.
pub fn run(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut out = vec![];
    for _ in 0..n {
        field_ops(&mut out, rng.fp(), rng.fp());
        field_ops(&mut out, rng.fq(), rng.fq());

        let (s, t, u) = (rng.fq(), rng.fq(), rng.fq());
        let p = pallas::Point::generator() * s;
        curve_ops(&mut out, p, pallas::Point::generator() * t, u);
        out.push(u8::from(bool::from(p.to_affine().is_on_curve())));

        let (s, t, u) = (rng.fp(), rng.fp(), rng.fp());
        curve_ops(
            &mut out,
            vesta::Point::generator() * s,
            vesta::Point::generator() * t,
            u,
        );
    }
    let points: Vec<_> = (1..=n as u64)
        .map(|i| pallas::Point::generator() * Fq::from(i))
        .chain(core::iter::once(pallas::Point::identity()))
        .collect();
    let mut affine = vec![pallas::Affine::identity(); points.len()];
    pallas::Point::batch_normalize(&points, &mut affine);
    for p in affine {
        out.extend_from_slice(&p.to_bytes());
    }
    out
}
