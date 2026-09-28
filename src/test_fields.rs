//! The two fields' constants and known answers for the tests of the assembly backends and of the
//! inversion. The moduli and the Montgomery constants are the field types' own. The other expected
//! values were computed independently with big-integer Montgomery arithmetic
//! (`a * b * 2^-256 mod p`) in Python, and a test checks each against the portable arithmetic of
//! the field types, which also supplies the reference products and inverses.

use crate::asm::Limbs;
use crate::fields::{fp, fq};

/// One field's constants, from its field type, and known answers.
pub(crate) struct Field {
    pub(crate) modulus: Limbs,
    /// `-modulus[0]^-1 mod 2^64`.
    pub(crate) inv: u64,
    /// `R = 2^256 mod p`, the Montgomery form of `1`.
    pub(crate) r: Limbs,
    /// `2R mod p`.
    pub(crate) two_r: Limbs,
    /// `3R mod p`.
    pub(crate) three_r: Limbs,
    /// `R^2 mod p`.
    pub(crate) r2: Limbs,
    /// `R^3 mod p`.
    pub(crate) r3: Limbs,
    /// `mul(R2, R3) = R^4 mod p`.
    pub(crate) r4: Limbs,
    /// `sqr_n_mul(R2, 2, R3) = R^7 mod p`.
    pub(crate) r7: Limbs,
    /// `mul(p - 1, p - 1)`.
    pub(crate) pm1_sq: Limbs,
    /// `p - 2`.
    pub(crate) pm2: Limbs,
    /// `from_mont` of the all-ones input, `(2^256 - 1) R^-1 mod p`.
    pub(crate) from_mont_ones: Limbs,
    /// `2^562 mod p`, the starting `v` of `invert`.
    pub(crate) v0: Limbs,
    /// Inputs and outputs of `invert`: `7R`, `0`, `1`, `p - 1`, and a small value, from the
    /// integer model of the algorithm.
    pub(crate) inversions: [(Limbs, Limbs); 5],
    /// The Montgomery product by the field type's portable arithmetic, independent of the backends.
    pub(crate) portable_mul: fn(&Limbs, &Limbs) -> Limbs,
    /// The Montgomery inverse by the field type's portable arithmetic, independent of the backends.
    pub(crate) portable_inverse: fn(&Limbs) -> Limbs,
}

/// The Pallas base field (`pasta_curves::Fp`).
pub(crate) const FP: Field = Field {
    modulus: fp::MODULUS.0,
    inv: fp::INV,
    r: fp::R.0,
    two_r: [
        0xcfc3a984fffffff9,
        0x1011d11bbee5303e,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ],
    three_r: [
        0x6b0ee5d0fffffff5,
        0x86f76d2b99b14bd0,
        0xfffffffffffffffe,
        0x3fffffffffffffff,
    ],
    r2: fp::R2.0,
    r3: fp::R3.0,
    r4: [
        0x1dfc65f6ad0492ae,
        0x84379b4cc10e927b,
        0x710d6cd04c692c97,
        0x21cce888a6cab566,
    ],
    r7: [
        0x7a3a1c29d1d1bd45,
        0x17023e5920bb6157,
        0x9004eaaf35c21e06,
        0x007efbf9151076fc,
    ],
    pm1_sq: [
        0xcf3f8e8753a769a9,
        0xac9fba6a4077fc57,
        0x70cb2996efc89a65,
        0x21f1c4ff1e2278d5,
    ],
    pm2: [
        0x992d30ecffffffff,
        0x224698fc094cf91b,
        0x0000000000000000,
        0x4000000000000000,
    ],
    from_mont_ones: [
        0xc9eda265ac589659,
        0x75a6de91c8d4fcc3,
        0x8f34d6691037659a,
        0x1e0e3b00e1dd872a,
    ],
    v0: [
        0x9a5f583ce5084635,
        0x4f417e233776c195,
        0x74634b1a733f7785,
        0x1c51de5ea66f0f25,
    ],
    inversions: [
        (
            [0xd83bd700ffffffe5, 0x628ddd6b04e1ba16, 0xfffffffffffffffc, 0x3fffffffffffffff],
            [0x8398bdd8b6db6db7, 0xbbc0f148939d4828, 0xdb6db6db6db6db6d, 0x2db6db6db6db6db6],
        ),
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ),
        (
            [0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x8c78ecb30000000f, 0xd7d30dbd8b0de0e7, 0x7797a99bc3c95d18, 0x096d41af7b9cb714],
        ),
        (
            [0x992d30ed00000000, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000],
            [0x0cb44439fffffff2, 0x4a738b3e7e3f1834, 0x886856643c36a2e7, 0x3692be50846348eb],
        ),
        (
            [0xfc962fc962fc9630, 0x369d0369d0369cd2, 0x0000000000000000, 0x0000000000000000],
            [0x33912c173eb52b5e, 0x8094d7a33b979988, 0x4c1c894cf5cc5f05, 0x2d05c75a616fc8d4],
        ),
    ],
    portable_mul: fp_mul,
    portable_inverse: fp_inverse,
};

/// The Vesta base field (`pasta_curves::Fq`).
pub(crate) const FQ: Field = Field {
    modulus: fq::MODULUS.0,
    inv: fq::INV,
    r: fq::R.0,
    two_r: [
        0x2a0f9218fffffff9,
        0x1011d11bbcef61f1,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ],
    three_r: [
        0xf8f3e594fffffff5,
        0x86f76d2b969cbe7a,
        0xfffffffffffffffe,
        0x3fffffffffffffff,
    ],
    r2: fq::R2.0,
    r3: fq::R3.0,
    r4: [
        0x569bba29179df5c1,
        0xf7abe57547cfa14c,
        0x8d0f36071632bdab,
        0x2c37a71489ba6088,
    ],
    r7: [
        0x56244c6793e8be1f,
        0xbe518f1c2a1b26e6,
        0xb6c73001c86b2b65,
        0x1b7d3aff1b7fd420,
    ],
    pm1_sq: [
        0x6119a3dd8e1a6f7f,
        0xc68de1279dc601eb,
        0x5790be58c050df13,
        0x1f7a89dd17647953,
    ],
    pm2: [
        0x8c46eb20ffffffff,
        0x224698fc0994a8dd,
        0x0000000000000000,
        0x4000000000000000,
    ],
    from_mont_ones: [
        0x2b2d474371e59083,
        0x5bb8b7d46bcea6f2,
        0xa86f41a73faf20ec,
        0x20857622e89b86ac,
    ],
    v0: [
        0xa3efbd8ee5083303,
        0xfbadea62cefef7a1,
        0xd6418abb493f6cf9,
        0x2aa5feb88c401333,
    ],
    inversions: [
        (
            [0x34853384ffffffe5, 0x628ddd6afd5230a2, 0xfffffffffffffffc, 0x3fffffffffffffff],
            [0x81c0fd04b6db6db7, 0xbbc0f14893a785d6, 0xdb6db6db6db6db6d, 0x2db6db6db6db6db6],
        ),
        (
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ),
        (
            [0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xfc9678ff0000000f, 0x67bb433d891a16e3, 0x7fae231004ccf590, 0x096d41af7ccfdaa9],
        ),
        (
            [0x8c46eb2100000000, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000],
            [0x8fb07221fffffff2, 0xba8b55be807a91f9, 0x8051dceffb330a6f, 0x3692be5083302556],
        ),
        (
            [0xfc962fc962fc9630, 0x369d0369d0369cd2, 0x0000000000000000, 0x0000000000000000],
            [0xe5c6fb7bddd0cf4b, 0x65ee805e3b7d0d89, 0x7562671be840d861, 0x1253ce66fd1d1868],
        ),
    ],
    portable_mul: fq_mul,
    portable_inverse: fq_inverse,
};

pub(crate) const FIELDS: [&Field; 2] = [&FP, &FQ];

pub(crate) const ONE: Limbs = [1, 0, 0, 0];
pub(crate) const ZERO: Limbs = [0, 0, 0, 0];

/// `p - 1`; `modulus[0]` is odd, so the subtraction does not borrow.
pub(crate) fn p_minus_1(f: &Field) -> Limbs {
    let mut limbs = f.modulus;
    limbs[0] -= 1;
    limbs
}

/// `a - b` as little-endian 256-bit integers, for `b ≤ a`.
pub(crate) fn sub_limbs(a: &Limbs, b: &Limbs) -> Limbs {
    let mut out = ZERO;
    let mut borrow = false;
    for i in 0..4 {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(u64::from(borrow));
        out[i] = d;
        borrow = b1 | b2;
    }
    assert!(!borrow, "{b:x?} exceeds {a:x?}");
    out
}

/// The Montgomery product of the Montgomery residues `x` and `y` by `Fp`'s portable arithmetic.
fn fp_mul(x: &Limbs, y: &Limbs) -> Limbs {
    (crate::fields::Fp(*x) * crate::fields::Fp(*y)).0
}

/// The Montgomery product of the Montgomery residues `x` and `y` by `Fq`'s portable arithmetic.
fn fq_mul(x: &Limbs, y: &Limbs) -> Limbs {
    (crate::fields::Fq(*x) * crate::fields::Fq(*y)).0
}

/// The Montgomery inverse of the Montgomery residue `x` by `Fp`'s portable arithmetic.
fn fp_inverse(x: &Limbs) -> Limbs {
    ff::Field::invert(&crate::fields::Fp(*x)).unwrap().0
}

/// The Montgomery inverse of the Montgomery residue `x` by `Fq`'s portable arithmetic.
fn fq_inverse(x: &Limbs) -> Limbs {
    ff::Field::invert(&crate::fields::Fq(*x)).unwrap().0
}

/// The canonical integer of a field element, from its little-endian `to_repr`.
pub(crate) fn canonical_integer<F: ff::PrimeField<Repr = [u8; 32]>>(element: F) -> Limbs {
    let repr = element.to_repr();
    core::array::from_fn(|i| u64::from_le_bytes(repr[8 * i..8 * (i + 1)].try_into().unwrap()))
}

/// The known answers above that are not the field types' own constants, recomputed by the portable
/// arithmetic of the field types, on the same limbs.
#[test]
fn known_answers_match_the_portable_arithmetic() {
    macro_rules! check {
        ($field:ident, $f:expr) => {{
            use crate::fields::$field;
            let f = $f;
            let el = |limbs: Limbs| $field(limbs);
            let pm1 = el(p_minus_1(f));
            assert_eq!((el(f.r) + el(f.r)).0, f.two_r);
            assert_eq!((el(f.two_r) + el(f.r)).0, f.three_r);
            assert_eq!((pm1 + pm1).0, f.pm2);
            assert_eq!((el(f.r2) * el(f.r3)).0, f.r4);
            assert_eq!((el(f.r2).square().square() * el(f.r3)).0, f.r7);
            assert_eq!((pm1 * pm1).0, f.pm1_sq);
            // The element whose Montgomery form is the all-ones input (mod `p`), as in
            // `entry_points_match_the_portable_arithmetic`; its canonical integer is `to_repr`.
            let ones = $field::from_u512([u64::MAX, u64::MAX, u64::MAX, u64::MAX, 0, 0, 0, 0]);
            assert_eq!(canonical_integer(ones * el(ONE)), f.from_mont_ones);
            // `v0` is the integer `2^562 mod p`, and each inversion's output is the Montgomery
            // form of the inverse of the element whose Montgomery form is its input.
            let two = $field::from_u512([2, 0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(canonical_integer(ff::Field::pow_vartime(&two, [562])), f.v0);
            for (x, z) in &f.inversions {
                let expected = if *x == ZERO { ZERO } else { (f.portable_inverse)(x) };
                assert_eq!(expected, *z, "{x:x?}");
            }
        }};
    }
    check!(Fp, &FP);
    check!(Fq, &FQ);
}
