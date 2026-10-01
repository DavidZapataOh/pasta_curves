use methods::{CHECK_ELF, CHECK_ID};
use risc0_zkvm::{default_executor, default_prover, ExecutorEnv};

fn execute(n: u32, seed: u64) -> risc0_zkvm::Journal {
    let env = ExecutorEnv::builder()
        .write(&(n, seed))
        .unwrap()
        .build()
        .unwrap();
    default_executor().execute(env, CHECK_ELF).unwrap().journal
}

fn costs() -> [u64; 14] {
    execute(0, 7).decode().unwrap()
}

/// Asserts that every measured cost is within its budget, listing the ones that are not.
fn within(costs: &[u64], budgets: &[(&str, usize, u64)]) {
    let over: Vec<_> = budgets
        .iter()
        .filter(|(_, i, budget)| costs[*i] > *budget)
        .map(|(op, i, budget)| format!("{op}: {} > {budget}", costs[*i]))
        .collect();
    assert!(over.is_empty(), "{over:?}");
}

#[test]
fn results_match_the_portable_implementation() {
    assert_eq!(execute(64, 42).bytes, ops::run(64, 42));
}

/// Proves for real, so the precompiles are shown to prove on the Pasta moduli
/// and not only to execute (about 18 minutes on 16 cores).
#[test]
#[ignore]
fn proves() {
    let env = ExecutorEnv::builder()
        .write(&(1u32, 5u64))
        .unwrap()
        .build()
        .unwrap();
    let receipt = default_prover().prove(env, CHECK_ELF).unwrap().receipt;
    receipt.verify(CHECK_ID).unwrap();
    assert_eq!(receipt.journal.bytes, ops::run(1, 5));
}

#[test]
fn field_operations_are_accelerated() {
    within(
        &costs(),
        &[
            ("Fp mul", 0, 250),
            ("Fp square", 1, 250),
            ("Fp invert", 2, 400),
            ("Fp to_repr", 3, 150),
            ("Fq mul", 4, 250),
            ("Fq invert", 5, 400),
        ],
    );
}

#[test]
fn curve_operations_are_accelerated() {
    within(
        &costs(),
        &[
            ("Pallas Point + Point", 6, 2_000),
            ("Pallas Point + Affine", 7, 2_000),
            ("Pallas double", 8, 1_200),
            ("Pallas to_affine", 9, 1_500),
            ("Pallas scalar mul", 10, 800_000),
            ("Vesta Point + Point", 11, 2_000),
            ("Vesta Point + Affine", 12, 2_000),
            ("Vesta double", 13, 1_200),
        ],
    );
}
