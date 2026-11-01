use super::*;

fn scan(source: &str) -> Sites {
    let mut sites = Sites {
        file: "corpus_sample.rs".into(),
        ..Sites::default()
    };
    sites.visit_file(&syn::parse_file(source).expect("test source"));
    sites
}

#[test]
fn producers_are_registered_by_literal_name() {
    let sites =
        scan(r#"#[tokio::test] async fn cell() { crate::goldens::golden_effects("a", &log); }"#);
    assert!(sites.failures.is_empty());
    assert_eq!(sites.agent["a"], ["corpus_sample.rs::cell"]);
    for source in [
        r#"fn cell() { crate::goldens::golden_effects(name, &log); }"#,
        r#"fn cell() { golden_effects("a", &log); }"#,
    ] {
        assert!(!scan(source).failures.is_empty(), "accepted {source}");
    }
}

#[test]
fn ignored_rows_are_not_producers() {
    let sites = scan(
        r#"
        golden_matrix! { wrapper: wrapper, wire: wire, run: run, oracle: crate::goldens::golden_effects;
            #[tokio::test] a: ("recording", CELL, "a");
            #[tokio::test] #[ignore = "unrecorded"] b: ("absent", CELL, "b");
        }
    "#,
    );
    assert!(sites.failures.is_empty());
    assert_eq!(sites.agent.len(), 1);
    assert!(sites.agent.contains_key("a"));
}

#[test]
fn missing_or_duplicate_producers_fail() {
    let producers = BTreeMap::from([
        (
            "duplicate".into(),
            vec!["first.rs".into(), "second.rs".into()],
        ),
        ("absent".into(), vec!["third.rs".into()]),
    ]);
    let failures = pairing_problems(
        &["duplicate".into(), "orphan".into()],
        &producers,
        &BTreeSet::new(),
        &[],
    );
    assert_eq!(failures.len(), 3, "{failures:?}");
}

#[test]
fn a_golden_checked_in_test_needs_one_producer_and_no_file() {
    let producers = BTreeMap::from([
        ("checked".into(), vec!["first.rs".into()]),
        ("twice".into(), vec!["second.rs".into(), "third.rs".into()]),
        ("committed".into(), vec!["fourth.rs".into()]),
    ]);
    let in_test = BTreeSet::from([
        "checked".to_owned(),
        "twice".to_owned(),
        "committed".to_owned(),
    ]);
    let failures = pairing_problems(&["committed".into()], &producers, &in_test, &[]);
    assert_eq!(failures.len(), 2, "{failures:?}");
    assert!(failures.iter().any(|failure| failure.contains("`twice`")));
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("`committed` is committed"))
    );
}

fn historical_reader_source() -> syn::File {
    syn::parse_file(
        r#"
        const GEMINI_CUSTOM_AT_OUTCOME: Program = Program {
            fixture: "gemini_breadth_custom_at_outcome",
            ..CUSTOM_AT_OUTCOME
        };
        both_interpreters! { gemini_custom_at_outcome: GEMINI_CUSTOM_AT_OUTCOME, }
    "#,
    )
    .expect("reader source")
}

#[test]
fn historical_file_has_its_named_readers_instead_of_a_producer() {
    let golden = &HISTORICAL_GOLDENS[0];
    let fixtures = [golden.label().expect("exact path").to_owned()];
    assert!(reader_problems(golden, &historical_reader_source()).is_empty());
    assert!(
        pairing_problems(
            &fixtures,
            &Producers::new(),
            &BTreeSet::new(),
            &HISTORICAL_GOLDENS[..1]
        )
        .is_empty()
    );
}

#[test]
fn a_listed_historical_file_cannot_disappear() {
    let failures = pairing_problems(
        &[],
        &Producers::new(),
        &BTreeSet::new(),
        &HISTORICAL_GOLDENS[..1],
    );
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains(HISTORICAL_GOLDENS[0].path));
    assert!(failures[0].contains("frozen reader-owned file is missing"));
}

#[test]
fn an_unlisted_orphan_still_needs_a_producer() {
    let fixtures = [
        HISTORICAL_GOLDENS[0]
            .label()
            .expect("exact path")
            .to_owned(),
        "unlisted_orphan".into(),
    ];
    let failures = pairing_problems(
        &fixtures,
        &Producers::new(),
        &BTreeSet::new(),
        &HISTORICAL_GOLDENS[..1],
    );
    assert_eq!(failures, ["golden `unlisted_orphan` has no producer"]);
}

#[test]
fn a_historical_reader_must_register_the_exact_program_and_fixture() {
    for source in [
        r#"
            // gemini_breadth_custom_at_outcome
            const UNUSED: &str = "gemini_breadth_custom_at_outcome";
            const GEMINI_CUSTOM_AT_OUTCOME: Program = Program { fixture: "another", ..CUSTOM_AT_OUTCOME };
            both_interpreters! { gemini_custom_at_outcome: GEMINI_CUSTOM_AT_OUTCOME, }
        "#,
        r#"
            const GEMINI_CUSTOM_AT_OUTCOME: Program = Program { #[cfg(any())] fixture: "gemini_breadth_custom_at_outcome", ..CUSTOM_AT_OUTCOME };
            both_interpreters! { gemini_custom_at_outcome: GEMINI_CUSTOM_AT_OUTCOME, }
        "#,
        r#"
            const GEMINI_CUSTOM_AT_OUTCOME: Program = Program { fixture: "gemini_breadth_custom_at_outcome", ..CUSTOM_AT_OUTCOME };
            both_interpreters! { gemini_custom_at_outcome: ANOTHER_PROGRAM, }
        "#,
        r#"
            const GEMINI_CUSTOM_AT_OUTCOME: Program = Program { fixture: "gemini_breadth_custom_at_outcome", ..CUSTOM_AT_OUTCOME };
            fn unused() { both_interpreters! { gemini_custom_at_outcome: GEMINI_CUSTOM_AT_OUTCOME, } }
        "#,
        r#"
            #[cfg(any())]
            const GEMINI_CUSTOM_AT_OUTCOME: Program = Program { fixture: "gemini_breadth_custom_at_outcome", ..CUSTOM_AT_OUTCOME };
            both_interpreters! { gemini_custom_at_outcome: GEMINI_CUSTOM_AT_OUTCOME, }
        "#,
        r#"
            const GEMINI_CUSTOM_AT_OUTCOME: Program = Program { fixture: "gemini_breadth_custom_at_outcome", ..CUSTOM_AT_OUTCOME };
            #[cfg(any())]
            both_interpreters! { gemini_custom_at_outcome: GEMINI_CUSTOM_AT_OUTCOME, }
        "#,
    ] {
        let syntax = syn::parse_file(source).expect("reader source");
        assert!(
            !reader_problems(&HISTORICAL_GOLDENS[0], &syntax).is_empty(),
            "accepted {source}"
        );
    }
}

#[test]
fn historical_reader_modules_must_be_included_by_the_verification_target() {
    for source in [
        r#"#[path = "../corpus_breadth.rs"] mod corpus_breadth;"#,
        r#"#[path = "../wrong.rs"] mod corpus_breadth;"#,
        r#"#[cfg(any())] #[path = "../corpus_breadth.rs"] mod corpus_breadth;"#,
    ] {
        let syntax = syn::parse_file(source).expect("module source");
        assert_eq!(
            module_reads(&syntax, "corpus_breadth", "../corpus_breadth.rs", false),
            source == r#"#[path = "../corpus_breadth.rs"] mod corpus_breadth;"#
        );
    }
}

#[test]
fn historical_readers_keep_the_actual_fixture_read_chain() {
    let source = std::fs::read_to_string(root().join("crates/rig-cassette/tests/corpus/mod.rs"))
        .expect("corpus source");
    let syntax = syn::parse_file(&source).expect("valid corpus source");
    assert!(reader_chain_problems(&syntax).is_empty());
    for (from, to) in [
        (
            "$crate::corpus::bus_engine_reproduces(&super::$program)",
            "$crate::corpus::unrelated(&super::$program)",
        ),
        (
            "let replay = Replay::open(program);",
            "let replay = Replay::open(other_program);",
        ),
        (
            "hand_drive(program, Resume::Never).await;",
            "hand_drive(other_program, Resume::Never).await;",
        ),
        (
            "let replay = Replay::open_checking(program, check);",
            "let replay = Replay::open_checking(other_program, check);",
        ),
        (
            "Self::open_checking(program, RequestCheck::Payload)",
            "Self::open_checking(other_program, RequestCheck::Payload)",
        ),
        (
            "let log = golden(program.fixture);",
            "let log = golden(\"unrelated\");",
        ),
        (
            "std::fs::read_to_string(&path)",
            "std::fs::read_to_string(&other_path)",
        ),
    ] {
        let changed = source.replace(from, to);
        assert_ne!(
            changed, source,
            "mutation must affect the actual chain: {from}"
        );
        let syntax = syn::parse_file(&changed).expect("changed corpus source");
        assert!(
            !reader_chain_problems(&syntax).is_empty(),
            "accepted broken chain: {from}"
        );
    }
}

#[test]
fn historical_ownership_cannot_claim_a_current_or_in_test_producer() {
    let name = HISTORICAL_GOLDENS[0]
        .label()
        .expect("exact path")
        .to_owned();
    let producers = BTreeMap::from([(name.clone(), vec!["current.rs".into()])]);
    let in_test = BTreeSet::from([name.clone()]);
    let failures = pairing_problems(&[name], &producers, &in_test, &HISTORICAL_GOLDENS[..1]);
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("must not claim a current producer"))
    );
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("cannot be checked in-test"))
    );
}
