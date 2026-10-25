use super::*;

const WRAPPERS: &[&str] = &["with_cassette"];

#[test]
fn direct_calls_specs_and_ignored_functions_keep_their_registration_semantics() {
    let source = r#"
        fn direct() { crate::support::with_cassette("direct", callback); }
        fn spec() { with_cassette((CassetteSpec::new("spec").timeout(5)), callback); }
        #[test] #[ignore = "not recorded"] fn ignored() { with_cassette("absent", callback); }
        fn unrelated() { with_other("other", callback); }
    "#;
    assert_eq!(
        cassette_scenarios(source, WRAPPERS).expect("registrations"),
        ["direct", "spec"]
    );
    for source in [
        "fn bad() { with_cassette(name, callback); }",
        "fn bad() { with_cassette(CassetteSpec::new(name), callback); }",
        "fn bad() { with_cassette(); }",
        "not Rust",
    ] {
        assert!(
            cassette_scenarios(source, WRAPPERS).is_err(),
            "accepted {source}"
        );
    }
}

#[test]
fn qualified_matrix_invocations_in_modules_claim_only_recorded_rows() {
    let source = r#"
        mod nested {
            crate::matrix::golden_matrix! {
                wrapper: super::with_cassette, wire: wire, run: run, oracle: crate::goldens::golden_effects;
                #[tokio::test] first: ("first", CELL, "first_golden");
                #[tokio::test] #[ignore = "unrecorded"] ignored: ("absent", CELL, "absent_golden");
            }
            crate::matrix::case_matrix! {
                family: wire_matrix_case;
                #[tokio::test] scripted: truncated_after_text;
                #[tokio::test] #[ignore = "unsupported"] skipped: refusal;
            }
            crate::matrix::case_matrix! {
                wrapper: with_cassette, family: tool_lifecycle_matrix_case;
                #[tokio::test] tools: ("tools", configured, cell(Blocking, Model, Shape));
            }
        }
    "#;
    assert_eq!(
        cassette_scenarios(source, WRAPPERS).expect("matrix rows"),
        ["first", "tools"]
    );
}

#[test]
fn comments_strings_and_macro_definitions_are_not_invocations() {
    let source = r#"
        // with_cassette("comment", callback);
        const TEXT: &str = "with_cassette(\"string\", callback)";
        macro_rules! definition { () => { with_cassette("definition", callback); }; }
    "#;
    assert!(
        cassette_scenarios(source, WRAPPERS)
            .expect("source")
            .is_empty()
    );
}

#[test]
fn malformed_matrix_rows_fail_instead_of_vanishing() {
    for source in [
        "golden_matrix! { wrapper: with_cassette, wire: wire, run: run, oracle: golden; #[tokio::test] a: (dynamic(), CELL, \"golden\"); }",
        "case_matrix! { family: wire_matrix_case; missing_attribute: truncation; }",
        "case_matrix! { wrapper: with_cassette, family: family; #[tokio::test] a: (\"a\", row); #[tokio::test] a: (\"b\", row); }",
    ] {
        assert!(
            cassette_scenarios(source, WRAPPERS).is_err(),
            "accepted {source}"
        );
    }
}

#[test]
fn sites_keep_the_wrapper_and_the_specs_declarations() {
    let source = r#"
        async fn cells() {
            with_x_cassette("plain/cell", |c| async {}).await;
            with_x_bogus_key(
                CassetteSpec::new("auth/cell")
                    .unordered()
                    .expects_account_failure(crate::cassettes::AccountFailure::Auth),
                |c| async {},
            )
            .await;
        }
    "#;
    let sites = cassette_scenario_sites(source, &["with_x_cassette", "with_x_bogus_key"])
        .expect("sites parse");
    assert_eq!(
        sites,
        [
            ScenarioSite {
                scenario: "plain/cell".into(),
                wrapper: "with_x_cassette".into(),
                declared: Vec::new(),
            },
            ScenarioSite {
                scenario: "auth/cell".into(),
                wrapper: "with_x_bogus_key".into(),
                declared: vec!["Auth".into()],
            },
        ]
    );
}

#[test]
fn a_wrapper_declares_through_its_session() {
    let source = r#"
        async fn with_x_bogus_key() {
            let cassette = ProviderCassette::start(root, "x", spec, url).await;
            cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
        }
        async fn with_x_rejected_key() {
            let cassette = ProviderCassette::start(root, "x", spec, url).await;
            let key = cassette.bogus_api_key();
        }
        async fn with_x_cassette() {
            let cassette = ProviderCassette::start(root, "x", spec, url).await;
            let key = cassette.api_key("X_API_KEY");
        }
    "#;
    let declaring = declaring_functions(source).expect("declarations parse");
    assert_eq!(
        declaring,
        [
            ("with_x_bogus_key".to_owned(), vec!["Auth".to_owned()]),
            ("with_x_rejected_key".to_owned(), vec!["Auth".to_owned()]),
        ]
    );
}

#[test]
fn read_only_readers_register_literal_and_same_file_constant_cassettes() {
    let source = r#"
        const PROVIDER: &str = "x";
        const SHARED: &str = "file/shared";
        fn literal() { crate::cassettes::recorded_interaction_bodies("y", "dir/literal"); }
        fn constants() { recorded_sse_json_frames(PROVIDER, SHARED); }
        fn local() {
            const SCENARIO: &str = "dir/local";
            let frames = chat::recorded_sole_usage_frame(PROVIDER, &SCENARIO);
        }
        fn other_local() {
            const SCENARIO: &str = "dir/other_local";
            stream_faults::recorded_sse_frames(PROVIDER, SCENARIO, 0);
        }
        fn helper(scenario: &str) { recorded_json_request(PROVIDER, scenario); }
        fn computed() { recorded_json_request(PROVIDER, &format!("dir/{}", 1)); }
        #[test] #[ignore = "not recorded"] fn ignored() { recorded_json_request("x", "absent"); }
        fn unrelated() { recorded_other("x", "other"); }
    "#;
    let pair = |provider: &str, scenario: &str| (provider.to_owned(), scenario.to_owned());
    assert_eq!(
        read_only_cassettes(source).expect("readers parse"),
        [
            pair("y", "dir/literal"),
            pair("x", "file/shared"),
            pair("x", "dir/local"),
            pair("x", "dir/other_local"),
        ]
    );
    assert!(read_only_cassettes("not Rust").is_err());
}

const IN_SOURCE_READERS: &[InSourceReader] = &[
    InSourceReader::Included {
        root: "/../rig-cassette/fixtures/cassettes/",
    },
    InSourceReader::Path {
        name: "recorded",
        provider: "openai",
    },
    InSourceReader::Path {
        name: "recorded_json",
        provider: "openai",
    },
    InSourceReader::ParityRow,
];

/// Each in-source reader registers the cassettes its literal and same-file
/// constant paths name. A path built at run time, through a `format!`
/// helper or a parameter, registers nothing, and a file nothing names stays
/// an orphan.
#[test]
fn in_source_readers_register_literal_and_same_file_constant_cassettes() -> anyhow::Result<()> {
    let source = r#"
        const SHARED: &str = "chat/shared.yaml";
        fn included() {
            drain(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
                "/../rig-cassette/fixtures/cassettes/ollama/grammar/included.yaml")));
        }
        fn local() {
            const FIXTURE: &str = "../llamacpp/content/local.yaml";
            let request = recorded_json("when", FIXTURE);
            let reply = recorded("then", SHARED);
        }
        fn fixture(scenario: &str) -> String {
            format!("originals/{scenario}.yaml")
        }
        async fn run_cell(scenario: &str, cell: Cell) {
            let body = recorded("then", &fixture(scenario));
        }
        async fn first() { run_cell("first", cell(Blocking)).await }
        fn passed(path: &str) { recorded("then", path); }
        fn calls_passed() { passed("originals/passed.yaml"); }
        const PAIRS: &[Pair] = &[Pair {
            provider: "openai",
            unary: "matrix/blocking.yaml",
            streamed: "matrix/streaming.yaml#2",
            minted: &[],
        }];
        fn named() { Recorded::named("gemini", "raw/named.yaml"); }
        #[test] #[ignore = "not recorded"] fn ignored() { recorded("then", "absent.yaml"); }
        fn escaping() { recorded("then", "../../outside.yaml"); }
        fn not_a_cassette() { recorded("then", "notes.txt"); }
        fn unrelated() {
            other("then", "unrelated.yaml");
            include_str!("../unrelated.yaml");
        }
        // recorded("then", "comment.yaml");
    "#;
    let read: std::collections::BTreeSet<String> = in_source_cassettes(source, IN_SOURCE_READERS)?
        .into_iter()
        .map(|(provider, scenario)| format!("{provider}/{scenario}"))
        .collect();
    let registered = [
        "ollama/grammar/included",
        "llamacpp/content/local",
        "openai/chat/shared",
        "openai/matrix/blocking",
        "openai/matrix/streaming",
        "gemini/raw/named",
    ];
    anyhow::ensure!(
        read == registered.iter().map(|path| (*path).to_owned()).collect(),
        "registered {read:?}"
    );

    let corpus = registered.iter().copied().chain([
        "openai/originals/first",
        "openai/originals/passed",
        "openai/originals/unreferenced",
        "openai/absent",
        "openai/unrelated",
    ]);
    let orphans: Vec<&str> = corpus.filter(|path| !read.contains(*path)).collect();
    anyhow::ensure!(
        orphans
            == [
                "openai/originals/first",
                "openai/originals/passed",
                "openai/originals/unreferenced",
                "openai/absent",
                "openai/unrelated"
            ],
        "orphans {orphans:?}"
    );

    anyhow::ensure!(in_source_cassettes("not Rust; recorded", IN_SOURCE_READERS).is_err());
    anyhow::ensure!(in_source_cassettes("not Rust", IN_SOURCE_READERS)?.is_empty());
    Ok(())
}
