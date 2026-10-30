//! Every current golden has one producer; the nine frozen historical logs
//! retain named offline readers. Pruned current producers check in-test.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use syn::visit::{self, Visit};

use rig_test_support::matrix_registry::Matrix;

#[path = "golden_pairing/tests.rs"]
mod tests;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixtures(suffix: &str) -> Vec<String> {
    let directory = root().join("crates/rig-cassette/fixtures/effects");
    let mut names: Vec<_> = std::fs::read_dir(directory)
        .expect("the corpus directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter_map(|name| name.strip_suffix(suffix).map(str::to_owned))
        .collect();
    names.sort();
    names
}

type Producers = BTreeMap<String, Vec<String>>;

#[derive(Default)]
struct Sites {
    agent: Producers,
    failures: Vec<String>,
    file: String,
    ignored: bool,
    test: Option<String>,
}

impl Sites {
    fn fail(&mut self, message: impl std::fmt::Display) {
        self.failures.push(format!("{}: {message}", self.file));
    }

    fn register(&mut self, name: &str, ignored: bool) {
        if ignored {
            return;
        }
        let location = self.test.as_ref().map_or_else(
            || self.file.clone(),
            |test| format!("{}::{test}", self.file),
        );
        self.agent
            .entry(name.to_owned())
            .or_default()
            .push(location);
    }
}

impl<'ast> Visit<'ast> for Sites {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let is_test = node.attrs.iter().any(|attr| {
            attr.path()
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "test")
        });
        self.ignored = node.attrs.iter().any(|attr| attr.path().is_ident("ignore"));
        self.test = is_test.then(|| node.sig.ident.to_string());
        visit::visit_item_fn(self, node);
        self.test = None;
        self.ignored = false;
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        match Matrix::of(node) {
            Some(Ok(Matrix::Golden(matrix))) => {
                let parts: Vec<_> = matrix
                    .oracle
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                if parts != ["crate", "goldens", "golden_effects"] {
                    self.fail("unknown agent matrix oracle");
                }
                for row in matrix.rows {
                    self.register(&row.golden.value(), row.ignored);
                }
            }
            Some(Ok(Matrix::Case(_))) | None => {}
            Some(Err(error)) => self.fail(error),
        }
        visit::visit_macro(self, node);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref() {
            let parts: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if parts.last().map(String::as_str) == Some("golden_effects") {
                if parts.iter().rev().nth(1).map(String::as_str) != Some("goldens") {
                    self.fail("qualify the golden helper with goldens");
                }
                if let Some(syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(name),
                    ..
                })) = call.args.first()
                {
                    self.register(&name.value(), self.ignored);
                } else {
                    self.fail("golden name must be a string literal");
                }
            }
        }
        visit::visit_expr_call(self, call);
    }
}

fn sites() -> Sites {
    let mut sites = Sites::default();
    let mut pending = vec![
        root().join("crates/rig-cassette/tests/providers"),
        root().join("tests/providers"),
        root().join("tests/core"),
    ];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("a tests directory") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = std::fs::read_to_string(&path).expect("source");
                let syntax = syn::parse_file(&source).expect("valid Rust source");
                sites.file = path
                    .strip_prefix(root())
                    .expect("under root")
                    .display()
                    .to_string();
                sites.visit_file(&syntax);
            }
        }
    }
    sites
}

/// The goldens the cassette prune replaced by an in-test check, by label:
/// their producers name a golden no file holds.
fn checked_in_test() -> BTreeSet<String> {
    let text = std::fs::read_to_string(root().join("crates/rig-cassette/coverage/pruned.tsv"))
        .unwrap_or_default();
    text.lines()
        .filter_map(|line| {
            let mut columns = line.split('\t');
            match (columns.next(), columns.next(), columns.next()) {
                (Some("golden"), Some(label), Some(covered)) if covered.starts_with("in-test ") => {
                    Some(label.to_owned())
                }
                _ => None,
            }
        })
        .collect()
}

/// These nine logs are frozen inputs of the named offline readers. Their
/// former producers now witness request refusal, not successful effect logs.
struct HistoricalGolden {
    path: &'static str,
    retired: &'static str,
    reader: HistoricalReader,
}

struct HistoricalReader {
    source: &'static str,
    module: &'static str,
    program: &'static str,
    row: &'static str,
}

const HISTORICAL_GOLDENS: &[HistoricalGolden] = &[
    HistoricalGolden {
        path: "crates/rig-cassette/fixtures/effects/gemini_breadth_custom_at_outcome.effects.json",
        retired: "case-080 gemini::cassette::corpus_breadth::custom_at_outcome_effect_log_is_the_golden_fixture",
        reader: HistoricalReader {
            source: "crates/rig-cassette/tests/corpus_breadth.rs",
            module: "corpus_breadth",
            program: "GEMINI_CUSTOM_AT_OUTCOME",
            row: "gemini_custom_at_outcome",
        },
    },
    HistoricalGolden {
        path: "crates/rig-cassette/fixtures/effects/gemini_retrieval_retrieved_tools_one.effects.json",
        retired: "case-155 gemini::cassette::corpus_retrieval::retrieved_tools_one_effect_log_is_the_golden_fixture",
        reader: HistoricalReader {
            source: "crates/rig-cassette/tests/corpus_retrieval.rs",
            module: "corpus_retrieval",
            program: "GEMINI_RETRIEVED_TOOLS_ONE",
            row: "gemini_retrieved_tools_one",
        },
    },
    HistoricalGolden {
        path: "crates/rig-cassette/fixtures/effects/gemini_breadth_output_tool_unary.effects.json",
        retired: "case-166 gemini::cassette::corpus_breadth::output_tool_unary_effect_log_is_the_golden_fixture",
        reader: HistoricalReader {
            source: "crates/rig-cassette/tests/corpus_breadth.rs",
            module: "corpus_breadth",
            program: "GEMINI_OUTPUT_TOOL_UNARY",
            row: "gemini_output_tool_unary",
        },
    },
    HistoricalGolden {
        path: "crates/rig-cassette/fixtures/effects/gemini_serving_two_turns_serial.effects.json",
        retired: "case-197 gemini::cassette::corpus_serving::two_turns_serial_effect_log_is_the_golden_fixture",
        reader: HistoricalReader {
            source: "crates/rig-cassette/tests/corpus_serving.rs",
            module: "corpus_serving",
            program: "GEMINI_TWO_TURNS_SERIAL",
            row: "gemini_two_turns_serial",
        },
    },
    HistoricalGolden {
        path: "crates/rig-cassette/fixtures/effects/gemini_retrieval_context_and_tools.effects.json",
        retired: "case-206 gemini::cassette::corpus_retrieval::context_and_tools_effect_log_is_the_golden_fixture",
        reader: HistoricalReader {
            source: "crates/rig-cassette/tests/corpus_retrieval.rs",
            module: "corpus_retrieval",
            program: "GEMINI_CONTEXT_AND_TOOLS",
            row: "gemini_context_and_tools",
        },
    },
    HistoricalGolden {
        path: "crates/rig-cassette/fixtures/effects/gemini_retrieval_retrieved_tools_with_static.effects.json",
        retired: "case-209 gemini::cassette::corpus_retrieval::retrieved_tools_with_static_effect_log_is_the_golden_fixture",
        reader: HistoricalReader {
            source: "crates/rig-cassette/tests/corpus_retrieval.rs",
            module: "corpus_retrieval",
            program: "GEMINI_RETRIEVED_TOOLS_WITH_STATIC",
            row: "gemini_retrieved_tools_with_static",
        },
    },
    HistoricalGolden {
        path: "crates/rig-cassette/fixtures/effects/gemini_breadth_output_tool_streamed.effects.json",
        retired: "case-219 gemini::cassette::corpus_breadth::output_tool_streamed_effect_log_is_the_golden_fixture",
        reader: HistoricalReader {
            source: "crates/rig-cassette/tests/corpus_breadth.rs",
            module: "corpus_breadth",
            program: "GEMINI_OUTPUT_TOOL_STREAMED",
            row: "gemini_output_tool_streamed",
        },
    },
    HistoricalGolden {
        path: "crates/rig-cassette/fixtures/effects/gemini_breadth_tool_dispatch_cancelled.effects.json",
        retired: "case-223 gemini::cassette::corpus_breadth::tool_dispatch_cancelled_effect_log_is_the_golden_fixture",
        reader: HistoricalReader {
            source: "crates/rig-cassette/tests/corpus_breadth.rs",
            module: "corpus_breadth",
            program: "GEMINI_DISPATCH_CANCELLED",
            row: "gemini_tool_dispatch_cancelled",
        },
    },
    HistoricalGolden {
        path: "crates/rig-cassette/fixtures/effects/gemini_tool_call_turns.effects.json",
        retired: "mec-rig-golden-ninth-1; mec-rig-132-dispositions-1 item 5 gemini::cassette::hook_stress::tool_call_turns_effect_log_is_the_golden_fixture",
        reader: HistoricalReader {
            source: "crates/rig-cassette/tests/golden_replay.rs",
            module: "golden_replay",
            program: "TOOL_CALL_TURNS",
            row: "reproduces_tool_call_turns",
        },
    },
];

impl HistoricalGolden {
    fn label(&self) -> Option<&str> {
        self.path
            .strip_prefix("crates/rig-cassette/fixtures/effects/")?
            .strip_suffix(".effects.json")
    }

    fn fail(&self, failures: &mut Vec<String>, reason: impl std::fmt::Display) {
        failures.push(format!(
            "historical golden `{}` (retired {}): {reason}",
            self.path, self.retired
        ));
    }
}

fn named_path(path: &syn::Path, names: &[&str]) -> bool {
    path.leading_colon.is_none()
        && path.segments.len() == names.len()
        && path.segments.iter().zip(names).all(|(segment, name)| {
            segment.ident == *name && matches!(segment.arguments, syn::PathArguments::None)
        })
}

fn path_expr(expr: &syn::Expr, names: &[&str]) -> bool {
    matches!(expr, syn::Expr::Path(path) if path.attrs.is_empty() && path.qself.is_none() && named_path(&path.path, names))
}

fn call_args<'a>(
    expr: &'a syn::Expr,
    names: &[&str],
) -> Option<&'a syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>> {
    match expr {
        syn::Expr::Call(call) if call.attrs.is_empty() && path_expr(&call.func, names) => {
            Some(&call.args)
        }
        _ => None,
    }
}

fn local_value<'a>(statement: &'a syn::Stmt, name: &str) -> Option<&'a syn::Expr> {
    let syn::Stmt::Local(local) = statement else {
        return None;
    };
    let syn::Pat::Ident(binding) = &local.pat else {
        return None;
    };
    let init = local.init.as_ref()?;
    (local.attrs.is_empty()
        && binding.ident == name
        && binding.subpat.is_none()
        && init.diverge.is_none())
    .then_some(init.expr.as_ref())
}

fn unconditional(attrs: &[syn::Attribute]) -> bool {
    !attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            || attr.path().is_ident("cfg_attr")
            || attr.path().is_ident("ignore")
    })
}

fn reader_problems(golden: &HistoricalGolden, syntax: &syn::File) -> Vec<String> {
    use syn::parse::Parser;
    let mut failures = Vec::new();
    let reader = &golden.reader;
    let programs: Vec<_> = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Const(item) if item.ident == reader.program => Some(item),
            _ => None,
        })
        .collect();
    let correct_program = match programs.as_slice() {
        [item] if item.attrs.is_empty() => {
            let typed = matches!(item.ty.as_ref(), syn::Type::Path(path) if path.qself.is_none() && named_path(&path.path, &["Program"]));
            let fixture = match item.expr.as_ref() {
                syn::Expr::Struct(value)
                    if value.attrs.is_empty()
                        && value.qself.is_none()
                        && named_path(&value.path, &["Program"]) =>
                {
                    let fields: Vec<_> = value.fields.iter().filter(|field| matches!(&field.member, syn::Member::Named(name) if name == "fixture")).collect();
                    matches!(fields.as_slice(), [field] if field.attrs.is_empty() && matches!(&field.expr, syn::Expr::Lit(value) if matches!(&value.lit, syn::Lit::Str(label) if Some(label.value().as_str()) == golden.label())))
                }
                _ => false,
            };
            typed && fixture
        }
        _ => false,
    };
    if !correct_program {
        golden.fail(
            &mut failures,
            format!(
                "{} must define {}: Program with its exact fixture literal",
                reader.source, reader.program
            ),
        );
    }
    let mut registered = 0;
    for item in &syntax.items {
        let syn::Item::Macro(item) = item else {
            continue;
        };
        if !item.mac.path.is_ident("both_interpreters") {
            continue;
        }
        if !item.attrs.is_empty() {
            golden.fail(&mut failures, "reader registration must be unconditional");
            continue;
        }
        let rows = syn::punctuated::Punctuated::<syn::FieldValue, syn::Token![,]>::parse_terminated
            .parse2(item.mac.tokens.clone());
        match rows {
            Ok(rows) => {
                for field in rows {
                    if matches!(&field.member, syn::Member::Named(name) if name == reader.row) {
                        if field.attrs.is_empty()
                            && field.colon_token.is_some()
                            && path_expr(&field.expr, &[reader.program])
                        {
                            registered += 1;
                        } else {
                            golden.fail(
                                &mut failures,
                                format!("reader row {} must use {}", reader.row, reader.program),
                            );
                        }
                    }
                }
            }
            Err(error) => golden.fail(
                &mut failures,
                format!("invalid both_interpreters rows: {error}"),
            ),
        }
    }
    if registered != 1 {
        golden.fail(&mut failures, format!("{} must register exactly one {} row for verify::{}::{{bus_engine,hand_driver}}::{} (found {registered})", reader.source, reader.row, reader.module, reader.row));
    }
    failures
}

fn module_reads(syntax: &syn::File, name: &str, source: &str, macro_use: bool) -> bool {
    let modules: Vec<_> = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Mod(module) if module.ident == name => Some(module),
            _ => None,
        })
        .collect();
    matches!(modules.as_slice(), [module] if module.content.is_none() && unconditional(&module.attrs)
        && module.attrs.iter().filter(|attr| attr.path().is_ident("path")).count() == 1
        && module.attrs.iter().any(|attr| matches!(&attr.meta, syn::Meta::NameValue(value) if value.path.is_ident("path") && matches!(&value.value, syn::Expr::Lit(value) if matches!(&value.lit, syn::Lit::Str(path) if path.value() == source))))
        && (!macro_use || module.attrs.iter().any(|attr| attr.path().is_ident("macro_use"))))
}

fn function<'a>(syntax: &'a syn::File, name: &str) -> Option<&'a syn::ItemFn> {
    let mut found = syntax.items.iter().filter_map(|item| match item {
        syn::Item::Fn(item) if item.sig.ident == name && unconditional(&item.attrs) => Some(item),
        _ => None,
    });
    let item = found.next()?;
    found.next().is_none().then_some(item)
}

fn replay_method<'a>(syntax: &'a syn::File, name: &str) -> Option<&'a syn::ImplItemFn> {
    let mut found = syntax.items.iter().filter_map(|item| match item {
        syn::Item::Impl(item) if item.trait_.is_none() && unconditional(&item.attrs)
            && matches!(item.self_ty.as_ref(), syn::Type::Path(path) if path.qself.is_none() && named_path(&path.path, &["Replay"])) => Some(item),
        _ => None,
    }).flat_map(|item| item.items.iter()).filter_map(|item| match item {
        syn::ImplItem::Fn(item) if item.sig.ident == name && unconditional(&item.attrs) => Some(item),
        _ => None,
    });
    let item = found.next()?;
    found.next().is_none().then_some(item)
}

fn statement_expr(statement: &syn::Stmt) -> Option<&syn::Expr> {
    match statement {
        syn::Stmt::Expr(expr, _) => Some(expr),
        _ => None,
    }
}

fn borrowed_ident(expr: &syn::Expr, name: &str) -> bool {
    matches!(expr, syn::Expr::Reference(value) if value.attrs.is_empty() && value.mutability.is_none() && path_expr(&value.expr, &[name]))
}

fn expect_receiver(expr: &syn::Expr) -> Option<&syn::Expr> {
    match expr {
        syn::Expr::MethodCall(call)
            if call.attrs.is_empty() && call.method == "expect" && call.args.len() == 1 =>
        {
            Some(&call.receiver)
        }
        _ => None,
    }
}

fn reads_effect_file(syntax: &syn::File) -> bool {
    let Some(function) = function(syntax, "golden") else {
        return false;
    };
    let [path, text, result] = function.block.stmts.as_slice() else {
        return false;
    };
    let Some(syn::Expr::MethodCall(join)) = local_value(path, "path") else {
        return false;
    };
    let format: syn::ExprMacro = syn::parse_quote!(format!("{fixture}.effects.json"));
    let correct_path = join.attrs.is_empty()
        && join.method == "join"
        && call_args(&join.receiver, &["fixtures", "effects_dir"])
            .is_some_and(|args| args.is_empty())
        && matches!(join.args.first(), Some(syn::Expr::Macro(value)) if value.mac.path.is_ident("format") && value.mac.tokens.to_string() == format.mac.tokens.to_string())
        && join.args.len() == 1;
    let correct_read = local_value(text, "text")
        .and_then(expect_receiver)
        .and_then(|expr| call_args(expr, &["std", "fs", "read_to_string"]))
        .is_some_and(|args| {
            args.len() == 1
                && args
                    .first()
                    .is_some_and(|expr| borrowed_ident(expr, "path"))
        });
    let correct_decode = statement_expr(result)
        .and_then(expect_receiver)
        .and_then(|expr| call_args(expr, &["serde_json", "from_str"]))
        .is_some_and(|args| {
            args.len() == 1
                && args
                    .first()
                    .is_some_and(|expr| borrowed_ident(expr, "text"))
        });
    correct_path && correct_read && correct_decode
}

fn reader_chain_problems(syntax: &syn::File) -> Vec<String> {
    let mut failures = Vec::new();
    // Lexed from text like the corpus source, so both token streams carry the
    // lexer's punctuation spacing: `parse_quote!` spaces `),*` and `::$`
    // differently, and its tokens never print equal to the parsed file's.
    let expected: syn::ItemMacro = syn::parse_str(
        r#"
macro_rules! both_interpreters {
    ($($test:ident: $program:ident),* $(,)?) => {
        mod bus_engine {
            $(
                #[tokio::test]
                async fn $test() {
                    $crate::corpus::bus_engine_reproduces(&super::$program).await;
                }
            )*
        }
        mod hand_driver {
            $(
                #[tokio::test]
                async fn $test() {
                    $crate::corpus::hand_driver_reproduces(&super::$program).await;
                }
            )*
        }
    };
}
"#,
    )
    .expect("the expected reader macro parses");
    let macros: Vec<_> = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Macro(item)
                if item
                    .ident
                    .as_ref()
                    .is_some_and(|name| name == "both_interpreters") =>
            {
                Some(item)
            }
            _ => None,
        })
        .collect();
    if !matches!(macros.as_slice(), [item] if unconditional(&item.attrs) && item.mac.path.is_ident("macro_rules") && item.mac.tokens.to_string() == expected.mac.tokens.to_string())
    {
        failures.push(
            "historical readers require the two original both_interpreters test bodies".into(),
        );
    }
    let bus_opens = function(syntax, "bus_engine_reproduces")
        .and_then(|item| item.block.stmts.first())
        .and_then(|stmt| local_value(stmt, "replay"))
        .and_then(|expr| call_args(expr, &["Replay", "open"]))
        .is_some_and(|args| {
            args.len() == 1 && args.first().is_some_and(|arg| path_expr(arg, &["program"]))
        });
    let hand_delegates = function(syntax, "hand_driver_reproduces").is_some_and(|item| {
        matches!(item.block.stmts.as_slice(), [stmt] if statement_expr(stmt).is_some_and(|expr| {
            matches!(expr, syn::Expr::Await(value) if value.attrs.is_empty() && call_args(&value.base, &["hand_drive"]).is_some_and(|args| args.len() == 2 && args.first().is_some_and(|arg| path_expr(arg, &["program"])) && args.last().is_some_and(|arg| path_expr(arg, &["Resume", "Never"]))))
        }))
    });
    let hand_opens = function(syntax, "hand_drive").is_some_and(|item| {
        let mut statements = item.block.stmts.iter();
        let check = statements.next().and_then(|stmt| local_value(stmt, "check"));
        let inert_check = matches!(check, Some(syn::Expr::Match(value)) if value.attrs.is_empty() && path_expr(&value.expr, &["resume"]) && value.arms.len() == 2 && value.arms.iter().all(|arm| arm.attrs.is_empty() && arm.guard.is_none() && (path_expr(&arm.body, &["check"]) || path_expr(&arm.body, &["RequestCheck", "Payload"]))));
        let open = statements.next().and_then(|stmt| local_value(stmt, "replay"))
            .and_then(|expr| call_args(expr, &["Replay", "open_checking"]));
        inert_check && open.is_some_and(|args| args.len() == 2 && args.first().is_some_and(|arg| path_expr(arg, &["program"])) && args.last().is_some_and(|arg| path_expr(arg, &["check"])))
    });
    let open_delegates = replay_method(syntax, "open").is_some_and(|item| {
        matches!(item.block.stmts.as_slice(), [stmt] if statement_expr(stmt).and_then(|expr| call_args(expr, &["Self", "open_checking"])).is_some_and(|args| args.len() == 2 && args.first().is_some_and(|arg| path_expr(arg, &["program"])) && args.last().is_some_and(|arg| path_expr(arg, &["RequestCheck", "Payload"]))))
    });
    let opens_fixture = replay_method(syntax, "open_checking").and_then(|item| item.block.stmts.first())
        .and_then(|stmt| local_value(stmt, "log")).and_then(|expr| call_args(expr, &["golden"]))
        .is_some_and(|args| args.len() == 1 && matches!(args.first(), Some(syn::Expr::Field(value)) if value.attrs.is_empty() && path_expr(&value.base, &["program"]) && matches!(&value.member, syn::Member::Named(name) if name == "fixture")));
    for (passes, link) in [
        (bus_opens, "bus_engine_reproduces -> Replay::open(program)"),
        (
            hand_delegates,
            "hand_driver_reproduces -> hand_drive(program, Resume::Never)",
        ),
        (
            hand_opens,
            "hand_drive -> Replay::open_checking(program, check)",
        ),
        (
            open_delegates,
            "Replay::open -> Self::open_checking(program, RequestCheck::Payload)",
        ),
        (
            opens_fixture,
            "Replay::open_checking -> golden(program.fixture)",
        ),
        (
            reads_effect_file(syntax),
            "golden -> read_to_string(effects_dir/fixture.effects.json)",
        ),
    ] {
        if !passes {
            failures.push(format!("historical reader chain changed: {link}"));
        }
    }
    failures
}

fn read_syntax(path: &str, failures: &mut Vec<String>) -> Option<syn::File> {
    match std::fs::read_to_string(root().join(path)) {
        Ok(source) => match syn::parse_file(&source) {
            Ok(syntax) => Some(syntax),
            Err(error) => {
                failures.push(format!("{path}: {error}"));
                None
            }
        },
        Err(error) => {
            failures.push(format!("{path}: {error}"));
            None
        }
    }
}

fn historical_reader_problems() -> Vec<String> {
    let mut failures = Vec::new();
    let entrypoint = read_syntax("crates/rig-cassette/tests/verify/main.rs", &mut failures);
    let corpus = read_syntax("crates/rig-cassette/tests/corpus/mod.rs", &mut failures);
    if let Some(syntax) = &corpus {
        failures.extend(reader_chain_problems(syntax));
    }
    if entrypoint
        .as_ref()
        .is_some_and(|syntax| !module_reads(syntax, "corpus", "../corpus/mod.rs", true))
    {
        failures
            .push("verify must include the original corpus macro and reader implementation".into());
    }
    let mut sources = BTreeMap::new();
    for golden in HISTORICAL_GOLDENS {
        if !root().join(golden.path).is_file() {
            golden.fail(&mut failures, "the exact frozen fixture path is not a file");
        }
        let reader = &golden.reader;
        let source = sources
            .entry(reader.source)
            .or_insert_with(|| read_syntax(reader.source, &mut failures));
        if let Some(syntax) = source {
            failures.extend(reader_problems(golden, syntax));
        }
        let relative = format!("../{}.rs", reader.module);
        if entrypoint
            .as_ref()
            .is_some_and(|syntax| !module_reads(syntax, reader.module, &relative, false))
        {
            golden.fail(
                &mut failures,
                format!("verify must include {} as {}", reader.source, reader.module),
            );
        }
    }
    failures
}

fn pairing_problems(
    fixtures: &[String],
    producers: &Producers,
    in_test: &BTreeSet<String>,
    historical: &[HistoricalGolden],
) -> Vec<String> {
    let mut problems = Vec::new();
    let mut historical_labels = BTreeSet::new();
    for golden in historical {
        let Some(label) = golden.label() else {
            golden.fail(&mut problems, "invalid exact historical fixture path");
            continue;
        };
        if !historical_labels.insert(label) {
            golden.fail(&mut problems, "duplicate historical fixture entry");
        }
        if !fixtures.iter().any(|name| name == label) {
            golden.fail(&mut problems, "the frozen reader-owned file is missing");
        }
        if producers.contains_key(label) {
            golden.fail(
                &mut problems,
                "a frozen historical log must not claim a current producer",
            );
        }
        if in_test.contains(label) {
            golden.fail(
                &mut problems,
                "a retained historical log cannot be checked in-test",
            );
        }
    }
    for name in fixtures {
        if historical_labels.contains(name.as_str()) {
            continue;
        }
        match producers.get(name).map(Vec::as_slice) {
            Some([_]) => {}
            Some(locations) => problems.push(format!(
                "golden `{name}` has {} producers: {locations:?}",
                locations.len()
            )),
            None => problems.push(format!("golden `{name}` has no producer")),
        }
    }
    for (name, locations) in producers {
        if in_test.contains(name) {
            if locations.len() != 1 {
                problems.push(format!(
                    "golden `{name}` checked in-test has {} producers: {locations:?}",
                    locations.len()
                ));
            }
            if fixtures.contains(name) {
                problems.push(format!(
                    "golden `{name}` is committed but listed as checked in-test"
                ));
            }
            continue;
        }
        if !fixtures.contains(name) {
            problems.push(format!("{locations:?} names missing golden `{name}`"));
        }
    }
    problems
}

#[test]
fn every_golden_has_exactly_one_producer_and_every_producer_a_golden() {
    let agent = fixtures(".effects.json");
    assert!(!agent.is_empty(), "the corpus must exist");
    let sites = sites();
    let mut problems = sites.failures.clone();
    problems.extend(historical_reader_problems());
    problems.extend(pairing_problems(
        &agent,
        &sites.agent,
        &checked_in_test(),
        HISTORICAL_GOLDENS,
    ));
    assert!(
        problems.is_empty(),
        "goldens and producers are not paired:\n{}",
        problems.join("\n")
    );
}
