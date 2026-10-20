//! Discover cassette references from direct calls and supported matrix rows.

use crate::matrix_registry;
use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprLit, ItemFn, Lit};

/// Malformed Rust or a scenario declaration that cannot be checked statically.
#[derive(Debug, thiserror::Error)]
pub enum ScenarioError {
    /// The source is not valid Rust syntax.
    #[error(transparent)]
    Parse(#[from] syn::Error),
    /// A wrapper or supported matrix has an invalid scenario declaration.
    #[error("{0}")]
    Invalid(String),
}

/// Collect recorded scenarios, excluding ignored functions and matrix rows.
/// Direct calls retain support for literal `CassetteSpec` builder expressions.
pub fn cassette_scenarios(
    source: &str,
    wrapper_names: &[&'static str],
) -> Result<Vec<String>, ScenarioError> {
    Ok(cassette_scenario_sites(source, wrapper_names)?
        .into_iter()
        .map(|site| site.scenario)
        .collect())
}

/// One recorded scenario and the call that records it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenarioSite {
    /// The scenario literal.
    pub scenario: String,
    /// The cassette wrapper the call goes through.
    pub wrapper: String,
    /// Account failures the spec itself declares
    /// (`.expects_account_failure(AccountFailure::Auth)` gives `"Auth"`).
    pub declared: Vec<String>,
}

/// Like [`cassette_scenarios`], keeping each scenario's wrapper and the
/// account failures its spec declares.
pub fn cassette_scenario_sites(
    source: &str,
    wrapper_names: &[&'static str],
) -> Result<Vec<ScenarioSite>, ScenarioError> {
    let syntax = syn::parse_file(source)?;
    let mut visitor = CassetteScenarioVisitor {
        wrapper_names,
        scenarios: Vec::new(),
        failures: Vec::new(),
    };
    visitor.visit_file(&syntax);
    if visitor.failures.is_empty() {
        Ok(visitor.scenarios)
    } else {
        Err(ScenarioError::Invalid(visitor.failures.join("\n")))
    }
}

/// The functions in `source` that declare account failures on their
/// cassette session, with what they declare: a call to
/// `expect_account_failure(AccountFailure::X)` declares `X`, and a call to
/// `bogus_api_key()` declares `Auth`.
pub fn declaring_functions(source: &str) -> Result<Vec<(String, Vec<String>)>, ScenarioError> {
    let syntax = syn::parse_file(source)?;
    let mut visitor = DeclaringVisitor::default();
    visitor.visit_file(&syntax);
    Ok(visitor.functions)
}

#[derive(Default)]
struct DeclaringVisitor {
    functions: Vec<(String, Vec<String>)>,
    current: Option<(String, Vec<String>)>,
}

impl<'ast> Visit<'ast> for DeclaringVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let outer = self
            .current
            .replace((node.sig.ident.to_string(), Vec::new()));
        visit::visit_item_fn(self, node);
        if let Some((name, declared)) = std::mem::replace(&mut self.current, outer)
            && !declared.is_empty()
        {
            self.functions.push((name, declared));
        }
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let declared = match node.method.to_string().as_str() {
            "expect_account_failure" => node.args.first().and_then(last_path_segment),
            "bogus_api_key" => Some("Auth".to_owned()),
            _ => None,
        };
        if let (Some(declared), Some((_, list))) = (declared, self.current.as_mut())
            && !list.contains(&declared)
        {
            list.push(declared);
        }
        visit::visit_expr_method_call(self, node);
    }
}

fn last_path_segment(expr: &Expr) -> Option<String> {
    let Expr::Path(path) = expr else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

/// Account failures a `CassetteSpec` builder chain declares.
fn spec_declarations(expr: &Expr) -> Vec<String> {
    match expr {
        Expr::MethodCall(call) => {
            let mut declared = spec_declarations(&call.receiver);
            if call.method == "expects_account_failure"
                && let Some(kind) = call.args.first().and_then(last_path_segment)
            {
                declared.push(kind);
            }
            declared
        }
        Expr::Paren(paren) => spec_declarations(&paren.expr),
        _ => Vec::new(),
    }
}

struct CassetteScenarioVisitor<'a> {
    wrapper_names: &'a [&'static str],
    scenarios: Vec<ScenarioSite>,
    failures: Vec<String>,
}

impl<'ast, 'a> Visit<'ast> for CassetteScenarioVisitor<'a> {
    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        match matrix_registry::Matrix::of(node).transpose() {
            Ok(matrix) => {
                if let Some((wrapper, scenarios)) =
                    matrix.and_then(matrix_registry::Matrix::recorded)
                    && self.wrapper_names.contains(&wrapper.as_str())
                {
                    self.scenarios
                        .extend(scenarios.into_iter().map(|scenario| ScenarioSite {
                            scenario,
                            wrapper: wrapper.clone(),
                            declared: Vec::new(),
                        }));
                }
            }
            Err(error) => self.failures.push(error.to_string()),
        }
        visit::visit_macro(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        // A `#[ignore]`d test documents that its cassette isn't recorded yet
        // (e.g. no provider API key available to record with); don't require
        // a file for scenarios it references.
        if node.attrs.iter().any(|attr| attr.path().is_ident("ignore")) {
            return;
        }

        visit::visit_item_fn(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Some(wrapper_name) = cassette_wrapper_name(node)
            && self.wrapper_names.contains(&wrapper_name.as_str())
        {
            match node.args.first() {
                Some(expr) => match cassette_scenario_value(expr) {
                    Some(scenario) => self.scenarios.push(ScenarioSite {
                        scenario,
                        wrapper: wrapper_name.clone(),
                        declared: spec_declarations(expr),
                    }),
                    None => self.failures.push(format!(
                        "calls {wrapper_name} without a string-literal cassette scenario"
                    )),
                },
                _ => self.failures.push(format!(
                    "calls {wrapper_name} without a string-literal cassette scenario"
                )),
            }
        }

        visit::visit_expr_call(self, node);
    }
}

/// The read-only cassette readers: each takes `(provider, scenario, ..)` and
/// reads that recorded cassette without replaying it, so a test that only
/// reads a fixture through one of them still uses it.
pub const READ_ONLY_READERS: &[&str] = &[
    "recorded_interaction_bodies",
    "recorded_json_request",
    "recorded_json_response",
    "recorded_request_header_pairs",
    "recorded_response_header_pairs",
    "recorded_response_header",
    "recorded_json_turns",
    "recorded_json_turn",
    "recorded_request_query_pairs",
    "recorded_request_paths",
    "recorded_statuses_and_bodies",
    "recorded_sse_json_frames",
    // `stream_faults::recorded_sse_frames`.
    "recorded_sse_frames",
    // `raw_capture::chat`.
    "recorded_sole_usage_frame",
    "recorded_frames_with_terminal",
    "recorded_agreeing_usage_frames",
];

/// The `(provider, scenario)` pairs `source` reads through a
/// [`READ_ONLY_READERS`] call whose first two arguments are string literals
/// or `&str` constants declared in the same file, a function's own
/// constants before the file's. A call with any other argument is not
/// counted, and neither is a call in an `#[ignore]`d function, as for
/// [`cassette_scenarios`].
pub fn read_only_cassettes(source: &str) -> Result<Vec<(String, String)>, ScenarioError> {
    let syntax = syn::parse_file(source)?;
    let mut visitor = ReaderVisitor {
        consts: Constants::of(&syntax),
        read: Vec::new(),
    };
    visitor.visit_file(&syntax);
    Ok(visitor.read)
}

/// String constants by name; a name declared twice with different values is
/// ambiguous and resolves to nothing.
#[derive(Default)]
struct ConstCollector {
    values: std::collections::BTreeMap<String, Option<String>>,
    /// Whether to stop at nested functions, which hold their own constants.
    outer_only: bool,
}

impl ConstCollector {
    fn unambiguous(self) -> std::collections::BTreeMap<String, String> {
        self.values
            .into_iter()
            .filter_map(|(name, value)| Some((name, value?)))
            .collect()
    }
}

impl<'ast> Visit<'ast> for ConstCollector {
    fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
        if let Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) = node.expr.as_ref()
        {
            let value = value.value();
            self.values
                .entry(node.ident.to_string())
                .and_modify(|known| {
                    if known.as_deref() != Some(value.as_str()) {
                        *known = None;
                    }
                })
                .or_insert(Some(value));
        }
        visit::visit_item_const(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        if !self.outer_only {
            visit::visit_item_fn(self, node);
        }
    }
}

/// The string constants in scope: the file's, and those of each enclosing
/// function, innermost last.
struct Constants {
    file: std::collections::BTreeMap<String, String>,
    local: Vec<std::collections::BTreeMap<String, String>>,
}

impl Constants {
    fn of(syntax: &syn::File) -> Self {
        let mut file = ConstCollector::default();
        file.visit_file(syntax);
        Self {
            file: file.unambiguous(),
            local: Vec::new(),
        }
    }

    /// Enter `function`, whose own constants shadow the file's.
    fn enter(&mut self, function: &ItemFn) {
        let mut consts = ConstCollector {
            outer_only: true,
            ..ConstCollector::default()
        };
        visit::visit_block(&mut consts, &function.block);
        self.local.push(consts.unambiguous());
    }

    fn leave(&mut self) {
        self.local.pop();
    }

    /// The value of a string literal or of a constant in scope, behind any
    /// references and parentheses.
    fn value(&self, expr: &Expr) -> Option<String> {
        match unwrapped(expr) {
            Expr::Lit(ExprLit {
                lit: Lit::Str(value),
                ..
            }) => Some(value.value()),
            Expr::Path(path) if path.qself.is_none() => {
                let name = path.path.get_ident()?.to_string();
                self.local
                    .iter()
                    .rev()
                    .find_map(|scope| scope.get(&name))
                    .or_else(|| self.file.get(&name))
                    .cloned()
            }
            _ => None,
        }
    }
}

/// `expr` behind any references and parentheses.
fn unwrapped(expr: &Expr) -> &Expr {
    match expr {
        Expr::Reference(reference) => unwrapped(&reference.expr),
        Expr::Paren(paren) => unwrapped(&paren.expr),
        expr => expr,
    }
}

fn is_ignored(function: &ItemFn) -> bool {
    function
        .attrs
        .iter()
        .any(|attr| attr.path().is_ident("ignore"))
}

struct ReaderVisitor {
    consts: Constants,
    read: Vec<(String, String)>,
}

impl<'ast> Visit<'ast> for ReaderVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        if is_ignored(node) {
            return;
        }
        self.consts.enter(node);
        visit::visit_item_fn(self, node);
        self.consts.leave();
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Some(name) = cassette_wrapper_name(node)
            && READ_ONLY_READERS.contains(&name.as_str())
            && let (Some(provider), Some(scenario)) = (
                node.args.first().and_then(|expr| self.consts.value(expr)),
                node.args
                    .iter()
                    .nth(1)
                    .and_then(|expr| self.consts.value(expr)),
            )
        {
            self.read.push((provider, scenario));
        }
        visit::visit_expr_call(self, node);
    }
}

/// How an in-source test of another crate reads a recorded cassette
/// without replaying it.
#[derive(Clone, Copy, Debug)]
pub enum InSourceReader {
    /// `include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "<root><path>"))`:
    /// the cassette at `<path>` under the cassette root, which is `root`
    /// relative to the reading crate's manifest directory.
    Included {
        /// The cassette root, relative to the manifest directory.
        root: &'static str,
    },
    /// A call `name(section, path)` that reads the cassette at `path`,
    /// relative to `provider`'s recordings.
    Path {
        /// The reader function.
        name: &'static str,
        /// The provider directory `path` is relative to.
        provider: &'static str,
    },
    /// rig-core's `raw_parity` rows, `Pair { provider, unary, streamed, .. }`,
    /// and its `Recorded::named(provider, path)` calls. Each path is
    /// `file.yaml` or `file.yaml#n` under `provider`'s recordings.
    ParityRow,
}

impl InSourceReader {
    /// Whether `source` holds an identifier every use of this reader names.
    fn named_in(self, source: &str) -> bool {
        match self {
            Self::Included { .. } => source.contains("include_str"),
            Self::Path { name, .. } => source.contains(name),
            Self::ParityRow => source.contains("Pair") || source.contains("Recorded"),
        }
    }
}

/// The `(provider, scenario)` pairs `source`, an in-source test module of
/// another crate, reads through `readers`. A source that names none of the
/// readers is not parsed.
///
/// As for [`read_only_cassettes`], each provider and path is a string
/// literal or a `&str` constant declared in the same file, a function's own
/// constants before the file's. The path argument of an
/// [`InSourceReader::Path`] call may also be:
/// - `f(value)`, where `f` is the file's one function of that name, it takes
///   one parameter, and its body is a `format!` of a literal that names only
///   that parameter;
/// - a parameter of the enclosing function, itself the file's one function
///   of its name, that the function never binds again, or `f(parameter)`.
///   It reads the value that parameter takes at each same-file call
///   `function(..)` passing a literal, a constant or `f(value)` there.
///
/// Nothing else counts: no other argument, no call in an `#[ignore]`d
/// function, no path that leaves the cassette root or names no `.yaml`
/// file, and no macro other than `include_str!` is looked into.
pub fn in_source_cassettes(
    source: &str,
    readers: &[InSourceReader],
) -> Result<Vec<(String, String)>, ScenarioError> {
    if !readers.iter().any(|reader| reader.named_in(source)) {
        return Ok(Vec::new());
    }
    let syntax = syn::parse_file(source)?;
    let mut functions = FunctionCollector::default();
    functions.visit_file(&syntax);
    let mut visitor = InSourceVisitor {
        readers,
        consts: Constants::of(&syntax),
        templates: functions.templates(),
        unique: functions.unique(),
        enclosing: Vec::new(),
        calls: std::collections::BTreeMap::new(),
        forwarded: Vec::new(),
        read: Vec::new(),
    };
    visitor.visit_file(&syntax);
    let InSourceVisitor {
        calls,
        forwarded,
        mut read,
        ..
    } = visitor;
    for forward in forwarded {
        for arguments in calls.get(&forward.function).into_iter().flatten() {
            if let Some(Some(value)) = arguments.get(forward.index)
                && let Some(path) = match &forward.template {
                    Some(template) => template.fill(value),
                    None => Some(value.clone()),
                }
                && let Some(cassette) = cassette_at(forward.provider, &path)
            {
                read.push(cassette);
            }
        }
    }
    Ok(read)
}

/// A same-file function whose body is `format!` of a literal naming only
/// its one parameter.
#[derive(Clone)]
struct Template {
    parameter: String,
    literal: String,
}

impl Template {
    fn of(function: &ItemFn) -> Option<Self> {
        let mut inputs = function.sig.inputs.iter();
        let (Some(syn::FnArg::Typed(input)), None) = (inputs.next(), inputs.next()) else {
            return None;
        };
        let parameter = plain_binding(&input.pat)?;
        let body = match function.block.stmts.as_slice() {
            [syn::Stmt::Expr(Expr::Macro(body), None)] => &body.mac,
            [syn::Stmt::Macro(body)] if body.semi_token.is_none() => &body.mac,
            _ => return None,
        };
        if !body.path.is_ident("format") {
            return None;
        }
        let literal = body.parse_body::<syn::LitStr>().ok()?.value();
        let template = Self { parameter, literal };
        template.fill("").map(|_| template)
    }

    /// The literal with each `{parameter}` replaced by `value`, or nothing
    /// when it holds any other placeholder.
    fn fill(&self, value: &str) -> Option<String> {
        let placeholder = format!("{{{}}}", self.parameter);
        let mut filled = String::new();
        let mut rest = self.literal.as_str();
        while let Some(at) = rest.find(['{', '}']) {
            filled.push_str(&rest[..at]);
            let tail = &rest[at..];
            if let Some(after) = tail.strip_prefix("{{") {
                filled.push('{');
                rest = after;
            } else if let Some(after) = tail.strip_prefix("}}") {
                filled.push('}');
                rest = after;
            } else if let Some(after) = tail.strip_prefix(placeholder.as_str()) {
                filled.push_str(value);
                rest = after;
            } else {
                return None;
            }
        }
        filled.push_str(rest);
        Some(filled)
    }
}

/// The name a pattern binds, when it is a plain immutable binding.
fn plain_binding(pattern: &syn::Pat) -> Option<String> {
    match pattern {
        syn::Pat::Ident(binding)
            if binding.by_ref.is_none()
                && binding.mutability.is_none()
                && binding.subpat.is_none() =>
        {
            Some(binding.ident.to_string())
        }
        _ => None,
    }
}

/// Every function of the file, nested ones included, by name.
#[derive(Default)]
struct FunctionCollector {
    functions: std::collections::BTreeMap<String, Vec<Option<Template>>>,
}

impl FunctionCollector {
    fn unique(&self) -> std::collections::BTreeSet<String> {
        self.functions
            .iter()
            .filter(|(_, definitions)| definitions.len() == 1)
            .map(|(name, _)| name.clone())
            .collect()
    }

    fn templates(&self) -> std::collections::BTreeMap<String, Template> {
        self.functions
            .iter()
            .filter_map(|(name, definitions)| match definitions.as_slice() {
                [Some(template)] => Some((name.clone(), template.clone())),
                _ => None,
            })
            .collect()
    }
}

impl<'ast> Visit<'ast> for FunctionCollector {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.functions
            .entry(node.sig.ident.to_string())
            .or_default()
            .push(Template::of(node));
        visit::visit_item_fn(self, node);
    }
}

/// The names a function's body binds, its nested functions aside.
#[derive(Default)]
struct BindingCollector {
    names: std::collections::BTreeSet<String>,
}

impl<'ast> Visit<'ast> for BindingCollector {
    fn visit_pat_ident(&mut self, node: &'ast syn::PatIdent) {
        self.names.insert(node.ident.to_string());
        visit::visit_pat_ident(self, node);
    }

    fn visit_item_fn(&mut self, _: &'ast ItemFn) {}
}

/// A reader's path that is a parameter of its enclosing function, awaiting
/// that function's same-file calls.
struct Forwarded {
    function: String,
    index: usize,
    template: Option<Template>,
    provider: &'static str,
}

/// An enclosing function: its name, when it is the file's only function of
/// that name, and each parameter that is a plain binding its body never
/// binds again.
struct Enclosing {
    name: Option<String>,
    parameters: Vec<Option<String>>,
}

struct InSourceVisitor<'a> {
    readers: &'a [InSourceReader],
    consts: Constants,
    templates: std::collections::BTreeMap<String, Template>,
    unique: std::collections::BTreeSet<String>,
    enclosing: Vec<Enclosing>,
    /// The arguments of each same-file call by a plain function name, as
    /// [`Self::value`] resolves them.
    calls: std::collections::BTreeMap<String, Vec<Vec<Option<String>>>>,
    forwarded: Vec<Forwarded>,
    read: Vec<(String, String)>,
}

impl InSourceVisitor<'_> {
    /// A literal, a constant in scope, or a template applied to either.
    fn value(&self, expr: &Expr) -> Option<String> {
        self.consts.value(expr).or_else(|| {
            let (template, argument) = self.template_call(expr)?;
            template.fill(&self.consts.value(argument)?)
        })
    }

    /// `f(argument)` for a same-file template `f`.
    fn template_call<'e>(&self, expr: &'e Expr) -> Option<(&Template, &'e Expr)> {
        let Expr::Call(call) = unwrapped(expr) else {
            return None;
        };
        let Expr::Path(function) = call.func.as_ref() else {
            return None;
        };
        let template = self
            .templates
            .get(&function.path.get_ident()?.to_string())?;
        let mut arguments = call.args.iter();
        match (arguments.next(), arguments.next()) {
            (Some(argument), None) => Some((template, argument)),
            _ => None,
        }
    }

    /// The index of the enclosing function's parameter `expr` names.
    fn parameter(&self, expr: &Expr) -> Option<(String, usize)> {
        let Expr::Path(path) = unwrapped(expr) else {
            return None;
        };
        let name = path.path.get_ident()?.to_string();
        let enclosing = self.enclosing.last()?;
        let index = enclosing
            .parameters
            .iter()
            .position(|parameter| parameter.as_deref() == Some(name.as_str()))?;
        Some((enclosing.name.clone()?, index))
    }

    fn path_read(&mut self, provider: &'static str, path: &Expr) {
        if let Some(value) = self.value(path) {
            self.read.extend(cassette_at(provider, &value));
            return;
        }
        let (template, argument) = match self.template_call(path) {
            Some((template, argument)) => (Some(template.clone()), argument),
            None => (None, path),
        };
        if let Some((function, index)) = self.parameter(argument) {
            self.forwarded.push(Forwarded {
                function,
                index,
                template,
                provider,
            });
        }
    }

    fn parity_read(&mut self, provider: &Expr, paths: &[&Expr]) {
        let Some(provider) = self.consts.value(provider) else {
            return;
        };
        for path in paths {
            if let Some(path) = self.consts.value(path)
                && let Some(file) = parity_file(&path)
            {
                self.read.extend(cassette_at(&provider, file));
            }
        }
    }

    fn reads_parity_rows(&self) -> bool {
        self.readers
            .iter()
            .any(|reader| matches!(reader, InSourceReader::ParityRow))
    }
}

impl<'ast> Visit<'ast> for InSourceVisitor<'_> {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        if is_ignored(node) {
            return;
        }
        let mut bound = BindingCollector::default();
        visit::visit_block(&mut bound, &node.block);
        let name = node.sig.ident.to_string();
        self.enclosing.push(Enclosing {
            name: self.unique.contains(&name).then_some(name),
            parameters: node
                .sig
                .inputs
                .iter()
                .map(|input| match input {
                    syn::FnArg::Typed(input) => {
                        plain_binding(&input.pat).filter(|name| !bound.names.contains(name))
                    }
                    syn::FnArg::Receiver(_) => None,
                })
                .collect(),
        });
        self.consts.enter(node);
        visit::visit_item_fn(self, node);
        self.consts.leave();
        self.enclosing.pop();
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(function) = node.func.as_ref()
            && let Some(name) = function.path.get_ident()
        {
            let arguments = node.args.iter().map(|expr| self.value(expr)).collect();
            self.calls
                .entry(name.to_string())
                .or_default()
                .push(arguments);
        }
        let arguments: Vec<&Expr> = node.args.iter().collect();
        if let (Some(name), [_, path]) = (cassette_wrapper_name(node), arguments.as_slice()) {
            let providers: Vec<&'static str> = self
                .readers
                .iter()
                .filter_map(|reader| match reader {
                    InSourceReader::Path {
                        name: reader,
                        provider,
                    } if *reader == name => Some(*provider),
                    _ => None,
                })
                .collect();
            for provider in providers {
                self.path_read(provider, path);
            }
        }
        if let ([provider, path], Expr::Path(function)) = (arguments.as_slice(), node.func.as_ref())
            && self.reads_parity_rows()
        {
            let mut segments = function.path.segments.iter().rev();
            if let (Some(method), Some(owner)) = (segments.next(), segments.next())
                && method.ident == "named"
                && owner.ident == "Recorded"
            {
                self.parity_read(provider, &[path]);
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_struct(&mut self, node: &'ast syn::ExprStruct) {
        if self.reads_parity_rows()
            && node
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "Pair")
        {
            let field = |wanted: &str| {
                node.fields.iter().find_map(|field| match &field.member {
                    syn::Member::Named(name) if name == wanted => Some(&field.expr),
                    _ => None,
                })
            };
            if let (Some(provider), Some(unary), Some(streamed)) =
                (field("provider"), field("unary"), field("streamed"))
            {
                self.parity_read(provider, &[unary, streamed]);
            }
        }
        visit::visit_expr_struct(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        for reader in self.readers {
            if let InSourceReader::Included { root } = reader
                && node.path.is_ident("include_str")
                && let Some(path) = included_path(node)
                && let Some(path) = path.strip_prefix(root)
            {
                self.read.extend(cassette_at("", path));
            }
        }
        visit::visit_macro(self, node);
    }
}

/// The literal `include_str!(concat!(env!("CARGO_MANIFEST_DIR"), literal))`
/// names after the manifest directory.
fn included_path(include: &syn::Macro) -> Option<String> {
    let Expr::Macro(concat) = include.parse_body::<Expr>().ok()? else {
        return None;
    };
    if !concat.mac.path.is_ident("concat") {
        return None;
    }
    let parts = concat
        .mac
        .parse_body_with(syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated)
        .ok()?;
    let parts: Vec<&Expr> = parts.iter().collect();
    let [
        Expr::Macro(env),
        Expr::Lit(ExprLit {
            lit: Lit::Str(path),
            ..
        }),
    ] = parts.as_slice()
    else {
        return None;
    };
    let variable = env.mac.parse_body::<syn::LitStr>().ok()?;
    (env.mac.path.is_ident("env") && variable.value() == "CARGO_MANIFEST_DIR").then(|| path.value())
}

/// A parity row's file: `file.yaml`, or `file.yaml#n` without `#n`.
fn parity_file(path: &str) -> Option<&str> {
    match path.split_once('#') {
        Some((file, interaction))
            if !interaction.is_empty() && interaction.bytes().all(|b| b.is_ascii_digit()) =>
        {
            Some(file)
        }
        Some(_) => None,
        None => Some(path),
    }
}

/// The `(provider, scenario)` of the cassette at `path`, relative to `base`
/// under the cassette root, when it stays under the root and names a
/// `.yaml` file inside a provider directory.
fn cassette_at(base: &str, path: &str) -> Option<(String, String)> {
    let mut segments: Vec<&str> = base
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    for segment in path.split('/') {
        match segment {
            "" => return None,
            "." => {}
            ".." => {
                segments.pop()?;
            }
            segment => segments.push(segment),
        }
    }
    let (file, directories) = segments.split_last()?;
    let (provider, directories) = directories.split_first()?;
    let stem = file.strip_suffix(".yaml").filter(|stem| !stem.is_empty())?;
    let scenario = directories
        .iter()
        .copied()
        .chain([stem])
        .collect::<Vec<_>>()
        .join("/");
    Some(((*provider).to_owned(), scenario))
}

fn cassette_scenario_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(scenario),
            ..
        }) => Some(scenario.value()),
        Expr::Call(call) if is_cassette_spec_new(call) => call.args.first().and_then(|expr| {
            let Expr::Lit(ExprLit {
                lit: Lit::Str(scenario),
                ..
            }) = expr
            else {
                return None;
            };

            Some(scenario.value())
        }),
        Expr::MethodCall(method_call) => cassette_scenario_value(&method_call.receiver),
        Expr::Paren(paren) => cassette_scenario_value(&paren.expr),
        _ => None,
    }
}

fn is_cassette_spec_new(call: &ExprCall) -> bool {
    let Expr::Path(path) = call.func.as_ref() else {
        return false;
    };

    let mut segments = path.path.segments.iter().rev();
    matches!(
        (segments.next(), segments.next()),
        (Some(method), Some(receiver))
            if method.ident == "new" && receiver.ident == "CassetteSpec"
    )
}

fn cassette_wrapper_name(node: &ExprCall) -> Option<String> {
    let Expr::Path(path) = node.func.as_ref() else {
        return None;
    };

    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

#[cfg(test)]
mod tests;
