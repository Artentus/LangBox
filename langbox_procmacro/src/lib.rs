//! Proc-macros for langbox
//!
//! This crate cannot be used stand-alone.
#![deny(missing_docs)]

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::*;
use syn::punctuated::Punctuated;
use syn::*;

fn ident_to_expr(ident: Ident) -> Expr {
    let mut segments = Punctuated::new();
    segments.push(PathSegment {
        ident,
        arguments: PathArguments::None,
    });

    let path = ExprPath {
        attrs: Vec::new(),
        qself: None,
        path: Path {
            leading_colon: None,
            segments,
        },
    };

    Expr::Path(path)
}

enum Combinator {
    Always,
    Parser(Expr),
    Optional(Box<Combinator>),
    OrDefault(Box<Combinator>),
    Many(Box<Combinator>),
    Many1(Box<Combinator>),
    MapTo {
        c: Box<Combinator>,
        value: Expr,
    },
    Map {
        c: Box<Combinator>,
        f: Expr,
    },
    MapError {
        c: Box<Combinator>,
        f: Expr,
    },
    Require {
        c: Box<Combinator>,
        f: Expr,
    },
    AndThen {
        lhs: Box<Combinator>,
        rhs: Box<Combinator>,
    },
    Prefix {
        lhs: Box<Combinator>,
        rhs: Box<Combinator>,
    },
    Suffix {
        lhs: Box<Combinator>,
        rhs: Box<Combinator>,
    },
    OrElse {
        lhs: Box<Combinator>,
        rhs: Box<Combinator>,
    },
    Parenthesized(Box<Combinator>),
}

fn parse_paren(input: ParseStream) -> Result<Option<Combinator>> {
    let lookahead = input.lookahead1();
    if lookahead.peek(token::Paren) {
        let content;
        let _ = parenthesized!(content in input);
        let c: Combinator = content.parse()?;

        if content.is_empty() {
            Ok(Some(Combinator::Parenthesized(Box::new(c))))
        } else {
            Err(content.error("unexpected token"))
        }
    } else {
        Ok(None)
    }
}

fn parse_leaf(input: ParseStream) -> Result<Option<Combinator>> {
    match parse_paren(input)? {
        Some(c) => Ok(Some(c)),
        None => {
            let lookahead = input.lookahead1();
            if lookahead.peek(Ident) {
                let ident = input.parse::<Ident>()?;
                Ok(Some(Combinator::Parser(ident_to_expr(ident))))
            } else if lookahead.peek(token::Brace) {
                let content;
                let _ = braced!(content in input);
                let expr = content.parse::<Expr>()?;

                Ok(Some(Combinator::Parser(expr)))
            } else {
                Ok(None)
            }
        }
    }
}

macro_rules! next {
    ($input:ident => if [$t:tt] $on_true:block $(else if [$ts:tt] $on_trues:block)* else $on_false:block) => {{
        let lookahead = $input.lookahead1();
        if lookahead.peek(Token![$t]) {
            let _ = $input.parse::<Token![$t]>()?;
            $on_true
        } $(else if lookahead.peek(Token![$ts]) {
            let _ = $input.parse::<Token![$ts]>()?;
            $on_trues
        })* else {
            $on_false
        }
    }};
}

fn parse_quantity(input: ParseStream) -> Result<Option<Combinator>> {
    next!(input => if [?] {
        Ok(if let Some(c) = parse_leaf(input)? {
            Some(Combinator::Optional(Box::new(c)))
        } else {
            None
        })
    } else if [%] {
        Ok(if let Some(c) = parse_leaf(input)? {
            Some(Combinator::OrDefault(Box::new(c)))
        } else {
            None
        })
    } else if [*] {
        Ok(if let Some(c) = parse_leaf(input)? {
            Some(Combinator::Many(Box::new(c)))
        } else {
            None
        })
    } else if [+] {
        Ok(if let Some(c) = parse_leaf(input)? {
            Some(Combinator::Many1(Box::new(c)))
        } else {
            None
        })
    } else {
        parse_leaf(input)
    })
}

fn parse_map(input: ParseStream) -> Result<Option<Combinator>> {
    if let Some(c) = parse_quantity(input)? {
        next!(input => if [=>] {
            let content;
            let _ = bracketed!(content in input);
            let expr = content.parse::<Expr>()?;

            Ok(Some(Combinator::MapTo { c: Box::new(c), value: expr }))
        } else if [->] {
            let content;
            let _ = bracketed!(content in input);
            let expr = content.parse::<Expr>()?;

            Ok(Some(Combinator::Map { c: Box::new(c), f: expr }))
        } else if [!] {
            next!(input => if [>] {
                let content;
                let _ = bracketed!(content in input);
                let expr = content.parse::<Expr>()?;

                Ok(Some(Combinator::MapError { c: Box::new(c), f: expr }))
            } else if [!] {
                let content;
                let _ = bracketed!(content in input);
                let expr = content.parse::<Expr>()?;

                Ok(Some(Combinator::Require { c: Box::new(c), f: expr }))
            } else {
                Err(input.error("invalid operator"))
            })
        } else {
            Ok(Some(c))
        })
    } else {
        Ok(None)
    }
}

macro_rules! rhs {
    ($input:ident, $lhs:ident, $parse_inner:ident => $comb:ident) => {
        if let Some(rhs) = $parse_inner(&$input)? {
            Ok((
                Combinator::$comb {
                    lhs: Box::new($lhs),
                    rhs: Box::new(rhs),
                },
                true,
            ))
        } else {
            Err($input.error("expected parser"))
        }
    };
}

fn parse_and_tail(input: ParseStream, lhs: Combinator) -> Result<(Combinator, bool)> {
    let forked_input = input.fork();

    next!(forked_input => if [<] {
        next!(forked_input => if [.] {
            next!(forked_input => if [>] {
                input.parse::<Token![<]>()?;
                input.parse::<Token![.]>()?;
                input.parse::<Token![>]>()?;

                rhs!(input, lhs, parse_map => AndThen)
            } else {
                input.parse::<Token![<]>()?;
                input.parse::<Token![.]>()?;

                rhs!(input, lhs, parse_map => Prefix)
            })
        } else {
            Ok((lhs, false))
        })
    } else if [.] {
        next!(forked_input => if [>] {
            input.parse::<Token![.]>()?;
            input.parse::<Token![>]>()?;

            rhs!(input, lhs, parse_map => Suffix)
        } else {
            Err(input.error("invalid operator"))
        })
    } else {
        Ok((lhs, false))
    })
}

fn parse_and(input: ParseStream) -> Result<Option<Combinator>> {
    if let Some(mut lhs) = parse_map(input)? {
        loop {
            let did_consume;
            (lhs, did_consume) = parse_and_tail(input, lhs)?;

            if !did_consume {
                return Ok(Some(lhs));
            }
        }
    } else {
        Ok(None)
    }
}

fn parse_or_tail(input: ParseStream, lhs: Combinator) -> Result<(Combinator, bool)> {
    next!(input => if [<] {
        next!(input => if [|] {
            next!(input => if [>] {
                rhs!(input, lhs, parse_and => OrElse)
            } else {
                Err(input.error("invalid operator"))
            })
        } else {
            Err(input.error("invalid operator"))
        })
    } else {
        Ok((lhs, false))
    })
}

fn parse_or(input: ParseStream) -> Result<Option<Combinator>> {
    if let Some(mut lhs) = parse_and(input)? {
        loop {
            let did_consume;
            (lhs, did_consume) = parse_or_tail(input, lhs)?;

            if !did_consume {
                return Ok(Some(lhs));
            }
        }
    } else {
        Ok(None)
    }
}

impl Parse for Combinator {
    fn parse(input: ParseStream) -> Result<Self> {
        let result = match parse_or(input)? {
            Some(c) => c,
            None => Self::Always,
        };

        if input.is_empty() {
            Ok(result)
        } else {
            Err(input.error("unexpected token"))
        }
    }
}

macro_rules! gen_unary {
    ($n:expr, $cident:ident, $c:ident, $v:ident, $fn:ident) => {{
        let p_fn = generate_parser(*$c, $n + 1);
        let p_ident = Ident::new(&format!("p{}", $n), Span::call_site());

        quote!({
            let #p_ident = #p_fn;
            #$cident::$fn(#p_ident, #$v)
        })
    }}
}

macro_rules! gen_binary {
    ($n:expr, $cident:ident, $lhs:ident, $rhs:ident, $fn:ident) => {{
        let lhs_fn = generate_parser(*$lhs, $n + 1);
        let rhs_fn = generate_parser(*$rhs, $n + 1);

        let lhs_ident = Ident::new(&format!("lhs{}", $n), Span::call_site());
        let rhs_ident = Ident::new(&format!("rhs{}", $n), Span::call_site());

        quote!({
            let #lhs_ident = #lhs_fn;
            let #rhs_ident = #rhs_fn;
            #$cident::$fn(#lhs_ident, #rhs_ident)
        })
    }};
}

fn generate_parser(c: Combinator, n: usize) -> syn::__private::TokenStream2 {
    use proc_macro_crate::*;
    use quote::__private::Span;

    let crate_name = crate_name("langbox").expect("langbox is not imported");
    let crate_ident = match crate_name {
        FoundCrate::Itself => Ident::new("crate", Span::call_site()),
        FoundCrate::Name(name) => Ident::new(&name, Span::call_site()),
    };

    match c {
        Combinator::Always => quote!(#crate_ident::_always()),
        Combinator::Parser(ident) => quote!(#ident),
        Combinator::Optional(c) => {
            let p_fn = generate_parser(*c, n + 1);
            let p_ident = Ident::new(&format!("p{}", n), Span::call_site());

            quote!({
                let #p_ident = #p_fn;
                #crate_ident::_opt(#p_ident)
            })
        }
        Combinator::OrDefault(c) => {
            let p_fn = generate_parser(*c, n + 1);
            let p_ident = Ident::new(&format!("p{}", n), Span::call_site());

            quote!({
                let #p_ident = #p_fn;
                #crate_ident::_or_default(#p_ident)
            })
        }
        Combinator::Many(c) => {
            let p_fn = generate_parser(*c, n + 1);
            let p_ident = Ident::new(&format!("p{}", n), Span::call_site());

            quote!({
                let #p_ident = #p_fn;
                #crate_ident::_many(#p_ident, true)
            })
        }
        Combinator::Many1(c) => {
            let p_fn = generate_parser(*c, n + 1);
            let p_ident = Ident::new(&format!("p{}", n), Span::call_site());

            quote!({
                let #p_ident = #p_fn;
                #crate_ident::_many(#p_ident, false)
            })
        }
        Combinator::MapTo { c, value } => gen_unary!(n, crate_ident, c, value, _map_to),
        Combinator::Map { c, f } => gen_unary!(n, crate_ident, c, f, _map),
        Combinator::MapError { c, f } => gen_unary!(n, crate_ident, c, f, _map_err),
        Combinator::Require { c, f } => gen_unary!(n, crate_ident, c, f, _require),
        Combinator::AndThen { lhs, rhs } => gen_binary!(n, crate_ident, lhs, rhs, _and_then),
        Combinator::Prefix { lhs, rhs } => gen_binary!(n, crate_ident, lhs, rhs, _prefix),
        Combinator::Suffix { lhs, rhs } => gen_binary!(n, crate_ident, lhs, rhs, _suffix),
        Combinator::OrElse { lhs, rhs } => gen_binary!(n, crate_ident, lhs, rhs, _or_else),
        Combinator::Parenthesized(c) => generate_parser(*c, n),
    }
}

/// Defines a parser using combinator DSL
///
/// # Operators
///
/// * `?a` - parse `a`; if no match return [`None`]
/// * `%a` - parse `a`; if no match return [`Default::default`]
/// * `*a` - parse `a` zero or more times
/// * `+a` - parse `a` one or more times
/// * `a <|> b` - parse `a`; if no match parse `b`
/// * `a <.> b` - parse `a`, then parse `b`
/// * `a <. b` - parse `a`, then parse `b`; return only `a`
/// * `a .> b` - parse `a`, then parse `b`; return only `b`
/// * `a!![f]` - parse `a`; if no match map the current input to an error using the function `f`
/// * `a=>[expr]` - parse `a` and return the expression `expr`
/// * `a->[f]` - parse `a` and map the result using the function `f`
/// * `a!>[f]` - parse `a`; on error map the error using the function `f`
/// * `{expr}` - evaluate `expr`, assuming it evaluates to a parser, and match it
///
/// # Precendence
///
/// Only one unary operator of the same level is allowed at a time.
///
/// Binary operators of the same level are left-associative.
///
/// 1. `( ... )`
/// 2. `{expr}`
/// 3. `?`, `%`, `*`, `+`
/// 4. `!!`, `=>`, `->`, `!>`
/// 5. `<.>`, `<.`, `.>`
/// 6. `<|>`
///
/// # Example
/// ```ignore
/// use langbox::*;
///
/// fn token(predicate: impl Fn(&JsonTokenKind) -> bool + Copy)
///     -> impl Parser<JsonTokenKind, JsonTokenKind, String> {
///     // ...
/// }
///
/// macro_rules! token {
///     ($kind:ident) => {
///         token(|kind| {
///             if let JsonTokenKind::$kind = kind {
///                 true
///             } else {
///                 false
///             }
///         })
///     };
/// }
///
/// fn bool() -> impl Parser<JsonTokenKind, bool, String> {
///     parser!({token!(False)}=>[false] <|> {token!(True)}=>[true])
/// }
/// ```
#[proc_macro]
pub fn parser(input: TokenStream) -> TokenStream {
    let c = parse_macro_input!(input as Combinator);
    generate_parser(c, 0).into()
}

struct Choice {
    parsers: Punctuated<Expr, Token![,]>,
}

impl Parse for Choice {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            parsers: input.parse_terminated(Expr::parse)?,
        })
    }
}

fn generate_choice(c: &Choice, crate_ident: Ident, n: usize) -> syn::__private::TokenStream2 {
    use quote::__private::Span;

    if n < c.parsers.len() {
        let p_fn = &c.parsers[n];
        let p_ident = Ident::new(&format!("p{}", n), Span::call_site());
        let inner = generate_choice(c, crate_ident.clone(), n + 1);

        quote!({
            let #p_ident = #p_fn;
            match #p_ident.run(input) {
                #crate_ident::ParseResult::Match(v) => #crate_ident::ParseResult::Match(v),
                #crate_ident::ParseResult::NoMatch => #inner,
                #crate_ident::ParseResult::Err(err) => return #crate_ident::ParseResult::Err(err),
            }
        })
    } else {
        quote!(#crate_ident::ParseResult::NoMatch)
    }
}

/// Defines a parser as a choice of other parsers
///
/// All parsers must have the same input, return and error types. Parsers are attempted to match in argument order.
/// The resulting parser returns the result of the first parser that matched.
///
/// # Example
/// ```ignore
/// use langbox::*;
///
/// fn jnull() -> impl Parser<JsonTokenKind, JsonValue, String> {
///     // ...
/// }
///
/// fn jbool() -> impl Parser<JsonTokenKind, JsonValue, String> {
///     // ...
/// }
///
/// fn jnumber() -> impl Parser<JsonTokenKind, JsonValue, String> {
///     // ...
/// }
///
/// fn jstring() -> impl Parser<JsonTokenKind, JsonValue, String> {
///     // ...
/// }
///
/// fn jarray() -> impl Parser<JsonTokenKind, JsonValue, String> {
///     // ...
/// }
///
/// fn jobject() -> impl Parser<JsonTokenKind, JsonValue, String> {
///     // ...
/// }
///
/// fn jvalue() -> impl Parser<JsonTokenKind, JsonValue, String> {
///     choice!(jnull(), jbool(), jnumber(), jstring(), jarray(), jobject())
/// }
/// ```
#[proc_macro]
pub fn choice(input: TokenStream) -> TokenStream {
    use proc_macro_crate::*;
    use quote::__private::Span;

    let crate_name = crate_name("langbox").expect("langbox is not imported");
    let crate_ident = match crate_name {
        FoundCrate::Itself => Ident::new("crate", Span::call_site()),
        FoundCrate::Name(name) => Ident::new(&name, Span::call_site()),
    };

    let c = parse_macro_input!(input as Choice);
    let body = generate_choice(&c, crate_ident.clone(), 0);

    quote!(
        #crate_ident::_constrain_parse_fn(move |input| {
            #body
        })
    )
    .into()
}

struct Sequence {
    parsers: Punctuated<Expr, Token![,]>,
}

impl Parse for Sequence {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            parsers: input.parse_terminated(Expr::parse)?,
        })
    }
}

fn generate_sequence(
    s: &Sequence,
    crate_ident: Ident,
    n: usize,
    mut vals: Vec<Ident>,
    mut spans: Vec<Ident>,
) -> syn::__private::TokenStream2 {
    use quote::__private::Span;

    if n < s.parsers.len() {
        let p_fn = &s.parsers[n];
        let p_ident = Ident::new(&format!("p{}", n), Span::call_site());

        let v_ident = Ident::new(&format!("v{}", n), Span::call_site());
        let s_ident = Ident::new(&format!("s{}", n), Span::call_site());

        vals.push(v_ident.clone());
        spans.push(s_ident.clone());
        let inner = generate_sequence(s, crate_ident.clone(), n + 1, vals, spans);

        quote!({
            let #p_ident = #p_fn;
            let #crate_ident::ParsedValue {
                value: #v_ident,
                span: #s_ident,
                remaining,
            } = #p_ident.run(remaining)?;
            #inner
        })
    } else {
        quote!({#crate_ident::ParseResult::Match(#crate_ident::ParsedValue {
            value: (#(#vals),*),
            span: #crate_ident::_join_spans(&[input.empty_span(), #(#spans),*]),
            remaining,
        })})
    }
}

/// Defines a parser as a sequence of other parsers
///
/// All parsers must have the same input and error types. Parsers are matched in argument order.
///
/// The resulting parser returns a flat tuple of all results of the parsers in the sequence, in order.
#[proc_macro]
pub fn sequence(input: TokenStream) -> TokenStream {
    use proc_macro_crate::*;
    use quote::__private::Span;

    let crate_name = crate_name("langbox").expect("langbox is not imported");
    let crate_ident = match crate_name {
        FoundCrate::Itself => Ident::new("crate", Span::call_site()),
        FoundCrate::Name(name) => Ident::new(&name, Span::call_site()),
    };

    let s = parse_macro_input!(input as Sequence);
    if s.parsers.len() == 0 {
        quote!(#crate_ident::_always()).into()
    } else {
        let body = generate_sequence(
            &s,
            crate_ident.clone(),
            0,
            Vec::with_capacity(s.parsers.len()),
            Vec::with_capacity(s.parsers.len()),
        );

        quote!(
            #crate_ident::_constrain_parse_fn(move |input| {
                let remaining = input;
                #body
            })
        )
        .into()
    }
}
