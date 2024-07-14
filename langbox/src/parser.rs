use crate::lexer::Token;
use crate::{TextPosition, TextSpan};
use std::convert::Infallible;
use std::ops::{ControlFlow, FromResidual, Try};

/// A stream of syntax tokens
///
/// # Example
/// ```ignore
/// use langbox::*;
///
/// let lexer: Lexer<_, _> = /* ... */;
/// let tokens = lexer.collect::<Vec<_>>();
/// let stream = TokenStream::new(&tokens);
/// ```
pub struct TokenStream<'a, Kind> {
    tokens: &'a [Token<Kind>],
    pos: usize,
}

impl<Kind> Clone for TokenStream<'_, Kind> {
    fn clone(&self) -> Self {
        Self {
            tokens: self.tokens,
            pos: self.pos,
        }
    }
}

impl<Kind> Copy for TokenStream<'_, Kind> {}

impl<'a, Kind> TokenStream<'a, Kind> {
    /// Creates a new token stream.
    #[inline]
    pub fn new(tokens: &'a [Token<Kind>]) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Gets the next token in the stream.
    #[inline]
    pub fn peek(&self) -> Option<&'a Token<Kind>> {
        self.tokens.get(self.pos)
    }

    /// Advances the stream by one token.
    #[inline]
    pub fn advance(&self) -> Self {
        Self {
            tokens: self.tokens,
            pos: (self.pos + 1).min(self.tokens.len()),
        }
    }

    /// Gets an empty span at the current position.
    pub fn empty_span(&self) -> TextSpan {
        let pos = match self.peek() {
            Some(t) => t.span.start_pos(),
            None => match self.tokens.iter().last() {
                Some(t) => t.span.end_pos(),
                None => TextPosition::NONE,
            },
        };

        TextSpan {
            file_id: pos.file_id,
            start_byte_offset: pos.byte_offset,
            end_byte_offset: pos.byte_offset,
        }
    }

    /// Returns the tokens that have already been consumed.
    #[inline]
    pub fn consumed(&self) -> &'a [Token<Kind>] {
        &self.tokens[..self.pos]
    }

    /// Returns the tokens that are remaining in the stream.
    #[inline]
    pub fn remaining(&self) -> &'a [Token<Kind>] {
        &self.tokens[self.pos..]
    }
}

/// The result of running a parser
#[must_use]
pub enum ParseResult<'a, TokenKind, T, E> {
    /// The input matched the parser pattern.
    Match {
        /// The value produced by the parser
        value: T,
        /// The remaining token stream
        remaining: TokenStream<'a, TokenKind>,
    },
    /// The input did not match the parser pattern.
    NoMatch,
    /// The input matched the parser pattern but was malformed or invalid.
    Err(E),
}

impl<'a, TokenKind, T, E> ParseResult<'a, TokenKind, T, E> {
    /// Maps a `ParseResult<'a, TokenKind, T, E>` to `ParseResult<'a, TokenKind, U, E>` by applying
    /// a function to a contained [`ParseResult::Match`] value, leaving an [`ParseResult::NoMatch`]
    /// and [`ParseResult::Err`] value untouched.
    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> ParseResult<'a, TokenKind, U, E> {
        match self {
            Self::Match { value, remaining } => ParseResult::Match {
                value: f(value),
                remaining,
            },
            Self::NoMatch => ParseResult::NoMatch,
            Self::Err(err) => ParseResult::Err(err),
        }
    }

    /// Maps a `ParseResult<'a, TokenKind, T, E>` to `ParseResult<'a, TokenKind, T, F>` by
    /// applying a function to a contained [`ParseResult::Err`] value, leaving
    /// an [`ParseResult::Match`] and [`ParseResult::NoMatch`] value untouched.
    #[inline]
    pub fn map_err<F>(self, f: impl FnOnce(E) -> F) -> ParseResult<'a, TokenKind, T, F> {
        match self {
            Self::Match { value, remaining } => ParseResult::Match { value, remaining },
            Self::NoMatch => ParseResult::NoMatch,
            Self::Err(err) => ParseResult::Err(f(err)),
        }
    }
}

impl<'a, TokenKind, T, E> Try for ParseResult<'a, TokenKind, T, E> {
    type Output = (T, TokenStream<'a, TokenKind>);
    type Residual = ParseResult<'a, TokenKind, Infallible, E>;

    #[inline]
    fn from_output((value, remaining): Self::Output) -> Self {
        Self::Match { value, remaining }
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Self::Match { value, remaining } => ControlFlow::Continue((value, remaining)),
            Self::NoMatch => ControlFlow::Break(ParseResult::NoMatch),
            Self::Err(err) => ControlFlow::Break(ParseResult::Err(err)),
        }
    }
}

impl<'a, TokenKind, T, E> FromResidual for ParseResult<'a, TokenKind, T, E> {
    #[inline]
    fn from_residual(residual: <Self as Try>::Residual) -> Self {
        match residual {
            ParseResult::Match { .. } => unreachable!(),
            ParseResult::NoMatch => Self::NoMatch,
            ParseResult::Err(err) => Self::Err(err),
        }
    }
}

// This function is a no-op but forces the compiler to constrain
// parser-closures with a higher-ranked lifetime bound.
#[doc(hidden)]
#[inline]
pub fn _constrain_parse_fn<TokenKind, T, E, F>(f: F) -> F
where
    for<'a> F: Fn(TokenStream<'a, TokenKind>) -> ParseResult<'a, TokenKind, T, E> + Copy,
{
    f
}

/// Defines a parser using a closure function.
///
/// # Examples
/// ```
/// use langbox::*;
///
/// // A parser that always matches, consumes no tokens and returns ()
/// fn always<TokenKind, E>() -> impl Parser<TokenKind, (), E> {
///     parse_fn!(|input| ParseResult::Match {
///         value: (),
///         remaining: input,
///     })
/// }
/// ```
///
/// ```
/// use langbox::*;
///
/// // A parser that returns None if the inner parser didn't match
/// fn opt<TokenKind, T, E>(
///     p: impl Parser<TokenKind, T, E>,
/// ) -> impl Parser<TokenKind, Option<T>, E> {
///     parse_fn!(|input| match p.run(input) {
///         ParseResult::Match { value, remaining } => ParseResult::Match { value: Some(value), remaining },
///         ParseResult::NoMatch => ParseResult::Match {
///             value: None,
///             remaining: input,
///         },
///         ParseResult::Err(err) => ParseResult::Err(err),
///     })
/// }
/// ```
#[macro_export]
macro_rules! parse_fn {
    (|$input:ident $(: $t:ty)?| $body:expr) => {
        $crate::_constrain_parse_fn(move |$input $(: $t)?| $body)
    };
    (|mut $input:ident $(: $t:ty)?| $body:expr) => {
        $crate::_constrain_parse_fn(move |mut $input $(: $t)?| $body)
    };
}

impl<TokenKind, T, E, F> Parser<TokenKind, T, E> for F
where
    for<'a> F: Fn(TokenStream<'a, TokenKind>) -> ParseResult<'a, TokenKind, T, E> + Copy,
{
    #[inline]
    fn run<'a>(&self, input: TokenStream<'a, TokenKind>) -> ParseResult<'a, TokenKind, T, E> {
        self(input)
    }
}

/// Transforms tokens from a token stream into structured data.
///
/// Parsers can be constructed using the [`parse_fn!`], [`parser!`],
/// [`choice!`] and [`sequence!`] macros.
pub trait Parser<TokenKind, T, E>: Copy {
    /// Runs the parser on the given input.
    fn run<'a>(&self, input: TokenStream<'a, TokenKind>) -> ParseResult<'a, TokenKind, T, E>;

    /// Returns an error if the parser doesn't match.
    fn require(
        self,
        f: impl Fn(TokenStream<TokenKind>) -> E + Copy,
    ) -> impl Parser<TokenKind, T, E> {
        parse_fn!(|input| match self.run(input) {
            ParseResult::Match { value, remaining } => ParseResult::Match { value, remaining },
            ParseResult::NoMatch => ParseResult::Err(f(input)),
            ParseResult::Err(err) => ParseResult::Err(err),
        })
    }

    /// Returns `None` if the parser doesn't match.
    fn opt(self) -> impl Parser<TokenKind, Option<T>, E> {
        parse_fn!(|input| match self.run(input) {
            ParseResult::Match { value, remaining } => ParseResult::Match {
                value: Some(value),
                remaining
            },
            ParseResult::NoMatch => ParseResult::Match {
                value: None,
                remaining: input,
            },
            ParseResult::Err(err) => ParseResult::Err(err),
        })
    }

    /// Returns a value if the parser doesn't match.
    fn or(self, val: T) -> impl Parser<TokenKind, T, E>
    where
        T: Copy,
    {
        parse_fn!(|input| match self.run(input) {
            ParseResult::Match { value, remaining } => ParseResult::Match { value, remaining },
            ParseResult::NoMatch => ParseResult::Match {
                value: val,
                remaining: input
            },
            ParseResult::Err(err) => ParseResult::Err(err),
        })
    }

    /// Returns a default value if the parser doesn't match.
    fn or_default(self) -> impl Parser<TokenKind, T, E>
    where
        T: Default,
    {
        parse_fn!(|input| match self.run(input) {
            ParseResult::Match { value, remaining } => ParseResult::Match { value, remaining },
            ParseResult::NoMatch => ParseResult::Match {
                value: T::default(),
                remaining: input,
            },
            ParseResult::Err(err) => ParseResult::Err(err),
        })
    }

    /// Matches either this parser or another.
    fn or_else(self, second: impl Parser<TokenKind, T, E>) -> impl Parser<TokenKind, T, E> {
        parse_fn!(|input| match self.run(input) {
            ParseResult::Match { value, remaining } => ParseResult::Match { value, remaining },
            ParseResult::NoMatch => second.run(input),
            ParseResult::Err(err) => ParseResult::Err(err),
        })
    }

    /// Matches another parser after this one.
    fn and_then<U>(
        self,
        second: impl Parser<TokenKind, U, E>,
    ) -> impl Parser<TokenKind, (T, U), E> {
        parse_fn!(|input| {
            let (v1, r1) = self.run(input)?;
            let (v2, r2) = second.run(r1)?;
            ParseResult::Match {
                value: (v1, v2),
                remaining: r2,
            }
        })
    }

    /// Maps the result of the parser on match to a value.
    fn map_to<U: Copy>(self, val: U) -> impl Parser<TokenKind, U, E> {
        parse_fn!(|input| self.run(input).map(|_| val))
    }

    /// Maps the result of the parser on match using a function.
    fn map<U>(self, f: impl Fn(T) -> U + Copy) -> impl Parser<TokenKind, U, E> {
        parse_fn!(|input| self.run(input).map(f))
    }

    /// Maps the error of the parser using a function.
    fn map_err<F>(self, f: impl Fn(E) -> F + Copy) -> impl Parser<TokenKind, T, F> {
        parse_fn!(|input| self.run(input).map_err(f))
    }

    /// Matches the parser multiple times.
    fn many(self, allow_empty: bool) -> impl Parser<TokenKind, Vec<T>, E> {
        parse_fn!(|mut input| {
            let mut result = Vec::new();

            loop {
                match self.run(input) {
                    ParseResult::Match { value, remaining } => {
                        result.push(value);
                        input = remaining;
                    }
                    ParseResult::NoMatch => break,
                    ParseResult::Err(err) => return ParseResult::Err(err),
                }
            }

            if allow_empty || (result.len() > 0) {
                ParseResult::Match {
                    value: result,
                    remaining: input,
                }
            } else {
                ParseResult::NoMatch
            }
        })
    }

    /// Matches the parser multiple times, separated by the given separator.
    fn sep_by<S>(
        self,
        sep: impl Parser<TokenKind, S, E>,
        allow_empty: bool,
        allow_trailing: bool,
    ) -> impl Parser<TokenKind, Vec<T>, E> {
        parse_fn!(|input| {
            match self.run(input) {
                ParseResult::Match { value, remaining } => {
                    let mut result = Vec::new();
                    result.push(value);

                    let mut input = remaining;

                    loop {
                        match sep.run(input) {
                            ParseResult::Match {
                                remaining: sep_remaining,
                                ..
                            } => match self.run(sep_remaining) {
                                ParseResult::Match { value, remaining } => {
                                    result.push(value);
                                    input = remaining;
                                }
                                ParseResult::NoMatch => {
                                    if allow_trailing {
                                        input = sep_remaining;
                                    }

                                    break;
                                }
                                ParseResult::Err(err) => return ParseResult::Err(err),
                            },
                            ParseResult::NoMatch => break,
                            ParseResult::Err(err) => return ParseResult::Err(err),
                        }
                    }

                    ParseResult::Match {
                        value: result,
                        remaining: input,
                    }
                }
                ParseResult::NoMatch => {
                    if allow_empty {
                        ParseResult::Match {
                            value: Vec::new(),
                            remaining: input,
                        }
                    } else {
                        ParseResult::NoMatch
                    }
                }
                ParseResult::Err(err) => ParseResult::Err(err),
            }
        })
    }

    /// Matches the parser exactly 'count' times.
    fn repeat(self, count: usize) -> impl Parser<TokenKind, Vec<T>, E> {
        parse_fn!(|mut input| {
            let mut result = Vec::with_capacity(count);

            for _ in 0..count {
                let (value, remaining) = self.run(input)?;
                result.push(value);
                input = remaining;
            }

            ParseResult::Match {
                value: result,
                remaining: input,
            }
        })
    }
}

/// A parser returning a tuple
pub trait TupleParser<TokenKind, U, V, E>: Parser<TokenKind, (U, V), E> {
    /// Returns the first matched value
    fn prefix(self) -> impl Parser<TokenKind, U, E> {
        parse_fn!(|input| self.run(input).map(|(val, _)| val))
    }

    /// Returns the second matched value
    fn suffix(self) -> impl Parser<TokenKind, V, E> {
        parse_fn!(|input| self.run(input).map(|(_, val)| val))
    }
}

impl<TokenKind, U, V, E, P> TupleParser<TokenKind, U, V, E> for P where
    P: Parser<TokenKind, (U, V), E>
{
}

/// Always matches.
pub fn always<TokenKind, E>() -> impl Parser<TokenKind, (), E> {
    parse_fn!(|input| ParseResult::Match {
        value: (),
        remaining: input,
    })
}

/// Matches the end of the token stream.
pub fn eof<TokenKind, E>() -> impl Parser<TokenKind, (), E> {
    parse_fn!(|input| if let Some(_) = input.peek() {
        ParseResult::NoMatch
    } else {
        ParseResult::Match {
            value: (),
            remaining: input,
        }
    })
}

/// Matches three parsers in sequence, returning only the second result.
pub fn between<TokenKind, T1, T2, T3, E>(
    prefix: impl Parser<TokenKind, T1, E>,
    p: impl Parser<TokenKind, T2, E>,
    suffix: impl Parser<TokenKind, T3, E>,
) -> impl Parser<TokenKind, T2, E> {
    parse_fn!(|input| {
        let (_, r1) = prefix.run(input)?;
        let (v2, r2) = p.run(r1)?;
        let (_, r3) = suffix.run(r2)?;
        ParseResult::Match {
            value: v2,
            remaining: r3,
        }
    })
}

pub use langbox_procmacro::{choice, parser, sequence};

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::lexer::test::*;
    use crate::lexer::whitespace_mode;
    use std::fmt;

    fn parse_test_token(kind: TestTokenKind) -> impl Parser<TestTokenKind, char, &'static str> {
        parse_fn!(|input: TokenStream<TestTokenKind>| {
            if let Some(next) = input.peek() {
                if next.kind == kind {
                    ParseResult::Match {
                        value: next.kind.to_char(),
                        remaining: input.advance(),
                    }
                } else {
                    ParseResult::NoMatch
                }
            } else {
                ParseResult::NoMatch
            }
        })
    }

    fn test_parser<T: fmt::Debug + Eq>(
        p: impl Parser<TestTokenKind, T, &'static str>,
        text: &'static str,
        expected_output: Option<T>,
    ) {
        let tokens = lex::<whitespace_mode::Keep>(text);
        let input = TokenStream::new(&tokens);

        let result = p.run(input);
        match result {
            ParseResult::Match { value, .. } => {
                if let Some(expected_output) = expected_output {
                    assert_eq!(expected_output, value);
                } else {
                    panic!("expected no match, but parser returned {value:?}");
                }
            }
            ParseResult::NoMatch => {
                if let Some(expected_output) = expected_output {
                    panic!("expected {expected_output:?}, but parser returned no match");
                }
            }
            ParseResult::Err(err) => {
                panic!("parser returned error: {err}");
            }
        }
    }

    #[test]
    fn empty() {
        test_parser(parser!(), "", Some(()));
    }

    #[test]
    fn single() {
        test_parser(parse_test_token(TestTokenKind::A), "a", Some('a'));
    }

    #[test]
    fn opt() {
        let p = parser!(?{parse_test_token(TestTokenKind::A)});

        test_parser(p, "", Some(None));
        test_parser(p, "a", Some(Some('a')));
        test_parser(p, "b", Some(None));
    }

    #[test]
    fn or_else() {
        let p =
            parser!({parse_test_token(TestTokenKind::A)} <|> {parse_test_token(TestTokenKind::B)});

        test_parser(p, "", None);
        test_parser(p, "a", Some('a'));
        test_parser(p, "b", Some('b'));

        test_parser(p, " a", None);
        test_parser(p, " b", None);
    }

    #[test]
    fn and_then() {
        let p =
            parser!({parse_test_token(TestTokenKind::A)} <.> {parse_test_token(TestTokenKind::B)});

        test_parser(p, "", None);
        test_parser(p, "a", None);
        test_parser(p, "b", None);
        test_parser(p, "ba", None);
        test_parser(p, "ab", Some(('a', 'b')));

        test_parser(p, " ab", None);
        test_parser(p, "a b", None);
    }

    #[test]
    fn prefix() {
        let p =
            parser!({parse_test_token(TestTokenKind::A)} <. {parse_test_token(TestTokenKind::B)});

        test_parser(p, "", None);
        test_parser(p, "a", None);
        test_parser(p, "b", None);
        test_parser(p, "ba", None);
        test_parser(p, "ab", Some('a'));

        test_parser(p, " ab", None);
        test_parser(p, "a b", None);
    }

    #[test]
    fn suffix() {
        let p =
            parser!({parse_test_token(TestTokenKind::A)} .> {parse_test_token(TestTokenKind::B)});

        test_parser(p, "", None);
        test_parser(p, "a", None);
        test_parser(p, "b", None);
        test_parser(p, "ba", None);
        test_parser(p, "ab", Some('b'));

        test_parser(p, " ab", None);
        test_parser(p, "a b", None);
    }

    #[test]
    fn many() {
        let p = parser!(*({parse_test_token(TestTokenKind::A)} <|> {parse_test_token(TestTokenKind::B)}));

        test_parser(p, "", Some(vec![]));
        test_parser(p, "a", Some(vec!['a']));
        test_parser(p, "b", Some(vec!['b']));

        test_parser(p, "abaaabb", Some(vec!['a', 'b', 'a', 'a', 'a', 'b', 'b']));

        test_parser(p, "abaa abb", Some(vec!['a', 'b', 'a', 'a']));
    }

    #[test]
    fn many1() {
        let p = parser!(+({parse_test_token(TestTokenKind::A)} <|> {parse_test_token(TestTokenKind::B)}));

        test_parser(p, "", None);
        test_parser(p, "a", Some(vec!['a']));
        test_parser(p, "b", Some(vec!['b']));

        test_parser(p, "abaaabb", Some(vec!['a', 'b', 'a', 'a', 'a', 'b', 'b']));

        test_parser(p, "abaa abb", Some(vec!['a', 'b', 'a', 'a']));
    }

    #[test]
    fn map_to() {
        let p = parser!({parse_test_token(TestTokenKind::A)}=>[false]);

        test_parser(p, "", None);
        test_parser(p, "a", Some(false));
    }

    #[test]
    fn map() {
        let p = parser!(({parse_test_token(TestTokenKind::A)} <|> {parse_test_token(TestTokenKind::B)})->[|c| c == 'b']);

        test_parser(p, "", None);
        test_parser(p, "a", Some(false));
        test_parser(p, "b", Some(true));
    }

    fn test_parser_error<T: fmt::Debug + Eq>(
        p: impl Parser<TestTokenKind, T, &'static str>,
        text: &'static str,
        expected_error: &'static str,
    ) {
        let tokens = lex::<whitespace_mode::Keep>(text);
        let input = TokenStream::new(&tokens);

        match p.run(input) {
            ParseResult::Match { value, .. } => {
                panic!("expected error, but parser returned {value:?}");
            }
            ParseResult::NoMatch => {
                panic!("expected error, but parser returned no match");
            }
            ParseResult::Err(err) => {
                assert_eq!(err, expected_error);
            }
        }
    }

    #[test]
    fn require() {
        let p = parser!({parse_test_token(TestTokenKind::A)}!![|_| "expected A"]);

        test_parser_error(p, "", "expected A");
        test_parser(p, "a", Some('a'));
        test_parser_error(p, "b", "expected A");
    }

    #[test]
    fn map_err() {
        let p = parser!(({parse_test_token(TestTokenKind::A)}!![|_| "wrong error"])!>[|_| "expected A"]);

        test_parser_error(p, "", "expected A");
        test_parser(p, "a", Some('a'));
        test_parser_error(p, "b", "expected A");
    }

    #[test]
    fn between() {
        let p = super::between(
            parse_test_token(TestTokenKind::B),
            parse_test_token(TestTokenKind::A),
            parse_test_token(TestTokenKind::B),
        );

        test_parser(p, "", None);
        test_parser(p, "a", None);
        test_parser(p, "b", None);
        test_parser(p, "ba", None);
        test_parser(p, "ab", None);
        test_parser(p, "bab", Some('a'));
    }

    #[test]
    fn sep_by() {
        let p = parse_test_token(TestTokenKind::A).sep_by(
            parse_test_token(TestTokenKind::B),
            true,
            false,
        );

        test_parser(p, "", Some(vec![]));
        test_parser(p, "a", Some(vec!['a']));
        test_parser(p, "ababa", Some(vec!['a', 'a', 'a']));
        test_parser(p, "baba", Some(vec![]));
    }

    #[test]
    fn sep_by_trailing() {
        let p = parse_test_token(TestTokenKind::A).sep_by(
            parse_test_token(TestTokenKind::B),
            true,
            true,
        );

        test_parser(p, "", Some(vec![]));
        test_parser(p, "b", Some(vec![]));
        test_parser(p, "a", Some(vec!['a']));
        test_parser(p, "ab", Some(vec!['a']));
        test_parser(p, "ababa", Some(vec!['a', 'a', 'a']));
        test_parser(p, "ababab", Some(vec!['a', 'a', 'a']));
        test_parser(p, "baba", Some(vec![]));
    }

    #[test]
    fn sep_by1() {
        let p = parse_test_token(TestTokenKind::A).sep_by(
            parse_test_token(TestTokenKind::B),
            false,
            false,
        );

        test_parser(p, "", None);
        test_parser(p, "a", Some(vec!['a']));
        test_parser(p, "ababa", Some(vec!['a', 'a', 'a']));
        test_parser(p, "baba", None);
    }

    #[test]
    fn sep_by1_trailing() {
        let p = parse_test_token(TestTokenKind::A).sep_by(
            parse_test_token(TestTokenKind::B),
            false,
            true,
        );

        test_parser(p, "", None);
        test_parser(p, "b", None);
        test_parser(p, "a", Some(vec!['a']));
        test_parser(p, "ab", Some(vec!['a']));
        test_parser(p, "ababa", Some(vec!['a', 'a', 'a']));
        test_parser(p, "ababab", Some(vec!['a', 'a', 'a']));
        test_parser(p, "baba", None);
    }

    #[test]
    fn repeat() {
        let p = parse_test_token(TestTokenKind::A).repeat(0);

        test_parser(p, "", Some(vec![]));
        test_parser(p, "a", Some(vec![]));

        let p = parse_test_token(TestTokenKind::A).repeat(3);

        test_parser(p, "", None);
        test_parser(p, "a", None);
        test_parser(p, "aaa", Some(vec!['a', 'a', 'a']));
        test_parser(p, "aaaa", Some(vec!['a', 'a', 'a']));
    }

    #[test]
    fn choice() {
        let p = choice!(
            parse_test_token(TestTokenKind::A),
            parse_test_token(TestTokenKind::B),
        );

        test_parser(p, "", None);
        test_parser(p, "a", Some('a'));
        test_parser(p, "b", Some('b'));

        test_parser(p, " a", None);
        test_parser(p, " b", None);
    }

    #[test]
    fn sequence() {
        let p = sequence!(
            parse_test_token(TestTokenKind::A),
            parse_test_token(TestTokenKind::B),
        );

        test_parser(p, "", None);
        test_parser(p, "a", None);
        test_parser(p, "b", None);
        test_parser(p, "ba", None);
        test_parser(p, "ab", Some(('a', 'b')));

        test_parser(p, " ab", None);
        test_parser(p, "a b", None);
    }

    #[test]
    fn eof() {
        test_parser(super::eof(), "", Some(()));
        test_parser(super::eof(), "a", None);

        let p = sequence!(parse_test_token(TestTokenKind::A), super::eof());

        test_parser(p, "a", Some(('a', ())));
    }
}
