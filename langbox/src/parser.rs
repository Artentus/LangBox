use super::*;
use std::ops::{ControlFlow, FromResidual, Try};

/// A stream of syntax tokens
///
/// # Example
/// ```
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
    /// Creates a new token stream
    #[inline]
    pub fn new(tokens: &'a [Token<Kind>]) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Gets the next token in the stream
    #[inline]
    pub fn peek(&self) -> Option<&'a Token<Kind>> {
        self.tokens.get(self.pos)
    }

    /// Advances the stream by one token
    #[inline]
    pub fn advance(&self) -> Self {
        Self {
            tokens: self.tokens,
            pos: (self.pos + 1).min(self.tokens.len()),
        }
    }

    /// Gets an empty span at the current position
    pub fn empty_span(&self) -> TextSpan {
        let pos = match self.peek() {
            Some(t) => t.span.start_pos,
            None => match self.tokens.iter().last() {
                Some(t) => t.span.end_pos,
                None => TextPosition {
                    file_id: FileId::NONE,
                    byte_offset: 0,
                    line: 0,
                    column: 0,
                },
            },
        };

        TextSpan {
            start_pos: pos,
            end_pos: pos,
        }
    }
}

/// The result of running a parser
#[must_use]
pub enum ParseResult<'a, TokenKind, T, E> {
    /// The input matched the parser pattern.
    Match {
        /// The value produced by the parser
        value: T,
        /// The span corresponding to all consumed tokens
        span: TextSpan,
        /// The remaining token stream
        remaining: TokenStream<'a, TokenKind>,
    },
    /// The input did not match the parser pattern.
    NoMatch,
    /// The input matched the parser pattern but was malformed or invalid.
    Err(E),
}

impl<'a, TokenKind, T, E> ParseResult<'a, TokenKind, T, E> {
    /// Maps a `ParseResult<TokenKind, T, E>` to `ParseResult<TokenKind, U, E>` by applying
    /// a function to a contained [`ParseResult::Match`] value, leaving an [`ParseResult::NoMatch`]
    /// and [`ParseResult::Err`] value untouched.
    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> ParseResult<'a, TokenKind, U, E> {
        match self {
            Self::Match {
                remaining,
                span,
                value,
            } => ParseResult::Match {
                remaining,
                span,
                value: f(value),
            },
            Self::NoMatch => ParseResult::NoMatch,
            Self::Err(err) => ParseResult::Err(err),
        }
    }

    /// Maps a `ParseResult<TokenKind, T, E>` to `ParseResult<TokenKind, T, F>` by
    /// applying a function to a contained [`ParseResult::Err`] value, leaving
    /// an [`ParseResult::Match`] and [`ParseResult::NoMatch`] value untouched.
    #[inline]
    pub fn map_err<F>(self, f: impl FnOnce(E) -> F) -> ParseResult<'a, TokenKind, T, F> {
        match self {
            Self::Match {
                remaining,
                span,
                value,
            } => ParseResult::Match {
                remaining,
                span,
                value,
            },
            Self::NoMatch => ParseResult::NoMatch,
            Self::Err(err) => ParseResult::Err(f(err)),
        }
    }

    /// Returns the contained [`ParseResult::Match`] or [`ParseResult::NoMatch`] value,
    /// consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`ParseResult::Err`], with a panic message including
    /// the passed message, and the content of the [`ParseResult::Err`].
    #[inline]
    #[track_caller]
    pub fn expect(self, msg: &str) -> InfallibleParseResult<'a, TokenKind, T>
    where
        E: Debug,
    {
        match self {
            Self::Match {
                value,
                span,
                remaining,
            } => InfallibleParseResult::Match {
                value,
                span,
                remaining,
            },
            Self::NoMatch => InfallibleParseResult::NoMatch,
            Self::Err(err) => panic!("{msg}: {err:?}"),
        }
    }

    /// Returns the contained [`ParseResult::Err`] value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`ParseResult::Match`] or [`ParseResult::NoMatch`],
    /// with a panic message including the passed message, and the content of the
    /// [`ParseResult::Match`].
    #[inline]
    #[track_caller]
    pub fn expect_err(self, msg: &str) -> E
    where
        T: Debug,
    {
        match self {
            Self::Match { value, .. } => panic!("{msg}: {value:?}"),
            Self::NoMatch => panic!("{msg}"),
            Self::Err(err) => err,
        }
    }

    /// Returns the contained [`ParseResult::Match`] or [`ParseResult::NoMatch`] value,
    /// consuming the `self` value.
    ///
    /// Because this function may panic, its use is generally discouraged.
    /// Instead, prefer to use pattern matching and handle the [`ParseResult::Err`]
    /// case explicitly, or call `unwrap_or_no_match`.
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`ParseResult::Err`], with a panic message provided
    /// by the [`ParseResult::Err`]'s value.
    #[inline]
    #[track_caller]
    pub fn unwrap(self) -> InfallibleParseResult<'a, TokenKind, T>
    where
        E: Debug,
    {
        match self {
            Self::Match {
                value,
                span,
                remaining,
            } => InfallibleParseResult::Match {
                value,
                span,
                remaining,
            },
            Self::NoMatch => InfallibleParseResult::NoMatch,
            Self::Err(err) => panic!("called `ParseResult::unwrap()` on an `Err` value: {err:?}"),
        }
    }

    /// Returns the contained [`ParseResult::Match`] or [`ParseResult::NoMatch`] value.
    #[inline]
    pub fn unwrap_or_no_match(self) -> InfallibleParseResult<'a, TokenKind, T> {
        match self {
            Self::Match {
                value,
                span,
                remaining,
            } => InfallibleParseResult::Match {
                value,
                span,
                remaining,
            },
            Self::NoMatch => InfallibleParseResult::NoMatch,
            Self::Err(_) => InfallibleParseResult::NoMatch,
        }
    }

    /// Returns the contained [`Err`] value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the value is a [`ParseResult::Match`] or [`ParseResult::NoMatch`] value,
    /// with a custom panic message provided by the [`ParseResult::Match`]'s value or
    /// [`ParseResult::NoMatch`].
    #[inline]
    #[track_caller]
    pub fn unwrap_err(self) -> E
    where
        T: Debug,
    {
        match self {
            Self::Match { value, .. } => panic!(
                "called `ParseResult::unwrap_err()` on a `ParseResult::Match` value: {value:?}"
            ),
            Self::NoMatch => {
                panic!("called `ParseResult::unwrap_err()` on a `ParseResult::NoMatch` value")
            }
            Self::Err(err) => err,
        }
    }
}

/// The result of running a parser
#[must_use]
pub enum InfallibleParseResult<'a, TokenKind, T> {
    /// The input matched the parser pattern.
    Match {
        /// The value produced by the parser
        value: T,
        /// The span corresponding to all consumed tokens
        span: TextSpan,
        /// The remaining token stream
        remaining: TokenStream<'a, TokenKind>,
    },
    /// The input did not match the parser pattern.
    NoMatch,
}

impl<'a, TokenKind, T> InfallibleParseResult<'a, TokenKind, T> {
    /// Maps an `InfallibleParseResult<TokenKind, T, E>` to `InfallibleParseResult<TokenKind, U, E>`
    /// by applying a function to a contained [`InfallibleParseResult::Match`] value, leaving an
    /// [`InfallibleParseResult::NoMatch`] value untouched.
    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> InfallibleParseResult<'a, TokenKind, U> {
        match self {
            Self::Match {
                remaining,
                span,
                value,
            } => InfallibleParseResult::Match {
                remaining,
                span,
                value: f(value),
            },
            Self::NoMatch => InfallibleParseResult::NoMatch,
        }
    }
}

#[doc(hidden)]
pub struct ParseResultResidual<E>(E);

impl<'a, TokenKind, T, E> Try for ParseResult<'a, TokenKind, T, E> {
    type Output = InfallibleParseResult<'a, TokenKind, T>;
    type Residual = ParseResultResidual<E>;

    fn from_output(output: Self::Output) -> Self {
        output.into()
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Self::Match {
                value,
                span,
                remaining,
            } => ControlFlow::Continue(InfallibleParseResult::Match {
                value,
                span,
                remaining,
            }),
            Self::NoMatch => ControlFlow::Continue(InfallibleParseResult::NoMatch),
            Self::Err(err) => ControlFlow::Break(ParseResultResidual(err)),
        }
    }
}

impl<'a, TokenKind, T, E> FromResidual for ParseResult<'a, TokenKind, T, E> {
    fn from_residual(residual: <Self as Try>::Residual) -> Self {
        Self::Err(residual.0)
    }
}

impl<'a, TokenKind, T, E> From<InfallibleParseResult<'a, TokenKind, T>>
    for ParseResult<'a, TokenKind, T, E>
{
    fn from(r: InfallibleParseResult<'a, TokenKind, T>) -> Self {
        match r {
            InfallibleParseResult::Match {
                value,
                span,
                remaining,
            } => Self::Match {
                value,
                span,
                remaining,
            },
            InfallibleParseResult::NoMatch => Self::NoMatch,
        }
    }
}

/// Transforms tokens from a token stream into structured data
///
/// Parsers can be constructed using the [`parse_fn!`], [`parser!`],
/// [`choice!`] and [`sequence!`] macros.
pub trait Parser<TokenKind, T, E>: Copy {
    /// Runs the parser on the given input
    fn run<'a>(&self, input: TokenStream<'a, TokenKind>) -> ParseResult<'a, TokenKind, T, E>;
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

/// Defines a parser using a closure function
///
/// # Examples
/// ```
/// // A parser that always matches, consumes no tokens and returns ()
/// fn always<TokenKind, E>() -> impl Parser<TokenKind, (), E> {
///     parse_fn!(|input| ParseResult::Match {
///         value: (),
///         span: input.empty_span(),
///         remaining: input,
///     })
/// }
/// ```
///
/// ```
/// // A parser that returns None if the inner parser didn't match
/// fn opt<TokenKind, T, E>(
///     p: impl Parser<TokenKind, T, E>,
/// ) -> impl Parser<TokenKind, Option<T>, E> {
///     parse_fn!(|input| match p.run(input)? {
///         InfallibleParseResult::Match {
///             value,
///             span,
///             remaining,
///         } => ParseResult::Match {
///             value: Some(value),
///             span,
///             remaining,
///         },
///         InfallibleParseResult::NoMatch => ParseResult::Match {
///             value: None,
///             span: input.empty_span(),
///             remaining: input,
///         },
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

#[doc(hidden)]
pub fn _always<TokenKind, E>() -> impl Parser<TokenKind, (), E> {
    parse_fn!(|input| ParseResult::Match {
        value: (),
        span: input.empty_span(),
        remaining: input,
    })
}

#[doc(hidden)]
pub fn _opt<TokenKind, T, E>(
    p: impl Parser<TokenKind, T, E>,
) -> impl Parser<TokenKind, Option<T>, E> {
    parse_fn!(|input| match p.run(input)? {
        InfallibleParseResult::Match {
            value,
            span,
            remaining,
        } => ParseResult::Match {
            value: Some(value),
            span,
            remaining,
        },
        InfallibleParseResult::NoMatch => ParseResult::Match {
            value: None,
            span: input.empty_span(),
            remaining: input,
        },
    })
}

#[doc(hidden)]
pub fn _or_default<TokenKind, T: Default, E>(
    p: impl Parser<TokenKind, T, E>,
) -> impl Parser<TokenKind, T, E> {
    parse_fn!(|input| match p.run(input)? {
        InfallibleParseResult::Match {
            value,
            span,
            remaining,
        } => ParseResult::Match {
            value,
            span,
            remaining,
        },
        InfallibleParseResult::NoMatch => ParseResult::Match {
            value: T::default(),
            span: input.empty_span(),
            remaining: input,
        },
    })
}

#[doc(hidden)]
pub fn _and_then<TokenKind, T1, T2, E>(
    first: impl Parser<TokenKind, T1, E>,
    second: impl Parser<TokenKind, T2, E>,
) -> impl Parser<TokenKind, (T1, T2), E> {
    parse_fn!(|input| match first.run(input)? {
        InfallibleParseResult::Match {
            value: v1,
            span: s1,
            remaining,
        } => match second.run(remaining)? {
            InfallibleParseResult::Match {
                value: v2,
                span: s2,
                remaining,
            } => ParseResult::Match {
                value: (v1, v2),
                span: s1.join(&s2),
                remaining,
            },
            InfallibleParseResult::NoMatch => ParseResult::NoMatch,
        },
        InfallibleParseResult::NoMatch => ParseResult::NoMatch,
    })
}

#[doc(hidden)]
pub fn _prefix<TokenKind, T1, T2, E>(
    first: impl Parser<TokenKind, T1, E>,
    second: impl Parser<TokenKind, T2, E>,
) -> impl Parser<TokenKind, T1, E> {
    parse_fn!(|input| match first.run(input)? {
        InfallibleParseResult::Match {
            value: v1,
            span: s1,
            remaining,
        } => match second.run(remaining)? {
            InfallibleParseResult::Match {
                span: s2,
                remaining,
                ..
            } => ParseResult::Match {
                value: v1,
                span: s1.join(&s2),
                remaining,
            },
            InfallibleParseResult::NoMatch => ParseResult::NoMatch,
        },
        InfallibleParseResult::NoMatch => ParseResult::NoMatch,
    })
}

#[doc(hidden)]
pub fn _suffix<TokenKind, T1, T2, E>(
    first: impl Parser<TokenKind, T1, E>,
    second: impl Parser<TokenKind, T2, E>,
) -> impl Parser<TokenKind, T2, E> {
    parse_fn!(|input| match first.run(input)? {
        InfallibleParseResult::Match {
            span: s1,
            remaining,
            ..
        } => match second.run(remaining)? {
            InfallibleParseResult::Match {
                value: v2,
                span: s2,
                remaining,
            } => ParseResult::Match {
                value: v2,
                span: s1.join(&s2),
                remaining,
            },
            InfallibleParseResult::NoMatch => ParseResult::NoMatch,
        },
        InfallibleParseResult::NoMatch => ParseResult::NoMatch,
    })
}

#[doc(hidden)]
pub fn _or_else<TokenKind, T, E>(
    first: impl Parser<TokenKind, T, E>,
    second: impl Parser<TokenKind, T, E>,
) -> impl Parser<TokenKind, T, E> {
    parse_fn!(|input| match first.run(input)? {
        InfallibleParseResult::Match {
            value,
            span,
            remaining,
        } => ParseResult::Match {
            value,
            span,
            remaining,
        },
        InfallibleParseResult::NoMatch => match second.run(input)? {
            InfallibleParseResult::Match {
                value,
                span,
                remaining,
            } => ParseResult::Match {
                value,
                span,
                remaining,
            },
            InfallibleParseResult::NoMatch => ParseResult::NoMatch,
        },
    })
}

#[doc(hidden)]
pub fn _map_to<TokenKind, T, U: Copy, E>(
    p: impl Parser<TokenKind, T, E>,
    val: U,
) -> impl Parser<TokenKind, U, E> {
    parse_fn!(|input| p.run(input).map(|_| val))
}

#[doc(hidden)]
pub fn _map<TokenKind, T, U, E>(
    p: impl Parser<TokenKind, T, E>,
    f: impl Fn(T) -> U + Copy,
) -> impl Parser<TokenKind, U, E> {
    parse_fn!(|input| p.run(input).map(f))
}

#[doc(hidden)]
pub fn _map_err<TokenKind, T, E, F>(
    p: impl Parser<TokenKind, T, E>,
    f: impl Fn(E) -> F + Copy,
) -> impl Parser<TokenKind, T, F> {
    parse_fn!(|input| p.run(input).map_err(f))
}

#[doc(hidden)]
pub fn _require<TokenKind, T, E>(
    p: impl Parser<TokenKind, T, E>,
    f: impl Fn(TokenStream<TokenKind>) -> E + Copy,
) -> impl Parser<TokenKind, T, E> {
    parse_fn!(|input| match p.run(input)? {
        InfallibleParseResult::Match {
            value,
            span,
            remaining,
        } => ParseResult::Match {
            value,
            span,
            remaining,
        },
        InfallibleParseResult::NoMatch => ParseResult::Err(f(input)),
    })
}

#[doc(hidden)]
pub fn _many<TokenKind, T, E>(
    p: impl Parser<TokenKind, T, E>,
    allow_empty: bool,
) -> impl Parser<TokenKind, Vec<T>, E> {
    parse_fn!(|mut input| {
        let mut result = Vec::new();
        let mut full_span = input.empty_span();

        loop {
            match p.run(input)? {
                InfallibleParseResult::Match {
                    value,
                    span,
                    remaining,
                } => {
                    result.push(value);
                    full_span = full_span.join(&span);
                    input = remaining;
                }
                InfallibleParseResult::NoMatch => break,
            }
        }

        if allow_empty || (result.len() > 0) {
            ParseResult::Match {
                value: result,
                span: full_span,
                remaining: input,
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

/// Matches three parsers in sequence, returning only the second result
pub fn between<TokenKind, T1, T2, T3, E>(
    prefix: impl Parser<TokenKind, T1, E>,
    p: impl Parser<TokenKind, T2, E>,
    suffix: impl Parser<TokenKind, T3, E>,
) -> impl Parser<TokenKind, T2, E> {
    parse_fn!(|input| match prefix.run(input)? {
        InfallibleParseResult::Match {
            span: s1,
            remaining,
            ..
        } => match p.run(remaining)? {
            InfallibleParseResult::Match {
                value, remaining, ..
            } => match suffix.run(remaining)? {
                InfallibleParseResult::Match {
                    span: s2,
                    remaining,
                    ..
                } => {
                    ParseResult::Match {
                        value,
                        span: s1.join(&s2),
                        remaining,
                    }
                }
                InfallibleParseResult::NoMatch => ParseResult::NoMatch,
            },
            InfallibleParseResult::NoMatch => ParseResult::NoMatch,
        },
        InfallibleParseResult::NoMatch => ParseResult::NoMatch,
    })
}

/// Matches a parser multiple times, separated by the given separator
pub fn sep_by<TokenKind, T, S, E>(
    p: impl Parser<TokenKind, T, E>,
    sep: impl Parser<TokenKind, S, E>,
    allow_empty: bool,
    allow_trailing: bool,
) -> impl Parser<TokenKind, Vec<T>, E> {
    parse_fn!(|input| {
        match p.run(input)? {
            InfallibleParseResult::Match {
                value,
                span,
                remaining,
            } => {
                let mut result = Vec::new();
                result.push(value);

                let mut full_span = span;
                let mut input = remaining;

                loop {
                    match sep.run(input)? {
                        InfallibleParseResult::Match {
                            span: sep_span,
                            remaining: sep_remaining,
                            ..
                        } => match p.run(sep_remaining)? {
                            InfallibleParseResult::Match {
                                value,
                                span,
                                remaining,
                            } => {
                                result.push(value);
                                full_span = full_span.join(&span);
                                input = remaining;
                            }
                            InfallibleParseResult::NoMatch => {
                                if allow_trailing {
                                    full_span = full_span.join(&sep_span);
                                    input = sep_remaining;
                                }

                                break;
                            }
                        },
                        InfallibleParseResult::NoMatch => break,
                    }
                }

                ParseResult::Match {
                    value: result,
                    span: full_span,
                    remaining: input,
                }
            }
            InfallibleParseResult::NoMatch => {
                if allow_empty {
                    ParseResult::Match {
                        value: Vec::new(),
                        span: input.empty_span(),
                        remaining: input,
                    }
                } else {
                    ParseResult::NoMatch
                }
            }
        }
    })
}

/// Matches a parser exactly 'count' times
pub fn repeat<TokenKind, T, E>(
    p: impl Parser<TokenKind, T, E>,
    count: usize,
) -> impl Parser<TokenKind, Vec<T>, E> {
    parse_fn!(|mut input| {
        let mut result = Vec::with_capacity(count);
        let mut full_span = input.empty_span();

        for _ in 0..count {
            match p.run(input)? {
                InfallibleParseResult::Match {
                    value,
                    span,
                    remaining,
                } => {
                    result.push(value);
                    full_span = full_span.join(&span);
                    input = remaining;
                }
                InfallibleParseResult::NoMatch => return ParseResult::NoMatch,
            }
        }

        ParseResult::Match {
            value: result,
            span: full_span,
            remaining: input,
        }
    })
}

pub use langbox_procmacro::{choice, parser, sequence};

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::lexer::test::*;

    fn parse_test_token(kind: TestTokenKind) -> impl Parser<TestTokenKind, char, &'static str> {
        parse_fn!(|input: TokenStream<TestTokenKind>| {
            if let Some(next) = input.peek() {
                if next.kind == kind {
                    ParseResult::Match {
                        value: next.kind.to_char(),
                        span: next.span,
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

    fn test_parser<T: Debug + Eq>(
        p: impl Parser<TestTokenKind, T, &'static str>,
        text: &'static str,
        expected_output: Option<T>,
    ) {
        let tokens = lex::<whitespace_mode::Keep>(text);
        let input = TokenStream::new(&tokens);

        let result = p.run(input).expect("parser produced an error");
        match result {
            InfallibleParseResult::Match { value, .. } => {
                if let Some(expected_output) = expected_output {
                    assert_eq!(expected_output, value);
                } else {
                    panic!("expected no match, but parser returned {value:?}");
                }
            }
            InfallibleParseResult::NoMatch => {
                if let Some(expected_output) = expected_output {
                    panic!("expected {expected_output:?}, but parser returned no match");
                }
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

    fn test_parser_error<T: Debug + Eq>(
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
        let p = super::sep_by(
            parse_test_token(TestTokenKind::A),
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
        let p = super::sep_by(
            parse_test_token(TestTokenKind::A),
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
        let p = super::sep_by(
            parse_test_token(TestTokenKind::A),
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
        let p = super::sep_by(
            parse_test_token(TestTokenKind::A),
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
        let p = super::repeat(parse_test_token(TestTokenKind::A), 0);

        test_parser(p, "", Some(vec![]));
        test_parser(p, "a", Some(vec![]));

        let p = super::repeat(parse_test_token(TestTokenKind::A), 3);

        test_parser(p, "", None);
        test_parser(p, "a", None);
        test_parser(p, "aaa", Some(vec!['a', 'a', 'a']));
        test_parser(p, "aaaa", Some(vec!['a', 'a', 'a']));
    }
}
