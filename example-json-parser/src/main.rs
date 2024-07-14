#![feature(trait_alias)]

use langbox::*;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone)]
enum JsonTokenKind {
    Error,
    OpenBrace,
    CloseBrace,
    OpenBracket,
    CloseBracket,
    Colon,
    Comma,
    Null,
    False,
    True,
    Number(f64),
    String(Rc<str>),
}

enum JsonTokenReader {}

impl JsonTokenReader {
    fn read_number(text: &str) -> Option<ReadTokenResult<<Self as TokenReader>::TokenKind>> {
        // This only supports numbers in a(.b)? format to keep the code simpler.

        let mut chars = text.char_indices();
        if let Some((_, c)) = chars.next() {
            if c.is_ascii_digit() {
                let mut end_pos = text.len();

                while let Some((pos, c)) = chars.next() {
                    if !c.is_ascii_digit() & (c != '.') {
                        end_pos = pos;
                        break;
                    }
                }

                let number_str = &text[..end_pos];
                Some(ReadTokenResult {
                    token: match number_str.parse::<f64>() {
                        Ok(value) => JsonTokenKind::Number(value),
                        Err(_) => JsonTokenKind::Error,
                    },
                    consumed_bytes: end_pos,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    fn read_string(text: &str) -> Option<ReadTokenResult<<Self as TokenReader>::TokenKind>> {
        // This does not support unicode escape sequences to keep the code simpler.

        let mut chars = text.char_indices();
        if let Some((_, '"')) = chars.next() {
            let mut result = String::new();
            let mut invalid_escape = false;

            while let Some((pos, c)) = chars.next() {
                if c == '"' {
                    return Some(ReadTokenResult {
                        token: if invalid_escape {
                            JsonTokenKind::Error
                        } else {
                            JsonTokenKind::String(result.into())
                        },
                        consumed_bytes: pos + '"'.len_utf8(),
                    });
                } else if c == '\\' {
                    if let Some((_, c)) = chars.next() {
                        match c {
                            '"' => result.push('"'),
                            '\\' => result.push('\\'),
                            '/' => result.push('/'),
                            'b' => result.push('\x08'),
                            'f' => result.push('\x0C'),
                            'n' => result.push('\n'),
                            'r' => result.push('\r'),
                            't' => result.push('\t'),
                            _ => invalid_escape = true, // continue lexing until the end of the string
                        }
                    } else {
                        // String ends in the escape character
                        return Some(ReadTokenResult {
                            token: JsonTokenKind::Error,
                            consumed_bytes: text.len(),
                        });
                    }
                } else {
                    result.push(c);
                }
            }

            // No closing double quotes
            return Some(ReadTokenResult {
                token: JsonTokenKind::Error,
                consumed_bytes: text.len(),
            });
        } else {
            None
        }
    }
}

impl TokenReader for JsonTokenReader {
    type TokenKind = JsonTokenKind;

    fn read_token(text: &str) -> ReadTokenResult<Self::TokenKind> {
        const KEYWORDS: &[(&str, JsonTokenKind)] = &[
            ("{", JsonTokenKind::OpenBrace),
            ("}", JsonTokenKind::CloseBrace),
            ("[", JsonTokenKind::OpenBracket),
            ("]", JsonTokenKind::CloseBracket),
            (":", JsonTokenKind::Colon),
            (",", JsonTokenKind::Comma),
            ("null", JsonTokenKind::Null),
            ("false", JsonTokenKind::False),
            ("true", JsonTokenKind::True),
        ];

        for (pattern, token) in KEYWORDS.iter() {
            if text.starts_with(pattern) {
                return ReadTokenResult {
                    token: token.clone(),
                    consumed_bytes: pattern.len(),
                };
            }
        }

        if let Some(result) = Self::read_number(text) {
            return result;
        }

        if let Some(result) = Self::read_string(text) {
            return result;
        }

        ReadTokenResult {
            token: JsonTokenKind::Error,
            consumed_bytes: text.chars().next().map(|c| c.len_utf8()).unwrap_or(0),
        }
    }
}

type JLexer<'a> = Lexer<'a, JsonTokenReader, whitespace_mode::Remove>;

#[derive(Debug)]
enum JsonValueKind {
    Null,
    Bool(bool),
    Number(f64),
    String(Rc<str>),
    Array(Vec<JsonValue>),
    Object(HashMap<Rc<str>, JsonValue>),
}

#[allow(dead_code)]
struct JsonValue {
    kind: JsonValueKind,
    span: TextSpan,
}

impl std::fmt::Debug for JsonValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.kind, f)
    }
}

trait JParser<T> = Parser<JsonTokenKind, T, String>;

fn token(predicate: impl Fn(&JsonTokenKind) -> bool + Copy) -> impl JParser<TextSpan> {
    parse_fn!(|input| if let Some(token) = input.peek() {
        if predicate(&token.kind) {
            ParseResult::Match {
                value: token.span,
                remaining: input.advance(),
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    })
}

macro_rules! token {
    ($kind:ident) => {
        token(|kind| matches!(kind, JsonTokenKind::$kind { .. }))
    };
}

fn jnull() -> impl JParser<JsonValue> {
    token!(Null).map(|span| JsonValue {
        kind: JsonValueKind::Null,
        span,
    })
}

fn jbool() -> impl JParser<JsonValue> {
    parse_fn!(|input| if let Some(token) = input.peek() {
        match &token.kind {
            JsonTokenKind::True => ParseResult::Match {
                value: JsonValue {
                    kind: JsonValueKind::Bool(true),
                    span: token.span,
                },
                remaining: input.advance(),
            },
            JsonTokenKind::False => ParseResult::Match {
                value: JsonValue {
                    kind: JsonValueKind::Bool(false),
                    span: token.span,
                },
                remaining: input.advance(),
            },
            _ => ParseResult::NoMatch,
        }
    } else {
        ParseResult::NoMatch
    })
}

fn jnumber() -> impl JParser<JsonValue> {
    parse_fn!(|input| if let Some(token) = input.peek() {
        if let JsonTokenKind::Number(value) = &token.kind {
            ParseResult::Match {
                value: JsonValue {
                    kind: JsonValueKind::Number(*value),
                    span: token.span,
                },
                remaining: input.advance(),
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    })
}

fn string() -> impl JParser<Rc<str>> {
    parse_fn!(|input| if let Some(token) = input.peek() {
        if let JsonTokenKind::String(value) = &token.kind {
            ParseResult::Match {
                value: Rc::clone(value),
                remaining: input.advance(),
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    })
}

fn jstring() -> impl JParser<JsonValue> {
    parse_fn!(|input| if let Some(token) = input.peek() {
        if let JsonTokenKind::String(value) = &token.kind {
            ParseResult::Match {
                value: JsonValue {
                    kind: JsonValueKind::String(Rc::clone(value)),
                    span: token.span,
                },
                remaining: input.advance(),
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    })
}

fn jarray() -> impl JParser<JsonValue> {
    let list = jvalue().sep_by(token!(Comma), true, false);
    let closing = token!(CloseBracket).require(|_| "expected closing bracket".to_string());
    let array = sequence!(token!(OpenBracket), list, closing);

    array.map(|(open, value, close)| JsonValue {
        kind: JsonValueKind::Array(value),
        span: open.join(close).unwrap(),
    })
}

fn jobject() -> impl JParser<JsonValue> {
    let field = parser!({string()} <. {token!(Colon)} <.> {jvalue()});
    let list = field.sep_by(token!(Comma), true, false);
    let closing = token!(CloseBrace).require(|_| "expected closing brace".to_string());
    let object = sequence!(token!(OpenBrace), list, closing);

    object.map(|(open, value, close)| JsonValue {
        kind: JsonValueKind::Object(value.into_iter().collect()),
        span: open.join(close).unwrap(),
    })
}

fn jvalue() -> impl JParser<JsonValue> {
    choice!(jnull(), jbool(), jnumber(), jstring(), jarray(), jobject())
}

const TEST_JSON: &str = r#"
[
  {
    "_id": "5973782bdb9a930533b05cb2",
    "isActive": true,
    "balance": "$1,446.35",
    "age": 32,
    "eyeColor": "green",
    "name": "Logan Keller",
    "gender": "male",
    "company": "ARTIQ",
    "email": "logankeller@artiq.com",
    "phone": "+1 (952) 533-2258",
    "friends": [
      {
        "id": 0,
        "name": "Colon Salazar"
      },
      {
        "id": 1,
        "name": "French Mcneil"
      },
      {
        "id": 2,
        "name": "Carol Martin"
      }
    ],
    "favoriteFruit": "banana"
  }
]
"#;

#[cfg(feature = "bench")]
fn main() {
    let mut file_server = FileServer::new();
    let file_id = file_server
        .register_file_memory("<test>", TEST_JSON)
        .unwrap();

    for _ in 0..10000 {
        let f = std::hint::black_box(|| {
            let lexer = JLexer::new(file_id, &file_server);
            let tokens = std::hint::black_box(lexer.collect::<Vec<_>>());
            let stream = TokenStream::new(&tokens);

            if let ParseResult::Err(_) = std::hint::black_box(jvalue().run(stream)) {
                panic!("malformed JSON input");
            }
        });

        f();
    }
}

#[cfg(not(feature = "bench"))]
fn main() {
    let mut file_server = FileServer::new();
    let file_id = file_server
        .register_file_memory("<test>", TEST_JSON)
        .unwrap();

    let lexer = JLexer::new(file_id, &file_server);
    let tokens = lexer.collect::<Vec<_>>();
    let stream = TokenStream::new(&tokens);

    match jvalue().run(stream) {
        ParseResult::Match { value, .. } => {
            println!("{value:#?}")
        }
        ParseResult::NoMatch => { /* empty input */ }
        ParseResult::Err(_) => panic!("malformed JSON input"),
    }
}
