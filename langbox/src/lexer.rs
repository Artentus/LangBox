use super::*;
use std::marker::PhantomData;

/// The result of reading a token
pub struct ReadTokenResult<Kind> {
    /// The read token
    pub token: Kind,
    /// How many bytes in the input string the token consumed
    ///
    /// The input text advanced by this many bytes MUST yield a valid UTF-8 string.
    pub consumed_bytes: usize,
}

/// Reads tokens from text input
///
/// # Example Implementation
/// ```
/// use langbox::*;
///
/// // Our implementation-specific kinds of tokens
/// #[derive(Debug, Clone)]
/// enum JsonTokenKind {
///     // The error kind is a good way to keep track of lexing errors while still being able to continue lexing
///     Error,
///
///     OpenBrace,
///     CloseBrace,
///     OpenBracket,
///     CloseBracket,
///     Colon,
///     Comma,
///     Null,
///     False,
///     True,
///     Number(f64),
///     String(std::rc::Rc<str>),
/// }
///
/// // Make the reader an uninhabited type because we don't construct any objects of it
/// enum JsonTokenReader {}
///
/// impl JsonTokenReader {
///     fn read_number(text: &str) -> Option<ReadTokenResult<<Self as TokenReader>::Token>> {
///         // ...
///         # None
///     }
///     
///     fn read_string(text: &str) -> Option<ReadTokenResult<<Self as TokenReader>::Token>> {
///         // ...
///         # None
///     }
/// }
///
/// impl langbox::TokenReader for JsonTokenReader {
///     type Token = JsonTokenKind;
///     
///     fn read_token(text: &str) -> ReadTokenResult<Self::Token> {
///         // `text` is guaranteed to have a length > 0
///
///         const KEYWORDS: &[(&str, JsonTokenKind)] = &[
///             ("{", JsonTokenKind::OpenBrace),
///             ("}", JsonTokenKind::CloseBrace),
///             ("[", JsonTokenKind::OpenBracket),
///             ("]", JsonTokenKind::CloseBracket),
///             (":", JsonTokenKind::Colon),
///             (",", JsonTokenKind::Comma),
///             ("null", JsonTokenKind::Null),
///             ("false", JsonTokenKind::False),
///             ("true", JsonTokenKind::True),
///         ];
///     
///         for (pattern, token) in KEYWORDS.iter() {
///             if text.starts_with(pattern) {
///                 return ReadTokenResult {
///                     token: token.clone(),
///                     consumed_bytes: pattern.len(),
///                 };
///             }
///         }
///     
///         if let Some(result) = Self::read_number(text) {
///             return result;
///         }
///     
///         if let Some(result) = Self::read_string(text) {
///             return result;
///         }
///     
///         ReadTokenResult {
///             token: JsonTokenKind::Error,
///             consumed_bytes: text.chars().next().map(|c| c.len_utf8()).unwrap_or(0),
///         }
///     }
/// }
/// ```
pub trait TokenReader {
    /// The type of token being produced
    type Token;

    /// Reads one token from the input text
    fn read_token(text: &str) -> ReadTokenResult<Self::Token>;
}

/// A syntax token
pub struct Token<Kind> {
    /// The implementation-specific token kind
    pub kind: Kind,
    /// The span corresponding to this token
    pub span: TextSpan,
}

mod private {
    pub trait Sealed {}

    impl Sealed for super::whitespace_mode::Keep {}
    impl Sealed for super::whitespace_mode::Remove {}
    impl Sealed for super::whitespace_mode::RemoveKeepNewLine {}
}

/// Defines how a lexer processes whitespace
pub trait WhitespaceMode: private::Sealed {}

/// Defines how a lexer processes whitespace
pub mod whitespace_mode {
    /// The lexer will keep all whitespace and pass it on to the token reader.
    pub enum Keep {}
    impl super::WhitespaceMode for Keep {}

    /// The lexer will remove all whitespace.
    pub enum Remove {}
    impl super::WhitespaceMode for Remove {}

    /// The lexer will remove all whitespace except newline ('\n') characters.
    pub enum RemoveKeepNewLine {}
    impl super::WhitespaceMode for RemoveKeepNewLine {}
}

#[inline]
const fn is_valid_utf8_first_byte(byte: u8) -> bool {
    matches!(
        byte,
          0b00000000..=0b01111111
        | 0b11000000..=0b11011111
        | 0b11100000..=0b11101111
        | 0b11110000..=0b11110111
    )
}

/// Transforms a source file into a token stream
///
/// # Example
/// ```no_run
/// use langbox::*;
///
/// let mut file_server = FileServer::new();
///
/// # enum YourTokenKind { None }
/// # enum YourTokenReader {}
/// # impl langbox::TokenReader for YourTokenReader {
/// #     type Token = YourTokenKind;
/// #     fn read_token(text: &str) -> ReadTokenResult<Self::Token> {
/// #         ReadTokenResult {
/// #             token: YourTokenKind::None,
/// #             consumed_bytes: 0,
/// #         }
/// #     }
/// # }
///
/// let file_path = "path/to/json/file.json";
/// let file_id = file_server.register_file(file_path).unwrap();
///
/// let lexer = Lexer::<YourTokenReader, whitespace_mode::Remove>::new(file_id, &file_server);
/// let tokens = lexer.collect::<Vec<_>>();
/// ```
pub struct Lexer<'a, Reader: TokenReader, WSM: WhitespaceMode> {
    _reader: PhantomData<Reader>,
    _wsm: PhantomData<WSM>,
    file_id: FileId,
    file: &'a SourceFile,
    pos: TextPosition,
}

impl<'a, Reader: TokenReader, WSM: WhitespaceMode> Lexer<'a, Reader, WSM> {
    /// Creates a new lexer for a source file
    pub fn new(file_id: FileId, file_server: &'a FileServer) -> Self {
        Self {
            _reader: PhantomData,
            _wsm: PhantomData,
            file_id,
            file: file_server.get_file(file_id).expect("invalid file ID"),
            pos: TextPosition {
                file_id,
                byte_offset: 0,
            },
        }
    }

    #[inline]
    fn remaining_text(&self) -> &str {
        &self.file.text()[(self.pos.byte_offset as usize)..]
    }

    fn advance_n(&mut self, n: usize) {
        let new_byte_offset = (self.pos.byte_offset as usize) + n;

        if new_byte_offset > self.file.text().len() {
            panic!("exceeded source code length");
        } else if let Some(&byte) = self.file.text().as_bytes().get(new_byte_offset) {
            if !is_valid_utf8_first_byte(byte) {
                panic!("invalid UTF-8 offset into source code");
            }
        }

        self.pos = TextPosition {
            file_id: self.file_id,
            byte_offset: new_byte_offset as u32,
        };
    }

    fn advance_while(&mut self, predicate: impl Fn(char) -> bool) {
        let mut new_byte_offset = self.file.text().len();

        for (p, c) in self.remaining_text().char_indices() {
            if !predicate(c) {
                new_byte_offset = (self.pos.byte_offset as usize) + p;
                break;
            }
        }

        self.pos = TextPosition {
            file_id: self.file_id,
            byte_offset: new_byte_offset as u32,
        };
    }

    fn next_inner(&mut self) -> Option<Token<Reader::Token>> {
        if (self.pos.byte_offset as usize) < self.file.text().len() {
            let start_byte_offset = self.pos.byte_offset;
            let ReadTokenResult {
                token,
                consumed_bytes,
            } = Reader::read_token(self.remaining_text());
            self.advance_n(consumed_bytes);
            let end_byte_offset = self.pos.byte_offset;

            Some(Token {
                kind: token,
                span: TextSpan {
                    file_id: self.pos.file_id,
                    start_byte_offset,
                    end_byte_offset,
                },
            })
        } else {
            None
        }
    }
}

impl<'a, Reader: TokenReader> Iterator for Lexer<'a, Reader, whitespace_mode::Keep> {
    type Item = Token<Reader::Token>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_inner()
    }
}

impl<'a, Reader: TokenReader> Iterator for Lexer<'a, Reader, whitespace_mode::Remove> {
    type Item = Token<Reader::Token>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.advance_while(char::is_whitespace);
        self.next_inner()
    }
}

impl<'a, Reader: TokenReader> Iterator for Lexer<'a, Reader, whitespace_mode::RemoveKeepNewLine> {
    type Item = Token<Reader::Token>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.advance_while(|c| c.is_whitespace() & (c != '\n'));
        self.next_inner()
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub(crate) enum TestTokenKind {
        Error,
        NewLine,
        A,
        B,
    }

    impl TestTokenKind {
        pub(crate) fn to_char(&self) -> char {
            match self {
                Self::Error => panic!("invalid token kind"),
                Self::NewLine => '\n',
                Self::A => 'a',
                Self::B => 'b',
            }
        }
    }

    pub(crate) enum TestTokenReader {}
    impl TokenReader for TestTokenReader {
        type Token = TestTokenKind;

        fn read_token(text: &str) -> ReadTokenResult<Self::Token> {
            let c = text.chars().next().expect("invalid text");

            let kind = match c {
                '\n' => TestTokenKind::NewLine,
                'a' => TestTokenKind::A,
                'b' => TestTokenKind::B,
                _ => TestTokenKind::Error,
            };

            ReadTokenResult {
                token: kind,
                consumed_bytes: c.len_utf8(),
            }
        }
    }

    pub(crate) fn lex<WSM: WhitespaceMode>(text: &'static str) -> Vec<Token<TestTokenKind>>
    where
        for<'a> Lexer<'a, TestTokenReader, WSM>: Iterator<Item = Token<TestTokenKind>>,
    {
        let mut file_server = crate::file_system::FileServer::new();
        let file_id = file_server.register_file_memory("<test>", text);

        let lexer = Lexer::<TestTokenReader, WSM>::new(file_id, &file_server);
        lexer.collect()
    }

    fn test_lexer<WSM: WhitespaceMode>(text: &'static str, expected_tokens: &[TestTokenKind])
    where
        for<'a> Lexer<'a, TestTokenReader, WSM>: Iterator<Item = Token<TestTokenKind>>,
    {
        let tokens = lex::<WSM>(text);
        assert_eq!(tokens.len(), expected_tokens.len());

        for (i, token) in tokens.into_iter().enumerate() {
            assert_eq!(token.kind, expected_tokens[i]);
        }
    }

    #[test]
    fn empty() {
        test_lexer::<whitespace_mode::Keep>("", &[]);
    }

    #[test]
    fn remove_whitespace() {
        test_lexer::<whitespace_mode::Remove>("  \n  \t   \r\n ", &[]);
    }

    #[test]
    fn remove_whitespace_keep_newline() {
        test_lexer::<whitespace_mode::RemoveKeepNewLine>(
            "  \n  \t   \r\n ",
            &[TestTokenKind::NewLine, TestTokenKind::NewLine],
        );
    }

    #[test]
    fn valid_text() {
        test_lexer::<whitespace_mode::Keep>(
            "abaaabb",
            &[
                TestTokenKind::A,
                TestTokenKind::B,
                TestTokenKind::A,
                TestTokenKind::A,
                TestTokenKind::A,
                TestTokenKind::B,
                TestTokenKind::B,
            ],
        );
    }

    #[test]
    fn valid_text_remove_whitespace() {
        test_lexer::<whitespace_mode::Remove>(
            "a\nb  aa\tab b",
            &[
                TestTokenKind::A,
                TestTokenKind::B,
                TestTokenKind::A,
                TestTokenKind::A,
                TestTokenKind::A,
                TestTokenKind::B,
                TestTokenKind::B,
            ],
        );
    }

    #[test]
    fn valid_text_remove_whitespace_keep_newline() {
        test_lexer::<whitespace_mode::RemoveKeepNewLine>(
            "a\nb  aa\tab b",
            &[
                TestTokenKind::A,
                TestTokenKind::NewLine,
                TestTokenKind::B,
                TestTokenKind::A,
                TestTokenKind::A,
                TestTokenKind::A,
                TestTokenKind::B,
                TestTokenKind::B,
            ],
        );
    }
}
