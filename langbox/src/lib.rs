//! A simple framework to build compilers and interpreters
//!
//! This crate requires a nightly compiler because of the try_trait_v2 feature.
//!
//! # Usage
//! ```
//! use langbox::*;
//!
//! enum JsonTokenKind {
//!     // ...
//! }
//!
//! // Make the reader an uninhabited type because we don't construct any objects of it
//! enum JsonTokenReader {}
//!
//! impl TokenReader for JsonTokenReader {
//!     type Token = JsonTokenKind;
//!     
//!     fn read_token(text: &str) -> ReadTokenResult<Self::Token> {
//!         // ...
//!     }
//! }
//!
//! struct JsonValue {
//!     // ...
//! }
//!
//! fn jvalue() -> impl Parser<JsonTokenKind, JsonValue, String> {
//!     // ...
//! }
//!
//! fn main() {
//!     // FileServer manages loading files for us.
//!     // It ensures the same physical file is never loaded twice.
//!     let mut file_server = FileServer::new();
//!     let file_id = file_server.register_file( /* ... */ );
//!
//!     // After we have loaded a file we can tokenize it using a Lexer.
//!     type JLexer<'a> = Lexer<'a, JsonTokenReader, whitespace_mode::Remove>;
//!     let lexer = JLexer::new(file_id, &file_server);
//!     let tokens = lexer.collect::<Vec<_>>();
//!     let stream = TokenStream::new(&tokens);
//!
//!     // Finally after all files have been tokenized we can parse the token stream.
//!     match jvalue().run(stream).expect("malformed JSON input") {
//!         InfallibleParseResult::Match { value, .. } => {
//!             // `value` contains the parsed JSON value
//!         }
//!         InfallibleParseResult::NoMatch => { /* empty input */ }
//!     }
//! }
//! ```
#![deny(missing_docs)]
#![feature(try_trait_v2)]
#![cfg_attr(target_family = "windows", feature(windows_by_handle))]

use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::Hash;

mod file_system;
pub use file_system::*;

/// Defines a position in the text of a source file
#[derive(Debug, Clone, Copy, Eq)]
pub struct TextPosition {
    file_id: FileId,
    byte_offset: usize,
    line: usize,
    column: usize,
}

impl TextPosition {
    /// The source file this position is referring to
    #[inline]
    pub fn file_id(&self) -> FileId {
        self.file_id
    }

    /// The line in the text
    #[inline]
    pub fn line(&self) -> usize {
        self.line
    }

    /// The column in the line
    #[inline]
    pub fn column(&self) -> usize {
        self.column
    }

    /// Finds the smaller of two text positions.
    /// The smaller position is defined as the position that appears first in the text.
    pub fn min(self, other: Self) -> Self {
        match self
            .partial_cmp(&other)
            .expect("positions belong to different files")
        {
            Ordering::Less => self,
            Ordering::Equal => self,
            Ordering::Greater => other,
        }
    }

    /// Finds the larger of two text positions.
    /// The larger position is defined as the position that appears last in the text.
    pub fn max(self, other: Self) -> Self {
        match self
            .partial_cmp(&other)
            .expect("positions belong to different files")
        {
            Ordering::Less => other,
            Ordering::Equal => self,
            Ordering::Greater => self,
        }
    }
}

impl PartialEq for TextPosition {
    fn eq(&self, other: &Self) -> bool {
        let eq = (self.file_id == other.file_id) && (self.byte_offset == other.byte_offset);

        if eq {
            debug_assert_eq!(self.line, other.line);
            debug_assert_eq!(self.column, other.column);
        }

        eq
    }
}

impl PartialOrd for TextPosition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.file_id == other.file_id {
            let ord = self.byte_offset.cmp(&other.byte_offset);

            match ord {
                Ordering::Less => {
                    debug_assert!((self.line < other.line) | (self.column < other.column));
                }
                Ordering::Equal => {
                    debug_assert_eq!(self.line, other.line);
                    debug_assert_eq!(self.column, other.column);
                }
                Ordering::Greater => {
                    debug_assert!((self.line > other.line) | (self.column > other.column));
                }
            }

            Some(ord)
        } else {
            None
        }
    }
}

impl Hash for TextPosition {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.byte_offset.hash(state);
    }
}

impl Display for TextPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line + 1, self.column + 1)
    }
}

/// Defines one contiguous region in the text of a source file
#[derive(Debug, Clone, Copy, Eq)]
pub struct TextSpan {
    start_pos: TextPosition,
    end_pos: TextPosition,
}

impl TextSpan {
    /// The source file this span is referring to
    #[inline]
    pub fn file_id(&self) -> FileId {
        self.start_pos.file_id
    }

    /// The inclusive start position of the span
    #[inline]
    pub fn start_pos(&self) -> TextPosition {
        self.start_pos
    }

    /// The exclusive end position of the span
    #[inline]
    pub fn end_pos(&self) -> TextPosition {
        self.end_pos
    }

    /// Gets the string slice corresponding to this span
    pub fn text<'a>(&self, file_server: &'a FileServer) -> &'a str {
        let file = file_server
            .get_file(self.file_id())
            .expect("invalid file ID");

        &file.text()[self.start_pos.byte_offset..self.end_pos.byte_offset]
    }

    /// Joins two spans.
    /// The resulting span will define a region that includes the regions of both spans entirely as well as anything in between.
    pub fn join(&self, other: &Self) -> Self {
        let start_pos = self.start_pos.min(other.start_pos);
        let end_pos = self.end_pos.max(other.end_pos);

        Self { start_pos, end_pos }
    }
}

impl PartialEq for TextSpan {
    fn eq(&self, other: &Self) -> bool {
        (self.file_id() == other.file_id())
            && (self.start_pos == other.start_pos)
            && (self.end_pos == other.end_pos)
    }
}

#[doc(hidden)]
pub fn _join_spans(spans: &[TextSpan]) -> TextSpan {
    assert!(spans.len() > 0);

    let mut result = spans[0];
    for span in spans[1..].iter() {
        result = result.join(span);
    }
    result
}

mod lexer;
pub use lexer::*;

mod parser;
pub use parser::*;
