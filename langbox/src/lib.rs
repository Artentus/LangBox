//! A simple framework to build compilers and interpreters
//!
//! This crate requires a nightly compiler because of the try_trait_v2 feature.
//!
//! # Usage
//! ```no_run
//! use langbox::*;
//!
//! enum JsonTokenKind {
//!     // ...
//!     # None
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
//!         # ReadTokenResult {
//!         #     token: JsonTokenKind::None,
//!         #     consumed_bytes: 0,
//!         # }
//!     }
//! }
//!
//! # #[derive(Clone, Copy)]
//! struct JsonValue {
//!     // ...
//! }
//!
//! fn jvalue() -> impl Parser<JsonTokenKind, JsonValue, String> {
//!     // ...
//!     # parser!(()=>[JsonValue {}])
//! }
//!
//! fn main() {
//!     // FileServer manages loading files for us.
//!     // It ensures the same physical file is never loaded twice.
//!     let mut file_server = FileServer::new();
//!
//!     let file_path = "path/to/json/file.json";
//!     let file_id = file_server.register_file(file_path).unwrap();
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
use std::fmt::Debug;
use std::hash::Hash;

mod file_system;
pub use file_system::*;

/// Defines a position in the text of a source file
#[derive(Debug, Clone, Copy, Eq)]
pub struct TextPosition {
    file_id: FileId,
    byte_offset: u32,
}

impl TextPosition {
    /// The source file this position is referring to
    #[inline]
    pub fn file_id(&self) -> FileId {
        self.file_id
    }

    /// Gets the zero-based line and column numbers corresponding to this position
    pub fn line_column(&self, file_server: &FileServer) -> (u32, u32) {
        let file = file_server.get_file(self.file_id).expect("invalid file ID");

        let mut line = 0;
        let mut column = 0;

        for (i, c) in file.text().char_indices() {
            if i >= (self.byte_offset as usize) {
                break;
            }

            if c == '\n' {
                line += 1;
                column = 0;
            } else {
                column += 1;
            }
        }

        (line, column)
    }

    /// Finds the smaller of two text positions.
    /// The smaller position is defined as the position that appears first in the text.
    pub fn min(&self, other: &Self) -> Self {
        match self
            .partial_cmp(&other)
            .expect("positions belong to different files")
        {
            Ordering::Less => *self,
            Ordering::Equal => *self,
            Ordering::Greater => *other,
        }
    }

    /// Finds the larger of two text positions.
    /// The larger position is defined as the position that appears last in the text.
    pub fn max(&self, other: &Self) -> Self {
        match self
            .partial_cmp(&other)
            .expect("positions belong to different files")
        {
            Ordering::Less => *other,
            Ordering::Equal => *self,
            Ordering::Greater => *self,
        }
    }
}

impl PartialEq for TextPosition {
    fn eq(&self, other: &Self) -> bool {
        (self.file_id == other.file_id) && (self.byte_offset == other.byte_offset)
    }
}

impl PartialOrd for TextPosition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.file_id == other.file_id {
            Some(self.byte_offset.cmp(&other.byte_offset))
        } else {
            None
        }
    }
}

impl Hash for TextPosition {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.file_id.hash(state);
        self.byte_offset.hash(state);
    }
}

/// Defines one contiguous region in the text of a source file
#[derive(Debug, Clone, Copy, Eq)]
pub struct TextSpan {
    file_id: FileId,
    start_byte_offset: u32,
    end_byte_offset: u32,
}

impl TextSpan {
    /// The source file this span is referring to
    #[inline]
    pub fn file_id(&self) -> FileId {
        self.file_id
    }

    /// The inclusive start position of the span
    #[inline]
    pub fn start_pos(&self) -> TextPosition {
        TextPosition {
            file_id: self.file_id,
            byte_offset: self.start_byte_offset,
        }
    }

    /// The exclusive end position of the span
    #[inline]
    pub fn end_pos(&self) -> TextPosition {
        TextPosition {
            file_id: self.file_id,
            byte_offset: self.end_byte_offset,
        }
    }

    /// Gets the string slice corresponding to this span
    pub fn text<'a>(&self, file_server: &'a FileServer) -> &'a str {
        let file = file_server.get_file(self.file_id).expect("invalid file ID");
        &file.text()[(self.start_byte_offset as usize)..(self.end_byte_offset as usize)]
    }

    /// Joins two spans.
    /// The resulting span will define a region that includes the regions of both spans entirely as well as anything in between.
    pub fn join(&self, other: &Self) -> Self {
        assert_eq!(
            self.file_id, other.file_id,
            "spans belong to different files"
        );

        let start_byte_offset = self.start_byte_offset.min(other.start_byte_offset);
        let end_byte_offset = self.end_byte_offset.max(other.end_byte_offset);

        Self {
            file_id: self.file_id,
            start_byte_offset,
            end_byte_offset,
        }
    }
}

impl PartialEq for TextSpan {
    fn eq(&self, other: &Self) -> bool {
        (self.file_id == other.file_id)
            && (self.start_byte_offset == other.start_byte_offset)
            && (self.end_byte_offset == other.end_byte_offset)
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
