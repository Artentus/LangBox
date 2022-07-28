__A simple framework to build compilers and interpreters__

This crate requires a nightly compiler because of the try_trait_v2 feature.

## Usage
```rust
use langbox::*;

enum JsonTokenKind {
    // ...
}

// Make the reader an uninhabited type because we don't construct any objects of it
enum JsonTokenReader {}

impl TokenReader for JsonTokenReader {
    type Token = JsonTokenKind;
    
    fn read_token(text: &str) -> ReadTokenResult<Self::Token> {
        // ...
    }
}

struct JsonValue {
    // ...
}

fn jvalue() -> impl Parser<JsonTokenKind, JsonValue, String> {
    // ...
}

fn main() {
    // FileServer manages loading files for us.
    // It ensures the same physical file is never loaded twice.
    let mut file_server = FileServer::new();
    let file_id = file_server.register_file( /* ... */ );

    // After we have loaded a file we can tokenize it using a Lexer.
    type JLexer<'a> = Lexer<'a, JsonTokenReader, whitespace_mode::Remove>;
    let lexer = JLexer::new(file_id, &file_server);
    let tokens = lexer.collect::<Vec<_>>();
    let stream = TokenStream::new(&tokens);

    // Finally after all files have been tokenized we can parse the token stream.
    match jvalue().run(stream).expect("malformed JSON input") {
        InfallibleParseResult::Match { value, .. } => {
            // `value` contains the parsed JSON value
        }
        InfallibleParseResult::NoMatch => { /* empty input */ }
    }
}
```
