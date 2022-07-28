use langbox::*;

fn p<TokenKind, T, E>(
    a: impl Parser<TokenKind, T, E>,
    b: impl Parser<TokenKind, T, E>,
) -> impl Parser<TokenKind, T, E> {
    parser!(a <|> b)
}

fn main() {}
