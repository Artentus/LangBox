use langbox::*;

fn p<TokenKind, T, E>(
    a: impl Parser<TokenKind, T, E>,
) -> impl Parser<TokenKind, Vec<T>, E> {
    parser!(+a)
}

fn main() {}
