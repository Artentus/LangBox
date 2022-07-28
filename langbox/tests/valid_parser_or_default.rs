use langbox::*;

fn p<TokenKind, T: Default, E>(
    a: impl Parser<TokenKind, T, E>,
) -> impl Parser<TokenKind, T, E> {
    parser!(%a)
}

fn main() {}
