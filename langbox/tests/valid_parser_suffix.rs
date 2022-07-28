use langbox::*;

fn p<TokenKind, T1, T2, E>(
    a: impl Parser<TokenKind, T1, E>,
    b: impl Parser<TokenKind, T2, E>,
) -> impl Parser<TokenKind, T2, E> {
    parser!(a .> b)
}

fn main() {}
