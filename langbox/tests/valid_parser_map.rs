use langbox::*;

fn p<TokenKind, T, E>(
    a: impl Parser<TokenKind, T, E>,
) -> impl Parser<TokenKind, bool, E> {
    parser!(a->[|_| false])
}

fn main() {}
