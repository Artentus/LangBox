use langbox::*;

fn p1<TokenKind, T1, T2, E>(
    a: impl Parser<TokenKind, T1, E>,
    b: impl Parser<TokenKind, T2, E>,
    c: impl Parser<TokenKind, (T1, T2), E>,
) -> impl Parser<TokenKind, (T1, T2), E> {
    parser!(a <.> b <|> c)
}

fn p2<TokenKind, T1, T2, E>(
    a: impl Parser<TokenKind, T1, E>,
    b: impl Parser<TokenKind, T2, E>,
    c: impl Parser<TokenKind, (T1, T2), E>,
) -> impl Parser<TokenKind, (T1, T2), E> {
    parser!(c <|> a <.> b)
}

fn main() {}
