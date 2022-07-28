use langbox::*;

fn p1<TokenKind, E>() -> impl Parser<TokenKind, (), E> {
    parser!()
}

fn p2<TokenKind, E>() -> impl Parser<TokenKind, (), E> {
    parser!({p1()})
}

fn main() {}
