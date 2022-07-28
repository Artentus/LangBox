#[test]
fn tests() {
    let t = trybuild::TestCases::new();
    t.pass("tests/empty_parser.rs");
    t.pass("tests/empty_choice.rs");
    t.pass("tests/empty_sequence.rs");
    t.pass("tests/valid_choice.rs");
    t.pass("tests/valid_sequence.rs");
    t.pass("tests/valid_parser_opt.rs");
    t.pass("tests/valid_parser_or_default.rs");
    t.compile_fail("tests/invalid_parser_or_default.rs");
    t.pass("tests/valid_parser_and_then.rs");
    t.pass("tests/valid_parser_prefix.rs");
    t.pass("tests/valid_parser_suffix.rs");
    t.pass("tests/valid_parser_or_else.rs");
    t.pass("tests/valid_parser_precendence.rs");
    t.pass("tests/valid_parser_parenthesized.rs");
    t.pass("tests/valid_parser_many.rs");
    t.pass("tests/valid_parser_many1.rs");
    t.pass("tests/valid_parser_curl_expr.rs");
    t.compile_fail("tests/invalid_parser_expr.rs");
    t.pass("tests/valid_parser_map_to.rs");
    t.pass("tests/valid_parser_map.rs");
    t.compile_fail("tests/invalid_parser_map.rs");
}
